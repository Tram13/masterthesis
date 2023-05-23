import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import swifter
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tools.config_parser import ConfigParser

# Trust me bro
f"{swifter.config}"


# This class works with the Yelp data set format. Download the data from https://www.yelp.com/dataset.
# It is expected that the data files are unpacked in the location defined by config.ini.
class DataReader:
    """
    Deze klasse omvat functies om de originele Yelp Dataset te parsen
    Hier gaan we dus data inlezen, de train-test split maken en in kleine mate feature engineering toepassen
    De code om de restaurants, users en reviews te parsen staat ook in `src/data/data analysis/*.ipynb`
    Het is makkelijker om de code daar te volgen dan in deze klasse
    """
    EXPECTED_FILES = [
        'yelp_academic_dataset_business.json',
        'yelp_academic_dataset_checkin.json',
        'yelp_academic_dataset_review.json',
        'yelp_academic_dataset_tip.json',
        'yelp_academic_dataset_user.json'
    ]

    RELEVANT_BUSINESS_FIELDS = [
        'business_id',
        'name',
        'city',
        'stars',
        'review_count',
        'attributes',  # Filtered in _parse_categories()
        'categories'  # Filtered in _parse_categories()
    ]

    RELEVANT_CHECKIN_FIELDS = [
        'business_id',
        'date'  # Will be transformed to 'average_checkins_per_week_normalised' and included in businesses dataframe
    ]

    RELEVANT_REVIEW_FIELDS = [
        'review_id',
        'user_id',
        'business_id',
        'stars',
        'useful',
        'funny',
        'cool',
        'text',
        'date'
    ]

    RELEVANT_USER_FIELDS = [
        'user_id',
        'useful',
        'funny',
        'cool',
        'fans',
        'compliments'  # Sum of all compliment fields
    ]

    def __init__(self, data_path: os.PathLike = None):
        # Default value for data_path is provided by config.ini file
        if data_path is None:
            data_path = Path(ConfigParser().get_value('data', 'data_path'))
        self.data_path = data_path
        self.cache_path = Path(self.data_path, ConfigParser().get_value('data', 'cache_directory'))
        self._assert_cache_dir_exists()
        self.file_paths = [Path(data_path, file) for file in self.EXPECTED_FILES]

    def read_data(self, use_cache: bool = True, save_as_cache: bool = True, part: int = None,
                  total_parts: int = None, no_train_test: bool = False, at_least: int = None, at_most: int = None) -> \
            tuple[tuple[DataFrame, DataFrame, DataFrame], tuple[DataFrame, DataFrame, DataFrame]]:
        """
        De ""main"" functie van deze klasse. Hiermee wordt alle data ingeladen, in 2 tuples die de train- en testset voorstellen
        Voor de overige methoden uit deze klasse te begrijpen, raden we aan om de notebooks te gebruiken
        :param use_cache: Lees data uit cache, indien mogelijk
        :param save_as_cache: Sla de verwerkte data op in de cache
        :param part: Not supported
        :param total_parts: Not supported
        :param no_train_test: Maak geen train-test-split. Alle data zal dan in het 0de element van de returned tuple zitten
        :param at_least: Filter data met minimaal `at_least` reviews
        :param at_most:  Filter data met maximaal `at_most` reviews
        :return:
        """
        if part is not None or total_parts is not None:
            raise NotImplementedError("Dit wordt niet meer ondersteund")
        if no_train_test:
            try:
                with tqdm(total=3, desc="Reading files from disk (no train test)", leave=False) as p_bar:
                    p_bar.set_postfix_str('(current: businesses)')
                    b = pd.read_parquet(Path(self.cache_path, 'businesses.parquet'), engine='fastparquet')
                    p_bar.update()
                    p_bar.set_postfix_str('(current: reviews)')
                    r = pd.read_parquet(Path(self.cache_path, 'reviews.parquet'), engine='fastparquet')
                    p_bar.update()
                    p_bar.set_postfix_str('(current: users)')
                    u = pd.read_parquet(Path(self.cache_path, 'users.parquet'), engine='fastparquet')
                    p_bar.update()
                return (b, r, u), (DataFrame(), DataFrame(), DataFrame())
            except OSError:
                raise FileNotFoundError('Please regenerate original caches')
        if use_cache:
            return self._read_from_cache(at_least, at_most)

        (b_train, r_train, u_train), (b_test, r_test, u_test) = self._read_from_disk(at_least, at_most)
        if save_as_cache:
            b_train.to_parquet(Path(self.cache_path, 'businesses_train_test.parquet'), engine='fastparquet')
            r_train.to_parquet(Path(self.cache_path, 'reviews_train.parquet'), engine='fastparquet')
            r_test.to_parquet(Path(self.cache_path, 'reviews_test.parquet'), engine='fastparquet')
            u_train.to_parquet(Path(self.cache_path, 'users_train.parquet'), engine='fastparquet')
            u_test.to_parquet(Path(self.cache_path, 'users_test.parquet'), engine='fastparquet')

        return (b_train, r_train, u_train), (b_test, r_test, u_test)

        # if part > total_parts or part < 1:
        #     raise ValueError(f"Cannot get part {part}/{total_parts}")
        # # Only partial read
        # else:
        #     raise NotImplementedError("Dit wordt niet meer ondersteund")
        # business_count = len(businesses)
        # start_index = (part - 1) * (business_count // total_parts)
        # end_index = part * (business_count // total_parts)
        # businesses = businesses[start_index:end_index]
        # return businesses, reviews, tips, users

    def _read_from_disk(self, at_least: int, at_most: int) -> tuple[tuple[DataFrame, DataFrame, DataFrame], tuple[DataFrame, DataFrame, DataFrame]]:
        with tqdm(total=5, desc="Reading files from disk", leave=False) as p_bar:
            p_bar.set_postfix_str('(current: businesses)')
            businesses = self._parse_businesses(self.file_paths[0])
            p_bar.update()
            p_bar.set_postfix_str('current: users')
            users_train, users_test = self._parse_users(self.file_paths[4])
            p_bar.update()
            p_bar.set_postfix_str('(current: reviews)')
            reviews_train, reviews_test = self._parse_reviews(self.file_paths[2], businesses, users_train, users_test)
            p_bar.update()
            p_bar.set_postfix_str('current: normalising')
            users_train = DataReader._process_users_split(users_train, reviews_train, businesses)
            users_test = DataReader._process_users_split(users_test, reviews_test, businesses)
            p_bar.update()
            p_bar.set_postfix_str('current: creating optimised indices')
            businesses, reviews_train, reviews_test, users_train, users_test = DataReader.fix_indices(
                businesses, reviews_train, reviews_test, users_train, users_test
            )

            p_bar.update()

            if at_least or at_most:
                original_size = len(reviews_train)
                if at_least:
                    reviews_train = self._get_coldstart_at_least_n_reviews(reviews_train, at_least)
                    reviews_test = self._get_coldstart_at_least_n_reviews(reviews_test, at_least)

                if at_most:
                    reviews_train = self._get_coldstart_at_most_n_reviews(reviews_train, at_most)
                    reviews_test = self._get_coldstart_at_most_n_reviews(reviews_test, at_most)

                logging.info(f"Kept {(len(reviews_train) / original_size * 100):.2f}% of dataset after coldstart with at least {at_least}, at most {at_most} filter")

            train_set = (businesses, reviews_train, users_train)
            test_set = (businesses.copy(deep=True), reviews_test, users_test)

        return train_set, test_set

    def _read_from_cache(self, at_least: int, at_most: int) -> tuple[tuple[DataFrame, DataFrame, DataFrame], tuple[DataFrame, DataFrame, DataFrame]]:
        try:
            with tqdm(total=3, desc="Reading files from cache", leave=False) as p_bar:
                p_bar.set_postfix_str('(current: businesses)')
                b_train = pd.read_parquet(Path(self.cache_path, 'businesses_train_test.parquet'), engine='fastparquet')
                b_test = b_train.copy(deep=True)
                p_bar.update()
                p_bar.set_postfix_str('(current: reviews)')
                r_train = pd.read_parquet(Path(self.cache_path, 'reviews_train.parquet'), engine='fastparquet')
                r_test = pd.read_parquet(Path(self.cache_path, 'reviews_test.parquet'), engine='fastparquet')
                p_bar.update()
                p_bar.set_postfix_str('current: users')
                u_train = pd.read_parquet(Path(self.cache_path, 'users.parquet'), engine='fastparquet')
                u_test = pd.read_parquet(Path(self.cache_path, 'users.parquet'), engine='fastparquet')
                p_bar.update()
        except OSError:
            print("Could not reach caches!", file=sys.stderr)
            return self._read_from_disk(at_least, at_most)

        if at_least or at_most:
            original_size = len(r_train)
            if at_least:
                r_train = self._get_coldstart_at_least_n_reviews(r_train, at_least)
                r_test = self._get_coldstart_at_least_n_reviews(r_test, at_least)

            if at_most:
                r_train = self._get_coldstart_at_most_n_reviews(r_train, at_most)
                r_test = self._get_coldstart_at_most_n_reviews(r_test, at_most)

            logging.info(f"Kept {(len(r_train) / original_size * 100):.2f}% of dataset after coldstart with at least {at_least}, at most {at_most} filter")

        return (b_train, r_train, u_train), (b_test, r_test, u_test)

    @staticmethod
    def _get_coldstart_at_least_n_reviews(reviews: pd.DataFrame, n: int) -> pd.DataFrame:
        user_review_counts = reviews.groupby(['user_id']).count()['business_id'].rename('count')
        user_at_least_n_reviews = user_review_counts[user_review_counts >= n].index
        filtered_reviews = reviews[reviews['user_id'].isin(user_at_least_n_reviews)]
        return filtered_reviews

    @staticmethod
    def _get_coldstart_at_most_n_reviews(reviews: pd.DataFrame, n: int) -> pd.DataFrame:
        user_review_counts = reviews.groupby(['user_id']).count()['business_id'].rename('count')
        user_at_most_n_reviews = user_review_counts[user_review_counts <= n].index
        filtered_reviews = reviews[reviews['user_id'].isin(user_at_most_n_reviews)]
        return filtered_reviews

    # Check if ALL and NOTHING BUT the data files are present in the provided directory
    def _assert_cache_dir_exists(self):
        cache_already_exists = os.path.isdir(self.cache_path)
        if not cache_already_exists:
            os.mkdir(self.cache_path)

    @staticmethod
    def _get_entries_from_file(file_path: os.PathLike) -> list[dict[str, any]]:
        with open(file_path, mode='r', encoding='utf-8') as json_file:
            return [json.loads(line) for line in json_file]

    # Keep only the required fields in the provided dictionary
    @staticmethod
    def _filter_entries(entries: list[dict[str, any]], fields: list[str]) -> list[dict[str, any]]:
        return [{key: entry[key] for key in fields} for entry in entries]

    def _parse_businesses(self, file_location: os.PathLike) -> DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_BUSINESS_FIELDS)
        businesses: DataFrame = DataFrame.from_records(filtered_entries)

        # Normalise data
        businesses = businesses.rename(columns={'stars': 'average_stars'})
        column_names_to_normalise = ['average_stars', 'review_count']
        normalised_series = [
            pd.Series(
                data=preprocessing.MinMaxScaler().fit_transform(
                    businesses[column_name].to_numpy().reshape(-1, 1)
                ).flatten(),
                name=f'business_{column_name}_normalised',
                dtype=np.float16,
            ).set_axis(businesses.index)  # To relink with the original dataframe
            for column_name in column_names_to_normalise
        ]
        businesses = businesses.drop(columns=column_names_to_normalise)
        businesses = pd.concat([businesses, *normalised_series], axis=1)

        # PARSING CATEGORIES
        categories_whitelist = {
            "Food Trucks",  # Data exploration shows that all restaurant-like businesses
            "Restaurants",  # either have the category "Food Truck" or "Restaurant".
        }  # Only keep businesses that contain at least 1 of the categories in this whitelist
        businesses['categories'] = [
            set(category_group.split(", "))  # Convert string of all categories to a set of individual categories
            if category_group and set(category_group.split(", ")).intersection(categories_whitelist)  # If in whitelist
            else None  # No category is provided by Yelp, or no category is in the whitelist
            for category_group in businesses['categories']
        ]
        businesses = businesses.dropna(subset=['categories']).copy()  # Remove businesses with no categories listed
        # We will only keep the categories with a high occurence
        all_remaining_categories = (category
                                    for business_categories in businesses['categories']
                                    for category in business_categories)
        categories_appearances = Counter(all_remaining_categories)
        common_categories = {item for item, count in categories_appearances.items() if count >= 500}
        businesses['categories'] = businesses['categories'].map(common_categories.intersection)
        all_remaining_categories = (category for business_categories in businesses['categories'] for category in
                                    business_categories)
        categories_appearances = Counter(all_remaining_categories)

        onehot_categories = [
            businesses['categories']
            .map(lambda business_categories: 1 if category in business_categories else 0)
            .rename(f"category_{category.replace(' ', '_').lower()}").astype(np.uint8)
            for category in categories_appearances.keys()
        ]
        businesses = pd.concat([businesses, *onehot_categories], axis=1)
        businesses = businesses.drop(columns=['categories'])

        # PARSING ATTRIBUTES
        businesses_attributes_filtered = []

        filtered_attributes_single = {
            'RestaurantsTakeOut',
            'RestaurantsDelivery',
            'RestaurantsPriceRange2',
            'GoodForKids',
            'RestaurantsGoodForGroups',
            'RestaurantsAttire',
            'NoiseLevel'
        }
        filtered_attributes_multi = {
            'Ambience',
            'GoodForMeal'
        }

        for business_attributes in businesses['attributes']:
            parsed_business_attributes = {}
            if business_attributes is not None:
                for attribute_key, attribute_value in business_attributes.items():
                    if attribute_key in filtered_attributes_multi and attribute_value.startswith(
                            '{'):  # Attribute is again a dict
                        json_string = re.sub(
                            ', u"',
                            ', "',
                            attribute_value.replace('\'', '\"').lower().replace('none', 'null')
                        ).replace('{u', '{')  # The provided JSON dict is not entirely up-to-spec
                        sub_attributes = json.loads(json_string)
                        for sub_key, sub_value in sub_attributes.items():
                            parsed_business_attributes[sub_key] = sub_value
                    elif attribute_key in filtered_attributes_single:
                        parsed_business_attributes[attribute_key] = attribute_value
            businesses_attributes_filtered.append(parsed_business_attributes)

        businesses['attributes'] = businesses_attributes_filtered
        all_remaining_attributes = (attribute_key for business_attributes in businesses['attributes'] for attribute_key
                                    in business_attributes.keys())
        attributes_appearances = Counter(all_remaining_attributes)
        onehot_attributes = [
            businesses['attributes']
            .map(
                lambda business_attributes_f:
                business_attributes_f[attribute] if attribute in business_attributes_f
                else None
            )
            .rename(f'attribute_{attribute.lower()}')
            .replace('None', None)
            .replace('True', True)
            .replace('False', False)
            for attribute, _ in attributes_appearances.most_common()  # Sorted list of attributes, since order matters
        ]
        onehot_attributes = [
            series.map(
                lambda attributes:
                re.sub("^u'", "", attributes).replace("'", "") if isinstance(attributes, str) else attributes)
            for series in onehot_attributes
        ]

        for index in range(len(onehot_attributes)):  # Convert string/boolean attributes to floats
            if index == 2:  # attribute_restaurantspricerange2
                onehot_attributes[index] = onehot_attributes[index].map(
                    lambda x: 0 if x == '1' else (
                        0.33 if x == '2' else (0.67 if x == '3' else (1 if x == '4' else 0.33)))
                ).astype(np.float16)  # '2' seems to be the most common value, thus default
            elif index == 14:  # attribute_noiselevel
                onehot_attributes[index] = onehot_attributes[index].map(
                    lambda x: 0 if x == 'quiet' else (
                        0.33 if x == 'average' else (0.67 if x == 'loud' else (1 if x == 'very_loud' else 0.33)))
                ).astype(np.float16)  # 'average' is the default value
            else:
                onehot_attributes[index] = onehot_attributes[index].map(
                    lambda x: 1 if x is True else (0 if x is False else 0.5)
                ).astype(np.float16)

        businesses = pd.concat([businesses, *onehot_attributes], axis=1)
        businesses = businesses.drop(columns=['attributes'])
        businesses = businesses.set_index('business_id')

        # ADD CHECK-INS
        checkins = DataReader._parse_checkins(self.file_paths[1])
        businesses = businesses.join(checkins, on='business_id')
        businesses['average_checkins_per_week_normalised'] = businesses['average_checkins_per_week_normalised'].replace(
            [np.nan], 0
        )

        return businesses

    @staticmethod
    def _parse_checkins(file_location: os.PathLike) -> DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_CHECKIN_FIELDS)
        checkins = DataFrame.from_records(filtered_entries)
        checkins['date'] = checkins['date'].map(
            lambda datelist: [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in datelist.split(', ')]
        )

        first_checkins = checkins['date'].map(lambda datelist: min(datelist))  # First check-in per restaurant
        last_checkin = checkins['date'].map(lambda datelist: max(datelist)).max()  # Last checkin date in entire dataset
        #  Amount of weeks between first check-in and last possible check-in
        amount_of_weeks = (last_checkin - first_checkins).map(lambda x: x.days / 7)
        amount_of_checkins = checkins['date'].transform(len)
        avg_checkins_per_week = (amount_of_checkins / amount_of_weeks).replace([np.inf, -np.inf, np.nan], 0)

        # remove extreme values
        avg_checkin_bottom_5 = avg_checkins_per_week.quantile(0.05)
        avg_checkin_top_5 = avg_checkins_per_week.quantile(0.95)
        avg_checkins_per_week = avg_checkins_per_week.transform(lambda x: DataReader.remove_extremes(x, avg_checkin_bottom_5, avg_checkin_top_5))

        avg_checkins_per_week_normalised = pd.Series(
            data=preprocessing.MinMaxScaler().fit_transform(avg_checkins_per_week.to_numpy().reshape(-1, 1)).flatten(),
            name="average_checkins_per_week_normalised"
        )

        checkins = pd.concat([checkins, avg_checkins_per_week_normalised], axis=1)
        checkins = checkins.drop(columns=['date'])
        checkins = checkins.set_index('business_id')
        return checkins

    @staticmethod
    def _parse_reviews(file_location: os.PathLike, businesses: DataFrame, users_train: DataFrame, users_test: DataFrame) -> tuple[DataFrame, DataFrame]:
        """
        :param file_location: Location of the reviews dataset in json format
        :param businesses: The businesses DataFrame as parsed by `_parse_businesses()`
        :return: A DataFrame containing all reviews for the given businesses
        """
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_REVIEW_FIELDS)
        reviews = DataFrame.from_records(filtered_entries)

        normalised_column = pd.Series(
            data=preprocessing.MinMaxScaler().fit_transform(
                reviews['stars'].to_numpy().reshape(-1, 1)
            ).flatten(),
            name='stars_normalised',
            dtype=np.float16,
        ).set_axis(reviews.index)  # To relink with the original dataframe
        reviews = reviews.drop(columns=['stars'])
        reviews = pd.concat([reviews, normalised_column], axis=1)

        # cleanup of other fields
        reviews['useful'] = reviews['useful'].transform(lambda x: 0 if x == 0 else 1).astype(np.uint8)
        reviews['funny_cool'] = reviews[['funny', 'cool']].apply(
            lambda row: 0 if row['funny'] == 0 and row['cool'] == 1 else 1, axis=1
        ).rename("funny_cool").astype(np.uint8)
        reviews = reviews.drop(columns=['funny', 'cool'])
        reviews['date'] = reviews['date'].map(lambda date_str: datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S'))

        # Only keep reviews for restaurants
        reviews = reviews[reviews['business_id'].isin(businesses.index)]
        reviews = reviews.set_index('review_id')
        reviews['text'] = reviews['text'].astype("string")

        # Transforming review IDs to integers
        reviews = reviews.reset_index(drop=True)
        reviews.index = reviews.index.rename('review_id')

        # Train - testset
        train_reviews = reviews[reviews['user_id'].isin(users_train.index.unique())]
        test_reviews = reviews[reviews['user_id'].isin(users_test.index.unique())]

        return train_reviews, test_reviews

    @staticmethod
    def _parse_users(file_location: os.PathLike) -> tuple[DataFrame, DataFrame]:
        entries = DataReader._get_entries_from_file(file_location)
        # Combine all compliments
        compliment_fields = [
            'compliment_hot',
            'compliment_more',
            'compliment_profile',
            'compliment_cute',
            'compliment_list',
            'compliment_note',
            'compliment_plain',
            'compliment_cool',
            'compliment_funny',
            'compliment_writer',
            'compliment_photos'
        ]
        combined_compliments = DataReader._filter_entries(entries, compliment_fields)
        combined_compliments = [sum(x.values()) for x in combined_compliments]
        for entry, sum_combined_for_entry in zip(entries, combined_compliments):
            entry['compliments'] = sum_combined_for_entry

        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_USER_FIELDS)
        users = DataFrame.from_records(filtered_entries)

        users['positive_interactions'] = users['useful'] + users['funny'] + users['cool']
        users = users.drop(columns=['useful', 'funny', 'cool'])

        # Normalisation
        users = DataReader.users_normalise_data_quantile_based(users, 'positive_interactions')
        users = DataReader.users_normalise_data_quantile_based(users, 'fans')
        users = DataReader.users_normalise_data_quantile_based(users, 'compliments')

        users = users.set_index('user_id')
        users.columns = [f"user_{column_name}" for column_name in users.columns]

        # Train - testset
        train_users, test_users = train_test_split(users, train_size=0.8)
        return train_users, test_users

    @staticmethod
    def _process_users_split(users, reviews, businesses):
        # Adding user profiles based on labels

        reviews = reviews.join(users, on='user_id', how='inner')
        reviews = reviews.join(businesses, on='business_id', how='inner')
        for column_name in [column_name for column_name in reviews.columns if column_name.startswith("category") or column_name.startswith("attribute")]:
            reviews[column_name] = reviews[column_name] * reviews['stars_normalised']  # Rescale impact label of restaurant to the rating of the user for that restaurant
        drop_columns = [
            column_name for column_name in reviews.columns
            if column_name != 'user_id' and not column_name.startswith("category") and not column_name.startswith(
                "attribute")
        ]
        reviews = reviews.drop(columns=drop_columns)
        user_label_profiles = reviews.groupby('user_id').sum()
        user_review_count = reviews.groupby('user_id').count()['category_food'].rename("user_review_count")

        user_label_profiles = user_label_profiles.join(user_review_count, on='user_id')
        for column_name in tqdm(user_label_profiles.columns,
                                desc="Applying normalisation for user profiles based on labels", leave=False):
            user_label_profiles[column_name] = user_label_profiles.swifter.apply(
                lambda row: row[column_name] / row['user_review_count'], axis=1).astype(
                np.float16)
        users = users.join(user_label_profiles, on='user_id', how='inner')
        users = users.drop(columns=['user_review_count'])
        users = users.astype(np.float16)
        return users

    @staticmethod
    def users_normalise_data_quantile_based(users: DataFrame, column_name: str):
        bottom = users[column_name].quantile(0.10)
        top = users[column_name].quantile(0.90)

        users[column_name] = users[column_name].transform(lambda val: DataReader.quantile_rescale(val, bottom, top))
        return users

    @staticmethod
    def quantile_rescale(val, bottom, top):
        if val <= bottom:
            return 0
        if val >= top:
            return 1
        return (val - bottom) / (top - bottom)

    @staticmethod
    def remove_extremes(val, bottom, top):
        if val < bottom:
            return bottom
        if val > top:
            return top
        return val

    @staticmethod
    def fix_indices(businesses: DataFrame, reviews_train: DataFrame, reviews_test: DataFrame,
                    users_train: DataFrame, users_test: DataFrame) \
            -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Performance 'hack': De originele data van de Yelp Dataset gebruikt strings als IDs. Dit is zeer traag en niet memory-efficient
        Met deze functie worden alle indices vervangen naar ints op een consistente manier
        """

        # Transforming business IDs to integers
        businesses_indices = pd.Series(range(len(businesses)), index=businesses.index).astype(np.uint16)
        businesses = businesses.reset_index(drop=True)
        businesses.index = businesses.index.rename('business_id')
        reviews_train['business_id'] = reviews_train['business_id'].transform(
            lambda b_id: businesses_indices[b_id]).astype(np.uint16)
        reviews_test['business_id'] = reviews_test['business_id'].transform(
            lambda b_id: businesses_indices[b_id]).astype(np.uint16)

        # Transforming user IDs to integers
        unique_users = pd.concat([reviews_train, reviews_test])['user_id'].unique()
        users_indices = pd.Series(range(len(unique_users)), index=unique_users).astype(np.uint32)
        reviews_train['user_id'] = reviews_train['user_id'].transform(lambda u_id: users_indices[u_id]).astype(
            np.uint32)
        reviews_test['user_id'] = reviews_test['user_id'].transform(lambda u_id: users_indices[u_id]).astype(np.uint32)

        # Reviews already have optimized indices

        # Applying original transformation based on reviews on the users
        # and removing users which don't have any reviews
        users_train = users_train.join(users_indices.rename("new_id"), on='user_id', how="inner")
        users_train.index = users_train['new_id']
        users_train.index = users_train.index.rename('user_id')
        users_train = users_train.drop(columns=['new_id'])

        users_test = users_test.join(users_indices.rename("new_id"), on='user_id', how="inner")
        users_test.index = users_test['new_id']
        users_test.index = users_test.index.rename('user_id')
        users_test = users_test.drop(columns=['new_id'])

        return businesses, reviews_train, reviews_test, users_train, users_test
