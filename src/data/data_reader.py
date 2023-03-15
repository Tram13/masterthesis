import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

from src.data.data_exception import DataException
from src.tools.config_parser import ConfigParser


# This class works with the Yelp data set format. Download the data from https://www.yelp.com/dataset.
# It is expected that the data files are unpacked in the location defined by config.ini.
class DataReader:
    EXPECTED_FILES = [
        'yelp_academic_dataset_business.json',
        'yelp_academic_dataset_checkin.json',
        'yelp_academic_dataset_review.json',
        'yelp_academic_dataset_tip.json',
        'yelp_academic_dataset_user.json'
    ]

    RELEVANT_BUSINESS_FIELDS = [  # TODO: set van maken, ook van alles hieronder
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

    RELEVANT_TIP_FIELDS = [  # TODO: prob dit mergen samen met de review, indien dit veld bestaat
        'user_id',
        'business_id',
        'text',
        'date',
        'compliment_count'
    ]

    RELEVANT_USER_FIELDS = [
        'user_id',
        'name',
        'review_count',  # See FAQ on Yelp.com, not entirely correct with the actual amount of reviews we have
        'friends',
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
        self._assert_correct_data_dir()
        self.file_paths = [Path(data_path, file) for file in self.EXPECTED_FILES]

    def read_data(self, use_cache: bool = True, save_as_cache: bool = True) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if use_cache:
            businesses, reviews, tips = self._read_from_cache()
        else:
            businesses, reviews, tips = self._read_from_disk()
        if save_as_cache:
            businesses.to_parquet(Path(self.cache_path, 'businesses.parquet'), engine='fastparquet')
            reviews.to_parquet(Path(self.cache_path, 'reviews.parquet'), engine='fastparquet')
            tips.to_parquet(Path(self.cache_path, 'tips.parquet'), engine='fastparquet')
            # users.to_parquet(Path(self.cache_path, 'users.parquet'), engine='fastparquet')
        return businesses, reviews, tips

    def _read_from_disk(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with tqdm(total=3, desc="Reading files from disk") as p_bar:
            p_bar.set_postfix_str('(current: businesses)')
            businesses = self._parse_businesses(self.file_paths[0])
            p_bar.update()
            p_bar.set_postfix_str('(current: reviews)')
            reviews = self._parse_reviews(self.file_paths[2], businesses)
            p_bar.update()
            p_bar.set_postfix_str('current: tips')
            tips = self._parse_tips(self.file_paths[3], businesses)
            p_bar.update()
            # p_bar.set_postfix_str('current: users')
            # users = self._parse_users(self.file_paths[4])
            # p_bar.update()
        return businesses, reviews, tips

    def _read_from_cache(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            businesses = pd.read_parquet(Path(self.cache_path, 'businesses.parquet'), engine='fastparquet')
            reviews = pd.read_parquet(Path(self.cache_path, 'reviews.parquet'), engine='fastparquet')
            tips = pd.read_parquet(Path(self.cache_path, 'tips.parquet'), engine='fastparquet')
            # users = pd.read_parquet(Path(self.cache_path, 'users.parquet'), engine='fastparquet')
        except OSError:
            print("Could not reach caches!", file=sys.stderr)
            businesses, reviews, tips = self._read_from_disk()
        return businesses, reviews, tips

    # Check if ALL and NOTHING BUT the data files are present in the provided directory
    def _assert_correct_data_dir(self):
        cache_already_exists = os.path.isdir(self.cache_path)
        if not cache_already_exists:
            os.mkdir(self.cache_path)
        if set(os.listdir(self.data_path)) != {self.cache_path.name, *self.EXPECTED_FILES}:
            if not cache_already_exists:  # We just created the cache directory
                os.rmdir(self.cache_path)
            raise DataException(
                f"\n\nInvalid files/directories found in {self.data_path}:\n"
                f"\tFound: {os.listdir(self.data_path)}\n"
                f"\tExpected: {self.cache_path.name, *self.EXPECTED_FILES}"
            )

    @staticmethod
    def _get_entries_from_file(file_path: os.PathLike) -> list[dict[str, any]]:
        with open(file_path, mode='r', encoding='utf-8') as json_file:
            return [json.loads(line) for line in json_file]

    # Keep only the required fields in the provided dictionary
    @staticmethod
    def _filter_entries(entries: list[dict[str, any]], fields: list[str]) -> list[dict[str, any]]:
        return [{key: entry[key] for key in fields} for entry in entries]

    def _parse_businesses(self, file_location: os.PathLike) -> pd.DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_BUSINESS_FIELDS)
        businesses: pd.DataFrame = pd.DataFrame.from_records(filtered_entries)

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
        businesses = pd.concat([businesses, *normalised_series], axis=1)  # TODO: what if there are no reviews for a business?

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
                lambda business_attributes:
                business_attributes[attribute] if attribute in business_attributes
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
        businesses['average_checkins_per_week_normalised'] = businesses['average_checkins_per_week_normalised'].replace([np.nan], 0)

        return businesses

    @staticmethod
    def _parse_checkins(file_location: os.PathLike) -> pd.DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_CHECKIN_FIELDS)
        checkins = pd.DataFrame.from_records(filtered_entries)
        checkins['date'] = checkins['date'].map(
            lambda datelist: [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in datelist.split(', ')]
        )

        first_checkins = checkins['date'].map(lambda datelist: min(datelist))  # First check-in per restaurant
        last_checkin = checkins['date'].map(lambda datelist: max(datelist)).max()  # Last checkin date in entire dataset
        #  Amount of weeks between first check-in and last possible check-in
        amount_of_weeks = (last_checkin - first_checkins).map(lambda x: x.days / 7)
        amount_of_checkins = checkins['date'].transform(len)
        avg_checkins_per_week = (amount_of_checkins / amount_of_weeks).replace([np.inf, -np.inf, np.nan], 0)
        avg_checkins_per_week_normalised = pd.Series(
            data=preprocessing.MinMaxScaler().fit_transform(avg_checkins_per_week.to_numpy().reshape(-1, 1)).flatten(),
            name="average_checkins_per_week_normalised"
        )

        checkins = pd.concat([checkins, avg_checkins_per_week_normalised], axis=1)
        checkins = checkins.drop(columns=['date'])
        checkins = checkins.set_index('business_id')
        return checkins

    @staticmethod
    def _parse_reviews(file_location: os.PathLike, businesses: pd.DataFrame) -> pd.DataFrame:
        """
        :param file_location: Location of the reviews dataset in json format
        :param businesses: The businesses DataFrame as parsed by `_parse_businesses()`
        :return: A DataFrame containing all reviews for the given businesses
        """
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_REVIEW_FIELDS)
        reviews = pd.DataFrame.from_records(filtered_entries)

        # TODO: extra normalisation: aan de hand van de gemiddelde rating van de gebruiker
        normalised_column = pd.Series(
            data=
            preprocessing.MinMaxScaler().fit_transform(
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

        return reviews

    @staticmethod
    def _parse_tips(file_location: os.PathLike, businesses: pd.DataFrame) -> pd.DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_TIP_FIELDS)
        tips = pd.DataFrame.from_records(filtered_entries)
        tips = tips[tips['business_id'].isin(businesses.index)]  # Only keep tips for restaurants
        tips['text'] = tips['text'].astype("string")
        return tips

    @staticmethod
    def _parse_users(file_location: os.PathLike) -> pd.DataFrame:
        # Currently not used in training of the model!
        # entries = DataReader._get_entries_from_file(file_location)
        # # Combine all compliments
        # compliment_fields = [
        #     'compliment_hot',
        #     'compliment_more',
        #     'compliment_profile',
        #     'compliment_cute',
        #     'compliment_list',
        #     'compliment_note',
        #     'compliment_plain',
        #     'compliment_cool',
        #     'compliment_funny',
        #     'compliment_writer',
        #     'compliment_photos'
        # ]
        # combined_compliments = DataReader._filter_entries(entries, compliment_fields)
        # combined_compliments = [sum(x.values()) for x in combined_compliments]
        # for entry, sum_combined_for_entry in zip(entries, combined_compliments):
        #     entry['compliments'] = sum_combined_for_entry
        #
        # filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_USER_FIELDS)
        # users = pd.DataFrame.from_records(filtered_entries)
        # users['friends'] = users['friends'].map(lambda friend_str: friend_str.split(', '))
        #
        # users = users.rename(columns={'review_count': 'user_review_count'})
        # users['name'] = users['name'].astype("string")
        # users = users.set_index('user_id')
        # return users
        raise NotImplementedError

    def fix_indices(self, businesses: pd.DataFrame, reviews: pd.DataFrame, tips: pd.DataFrame, users: pd.DataFrame) -> \
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # TODO: fix
        raise NotImplementedError
