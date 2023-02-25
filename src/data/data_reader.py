import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing

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

    RELEVANT_REVIEW_FIELDS = [  # TODO: uitzoeken of een gebruiker meerdere reviews over hetzelfde restaurant kan hebben?
        'review_id',
        'user_id',
        'business_id',
        'stars',
        'useful',
        'funny',  # TODO: onderzoeken of 'cool' en 'funny' velden nuttig zijn? Indien ja, combineren met useful
        'cool',  # TODO: onderzoeken of 'cool' en 'funny' velden nuttig zijn? Indien ja, combineren met useful
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
        'review_count',  # TODO: Check of dit correct is?
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
            data_path = Path(ConfigParser.get_value('data', 'data_path'))
        self._assert_correct_data_dir(data_path)
        self.file_paths = [Path(data_path, file) for file in self.EXPECTED_FILES]

    def read_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        businesses = self._parse_businesses(self.file_paths[0])
        reviews = self._parse_reviews(self.file_paths[2])
        tips = self._parse_tips(self.file_paths[3])
        users = self._parse_users(self.file_paths[4])
        return businesses, reviews, tips, users

    # Check if ALL and NOTHING BUT the data files are present in the provided directory
    def _assert_correct_data_dir(self, data_path):
        if set(os.listdir(data_path)) != set(self.EXPECTED_FILES):
            raise DataException(
                f"\n\nInvalid files found in {data_path}:\n"
                f"\tFound: {os.listdir(data_path)}\n"
                f"\tExpected: {self.EXPECTED_FILES}"
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
            .rename(f"category_{category.replace(' ', '_').lower()}") for category in categories_appearances.keys()
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
                )  # '2' seems to be the most common value, thus default
            elif index == 14:  # attribute_noiselevel
                onehot_attributes[index] = onehot_attributes[index].map(
                    lambda x: 0 if x == 'quiet' else (
                        0.33 if x == 'average' else (0.67 if x == 'loud' else (1 if x == 'very_loud' else 0.33)))
                )  # 'average' is the default value
            else:
                onehot_attributes[index] = onehot_attributes[index].map(
                    lambda x: 1 if x is True else (0 if x is False else 0.5)
                )

        businesses = pd.concat([businesses, *onehot_attributes], axis=1)
        businesses = businesses.drop(columns=['attributes'])
        businesses = businesses.set_index('business_id')

        # ADD CHECK-INS
        checkins = DataReader._parse_checkins(self.file_paths[1])
        businesses = businesses.join(checkins, on='business_id')

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
        avg_checkins_per_week = (amount_of_checkins / amount_of_weeks).replace([np.inf, -np.inf], 0)
        avg_checkins_per_week_normalised = pd.Series(
            data=preprocessing.MinMaxScaler().fit_transform(avg_checkins_per_week.to_numpy().reshape(-1, 1)).flatten(),
            name="average_checkins_per_week_normalised")

        checkins = pd.concat([checkins, avg_checkins_per_week_normalised], axis=1)
        checkins = checkins.drop(columns=['date'])
        checkins = checkins.set_index('business_id')
        return checkins

    @staticmethod
    def _parse_reviews(file_location: os.PathLike) -> pd.DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_REVIEW_FIELDS)
        reviews: pd.DataFrame = pd.DataFrame.from_records(filtered_entries)
        return reviews

    @staticmethod
    def _parse_tips(file_location: os.PathLike) -> pd.DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_TIP_FIELDS)
        tips: pd.DataFrame = pd.DataFrame.from_records(filtered_entries)
        return tips

    @staticmethod
    def _parse_users(file_location: os.PathLike) -> pd.DataFrame:
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
        users: pd.DataFrame = pd.DataFrame.from_records(filtered_entries)
        return users
