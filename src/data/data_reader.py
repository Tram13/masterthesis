from pathlib import Path

import json
import pandas as pd
import os

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

    RELEVANT_BUSINESS_FIELDS = [
        'business_id',
        'name',
        'city',
        'stars',
        'review_count',
        'attributes',
        'categories'
    ]

    RELEVANT_CHECKIN_FIELDS = [
        'business_id',
        'date'  # TODO: uitzoeken hoe deze lijst van dates kan gebruikt worden (bvb: average checkins per week?)
    ]

    RELEVANT_REVIEW_FIELDS = [
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

    RELEVANT_TIP_FIELDS = [
        'user_id',
        'business_id',
        'text',
        'date',
        'compliment_count'
    ]

    RELEVANT_USER_FIELDS = [
        'user_id',
        'name',
        'review_count',
        'yelping_since',
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

    def read_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        businesses = self._parse_businesses(self.file_paths[0])
        checkins = self._parse_checkins(self.file_paths[1])
        reviews = self._parse_reviews(self.file_paths[2])
        tips = self._parse_tips(self.file_paths[3])
        users = self._parse_users(self.file_paths[4])
        return businesses, checkins, reviews, tips, users

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

    @staticmethod
    def _parse_businesses(file_location: os.PathLike) -> pd.DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_BUSINESS_FIELDS)
        businesses: pd.DataFrame = pd.DataFrame.from_records(filtered_entries)
        # TODO: attributes en categories deftig parsen
        return businesses

    @staticmethod
    def _parse_checkins(file_location: os.PathLike) -> pd.DataFrame:
        entries = DataReader._get_entries_from_file(file_location)
        filtered_entries = DataReader._filter_entries(entries, DataReader.RELEVANT_CHECKIN_FIELDS)
        checkins: pd.DataFrame = pd.DataFrame.from_records(filtered_entries)
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
