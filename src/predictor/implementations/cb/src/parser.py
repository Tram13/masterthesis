# author: Arnoud De Jonge
import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.business import Business, BusinessDictionary
from src.rating import Rating
from src.user import User


def parse_businesses() -> Dict[int, Business]:
    print("Parsing businesses")
    businesses = pd.read_parquet(Path("..", "data", "businesses.parquet"))
    businesses = [
        Business(
            business_id=index,
            name=data['name'],
            categories=[column_name for column_name in data.index if column_name.startswith("category") and data[column_name] == 1]
        )
        for index, data in businesses.iterrows()
    ]
    gc.collect()
    return BusinessDictionary(businesses).businesses


def parse_ratings() -> tuple[defaultdict[int, list], int]:
    print("Parsing ratings")
    reviews = pd.read_parquet(Path("..", "data", "reviews.parquet"))
    ratings_dict = defaultdict(list)
    for index, data in reviews.iterrows():
        rating = Rating(
            user_id=data['user_id'],
            business_id=data['business_id'],
            score=data['stars_normalised']
        )
        ratings_dict[data['user_id']].append(rating)
    amount_of_ratings = len(reviews)
    gc.collect()
    return ratings_dict, amount_of_ratings


def parse_users(rating_dict: Dict[int, List[Rating]], businesses_dict) -> List[User]:
    print("Parsing users")
    return [
        User(
            user_id=user_id,
            ratings=ratings,
            businesses_dict=businesses_dict
        )
        for user_id, ratings in rating_dict.items()]


def parse():
    businesses_dict = parse_businesses()
    rating_dict, amount_of_ratings = parse_ratings()
    users_list = parse_users(rating_dict=rating_dict, businesses_dict=businesses_dict)
    return users_list, businesses_dict, rating_dict, amount_of_ratings
