# author: Arnoud De Jonge
import gc
from collections import defaultdict
from pathlib import Path

import pandas as pd

from business import Business
from rating import Rating
from user import User


def parse_ratings() -> defaultdict[int, list]:
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
    gc.collect()
    return ratings_dict


def user_rating_dict_to_business_id_rating_dict(user_rating_dict: defaultdict[int, list]):
    business_ratings_dict = defaultdict(list)
    for user_id, rating_list in user_rating_dict.items():
        for rating in rating_list:
            business_ratings_dict[rating.business_id].append(rating)
    return business_ratings_dict


def parse_businesses(rating_dict: defaultdict[int, list]) -> dict[int, Business]:
    print("Parsing businesses")
    business_id_rating_dict = user_rating_dict_to_business_id_rating_dict(rating_dict)

    businesses = pd.read_parquet(Path("..", "data", "businesses.parquet"))
    businesses = [
        Business(
            business_id=index,
            name=data['name'],
            categories=[column_name for column_name in data.index if column_name.startswith("category") and data[column_name] == 1],
            ratings=business_id_rating_dict[index]
        )
        for index, data in businesses.iterrows()
    ]
    gc.collect()
    return {business.business_id: business for business in businesses}


def parse_users(rating_dict: dict[int, list[Rating]]) -> dict[int, User]:
    print("Parsing users")
    return {user_id: User(
        user_id=user_id,
        ratings={rating.business_id: rating for rating in rating_list}
    )
        for user_id, rating_list in rating_dict.items()}


def parse():
    rating_dict = parse_ratings()
    movie_dict = parse_businesses(rating_dict=rating_dict)
    user_dict = parse_users(rating_dict=rating_dict)
    return user_dict, movie_dict
