# author: Arnoud De Jonge

from typing import Dict

import numpy as np

from rating import Rating


class User:
    def __init__(self, user_id: int, ratings: Dict[int, Rating]):
        self.user_id = user_id
        self.ratings = ratings
        self.ratings_business_ids = set(ratings.keys())
        self.average_rating_score = np.average([rating.score for rating in ratings.values()])

    def update(self):
        self.ratings_business_ids = set(self.ratings.keys())
        self.average_rating_score = np.average([rating.score for rating in self.ratings.values()])

    def __str__(self):
        return f'User: {self.user_id}\n' \
               f'Rated Businesses: {self.ratings_business_ids}\n'

    def contains_rating_for_movie_id(self, business_id: int) -> bool:
        return business_id in self.ratings_business_ids
