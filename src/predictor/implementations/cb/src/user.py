# author: Arnoud De Jonge

from typing import List, Dict

import numpy as np

from genre_converter import CATEGORY_STRING_TO_INT
from src.business import Business
from src.rating import Rating


# saving the ratings in the user class is easy to work with but it is less efficient
class User:
    def __init__(self, user_id: int, ratings: List[Rating], businesses_dict: Dict[int, Business]):
        self.user_id = user_id
        self.ratings = ratings
        self.ratings.sort(key=lambda rating: rating.business_id)
        self.profile = self._create_profile(businesses_dict)
        self.profile_normalized = self._create_profile(businesses_dict, normalized=True)

    def __str__(self):
        return f'User: {self.user_id}\n' \
               f'Profile: {list(self.profile[0])}\n'

    def _create_profile(self, businesses_dict, normalized=False) -> np.ndarray:
        profile = [0 for _ in range(len(CATEGORY_STRING_TO_INT))]
        for rating in self.ratings:
            business = businesses_dict[rating.business_id]
            for category in business.categories:
                if normalized:
                    profile[category] += rating.score * business.one_hot_encoded_normalized[category]
                else:
                    profile[category] += rating.score
        return np.array([profile])

    def contains_rating(self, business_id: int):
        # we can do a binary search because we sorted the ratings on movie_id
        low = 0
        high = len(self.ratings) - 1
        while low <= high:
            mid = (high + low) // 2

            # ignore left half
            if self.ratings[mid].business_id < business_id:
                low = mid + 1

            # ignore right half
            elif self.ratings[mid].business_id > business_id:
                high = mid - 1

            # found
            else:
                return True

        # not present
        return False
