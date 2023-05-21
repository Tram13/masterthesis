# author: Arnoud De Jonge

from typing import List

import numpy as np

from genre_converter import convert_category_to_int, convert_category_to_string
from rating import Rating


class Business:
    def __init__(self, business_id: int, name: str, categories: List[str], ratings: List[Rating]):
        self.business_id = business_id
        self.name = name
        self.categories = [convert_category_to_int(category) for category in categories]
        self.rating_user_ids = set([rating.user_id for rating in ratings])
        self.ratings_dict_by_user = {rating.user_id: rating for rating in ratings}
        if not ratings:
            self.average_score = 0
        else:
            self.average_score = np.average([rating.score for rating in ratings])

    def __str__(self):
        return f'Business ID {self.business_id}: {self.name} --- {[convert_category_to_string(genre) for genre in self.categories]}\n'
