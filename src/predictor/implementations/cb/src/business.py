# author: Arnoud De Jonge

from typing import List

import numpy as np

from genre_converter import convert_category_to_int, convert_category_to_string, CATEGORY_STRING_TO_INT


class Business:
    def __init__(self, business_id: int, name: str, categories: List[str]):
        self.business_id = business_id
        self.name = name
        self.categories = [convert_category_to_int(category) for category in categories]
        self.one_hot_encoded = self._one_hot_encode_business()
        self.one_hot_encoded_normalized = self._one_hot_encode_business(normalized=True)

    def _one_hot_encode_business(self, normalized=False) -> np.ndarray:
        output = np.zeros((len(CATEGORY_STRING_TO_INT),), dtype=float)
        output[self.categories] += 1
        if normalized and len(self.categories):
            output = np.divide(output, np.sqrt(len(self.categories)))
        return output

    def __str__(self):
        return f'Business ID {self.business_id}: {self.name} --- {[convert_category_to_string(genre) for genre in self.categories]}\n'


class BusinessDictionary:
    def __init__(self, businesses: List[Business]):
        self.businesses = {business.business_id: business for business in businesses}

    def __str__(self):
        return "".join([str(business) for business in self.businesses.values()])
