# author: Arnoud De Jonge
from typing import Dict

from tqdm import tqdm

from business import Business
from recommender import item_item_similarity


class IICF_model:
    def __init__(self, movie_dict: Dict[int, Business]):
        self.model = {movie_id: {} for movie_id in movie_dict.keys()}
        self._fill_model(movie_dict)

    def _fill_model(self, movie_dict: Dict[int, Business]):
        for movie_id_one, movie_one in tqdm(movie_dict.items()):
            for movie_id_two, movie_two in movie_dict.items():
                if movie_id_one < movie_id_two:
                    self.model[movie_id_one][movie_id_two] = item_item_similarity(movie_one, movie_two)
