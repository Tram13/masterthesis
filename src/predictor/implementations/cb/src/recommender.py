# author: Arnoud De Jonge

import functools
import operator
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np

from genre_converter import CATEGORY_STRING_TO_INT, CATEGORY_INT_TO_STRING
from src.business import Business
from src.user import User


def get_genre_frequency(movie_dict: Dict[int, Business]):
    genre_ids = functools.reduce(operator.iconcat, [movie.categories for movie in movie_dict.values()], [])
    genres = [CATEGORY_INT_TO_STRING[genre_id] for genre_id in genre_ids]
    return Counter(genres)


def calculate_idf(business_dict: Dict[int, Business]):
    idf = np.ones((len(CATEGORY_STRING_TO_INT),), dtype=float)
    for genre, freqency in get_genre_frequency(business_dict).items():
        idf[CATEGORY_STRING_TO_INT[genre]] /= freqency
    return idf


def one_hot_encode_movie_dict(movie_dict: Dict[int, Business], normalized=False) -> Tuple[List[int], np.ndarray]:
    order = []
    one_hot_movies = []
    for key, value in movie_dict.items():
        order.append(key)
        if normalized:
            one_hot_movies.append(value.one_hot_encoded_normalized)
        else:
            one_hot_movies.append(value.one_hot_encoded)
    return order, np.array(one_hot_movies)


def best_basic_content_recommender(businesses_dict: Dict[int, Business], user: User, normalized=False, idf=None) -> List[Tuple[int, float]]:
    if normalized:
        user_profile = np.array(user.profile_normalized)
    else:
        user_profile = np.array(user.profile)
    order, one_hot_encoded_movies = one_hot_encode_movie_dict(businesses_dict, normalized=normalized)
    # MxG @ (1xG)^t = Mx1
    if idf is not None:
        user_profile = np.multiply(user_profile, idf)
    basic_recommender = np.dot(one_hot_encoded_movies, user_profile.transpose())
    return sorted([(movie_id, float(basic_recommender[index][0])) for index, movie_id in enumerate(order)],
                  key=lambda t: (t[1], t[0]), reverse=True)


def diversify_with_mmr(related_items: List[Business], user: User, num_recommendations: int, lam: float = 0.5, idf=None):
    diversified_items = []
    while len(diversified_items) < num_recommendations and related_items:
        best_item_score = None
        best_item_index = 0
        for index, item in enumerate(related_items):
            relevance_to_query = calculate_relevance_to_user_profile(item, np.multiply(user.profile_normalized, idf)[0])
            similarity_with_selected_items = calculate_max_similarity_with_other_items(item, [business for score, business in diversified_items])
            combined_score = lam * relevance_to_query - (1 - lam) * similarity_with_selected_items

            if best_item_score is None or combined_score > best_item_score:
                best_item_score = combined_score
                best_item_index = index

        diversified_items.append((best_item_score, related_items.pop(best_item_index)))

    return diversified_items


def calculate_relevance_to_user_profile(item: Business, user_profile_normalized: np.ndarray):
    return cosine_similarity(item.one_hot_encoded_normalized, user_profile_normalized)


def calculate_max_similarity_with_other_items(item: Business, other_items: List[Business]):
    if not other_items:
        return 0
    return max([cosine_similarity(item.one_hot_encoded_normalized, other_item.one_hot_encoded_normalized)
                for other_item in other_items if other_item != item])


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
