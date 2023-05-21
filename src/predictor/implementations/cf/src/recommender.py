# author: Arnoud De Jonge
from collections import defaultdict
from typing import Set, Dict, List

import numpy as np
from tqdm import tqdm

from business import Business
from user import User


def get_overlapping_ratings_movie_ids(user_a: User, user_u: User) -> Set[int]:
    return user_a.ratings_business_ids.intersection(user_u.ratings_business_ids)


def calculate_significance_weighting(amount_common_items: int, y: int = 10) -> float:
    return min(y, amount_common_items) / y


def pearson_correlation_with_significance_weighting(user_a: User, user_u: User) -> float:
    return pearson_correlation_without_significance_weighting(user_a, user_u) * calculate_significance_weighting(
        len(get_overlapping_ratings_movie_ids(user_a, user_u)))


def pearson_correlation_without_significance_weighting(user_a: User, user_u: User) -> float:
    overlapping_movie_ids = get_overlapping_ratings_movie_ids(user_a, user_u)

    # not many in common movies
    if len(overlapping_movie_ids) <= 1:
        return 0

    # rating score of overlapping movies
    overlapping_ratings_scores_user_a = np.array([user_a.ratings[movie_id].score for movie_id in overlapping_movie_ids])
    overlapping_ratings_scores_user_u = np.array([user_u.ratings[movie_id].score for movie_id in overlapping_movie_ids])

    # average of the rating score of alL movies
    average_user_a = user_a.average_rating_score
    average_user_u = user_u.average_rating_score

    # every rating score - the average
    overlapping_ratings_scores_user_a_minus_average = overlapping_ratings_scores_user_a - average_user_a
    overlapping_ratings_scores_user_u_minus_average = overlapping_ratings_scores_user_u - average_user_u

    # calculate nominator
    nominator = np.sum(
        overlapping_ratings_scores_user_a_minus_average * overlapping_ratings_scores_user_u_minus_average)

    # calculate denominator
    denominator = np.sqrt(np.sum(
        overlapping_ratings_scores_user_a_minus_average * overlapping_ratings_scores_user_a_minus_average)) * np.sqrt(
        np.sum(overlapping_ratings_scores_user_u_minus_average * overlapping_ratings_scores_user_u_minus_average))

    # denominator == 0
    if denominator == 0:
        return 0

    # calculate nominator/denominator
    return nominator / denominator


def UUCF_recommendation(user: User, user_dict: Dict[int, User], movie_dict: Dict[int, Business], top_k_users: int = 20):
    user_similarities = [(similar_user, pearson_correlation_with_significance_weighting(user, similar_user))
                         for similar_user in user_dict.values() if similar_user.user_id != user.user_id]

    # sort similar users by their score (tie -> smallest id)
    user_similarities.sort(key=lambda t: (t[1], -t[0].user_id), reverse=True)

    # only take positive similarity
    user_similarities = [u for u in user_similarities if u[1] > 0]

    # recommendation
    recommendations_with_score = []
    for movie_id, movie in movie_dict.items():
        # user has already rated this movie
        if user.contains_rating_for_movie_id(movie_id):
            continue

        # get top k users who also rated the movie
        top_k_user_similarities_who_rated_movie = [u for u in user_similarities if
                                                   u[0].contains_rating_for_movie_id(movie_id)][:top_k_users]

        # no neighbours
        if not top_k_user_similarities_who_rated_movie:
            recommendations_with_score.append((user.average_rating_score, movie))
        else:
            nominator = np.sum(
                [(similar_user.ratings[movie_id].score - similar_user.average_rating_score) * similarity_score
                 for similar_user, similarity_score in top_k_user_similarities_who_rated_movie])

            denominator = np.sum([similarity_score for _, similarity_score in top_k_user_similarities_who_rated_movie])
            recommendations_with_score.append((user.average_rating_score + nominator / denominator, movie))

    # sort the recommendations based on score (tie: lowest id)
    recommendations_with_score.sort(key=lambda t: (t[0], -t[1].business_id), reverse=True)

    # return the sorted items
    return recommendations_with_score


def UUCF_recommendation_top_n(user: User, user_dict: Dict[int, User], movie_dict: Dict[int, Business],
                              top_k_users: int = 20,
                              top_n_items: int = 10):
    return UUCF_recommendation(user, user_dict, movie_dict, top_k_users)[:top_n_items]


def item_item_similarity(item_x: Business, item_y: Business) -> float:
    # all user_ids who rated movie x
    user_ids_who_rated_x = item_x.rating_user_ids
    # all user_ids who rated movie y
    user_ids_who_rated_y = item_y.rating_user_ids
    # all user_ids who rated movie x and y
    user_ids_who_rated_both_x_and_y = user_ids_who_rated_x.intersection(user_ids_who_rated_y)

    # all rating scores for x by users who rated movie x
    rated_by_user_score_x = np.array([item_x.ratings_dict_by_user[user_id].score for user_id in
                                      user_ids_who_rated_x])
    # all rating scores for y by users who rated movie y
    rated_by_user_score_y = np.array([item_y.ratings_dict_by_user[user_id].score for user_id in
                                      user_ids_who_rated_y])

    # all rating scores minus the average score for that item
    rated_by_user_score_x_minus_average = rated_by_user_score_x - item_x.average_score
    rated_by_user_score_y_minus_average = rated_by_user_score_y - item_y.average_score

    # denominator part for item x
    denominator_item_x = np.sqrt(np.dot(rated_by_user_score_x_minus_average, rated_by_user_score_x_minus_average))
    # denominator part for item y
    denominator_item_y = np.sqrt(np.dot(rated_by_user_score_y_minus_average, rated_by_user_score_y_minus_average))
    # denominator
    denominator = denominator_item_x * denominator_item_y

    # denominator == 0 => return 0
    if denominator == 0:
        return 0

    # all rating scores for item x by users who rated movie x and y
    both_rated_by_user_score_x = np.array([item_x.ratings_dict_by_user[user_id].score for user_id in
                                           user_ids_who_rated_both_x_and_y])
    # all rating scores for item y by users who rated movie x and y
    both_rated_by_user_score_y = np.array([item_y.ratings_dict_by_user[user_id].score for user_id in
                                           user_ids_who_rated_both_x_and_y])

    #  all rating scores for item by users who rated movie x and y minus the average
    both_rated_by_user_score_x_minus_average = both_rated_by_user_score_x - item_x.average_score
    both_rated_by_user_score_y_minus_average = both_rated_by_user_score_y - item_y.average_score

    # numerator
    numerator = np.dot(both_rated_by_user_score_x_minus_average, both_rated_by_user_score_y_minus_average)

    return numerator / denominator


def get_item_item_similarity_from_model(model: Dict[int, Dict[int, float]], movie_id_one: int,
                                        movie_id_two: int) -> float:
    return model[movie_id_one][movie_id_two] if movie_id_one < movie_id_two else model[movie_id_two][movie_id_one]


def get_top_k_similar_items(user: User, movie: Business, movie_dict: Dict[int, Business], top_k: int = 20):
    # get similarity scores
    item_item_similarities_with_movie = [
        (item_item_similarity(movie_dict[user_rated_movie_id], movie), user_rated_movie_id)
        for user_rated_movie_id in user.ratings_business_ids]

    # sort them by similarity score (tie: lowest movie_id)
    item_item_similarities_with_movie.sort(key=lambda t: (t[0], -t[1]), reverse=True)

    # get top-k similar items (with score > 0) rated by the user with it's rating
    return [
               (similarity_score, movie_dict[user_rated_movie_id].ratings_dict_by_user[user.user_id].score)
               for similarity_score, user_rated_movie_id in item_item_similarities_with_movie if similarity_score > 0
           ][:top_k]


def IICF_recommendation(user: User, movie_dict: Dict[int, Business], top_k_similar: int = 20):
    user_rated_movie_ids = user.ratings_business_ids

    recommendations_with_score = []
    for movie_id, movie in tqdm(movie_dict.items(), desc="Comparing against all Restaurants", leave=False):
        # user has already rated this movie
        if movie_id in user_rated_movie_ids:
            continue

        # get top k similar items
        top_k_similar_items = get_top_k_similar_items(user, movie, movie_dict, top_k=top_k_similar)

        # no similar items
        if not top_k_similar_items:
            score = 0
        else:
            numerator = np.sum(
                [similarity_score * rating_score for similarity_score, rating_score in top_k_similar_items])
            denominator = np.sum([np.abs(similarity_score) for similarity_score, _ in top_k_similar_items])
            score = numerator / denominator

        recommendations_with_score.append((score, movie))

    # sort the recommendations based on score (tie: lowest id)
    recommendations_with_score.sort(key=lambda t: (t[0], -t[1].business_id), reverse=True)

    # return the sorted items
    return recommendations_with_score


def IICF_recommendation_top_n(user: User, movie_dict: Dict[int, Business], model: Dict[int, Dict[int, float]],
                              top_k_similar: int = 20,
                              top_n_items: int = 10):
    return IICF_recommendation(user, movie_dict, model, top_k_similar)[:top_n_items]


def shopping_basket_recommendation(basket_movie_ids: List[int], movie_dict: Dict[int, Business],
                                   model: Dict[int, Dict[int, float]], top_n_items: int = 10,
                                   positive_only: bool = True):
    recommendations_with_score = []
    for movie_id, movie in movie_dict.items():

        # movie already in shopping basket
        if movie_id in basket_movie_ids:
            continue

        # get similarity of current movie with each item in the basket
        item_similarities = [
            get_item_item_similarity_from_model(model, basket_movie_id_item, movie_id)
            for basket_movie_id_item in basket_movie_ids
        ]

        # if we only use positive similarities, filter out the negative ones
        if positive_only:
            item_similarities = [item_similarity for item_similarity in item_similarities if item_similarity > 0]

        # score is sum of all (positive) similarities
        score = np.sum(item_similarities)

        recommendations_with_score.append((score, movie))

    # sort the recommendations based on score (tie: lowest id)
    recommendations_with_score.sort(key=lambda t: (t[0], -t[1].business_id), reverse=True)

    # return the top n items
    return recommendations_with_score[:top_n_items]


def hybrid_recommendation(user: User, user_dict: Dict[int, User], movie_dict: Dict[int, Business],
                          model: Dict[int, Dict[int, float]],
                          top_k_similar: int = 20, weight_uucf: float = 0.5, weight_iicf: float = 0.5):
    uucf_recommendations = UUCF_recommendation(user, user_dict, movie_dict, top_k_similar)
    iicf_recommendations = IICF_recommendation(user, movie_dict, model, top_k_similar)

    hybrid_recommendation_dictionary = defaultdict(int)
    # add uucf score to hybrid
    for uucf_score, movie in uucf_recommendations:
        hybrid_recommendation_dictionary[movie.business_id] += weight_uucf * uucf_score

    # add iicf score to hybrid
    for iicf_score, movie in iicf_recommendations:
        hybrid_recommendation_dictionary[movie.business_id] += weight_iicf * iicf_score

    # collect the recommendations from the dictionary
    recommendations_with_score = [(score, movie_dict[movie_id]) for movie_id, score in
                                  hybrid_recommendation_dictionary.items()]

    # sort the recommendations based on score (tie: lowest id)
    recommendations_with_score.sort(key=lambda t: (t[0], -t[1].business_id), reverse=True)

    # return the sorted items
    return recommendations_with_score


def hybrid_recommendation_top_n(user: User, user_dict: Dict[int, User], movie_dict: Dict[int, Business],
                                model: Dict[int, Dict[int, float]],
                                top_k_similar: int = 20, weight_uucf: float = 0.5, weight_iicf: float = 0.5,
                                top_n_items: int = 10):
    return hybrid_recommendation(user, user_dict, movie_dict, model, top_k_similar, weight_uucf, weight_iicf)[:top_n_items]
