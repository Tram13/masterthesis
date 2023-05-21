from random import randint, choice

from business import Business
from rating import Rating
from user import User


def get_random_testset(users: dict[int, User], businesses: dict[int, Business], min_reviews_left_in_test: int, n: int = 500) -> tuple[dict[int, User], dict[int, Business], list[Rating]]:
    testset = []
    while len(testset) < n:
        user_id = choice(list(users.keys()))
        amount_of_ratings = len(users[user_id].ratings)
        if amount_of_ratings > min_reviews_left_in_test - 1:
            rating_key = choice(list(users[user_id].ratings.keys()))
            rating = users[user_id].ratings.pop(rating_key)
            testset.append(rating)
            users[user_id].update()

    for rating in testset:
        businesses[rating.business_id].rating_user_ids.remove(rating.user_id)
        businesses[rating.business_id].ratings_dict_by_user.pop(rating.user_id)

    return users, businesses, testset
