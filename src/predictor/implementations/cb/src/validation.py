from random import randint

from src.rating import Rating
from src.user import User


def get_random_testset(users: list[User], min_ratings_in_train: int, n=500) -> tuple[list[User], list[Rating]]:
    testset = []
    amount_of_users = len(users)
    while len(testset) < n:
        user_id = randint(0, amount_of_users - 1)
        amount_of_ratings = len(users[user_id].ratings)
        if amount_of_ratings > min_ratings_in_train:
            rating_index = randint(0, amount_of_ratings - 1)
            rating = users[user_id].ratings.pop(rating_index)
            testset.append(rating)

    return users, testset
