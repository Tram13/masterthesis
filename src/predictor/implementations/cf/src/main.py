# author: Arnoud De Jonge
import math
import os
from statistics import mean

from parser import parse
from recommender import *
from validation import get_random_testset


def user_user_test():
    print("User-User")
    predictions = []
    actuals = []
    diffs = []
    mses = []

    for test_rating in tqdm(testset, desc="Validating User-User"):
        test_user = users_dict[test_rating.user_id]
        test_restaurant = businesses_dict[test_rating.business_id]
        score_with_restaurant = UUCF_recommendation(test_user, users_dict, businesses_dict, 20)

        # Find restaurant prediction:
        prediction_test_restaurant = [
                                         score for score, restaurant in score_with_restaurant
                                         if restaurant.business_id == test_restaurant.business_id
                                     ][0] * 4 + 1
        # Actual score
        actual = test_rating.score * 4 + 1
        # Difference
        diff = abs(prediction_test_restaurant - actual)
        # Saving results
        predictions.append(prediction_test_restaurant)
        actuals.append(actual)
        diffs.append(diff)
        mses.append(math.pow(diff, 2))

    with open(f"results_cf_user_user_RMSE={math.sqrt(mean(mses))}.csv", 'w+') as results_file:
        results_file.write("prediction,actual,error,mse\n")
        for i in range(len(diffs)):
            results_file.write(f"{predictions[i]},{actuals[i]},{diffs[i], mses[i]}\n")
    print(f"MSE: {mean(mses)}")
    print(f"RMSE: {math.sqrt(mean(mses))}")


def item_item_test():
    print("Item-Item")
    predictions = []
    actuals = []
    diffs = []
    mses = []

    for test_rating in tqdm(testset, desc="Validating Item-Item"):
        test_user = users_dict[test_rating.user_id]
        test_restaurant = businesses_dict[test_rating.business_id]
        score_with_restaurant = IICF_recommendation(test_user, businesses_dict, top_k_similar=20)

        # Find restaurant prediction:
        prediction_test_restaurant = \
            [score for score, restaurant in score_with_restaurant if
             restaurant.business_id == test_restaurant.business_id][
                0] * 4 + 1
        # Actual score
        actual = test_rating.score * 4 + 1
        # Difference
        diff = abs(prediction_test_restaurant - actual)
        # Saving results
        predictions.append(prediction_test_restaurant)
        actuals.append(actual)
        diffs.append(diff)
        mses.append(math.pow(diff, 2))

    with open(f"results_cf_item_item_RMSE={math.sqrt(mean(mses))}.csv", 'w+') as results_file:
        results_file.write("prediction,actual,error,mse\n")
        for i in range(len(diffs)):
            results_file.write(f"{predictions[i]},{actuals[i]},{diffs[i], mses[i]}\n")
    print(f"MSE: {mean(mses)}")
    print(f"RMSE: {math.sqrt(mean(mses))}")


if __name__ == '__main__':
    print("Parsing...")
    users_dict, businesses_dict = parse()
    print("Generating test set")
    users_dict, businesses_dict, testset = get_random_testset(users_dict, businesses_dict, 5)

    print('Done parsing')
    item_item_test()
