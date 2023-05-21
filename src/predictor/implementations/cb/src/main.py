# author: Arnoud De Jonge
import math
from statistics import mean

from tqdm import tqdm

from src.parser import parse
from src.recommender import best_basic_content_recommender, calculate_idf
from src.validation import get_random_testset

if __name__ == '__main__':
    for MIN_RATINGS_IN_TRAIN in [1, 4]:
        print("Parsing...")
        users_list, businesses_dict, rating_dict, amount_of_ratings = parse()
        print("Generating testset")
        users_list, testset = get_random_testset(users_list, MIN_RATINGS_IN_TRAIN)
        errors = []
        mses = []
        actuals = []
        predictions = []
        print("Validating")
        for rating in tqdm(testset, desc="Validating"):
            idf = calculate_idf(businesses_dict)
            user = None
            u_i = 0
            while user is None:
                if users_list[u_i].user_id == rating.user_id:
                    user = users_list[u_i]
                u_i += 1
            scores = best_basic_content_recommender(businesses_dict, user, True, idf)

            business = None
            b_i = 0
            while business is None:
                if scores[b_i][0] == rating.business_id:
                    business = scores[b_i]
                b_i += 1

            prediction = business[1] * 4 + 1
            actual = (rating.score + 0.5) * 4 + 1
            error = abs(actual - prediction)
            mse = math.pow(error, 2)
            predictions.append(prediction)
            actuals.append(actual)
            errors.append(error)
            mses.append(mse)

        with open(f"results_cb_min_{MIN_RATINGS_IN_TRAIN}_rating_train.csv", 'w+') as results_file:
            results_file.write("prediction,actual,error,mse\n")
            for i in range(len(errors)):
                results_file.write(f"{predictions[i]},{actuals[i]},{errors[i]},{mses[i]}\n")
        print(f"MSE: {mean(mses)}")
        print(f"RMSE: {math.sqrt(mean(mses))}")
