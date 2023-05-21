import math

import pandas as pd

from data.data_reader import DataReader


class BiasRecommender:

    def __init__(self):
        self.global_mean = 0

    def run(self) -> pd.DataFrame:
        # Loading data
        train_data, test_data = DataReader().read_data()
        (b_train, r_train, u_train), (b_test, r_test, u_test) = train_data, test_data

        # Calculating global mean
        self.global_mean = r_train['stars_normalised'].mean()
        # Calculating restaurant bias
        restaurant_bias = (r_train.groupby("business_id").mean(numeric_only=True)['stars_normalised'] - self.global_mean).rename("restaurant_bias")
        # Calculating user bias
        ratings_in_testset_per_user = r_test.groupby("user_id").count()['funny_cool'].rename('review_count')
        verwerk_data = r_test[['stars_normalised', 'user_id', "business_id"]].join(restaurant_bias, on='business_id').join(ratings_in_testset_per_user, on='user_id')
        verwerk_data = verwerk_data[verwerk_data['review_count'] > 1]

        all_user_biases_list = []
        to_drop = []
        all_users = r_test['user_id'].unique()
        for current_user_id in all_users:
            reviews_of_user_all = r_test[r_test['user_id'] == current_user_id]
            reviews_of_user_gen = reviews_of_user_all.sample(frac=0.5)
            to_drop.append(reviews_of_user_gen.index)
            user_bias = reviews_of_user_gen.join(restaurant_bias, on='business_id').apply(lambda row: row['stars_normalised'] - self.global_mean - row['restaurant_bias'], axis=1).mean()
            all_user_biases_list.append(user_bias)
        user_bias = pd.Series(data=all_user_biases_list, index=all_users).rename("user_bias").fillna(0)

        # Making predictions
        drops = pd.concat([pd.Series(index=x) for x in to_drop])
        verwerk_data = verwerk_data.drop(drops.index)
        verwerk_data = verwerk_data.join(user_bias, on='user_id')
        predictions = verwerk_data.progress_apply(self._get_bias, axis=1).rename('predicted')
        scaled_predictions = round((predictions * 4) + 1)
        actual = (verwerk_data['stars_normalised'] * 4) + 1
        difference = abs(scaled_predictions - actual).rename("difference")

        # Calculating statistics
        mse = (predictions - verwerk_data['stars_normalised']).transform(lambda x: x * x).mean()
        print(f"MSE: {mse}")
        print(f"adjusted RMSE: {math.sqrt(mse) * 4}")
        results = pd.concat([scaled_predictions, actual, difference], axis=1)
        return results

    def _get_bias(self, row) -> float:
        return self.global_mean + row['restaurant_bias'] + row['user_bias']
