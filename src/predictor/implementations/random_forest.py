import gc
import logging
from random import randint

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBRFRegressor

from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from tools.restaurant_profiles_manager import RestaurantProfilesManager
from tools.user_profiles_manager import UserProfilesManager


class RandomForest:

    def __init__(self, input_ml_train: pd.DataFrame, input_ml_test: pd.DataFrame, output_ml_train: pd.Series, output_ml_test: pd.Series):
        self.model = XGBRFRegressor(
            n_estimators=100,
            subsample=0.8,
            colsample_bynode=0.2,
        )  # Parameters optimized by grid search
        self.input_ml_train = input_ml_train
        self.input_ml_test = input_ml_test
        self.output_ml_train = output_ml_train
        self.output_ml_test = output_ml_test
        self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    def _train(self, save_to_disk: bool = True):
        self.model.fit(self.input_ml_train, self.output_ml_train, eval_set=[(self.input_ml_test, self.output_ml_test)])
        if save_to_disk:
            self.model.save_model("random_forest.json")

    def _validate(self, n: int = 500) -> tuple[float, pd.DataFrame]:
        test_input = self.input_ml_test.head(n)
        test_output = self.output_ml_test.head(n)

        predictions = []
        errors = []

        for df_index in test_input.index:
            input_row = test_input[df_index]
            expected_output = test_output[df_index] * 4 + 1
            prediction = self.model.predict(input_row) * 4 + 1
            predictions.append(prediction)
            error = abs(prediction - expected_output)
            errors.append(error)

        validation_results = pd.DataFrame([predictions, test_output, errors], columns=['prediction', 'actual', 'difference'])

        mse = np.mean(np.square(np.array(validation_results['difference'])))
        return float(mse), validation_results

    @staticmethod
    def run():
        train_data, test_data = DataReader().read_data()
        up_params = UserProfilesManager().get_best()
        rp_params = RestaurantProfilesManager().get_best()
        gc.collect()

        training_input, test_input, training_output, test_output = DataPreparer.parse_data_train_test(
            train_data, test_data, (up_params, rp_params), cache_index_if_available=randint(0, 20)
        )
        forest = RandomForest(training_input, test_input, training_output, test_output)
        logging.info("Fitting Random Forest")
        forest._train()
        logging.info("Testing Random Forest")
        results = forest._validate()
        logging.info("Random Forest results:")
        logging.info(f"{results[0]}")
        logging.info(f"{results[1]}")
