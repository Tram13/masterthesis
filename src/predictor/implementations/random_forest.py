import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from xgboost import XGBRFRegressor


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

    def train(self, save_to_disk: bool = True):
        self.model.fit(self.input_ml_train, self.output_ml_train, eval_set=[(self.input_ml_test, self.output_ml_test)])
        if save_to_disk:
            self.model.save_model("testlocatie.json")  # TODO: deftige locatie opgeven

    def validate(self, n: int = 500) -> tuple[float, pd.DataFrame]:
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
