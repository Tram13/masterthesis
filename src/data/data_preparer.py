import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class DataPreparer:

    @staticmethod
    def get_train_test_validate(businesses: pd.DataFrame, reviews: pd.DataFrame, tips: pd.DataFrame):
        ml_data = DataPreparer.get_df_for_ml(businesses, reviews, tips).reset_index()
        output_ml = ml_data['stars_normalised']
        input_ml = ml_data.drop(columns=['stars_normalised', 'review_id', 'user_id', 'business_id'])
        input_ml_train, input_ml_test, output_ml_train, output_ml_test = train_test_split(input_ml, output_ml, test_size=0.2)
        return input_ml_train, input_ml_test, output_ml_train, output_ml_test

    @staticmethod
    def get_df_for_ml(businesses: pd.DataFrame, reviews: pd.DataFrame, tips: pd.DataFrame) -> pd.DataFrame:
        """
        Get all combinations of business attributes and user attributes
        :return: A DataFrame where each row represents an input for the ML model
        """
        # TODO: hier en daar is nog normalisatie nodig! zie test.ipynb
        businesses = businesses.drop(columns=['name', 'city'])
        # users = users.drop(columns=['name', 'friends'])  # TODO: hoe friends verwerken?
        user_reviewed_restaurant = reviews[['user_id', 'business_id', 'stars_normalised']]

        user_reviewed_restaurant = user_reviewed_restaurant.join(businesses, on='business_id')
        # user_reviewed_restaurant = user_reviewed_restaurant.join(users, on='user_id')
        user_reviewed_restaurant.set_index(['user_id', 'business_id'], append=True)

        return user_reviewed_restaurant

    @staticmethod
    def get_tensor_for_ml(restaurant_reviews: torch.Tensor, ratings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ratings = ratings.unsqueeze(-1)
        restaurant_reviews, ratings = restaurant_reviews.to(device), ratings.to(device)  # Copy to GPU
        return restaurant_reviews, ratings
