import gc
import logging

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from NLP.profiles_creator import ProfileCreator


class DataPreparer:
    @staticmethod
    def get_profiles_split(reviews: pd.DataFrame, profile_dataframe_size: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the reviews in 2 parts: one part used for generating the user and business profiles, the other for rating predictions
        :param reviews: The reviews dataframe to split into two parts
        :param profile_dataframe_size: The size of the DataFrame for generation of user and business profiles, after splitting
        :return: The splitted reviews DataFrame
        """
        # Calculate absolute sizes of generation set and prediction set
        profile_generation_size = int(len(reviews) * profile_dataframe_size)
        prediction_size = len(reviews) - profile_generation_size

        # Select 1 review for each user and for each restaurant
        drop_for_users = reviews.drop_duplicates(subset=['user_id']).index
        drop_for_restaurant = reviews.drop_duplicates(subset=['business_id']).index
        not_selectable = drop_for_users.union(drop_for_restaurant)

        # Remove reviews for profile generation set from selectable set
        reviews_selectable = reviews.drop(not_selectable)

        # Random sample from selectable reviews to serve as prediction set
        reviews_prediction = reviews_selectable.sample(min(prediction_size, len(reviews_selectable)))
        # Inverse of the sampled reviews serve as generation set
        reviews_profile_generation = reviews.drop(reviews_prediction.index)

        return reviews_profile_generation, reviews_prediction

    @staticmethod
    def get_df_for_ml(businesses: pd.DataFrame, reviews_prediction: pd.DataFrame, users: pd.DataFrame, user_profiles: pd.DataFrame,
                      business_profiles: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Transforms the segregated parsed data into a structure to use for Machine Learning: DataFrame represents the input, Series represents the output
        :return: A pd.DataFrame where each row represents an input for the ML model, A pd.Series where each row represents the expected output for the ML model
        """
        users.columns = [f"user_{column_name}" if not column_name.startswith("user") else column_name for column_name in users.columns]
        user_profiles.columns = [f"user_profile_{column_id}" for column_id in user_profiles.columns]
        reviews = reviews_prediction.join(user_profiles, on='user_id', how='inner')
        businesses = businesses.drop(columns=['name', 'city'])
        if business_profiles is not None:
            business_profiles.columns = [f"business_profile_{column_id}" for column_id in business_profiles.columns]
            businesses = businesses.join(business_profiles, on='business_id', how='inner')
        user_reviewed_restaurant = reviews[['user_id', 'business_id', 'stars_normalised', *user_profiles.columns]]
        user_reviewed_restaurant = user_reviewed_restaurant.join(businesses, on='business_id', how="inner")

        user_reviewed_restaurant = user_reviewed_restaurant.join(users, on='user_id', how="inner")
        user_reviewed_restaurant = user_reviewed_restaurant.set_index(['user_id', 'business_id'], append=True)

        ml_data = user_reviewed_restaurant.reset_index()
        output_ml = ml_data['stars_normalised']
        input_ml = ml_data.drop(columns=['stars_normalised', 'review_id', 'user_id', 'business_id'])

        return input_ml, output_ml

    @staticmethod
    def get_tensor_for_ml(restaurant_reviews: torch.Tensor, ratings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ratings = ratings.unsqueeze(-1)
        restaurant_reviews, ratings = restaurant_reviews.to(device), ratings.to(device)  # Copy to GPU
        return restaurant_reviews, ratings

    @staticmethod
    def transform_data(businesses: pd.DataFrame, reviews: pd.DataFrame, users: pd.DataFrame, up_creator: ProfileCreator, rp_creator: ProfileCreator) -> tuple[pd.DataFrame, pd.Series, str, str]:
        logging.info("Splitting in generation and prediction sets")
        reviews_generation, reviews_prediction = DataPreparer.get_profiles_split(reviews, profile_dataframe_size=0.7)

        logging.info("Creating User Profile")

        user_profiles_nlp = up_creator.get_user_profile(reviews_generation)
        user_profiles_parameters = up_creator.get_parameters_string()

        logging.info("Creating Restaurant Profile")

        business_profiles_nlp = rp_creator.get_restaurant_profile(reviews_generation)
        business_profiles_parameters = rp_creator.get_parameters_string()

        logging.info("Transforming to ML input")
        input_ml, output_ml = DataPreparer.get_df_for_ml(businesses, reviews_prediction, users, user_profiles_nlp, business_profiles_nlp)
        gc.collect()
        return input_ml, output_ml, user_profiles_parameters, business_profiles_parameters
