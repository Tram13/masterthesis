import gc
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from NLP.profiles_creator import ProfileCreator
from tools.config_parser import ConfigParser
from tools.restaurant_profiles_manager import RestaurantProfilesManager
from tools.user_profiles_manager import UserProfilesManager


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
        """
        Prepares the provided tensor for GPU learning
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ratings = ratings.unsqueeze(-1)
        restaurant_reviews, ratings = restaurant_reviews.to(device), ratings.to(device)  # Copy to GPU
        return restaurant_reviews, ratings

    @staticmethod
    def transform_data(businesses: pd.DataFrame, reviews: pd.DataFrame, users: pd.DataFrame, up_creator_params: dict, rp_creator_params: dict,
                       profile_size: float = 0.7) -> tuple[pd.DataFrame, pd.Series]:
        logging.info("Splitting in generation and prediction sets")
        reviews_generation, reviews_prediction = DataPreparer.get_profiles_split(reviews, profile_dataframe_size=profile_size)

        logging.info("Creating User Profile")
        user_profiles_nlp = ProfileCreator.load_from_dict(up_creator_params).get_user_profile(reviews_generation)

        logging.info("Creating Restaurant Profile")
        business_profiles_nlp = ProfileCreator.load_from_dict(rp_creator_params).get_restaurant_profile(reviews_generation)

        logging.info("Transforming to ML input")
        input_ml, output_ml = DataPreparer.get_df_for_ml(businesses, reviews_prediction, users, user_profiles_nlp, business_profiles_nlp)
        gc.collect()
        return input_ml, output_ml

    @staticmethod
    def parse_data_train_test(
            train_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
            test_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
            profile_params: tuple[dict, dict],
            cache_index_if_available: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Combineert de aparte Users, Reviews, Restaurants DataFrames naar één DataFrame a.d.h.v. joins
        Daarna wordt bij zowel de train- als testset een deel van de data gebruikt als generation set voor de profielen zoals  gespecifieerd in `profile_params`
        Deze profielen worden dan toegevoegd aan de overige data en teruggegeven
        Zie HS4: Data Flow
        :param train_data:
        :param test_data:
        :param profile_params:
        :param cache_index_if_available:
        :return:
        """

        if cache_index_if_available is not None \
                and profile_params[0] == UserProfilesManager().get_best() \
                and profile_params[1] == RestaurantProfilesManager().get_best():
            try:
                logging.info(f"Reading ML input/output from cache {cache_index_if_available}")
                return DataPreparer._read_nn_cache(cache_index_if_available)
            except OSError:
                logging.warning(f"ML input/output not available in cache {cache_index_if_available}")
                pass
        logging.info("Parsing train set")
        training_input, training_output = DataPreparer.transform_data(*train_data, *profile_params, profile_size=0.5)
        logging.info("Parsing test set")
        test_input, test_output = DataPreparer.transform_data(*test_data, *profile_params, profile_size=0.7)
        gc.collect()
        return training_input, test_input, training_output, test_output

    @staticmethod
    def make_nn_caches(train_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], test_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], up_params: dict,
                       rp_params: dict, n: int = 30, resume_from: int = 0):
        """
        Helper functions to create caches so repeated computations go faster
        """
        for i in tqdm(range(resume_from, n), desc="Creating input/output data"):
            DataPreparer._make_nn_cache(train_data, test_data, up_params, rp_params, i)

    @staticmethod
    def _make_nn_cache(train_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], test_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], up_params: dict,
                       rp_params: dict, i: int):
        save_dir = Path(ConfigParser().get_value("best_profiles", "save_dir"))
        training_input, test_input, training_output, test_output = DataPreparer.parse_data_train_test(train_data, test_data, (up_params, rp_params),
                                                                                                      cache_index_if_available=None)
        sub_dir = Path(save_dir, f"split_{str(i).zfill(5)}")
        os.makedirs(sub_dir, exist_ok=True)
        training_input.to_parquet(Path(sub_dir, "training_input.parquet"), engine="fastparquet")
        test_input.to_parquet(Path(sub_dir, "test_input.parquet"), engine="fastparquet")
        training_output.to_csv(Path(sub_dir, "training_output.csv"))
        test_output.to_csv(Path(sub_dir, "test_output.csv"))

    @staticmethod
    def _read_nn_cache(index: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        save_dir = Path(ConfigParser().get_value("best_profiles", "save_dir"))
        sub_dir = Path(save_dir, f"split_{str(index).zfill(5)}")
        training_input = pd.read_parquet(Path(sub_dir, "training_input.parquet"))
        test_input = pd.read_parquet(Path(sub_dir, "test_input.parquet"))
        training_output = pd.read_csv(Path(sub_dir, "training_output.csv"), index_col=0).squeeze()
        test_output = pd.read_csv(Path(sub_dir, "test_output.csv"), index_col=0).squeeze()
        return training_input, test_input, training_output, test_output
