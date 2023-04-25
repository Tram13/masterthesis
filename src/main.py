import logging
import os
from pathlib import Path

from torch import optim
from tqdm import tqdm

from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.neural_network_trainer import NeuralNetworkTrainer
from tools.config_parser import ConfigParser
from tools.profiles_manager import ProfilesManager


def main():

    logging.info("Starting training: with user profiles and business profiles")
    user_profiles_name = "BASIC_USER_PROFILES_400_no_sentiment.parquet"
    for business_profiles_name in tqdm(ProfilesManager().get_business_profiles_names(), desc="Business Profiles"):
        # To skip duplicate work
        short_name_user_profile = "".join(user_profiles_name.split(".")[:-1])
        short_name_business_profile = "".join(business_profiles_name.split(".")[:-1])
        # Check if model already exists
        save_dir = ConfigParser().get_value('predictor_model', 'model_dir')
        if len([file.name for file in os.scandir(Path(save_dir)) if short_name_user_profile in file.name and short_name_business_profile in file.name]) == 0:  # if not found

            businesses, reviews, tips, users = DataReader().read_data()
            user_profiles = ProfilesManager().get_user_profiles(user_profiles_name)
            business_profiles = ProfilesManager().get_business_profiles(business_profiles_name)

            logging.info("Processing original data")
            train_test_data = DataPreparer.get_train_test_validate(businesses, reviews, tips, users, user_profiles, business_profiles)

            # Cleanup memory
            logging.info("Cleaning up original data")
            del businesses
            del reviews
            del tips
            del users
            del user_profiles
            del business_profiles

            nn_trainer = NeuralNetworkTrainer(user_profiles_name, business_profiles_name, *train_test_data)

            logging.info("Cleaning up memory on CPU")
            del train_test_data

            model = MultiLayerPerceptronPredictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1], output_size=1)

            optimizer = optim.Adam(model.parameters(), lr=0.002)
            model, optimizer = nn_trainer.train(model, optimizer, epochs=100, save_to_disk=True, verbose=False)
            model.plot_loss_progress(save_location=Path("predictor", f"loss_mlp_{user_profiles_name}_{business_profiles_name}.png"))
        else:
            logging.info(f"Skipped model {short_name_user_profile}  | {short_name_business_profile}, another version already exists")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
