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
    businesses, reviews, tips = DataReader().read_data()

    logging.info("Starting training: with user profiles and business profiles")
    for user_profiles_name in tqdm(ProfilesManager().get_user_profiles_names(), desc="User Profiles"):
        for business_profiles_name in tqdm(ProfilesManager().get_business_profiles_names(), desc="Business Profiles"):

            user_profiles = ProfilesManager().get_user_profiles(user_profiles_name)
            business_profiles = ProfilesManager().get_business_profiles(business_profiles_name)

            train_test_data = DataPreparer.get_train_test_validate(businesses, reviews, tips, user_profiles, business_profiles)

            model = MultiLayerPerceptronPredictor(input_size=train_test_data[0].columns.size, output_size=1)
            # Check if model already exists
            save_dir = ConfigParser().get_value('predictor_model', 'model_dir')

            # To skip duplicate work
            short_name_user_profile = "".join(user_profiles_name.split(".")[:-1])
            short_name_business_profile = "".join(business_profiles_name.split(".")[:-1])
            if len([file.name for file in os.scandir(Path(save_dir)) if short_name_user_profile in file.name and short_name_business_profile in file.name]) == 0:  # if not found

                optimizer = optim.Adam(model.parameters(), lr=0.002)
                nn_trainer = NeuralNetworkTrainer(user_profiles_name, business_profiles_name, *train_test_data)
                model, optimizer = nn_trainer.train(model, optimizer, epochs=100, save_to_disk=True, verbose=False)
                model.plot_loss_progress(save_location=Path("predictor", f"loss_mlp_{user_profiles_name}_{business_profiles_name}.png"))
            else:
                logging.info(f"Skipped model {model.get_default_save_location()}, another version already exists")

    # To skip duplicate work, again
    user_profile_names_all = ProfilesManager().get_user_profiles_names()
    user_profile_names = []
    for user_profiles_name in user_profile_names_all:
        save_dir = ConfigParser().get_value('predictor_model', 'model_dir')
        user_profiles_name_short = "".join(user_profiles_name.split(".")[:-1])
        if len([file.name for file in os.scandir(Path(save_dir)) if user_profiles_name_short in file.name and "None" in file.name]) == 0:
            user_profile_names.append(user_profiles_name)

    for user_profiles_name in tqdm(user_profile_names, desc="User Profiles"):
        user_profiles = ProfilesManager().get_user_profiles(user_profiles_name)

        train_test_data = DataPreparer.get_train_test_validate(businesses, reviews, tips, user_profiles)

        model = MultiLayerPerceptronPredictor(input_size=train_test_data[0].columns.size, output_size=1)
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        nn_trainer = NeuralNetworkTrainer(user_profiles_name, None, *train_test_data)
        model, optimizer = nn_trainer.train(model, optimizer, epochs=100, save_to_disk=True, verbose=False)
        model.plot_loss_progress(save_location=Path("predictor", f"loss_mlp_{user_profiles_name}_None.png"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
