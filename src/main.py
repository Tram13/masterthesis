import logging
from pathlib import Path

from torch import optim
from tqdm import tqdm

from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.neural_network_trainer import NeuralNetworkTrainer
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
            optimizer = optim.Adam(model.parameters(), lr=0.002)

            nn_trainer = NeuralNetworkTrainer(user_profiles_name, business_profiles_name, *train_test_data)
            model, optimizer = nn_trainer.train(model, optimizer, epochs=100, save_to_disk=True, verbose=False)
            model.plot_loss_progress(save_location=Path("predictor", f"loss_mlp_{user_profiles_name}_{business_profiles_name}.png"))

    logging.info("Starting training: with user profiles, NO business profiles")
    for user_profiles_name in tqdm(ProfilesManager().get_user_profiles_names(), desc="User Profiles, No business profiles"):
        user_profiles = ProfilesManager().get_user_profiles(user_profiles_name)

        train_test_data = DataPreparer.get_train_test_validate(businesses, reviews, tips, user_profiles)

        model = MultiLayerPerceptronPredictor(input_size=train_test_data[0].columns.size, output_size=1)
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        nn_trainer = NeuralNetworkTrainer(user_profiles_name, 'None', *train_test_data)
        model, optimizer = nn_trainer.train(model, optimizer, epochs=100, save_to_disk=True, verbose=False)
        model.plot_loss_progress(save_location=Path("predictor", f"loss_mlp_{user_profiles_name}_None.png"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
