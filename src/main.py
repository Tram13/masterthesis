import gc
import logging
import os
from pathlib import Path

from torch import optim

from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.neural_network_trainer import NeuralNetworkTrainer
from tools.config_parser import ConfigParser
from tools.profiles_manager import ProfilesManager


def get_data(part: int = None, total_parts: int = None):
    businesses, reviews, tips, users = DataReader().read_data(part=part, total_parts=total_parts)
    logging.info("Reading profiles")
    user_profiles = ProfilesManager().get_user_profiles()
    business_profiles = ProfilesManager().get_business_profiles()

    logging.info("Processing original data")
    train_test_data = DataPreparer.get_train_test_validate(businesses, reviews, tips, users, user_profiles, business_profiles)

    # Cleanup memory
    logging.info("Cleaning up original data")
    gc.collect()
    return train_test_data


def main_single_model():
    # Parameters
    TOTAL_PARTS = 1  # For dataset splitting
    EPOCHS = 1000
    LR = 0.0002

    logging.info("Starting training: with user profiles and business profiles")
    # Initialisation
    user_profiles_name = ConfigParser().get_value("cache", "best_user")
    business_profiles_name = ConfigParser().get_value("cache", "best_business")
    short_name_user_profile = Path(ConfigParser().get_value("cache", "best_user")).stem
    short_name_business_profile = Path(ConfigParser().get_value("cache", "best_business")).stem
    logging.info(f"Running {user_profiles_name} | {business_profiles_name}")

    # Check if model already exists
    save_dir = ConfigParser().get_value('predictor_model', 'model_dir')
    if len([file.name for file in os.scandir(Path(save_dir)) if short_name_user_profile in file.name and short_name_business_profile in file.name]) != 0:
        logging.info(f"Skipped model {short_name_user_profile}  | {short_name_business_profile}, another version already exists")
        return 1

    # Getting first part of dataset
    train_test_data = get_data(part=1, total_parts=TOTAL_PARTS)
    nn_trainer = NeuralNetworkTrainer(user_profiles_name, business_profiles_name, *train_test_data)
    logging.info("Cleaning up memory on CPU")
    gc.collect()

    # Creating model
    model = MultiLayerPerceptronPredictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1], output_size=1)
    optimizer = optim.Adagrad(model.parameters(), lr=LR)

    # Training on first part of dataset
    logging.info(f"Training with part 1/{TOTAL_PARTS} of dataset")
    model, optimizer = nn_trainer.train(model, optimizer, epochs=EPOCHS, save_to_disk=1 == TOTAL_PARTS, verbose=True)

    # Memory cleanup after run
    gc.collect()

    for index_part in range(2, TOTAL_PARTS + 1):
        # Getting next part of dataset
        train_test_data = get_data(part=index_part, total_parts=TOTAL_PARTS)
        nn_trainer = NeuralNetworkTrainer(user_profiles_name, business_profiles_name, *train_test_data)
        logging.info("Cleaning up memory on CPU")
        del train_test_data

        logging.info(f"Training with part {index_part}/{TOTAL_PARTS} of dataset")
        model, optimizer = nn_trainer.train(model, optimizer, epochs=EPOCHS, save_to_disk=index_part == TOTAL_PARTS, verbose=True)

        # Memory cleanup after run
        del nn_trainer.train_loader.dataset
        del nn_trainer.test_loader.dataset
        del nn_trainer.train_loader
        del nn_trainer.test_loader
        del nn_trainer
        gc.collect()

    # Save statistics
    model.plot_loss_progress(save_location=Path("predictor", f"loss_mlp_{user_profiles_name}_{business_profiles_name}.png"))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main_single_model()
