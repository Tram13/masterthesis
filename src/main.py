import gc
import logging
import os
from pathlib import Path

import pandas as pd
from torch import optim

from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.neural_network_trainer import NeuralNetworkTrainer
from tools.config_parser import ConfigParser
from tools.profiles_manager import ProfilesManager


def get_data_train_test():
    logging.info("Reading Yelp Dataset")
    (b_train, r_train, u_train), (b_test, r_test, u_test) = DataReader().read_data()
    gc.collect()
    logging.info("Parsing train set")
    training_input, training_output = transform_data(b_train, r_train, u_train)
    logging.info("Parsing test set")
    test_input, test_output = transform_data(b_test, r_test, u_test)
    gc.collect()
    return training_input, test_input, training_output, test_output


def transform_data(businesses, reviews, users):
    logging.info("Splitting in generation and prediction sets")
    reviews_generation, reviews_prediction = DataPreparer.get_profiles_split(reviews, profile_dataframe_size=0.7)

    logging.info("Creating profiles")
    # TODO: onderstaand vervangen met nieuwe code
    user_profiles_nlp = ProfilesManager().get_user_profiles()
    business_profiles_nlp = ProfilesManager().get_business_profiles()

    logging.info("Transforming to ML input")
    input_ml, output_ml = DataPreparer.get_df_for_ml(businesses, reviews_prediction, users, user_profiles_nlp, business_profiles_nlp)
    gc.collect()
    return input_ml, output_ml


def main_single_model():
    # Parameters
    EPOCHS = 1000
    LR = 0.0002

    logging.info("Initialising")
    # Initialisation TODO: uitzoeken wat hiervan nog nodig is
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
    train_test_data = get_data_train_test()
    nn_trainer = NeuralNetworkTrainer(user_profiles_name, business_profiles_name, *train_test_data)
    logging.info("Cleaning up memory on CPU")
    gc.collect()

    # Creating model
    model = MultiLayerPerceptronPredictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1], output_size=1)
    optimizer = optim.Adagrad(model.parameters(), lr=LR)

    # Training on first part of dataset
    logging.info("Starting training")
    model, optimizer = nn_trainer.train(model, optimizer, epochs=EPOCHS, save_to_disk=True, verbose=True)

    # Save statistics
    model.plot_loss_progress(save_location=Path("predictor", f"loss_mlp_{user_profiles_name}_{business_profiles_name}.png"))
    return 0


if __name__ == '__main__':
    # Note: force manual garbage collection is used to save on memory after heavy RAM and I/O instructions
    logging.basicConfig(level=logging.INFO)
    main_single_model()
