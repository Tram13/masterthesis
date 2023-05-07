import gc
import logging
from pathlib import Path

import pandas as pd
from torch import optim
from tqdm import tqdm

from NLP.profiles_creator import ProfileCreator
from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.neural_network_trainer import NeuralNetworkTrainer
from tools.restaurant_profiles_manager import RestaurantProfilesManager
from tools.user_profiles_manager import UserProfilesManager


def parse_data_train_test(train_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], test_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], profile_params: tuple[dict, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logging.info("Parsing train set")
    training_input, training_output = DataPreparer.transform_data(*train_data, *profile_params)
    logging.info("Parsing test set")
    test_input, test_output = DataPreparer.transform_data(*test_data, *profile_params)
    gc.collect()
    return training_input, test_input, training_output, test_output


def main_single_model(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS, LR):
    with tqdm(total=EPOCHS, desc="Epochs", leave=False) as p_bar:
        train_test_data = parse_data_train_test(train_data, test_data, (up_params, rp_params))

        logging.info("Transforming data to DataLoaders")
        nn_trainer = NeuralNetworkTrainer(up_params, rp_params, *train_test_data)
        gc.collect()

        logging.info("Creating Multi-Layer Perceptron model")
        model = MultiLayerPerceptronPredictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1], output_size=1)
        optimizer = optim.Adagrad(model.parameters(), lr=LR)

        logging.info("Starting training")
        model, optimizer = nn_trainer.train(model, optimizer, sub_epochs=SUB_EPOCHS, save_to_disk=(EPOCHS == 1), verbose=True)
        p_bar.update()

        for epoch in range(2, EPOCHS + 1):
            # Creates new user profiles based on subset of train or test data
            train_test_data = parse_data_train_test(train_data, test_data, (up_params, rp_params))

            logging.info("Transforming data to DataLoaders")
            nn_trainer = NeuralNetworkTrainer(up_params, rp_params, *train_test_data)
            gc.collect()

            logging.info("Creating Multi-Layer Perceptron model")

            logging.info("Starting training")
            model, optimizer = nn_trainer.train(model, optimizer, sub_epochs=SUB_EPOCHS, save_to_disk=(epoch == EPOCHS), verbose=True)
            logging.info(f"Current loss history: {[f'{val:.3}' for val in model.loss_history[-5:]]}")
            p_bar.update()
            gc.collect()

        # Save statistics
    model.plot_loss_progress(save_location=Path(f"{str(model.get_default_save_location())[:-2]}.png"))
    gc.collect()


def main_all_models():
    # Parameters
    EPOCHS = 30
    SUB_EPOCHS = 20
    LR = 0.0002

    logging.info("Reading Yelp Dataset")
    train_data, test_data = DataReader().read_data()
    gc.collect()
    for up_params in tqdm(UserProfilesManager(), desc="User Profiles"):
        for rp_params in tqdm(RestaurantProfilesManager(), desc="Restaurant Profiles"):
            main_single_model(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS, LR)
    return 0


if __name__ == '__main__':
    # Note: force manual garbage collection is used to save on memory after heavy RAM and I/O instructions
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%H:%M:%S',
        format='%(asctime)s %(levelname)-8s %(message)s',
    )
    main_all_models()
