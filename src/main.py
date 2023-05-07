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


def parse_data_train_test(train_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], test_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], tuple[dict, dict]]:
    profile_creators_params = get_profile_creators()
    logging.info("Parsing train set")
    training_input, training_output = DataPreparer.transform_data(*train_data, *profile_creators_params)
    logging.info("Parsing test set")
    test_input, test_output = DataPreparer.transform_data(*test_data, *profile_creators_params)
    gc.collect()
    return (training_input, test_input, training_output, test_output), profile_creators_params


def get_profile_creators() -> tuple[dict, dict]:
    up_creator = ProfileCreator(
        model_name="online_model_400top_97.bert",
        use_sentiment_in_scores=False,
        approx_mode=False,
        approx_normalization=True,
        approx_amount_top_n=5,
        filter_useful_topics=False
    ).get_build_parameters()

    rp_creator = ProfileCreator(
        model_name="online_model_50top_85.bert",
        use_sentiment_in_scores=True,
        approx_mode=True,
        approx_normalization=True,
        approx_amount_top_n=5,
        filter_useful_topics=False
    ).get_build_parameters()

    return up_creator, rp_creator


def main_single_model():
    # Parameters
    EPOCHS = 1
    SUB_EPOCHS = 10
    LR = 0.0002

    logging.info("Reading Yelp Dataset")
    train_data, test_data = DataReader().read_data()
    gc.collect()

    with tqdm(total=EPOCHS, desc="Epochs") as p_bar:
        train_test_data, (up_params, rp_params) = parse_data_train_test(train_data, test_data)

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
            train_test_data, (up_params, rp_params) = parse_data_train_test(train_data, test_data)

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
    return 0


if __name__ == '__main__':
    # Note: force manual garbage collection is used to save on memory after heavy RAM and I/O instructions
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%H:%M:%S',
        format='%(asctime)s %(levelname)-8s %(message)s',
    )
    main_single_model()
