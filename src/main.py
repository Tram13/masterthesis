import logging
from pathlib import Path

import pandas as pd
from torch import optim

from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.neural_network_trainer import NeuralNetworkTrainer


def main():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    businesses, reviews, tips = DataReader().read_data()
    user_profiles = pd.read_parquet("NLP/FIRST_USER_PROFILES.parquet")
    train_test_data = DataPreparer.get_train_test_validate(businesses, reviews, tips, user_profiles)

    model = MultiLayerPerceptronPredictor(input_size=train_test_data[0].columns.size, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    nn_trainer = NeuralNetworkTrainer(*train_test_data)
    model, optimizer = nn_trainer.train(model, optimizer, epochs=50 * 4, save_to_disk=True)
    model.plot_loss_progress(save_location=Path("predictor", "loss.png"))


if __name__ == '__main__':
    main()
