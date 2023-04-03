import logging
from pathlib import Path

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.neural_network_trainer import NeuralNetworkTrainer
from tools.RestaurantReviewsDataset import RestaurantReviewsDataset


def main():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    businesses, reviews, tips = DataReader().read_data()
    user_profiles = pd.read_parquet("NLP/FIRST_USER_PROFILES.parquet")
    train_test_data = DataPreparer.get_train_test_validate(businesses, reviews, tips, user_profiles)

    model = MultiLayerPerceptronPredictor(input_size=train_test_data[0].columns.size, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    nn_trainer = NeuralNetworkTrainer(*train_test_data)
    # model, optimizer = nn_trainer.train(model, optimizer, epochs=50 * 4, save_to_disk=True)
    model.load(optimizer)
    model.plot_loss_progress(save_location=Path("predictor", "loss.png"))

    model.eval()  # Prepare layers of model for evaluation
    with torch.no_grad():
        testX, testY = train_test_data[1], train_test_data[3]

        testX = testX.head(50)
        testY = testY.head(50)

        dl = RestaurantReviewsDataset(testX.to_numpy(), testY.to_numpy())
        test_loader = DataLoader(dl, batch_size=50)
        for x, y in test_loader:
        # Prepare data
            x, y = DataPreparer.get_tensor_for_ml(x, y)
        # Compute predictions and loss

            predictions = model(x)
            df = pd.DataFrame(data=[predictions.cpu(), y.cpu()])
            print(df)


        # loss = loss_fn(predictions, ratings)
        # Calculate statistics
        # total_loss += loss.item()
        # correct += ((ratings - 0.125 <= predictions) & (predictions <= ratings + 0.125)).type(torch.float).sum().item()


if __name__ == '__main__':
    main()
