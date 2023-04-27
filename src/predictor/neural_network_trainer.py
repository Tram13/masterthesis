import logging
import math
import os
from typing import Union

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn import Module, MSELoss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_preparer import DataPreparer
from tools.RestaurantReviewsDataset import RestaurantReviewsDataset


class NeuralNetworkTrainer:
    __slots__ = ['train_loader', 'test_loader', 'user_profiles_location', 'business_profiles_location']
    BATCH_SIZE = 8192

    def __init__(self, user_profiles_path: Union[os.PathLike, str], business_profiles_path: Union[os.PathLike, str, None], input_ml_train: pd.DataFrame, input_ml_test: pd.DataFrame, output_ml_train: pd.DataFrame,
                 output_ml_test: pd.DataFrame):

        train_data = RestaurantReviewsDataset(input_ml_train.to_numpy(), output_ml_train.to_numpy())
        test_data = RestaurantReviewsDataset(input_ml_test.to_numpy(), output_ml_test.to_numpy())

        self.train_loader = DataLoader(train_data, batch_size=self.BATCH_SIZE)
        self.test_loader = DataLoader(test_data, batch_size=self.BATCH_SIZE)

        self.user_profiles_location = user_profiles_path
        self.business_profiles_location = business_profiles_path if business_profiles_path not in {"None", "", "none", None} else "None"

    @staticmethod
    def _get_parameters_string(model: Module, optimizer: Optimizer, epochs: int):
        # Save all parameters as string, which will then be saved with the model
        note = {
            'model': model.__class__,
            'optimizer': optimizer.__class__,
            'learning_rate': optimizer.defaults['lr'],
            'epochs': epochs
        }
        return str(note)

    @staticmethod
    def train_epoch(model: Module, optimizer: Optimizer, dataloader: DataLoader, loss_fn) -> float:
        model.train()  # Prepare layers of model for training
        # Prepare statistics
        total_loss = 0
        num_batches = len(dataloader)
        for restaurant_reviews, ratings in tqdm(dataloader, desc=f"Training network in batches", leave=False):
            # Prepare data
            restaurant_reviews, ratings = DataPreparer.get_tensor_for_ml(restaurant_reviews, ratings)
            # Compute predictions and loss
            predictions = model(restaurant_reviews)
            loss = loss_fn(predictions, ratings)
            # Compute statistics
            total_loss += math.sqrt(loss.item())  # TODO: aanpassen als we geen MSE meer gebruiken
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = total_loss / num_batches
        return mean_loss

    @staticmethod
    def validate_epoch(model: Module, dataloader: DataLoader, loss_fn) -> tuple[float, float]:
        model.eval()  # Prepare layers of model for evaluation
        # Prepare statistics
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for restaurant_reviews, ratings in dataloader:
                # Prepare data
                restaurant_reviews, ratings = DataPreparer.get_tensor_for_ml(restaurant_reviews, ratings)
                # Compute predictions and loss
                predictions = model(restaurant_reviews)
                loss = loss_fn(predictions, ratings)
                # Calculate statistics
                total_loss += loss.item()
                correct += ((ratings - 0.125 <= predictions) & (predictions <= ratings + 0.125)).type(torch.float).sum().item()

        mean_loss = total_loss / num_batches
        accuracy = correct / size
        return mean_loss, accuracy

    def train(self, model: Module, optimizer: Optimizer, epochs: int = 100, plot_loss: bool = False, save_to_disk: bool = True, verbose=True) -> tuple[Module, Optimizer]:
        model.parameters_configuration = self._get_parameters_string(model, optimizer, epochs)
        model.user_profiles_location = self.user_profiles_location
        model.business_profiles_location = self.business_profiles_location
        if verbose:
            logging.info(f"Training network with following parametes: {model.parameters_configuration}")
        history = {
            'train_loss': [],
            'test_loss': [],
            'test_acc': []
        }
        criterion = MSELoss()

        epochs_with_progressbar = tqdm(range(epochs), desc="Epochs")
        for epoch in epochs_with_progressbar:
            train_loss = self.train_epoch(model, optimizer, self.train_loader, criterion)
            test_loss, test_acc = self.validate_epoch(model, self.test_loader, criterion)

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            model.update_epoch(test_loss)
            epochs_with_progressbar.set_description_str(f"Epochs (RMSEloss of last 5 epochs: {[f'{val:.3}' for val in history['test_loss'][-5:]]}")
            if plot_loss and epoch % (epochs // (100 / 5)) == 0:  # Every 5%
                model.plot_loss_progress()
                plt.show()

        if save_to_disk:
            model.save(optimizer, verbose=verbose)
        if plot_loss:
            model.plot_loss_progress()
        return model, optimizer
