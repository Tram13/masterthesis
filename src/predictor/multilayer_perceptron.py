import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim

from tools.config_parser import ConfigParser


class MultiLayerPerceptronPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(MultiLayerPerceptronPredictor, self).__init__()
        self.flatten = nn.Flatten()
        # Definition of netwok architecture
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Sigmoid(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 8),
            nn.Sigmoid(),
            nn.Linear(input_size // 8, input_size // 16),
            nn.Sigmoid(),
            nn.Linear(input_size // 16, output_size)
        )

        # Run on GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Statistics
        self.current_epoch = 0
        self.loss_history = []
        self.note = ""
        self.user_profiles_location = "None"
        self.business_profiles_location = "None"
        self.parameters_configuration = ""

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_stack(x)

    def update_epoch(self, loss: float):
        self.loss_history.append(loss)
        self.current_epoch += 1

    def get_default_save_location(self, use_inference_model: bool = False) -> Path:
        save_dir = ConfigParser().get_value('predictor_model', 'model_dir')
        if use_inference_model:
            file_name = ConfigParser().get_value('predictor_model', 'mlp_inference_model_name').split('.')
        else:
            file_name = ConfigParser().get_value('predictor_model', 'mlp_full_model_name').split('.')
        uuid = datetime.now().strftime("%Y-%m-%d_%Hh%M")
        return Path(save_dir, f'{file_name[0]}_{uuid}__{"".join(self.user_profiles_location.split(".")[:-1])}_{"".join(self.business_profiles_location.split(".")[:-1])}.{file_name[1]}')

    @staticmethod
    def get_latest_model_from_default_location(use_inference_model: bool = False) -> Path:  # Only for full model
        save_dir = ConfigParser().get_value('predictor_model', 'model_dir')
        if use_inference_model:
            file_name = ConfigParser().get_value('predictor_model', 'mlp_inference_model_name').split('.')
        else:
            file_name = ConfigParser().get_value('predictor_model', 'mlp_full_model_name').split('.')
        # Search disk for models in that location
        found_models = [str(model.name) for model in list(os.scandir(save_dir))]
        # Just some date parsing to find the most recent model
        found_models = [(model_name[len(file_name[0]) + 1: len(file_name[0]) + 17], model_name) for model_name in found_models]
        found_models = [(datetime.strptime(ts, "%Y-%m-%d_%Hh%M"), model_name) for ts, model_name in found_models]
        found_models.sort(reverse=True)

        return Path(save_dir, found_models[0][1])

    def save(self, optimizer: torch.optim.Optimizer, path: os.PathLike = None, overwrite: bool = True):
        if path is None:
            path = self.get_default_save_location()
            logging.info(f"Using default save location: {path}")

        if os.path.exists(path) and os.path.isfile(path):  # Model already exists
            if overwrite:
                logging.warning(f'Overwriting existing model at {path}')
                os.remove(path)
            else:
                logging.warning(f"Existing model found at {path}. Aborting save!")
                return
        save_directory = os.path.dirname(path)
        if not os.path.exists(save_directory):  # The save directory does not exist
            os.makedirs(save_directory, exist_ok=True)

        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": self.loss_history,
            "note": self.note,
            "user_profiles_location": self.user_profiles_location,
            "business_profiles_location": self.business_profiles_location,
            "parameters_configuration": self.parameters_configuration
        }, path)
        logging.info(f"Model saved at {path}.")

    def save_for_inference(self, path: os.PathLike, overwrite: bool = True):
        if path is None:
            path = self.get_default_save_location(use_inference_model=True)
            logging.info(f"Using default save location: {path}")

        if os.path.exists(path) and os.path.isfile(path):  # Model already exists
            if overwrite:
                logging.warning(f'Overwriting existing model at {path}')
                os.remove(path)
            else:
                logging.warning(f"Existing model found at {path}. Aborting save!")
                return
        save_directory = os.path.dirname(path)
        if not os.path.exists(save_directory):  # The save directory does not exist
            os.makedirs(save_directory, exist_ok=True)

        torch.save(self.state_dict(), path)
        logging.info(f"Inference Model saved at {path}.")

    def load(self, optimizer: torch.optim.Optimizer, path: os.PathLike = None) -> tuple[nn.Module, torch.optim.Optimizer]:
        if path is None:
            path = self.get_latest_model_from_default_location()
            logging.info(f"Using default save location: {path}")
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.loss_history = checkpoint['loss']
        self.note = checkpoint['note']
        self.user_profiles_location = checkpoint['user_profiles_location']
        self.business_profiles_location = checkpoint['business_profiles_location']
        self.parameters_configuration = checkpoint['parameters_configuration']
        logging.info(f"Model loaded from {path}.")

        return self, optimizer

    def load_for_inference(self, path: os.PathLike = None) -> nn.Module:
        if path is None:
            path = self.get_latest_model_from_default_location(use_inference_model=True)
            logging.info(f"Using default save location: {path}")
        if not os.path.exists(path) or not os.path.isfile(path):  # Model not found
            logging.error(f"Model not found at {path}. Aborting!")
            raise FileNotFoundError(f"Could not load {path}")
        self.load_state_dict(torch.load(path))
        logging.info(f"Inference Model loaded from {path}.")
        return self

    def plot_loss_progress(self, title: str = "MLP", display_note: bool = True, save_location: os.PathLike = None):
        fig, ax = plt.subplots()
        ax.plot(range(self.current_epoch), self.loss_history)
        if display_note:
            title = f"{title} Loss: {self.note}"
        ax.set_title(title)
        ax.set_xlabel("Epoch")

        if save_location:
            fig.savefig(save_location)

        return ax
