import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn

from tools.config_parser import ConfigParser


class MultiLayerPerceptronPredictor(nn.Module):

    def __init__(self, input_size: int, version: int):
        super(MultiLayerPerceptronPredictor, self).__init__()

        if type(self) == MultiLayerPerceptronPredictor:
            raise NotImplementedError("Cannot initialise base class for MultiLayer Perceptron model")

        self.flatten = nn.Flatten()

        # Statistics
        self.current_epoch = 0
        self.loss_history = []
        self.note = ""
        self.user_profiles_params = {}
        self.business_profiles_params = {}
        self.parameters_configuration = ""

        # Config parameters
        self.input_size = input_size
        self.version = version
        self.optimizer = "ADAGRAD"  # Default optimizer
        self.lr = 0  # Default LR

        # Needs to be implemented by inheritance class
        self.linear_stack = None
        self.device = None

    def update_epoch(self, loss: float):
        self.loss_history.append(loss)
        self.current_epoch += 1

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_stack(x)

    def get_default_save_location(self, use_inference_model: bool = False) -> Path:
        save_dir = ConfigParser().get_value('predictor_model', 'model_dir')
        if use_inference_model:
            file_name = ConfigParser().get_value('predictor_model', 'mlp_inference_model_name').split('.')
        else:
            file_name = ConfigParser().get_value('predictor_model', 'mlp_full_model_name').split('.')
        uuid = datetime.now().strftime("%Y-%m-%d_%Hh%M")
        return Path(save_dir, f'{file_name[0]}{self.version}_{uuid}__EPOCHS={self.current_epoch}_LR={self.lr}_LOSS={self.loss_history[-1]:.3}.{file_name[1]}')

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
        found_models = [(model_name[len(file_name[0]) + 2: len(file_name[0]) + 17], model_name) for model_name in found_models]
        found_models = [(datetime.strptime(ts, "%Y-%m-%d_%Hh%M"), model_name) for ts, model_name in found_models]
        found_models.sort(reverse=True)

        return Path(save_dir, found_models[0][1])

    def save(self, optimizer: torch.optim.Optimizer, path: os.PathLike = None, overwrite: bool = True, verbose=True):
        if path is None:
            path = self.get_default_save_location()
            if verbose:
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
            "user_profiles_params": self.user_profiles_params,
            "business_profiles_params": self.business_profiles_params,
            "parameters_configuration": self.parameters_configuration,
            "input_size": self.input_size,
            "version": self.version,
            "optimizer": self.optimizer,
            "lr": self.lr
        }, path)

        with open(f"{str(path)[:-3]}.txt", 'w+', encoding='utf-8') as params_file:
            params_file.write(f'{self.user_profiles_params}\n')
            params_file.write(f'{self.business_profiles_params}\n')

        if verbose:
            logging.info(f"Model saved at {path}.")

    @staticmethod
    def load(optimizer: torch.optim.Optimizer, path: os.PathLike, model_class: nn.Module) -> tuple[nn.Module, torch.optim.Optimizer]:
        checkpoint = torch.load(path)

        input_size = checkpoint['input_size']
        loaded_model = model_class(input_size)

        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_model.current_epoch = checkpoint['epoch']
        loaded_model.loss_history = checkpoint['loss']
        loaded_model.note = checkpoint['note']
        loaded_model.version = checkpoint['version']
        loaded_model.user_profiles_params = checkpoint['user_profiles_params']
        loaded_model.business_profiles_params = checkpoint['business_profiles_params']
        loaded_model.parameters_configuration = checkpoint['parameters_configuration']
        if 'optimizer' in checkpoint:
            loaded_model.optimizer = checkpoint['optimizer']
        if 'lr' in checkpoint:
            loaded_model.lr = checkpoint['lr']
        logging.info(f"Model loaded from {path}.")

        return loaded_model, optimizer

    @staticmethod
    def get_profile_names(path: os.PathLike) -> tuple[str, str]:
        checkpoint = torch.load(path)
        return checkpoint['user_profiles_params'], checkpoint['business_profiles_params']

    @staticmethod
    def get_input_size_from_file(path: os.PathLike) -> int:
        checkpoint = torch.load(path)
        return checkpoint['input_size']

    @staticmethod
    def get_version_from_file(path: os.PathLike) -> int:
        checkpoint = torch.load(path)
        return checkpoint['version']

    def plot_loss_progress(self, title: str = "Model", display_note: bool = True, save_location: os.PathLike = None) -> tuple[plt.Figure, plt.Axes]:
        subplot = plt.subplots()
        fig: plt.Figure = subplot[0]
        ax: plt.Axes = subplot[1]
        ax.plot(range(self.current_epoch), self.loss_history)
        if display_note:
            title = f"{title} Loss: {self.note}{' - ' if self.note else ''}{self.version} verborgen lagen"
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (normalized scores)")

        if save_location:
            fig.savefig(save_location)

        plt.close(fig)
        return fig, ax
