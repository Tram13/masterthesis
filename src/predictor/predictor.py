from abc import ABC, abstractmethod
import torch.nn as nn
import pandas as pd


class Predictor(ABC, nn.Module):

    def get_train_test(self, businesses: pd.DataFrame, users: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
