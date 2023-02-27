from abc import ABC, abstractmethod


class Predictor(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
