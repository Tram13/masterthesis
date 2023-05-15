import os

from torch import optim

from predictor.implementations.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.implementations.multilayer_perceptron1 import MultiLayerPerceptron1Predictor
from predictor.implementations.multilayer_perceptron2 import MultiLayerPerceptron2Predictor
from predictor.implementations.multilayer_perceptron3 import MultiLayerPerceptron3Predictor
from predictor.implementations.multilayer_perceptron4 import MultiLayerPerceptron4Predictor
from predictor.implementations.multilayer_perceptron5 import MultiLayerPerceptron5Predictor
from predictor.implementations.multilayer_perceptron6 import MultiLayerPerceptron6Predictor
from predictor.implementations.multilayer_perceptron7 import MultiLayerPerceptron7Predictor
from predictor.implementations.multilayer_perceptron8 import MultiLayerPerceptron8Predictor


class ModelLoader:
    @staticmethod
    def _get_model_by_version(version: int) -> MultiLayerPerceptronPredictor:
        return {
            1: MultiLayerPerceptron1Predictor,
            2: MultiLayerPerceptron2Predictor,
            3: MultiLayerPerceptron3Predictor,
            4: MultiLayerPerceptron4Predictor,
            5: MultiLayerPerceptron5Predictor,
            6: MultiLayerPerceptron6Predictor,
            7: MultiLayerPerceptron7Predictor,
            8: MultiLayerPerceptron8Predictor
        }[version]

    @staticmethod
    def load_mlp_model(model_path: os.PathLike) -> MultiLayerPerceptronPredictor:
        # Find model class
        version = MultiLayerPerceptronPredictor.get_version_from_file(model_path)
        model_class = ModelLoader._get_model_by_version(version)
        input_size = MultiLayerPerceptronPredictor.get_input_size_from_file(model_path)
        # Optimizer doesn't matter anymore, so dummy model
        trained_optimizer = optim.Adam(MultiLayerPerceptron1Predictor(input_size=input_size).parameters(), lr=0.002)
        trained_model, _ = MultiLayerPerceptronPredictor.load(trained_optimizer, model_path, model_class)
        return trained_model
