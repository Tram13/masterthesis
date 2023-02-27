from predictor.predictor import Predictor


class MultiLayerPerceptronPredictor(Predictor):

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
