import torch.nn as nn
import torch.optim

from predictor.implementations.multilayer_perceptron import MultiLayerPerceptronPredictor


class MultiLayerPerceptron1Predictor(MultiLayerPerceptronPredictor):
    def __init__(self, input_size: int):
        super(MultiLayerPerceptron1Predictor, self).__init__(input_size, 1)
        # Definition of netwok architecture
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(input_size // 2, 1)
        )

        # Random initialisation
        torch.nn.init.xavier_uniform_(self.linear_stack[0].weight)
        torch.nn.init.xavier_uniform_(self.linear_stack[3].weight)

        # Run on GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
