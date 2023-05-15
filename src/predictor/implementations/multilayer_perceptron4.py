import torch.optim
from torch import nn

from predictor.implementations.multilayer_perceptron import MultiLayerPerceptronPredictor


class MultiLayerPerceptron4Predictor(MultiLayerPerceptronPredictor):
    def __init__(self, input_size: int):
        super(MultiLayerPerceptron4Predictor, self).__init__(input_size, 4)
        # Definition of netwok architecture
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(input_size // 4, input_size // 8),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(input_size // 8, input_size // 16),
            nn.Sigmoid(),
            nn.Linear(input_size // 16, 1)
        )

        # Random initialisation
        torch.nn.init.xavier_uniform_(self.linear_stack[0].weight)
        torch.nn.init.xavier_uniform_(self.linear_stack[3].weight)
        torch.nn.init.xavier_uniform_(self.linear_stack[6].weight)
        torch.nn.init.xavier_uniform_(self.linear_stack[9].weight)
        torch.nn.init.xavier_uniform_(self.linear_stack[11].weight)

        # Run on GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
