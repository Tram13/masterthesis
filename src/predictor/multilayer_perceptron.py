import torch.nn as nn
import torch.optim


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

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_stack(x)
