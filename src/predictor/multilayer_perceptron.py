import torch.nn as nn
import torch.optim


class MultiLayerPerceptronPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(MultiLayerPerceptronPredictor, self).__init__()
        self.l1 = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(5, output_size)
        # Run on GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output


if __name__ == '__main__':
    model = MultiLayerPerceptronPredictor(80, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    epochs = 1500

    costval = []
    for j in range(epochs):
        for i, (x_train, y_train) in enumerate(dataloader):    #prediction
            y_pred = model(x_train)
            # calculating loss
            cost = criterion(y_pred, y_train.reshape(-1, 1))
            # backprop
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        if j % 50 == 0:
            print(cost)
            costval.append(cost)

