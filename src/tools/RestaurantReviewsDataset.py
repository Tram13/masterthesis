import torch
from torch.utils.data import Dataset


# Class only used for torch compatibility
# This does NOT contain the entire dataset, only fractions of it (such as train or test subsets)
class RestaurantReviewsDataset(Dataset):
    def __init__(self, input_features, ratings):
        self.x_train = torch.tensor(input_features, dtype=torch.float32)
        self.y_train = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
