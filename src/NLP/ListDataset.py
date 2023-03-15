from torch.utils.data import Dataset

# todo place this somewhere
class ListDataset(Dataset):

    def __init__(self, data: list):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
