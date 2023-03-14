import torch
from torch.utils.data import Dataset, DataLoader


class LinceDataset(Dataset):
    def __init__(self, data):
        x, y = self.__create_dataset(data)

    def __create_dataset(self, data):
        x = []
        y = []
        for line in data:
            line = line.strip()
            x.append(line.split('\t')[0])
            y.append(line.split('\t')[1])
        return x, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# create a function to read the data from a file


def read_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    return data


train_data = read_data('./data/train.txt')
test_data = read_data('./data/test.txt')
dev_data = read_data('./data/dev.txt')
