import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class CodeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding()
        self.lstm = nn.LSTM()
        self.fc = nn.Linear()

    def forward(self, x):
        pass

    def run_training(self, train_loader, dev_loader, epochs):
        pass

    def evaluate(self, test_loader):
        pass
