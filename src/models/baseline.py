import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np

class GramNet(nn.Module):
    def __init__(self,vocab_size, n_hidden=256, n_layers=4,embedding_dim=200, dropout=0.2, lr=0.001,device='cuda'):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.device = device
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, n_hidden, n_layers, dropout=dropout,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hidden, vocab_size)      
    
    def forward(self, x, hidden):
        embedded = self.embedding(x)     
        lstm_output, hidden = self.rnn(embedded, hidden)
        out = self.dropout(lstm_output)
        out = out.reshape(-1, self.n_hidden) 
        out = self.fc(out)
        return out, hidden
    
    def __init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return hidden
    
    def accuracy(self,true, pred):
        true = np.array(true)
        pred = np.array(pred)
        num_correct = sum(true == pred)
        num_total = len(true)
        return num_correct / num_total


    def run_training(self,train_dataset,valid_dataset, epochs=10, batch_size=32, clip = 1,print_every=1):
        device = self.device
        if str(device) == 'cpu':
            print("Training only supported in GPU environment")
            return
        torch.cuda.empty_cache()
        self.to(device)
        train_loader = train_dataset.get_dataloader(batch_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(epochs):
            hidden = self.__init_hidden(batch_size)
            for i, (x, y) in enumerate(train_loader):
                hidden = tuple([each.data for each in hidden])
                x, y = x.to(device), y.to(device)
                output, hidden = self.forward(x, hidden)
                loss = criterion(output, y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.parameters(), clip)
                optimizer.step()
            if i % print_every == 0:
                acc,_ = self.evaluate(train_dataset)
                acc2,_ = self.evaluate(valid_dataset)
                self.train()
                print("Epoch: {}/{}".format(epoch+1, epochs),
#                       "Step: {}".format(i),
                      "Loss: {}".format(loss.item()),
                      "Training Accuracy: {}".format(acc),
                      "Validation Accuracy: {}".format(acc2))
                    
    def evaluate(self, dataset, batch_size=32):
        device = self.device
        self.to(device)
        self.eval()
        loader = dataset.get_dataloader(batch_size)
        hidden = self.__init_hidden(batch_size)
        preds = []
        trues = []
        for i, (x, y) in enumerate(loader):
            hidden = tuple([each.data for each in hidden])
            x, y = x.to(device), y
            output, hidden = self.forward(x, hidden)
            preds.extend(output.argmax(dim=1).cpu().numpy())
            trues.extend(y.view(-1).numpy())
        accuracy = self.accuracy(trues, preds)
        return accuracy, preds