import torch
from torch.utils.data import Dataset, DataLoader

class L3CubeDataset(Dataset):
    def __init__(self, filename):
        data = self.read_data(filename)
        self.vocab, self.ind2vocab = self.build_vocab(data)
        x = self.__create_dataset(data)

    def __init__(self,filename,vocab):
        self.vocab = vocab
        self.ind2vocab = {v:k for k,v in vocab.items()}
        data = self.read_data(filename)
        x = self.__create_dataset(data)
        
    def get_vocab(self):
        return self.vocab

    def read_data(filename):
        lines = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                lines.append(line.strip().split(' '))
        return lines

    def build_vocab(self,data):
        word_set = set()
        for line in data:
            for word in line:
                if word not in word_set:
                    word_set.add(word)
        # sort the vocab
        word_list = sorted(list(word_set))
        vocab_dict = {}
        for i,word in enumerate(word_list):
            vocab_dict[word]=i+2
        ind2word = {v:k for k,v in vocab_dict.items()}
        return vocab_dict, ind2word
    
    
    def __create_dataset(self, data):
        x = []
        y = []
        for line in data:
            line = line.strip()
            x.append(line.split('\t')[0])
            y.append(line.split('\t')[1])
        return x,y  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
