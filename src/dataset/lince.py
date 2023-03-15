import torch
from torch.utils.data import Dataset, DataLoader


class LinceDataset(Dataset):
    def __init__(self, filename, vocab_english=None, vocab_hinglish=None, ngram=5):
        data_english, data_hinglish = self.read_data(filename)
        if vocab_hinglish is None:
            self.vocab_h, self.ind2vocab_h = self.build_vocab(data_hinglish)
        else:
            self.vocab_h = vocab_hinglish
            self.ind2vocab_h = {v: k for k, v in vocab_hinglish.items()}
        self.n = ngram
        self.x, self.y = self.__create_dataset(data_hinglish)

    def get_vocab(self):
        return self.vocab_h

    def read_data(self, filename):
        english = []
        hinglish = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                e = line.strip().split('\t')[0]
                english.append(e.strip().split(' '))
                try:
                    h = line.strip().split('\t')[1]
                except:
                    h = ""
                hinglish.append(h.strip().split(' '))
        return english, hinglish

    def build_vocab(self, data):
        word_set = set()
        for line in data:
            for word in line:
                if word not in word_set:
                    word_set.add(word)
        # sort the vocab
        word_list = sorted(list(word_set))
        vocab_dict = {"<unk>": 0}
        for i, word in enumerate(word_list):
            vocab_dict[word] = i+1
        ind2word = {v: k for k, v in vocab_dict.items()}
        return vocab_dict, ind2word

    def get_ngram(self, tokens):
        n = self.n
        ngram = []
        if len(tokens) == 0:
            return None
        tokens = ["<begin>" for _ in range(n-2)] + tokens
        for i in range(len(tokens)-n+1):
            ngram.append(tokens[i:i+n])
        return ngram

    def __get_seq(self, tokens):
        vec = []
        for word in tokens:
            if word in self.vocab_h:
                vec.append(self.vocab_h[word])
            else:
                vec.append(self.vocab_h["<unk>"])
        return vec

    def __create_dataset(self, data):
        x = []
        y = []
        ngrams = []
        for line in data:
            ngrams.extend(self.get_ngram(line))

        for ngram in ngrams:
            x_tokens = ngram[:-1]
            y_tokens = ngram[1:]
            x.append(self.__get_seq(x_tokens))
            y.append(self.__get_seq(y_tokens))
        return torch.LongTensor(x), torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=True)


train_data = LinceDataset('processed_data/lince/train.txt')
valid_data = LinceDataset('processed_data/lince/valid.txt',
                          vocab_hinglish=train_data.get_vocab())