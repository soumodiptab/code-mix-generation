import torchtext
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
import re
from cleantext import clean
import os
#tweet_tokenizer = TweetTokenizer()
tokenizer = get_tokenizer('basic_english')


def replace_concurrent_punctuation(text):
    # replace concurrent punctuation with single punctuation
    return re.sub(r'(!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~){2,}', r' ', text)


def read_data(filename, n_lines):
    with open(filename, 'r') as f:
        lines = []
        for _ in range(n_lines):
            line = f.readline()
            line = clean(line, no_emoji=True)
            line = replace_concurrent_punctuation(line)
            lines.append(tokenizer(line))
    return lines


def save_data(filename, lines):
    # Save the data to a file
    with open(filename, 'w')as f:
        for line in lines:
            line = ' '.join(line)
            f.write(line.strip()+'\n')

if not os.path.exists('./processed_data'):
    os.mkdir('processed_data')

data = read_data('data/alternate/L3Cube-HingCorpus_roman/R11_final_data/concatenated_train_final_shuffled.txt',500000)
train,valid = train_test_split(data, test_size=0.3, random_state=42)
valid,test=train_test_split(valid, test_size=0.5, random_state=42)
save_data('processed_data/train.txt', train)
save_data('processed_data/valid.txt', valid)
save_data('processed_data/test.txt', test)