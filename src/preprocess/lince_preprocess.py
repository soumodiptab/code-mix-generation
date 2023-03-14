import torchtext
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
import re
from cleantext import clean
import os
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()
tokenizer = get_tokenizer('basic_english')


def replace_dates(text):
    date_format_a = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', ' <DATE> ', text)
    date_format_b = re.sub(
        r'[A-Za-z]{2,8}\s\d{1,2},?\s\d {4}', ' <DATE> ', date_format_a)
    date_format_c = re.sub(
        r'\d{2} [A-Z][a-z]{2,8} \d{4}', ' <DATE> ', date_format_b)
    return date_format_c


def replace_concurrent_punctuation(text):
    # replace concurrent punctuation with single punctuation
    return re.sub(r'(!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~){2,}', r' ', text)


def replace_hash_tags(text):
    return re.sub(r'(\s|^)#(\w+)', ' <HASHTAG> ', text)


def remove_special_characters(text):
    # remove special characters other than punctuation
    return re.sub(r'[^A-Za-z0-9\s\.\,\!\?\'\"\:\;]', ' ', text)


def remove_extra_spaces(text):
    return re.sub(r'\s{2,}', ' ', text)


def replace_hyphenated_words(text):
    # replace hyphenated words with words seperated by space
    return re.sub(r'(\w+)-(\w+)', r'\1 \2', text)

def custom_tokenizer(line):
    line = re.sub(r'<|>', ' ', line)
    line = replace_dates(line)
    line = replace_hyphenated_words(line)
    line = replace_hash_tags(line)
    # remove < and > from the text
    line = clean(line, no_emoji=True,
                    no_urls=True,
                    no_emails=True,
                    no_phone_numbers=True,
                    no_currency_symbols=True,
                    replace_with_url=" <URL> ",
                    replace_with_email=" <EMAIL> ",
                    replace_with_phone_number=" <PHONE> ",
                    replace_with_currency_symbol=" <CURRENCY> ",
                    lower=True)
    line = remove_special_characters(line)
    #line = replace_concurrent_punctuation(line)
    line = clean(line, no_numbers=True, no_digits=True, no_punct=True,
                    replace_with_number=" <NUMBER> ", replace_with_digit=" ", replace_with_punct="")
    line = "<BEGIN> " + line + " <END>"
    line = remove_extra_spaces(line)


def read_data(filename):
    with open(filename, 'r') as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            english = line.split('\t')[0]
            hinglish = line.split('\t')[1]
            
            tokens = tokenizer(line)
            if len(tokens) > 1:
                lines.append(tokens)
    return lines


def save_data(filename, lines):
    # Save the data to a file
    with open(filename, 'w')as f:
        for line in lines:
            line = ' '.join(line)
            f.write(line.strip()+'\n')


if not os.path.exists('./processed_data'):
    os.mkdir('processed_data')

train = read_data('data/train.txt')

# train, valid = train_test_split(data, test_size=0.3, random_state=42)
# valid, test = train_test_split(valid, test_size=0.5, random_state=42)
# print(train[1:100])
save_data('processed_data/train.txt', train)
save_data('processed_data/valid.txt', valid)
save_data('processed_data/test.txt', test)
