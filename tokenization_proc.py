## Trains and loads the SentencePiece tokenizer ##

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import TextVectorization

paths = [f'/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/wikitext-103/wiki.{variable}.tokens' for variable in ['valid']]

def generate_wiki(paths):
    data = []
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                data.append(line)

    return data

MAX_SEQUENCE_LENGTH = 10
VOCAB_SIZE = 10000
inp_tokenizer = TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SEQUENCE_LENGTH)

print('building tokenizer')
inp_tokenizer.adapt(generate_wiki(paths))

print(inp_tokenizer('Testing this tokenizer! Ahan should be out of vocabulary indeed!'))

