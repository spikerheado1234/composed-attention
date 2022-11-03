## Trains and loads a tokenizer Bert-Style WordPiece tokenizer ##

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
import numpy as np

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

paths = [f'/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/wikitext-103/wiki.{variable}.tokens' for variable in ['valid']]

def generate_wiki(paths):
    data = []
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                data.append(line)

    return data


## A global list of reserved tokens we can use. ##
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]", "[MASK]"]

bert_tokenizer_params=dict(lower_case=True)
def generate_vocabulary(ds, map_fn):
    global bert_tokenizer_params, reserved_tokens

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size = 8000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    ## We have to change the pt, en pair into just en.
    train_en = ds.map(map_fn)

    print('generating bert vocab from dataset')
    en_vocab = bert_vocab.bert_vocab_from_dataset(
        train_en.batch(1000).prefetch(10),
        **bert_vocab_args
    )
    return en_vocab


SEQUENCE_LENGTH = 11

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)

## TODO, comment this out after tokenization has been run. ##
## To not keep repeating this, comment out eventually. ##
#en_vocab = generate_vocabulary(train_examples, lambda en_one, en_two : en_one)

#write_vocab_file('en_vocab.txt', en_vocab)

def generate_tokenizer(vocab_path, tokenizer_params):
    return text.BertTokenizer(vocab_path, **tokenizer_params)

#en_tokenizer = generate_tokenizer('en_vocab.txt', bert_tokenizer_params)

## Use only if needed. ##
def add_start_end(inp):
    global reserved_tokens

    START = tf.get_static_value(tf.math.argmax(tf.constant(reserved_tokens) == "[START]"))
    END = tf.get_static_value(tf.math.argmax(tf.constant(reserved_tokens) == "[END]"))
    batch_size = inp.shape[0]
    start = tf.ones(shape=(batch_size, 1)) * START
    end = tf.ones(shape=(batch_size, 1)) * END
    start = tf.cast(start, dtype=inp.dtype)
    end = tf.cast(end, dtype=inp.dtype)
    return tf.concat([start, inp, end], axis=1)

def masker(inp):
    mask = tf.random.uniform(shape=inp.shape, minval=0, maxval=1) < 0.15
    mask = mask.numpy()

    ## Then use this to mask the input. ##
    inp = inp.numpy()
    MASK = tf.argmax(tf.constant(reserved_tokens) == "[MASK]")
    inp[mask] = MASK

    return tf.convert_to_tensor(inp)

def pad(inp):
    global SEQUENCE_LENGTH
    ## Now, in the event that inp is less than sequence length, we must pad with zeros accordingly. ##
    max_seq_len = inp.shape[1]
    current_batch_size = inp.shape[0]
    if SEQUENCE_LENGTH - max_seq_len > 0:
        zero_vector = tf.zeros(shape=(current_batch_size, SEQUENCE_LENGTH - max_seq_len), dtype=tf.int64)
        return tf.concat([inp, zero_vector], axis=1)
    else:
        return inp

## Masks 15% of the input. Ragged Tensor should be tokenized via Bert-Style tokenization. ##
def mask(inp):
    global SEQUENCE_LENGTH

    inp = inp.merge_dims(-2, -1).to_tensor()
    inp = inp[:, :SEQUENCE_LENGTH]
    inp = pad(inp)

    ## Then we drop the last two tokens and add the start and last token.
    inp = inp[:, :inp.shape[1] - 2]
    inp = add_start_end(inp)

    tar_input = inp # Fed into the decoder.
    tar_real = inp # Take categorical cross entropy loss against this.

    inp = masker(inp) # Fed into the encoder.

    return (inp, tar_input, tar_real)



"""
So the pipeline will look like this.

for index, example in enumerate(ds):
    train_tok = en_tokenizer.tokenize(example)
    inp, tar_input, tar_real = mask(train_tok)

## A unit test to sanity check everything. ##

## We train a simple tokenizer using the pt-en dataset. ##
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

output = en_tokenizer.tokenize([bytes("Hello, my name is ahan and this is a test!", 'utf-8'),
                                bytes("Hello, my name is bob and this is a test!", 'utf-8')])

mask(output)
"""
