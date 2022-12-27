## Trains and loads a tokenizer Bert-Style WordPiece tokenizer ##

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
import numpy as np
from constants import Constants

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

paths = [f'/Users/Ahan/Desktop/Ahan/UIUC/PL-FOR-NAS/attention/wikitext-103/wiki.{variable}.tokens' for variable in ['valid', 'test', 'train']]

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
        vocab_size = Constants.wiki_vocab_size,
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

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)

## TODO, comment this out after tokenization has been run. ##
## To not keep repeating this, comment out eventually. ##

## I believe it is safe to delete this. ##
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
    ## Put back into the input the start and end tokens if masked out initially.
    inp[:, 0] = tf.argmax(tf.constant(reserved_tokens) == "[START]")
    inp[:, -1] = tf.argmax(tf.constant(reserved_tokens) == "[END]")

    return tf.convert_to_tensor(inp)

def get_masked_input_and_labels(encoded_texts):
    encoded_texts = encoded_texts.numpy()
    mask_token_id = tf.argmax(tf.constant(reserved_tokens) == "[MASK]")

    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 4] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights

def pad(inp, seq_length):
    ## Now, in the event that inp is less than sequence length, we must pad with zeros accordingly. ##
    max_seq_len = inp.shape[1]
    current_batch_size = inp.shape[0]
    if seq_length - max_seq_len > 0:
        zero_vector = tf.zeros(shape=(current_batch_size, seq_length - max_seq_len), dtype=tf.int64)
        return tf.concat([inp, zero_vector], axis=1)
    else:
        return inp

## Masks 15% of the input. Ragged Tensor should be tokenized via Bert-Style tokenization. ##
def mask(inp, seq_length):

    inp = inp.merge_dims(-2, -1).to_tensor()

    inp = inp[:, :seq_length]
    inp = pad(inp, seq_length)

    ## Then we drop the last two tokens and add the start and last token.
    inp = inp[:, :-2]
    inp = add_start_end(inp)

    tar_input = inp # Fed into the decoder.
    tar_real = inp # Take categorical cross entropy loss against this.

    #inp = masker(inp) # Fed into the encoder.
    inp, _, sample_weights = get_masked_input_and_labels(inp)

    return (tf.convert_to_tensor(inp), tar_input, tar_real, sample_weights)

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
