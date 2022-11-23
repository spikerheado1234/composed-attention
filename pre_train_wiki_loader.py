import datasets
import tensorflow as tf
from tokenization_proc import mask, generate_vocabulary, generate_tokenizer, write_vocab_file, bert_tokenizer_params
from tensorflow.data import TextLineDataset
import os

print('loading wikitext dataset')

train_dir = os.getcwd() + "/wikitext-103/wiki.train.tokens"
valid_dir = os.getcwd() + "/wikitext-103/wiki.valid.tokens"
test_dir = os.getcwd() + "/wikitext-103/wiki.test.tokens"

## TODO, standardize later. ##
seed = 42

def get_train_ds():
    ds = TextLineDataset(
            [train_dir]
    )
    return ds

def get_all_ds():
    ds = TextLineDataset(
            [train_dir,
            valid_dir,
            test_dir]
    )
    return ds

def get_val_ds():
    ds = TextLineDataset(
        [valid_dir]
    )
    return ds

## Next, we use the dataset to generate the vocabulary. ##
print('generating vocabulary')
en_vocab = generate_vocabulary(get_all_ds(), lambda en_one: en_one)

write_vocab_file('en_vocab.txt', en_vocab)

print('generating tokenizer')
en_tokenizer = generate_tokenizer('en_vocab.txt', bert_tokenizer_params)

def prepare_batch(inp):
    global en_tokenizer
    ## Note, inp and tar are the same. ##
    inp_tok = en_tokenizer.tokenize(inp)

    return (inp_tok, inp_tok), inp_tok 

def make_batches(ds, BUFFER_SIZE, BATCH_SIZE):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))



