import datasets
import tensorflow as tf
from tokenization_proc import mask, generate_vocabulary, generate_tokenizer, write_vocab_file, bert_tokenizer_params

ds = datasets.load_data('c4', 'en') ## Loads the c4 dataset.

ds = ds.to_tf_dataset(
    colums=["text"],
    label_cols=["text"],
    batch_size=64,
    shuffle=True
)

## Next, we use the dataset to generate the vocabulary. ##
en_vocab = generate_vocabulary(ds, lambda en_one, en_two: en_one)

write_vocab_file('en_vocab.txt', en_vocab)

en_tokenizer = generate_tokenizer('en_vocab.txt', bert_tokenizer_params)

def prepare_batch(inp, tar):
    global en_tokenizer
    ## Note, inp and tar are the same. ##
    inp_tok = en_tokenizer.tokenize(inp)

    inp, tar_inp, tar_real = mask(inp_tok)
    tar_inp = tar_inp[:, :-1] # Drop the end token for the Decoder Input.
    tar_real = tar_real[:, 1:] # Drop the start token for what we compare to.

    return (inp, tar_inp), tar_real

def make_batches(ds, BUFFER_SIZE):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))



