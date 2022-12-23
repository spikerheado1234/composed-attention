import sys

## First append the parent directory.
sys.path.append('/home/ag82/composed-attention/')

from vanilla_transformer import Transformer
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import argparse
import time
import pdb ## For debugging only, TODO remove.

from pre_train_wiki_loader import en_tokenizer
from stats import Stats 
from constants import Constants
import tensorflow_models as tfm
from tokenization_proc import pad, add_start_end

## Basic command line arguments. ##
parser = argparse.ArgumentParser(description='Train compositions of efficient Transformer Variants.')
parser.add_argument('--type', dest='attention_type', default='MHA', choices=['MHA', 'LinMHA', 'PerfMHA', 'CompMHA'], help='The type of attention mechanism you wish to train a Transformer on. (Possible Values are: MHA, LinMHA or PerfMHA)')
parser.add_argument('--downsampling_k', dest='downsampling_k', default=32, type=int, help='The dimension you wish to downsample the sequence length to in accordance to the LinFormer Paper.')
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='the batch size used for training & inference purposes')
parser.add_argument('--layers', dest='layers', default=4, type=int, help='the number of layers in the transformer.')
parser.add_argument('--sequence_length', dest='sequence_length', type=int, default=128, help='the sequence length of the input to the transformer')
parser.add_argument('--step_count', dest='num_steps', type=int, default=500000, help='the number of steps as input to pre-training.')
parser.add_argument('--rank', dest='rank', type=int, default=1, help='The rank of the process, to distinguish output.')
parser.add_argument('--encoder_only', dest='enc_only', action='store_true', help='Whether we are training in encoder only mode')
parser.add_argument('--warmup', dest='warmup', default=10000, type=int, help='The number of warmup steps required during pre-training.')

args = parser.parse_args()

## Define global vars here. ##

BUFFER_SIZE = 20000

MAX_TOKENS = args.sequence_length

BATCH_SIZE = args.batch_size

### Data preparation 

curr_dir = os.getcwd() + "/"

## Load the IMDB dataset. ##
train_data, val_data, test_data = tfds.load(name="imdb_reviews", split=('train[:60%]', 'test[60%:]', 'test'), as_supervised=True)

## A basic preparatory function. ##
def prepare_batch(inps, labels):
    global en_tokenizer
    ## Take special care to tokenize ONLY the inps. ##
    inps_tok = en_tokenizer.tokenize(inps)

    return inps_tok, labels

## A function to tokenize the data. ##
def make_batches(ds, BUFFER_SIZE, BATCH_SIZE):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

## Basic tokenized dataset that is batched accordingly. ##
train_batches = make_batches(train_data, BUFFER_SIZE, BATCH_SIZE)

## We define our transformer here. ##
num_layers = args.layers
d_model = 512
dff = 2048
num_attention_heads = 8
dropout_rate = 0.1
rank = args.rank

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_attention_heads=num_attention_heads,
    dff=dff,
    input_vocab_size=Constants.wiki_vocab_size,
    target_vocab_size=Constants.wiki_vocab_size,
    dropout_rate=dropout_rate,
    downsampling_value=args.downsampling_k if args.attention_type == 'LinMHA' or args.attention_type == 'CompMHA' else 32, # Just default to 32 otherwise, doesn't matter since it won't be used.
    attention_type=args.attention_type,
    sequence_length=args.sequence_length,
    encoder_only=args.enc_only)

## Then, we create the learning rate schedule. ##
initial_learning_rate = 0.0001
num_train_steps = args.num_steps
warmup_steps = args.warmup
linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=0,
    decay_steps=num_train_steps)

warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = linear_decay,
    warmup_steps = warmup_steps
)
optimizer = tf.keras.optimizers.Adam(warmup_schedule, beta_1=0.9, beta_2=0.999,
                                    epsilon=1e-9)

## We create our loss function. ##
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')

## Next, we create helper methods to compute accuracy and loss. ##

## TODO, come back to this later. ## TODO, sanity check this.
def accuracy_function(real, pred):
  ## We delete whatever corresponds to the [END] token over here. 
  pdb.set_trace()
  accuracies = tf.math.equal(tf.cast(real, dtype=tf.int64), tf.cast(tf.round(pred), dtype=tf.int64))
  accuracies = tf.cast(accuracies, dtype=tf.float32)
  return tf.reduce_sum(accuracies) / tf.reduce_sum(tf.ones(shape=real.shape, dtype=tf.float32)) ## Divide by batch * seq to get accuracy over everything.
  
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

checkpoint_path = './checkpoints/train/' + str(args.attention_type) + '/' + str(initial_learning_rate) + '/' + str(args.warmup)

ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')
else:
    ## Otherwise we raise an exception. We should ALWAYS load from a checkpoint over here. ##
    raise Exception

train_step_signature = [
    (
         tf.TensorSpec(shape=(None, None), dtype=tf.int64),
         tf.TensorSpec(shape=(None, None), dtype=tf.int64)),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

## Then, we create a new model to finetune. ##
class DownstreamModel(tf.keras.Model):
  def __init__(self, transformer):
    super(DownstreamModel, self).__init__()
    self.transformer = transformer

    ## We create more interim dense layers as well. ##
    self.layer_one = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu) ## A hyperparameter that we will tune later.
    self.layer_two = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

  def call(self, inp): ## Keras requires only one input, so we pass in a list.
    out , _ = self.transformer(inp)
    out = self.layer_one(out)
    out = self.layer_two(out)
    return out

## Instantiate our new model over here. ##
downstream_model = DownstreamModel(transformer)

## TODO, check for correctness. ##
def pad_data(inp_tok, review):
  global MAX_TOKENS 
  """
  Pads the vector inputs (of size (BATCH_SIZE, SEQUENCE LENGTH)) to ensure each
  sequence length is standardized to MAX_TOKENS.
  """
  inp = inp_tok.merge_dims(-2, -1).to_tensor()

  inp = inp[:, :MAX_TOKENS]
  inp = pad(inp, MAX_TOKENS)

  ## Then we drop the last two tokens and add the start and last token.
  inp = inp[:, :-2]
  inp = add_start_end(inp)
  ## Now the inp will be fed into the encoder. ##

  ## first resize the reviews. ##
  review = tf.reshape(review, [review.shape[0], 1])
  review = add_start_end(review)
  tar_inp = review[:, :-1] ## Remove the end token for the review. Will be fed into the decoder.
  tar_real = review[:, 1:] ## Remove the start token for the comparison of the reivew. 

  return (inp, tar_inp), tar_real[:, :-1] ## Also remove the last token from tar_real.

def train_step(inputs, labels):

  (inp, tar_inp), tar_real = pad_data(inputs, labels)

  with tf.GradientTape() as tape:
    predictions, _ = downstream_model([inp, tar_inp],
                                      training = True)
    ## we have to recast the predictions. 
    loss = loss_object(tar_real, predictions) 
    accuracy = accuracy_function(tar_real, predictions)

  gradients = tape.gradient(loss, downstream_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, downstream_model.trainable_variables))

  train_loss.update_state(loss)
  train_accuracy(accuracy)


EPOCHS = args.num_steps
total_steps_required = args.num_steps

steps_elapsed = 0

train_start = time.time()
for epoch in range(EPOCHS):
  start = time.time()
  if steps_elapsed > total_steps_required:
    break

  train_loss.reset_states()
  train_accuracy.reset_states()

  for (batch, (inp, tar)) in enumerate(train_batches):
    if steps_elapsed > total_steps_required:
      break
    train_step(inp, tar)
    if (steps_elapsed % 1000 == 0):
      # We print end-to-end time here just in case.
      print(f'----------- End-to-End: {time.time() - train_start} -----------')

    print(f'Steps {steps_elapsed} Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Perplexity: {train_perplexity.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

    with open(f'./train_data_{args.attention_type}_{args.rank}.txt', 'a+') as f:
      f.write(f'{steps_elapsed} {train_loss.result():.4f} {train_accuracy.result():.4f}\n')

    with open(f'./train_stats_{args.attention_type}_{args.rank}.txt', 'a+') as f:
        f.write(f'{steps_elapsed} MHA {Stats.mha_time:.4f} MHA-Enc {Stats.mha_enc_time:.4f} MHA-Causal {Stats.mha_causal_time:.4f} MHA-Enc-Dec {Stats.mha_enc_dec_time:.4f} FFN {Stats.ffn_time:.4f} Downsampling {Stats.downsampling_time:.4f} Kernel-Transformation {Stats.transformation_time:.4f}\n')

    steps_elapsed += 1

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n', flush=True)


train_end = time.time()

print(f'Total training time: {train_end-train_start}\n', flush=True)
