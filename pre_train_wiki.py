import tensorflow as tf
from vanilla_transformer import Transformer
import tensorflow_datasets as tfds
import tensorflow_text
import time
import argparse
from stats import Stats 
import os
from pre_train_wiki_loader import get_all_ds, get_train_ds, get_val_ds, make_batches
from constants import Constants
from tokenization_proc import mask
import tensorflow_models as tfm
import numpy as np
import pdb

## Define argument parsing and help over here. ##

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
parser.add_argument('--learning_rate', dest='lr_rate', type=float, default=0.0001, help='the largest constant in the lr schedule.')

args = parser.parse_args()

## Define global vars here. ##

MAX_TOKENS = args.sequence_length

### Data preparation 

curr_dir = os.getcwd() + "/"

train_ds = get_train_ds()
val_ds = get_val_ds()

BUFFER_SIZE = 20000
BATCH_SIZE = args.batch_size

## We make the batches here. ##
train_batches = make_batches(train_ds, BUFFER_SIZE, BATCH_SIZE)
val_batches = make_batches(val_ds, BUFFER_SIZE,BATCH_SIZE)
## Hyperparameters ##
num_layers = args.layers
d_model = 512
dff = 2048
num_attention_heads = 8
dropout_rate = 0.1
rank = args.rank
val_iter_freq = 10000

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

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
# Lets use a new learning rate here.
initial_learning_rate = args.lr_rate
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

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def accuracy_function(real, pred, weights):

  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(weights, 0))

  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
  

# This seems to be a tad buggy.
def loss_function(real, pred, sample_weight):
  loss_ = loss_object(real, pred, sample_weight=sample_weight)
  reduced_sum = tf.cast(tf.reduce_sum(sample_weight), dtype=tf.float32)
  reduced_loss = tf.cast(tf.reduce_sum(loss_), dtype=tf.float32)
  return reduced_loss / reduced_sum

def perplexity_function(_loss):

    return tf.math.exp(_loss)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
train_perplexity = tf.keras.metrics.Mean(name='train_perplexity')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

checkpoint_path = './checkpoints/train/' + str(args.attention_type) + '/' + str(initial_learning_rate) + '/' + str(args.warmup)

ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

# If a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')

class MaskedLM(tf.keras.Model):
    def __init__(self, transformer, target_vocab_size, encoder_only):
        super(MaskedLM, self).__init__()

        self.transformer = transformer
        self.encoder_only = encoder_only

        # The final linear layer.
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    @tf.function
    def call(self, inp):
      output, attention_weights = self.transformer(inp)

      final_output = self.final_layer(output)  # Shape `(batch_size, tar_seq_len, target_vocab_size)`.

      # Return the final output and the attention weights.
      return final_output, attention_weights

masked_lm = MaskedLM(transformer, Constants.wiki_vocab_size, False)

train_step_signature = [
    (
         tf.TensorSpec(shape=(None, None), dtype=tf.int64),
         tf.TensorSpec(shape=(None, None), dtype=tf.int64)),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

def mask_data(inp_tok):
  global MAX_TOKENS 
  """
  Pads the vector inputs (of size (BATCH_SIZE, SEQUENCE LENGTH)) to ensure each
  sequence length is standardized to MAX_TOKENS.
  """
  inp, tar_inp, tar_real, sample_weights = mask(inp_tok, MAX_TOKENS)
  tar_inp = tar_inp[:, :-1] # Drop the end token for the Decoder Input.
  tar_real = tar_real[:, 1:] # Drop the start token for what we compare to.

  return (inp, tar_inp), tar_real, sample_weights

def val_step(inputs, labels):

  (inp, tar_inp) = inputs
  tar_real = labels

  (inp, tar_inp), tar_real, weight = mask_data(inp)

  if args.enc_only:
    inp = inp[:, 1:]

  predictions, _ = masked_lm([inp, tar_inp],
                                training = False)
  loss = loss_object(tar_real, predictions, sample_weight=weight[:, 1:]) 
  accuracy = accuracy_function(tar_real, predictions, weight[:, 1:])

  val_loss.update_state(loss, sample_weight=weight[:, 1:])
  val_accuracy(accuracy)

def train_step(inputs, labels):

  (inp, tar_inp) = inputs
  tar_real = labels

  (inp, tar_inp), tar_real, weight = mask_data(inp)

  # We must drop the start token if we train in the encoder only regime.
  if args.enc_only:
    inp = inp[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = masked_lm([inp, tar_inp],
                                 training = True)
    loss = loss_object(tar_real, predictions, sample_weight=weight[:, 1:]) 
    accuracy = accuracy_function(tar_real, predictions, weight[:, 1:])

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss.update_state(loss, sample_weight=weight[:, 1:])
  train_accuracy(accuracy)
  train_perplexity(perplexity_function(train_loss.result()))

EPOCHS = 30

train_start = time.time()
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()
  train_perplexity.reset_states()

  for (batch, (inp, tar)) in enumerate(train_batches):
    train_step(inp, tar)

  ## We checkpoint at every epoch now.
  save_path = ckpt_manager.save()
  ckpt.step.assign_add(1)

  ## Here, we validate the data.
  val_loss.reset_states()
  val_accuracy.reset_states()
  for (batch, (inp, tar)) in enumerate(val_batches):
    val_step(inp, tar)

  print(f'Epoch {epoch + 1} Loss {val_loss.result():.4f} Accuracy {val_accuracy.result():.4f}', flush=True)

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n', flush=True)

train_end = time.time()

print(f'Total training + validation time: {train_end-train_start}\n', flush=True)
