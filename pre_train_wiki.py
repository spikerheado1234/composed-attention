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

## Define argument parsing and help over here. ##

parser = argparse.ArgumentParser(description='Train compositions of efficient Transformer Variants.')
parser.add_argument('--type', dest='attention_type', default='MHA', choices=['MHA', 'LinMHA', 'PerfMHA', 'CompMHA'], help='The type of attention mechanism you wish to train a Transformer on. (Possible Values are: MHA, LinMHA or PerfMHA)')
parser.add_argument('--downsampling_k', dest='downsampling_k', default=32, type=int, help='The dimension you wish to downsample the sequence length to in accordance to the LinFormer Paper.')
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='the batch size used for training & inference purposes')
parser.add_argument('--layers', dest='layers', default=4, type=int, help='the number of layers in the transformer.')
parser.add_argument('--sequence_length', dest='sequence_length', type=int, default=128, help='the sequence length of the input to the transformer')
parser.add_argument('--step_count', dest='num_steps', type=int, default=500000, help='the number of steps as input to pre-training.')
parser.add_argument('--rank', dest='rank', type=int, default=1, help='The rank of the process, to distinguish output.')
parser.add_argument('--encoder_only', dest='enc_only', type=bool, default=False, help='Whether we are training in encoder only mode')
parser.add_argument('--validate_mode', dest='is_val', type=bool, default=False, help='Whether we are concurrently validating as well')

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

## Hyperparameters ##
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
    downsampling_value=args.downsampling_k if args.attention_type == 'LinMHA' else 32, # Just default to 32 otherwise, doesn't matter since it won't be used.
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
initial_learning_rate = 2e-5
num_train_steps = args.num_steps
warmup_steps = 10000
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

checkpoint_path = './checkpoints/train/' + str(args.attention_type)

ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

# If a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')

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

distribution = [0, 0, 0, 0] # 0, 0 < 10, 10 -50, > 50 -> gives number of NON-0 elements excluding start and end token.

def compute_distribution(inp):
  global distribution

  count = tf.reduce_sum(tf.cast(tf.math.logical_not(tf.equal(tf.convert_to_tensor(inp.numpy()[:, :-1]), 0)), dtype=tf.int64))

  if count == 0:
    distribution[0] += 1
  elif 0 < count < 10:
    distribution[1] += 1
  elif 10 <= count < 50:
    distribution[2] += 1
  else:
    distribution[3] += 1

def train_step(inputs, labels):

  (inp, tar_inp) = inputs
  tar_real = labels

  (inp, tar_inp), tar_real, weight = mask_data(inp)

  compute_distribution(tar_real)

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_object(tar_real, predictions, sample_weight=weight[:, 1:]) 
    accuracy = accuracy_function(tar_real, predictions, weight[:, 1:])

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss.update_state(loss, sample_weight=weight[:, 1:])
  train_accuracy(accuracy)
  train_perplexity(perplexity_function(train_loss.result()))

EPOCHS = args.num_steps
total_steps_required = args.num_steps

profiling_steps = 10

steps_elapsed = 0

# Right before we run about 20 iterations to ensure all the caches and buffers are in use to simulate "training."

pre_profile_steps = 20

print('running pre_profiling steps')
for (idx, (inp, tar)) in enumerate(train_batches):
  train_step(inp, tar)

  if idx >= pre_profile_steps:
    break

train_start = time.time()
for epoch in range(EPOCHS):
  start = time.time()
  if steps_elapsed > total_steps_required:
    break

  train_loss.reset_states()
  train_accuracy.reset_states()
  train_perplexity.reset_states()
  print('started profiling')
  tf.profiler.experimental.start(f'logs/{args.attention_type}')

  with tf.profiler.experimental.Trace('train', step_num=steps_elapsed, _r=1):
    for (batch, (inp, tar)) in enumerate(train_batches):
      if steps_elapsed > total_steps_required:
        break
      train_step(inp, tar)
      if (steps_elapsed % 1000 == 0):
        # We print end-to-end time here just in case.
        print(f'----------- End-to-End: {time.time() - train_start} -----------')
      if (steps_elapsed % 5000 == 0):
        save_path = ckpt_manager.save()
        print(f'Saved checkpoint for step: {steps_elapsed} path: {save_path}')
        ckpt.step.assign_add(1)

      print(f'Steps {steps_elapsed} Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Perplexity: {train_perplexity.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

      with open(f'./train_data_{args.attention_type}.txt', 'a+') as f:
        f.write(f'{steps_elapsed} {train_loss.result():.4f} {train_accuracy.result():.4f}\n')

      with open(f'./train_stats_{args.attention_type}.txt', 'a+') as f:
          f.write(f'{steps_elapsed} MHA {Stats.mha_time:.4f} MHA-Enc {Stats.mha_enc_time:.4f} MHA-Causal {Stats.mha_causal_time:.4f} MHA-Enc-Dec {Stats.mha_enc_dec_time:.4f} FFN {Stats.ffn_time:.4f} Downsampling {Stats.downsampling_time:.4f} Kernel-Transformation {Stats.transformation_time:.4f}\n')

      steps_elapsed += 1

      if steps_elapsed == profiling_steps:
        print('closing profiler')
        tf.profiler.experimental.stop()

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}', flush=True)


  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n', flush=True)


train_end = time.time()

print(f'Total training time: {train_end-train_start}\n', flush=True)
