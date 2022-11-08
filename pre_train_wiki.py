import tensorflow as tf
from vanilla_transformer import Transformer
import tensorflow_datasets as tfds
import tensorflow_text
import time
import argparse
from stats import Stats 
import os
from pre_train_wiki_loader import get_dataset, make_batches
from constants import Constants
from tokenization_proc import mask

## Define argument parsing and help over here. ##

parser = argparse.ArgumentParser(description='Train compositions of efficient Transformer Variants.')
parser.add_argument('--type', dest='attention_type', default='MHA', choices=['MHA', 'LinMHA', 'PerfMHA', 'CompMHA'], help='The type of attention mechanism you wish to train a Transformer on. (Possible Values are: MHA, LinMHA or PerfMHA)')
parser.add_argument('--downsampling_k', dest='downsampling_k', default=32, type=int, help='The dimension you wish to downsample the sequence length to in accordance to the LinFormer Paper.')
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='the batch size used for training & inference purposes')
parser.add_argument('--layers', dest='layers', default=4, type=int, help='the number of layers in the transformer.')
parser.add_argument('--sequence_length', dest='sequence_length', type=int, default=128, help='the sequence length of the input to the transformer')
parser.add_argument('--step_count', dest='num_steps', type=int, default=500000, help='the number of steps as input to pre-training.')

args = parser.parse_args()

## Define global vars here. ##

MAX_TOKENS = args.sequence_length

### Data preparation 

curr_dir = os.getcwd() + "/"

train_ds = get_dataset()

BUFFER_SIZE = 20000
BATCH_SIZE = args.batch_size

## We make the batches here. ##
train_batches = make_batches(train_ds, BUFFER_SIZE, BATCH_SIZE)

## Hyperparameters ##
num_layers = args.layers
d_model = 1024
dff = 3072
num_attention_heads = 10
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_attention_heads=num_attention_heads,
    dff=dff,
    input_vocab_size=Constants.wiki_vocab_size,
    target_vocab_size=Constants.wiki_vocab_size,
    dropout_rate=dropout_rate,
    downsampling_value=args.downsampling_k if args.attention_type == 'LinMHA' else 32, # Just default to 32 otherwise, doesn't matter since it won't be used.
    attention_type=args.attention_type)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=10000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = tf.constant(warmup_steps)
    self.warmup_steps = tf.cast(self.warmup_steps, tf.float32)

  def __call__(self, step):
    step_one = tf.constant(step)
    step_one = tf.cast(step_one, tf.float32)
    max_val = tf.math.maximum(step_one, self.warmup_steps)
    return tf.math.rsqrt(max_val)


learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def accuracy_function(real, pred):

  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))

  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)
  

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def perplexity_function(_loss):

    return tf.math.exp(_loss * -1)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
train_perplexity = tf.keras.metrics.Mean(name='train_perplexity')

checkpoint_path = './checkpoints/train'

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

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
  inp, tar_inp, tar_real = mask(inp_tok, MAX_TOKENS)
  tar_inp = tar_inp[:, :-1] # Drop the end token for the Decoder Input.
  tar_real = tar_real[:, 1:] # Drop the start token for what we compare to.

  return (inp, tar_inp), tar_real

def train_step(inputs, labels):
  (inp, tar_inp) = inputs
  tar_real = labels

  (inp, tar_inp), tar_real = mask_data(inp)

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)
    accuracy = accuracy_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy)
  train_perplexity(perplexity_function(loss))

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
  train_perplexity.reset_states()

  for (batch, (inp, tar)) in enumerate(train_batches):
    if steps_elapsed > total_steps_required:
      break
    train_step(inp, tar)
    steps_elapsed += 1

    print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

    with open('./train_data.txt', 'a+') as f:
      f.write(f'{train_loss.result():.4f} {train_accuracy.result():.4f}\n')

    with open('./train_stats.txt', 'a+') as f:
        f.write(f'MHA {Stats.mha_time:.4f} MHA-Enc {Stats.mha_enc_time:.4f} MHA-Causal {Stats.mha_causal_time:.4f} MHA-Enc-Dec {Stats.mha_enc_dec_time:.4f} FFN {Stats.ffn_time:.4f} Downsampling {Stats.downsampling_time:.4f} Kernel-Transformation {Stats.transformation_time:.4f}\n')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}', flush=True)


  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n', flush=True)


train_end = time.time()

print(f'Total training time: {train_end-train_start}\n', flush=True)