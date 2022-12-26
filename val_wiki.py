"""
This is a validation script, it will run through the checkpoints in the following order:
2, 4, 6, 8, ..., last_checkpoint

And run eachcheckpoint on the validation dataset.
"""
import tensorflow as tf
from vanilla_transformer import Transformer
import argparse
import os
from pre_train_wiki_loader import get_val_ds, make_batches
from constants import Constants
from tokenization_proc import mask
import tensorflow_models as tfm

## Define argument parsing and help over here. ##

parser = argparse.ArgumentParser(description='Train compositions of efficient Transformer Variants.')
parser.add_argument('--type', dest='attention_type', default='MHA', choices=['MHA', 'LinMHA', 'PerfMHA', 'CompMHA'], help='The type of attention mechanism you wish to train a Transformer on. (Possible Values are: MHA, LinMHA or PerfMHA)')
parser.add_argument('--downsampling_k', dest='downsampling_k', default=32, type=int, help='The dimension you wish to downsample the sequence length to in accordance to the LinFormer Paper.')
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='the batch size used for training & inference purposes')
parser.add_argument('--layers', dest='layers', default=4, type=int, help='the number of layers in the transformer.')
parser.add_argument('--sequence_length', dest='sequence_length', type=int, default=128, help='the sequence length of the input to the transformer')
parser.add_argument('--rank', dest='rank', type=int, default=1, help='The rank of the process, to distinguish output.')
parser.add_argument('--encoder_only', dest='enc_only', type=bool, default=False, help='Whether we are training in encoder only mode')
parser.add_argument('--num_checkpoints', dest='checkpoint_num', type=int, default=30, help='The total number of checkpoints to validate')
parser.add_argument('--step_count', dest='num_steps', type=int, default=500000, help='the number of steps as input to pre-training.')
parser.add_argument('--learning_rate', dest='lr_rate', type=float, default=0.1, help='the largest constant in the lr schedule.')
parser.add_argument('--warmup', dest='warmup', default=10000, type=int, help='The number of warmup steps required during pre-training.')

args = parser.parse_args()

## Define global vars here. ##

MAX_TOKENS = args.sequence_length

### Data preparation 

curr_dir = os.getcwd() + "/"

val_ds = get_val_ds()

BUFFER_SIZE = 20000
BATCH_SIZE = args.batch_size

## We make the batches here. ##
val_batches = make_batches(val_ds, BUFFER_SIZE, BATCH_SIZE)

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
# May be buggy, but be we do NOT want to use Mean as used above. ##
train_perplexity = tf.keras.metrics.Mean(name='train_perplexity')

checkpoint_path = './checkpoints/train/' + str(args.attention_type) + f'/{str(args.lr_rate)}/{args.warmup}'

ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

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

def train_step(inputs, labels):

  (inp, tar_inp) = inputs
  tar_real = labels

  (inp, tar_inp), tar_real, weight = mask_data(inp)

  predictions, _ = transformer([inp, tar_inp],
                                 training = True)
  loss = loss_object(tar_real, predictions, sample_weight=weight[:, 1:]) 
  accuracy = accuracy_function(tar_real, predictions, weight[:, 1:])

  train_loss.update_state(loss, sample_weight=weight[:, 1:])
  train_accuracy(accuracy)
  train_perplexity(perplexity_function(train_loss.result()))

num_checkpoints = args.checkpoint_num
curr_checkpoint = 2

## We first restore the initial model checkpoint we wish to validate. ##
ckpt.restore(f'{checkpoint_path}/ckpt-{curr_checkpoint}')
while curr_checkpoint <= num_checkpoints:

  train_loss.reset_states()
  train_accuracy.reset_states()
  train_perplexity.reset_states()

  for (batch, (inp, tar)) in enumerate(val_batches):
    train_step(inp, tar)

  print(f'Checkpoint {curr_checkpoint} Batch {batch} Loss {train_loss.result():.4f} Perplexity: {train_perplexity.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  with open(f'./{args.attention_type}_val_data_{args.lr_rate}_{args.warmup}.txt', 'a+') as f:
    f.write(f'{train_loss.result():.4f} {train_accuracy.result():.4f}\n')
  
  curr_checkpoint += 2
  ## Restore the next checkpoint and continue. ##
  ckpt.restore(f'{checkpoint_path}/ckpt-{curr_checkpoint}')
