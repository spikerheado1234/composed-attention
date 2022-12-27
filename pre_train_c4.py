import tensorflow as tf
from vanilla_transformer import Transformer
import tensorflow_datasets as tfds
import tensorflow_text
import time
import argparse
from stats import Stats 
import os
from constants import Constants
from tokenization_proc import mask
import tensorflow_models as tfm
import numpy as np
from datasets import load_dataset
from pre_train_wiki_loader import en_tokenizer

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

args = parser.parse_args()

## Define global vars here. ##

MAX_TOKENS = args.sequence_length

### Data preparation ##

## We create a cusstom C4dataloader to interface with HuggingFace's APIs. ##
class C4Loader:
    def __init__(self, inp_dict, batch_size):
        self.inp_dict = inp_dict
        self.inp_iter = iter(inp_dict)
        self.batch_size = batch_size
        
        self.has_more_data = True
        
    def gen_next_train_data(self):
        inp_tensor = None
        count = 0
        try:
            while count < self.batch_size:
                if count == 0:
                    inp_tensor = tf.convert_to_tensor([next(self.inp_iter)['text']])
                else:
                    current_input = tf.convert_to_tensor([next(self.inp_iter)['text']])
                    inp_tensor = tf.concat([inp_tensor, current_input], axis=0)
                    
                count += 1
                    
            return inp_tensor
        except StopIteration:
            ## We terminate with whatever we have and return the data. ##
            self.has_more_data = False
            return inp_tensor
    
    def is_data_rem(self):
        return self.has_more_data

    def reset_data(self):
        self.inp_iter = next(self.inp_dict)

curr_dir = os.getcwd() + "/"

train_ds = load_dataset("c4", "en", split="train")
val_ds = load_dataset("c4", "en", split="validation")

BUFFER_SIZE = 20000
BATCH_SIZE = args.batch_size

## We make the batches here. ##
train_loader = C4Loader(train_ds, args.batch_size)
val_loader = C4Loader(val_ds, args.batch_size)

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

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def accuracy_function(real, pred, weights):

  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(weights, 0))

  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  ## Corner case, when nothing is masked (prob of 0.85^(Batch Size * Sequence Length)) then we will get NaNs propagated. ##
  epsilon = 1e-9

  return tf.reduce_sum(accuracies) / (tf.reduce_sum(mask) + epsilon)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

checkpoint_path = './checkpoints/train/' + str(args.attention_type) + '/' + str(initial_learning_rate) + '/' + str(args.warmup)

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

def train_step(inputs):
  global en_tokenizer

  input_tok = en_tokenizer.tokenize(inputs)

  (inp, tar_inp), tar_real, weight = mask_data(input_tok)

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_object(tar_real, predictions, sample_weight=weight[:, 1:]) 
    accuracy = accuracy_function(tar_real, predictions, weight[:, 1:])

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss.update_state(loss, sample_weight=weight[:, 1:])
  train_accuracy(accuracy)

def val_step(inputs):
  global en_tokenizer

  input_tok = en_tokenizer.tokenize(inputs)

  (inp, tar_inp), tar_real, weight = mask_data(input_tok)

  predictions, _ = transformer([inp, tar_inp],
                                training = True)
  loss = loss_object(tar_real, predictions, sample_weight=weight[:, 1:]) 

  accuracy = accuracy_function(tar_real, predictions, weight[:, 1:])

  val_loss.update_state(loss, sample_weight=weight[:, 1:])
  val_accuracy(accuracy)

EPOCHS = 30
total_steps_required = args.num_steps

steps_elapsed = 0

train_start = time.time()
for epoch in range(EPOCHS):
  start = time.time()
  if steps_elapsed > total_steps_required:
    break

  batch = 0 ## We initialize the batch to start at 0. ##

  train_loss.reset_states()
  train_accuracy.reset_states()

  while train_loader.is_data_rem():
    if steps_elapsed > total_steps_required:
      break
    inp = train_loader.gen_next_train_data()
    train_step(inp)
    if (steps_elapsed % 1000 == 0):
      # We print end-to-end time here just in case.
      print(f'----------- End-to-End: {time.time() - train_start} -----------')
    if (steps_elapsed % 5000 == 0):
      save_path = ckpt_manager.save()
      print(f'Saved checkpoint for step: {steps_elapsed} path: {save_path}')
      ckpt.step.assign_add(1)

    print(f'Steps {steps_elapsed} Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

    with open(f'./train_data_{args.attention_type}_{args.rank}.txt', 'a+') as f:
      f.write(f'{steps_elapsed} {train_loss.result():.4f} {train_accuracy.result():.4f}\n')

    with open(f'./train_stats_{args.attention_type}_{args.rank}.txt', 'a+') as f:
        f.write(f'{steps_elapsed} MHA {Stats.mha_time:.4f} MHA-Enc {Stats.mha_enc_time:.4f} MHA-Causal {Stats.mha_causal_time:.4f} MHA-Enc-Dec {Stats.mha_enc_dec_time:.4f} FFN {Stats.ffn_time:.4f} Downsampling {Stats.downsampling_time:.4f} Kernel-Transformation {Stats.transformation_time:.4f}\n')

    steps_elapsed += 1

    ## Here, we validate the data.
    if (steps_elapsed % val_iter_freq == 0 and steps_elapsed != 0):
      val_loss.reset_states()
      val_accuracy.reset_states()

      while val_loader.has_more_data():
        inp = val_loader.gen_next_train_data()
        val_step(inp)

      with open(f'./val_data_{args.attention_type}.txt', "a+") as f:
        f.write(f'{val_loss.result():.4f} {val_accuracy.result():.4f}\n')

      ## We reset the data.
      val_loader.reset_data()

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}', flush=True)

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n', flush=True)

  batch += 1

train_end = time.time()

print(f'Total training time: {train_end-train_start}\n', flush=True)
