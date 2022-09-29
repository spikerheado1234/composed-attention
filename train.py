import tensorflow as tf
from vanilla_transformer import Transformer
import tensorflow_datasets as tfds
import tensorflow_text
import time
import argparse
from stats import Stats 
import os

## Define argument parsing and help over here. ##

parser = argparse.ArgumentParser(description='Train compositions of efficient Transformer Variants.')
parser.add_argument('--type', dest='attention_type', default='MHA', choices=['MHA', 'LinMHA', 'PerfMHA'], help='The type of attention mechanism you wish to train a Transformer on. (Possible Values are: MHA, LinMHA or PerfMHA)')
parser.add_argument('--downsampling_k', dest='downsampling_k', default=32, type=int, help='The dimension you wish to downsample the sequence length to in accordance to the LinFormer Paper.')
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='the batch size used for training & inference purposes')
parser.add_argument('--layers', dest='layers', default=4, type=int, help='the number of layers in the transformer.')
parser.add_argument('--sequence_length', dest='sequence_length', type=int, default=128, help='the sequence length of the input to the transformer')

args = parser.parse_args()

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    if args.attention_type == 'MHA':
      tf.config.set_visible_devices(gpus[0], 'GPU')
    elif args.attention_type == 'LinMHA' and len(gpus) > 1:
      tf.config.set_visible_devices(gpus[1], 'GPU') 
    elif args.attention_type == 'PerfMHA' and len(gpus) > 2:
      tf.config.set_visible_devices(gpus[2], 'GPU')
    else:
      tf.config.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    print(e)

## Define global vars here. ##

MAX_TOKENS = args.sequence_length

### Data preparation 

curr_dir = os.getcwd() + "/"

def load_data():
  examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                              with_info=True,
                              as_supervised=True)

  train_examples, val_examples = examples['train'], examples['validation']
  return train_examples, val_examples


def load_tokenizer():
  global curr_dir

  model_name = 'ted_hrlr_translate_pt_en_converter'
  tf.keras.utils.get_file(
      f'{model_name}.zip',
      f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
      cache_dir='.', cache_subdir='', extract=True
  )

  tf.saved_model.LoadOptions(experimental_io_device="CPU:0")
  tokenizers = tf.saved_model.load(curr_dir + model_name)
  return tokenizers

train_examples, val_examples = load_data()
tokenizers = load_tokenizer()

def prepare_batch(pt, en):
  pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
  pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
  pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

  en = tokenizers.en.tokenize(en)
  en = en[:, :(MAX_TOKENS+1)]
  en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
  en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

  return (pt, en_inputs), en_labels

BUFFER_SIZE = 20000
BATCH_SIZE = args.batch_size

def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))


## We make the batches here. ##
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

## Hyperparameters ##
num_layers = args.layers
d_model = 128
dff = 512
num_attention_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_attention_heads=num_attention_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate,
    downsampling_value=args.downsampling_k if args.attention_type == 'LinMHA' else 32, # Just default to 32 otherwise, doesn't matter since it won't be used.
    attention_type=args.attention_type)

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

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

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

def pad_vector(inputs):
  global MAX_TOKENS 
  """
  Pads the vector inputs (of size (BATCH_SIZE, SEQUENCE LENGTH)) to ensure each
  sequence length is standardized to MAX_TOKENS.
  """
  max_seq_len = inputs.shape[1]
  current_batch_size = inputs.shape[0]
  zero_vector = tf.zeros(shape=(current_batch_size, MAX_TOKENS - max_seq_len), dtype=tf.int64)

  result = tf.concat([inputs, zero_vector], axis=1)
  return result

def train_step(inputs, labels):
  (inp, tar_inp) = inputs
  tar_real = labels

  inp = pad_vector(inp)
  tar_inp = pad_vector(tar_inp)
  tar_real = pad_vector(tar_real)

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))

EPOCHS = 30

train_start = time.time()

## This is the code for the benchmarking experiment.
def train_step_prac(inputs, labels):
  (inp, tar_inp) = inputs
  tar_real = labels

  inp = pad_vector(inp)
  tar_inp = pad_vector(tar_inp)
  tar_real = pad_vector(tar_real)

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

repititions = 100
exp_file = curr_dir + "data/benchmarks/" + args.attention_type + f"/{BATCH_SIZE}bs_{MAX_TOKENS}sl_{num_layers}nl_data.txt"

train_start = time.time()
for (batch, (inp, tar)) in enumerate(train_batches):

  # We basically run the same batch over and over again 100 times.
  for _ in range(repititions):
    try:
      train_step_prac(inp, tar)
    except Exception:
      with open(exp_file, "a+") as f:
        f.write(f'Exception encountered, probably OOM\n')
      raise Exception

  # We only want to run the same data over and over again.
  break

with open(exp_file, "a+") as f:
  f.write(f'MHA {Stats.mha_time:.4f} MHA-Enc {Stats.mha_enc_time:.4f} MHA-Causal {Stats.mha_causal_time:.4f} MHA-Enc-Dec {Stats.mha_enc_dec_time:.4f} FFN {Stats.ffn_time:.4f} Downsampling {Stats.downsampling_time:.4f} Kernel-Transformation {Stats.transformation_time:.4f}\n')

train_end = time.time()

with open(exp_file, "a+") as f:
  f.write(f'End-to-End: {train_end - train_start}\n')

## The code has finished here.

"""
Actual training loop, put it back into the code post inference experiment.
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_batches):
    train_step(inp, tar)

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
"""
