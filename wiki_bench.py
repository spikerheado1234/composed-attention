import tensorflow as tf
from vanilla_transformer import Transformer
import tensorflow_datasets as tfds
import tensorflow_text
import time
import argparse
from stats import Stats 
import os
from pre_train_wiki_loader import get_train_ds, make_batches
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
parser.add_argument('--rank', dest='rank', type=int, default=1, help='The rank of the process, to distinguish output.')
parser.add_argument('--encoder_only', dest='enc_only', action='store_true', help='Whether we are training in encoder only mode')
parser.add_argument('--hidden_dim', dest='hid_dim', type=int, default=512, help='The size of the hidden dimension.')

args = parser.parse_args()

## Define global vars here. ##

MAX_TOKENS = args.sequence_length

### Data preparation 

curr_dir = os.getcwd() + "/"

train_ds = get_train_ds()

BUFFER_SIZE = 20000
BATCH_SIZE = args.batch_size

## We make the batches here. ##
train_batches = make_batches(train_ds, BUFFER_SIZE, BATCH_SIZE)

## Hyperparameters ##
num_layers = args.layers
d_model = args.hid_dim
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

def accuracy_function(real, pred, weights):

  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(weights, 0))

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

checkpoint_path = './checkpoints/train/' + str(rank)

ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

train_step_signature = [
    (
         tf.TensorSpec(shape=(None, None), dtype=tf.int64),
         tf.TensorSpec(shape=(None, None), dtype=tf.int64)),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

## Finally, our masked language model. ##
class MaskedLM(tf.keras.Model):
    def __init__(self, transformer, target_vocab_size, encoder_only):
        super(MaskedLM, self).__init__()

        self.transformer = transformer
        self.encoder_only = encoder_only

        # The final linear layer.
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    ##@tf.function
    def call(self, inp):
      output, attention_weights = self.transformer(inp)

      final_output = self.final_layer(output)  # Shape `(batch_size, tar_seq_len, target_vocab_size)`.

      # Return the final output and the attention weights.
      return final_output, attention_weights

masked_lm = MaskedLM(transformer, Constants.wiki_vocab_size, True)

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

  train_step_start = time.time()
  if args.enc_only: ## We must then remove the end token from the input into the encoder.
    inp = tf.convert_to_tensor(inp.numpy()[:, :-1])
  forward_prop_start = time.time()
  predictions, _ = masked_lm([inp, tar_inp], training = True)
  forward_prop_end = time.time()
  Stats.total_forward_prop_time += (forward_prop_end - forward_prop_start)
  loss = loss_object(tar_real, predictions, sample_weight=weight[:, 1:]) 
  accuracy = accuracy_function(tar_real, predictions, weight[:, 1:])

  train_step_end = time.time()
  Stats.train_step_time += train_step_end - train_step_start

  train_loss.update_state(loss, sample_weight=weight[:, 1:])
  train_accuracy(accuracy)
  train_perplexity(perplexity_function(train_loss.result()))

def train_step(inputs, labels):
  (inp, tar_inp) = inputs
  tar_real = labels

  (inp, tar_inp), tar_real, weight = mask_data(inp)

  train_step_start = time.time()
  with tf.GradientTape() as tape:
    if args.enc_only: ## We must then remove the end token from the input into the encoder.
      inp = tf.convert_to_tensor(inp.numpy()[:, :-1])
    forward_prop_start = time.time()
    predictions, _ = masked_lm([inp, tar_inp], training = True)
    forward_prop_end = time.time()
    Stats.total_forward_prop_time += (forward_prop_end - forward_prop_start)
    loss = loss_object(tar_real, predictions, sample_weight=weight[:, 1:]) 
    accuracy = accuracy_function(tar_real, predictions, weight[:, 1:])

  gradient_start = time.time()
  gradients = tape.gradient(loss, transformer.trainable_variables)
  gradient_end = time.time()
  Stats.gradient_computation += (gradient_end - gradient_start)

  optimiser_start = time.time()
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  optimiser_end = time.time()
  Stats.optimser_step += (optimiser_end - optimiser_start)
  train_step_end = time.time()
  Stats.train_step_time += train_step_end - train_step_start

  train_loss.update_state(loss, sample_weight=weight[:, 1:])
  train_accuracy(accuracy)
  train_perplexity(perplexity_function(train_loss.result()))


EPOCHS = 30
total_steps_required = 100

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
    val_step(inp, tar)

    steps_elapsed += 1

train_end = time.time()

with open(f"benchmark_results_{args.attention_type}.txt", "a+") as f:
    f.write(f"Train Step Time: {Stats.train_step_time:.3f}, sequence length: {args.sequence_length}, downsampling value: {args.downsampling_k}, hidden_dim: {d_model} Attn Heads: {num_attention_heads}\n")
    f.write(f"Forward Prop: {Stats.total_forward_prop_time:.3f} MHA {Stats.mha_time:.3f} MHA FFN {Stats.mha_ffn:.3f} Favour Total Time {Stats.favour_time:.3f} Kernel Transformation {Stats.transformation_time:.3f} Downsampling {Stats.downsampling_time:.3f} Downsample-Mat-Mul {Stats.downsampling_mat_mul:.3f} Downsample-Mat-Gen {Stats.downsampling_mat_gen:.3f} Linear Transformation {Stats.linear_transformation:.3f} QKV Product: {Stats.q_k_v_product:.3f} Transpose Time: {Stats.transpose_time:.3f} Expand Dims Time: {Stats.expand_dims_time:.3f}\n")
    #f.write(f"Forward Prop: {Stats.total_forward_prop_time}\n")
    #f.write(f"Forward Prop: {Stats.total_forward_prop_time} MHA {tf.get_static_value(Stats.mha_time):.4f} MHA-Enc {tf.get_static_value(Stats.mha_enc_time):.4f} MHA-FFN: {tf.get_static_value(Stats.mha_ffn):.4f} Favour Attn: {tf.get_static_value(Stats.favour_time):.4f} FFN {tf.get_static_value(Stats.ffn_time):.4f} Downsampling {tf.get_static_value(Stats.downsampling_time):.4f} Kernel-Transformation {tf.get_static_value(Stats.transformation_time):.4f}  Computation: {Stats.gradient_computation} Optimisation Step: {Stats.optimser_step}\n")
    #f.write(f"Q-K-V Product: {tf.get_static_value(Stats.q_k_v_product):.4f} Linear Transformation: {tf.get_static_value(Stats.linear_transformation):.4f}\n")
