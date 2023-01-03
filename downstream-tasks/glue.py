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
parser.add_argument('--task', dest='task', default="cola", type=str, help='The GLUE task to fine-tune on.')

args = parser.parse_args()

## Define global vars here. ##

BUFFER_SIZE = 20000

MAX_TOKENS = args.sequence_length

BATCH_SIZE = args.batch_size

### Data preparation 

curr_dir = os.getcwd() + "/"

train_data = None
val_data = None
prepare_helper = None
loss_object = None
accuracy_function = None

## Some helper methods. ##

## The invariant is that the enc_part and dec_part can be readily 
## fed into the transformer. All that is required is:
## 1.) Morphing each sequence length to MAX_TOKENS.
## 2.) Adding the start and end tokens.
## 3.) Producing the real output. Which is the "ground truth" we will compare to for teacher forcing.
def prepare_transformer_input(enc_part, dec_part):
    global MAX_TOKENS
    enc_part = enc_part[:, :MAX_TOKENS]
    enc_part = pad(enc_part, MAX_TOKENS)
    enc_part = enc_part[:, :-2]
    enc_part = add_start_end(enc_part)
    
    dec_part = add_start_end(dec_part)
    real_dec_part = dec_part[:, :-1]
    output_comparison = dec_part[:, 1:]
    
    return enc_part, real_dec_part, output_comparison

def prepare_cola(inp):
    global en_tokenizer, MAX_TOKENS

    sentence = inp['sentence']
    label = inp['label']
        
    inp_tok = en_tokenizer.tokenize(sentence)

    return inp_tok.merge_dims(-2, -1).to_tensor(), tf.reshape(label, shape=(label.shape[0], 1)) ## Tuple of (Tokenized input, answer)

def cola_accuracy(real, pred):
    accuracies = tf.math.equal(tf.cast(pred, dtype=tf.int64), real)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.sum(tf.ones(shape=real.shape, dtype=tf.float32))

if args.task == "cola":
    train_data = tfds.load(name="glue/cola", split="train").map(prepare_cola)
    val_data = tfds.load(name="glue/cola", split="validation").map(prepare_cola)
    prepare_helper = prepare_cola 
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
    accuracy_function = cola_accuracy
elif args.task == "sst2":
    pass
elif args.task == "mrpc":
    pass
else:
    raise Exception("Incorrect task specified")

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


class DownstreamModel(tf.keras.Model):
    def __init__(self, transformer, glue_task):
        self.glue_task = glue_task
        self.transformer = transformer

        if glue_task == "cola":
            self.layer_out = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu) ## Again, another hyperparameter to be tuned.
            self.final_out = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

    def call(self, input):
        output, _ = self.transformer(input)

        if self.glue_task == "cola":
            output = self.layer_out(output)
            output = self.final_out(output)
            return output


downstream_model = DownstreamModel(transformer, args.task)

def val_step(enc_part, dec_part):

  enc_part, dec_part, real_val = prepare_transformer_input(enc_part, dec_part)

  predictions, _ = downstream_model([enc_part, dec_part],
                                      training = False)

  loss = loss_object(real_val, predictions) 
  accuracy = accuracy_function(real_val, predictions)

  train_loss.update_state(loss)
  train_accuracy(accuracy)

def train_step(enc_part, dec_part):

  enc_part, dec_part, real_val = prepare_transformer_input(enc_part, dec_part)

  with tf.GradientTape() as tape:
    predictions, _ = downstream_model([enc_part, dec_part],
                                      training = True)

    loss = loss_object(real_val, predictions) 
    accuracy = accuracy_function(real_val, predictions)

  gradients = tape.gradient(loss, downstream_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, downstream_model.trainable_variables))

  train_loss.update_state(loss)
  train_accuracy(accuracy)

EPOCHS = 30
total_steps_required = args.num_steps

steps_elapsed = 0

train_start = time.time()
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  pdb.set_trace()
  for (batch, (enc_part, dec_part)) in enumerate(train_data):
    train_step(enc_part, dec_part)

  train_loss.reset_states()
  train_accuracy.reset_states()
  for (batch, (enc_part, dec_part)) in enumerate(val_data):
    val_step(enc_part, dec_part)

  ## We write to file the validation information.
  with open(f'{args.attention_type}_val_data_glue_{args.task}.txt', "a+") as f:
    f.write(f'{train_loss.result():.4f} {train_accuracy.result():.4f}\n')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n', flush=True)

train_end = time.time()

print(f'Total training time: {train_end-train_start}\n', flush=True)
