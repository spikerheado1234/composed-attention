import sys

## First append the parent directory.
sys.path.append('/home/ag82/composed-attention/')

from vanilla_transformer import Transformer
import tensorflow as tf
import os
import tensorflow_datasets as tfds
import argparse
import time
import numpy as np

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
parser.add_argument('--learning_rate', dest='lr_rate', type=float, default=0.0002, help='the largest constant in the lr schedule.')
parser.add_argument('--pre-train-data', dest='pt_data', type=str, default="wiki-text", help=' the pre-train dataset we are taking the best checkpoint of.')

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
    ## May have to add additional padding just so that it doesn't complain for Linformer type architectures due to poor state capturing whilst checkpointing.

    ## This is the part that is added, may impact accuracy deliteriously.##
    dec_part = pad(dec_part, MAX_TOKENS)
    dec_part = dec_part[:, :-2]
    ## Over here, to prevent any causal leakage, we do NOT feed in the input into the decoder.
    ## First, we create a tensor of all zeros.
    real_dec_part = tf.zeros(shape=dec_part.shape, dtype=tf.int64)
    ## We prepend the start token and append the end token.
    real_dec_part = add_start_end(real_dec_part)
    real_dec_part = real_dec_part[:, :-1] ## We then remove the end token.
    
    dec_part = add_start_end(dec_part)
    output_comparison = dec_part[:, 1:]
    ones = np.ones(shape=(output_comparison.shape))
    zeros = np.zeros(shape=(output_comparison.shape))
    weights = tf.cast(tf.convert_to_tensor(np.where(output_comparison.numpy() > 5, ones, zeros)), dtype=tf.int64)
    
    return enc_part, real_dec_part, output_comparison, weights

def compute_accuracy(real, pred, weights):
    accuracies = tf.math.equal(real, tf.math.argmax(pred, axis=-1))
    mask = tf.math.equal(weights, tf.ones(shape=weights.shape, dtype=tf.int64))
    accuracies &= mask
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(tf.cast(weights, dtype=tf.float32))

## COLA HELPER METHODS. ##
def prepare_cola(inp):
    global en_tokenizer, MAX_TOKENS

    prefix = tf.convert_to_tensor(['cola sentence: '])
    sentence = prefix + inp['sentence']
    label = inp['label']
    label = en_tokenizer.tokenize(tf.convert_to_tensor(label.numpy().astype('S')))
        
    inp_tok = en_tokenizer.tokenize(sentence)

    return inp_tok.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor() ## Tuple of (Tokenized input, answer)

## SST-2 HELPER METHODS. ##
def prepare_sst2(inp):
    global en_tokenizer, MAX_TOKENS

    prefix = tf.convert_to_tensor(["sst2 sentence: "])
    sentence = prefix + inp['sentence']
    label = inp['label']
    label = en_tokenizer.tokenize(tf.convert_to_tensor(label.numpy().astype('S')))
        
    inp_tok = en_tokenizer.tokenize(sentence)
    return inp_tok.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor()

## MRPC HELPER METHODS ##
def prepare_mrpc(inp):
    global en_tokenizer
    
    prefix_one = tf.convert_to_tensor(['mrpc sentence 1: '])
    prefix_two = tf.convert_to_tensor(['sentence 2: '])
    
    sentence_one = inp['sentence1']
    sentence_two = inp['sentence2']
    label = inp['label']
    sentence_one = prefix_one + sentence_one
    sentence_two = prefix_two + sentence_two
    enc_input = tf.convert_to_tensor(sentence_one.numpy() + sentence_two.numpy())
    enc_input = en_tokenizer.tokenize(enc_input)
    label = en_tokenizer.tokenize(tf.convert_to_tensor(label.numpy().astype('S')))
    
    return enc_input.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor()

## QQP HELPER METHODS. ##
def prepare_qqp(inp):
    global en_tokenizer
    
    prefix_one = tf.convert_to_tensor(['qqp question 1: '])
    prefix_two = tf.convert_to_tensor(['question 2: '])
    
    sentence_one = inp['question1']
    sentence_two = inp['question2']
    label = inp['label']
    sentence_one = prefix_one + sentence_one
    sentence_two = prefix_two + sentence_two
    enc_input = tf.convert_to_tensor(sentence_one.numpy() + sentence_two.numpy())
    enc_input = en_tokenizer.tokenize(enc_input)
    label = en_tokenizer.tokenize(tf.convert_to_tensor(label.numpy().astype('S')))
    
    return enc_input.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor()

## STSB HELPER METHODS. ##
def prepare_stsb(inp):
    global en_tokenizer
    
    prefix_one = tf.convert_to_tensor(['stsb sentence 1: '])
    prefix_two = tf.convert_to_tensor(['sentence 2: '])
    
    sentence_one = inp['sentence1']
    sentence_two = inp['sentence2']
    label = inp['label']
    sentence_one = prefix_one + sentence_one
    sentence_two = prefix_two + sentence_two
    enc_input = tf.convert_to_tensor(sentence_one.numpy() + sentence_two.numpy())
    enc_input = en_tokenizer.tokenize(enc_input)
    label = tf.cast(tf.math.round(label), dtype=tf.int64) ## We cast this simple classification by rounding to nearest integer.
    label = en_tokenizer.tokenize(tf.convert_to_tensor(label.numpy().astype('S')))
    
    return enc_input.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor()

## MNLI Helper methods. ##
def prepare_mnli(inp):
    global en_tokenizer
    
    prefix_one = tf.convert_to_tensor(['mnli hypothesis: '])
    prefix_two = tf.convert_to_tensor(['premise: '])
    
    premise = inp['premise']
    hypothesis = inp['hypothesis']
    label = inp['label']
    sentence_one = prefix_one + hypothesis
    sentence_two = prefix_two + premise
    enc_input = tf.convert_to_tensor(sentence_one.numpy() + sentence_two.numpy())
    enc_input = en_tokenizer.tokenize(enc_input)
    label = en_tokenizer.tokenize(label.numpy().astype('S'))
    
    return enc_input.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor()

## QNLI helper methods. ##
def prepare_qnli(inp):
    global en_tokenizer
    
    prefix_one = tf.convert_to_tensor(['qnli question: '])
    prefix_two = tf.convert_to_tensor(['sentence: '])
    
    question = inp['question']
    sentence = inp['sentence']
    label = inp['label']
    sentence_one = prefix_one + question
    sentence_two = prefix_two + sentence
    enc_input = tf.convert_to_tensor(sentence_one.numpy() + sentence_two.numpy())
    enc_input = en_tokenizer.tokenize(enc_input)
    label = en_tokenizer.tokenize(label.numpy().astype('S'))
    
    return enc_input.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor()

## RTE helper methods. ##
def prepare_rte(inp):
    global en_tokenizer
    
    prefix_one = tf.convert_to_tensor(['rte sentence 1: '])
    prefix_two = tf.convert_to_tensor(['sentence 2: '])
    
    sentence_one = inp['sentence1']
    sentence_two = inp['sentence2']
    label = inp['label']
    sentence_one = prefix_one + sentence_one
    sentence_two = prefix_two + sentence_two
    enc_input = tf.convert_to_tensor(sentence_one.numpy() + sentence_two.numpy())
    enc_input = en_tokenizer.tokenize(enc_input)
    label = en_tokenizer.tokenize(label.numpy().astype('S'))
    
    return enc_input.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor()

## WNLI Helper methods. ##
def prepare_wnli(inp):
    global en_tokenizer
    
    prefix_one = tf.convert_to_tensor(['wnli sentence 1: '])
    prefix_two = tf.convert_to_tensor(['sentence 2: '])
    
    sentence_one = inp['sentence1']
    sentence_two = inp['sentence2']
    label = inp['label']
    sentence_one = prefix_one + sentence_one
    sentence_two = prefix_two + sentence_two
    enc_input = tf.convert_to_tensor(sentence_one.numpy() + sentence_two.numpy())
    enc_input = en_tokenizer.tokenize(enc_input)
    label = en_tokenizer.tokenize(label.numpy().astype('S'))
    
    return enc_input.merge_dims(-2, -1).to_tensor(), label.merge_dims(-2, -1).to_tensor()

if args.task == "cola":
    train_data = tfds.load(name="glue/cola", split="train").batch(args.batch_size)
    val_data = tfds.load(name="glue/cola", split="validation").batch(args.batch_size)
    prepare_helper = prepare_cola 
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
elif args.task == "sst2":
    train_data = tfds.load(name="glue/sst2", split="train").batch(args.batch_size)
    val_data = tfds.load(name="glue/sst2", split="validation").batch(args.batch_size)
    prepare_helper = prepare_sst2
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
elif args.task == "mrpc":
    train_data = tfds.load(name='glue/mrpc', split='train').batch(args.batch_size)
    val_data = tfds.load(name='glue/mrpc', split='validation').batch(args.batch_size)
    prepare_helper = prepare_mrpc
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
elif args.task == "qqp":
    train_data = tfds.load(name='glue/qqp', split='train').batch(args.batch_size)
    val_data = tfds.load(name='glue/qqp', split='validation').batch(args.batch_size)
    prepare_helper = prepare_qqp
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
elif args.task == "stsb":
    train_data = tfds.load(name='glue/stsb', split='train').batch(args.batch_size)
    val_data = tfds.load(name='glue/stsb', split='validation').batch(args.batch_size)
    prepare_helper = prepare_stsb
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
elif args.task == "mnli":
    train_data = tfds.load(name='glue/mnli', split='train').batch(args.batch_size)
    val_data_mismatched = tfds.load(name='glue/mnli', split='validation_mismatched').batch(args.batch_size)
    val_data_matched = tfds.load(name='glue/mnli', split='validation_matched').batch(args.batch_size)
    ## TODO, check if concatenation is correct. ##
    val_data = val_data_mismatched.concatenate(val_data_matched)
    prepare_helper = prepare_mnli
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
elif args.task == "qnli":
    train_data = tfds.load(name='glue/qnli', split='train').batch(args.batch_size)
    val_data = tfds.load(name='glue/qnli', split='validation').batch(args.batch_size)
    prepare_helper = prepare_qnli 
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
elif args.task == "rte":
    train_data = tfds.load(name='glue/rte', split='train').batch(args.batch_size)
    val_data = tfds.load(name='glue/rte', split='validation').batch(args.batch_size)
    prepare_helper = prepare_rte 
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
elif args.task == "wnli":
    train_data = tfds.load(name='glue/wnli', split='train').batch(args.batch_size)
    val_data = tfds.load(name='glue/wnli', split='validation').batch(args.batch_size)
    prepare_helper = prepare_wnli 
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
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
optimizer = tf.keras.optimizers.Adam(warmup_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

checkpoint_path = './checkpoints/train/' + str(args.attention_type) + '/' + str(args.pt_data) + '/' + str(initial_learning_rate) + '/' + str(args.warmup)

ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')
else:
    ## Otherwise we raise an exception. We should ALWAYS load from a checkpoint over here. ##
    raise Exception("Did not find any suitable model checkpoint to load from.")

class DownstreamModel(tf.keras.Model):
    def __init__(self, transformer, glue_task, target_vocab_size):
        super(DownstreamModel, self).__init__()
        self.glue_task = glue_task
        self.transformer = transformer

        if args.task == "cola" or args.task == "sst2" or args.task == "mrpc" or args.task == "qqp" or args.task == "stsb" or args.task == "mnli" or args.task == "qnli" or args.task == "rte" or args.task == "wnli":
            self.layer_out = tf.keras.layers.Dense(target_vocab_size) ## Again, another hyperparameter to be tuned.
        else:
            self.layer_expand = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
            self.layer_out = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

    @tf.function
    def call(self, inp):
        output, _ = self.transformer(inp)
        if args.task == "cola" or args.task == "sst2" or args.task == "mrpc" or args.task == "qqp" or args.task == "stsb" or args.task == "mnli" or args.task == "qnli" or args.task == "rte" or args.task == "wnli":
            return self.layer_out(output)
        else:
            output = self.layer_expand(output)
            output = self.layer_out(output)
            return output

downstream_model = DownstreamModel(transformer, args.task, Constants.wiki_vocab_size)

def val_step(inp):

  enc_part, dec_part = prepare_helper(inp)
  enc_part, dec_part, real_val, weights = prepare_transformer_input(enc_part, dec_part)

  predictions = downstream_model([enc_part, dec_part],
                                  training = False)

  loss = loss_object(real_val, predictions, sample_weight=weights) 
  accuracy = compute_accuracy(real_val, predictions, weights)

  train_loss.update_state(loss)
  train_accuracy(accuracy)

def train_step(inp):

  enc_part, dec_part = prepare_helper(inp)
  enc_part, dec_part, real_val, weights = prepare_transformer_input(enc_part, dec_part)

  with tf.GradientTape() as tape:
    predictions = downstream_model([enc_part, dec_part],
                                    training = True)

    loss = loss_object(real_val, predictions, sample_weight=weights) 
    accuracy = compute_accuracy(real_val, predictions, weights)

  gradients = tape.gradient(loss, downstream_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, downstream_model.trainable_variables))

  train_loss.update_state(loss)
  train_accuracy(accuracy)

EPOCHS = 3

## We write everything to one file.
with open(f'{args.attention_type}_val_data_glue.txt', "a+") as f:
    f.write(f"--------{args.task}--------\n")

train_start = time.time()
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  for batch, inp in enumerate(train_data):
    train_step(inp)
    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  train_loss.reset_states()
  train_accuracy.reset_states()
  for batch, inp in enumerate(val_data):
    val_step(inp)

  ## We write to file the validation information.
  with open(f'{args.attention_type}_val_data_glue.txt', "a+") as f:
    f.write(f'{train_loss.result():.4f} {train_accuracy.result():.4f}\n')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n', flush=True)

train_end = time.time()

print(f'Total training time: {train_end-train_start}\n', flush=True)
