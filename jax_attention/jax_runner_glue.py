import sys
import os

sys.path.append(os.getcwd())

import pdb
import numpy as np
from functools import partial

import jax
from jax import value_and_grad, grad, jit
import jax.numpy as jnp
from jax import random
import argparse
from constants import Constants
import time
from jax.interpreters import xla

import flax.linen as nn
from flax.training import checkpoints, train_state
from flax.core.frozen_dict import freeze
import optax

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

args = parser.parse_args()
param_key = random.PRNGKey(42)
dropout_key = random.PRNGKey(43)
enc_input = jnp.round(random.uniform(random.PRNGKey(44), (args.batch_size,args.sequence_length)) * Constants.wiki_vocab_size).astype(jnp.int32)
dec_input = jnp.round(random.uniform(random.PRNGKey(45), (args.batch_size, args.sequence_length)) * Constants.wiki_vocab_size).astype(jnp.int32)

import tensorflow as tf
import tensorflow_datasets as tfds

from tokenization_proc import pad, add_start_end
from transformer_skeleton import Transformer


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

## Maps the token, classify, to its token ID. I manually looked at the 
## vocabublary and extracted this.
key_words = {"classify" : 17693}
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

    ## Finally, if we are running in encoder only mode, we replace the first, start, 
    ## token with the classify token (this is vocab dependent).
    if args.enc_only:
        enc_part_numpy = enc_part.numpy()
        enc_part_numpy[:, 0] = key_words["classify"]
        enc_part = tf.convert_to_tensor(enc_part_numpy, dtype=tf.int64)
        ## We also remove the end token to keep tensor sizes the same
        ## and tensorflow happy.
        enc_part = enc_part[:, :-1]

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

## Hyperparameters ##
num_layers = args.layers
d_model = 768
dff = 3072
num_attention_heads = 12
dropout_rate = 0.1
rank = args.rank
learning_rate = args.lr_rate

## Then, we create the learning rate schedule. ##
initial_learning_rate = 0.0001
num_train_steps = args.num_steps
warmup_steps = args.warmup
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-9)

class DownsStreamModel(nn.Module):
    transformer : Transformer 
    vocabulary_size : int

    def setup(self):
        self.last_ffn = nn.Dense(self.vocabulary_size)
        self.last_ffn_binary = nn.Dense(2)
    
    def __call__(self, encoder_input, decoder_input, *, train):
        output = self.transformer(encoder_input, decoder_input, train=train)

        ## Need to resort to this hack to compute output.
        ## Will the gradient computation be correct as well?
        output_one = self.last_ffn(output)
        output_two = self.last_ffn_binary(output)
        output = jax.lax.cond(args.task == "mnli", output_one, output_two)
        return output

## We create a triangular schedule here. ##
def create_lr_schedule(peak_lr, warmup_steps, total_step_count):
    rem_steps = total_step_count - warmup_steps
    return optax.join_schedules([
        optax.linear_schedule(init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps),
        optax.linear_schedule(init_value=peak_lr, end_value=0.0, transition_steps=rem_steps)
    ], [warmup_steps])

transformer = Transformer(d_model, int(d_model / num_attention_heads), num_attention_heads, dropout_rate, (args.sequence_length - 1) if args.enc_only else args.sequence_length, dff, num_layers, Constants.wiki_vocab_size, args.enc_only, args.downsampling_k, args.attention_type)

## We have to first initialize the model & optimizer. ##
downstream_model = DownsStreamModel(transformer, Constants.wiki_vocab_size)
params = downstream_model.init({'params': param_key, 'dropout': dropout_key}, enc_input, dec_input, train=True)
optimizer = optax.adam(create_lr_schedule(args.lr_rate, args.warmup, args.num_steps))
opt_state = optimizer.init(params) ## TODO, check for correctness over here.

## We prepare the checkpoint information. ##
checkpoint_path = './checkpoints/train/' + str(args.attention_type) + '/' + str(args.lr_rate) + '/' + str(args.warmup)
ckpt_count = 0
state = train_state.TrainState.create(apply_fn=transformer.apply, params=params['params']['transformer'], tx=optimizer)
## We attempt to restore the latest checkpoint. ##
if checkpoints.latest_checkpoint(ckpt_dir=checkpoint_path):

    ## Honestly, not super confident this is correct, but it's at least a start. I need to debug this
    ## extremely carefully. 
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=state)
    unfrozen_dict = params.unfreeze()
    unfrozen_loaded_params = restored_state.params.unfreeze()
    unfrozen_dict['params']['transformer'] = unfrozen_loaded_params
    params = freeze(unfrozen_dict)
    print('Latest checkpoint restored!', flush=True)
else: ## Otherwise we create a checkpoint. ##
    checkpoints.save_checkpoint(ckpt_dir=checkpoint_path, target=state, step=ckpt_count)
    print('checkpoint succesfully created.', flush=True)

@partial(jax.jit, static_argnames=['train'])
def compute_loss(parameters, inp, tar_inp, train, dropout_key, real, weights):
    logits = downstream_model.apply(parameters, inp, tar_inp, train=train, rngs={'dropout': dropout_key})
    ## TODO, is this the correct way to implement MLM? How can I know if this is correct?
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, real)
    loss *= weights
    return loss.sum() / jnp.sum(weights) 

def compute_wnli_accuracy(logits, real):
    raise Exception("Not implemented yet.")

def compute_binary_accuracy(logits, real):
    ## We take the first token's output and compare it to real.
    logits = logits[:, :1]
    logits = jnp.argmax(logits, axis=-1)
    accuracy = jnp.where(logits == real, 1, 0)
    return jnp.sum(accuracy) / jnp.array(real.shape[0])

def train_step(parameters, inp, tar_inp, data_real, train, weights, dropout_key, opt_state, optimizer, batch_number):
    loss, grads = value_and_grad(compute_loss)(parameters, inp, tar_inp, train, dropout_key, data_real, weights)
    updates, opt_state = optimizer.update(grads, opt_state, parameters)
    params = optax.apply_updates(parameters, updates)
    ## Returns the tuple of: (new model parameters, optimizer state, loss).
    return params, opt_state, loss

@jax.jit
def apply_model(parameters, inp, tar_inp, dropout_key):
    logits = downstream_model.apply(parameters, inp, tar_inp, train=False, rngs={'dropout': dropout_key})
    return logits

def val_step(parameters, inp, tar_inp, data_real, dropout_key):
    logits = apply_model(parameters, inp, tar_inp, dropout_key) 

    ## Then we compute the accuracy over here.
    if args.task == "wnli":
        return compute_wnli_accuracy(logits, data_real)
    else:
        return compute_binary_accuracy(logits, data_real)

## Over here, we have the main training loop. ##
EPOCHS = 4
for epoch in range(EPOCHS):
    loss_accum = 0

    for batch, inp in enumerate(train_data):
        ## We take one train_step. ##

        pdb.set_trace()
        ## We have to do all the pre-processing first. ##
        enc_part, dec_part = prepare_helper(inp)
        enc_part, dec_part, real_val, weights = prepare_transformer_input(enc_part, dec_part)

        ## Then we have to convert to jax arrays.
        enc_part = jnp.array(enc_part.numpy())
        dec_part = jnp.array(dec_part.numpy())
        real_val = jnp.array(real_val.numpy())
        weights = jnp.array(weights)
        ## Refresh the dropout_key as well.## 
        dropout_key = random.split(dropout_key)[1]

        ## Finally, call one train_step. ##
        params, opt_state, loss = train_step(params, enc_part, dec_part, real_val, True, weights, dropout_key, opt_state, optimizer, batch)
        loss_accum += loss
        print(f'Epoch: {epoch + 1} Batch: {batch+1} Loss: {loss_accum/float(batch+1):.3f}', flush=True)

    ## Then we validate. ##
    total_val_accuracy = 0
    num_batches = 0
    for batch, (inputs, labels) in enumerate(val_data):
        
        ## We have to do all the pre-processing first. ##
        enc_part, dec_part = prepare_helper(inp)
        enc_part, dec_part, real_val, weights = prepare_transformer_input(enc_part, dec_part)

        ## Then we have to convert to jax arrays.
        enc_part = jnp.array(enc_part.numpy())
        dec_part = jnp.array(dec_part.numpy())
        real_val = jnp.array(real_val.numpy())
        weights = jnp.array(weights)
        ## Refresh the dropout_key as well.## 
        dropout_key = random.split(dropout_key)[1]

        ## Lastly, we call the validation function. ##
        accuracy = val_step(params, enc_part, dec_part, real_val, dropout_key)
        total_val_accuracy += accuracy 
        num_batches += 1
        

    total_val_accuracy /= float(num_batches)
    print(f'Epoch {epoch + 1} Validation-Accuracy: {total_val_accuracy:.3f}', flush=True)
    with open(f'{args.attention_type}_val_data_glue.txt', 'a+') as f:
        f.write(f'{total_val_accuracy:.3f} {epoch+1}\n')
