import sys
import os

sys.path.append(os.getcwd())

import jax
from jax import value_and_grad, grad, jit
import jax.numpy as jnp
from jax import random
from functools import partial
import argparse
import flax.linen as nn
from flax.training import checkpoints, train_state
from transformer_skeleton import Transformer
from pre_train_wiki_loader import get_train_ds, get_val_ds, make_batches
from tokenization_proc import mask
from constants import Constants
import optax
import pdb

## Custom command line arguments over here. ##
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
d_model = 768
dff = 3072
num_attention_heads = 12
dropout_rate = 0.1
rank = args.rank
learning_rate = args.lr_rate

assert d_model % num_attention_heads == 0, "Hidden dimension needs to be a multiple of the number of attention heads!"

class MaskedLM(nn.Module):
    transformer : Transformer 
    vocabulary_size : int

    def setup(self):
        self.last_ffn = nn.Dense(self.vocabulary_size)
    
    def __call__(self, encoder_input, decoder_input, *, train):
        output = self.transformer(encoder_input, decoder_input, train=train)
        output = self.last_ffn(output)
        return output

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

transformer = Transformer(d_model, int(d_model / num_attention_heads), num_attention_heads, 
                            dropout_rate, args.sequence_length, dff, num_layers, Constants.wiki_vocab_size, args.enc_only)

## We have to first initialize the model & optimizer. ##
masked_lm = MaskedLM(transformer, Constants.wiki_vocab_size)
param_key = random.PRNGKey(42)
dropout_key = random.PRNGKey(43)
enc_input = jnp.round(random.uniform(random.PRNGKey(44), (args.batch_size, args.sequence_length)) * Constants.wiki_vocab_size).astype(jnp.int32)
dec_input = jnp.round(random.uniform(random.PRNGKey(45), (args.batch_size, args.sequence_length)) * Constants.wiki_vocab_size).astype(jnp.int32)
params = masked_lm.init({'params': param_key, 'dropout': dropout_key}, enc_input, dec_input, train=True)
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params) ## TODO, check for correctness over here.

## We prepare the checkpoint information. ##
checkpoint_path = './checkpoints/train/' + str(args.attention_type) + '/' + str(learning_rate) + '/' + str(args.warmup)
ckpt_count = 0
pdb.set_trace()
## We attempt to restore the latest checkpoint. ##
if checkpoints.latest_checkpoint(ckpt_dir=checkpoint_path):
    checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=params['params']['transformer'])
    print('Latest checkpoint restored!')
else: ## Otherwise we create a checkpoint. ##
    state = train_state.TrainState.create(apply_fn=transformer.apply, params=params['params']['transformer'], tx=optimizer)
    checkpoints.save_checkpoint(ckpt_dir=checkpoint_path, target=state, step=ckpt_count)
    print('checkpoint succesfully created.')

def train_step(parameters, inp, tar_inp, data_real, train, weights, dropout_key, opt_state, optimizer):
    @partial(jax.jit, static_argnames=['train'])
    def compute_loss(parameters, inp, tar_inp, train, dropout_key, real, weights):
        logits = masked_lm.apply(parameters, inp, tar_inp, train=train, rngs={'dropout': dropout_key})

        ## TODO, is this the correct way to implement MLM? How can I know if this is correct?
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, real)
        loss *= weights
        return loss.sum() / jnp.sum(weights) 
    
    loss, grads = value_and_grad(compute_loss)(parameters, inp, tar_inp, train, dropout_key, data_real, weights)
    updates, opt_state = optimizer.update(grads, opt_state, parameters)
    params = optax.apply_updates(parameters, updates)

    ## Returns the tuple of: (new model parameters, optimizer state, loss).
    return params, opt_state, loss

def val_step(parameters, inp, tar_inp, data_real, train, weights, dropout_key):
    @partial(jax.jit, static_argnames=['train'])
    def compute_loss(parameters, inp, tar_inp, train, dropout_key, real, weights):
        logits = masked_lm.apply(parameters, inp, tar_inp, train=train, rngs={'dropout': dropout_key})

        ## TODO, is this the correct way to implement MLM? How can I know if this is correct?
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, real)
        loss *= weights
        return loss.sum() / jnp.sum(weights)      

    loss = compute_loss(parameters, inp, tar_inp, train, dropout_key, data_real, weights) 

    return loss

## Over here, we have the main training loop. ##
EPOCHS = 15

for epoch in range(EPOCHS):

    for batch, (inputs, labels) in enumerate(train_batches):
        ## We take one train_step. ##

        ## We have to do all the pre-processing first. ##
        (inp, tar_inp) = inputs
        tar_real = labels
        (inp, tar_inp), tar_real, weight = mask_data(inp)
        ## Convert from tensors to jnp ndarrays.
        inp = jnp.array(inp.numpy())
        tar_inp = jnp.array(tar_inp.numpy())
        tar_real = jnp.array(tar_real.numpy())
        weight = jnp.array(weight)
        ## Refresh the dropout_key as well.## 
        dropout_key = random.split(dropout_key)[1]

        ## Finally, call one train_step. ##
        params, opt_state, loss = train_step(params, inp, tar_inp, tar_real, True, weight[:, 1:], dropout_key, opt_state, optimizer)

    ## Here we checkpoint our model.
    ckpt_count += 1
    checkpoints.save_checkpoint(ckpt_dir=checkpoint_path, target=params['params']['transformer'], step=ckpt_count)
    print('succesfully checkpointed.')

    ## Then we validate. ##
    for batch, (inputs, labels) in enumerate(train_batches):
        
        ## We pre-process. ##
        (inp, tar_inp) = inputs
        tar_real = labels
        (inp, tar_inp), tar_real, weight = mask_data(inp)
        ## Convert from tensors to jnp ndarrays.
        inp = jnp.array(inp.numpy())
        tar_inp = jnp.array(tar_inp.numpy())
        tar_real = jnp.array(tar_real.numpy())
        weight = jnp.array(weight)
        ## Refresh the dropout_key as well.## 
        dropout_key = random.split(dropout_key)[1]

        ## Lastly, we call the validation function. ##
        loss = val_step(params, inp, tar_inp, tar_real, True, weight[:, 1:], dropout_key)
