import sys
import os

sys.path.append(os.getcwd())

import flax.linen as nn

from jax_mha import MHA as VanillaMHA
from jax_sonic_mha import MHA as CompMHA
from jax_lin_mha import MHA as LinMHA
from jax_performer_mha import MHA as PerfMHA
import pdb
import math
import jax.numpy as jnp
import jax

class PositionalEmbedding(nn.Module):
    vocabulary_size : int
    embedding_dim : int
    dropout : float
    
    def setup(self):
        ## We initialize a normal embedding layer.
        self.embedding_layer = nn.Embed(self.vocabulary_size, self.embedding_dim)
        self.dropout_embedding = nn.Dropout(rate=self.dropout)

    def __call__(self, x, *, train): # Is is a tensor of shape: [batch_size, sequence_length].
        assert self.embedding_dim % 2 == 0, "Embedding Dimension should be divisible by two!"

        x = self.embedding_layer(x)
        ## Next, we have to compute a vector of positional embeddings to add.
        pos_embed = jnp.zeros((x.shape[1], self.embedding_dim))
        position = jnp.arange(0, x.shape[1], dtype=jnp.float32)
        div_term = jnp.exp(jnp.arange(0, self.embedding_dim, 2) * (-math.log(10000.0) / self.embedding_dim))
        #pos_embed.at[:, 0::2].set(jnp.sin(jnp.einsum('a,b -> ab', position, div_term)))
        pos_embed = pos_embed.at[:, 0::2].set(jnp.sin(jnp.einsum('a,b -> ab', position, div_term)))
        pos_embed = pos_embed.at[:, 1::2].set(jnp.cos(jnp.einsum('a,b -> ab', position, div_term)))

        ## Finally, we return the dropped out summed result.
        return self.dropout_embedding(x + pos_embed, deterministic=not train)

class EncoderLayer(nn.Module):
    hidden_dim: int
    head_dim : int
    num_heads : int
    dropout : float
    sequence_length : int
    ffn_size : int
    downsampling_k : int = 64
    attention_type : str = "MHA"

    def setup(self):
        ## We first have the pre-ambulatory initialization.
        if self.attention_type == "PerfMHA":
            self.mha = PerfMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                dropout=self.dropout, mask=False)
        elif self.attention_type == "LinMHA":
            self.mha = LinMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                dropout=self.dropout, mask=False, sequence_length=self.sequence_length, downsampling_k=self.downsampling_k)
        elif self.attention_type == "CompMHA":
            self.mha = CompMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                dropout=self.dropout, mask=False, sequence_length=self.sequence_length, downsampling_k=self.downsampling_k)
        else: ## We default to vanilla attention.
            self.mha = VanillaMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                dropout=self.dropout, mask=False)

        self.dense_expand = nn.Dense(self.ffn_size)
        self.dense_contract = nn.Dense(self.hidden_dim)

        ## We need two layer_norm layers.
        self.layer_norm_one = nn.LayerNorm()
        self.layer_norm_two = nn.LayerNorm()

        ## We need three dropout layers.
        self.dropout_one = nn.Dropout(self.dropout)
        self.dropout_two = nn.Dropout(self.dropout)
        self.dropout_three = nn.Dropout(self.dropout)

    def __call__(self, x, *, train):
        queries, keys, values = x, x, x

        ## we first compute the attention value.
        attn = self.mha([queries, keys, values], train=train)
        ## We drop out the values of attn.
        attn = self.dropout_one(attn, deterministic=not train)

        ## Then we have to compute the layer norm of the addition.
        attn_prev_ffn = self.layer_norm_one(attn + x)

        ## Then we have to put it through a dense ffn Layer.
        attn = self.dense_expand(attn_prev_ffn)
        attn = self.dropout_two(attn, deterministic=not train)
        attn = self.dense_contract(attn)
        attn = self.dropout_three(attn, deterministic=not train)

        ## Then we have to put it through the last layer-norm.
        attn = self.layer_norm_two(attn + attn_prev_ffn)

        ## Then this is our final value.
        return attn

class Encoder(nn.Module):
    hidden_dim: int
    head_dim : int
    num_heads : int
    dropout : float
    sequence_length : int
    ffn_size : int
    encoder_layers : int
    downsampling_k : int = 64
    attention_type : str = "MHA"

    def setup(self):
        self.encoders = [EncoderLayer(self.hidden_dim, self.head_dim, 
                                        self.num_heads, self.dropout, 
                                        self.sequence_length, self.ffn_size, 
                                        self.downsampling_k, self.attention_type) for _ in range(self.encoder_layers)]

    def __call__(self, x, *, train):
        for enc in self.encoders:
            x = enc(x, train=train)

        return x

class DecoderLayer(nn.Module):
    hidden_dim: int
    head_dim : int
    num_heads : int
    dropout : float
    sequence_length : int
    ffn_size : int
    downsampling_k : int = 64
    attention_type : str = "MHA"

    def setup(self):
        if self.attention_type == "PerfMHA":
            self.masked_mha = PerfMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                        dropout=self.dropout, mask=True)
                
            self.enc_dec_mha = PerfMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                            dropout=self.dropout, mask=False)
        elif self.attention_type == "LinMHA":
            self.masked_mha = LinMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                        dropout=self.dropout, mask=True, sequence_length=self.sequence_length, 
                                        downsampling_k=self.downsampling_k)
                
            self.enc_dec_mha = LinMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                            dropout=self.dropout, mask=False, sequence_length=self.sequence_length,
                                            downsampling_k=self.downsampling_k)
        elif self.attention_type == "CompMHA":
            self.masked_mha = CompMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                        dropout=self.dropout, mask=True, sequence_length=self.sequence_length, 
                                        downsampling_k=self.downsampling_k)
                
            self.enc_dec_mha = CompMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                            dropout=self.dropout, mask=False, sequence_length=self.sequence_length,
                                            downsampling_k=self.downsampling_k)
        else: ## Default to Vanilla Attention.
            self.masked_mha = VanillaMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                        dropout=self.dropout, mask=True)
                
            self.enc_dec_mha = VanillaMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads, 
                                            dropout=self.dropout, mask=False)

        self.dense_expand = nn.Dense(self.ffn_size)
        self.dense_contract = nn.Dense(self.hidden_dim)

        self.dropout_one = nn.Dropout(rate=self.dropout)
        self.dropout_two = nn.Dropout(rate=self.dropout)
        self.dropout_three = nn.Dropout(rate=self.dropout)
        self.dropout_four = nn.Dropout(rate=self.dropout)

        ## Three layernorms required
        self.layer_norm_one = nn.LayerNorm()
        self.layer_norm_two = nn.LayerNorm()
        self.layer_norm_three = nn.LayerNorm()

    ## Note here, x is a list of: [Encoder Input, Decoder Input]
    def __call__(self, x, *, train):
        encoder_input, decoder_input = x

        ## This is the Masked Attention.
        attn_output_masked = self.masked_mha([decoder_input, decoder_input, decoder_input], train=train)
        attn_output_masked = self.dropout_one(attn_output_masked, deterministic=not train)
        attn_output_masked = self.layer_norm_one(decoder_input + attn_output_masked)

        ## This is the Enc-Dec Attention.
        attn_output = self.enc_dec_mha([attn_output_masked, encoder_input, encoder_input], train=train)
        attn_output = self.dropout_two(attn_output, deterministic=not train)
        attn_output = self.layer_norm_two(attn_output + attn_output_masked)

        ## Lastly, the feed forward network.
        attn_ffn = self.dense_expand(attn_output)
        attn_ffn = self.dropout_three(attn_ffn, deterministic=not train)
        attn_ffn = self.dense_contract(attn_ffn)
        attn_ffn = self.dropout_four(attn_ffn, deterministic=not train)
        attn_ffn = self.layer_norm_three(attn_ffn + attn_output)

        return attn_ffn

class Decoder(nn.Module):
    hidden_dim: int
    head_dim : int
    num_heads : int
    dropout : float
    sequence_length : int
    ffn_size : int
    decoder_layers : int
    downsampling_k : int = 64
    attention_type : str = "MHA"

    def setup(self):
        self.decoders = [DecoderLayer(self.hidden_dim, self.head_dim, self.num_heads, 
                                        self.dropout, self.sequence_length, self.ffn_size, self.downsampling_k, self.attention_type) for _ in range(self.decoder_layers)]

    ## x should be a list: [encoder_input, decoder_input]
    def __call__(self, x, *, train):
        encoder_input, decoder_input = x

        for dec in self.decoders:
            decoder_input = dec([encoder_input, decoder_input], train=train)

        return decoder_input

class Transformer(nn.Module):
    hidden_dim: int
    head_dim : int
    num_heads : int
    dropout : float
    sequence_length : int
    ffn_size : int
    num_layers : int
    vocabulary_size : int
    encoder_only : bool
    downsampling_k : int = 64 ## Default this to 64.
    attention_type : str = "MHA" ## Default this to vanilla attention.

    def setup(self):
        self.positional_embedding = PositionalEmbedding(self.vocabulary_size, self.hidden_dim, self.dropout)

        self.encoder = Encoder(self.hidden_dim, self.head_dim, self.num_heads, 
                               self.dropout, self.sequence_length, self.ffn_size, 
                               self.num_layers, self.downsampling_k, self.attention_type)

        self.decoder = Decoder(self.hidden_dim, self.head_dim, self.num_heads, 
                               self.dropout, self.sequence_length, self.ffn_size, 
                               self.num_layers, self.downsampling_k, self.attention_type)

        self.last_ffn = nn.Dense(self.vocabulary_size)

    def __call__(self, encoder_input, decoder_input, *, train):
        if self.encoder_only:
            ## Over here, x is one input.
            encoder_input = self.positional_embedding(encoder_input, train=train)
            encoder_output = self.encoder(encoder_input, train=train)
            return encoder_output 
        else:
            ## Over here, x is a list: [encoder_input, decoder_input]
            encoder_input = self.positional_embedding(encoder_input, train=train)
            decoder_input = self.positional_embedding(decoder_input, train=train)
            encoder_output = self.encoder(encoder_input, train=train)
            decoder_output = self.decoder([encoder_output, decoder_input], train=train)
            return decoder_output 

## This section is for unit testing all my code.
import jax.numpy as jnp
from jax import random

hidden_dim = 8
head_dim = 4
num_heads = 2 
dropout = 0.1
sequence_length = 4
ffn_size = 10
num_layers = 2
vocabulary_size = 10
attnetion_type = "MHA"
downsampling_k = 3
batch_size = 2

transformer = Transformer(hidden_dim, head_dim, num_heads, 0.1, sequence_length, ffn_size, num_layers, vocabulary_size, True, downsampling_k, attnetion_type)
param_key = random.PRNGKey(42)
dropout_key = random.PRNGKey(43)
enc_input = jnp.round(random.uniform(random.PRNGKey(44), (batch_size, sequence_length)) * vocabulary_size).astype(jnp.int32)
dec_input = jnp.round(random.uniform(random.PRNGKey(45), (batch_size, sequence_length)) * vocabulary_size).astype(jnp.int32)
params = transformer.init({'params': param_key, 'dropout': dropout_key}, enc_input, dec_input, train=True)

## One thing to keep note is that a new dropout_key must constantly be passed into the function.
last_dropout_key = dropout_key
for _ in range(2):
    attention_mat = transformer.apply(params, enc_input, dec_input, train=True, rngs={'dropout': dropout_key})
    print(attention_mat)
    last_dropout_key = random.split(last_dropout_key)[1]