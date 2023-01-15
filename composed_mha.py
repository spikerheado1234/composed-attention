# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of multiheaded FAVOR-attention & FAVOR-self-attention layers.
Prefix Sum Tensorflow implementation by Valerii Likhosherstov.
"""
import math
import numpy as np
import tensorflow as tf
import fast_attention.util as util 
from keras import constraints
from keras import initializers
from keras import regularizers
from stats import Stats 
import string
import time
from local_einsumdense import EinsumDense as ED

BIG_CONSTANT = 1e8

_CHR_IDX = string.ascii_lowercase

def _downsample_mat(mat, rand_mat):
    """
    The shape of mat is: [B, S, N, H] -> batch size, seq length, num heads, head dimension.
    This function down-samples mat using the random matrix rand_mat as described in the 
    Linformer paper.

    Rand_mat should be of size: [K, S] -> down-sample size, sequence length.

    Returns a down-sampled matrix of size: [B, K, N, H].
    """

    #output = tf.einsum('ks, bsnh -> bknh', rand_mat, mat) # Output is of shape: [B, N, K, H] after completing this step.
    output = tf.einsum('ks, bsd -> bkd', rand_mat, mat) # Output is of shape: [B, N, K, H] after completing this step.
    # We then have to return back to the normal shape of: [B, K, N, H]

    return output

def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)

def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = f"{input_str},{kernel_str}->{output_str}"

    return equation, bias_axes, len(output_str)

def _downsampling_shape_correct(mat_shape, rand_mat_shape):
    """
    Checks if rand_mat's shape is fit to downsample mat_shape.
    Since, _build_from_signature does not have access to the keys 
    we must check this to be sure.

    mat_shape -> (Batch Size, Sequence Length, Num Heads, Inner Dimension)
    Rand_mat -> (Down Sample Size, Sequence Length)
    """
    return mat_shape[1] == rand_mat_shape[1] and rand_mat_shape[0] < mat_shape[1] 

def _build_downsample_proj(k, 
                           shape,
                           seed=1):
    """
    Builds a randomly generated matrix of dimension: num_heads x k x sequence_length.
    This is a non-trainable matrix.
    k -> dimension of the space you would like to down-sample to.
    shape -> the shape of the randomly generated matrix. Must be a tuple.
    """
    return tf.Variable(tf.random.normal(shape=shape, mean=0.0, stddev=1/k, seed=seed, dtype=tf.float32), trainable=False, dtype=tf.float32)


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
  r"""Constructs the matrix of random projections.
  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random and either deterministic length
  \sqrt{d} or length taken from the \chi(d) distribution (in the latter case
  marginal distributions of the projections are d-dimensional Gaussian vectors
  with associated identity covariance matrix).
  Args:
    m: number of random projections.
    d: dimensionality of each random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{d}, 0 if the lengths of random projections should follow
      \chi(d) distribution.
    struct_mode: if True then products of Givens rotations will be used to
      construct random orthogonal matrix. This bypasses Gram-Schmidt
      orthogonalization.
  Returns:
    The matrix of random projections of the shape [m, d].
  """
  nb_full_blocks = int(m / d)
  block_list = []
  current_seed = seed
  for _ in range(nb_full_blocks):
    if struct_mode:
      q = create_products_of_givens_rotations(d, seed)
    else:
      unstructured_block = tf.random.normal((d, d), seed=current_seed)
      q, _ = tf.linalg.qr(unstructured_block)
      q = tf.transpose(q)
    block_list.append(q)
    current_seed += 1
  remaining_rows = m - nb_full_blocks * d
  if remaining_rows > 0:
    if struct_mode:
      q = create_products_of_givens_rotations(d, seed)
    else:
      unstructured_block = tf.random.normal((d, d), seed=current_seed)
      q, _ = tf.linalg.qr(unstructured_block)
      q = tf.transpose(q)
    block_list.append(q[0:remaining_rows])
  final_matrix = tf.experimental.numpy.vstack(block_list)
  current_seed += 1

  if scaling == 0:
    multiplier = tf.norm(tf.random.normal((m, d), seed=current_seed), axis=1)
  elif scaling == 1:
    multiplier = tf.math.sqrt(float(d)) * tf.ones((m))
  else:
    raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

  return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)


def create_products_of_givens_rotations(dim, seed):
  r"""Constructs a 2D-tensor which is a product of Givens random rotations.
  Constructs a 2D-tensor of the form G_1 * ... * G_k, where G_i is a Givens
  random rotation. The resulting tensor mimics a matrix taken uniformly at
  random form the orthogonal group.
  Args:
    dim: number of rows/columns of the resulting 2D-tensor.
    seed: random seed.
  Returns:
    The product of Givens random rotations.
  """
  nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
  q = np.eye(dim, dim)
  np.random.seed(seed)
  for _ in range(nb_givens_rotations):
    random_angle = math.pi * np.random.uniform()
    random_indices = np.random.choice(dim, 2)
    index_i = min(random_indices[0], random_indices[1])
    index_j = max(random_indices[0], random_indices[1])
    slice_i = q[index_i]
    slice_j = q[index_j]
    new_slice_i = math.cos(random_angle) * slice_i + math.sin(
        random_angle) * slice_j
    new_slice_j = -math.sin(random_angle) * slice_i + math.cos(
        random_angle) * slice_j
    q[index_i] = new_slice_i
    q[index_j] = new_slice_j
  return tf.cast(tf.constant(q), dtype=tf.float32)


def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
  """Computes features for the ReLU-kernel.
  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.
  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
  Returns:
    Corresponding kernel feature map.
  """
  del is_query
  if projection_matrix is None:
    return tf.nn.relu(data) + numerical_stabilizer
  else:
    ratio = 1.0 / tf.math.sqrt(
        tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
    data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
    return tf.nn.relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
  """Computes random features for the softmax kernel using FAVOR+ mechanism.
  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.
  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
  Returns:
    Corresponding kernel feature map.
  """
  data_normalizer = 1.0 / (
      tf.math.sqrt(tf.math.sqrt(tf.dtypes.cast(data.shape[-1], tf.float32))))
  data = data_normalizer * data
  ratio = 1.0 / tf.math.sqrt(
      tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
  data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
  diag_data = tf.math.square(data)
  diag_data = tf.math.reduce_sum(
      diag_data, axis=tf.keras.backend.ndim(data) - 1)
  diag_data = diag_data / 2.0
  diag_data = tf.expand_dims(diag_data, axis=tf.keras.backend.ndim(data) - 1)
  last_dims_t = (len(data_dash.shape) - 1,)
  attention_dims_t = (len(data_dash.shape) - 3,)
  if is_query:
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t, keepdims=True)) + numerical_stabilizer)
  else:
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t + attention_dims_t, keepdims=True)) +
        numerical_stabilizer)

  return data_dash


def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = tf.einsum('bshd, bshe -> bhde', ks, vs) 
  return tf.einsum('bshd, bhde -> bshe', qs, kvs)


def noncausal_denominator(qs, ks):
  """Computes FAVOR normalizer in noncausal attention.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
  Returns:
    FAVOR normalizer in noncausal attention.
  """
  ks_sum = tf.einsum("blhm -> bhm", ks)
  return tf.einsum("blhm,bhm->blh", qs, ks_sum)

@tf.custom_gradient
def causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """

  result = []
  sums = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

  for index in range(qs.shape[0]):
    sums = sums + tf.einsum("ijk,ijl->ijkl", ks[index], vs[index]) 
    result.append(tf.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    grads = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    gr_sums = sums

    q_grads = []
    k_grads = []
    v_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijkl,ijl->ijk", gr_sums, res_grad[index])[None, Ellipsis])
      grads = grads + tf.einsum("ijk,ijl->ijkl", qs[index], res_grad[index])
      k_grads.append(tf.einsum("ijkl,ijl->ijk", grads, vs[index])[None, Ellipsis])
      v_grads.append(tf.einsum("ijkl,ijk->ijl", grads, ks[index])[None, Ellipsis])
      gr_sums = gr_sums - tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)
    v_grads = tf.concat(v_grads[::-1], axis=0)

    return q_grads, k_grads, v_grads

  return result, grad


@tf.custom_gradient
def causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
  Returns:
    FAVOR normalizer in causal attention.
  """

  result = []
  sums = tf.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(tf.reduce_sum(qs[index] * sums, axis=2)[None, Ellipsis])

  result = tf.concat(result, axis=0)

  def grad(res_grad):

    k_grad = tf.zeros_like(ks[0])

    gr_sums = sums

    q_grads = []
    k_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          tf.einsum("ijk,ij->ijk", gr_sums, res_grad[index])[None, Ellipsis])
      k_grad = k_grad + tf.einsum("ijk,ij->ijk", qs[index], res_grad[index])
      k_grads.append(k_grad[None, Ellipsis])
      gr_sums = gr_sums - ks[index]

    q_grads = tf.concat(q_grads[::-1], axis=0)
    k_grads = tf.concat(k_grads[::-1], axis=0)

    return q_grads, k_grads

  return result, grad


def favor_attention(query,
                    key,
                    value,
                    kernel_transformation,
                    causal,
                    projection_matrix=None):
  """Computes FAVOR normalized attention.
  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    kernel_transformation: transformation used to get finite kernel features.
    causal: whether attention is causal or not.
    projection_matrix: projection matrix to be used.
  Returns:
    FAVOR normalized attention.
  """
  transformation_start = time.time()
  query_prime = kernel_transformation(query, True,
                                      projection_matrix)  # [B,L,H,M]
  key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
  transformation_end = time.time()
  Stats.transformation_time += (transformation_end - transformation_start)

  transpose_time_start = time.time()
  transpose_time_end = time.time()
  Stats.transpose_time += transpose_time_end - transpose_time_start

  attn_product_start = time.time()
  if causal:
    av_attention = causal_numerator(query_prime, key_prime, value)
    attention_normalizer = causal_denominator(query_prime, key_prime)
  else:
    attention_normalizer = noncausal_denominator(query_prime, key_prime)
    av_attention = noncausal_numerator(query_prime, key_prime, value)
  attn_product_end = time.time()
  Stats.q_k_v_product += (attn_product_end - attn_product_start)
  # TODO(kchoro): Add more comments.
  transpose_time_start = time.time()
  transpose_time_end = time.time()
  Stats.transpose_time += transpose_time_end - transpose_time_start

  expand_dims_start = time.time()
  attention_normalizer = tf.expand_dims(attention_normalizer,
                                        len(attention_normalizer.shape))
  expand_dims_end = time.time()
  Stats.expand_dims_time += (expand_dims_end - expand_dims_start)
  return av_attention / attention_normalizer


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self,
               hidden_size,
               num_heads,
               attention_dropout,
               kernel_transformation=relu_kernel_transformation,
               numerical_stabilizer=0.001,
               causal=False,
               projection_matrix_type=None,
               downsample_k=32, # We default this value to 32. (Currently half the batch size.)
               nb_random_features=0,
               use_bias=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None):
    """Initialize Attention.
    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
      kernel_transformation: transformation used to produce kernel features for
        attention.
      numerical_stabilizer: used to bound away from zero kernel values.
      causal: whether attention is causal or not.
      projection_matrix_type: None if Identity should be used, otherwise random
        projection matrix will be applied.
      nb_random_features: number of random features to be used (relevant only if
        projection_matrix is not None).
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))
    
    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.kernel_transformation = kernel_transformation
    self.numerical_stabilizer = numerical_stabilizer
    self.causal = causal
    self.projection_matrix_type = projection_matrix_type
    self.nb_random_features = nb_random_features
    self._built_proj_mat = False
    self._downsample_k = downsample_k
    self.use_bias = use_bias
    self._kernel_initializer = initializers.get(kernel_initializer)
    self._bias_initializer = initializers.get(bias_initializer)
    self._kernel_regularizer = regularizers.get(kernel_regularizer)
    self._bias_regularizer = regularizers.get(bias_regularizer)
    self._activity_regularizer = regularizers.get(activity_regularizer)
    self._kernel_constraint = constraints.get(kernel_constraint)
    self._bias_constraint = constraints.get(bias_constraint)

  def build(self, input_shape):
    """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
    size_per_head = self.hidden_size // self.num_heads
    
    def _glorot_initializer(fan_in, fan_out):
      limit = math.sqrt(6.0 / (fan_in + fan_out))
      return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

    if hasattr(input_shape, "shape"):
      self._query_shape = tf.TensorShape(input_shape.shape) 
      self._key_shape = tf.TensorShape(input_shape.shape)
      self._value_shape = tf.TensorShape(input_shape.shape)
    else:
      self._query_shape = tf.TensorShape(input_shape) 
      self._key_shape = tf.TensorShape(input_shape)
      self._value_shape = tf.TensorShape(input_shape)

    free_dims = self._query_shape.rank - 1
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=1, output_dims=2
    )
    self._query_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(
            output_rank - 1, [self.num_heads, self.hidden_size]
        ),
        bias_axes=bias_axes if self.use_bias else None,
        name="query",
        **self._get_common_kwargs_for_sublayer(),
    )
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        self._key_shape.rank - 1, bound_dims=1, output_dims=2
    )
    self._key_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(
            output_rank - 1, [self.num_heads, self.hidden_size]
        ),
        bias_axes=bias_axes if self.use_bias else None,
        name="key",
        **self._get_common_kwargs_for_sublayer(),
    )
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        self._value_shape.rank - 1, bound_dims=1, output_dims=2
    )
    self._value_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(
            output_rank - 1, [self.num_heads, self.hidden_size]
        ),
        bias_axes=bias_axes if self.use_bias else None,
        name="value",
        **self._get_common_kwargs_for_sublayer(),
    )

    output_initializer = _glorot_initializer(self.hidden_size, self.hidden_size)
    self.output_dense_layer = util.DenseEinsum(
        output_shape=self.hidden_size,
        num_summed_dimensions=2,
        kernel_initializer=output_initializer,
        use_bias=False,
        name="output_transform")
    super(Attention, self).build(input_shape)

  def _get_common_kwargs_for_sublayer(self):
      common_kwargs = dict(
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activity_regularizer=self._activity_regularizer,
          kernel_constraint=self._kernel_constraint,
          bias_constraint=self._bias_constraint,
      )
      # Create new clone of kernel/bias initializer, so that we don't reuse
      # the initializer instance, which could lead to same init value since
      # initializer is stateless.
      kernel_initializer = self._kernel_initializer.__class__.from_config(
          self._kernel_initializer.get_config()
      )
      bias_initializer = self._bias_initializer.__class__.from_config(
          self._bias_initializer.get_config()
      )
      common_kwargs["kernel_initializer"] = kernel_initializer
      common_kwargs["bias_initializer"] = bias_initializer
      return common_kwargs
 
  def get_config(self):
    config = {
        "num_heads": self._num_heads,
        "key_dim": self._key_dim,
        "value_dim": self._value_dim,
        "dropout": self._dropout,
        "use_bias": self._use_bias,
        "output_shape": self._output_shape,
        "attention_axes": self._attention_axes,
        "kernel_initializer": initializers.serialize(
            self._kernel_initializer
        ),
        "bias_initializer": initializers.serialize(self._bias_initializer),
        "kernel_regularizer": regularizers.serialize(
            self._kernel_regularizer
        ),
        "bias_regularizer": regularizers.serialize(self._bias_regularizer),
        "activity_regularizer": regularizers.serialize(
            self._activity_regularizer
        ),
        "kernel_constraint": constraints.serialize(self._kernel_constraint),
        "bias_constraint": constraints.serialize(self._bias_constraint),
        "query_shape": self._query_shape,
        "key_shape": self._key_shape,
        "value_shape": self._value_shape,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           query,
           value,
           key,
           training,
           attention_mask=None, # This will not be used anyway. Put in here to make calls to Attention the same as MHA and LinMHA upstream.
           return_attention_scores=False, # This will also not be used, similar reasoning as above.
           use_causal_mask=True, # This will not be used, similar reasoning as above.
           cache=None,
           decode_loop_step=None):
    """Apply attention mechanism to query_input and source_input.
    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]} where
               i is the current decoded length for non-padded decode, or max
               sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.
    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
    ## We build the projection matrices if we have not already.
    downsample_trfr_start = time.time()
    if not self._built_proj_mat:
        self._rand_mat_keys = _build_downsample_proj(self._downsample_k, (self._downsample_k, key.shape[1]))
        self._rand_mat_values = _build_downsample_proj(self._downsample_k, (self._downsample_k, value.shape[1]))
        self._built_proj_mat = True
    downsample_mat_gen_end = time.time()
    Stats.downsampling_mat_gen += (downsample_mat_gen_end - downsample_trfr_start)

    downsample_mat_mul_start = time.time()
    ## We then transform the keys and values accordingly.
    key = _downsample_mat(key, self._rand_mat_keys)
    value = _downsample_mat(value, self._rand_mat_values)
    downsample_trfr_end = time.time()
    Stats.downsampling_time += (downsample_trfr_end - downsample_trfr_start)
    Stats.downsampling_mat_mul += (downsample_trfr_end - downsample_mat_mul_start) 

    # Original dimension of Q, K and V are -> [batch_size, seq_length, hidden_dimension]
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
    # `query` = [B, T, N ,H]
    lin_trnfrm_start = time.time()
    query = self._query_dense(query)

    # `key` = [B, S, N, H]
    key = self._key_dense(key)

    # `value` = [B, S, N, H]
    value = self._value_dense(value)
    lin_trnfrm_end = time.time()
    Stats.linear_transformation += (lin_trnfrm_end - lin_trnfrm_start)

    if self.projection_matrix_type is None:
      projection_matrix = None
    else:
      dim = query.shape[-1]
      seed = tf.math.ceil(tf.math.abs(tf.math.reduce_sum(query) * BIG_CONSTANT))
      seed = tf.dtypes.cast(seed, tf.int32)
      projection_matrix = create_projection_matrix(
          self.nb_random_features, dim, seed=seed)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        key = cache["k"] + key * indices
        cache_v_shape = cache["v"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        value = cache["v"] + value * indices
      else:
        key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
        value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

      # Update cache
      cache["k"] = key
      cache["v"] = value

    favour_start = time.time()
    attention_output = favor_attention(query, key, value,
                                       self.kernel_transformation, self.causal,
                                       projection_matrix)
    favour_end = time.time()
    Stats.favour_time += (favour_end - favour_start)
    local_ffn_start = time.time()
    attention_output = self.output_dense_layer(attention_output)
    local_ffn_end = time.time()
    Stats.ffn_time += (local_ffn_end - local_ffn_start)
    Stats.mha_ffn += (local_ffn_end - local_ffn_start)
    if return_attention_scores:
      return attention_output, None # We return phony attention weights since it is never explicitly computed in the PerFormer.

    return attention_output 


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self,
           query_input,
           bias,
           training,
           cache=None,
           decode_loop_step=None):
    return super(SelfAttention, self).call(query_input, query_input, bias,
                                           training, cache, decode_loop_step)
