"""
Here, we have a series of benchmarks that investigate how good 
einsum in tensorflow for the GPU backend really is.
"""
import tensorflow as tf
import time

## Just some helper methods that we will use later.
def create_rng_mat(shape):
    return tf.random.uniform(shape=shape, maxval=1000, dtype=tf.int64)

## Determines if two tensors are elementwise equal.
def is_equal(one, two):
    if len(one.shape) != len(two.shape):
        return False

    for a,b in zip(one.shape, two.shape):
        if a != b:
            return False

    equality = tf.math.equal(one, two)
    equality = tf.cast(equality, dtype=tf.int64)
    return tf.reduce_sum(equality) == tf.cast(tf.math.reduce_prod(tf.convert_to_tensor(one.shape)), dtype=tf.int64)

## Over here, qs, ks and vs should be 4-d tensors.
def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs) ## Why on earth is there an outer product?
  return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)

def noncausal_numerator_matmul(qs, ks, vs):
    ## Equivalent of: tf.einsum('lbhm, lbhd -> bhmd', ks, vs). 
    kvs = tf.matmul(tf.expand_dims(ks, axis=-1), tf.expand_dims(vs, axis=-1), transpose_b=True)
    kvs = tf.reduce_sum(kvs, axis=0)
    # Equivalent of: tf.einsum('lbhm, bhmd -> lbhd', qs, kvs).
    ## This is buggy, come back to this. TODO.
    tf.tensordot(qs, kvs, axes=((3), (2)))

## Here, we compare the linear transformation in matmul form and einsum form.
@tf.function
def lin_matmul(xs, ws):
    ## Equivalent of: tf.einsum('bsd, dhk -> bshk', xs, ws)
    return tf.tensordot(xs, ws, axes=((2), (0)))

## xs should be [batch_size, seq_length, hidden_dimension]
## ws should be [hidden_dimension num_heads, head_dimension]
## resultant tensor is: [batch_size, seq_length, heads, head_dimension]
@tf.function
def lin_einsum(xs, ws):
    return tf.einsum('bsd, dhk -> bshk', xs, ws)

## We test out multiplying three matrices.

## Next, we see the effect of baking in vs. two separate einsums.
## xs -> inputs, ws -> weights for lin trfm, ds -> downsampling matrix. 
## xs -> [batch_size, sequence_length, hidden dimension], ws -> [hidden dimension, num_heads, head_dimension]
## ds -> [downsampling_factor, sequence_length].
@tf.function
def non_baked_matmul_einsum(xs, ws, ds):
    a = tf.einsum('ds, bsh -> bdh', ds, xs) ## First we downsample.
    return lin_einsum(a, ws) ## Then we apply the linear transformation.

@tf.function
def baked_matmul_einsum(xs,ws,ds):
    return tf.einsum('ds, bsh, hnf -> bdnf', xs, ws, ds)

@tf.function
def non_baked_three_matmul(xs,ys,zs):
    a = tf.matmul(xs, ys)
    return tf.tensordot(a, zs, axes=((2), (0)))

xs = create_rng_mat((32,14000,1024))
ws = create_rng_mat((xs.shape[-1],8,128))
ds = create_rng_mat((16,xs.shape[1]))

def baking_matmul_exp():
    a = time.time()
    for _ in range(100):
        non_baked_three_matmul(ds, xs, ws)
    b = time.time()

    c = time.time()
    for _ in range(100):
        baked_matmul_einsum(ds, xs, ws)
    d = time.time()

    with open("perf_benchmark.txt", "a+") as f:
        f.write(f'Non_baked_three_matmul: {b-a} baked_matmul_einsum: {d-c}\n')

baking_matmul_exp()

## simple test cases. ##
## l = 2, b = 3, h = 4, m = 5, d = 5
#qs = create_rng_mat((2,3,4,5))
#ks = create_rng_mat((2,3,4,5))
#vs = create_rng_mat((2,3,4,5))
#a = noncausal_numerator(qs, ks, vs)
#b = noncausal_numerator_matmul(qs, ks, vs) 
#print(is_equal(a, b))
#print(a.shape)
#print(b.shape)