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
    a = tf.matmul(xs, ys) ## Downsampling
    return tf.tensordot(a, zs, axes=((2), (0))) ## Linear transformation.

xs = create_rng_mat((32,14000,1024))
ys = create_rng_mat(xs.shape)
zs = create_rng_mat(xs.shape)
ws = create_rng_mat((xs.shape[-1],8,128))
wy = create_rng_mat(ws.shape)
wz = create_rng_mat(ws.shape)
ds = create_rng_mat((16,xs.shape[1]))
dsv = create_rng_mat(ds.shape)
downsampling_value = ds.shape[0]
sequence_length = xs.shape[1]

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

def linear_trfm_exp():
    a = time.time()
    for _ in range(1000):
        lin_matmul(xs, ws)
    b = time.time()

    c = time.time()
    for _ in range(1000):
        lin_einsum(xs, ws)
    d = time.time()

    with open("perf_benchmark.txt", "a+") as f:
        f.write(f'matmul time: {b-a} einsum time: {d-c}\n')

def locality_exp_einsum(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
    @tf.function
    def locality(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
        ## We interleave the einsums for better data locality.
        ks = tf.einsum('ks, bsd -> bkd', ds_ks, ks)
        ks = tf.einsum('bsd, dnh -> bsnh', ks, ws_ks)
        vs = tf.einsum('ks, bsd -> bkd', ds_vs, vs)
        vs = tf.einsum('bsd, dnh -> bsnh', vs, ws_vs)
        qs = tf.einsum('bsd, dnh -> bsnh', qs, ws_qs)

    @tf.function
    def not_local(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
        ## First we do the einsums to downsample. 
        ks = tf.einsum('ks, bsd -> bkd', ds_ks, ks)
        vs = tf.einsum('ks, bsd -> bkd', ds_vs, vs)

        ## Then, we do the einsums to map to attn heads.
        qs = tf.einsum('bsd, dnh -> bsnh', qs, ws_qs)
        ks = tf.einsum('bsd, dnh -> bsnh', ks, ws_ks)
        vs = tf.einsum('bsd, dnh -> bsnh', vs, ws_vs)

    a = time.time()
    for _ in range(100):
        locality(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs)
    b = time.time()

    c = time.time()
    for _ in range(100):
        not_local(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs)
    d = time.time()

    with open("perf_benchmark.txt", "a+") as f:
        f.write(f'Locality: {b-a}  Non-locality: {d-c}\n')

def locality_exp_matmul(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
    @tf.function
    def locality(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
        ## We interleave the matmuls and tensordots for better data locality.
        ks = tf.matmul(ds_ks, ks)
        ks = tf.tensordot(ks, ws_ks, axes=((2), (0)))
        vs = tf.matmul(ds_vs, vs)
        vs = tf.tensordot(vs, ws_vs, axes=((2), (0)))
        qs = tf.tensordot(qs, ws_qs, axes=((2), (0)))

    @tf.function
    def not_local(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
        ## First we do the matmuls to downsample. 
        ks = tf.matmul(ds_ks, ks)
        vs = tf.matmul(ds_vs, vs)

        ## Then, we do the tensordots to map to attn heads.
        ks = tf.tensordot(ks, ws_ks, axes=((2), (0)))
        vs = tf.tensordot(vs, ws_vs, axes=((2), (0)))
        qs = tf.tensordot(qs, ws_qs, axes=((2), (0)))

    a = time.time()
    for _ in range(100):
        locality(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs)
    b = time.time()

    c = time.time()
    for _ in range(100):
        not_local(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs)
    d = time.time()

    with open("perf_benchmark.txt", "a+") as f:
        f.write(f'Locality: {b-a}  Non-locality: {d-c}\n')

def ginormous_einsum(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
    global downsampling_value, sequence_length
    @tf.function
    def big_einsum(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
        ## First we have to conacatenate all the qs, ks and vs
        ## First we downsample the ks and vs.
        inter_result = tf.concat([ks, vs], axis=1)
        downsampling_mats = tf.concat([ds_ks, ds_vs], axis=1)
        ## Seems like this tensor is too big. 
        downsampled_values = tf.einsum('ks, bsd -> bksd', downsampling_mats, inter_result)

        ## We then slice out and reduce_sum everything.
        ks, vs = downsampled_values[:, :, :sequence_length, :], downsampled_values[:, :, sequence_length:, :]
        ks = tf.reduce_sum(ks, axis=2)
        vs = tf.reduce_sum(vs, axis=2)
        return ks, vs

    @tf.function
    def little_einsum(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
        ks = tf.einsum('ks, bsd -> bkd', ds_ks, ks)
        vs = tf.einsum('ks, bsd -> bkd', ds_vs, vs)
        return ks, vs

    a,b = big_einsum(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs)
    c,d = little_einsum(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs)
    print(is_equal(a, c))
    print(is_equal(b, d))

## Compares the best matmul schedule with the best einsum schedule.
def matmul_einsum_schedule_exp(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):

    @tf.function
    def matmul_schedule(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
        ## First we do the matmuls to downsample. 
        ks = tf.matmul(ds_ks, ks)
        vs = tf.matmul(ds_vs, vs)

        ## Then, we do the tensordots to map to attn heads.
        ks = tf.tensordot(ks, ws_ks, axes=((2), (0)))
        vs = tf.tensordot(vs, ws_vs, axes=((2), (0)))
        qs = tf.tensordot(qs, ws_qs, axes=((2), (0)))

    @tf.function
    def einsum_schedule(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs):
        ## First we do the einsums to downsample. 
        ks = tf.einsum('ks, bsd -> bkd', ds_ks, ks)
        vs = tf.einsum('ks, bsd -> bkd', ds_vs, vs)

        ## Then, we do the einsums to map to attn heads.
        ks = tf.einsum('bsd, dnh -> bsnh', ks, ws_ks)
        vs = tf.einsum('bsd, dnh -> bsnh', vs, ws_vs)
        qs = tf.einsum('bsd, dnh -> bsnh', qs, ws_qs)

    a = time.time()
    for _ in range(100):
        matmul_schedule(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs)
    b = time.time()

    c = time.time()
    for _ in range(100):
        einsum_schedule(qs, ks, vs, ds_ks, ds_vs, ws_qs, ws_ks, ws_vs)
    d = time.time()

    with open("perf_benchmark.txt", "a+") as f:
        f.write(f'Matmul-Schedule: {b-a}  Einsum-Schedule: {d-c}\n')
    

# Call whichever experiment over here.
#baking_matmul_exp()
#locality_exp_einsum(xs, ys, zs, ds, dsv, ws, wy, wz)
#ginormous_einsum(xs, ys, zs, ds, dsv, ws, wy, wz)
matmul_einsum_schedule_exp(xs, ys, zs, ds, dsv, ws, wy, wz)

