"""
A class for gathering run-time statistics of different operations.
"""

class Stats:
    ## For all attention mechanisms ##
    mha_time = 0 # Total time spent in multi-head-attention.
    mha_enc_time = 0 # Total time spent in encoder attention.
    mha_causal_time = 0 # Total time spent in Masked Multi-Head Self Attention in the Decoder.
    mha_enc_dec_time = 0 # Total time spent in enc-dec attention in the Decoder.
    ffn_time = 0 # Time spent in feed forward network.
    embedding_time = 0 # Time spent in the positional embedding layer.
    total_forward_prop_time = 0 ## Total time spent in forward prop.

    train_step_time = 0 ## Time spent only in the Train Step.

    gradient_computation = 0 ## Time spent in computing gradients.

    optimser_step = 0 ## Time spent in the optimisation step.

    ## For LinFormer. ##
    downsampling_time = 0 # Time spent down_sampling the matrix.

    ## For PerFormer. ##
    transformation_time = 0 # Time taken to map queries and keys to another domain.

    ## MHA specific benchmarks. ##
    linear_transformation = 0 ## Time taken for the X*W1, X*W2 and X*W3 products.

    q_k_v_product = 0 ## Time taken for the efficient way of multiplying out the matrices (in accordance with the performer style.)
    
    mha_ffn = 0 ## Time in the MHA spent on FFN.

    favour_time = 0 ## Time in the MHA spent on the favour attention.

    transpose_time = 0 ## Time taken to transpose keys queries and values in favour attention

    expand_dims_time = 0 ## Time taken to expand the dimension in favour attention.
    
