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

    train_step_time = 0 ## Time spent only in the Train Step.

    gradient_computation = 0 ## Time spent in computing gradients.

    optimser_step = 0 ## Time spent in the optimisation step.

    ## For LinFormer. ##
    downsampling_time = 0 # Time spent down_sampling the matrix.

    ## For PerFormer. ##
    transformation_time = 0 # Time taken to map queries and keys to another domain.
