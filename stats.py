"""
A class for gathering run-time statistics of different operations.
"""

class Stats:
    ## For all attention mechanisms ##
    mha_time = 0 # Total time spent in multi-head-attention.
    ffn_time = 0 # Time spent in feed forward network.

    ## For LinFormer. ##
    downsampling_time = 0 # Time spent down_sampling the matrix.
