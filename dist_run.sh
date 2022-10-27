#!/bin/bash


python3 dist_train.py --type MHA --batch_size 64 --layers 4 --sequence_length 124 --downsampling_k 64 > output.in 2> err.in
