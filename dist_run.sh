#!/bin/bash


python3 dist_train.py --type MHA --batch_size 32 --layers 10 --sequence_length 256 --downsampling_k 64 > output.in 2> err.in
