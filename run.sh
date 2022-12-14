#!/bin/bash


python pre_train_wiki.py --type MHA --batch_size 4 --layers 16 --sequence_length 124 --downsampling_k 64 --rank 1 > output.in 2> err.in

