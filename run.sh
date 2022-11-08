#!/bin/bash


python pre_train_wiki.py --type CompMHA --batch_size 8 --layers 16 --sequence_length 124 --downsampling_k 64 > output.in 2> err.in

