#!/bin/bash

source env/bin/activate

python3 train.py --type CompMHA --batch_size 64 --layers 4 --sequence_length 128 --downsampling_k 64 > output_compmha.in 2> err_compmha.in

deactivate
