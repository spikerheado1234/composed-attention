#!/bin/bash

source env/bin/activate 

python3 train.py --type MHA --batch_size 10 --layers 4 --sequence_length 10 --downsampling_k 64 > output.in 2> err.in

deactivate
