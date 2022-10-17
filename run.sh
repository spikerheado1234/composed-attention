#!/bin/bash

source env/bin/activate

python3 train.py --type MHA --batch_size 64 --layers 24 --sequence_length 512 --downsampling_k 64 > output.in 2> err.in

deactivate
