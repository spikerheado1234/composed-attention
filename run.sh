#!/bin/bash

source env/bin/activate

python3 train.py --type LinMHA --downsampling_k 40 > output.in 2> err.in

deactivate
