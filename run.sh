#!/bin/bash

source env/bin/activate

python3 train.py --type CompMHA > output_compmha.in 2> err_compmha.in

deactivate
