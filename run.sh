#!/bin/bash

source env/bin/activate

python3 train.py --type PerfMHA > output.in 2> err.in

deactivate
