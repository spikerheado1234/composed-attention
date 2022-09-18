#!/bin/bash

source env/bin/activate

python3 --type MHA > output.in 2> err.in

source deactivate
