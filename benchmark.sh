#!/bin/bash

models=("MHA" "CompMHA" "LinMHA")


for model in ${models[@]}; do
    for (( i=50; i<1024; i=i+50 )) do 
        python pre_train_wiki.py --type MHA --batch_size 4 --layers 16 --sequence_length $i --downsampling_k 64 --rank 1 1> output_$model.in 2> err_$model.in
        echo "model: $model, seq length: $i finished"
    done
done
