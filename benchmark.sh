#!/bin/bash

# For now, we have a tapered down version
models=("MHA" "LinMHA" "PerfMHA")
layers=(2 4)
batchSizes=(2 4)
sequenceLengths=(2 4)

for model in ${models[@]}; do
    for layer in ${layers[@]}; do
        for batchSize in ${batchSizes[@]}; do
            for sequenceLength in ${sequenceLengths[@]}; do
                python3 train.py --type ${model} --downsampling_k $((sequenceLength / 2)) --batch_size ${batchSize} --layers ${layer} --sequence_length ${sequenceLength}
            done 
        done
    done
done