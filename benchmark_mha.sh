#!/bin/bash

models=("MHA")
layers=(2 4 8 16 32 64)
batchSizes=(2 4 8 16 32 64 128 256 512 1024)
sequenceLengths=(2 4 8 16 32 64 128 256 512 1024)

source env/bin/activate

for model in ${models[@]}; do
    for layer in ${layers[@]}; do
        for batchSize in ${batchSizes[@]}; do
            for sequenceLength in ${sequenceLengths[@]}; do
                python3 benchmark.py --type ${model} --downsampling_k $((sequenceLength / 2)) --batch_size ${batchSize} --layers ${layer} --sequence_length ${sequenceLength} > output_mha.in 2> err_mha.in;
		        echo "model: ${model}, layer: ${layer}, batch size: ${batchSize}, sequence length: ${sequenceLength} finished"
            done 
        done
    done
done

deactivate
