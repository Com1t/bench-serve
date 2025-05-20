#!/bin/bash

PYTHON=/home/mona/miniconda3/envs/llm-serving/bin/python
OUTPUT_DIR_PREFIX=prefill
PREFILL_EXE=llama-distserve_p.py
NUM_ITERS=50

benchmark_batch_size=(4)
benchmark_seq_len=(1024)
MPS_levels=(10 20 30 40 50 60 70 80 90 100)
for mps_level in ${MPS_levels[@]}; do
    for batch_size in ${benchmark_batch_size[@]}; do
        for seq_len in ${benchmark_seq_len[@]}; do
            echo "Prefill, batch_size: $batch_size, max_seq_len: $seq_len"

            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mps_level nsys profile \
                --gpu-metrics-devices=0 \
                -o ${OUTPUT_DIR_PREFIX}_mps${mps_level}_${batch_size}_${seq_len} \
                --export sqlite \
                --force-overwrite true \
                --trace "cuda,nvtx,osrt,cudnn,cublas" \
                $PYTHON $PREFILL_EXE \
                --batch_size $batch_size \
                --seq_len $seq_len \
                --num_iterations $NUM_ITERS
        done
    done
done
######################################################################

PYTHON=/home/mona/miniconda3/envs/llm-serving/bin/python
OUTPUT_DIR_PREFIX=decode
PREFILL_EXE=llama-distserve_d.py

benchmark_batch_size=(4)
benchmark_prefill_len=1024
benchmark_generate_tks=(128)
MPS_levels=(10 20 30 40 50 60 70 80 90 100)
for mps_level in ${MPS_levels[@]}; do
    for batch_size in ${benchmark_batch_size[@]}; do
        for seq_len in ${benchmark_seq_len[@]}; do
            echo "Decode, batch_size: $batch_size, max_seq_len: $seq_len"

            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mps_level nsys profile \
                --gpu-metrics-devices=0 \
                -o ${OUTPUT_DIR_PREFIX}_mps${mps_level}_${batch_size}_${seq_len} \
                --export sqlite \
                --force-overwrite true \
                --trace "cuda,nvtx,osrt,cudnn,cublas" \
                $PYTHON $PREFILL_EXE \
                --batch_size $batch_size \
                --seq_len $seq_len \
                --num_iterations $NUM_ITERS \
                --num_generate_tokens $benchmark_generate_tks
        done
    done
done
