#!/bin/bash

PYTHON=/home/mona/miniconda3/envs/llm-serving/bin/python
OUTPUT_DIR=decode_time_2048
EXE=llama-distserve_d.py

nsys profile \
    --gpu-metrics-devices=0 \
    -o $OUTPUT_DIR --force-overwrite true \
    --trace "cuda,nvtx,osrt,cudnn,cublas" \
    $PYTHON $EXE
