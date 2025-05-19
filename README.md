# Benchmark serving efficiency

# 1. Introduction
This repository contains the code and instructions to benchmark the serving efficiency of different models. The goal is to provide a comprehensive evaluation of the performance of various models in terms of latency, throughput, and resource utilization.

# 2. Prerequisites
- Python 3.7 or higher
- Torch 2.0 or higher
- Transformers 4.0 or higher

# 3. Usage
Prefill
```
python benchmark-serving/llama-distserve_d.py --batch_size 4 --seq_len 2048 --num_iterations 50 --num_generate_tokens 100 --num_warmup_iterations 10 --use_profiler
```

Decode
```
python benchmark-serving/llama-distserve_d.py --batch_size 4 --seq_len 2048 --num_iterations 50 --num_generate_tokens 100 --num_warmup_iterations 10 --use_profiler

```

-- `--batch_size`: The batch size for the benchmark. Default is 4.
-- `--seq_len`: The sequence length for the benchmark. Default is 2048.
-- `--num_iterations`: The number of iterations to run the benchmark. Default is 50.
-- `--num_generate_tokens`: The number of tokens to generate. Default is 100.
-- `--num_warmup_iterations`: The number of warmup iterations to run before the benchmark. Default is 10.
-- `--use_profiler`: Whether to use the profiler for benchmarking. Default is False.

