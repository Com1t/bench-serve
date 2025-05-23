import os
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from modeling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from torch.profiler import profile, record_function, ProfilerActivity

import nvtx

def init_prof(use_profiler):
    activities = []
    # activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.CUDA)

    from contextlib import nullcontext

    ctx = (
        torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile/"),
            record_shapes=True,
            with_stack=True,
        )
        if use_profiler
        else nullcontext()
    )
    return ctx


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

    cfg = LlamaConfig()
    cfg.hidden_size = 4096
    cfg.intermediate_size = 11008
    cfg.max_position_embeddings = 4096
    cfg.num_attention_heads = 32
    cfg.num_key_value_heads = 32
    cfg.num_hidden_layers = 1
    cfg.rms_norm_eps = 1e-05
    cfg._attn_implementation = "sdpa"
    cfg.torch_dtype = torch.float16

    model = LlamaForCausalLM(cfg).to(device)
    print(
        f"After init model, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
    )

    # Example input and configuration
    batch_size = 1
    seq_len = 1024
    vocab_size = cfg.vocab_size

    use_profiler = False
    num_iterations = 50
    num_warmup_iterations = 10
    num_inf_iterations = num_iterations + num_warmup_iterations

    num_generate_tokens = 100

    with torch.no_grad():
        print(
            f"During infer, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
        )

        ctx = init_prof(use_profiler)
        with ctx as prof:
            elapse = 0.0
            for step in range(num_iterations):
                if step >= num_warmup_iterations:
                    torch.cuda.synchronize()
                    start_time = time.time()

                prefill_time = 0.0
                decoding_time = 0.0

                # Initialize past_key_values
                past_key_values = DynamicCache()

                # Generate random input
                input_ids = torch.randint(vocab_size, (batch_size, seq_len))
                position_ids = (
                    torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1)
                )
    
                if step >= num_warmup_iterations:
                    torch.cuda.synchronize()
                    prefill_rng = nvtx.start_range(message="prefill", color="blue")
                    prefill_start_time = time.time()


                past_key_values = DynamicCache()
                logits, outputs, past_key_values = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    num_logits_to_keep=1,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                # Update position_ids
                next_position_id = position_ids[:, -1] + 1
                position_ids = next_position_id.unsqueeze(1)
                
                # Get the next input token
                input_ids = torch.argmax(logits, dim=2).reshape(batch_size, -1)

                if step >= num_warmup_iterations:
                    torch.cuda.synchronize()
                    nvtx.end_range(prefill_rng)
                    prefill_end_time = time.time()
                    prefill_time = prefill_end_time - prefill_start_time
                
                # decoding
                for i_tk in range(num_generate_tokens):
                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        decode_rng = nvtx.start_range(message="decode", color="blue")
                        itr_start_time = time.time()

                    with torch.no_grad():
                        logits, outputs, past_key_values = model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            num_logits_to_keep=1,
                            use_cache=True,
                            past_key_values=past_key_values,
                        )

                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        nvtx.end_range(decode_rng)
                        itr_end_time = time.time()
                        itr_elapse_time = itr_end_time - itr_start_time
                        decoding_time += itr_elapse_time

                    # Update position_ids
                    next_position_id = position_ids[:, -1] + 1
                    position_ids = next_position_id.unsqueeze(1)

                    # Get the next input token
                    input_ids = torch.argmax(logits, dim=2).reshape(batch_size, -1)

                print(
                    f"step {step} CUDA memory allocated/reserved: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f}/{torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
                )

                if step >= num_warmup_iterations:
                    end_time = time.time()
                    elapse += end_time - start_time
                    print(f"time for prefill: {prefill_time * 1000:.3f} ms")
                    print(f"time for decode: {decoding_time * 1000:.3f} ms")

                if use_profiler:
                    prof.step()


    print(f"time for local attention: {elapse / num_inf_iterations * 1000:.3f} ms")
