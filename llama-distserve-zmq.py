import os
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from modeling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import zmq
from torch.profiler import profile, record_function, ProfilerActivity
import nvtx


def init_prof(use_profiler):
    activities = [torch.profiler.ProfilerActivity.CUDA]
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

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        dist.init_process_group("gloo")

    # Setup ZMQ context
    ctx_zmq = zmq.Context()
    if rank == 0:
        socket = ctx_zmq.socket(zmq.PUSH)
        socket.bind("tcp://*:5555")
    else:
        socket = ctx_zmq.socket(zmq.PULL)
        socket.connect("tcp://localhost:5555")

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

    batch_size = 1
    seq_len = 1024
    vocab_size = cfg.vocab_size

    use_profiler = False
    num_iterations = 20
    num_warmup_iterations = 10
    num_inf_iterations = num_iterations + num_warmup_iterations
    num_generate_tokens = 100

    with torch.no_grad():
        ctx = init_prof(use_profiler)
        with ctx as prof:
            elapse = 0.0
            for step in range(num_iterations):
                if rank == 0:
                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        start_time = time.time()

                    prefill_time = 0.0
                    past_key_values = DynamicCache()
                    input_ids = torch.randint(vocab_size, (batch_size, seq_len))
                    position_ids = torch.arange(seq_len).unsqueeze(0).expand(input_ids.shape[0], -1)

                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        prefill_rng = nvtx.start_range(message="prefill", color="blue")
                        prefill_start_time = time.time()

                    logits, outputs, past_key_values = model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        num_logits_to_keep=1,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    next_position_id = position_ids[:, -1] + 1
                    position_ids = next_position_id.unsqueeze(1)
                    input_ids = torch.argmax(logits, dim=2).reshape(batch_size, -1)

                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        nvtx.end_range(prefill_rng)
                        prefill_end_time = time.time()
                        prefill_time = prefill_end_time - prefill_start_time

                    # Serialize and send with ZMQ
                    tensor_package = {
                        "input_ids": input_ids.cpu().numpy(),
                        "position_ids": position_ids.cpu().numpy(),
                        "key_cache": past_key_values.key_cache,
                        "value_cache": past_key_values.value_cache,
                    }
                    socket.send_pyobj(tensor_package)

                    if step >= num_warmup_iterations:
                        end_time = time.time()
                        elapse += end_time - start_time
                        print(f"time for prefill: {prefill_time * 1000:.3f} ms")

                    if use_profiler:
                        prof.step()

                elif rank == 1:
                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        start_time = time.time()

                    decoding_time = 0.0
                    past_key_values = DynamicCache()

                    # Receive and unpack
                    tensor_package = socket.recv_pyobj()
                    input_ids = torch.tensor(tensor_package["input_ids"]).to(device)
                    position_ids = torch.tensor(tensor_package["position_ids"]).to(device)
                    past_key_values.key_cache = tensor_package["key_cache"]
                    past_key_values.value_cache = tensor_package["value_cache"]

                    for i_tk in range(num_generate_tokens):
                        if step >= num_warmup_iterations:
                            torch.cuda.synchronize()
                            decode_rng = nvtx.start_range(message="decode", color="green")
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
                            decoding_time += itr_end_time - itr_start_time

                        next_position_id = position_ids[:, -1] + 1
                        position_ids = next_position_id.unsqueeze(1)
                        input_ids = torch.argmax(logits, dim=2).reshape(batch_size, -1)

                    print(
                        f"step {step} CUDA memory allocated/reserved: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f}/{torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
                    )

                    if step >= num_warmup_iterations:
                        end_time = time.time()
                        elapse += end_time - start_time
                        print(f"time for decode: {decoding_time * 1000:.3f} ms")

                    if use_profiler:
                        prof.step()
    
    tensor = torch.ones(1).cuda() * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"time for local attention: {elapse / num_inf_iterations * 1000:.3f} ms")
