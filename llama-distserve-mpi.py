import os
import time
import torch
import torch.distributed as dist
from modeling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from utils import init_prof, get_model_config
import nvtx


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not dist.is_initialized():
        dist.init_process_group("gloo")

    cfg = get_model_config()

    model = LlamaForCausalLM(cfg).to(device)
    print(
        f"After init model, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB"
    )

    # Example input and configuration
    batch_size = 1
    seq_len = 1024
    vocab_size = cfg.vocab_size

    use_profiler = False
    num_iterations = 20
    num_warmup_iterations = 10
    num_inf_iterations = num_iterations + num_warmup_iterations

    num_generate_tokens = 100

    with torch.no_grad():
        print(
            f"During infer, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB"
        )

        ctx = init_prof(use_profiler)
        with ctx as prof:
            elapse = 0.0
            if rank == 0:
                for step in range(num_iterations):
                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        start_time = time.time()

                    prefill_time = 0.0

                    # Initialize past_key_values
                    past_key_values = DynamicCache()

                    # Generate random input
                    input_ids = torch.randint(vocab_size, (batch_size, seq_len))
                    position_ids = (
                        torch.arange(seq_len)
                        .unsqueeze(0)
                        .expand(input_ids.shape[0], -1)
                    )

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

                    dist.send_object_list(
                        [
                            input_ids,
                            position_ids,
                            past_key_values.key_cache,
                            past_key_values.value_cache,
                        ],
                        dst=1,
                    )

                    if step >= num_warmup_iterations:
                        end_time = time.time()
                        elapse += end_time - start_time
                        print(f"time for prefill: {prefill_time * 1000:.3f} ms")

                    if use_profiler:
                        prof.step()

            if rank == 1:
                for step in range(num_iterations):
                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        start_time = time.time()
                    decoding_time = 0.0
                    objects = [None, None, None, None]
                    past_key_values = DynamicCache()
                    dist.recv_object_list(objects, src=0)
                    (
                        input_ids,
                        position_ids,
                        past_key_values.key_cache,
                        past_key_values.value_cache,
                    ) = objects

                    # decoding
                    for i_tk in range(num_generate_tokens):
                        if step >= num_warmup_iterations:
                            torch.cuda.synchronize()
                            decode_rng = nvtx.start_range(
                                message="decode", color="green"
                            )
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
                        f"step {step} CUDA memory allocated/reserved: {torch.cuda.memory_allocated(device) / 1024**3:.2f}/{torch.cuda.memory_reserved(device) / 1024**3:.2f} GB"
                    )

                    if step >= num_warmup_iterations:
                        end_time = time.time()
                        elapse += end_time - start_time
                        print(f"time for decode: {decoding_time * 1000:.3f} ms")

                    if use_profiler:
                        prof.step()

    print(f"time for local attention: {elapse / num_inf_iterations * 1000:.3f} ms")
