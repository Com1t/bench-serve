import os
import time
import torch
import argparse
import torch.distributed as dist
from modeling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import copy
from utils import init_prof, get_model_config

import nvtx


def main(
    batch_size=4,
    seq_len=2048,
    num_iterations=50,
    num_generate_tokens=100,
    num_warmup_iterations=10,
    num_prefill_instances=1,
    num_decode_instances=1,
    use_profiler=False,
    prefill_mps_pct=None,
):
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    
    if prefill_mps_pct is not None:
        if rank < num_prefill_instances:
            mps_pct = prefill_mps_pct
            mps_pct //= num_prefill_instances
        else:
            mps_pct = 100 - prefill_mps_pct
            mps_pct //= num_decode_instances

        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_pct)
        print(f"[RANK Prefill] Set MPS to {mps_pct}%")

    if not dist.is_initialized():
        dist.init_process_group("gloo")

    cfg = get_model_config()

    model = LlamaForCausalLM(cfg).to(device)
    print(
        f"After init model, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
    )

    # Example input and configuration
    vocab_size = cfg.vocab_size

    use_profiler = False
    num_inf_iterations = num_iterations + num_warmup_iterations

    with torch.no_grad():
        print(
            f"During infer, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
        )

        ctx = init_prof(use_profiler)
        with ctx as prof:
            elapse = 0.0
            if rank < num_prefill_instances:
                for step in range(num_iterations):
                    dist.barrier()
                    prefill_time = 0.0

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
                        print(f"time for prefill: {prefill_time * 1000:.3f} ms")

                    if use_profiler:
                        prof.step()

            else:
                prefill_input_ids = torch.randint(vocab_size, (batch_size, seq_len))
                prefill_position_ids = (
                torch.arange(seq_len).unsqueeze(0).expand(prefill_input_ids.shape[0], -1)
                )
                prefill_past_key_values = DynamicCache()
                prefill_logits, prefill_outputs, prefill_past_key_values = model(
                    input_ids=prefill_input_ids,
                    position_ids=prefill_position_ids,
                    num_logits_to_keep=1,
                    use_cache=True,
                    past_key_values=prefill_past_key_values,
                )
                # Update position_ids
                prefill_next_position_id = prefill_position_ids[:, -1] + 1
                prefill_position_ids = prefill_next_position_id.unsqueeze(1)
                # Get the next input token
                prefill_input_ids = torch.argmax(prefill_logits, dim=2).reshape(batch_size, -1)

                for step in range(num_iterations):
                    decoding_time = 0.0
                    input_ids = prefill_input_ids
                    position_ids = prefill_position_ids
                    past_key_values = copy.deepcopy(prefill_past_key_values)

                    dist.barrier()
                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        decode_rng = nvtx.start_range(message="decode", color="green")
                        start_time = time.time()

                    # decoding
                    for i_tk in range(num_generate_tokens):
                        if step >= num_warmup_iterations:
                            torch.cuda.synchronize()
                            tkd_rng = nvtx.start_range(message="tkd", color="orange")
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
                            nvtx.end_range(tkd_rng)
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
                        nvtx.end_range(decode_rng)
                        end_time = time.time()
                        elapse += end_time - start_time
                        print(f"time for decode: {decoding_time * 1000:.3f} ms")

                    if use_profiler:
                        prof.step()


    print(f"[rank {rank}] time for local attention: {elapse / num_inf_iterations * 1000:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048, help="Sequence length for prefill"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Number of iterations for inference",
    )
    parser.add_argument(
        "--num_generate_tokens",
        type=int,
        default=100,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--prefill_instance",
        type=int,
        default=1,
        help="Number of prefill instances to use",
    )
    parser.add_argument(
        "--decode_instance",
        type=int,
        default=1,
        help="Number of decoding instances to use",
    )
    parser.add_argument(
        "--use_profiler", action="store_true", default=False, help="Use profiler"
    )
    parser.add_argument(
        "--prefill_mps_pct",
        type=int,
        default=None,
        help="Percentage of MPS threads to use (0-100). Optional.",
    )
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iterations=args.num_iterations,
        num_generate_tokens=args.num_generate_tokens,
        num_warmup_iterations=args.num_warmup_iterations,
        num_prefill_instances=args.prefill_instance,
        num_decode_instances=args.decode_instance,
        use_profiler=args.use_profiler,
        prefill_mps_pct=args.prefill_mps_pct,
    )
