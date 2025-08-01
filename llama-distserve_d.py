import os
import time
import copy
import nvtx
import torch
import argparse
from modeling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from utils import init_prof, get_model_config


def main(
    batch_size=4,
    seq_len=2048,
    num_iterations=50,
    num_generate_tokens=100,
    num_warmup_iterations=10,
    use_profiler=False,
    mps_pct=None,
):
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

    if mps_pct is not None:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_pct)
        print(f"[RANK Prefill] Set MPS to {mps_pct}%")

    cfg = get_model_config()
    model = LlamaForCausalLM(cfg).to(device)
    print(
        f"After init model, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB"
    )

    # Example input and configuration
    vocab_size = cfg.vocab_size
    num_inf_iterations = num_iterations + num_warmup_iterations

    with torch.no_grad():
        print(
            f"During infer, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB"
        )

        ctx = init_prof(use_profiler)
        with ctx as prof:
            elapse = 0.0

            prefill_input_ids = torch.randint(vocab_size, (batch_size, seq_len))
            prefill_position_ids = (
                torch.arange(seq_len)
                .unsqueeze(0)
                .expand(prefill_input_ids.shape[0], -1)
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
            prefill_input_ids = torch.argmax(prefill_logits, dim=2).reshape(
                batch_size, -1
            )

            for step in range(num_iterations):
                if step >= num_warmup_iterations:
                    torch.cuda.synchronize()
                    decode_rng = nvtx.start_range(message="decode", color="blue")
                    start_time = time.time()

                prefill_time = 0.0
                decoding_time = 0.0
                input_ids = prefill_input_ids
                position_ids = prefill_position_ids
                past_key_values = copy.deepcopy(prefill_past_key_values)
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
                        if i_tk == 0:
                            prefill_time = itr_elapse_time
                        else:
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
                    nvtx.end_range(decode_rng)
                    end_time = time.time()
                    elapse += end_time - start_time
                    print(f"time for prefill: {prefill_time * 1000:.3f} ms")
                    print(f"time for decode: {decoding_time * 1000:.3f} ms")

                if use_profiler:
                    prof.step()

    print(f"time for local attention: {elapse / num_inf_iterations * 1000:.3f} ms")


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
        "--use_profiler", action="store_true", default=False, help="Use profiler"
    )
    parser.add_argument(
        "--mps_pct",
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
        use_profiler=args.use_profiler,
        mps_pct=args.mps_pct,
    )
