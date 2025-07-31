import os
import time
from random import randint, seed
from benchmark.benchmark_utils import save_to_pytorch_benchmark_format
import argparse
import json
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main(args: argparse.Namespace):
    print(args)
    seed(0)
    num_seqs = args.num_seqs
    max_input_len = args.input_len
    max_ouput_len = args.output_len

    path = os.path.expanduser("/home/my_ubuntu/AI_deployment/nano-vllm/model/Qwen3-0.6B/")

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    total_num_prompt_tokens = sum(len(prompt) for prompt in prompt_token_ids)

    if args.backend == "vllm":
        llm = LLM(path, enforce_eager=not bool(args.use_cudagraph), max_model_len=4096, enable_chunked_prefill=args.enable_chunked_prefill)
        prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]
    else:
        llm = LLM(path, enforce_eager=not bool(args.use_cudagraph), max_model_len=4096)

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_num_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    total_num_token = total_num_output_tokens + total_num_prompt_tokens
    throughput_total_token = (total_num_prompt_tokens + total_num_output_tokens) / t
    throughput_output_token = total_num_output_tokens / t
    throughput_request = num_seqs / t
    # used = llm.scheduler.num_cached_block
    print(f"Total token: {total_num_token} tok, Total output token: {total_num_output_tokens} tok\n"
          f"{throughput_total_token:.2f} total tok/s, {throughput_output_token:.2f} total output tok/s \n"
          f"{throughput_request:.2f} req/s, Time: {t:.2f}s")
    
    # Output JSON results if specified
    if args.output_json:
        results = {
            "CUDA graphy": bool(args.use_cudagraph),
            "elapsed_time": t,
            "num_requests": num_seqs,
            "total_num_tokens": total_num_token,
            "total_num_output_tokens": total_num_output_tokens,
            "requests_per_second": num_seqs / t,
            "tokens_per_second": total_num_token / t,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serving benchmark for lite_infer.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--input-len",
                        type=int,
                        default=1024,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=1024,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--num-seqs",
                        type=int,
                        default=128,
                        help="Number of prompts to process.")
    parser.add_argument("--use-cudagraph",
                        type=int,
                        default=1,
                        help="disable CUDA graphy.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "lite_infer"],
                        default="vllm")
    parser.add_argument(
        '--enable-chunked-prefill',
        type=bool,
        default=True,
        help='enable chunked prefill.')
    args = parser.parse_args()
    main(args)
