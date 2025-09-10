import os
import time
import numpy as np
import argparse
from random import randint, seed
from tqdm.auto import tqdm
from lite_infer import LLM, SamplingParams

# --- Constants ---
MODEL_PATH = os.path.expanduser("/home/my_ubuntu/AI_deployment/lite_infer/model/Qwen3-0.6B")
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
os.environ['TORCH_DYNAMO_CACHE_SIZE_LIMIT'] = '32'

# --- Seed for reproducibility ---
seed(0)
np.random.seed(0)

class RequestMetrics:
    """Stores metrics for a single request."""
    def __init__(self, request_id, input_len, max_output_len):
        self.request_id = request_id
        self.input_len = input_len
        self.max_output_len = max_output_len
        self.submission_time = -1
        self.first_token_time = -1
        self.completion_time = -1
        self.output_len = -1

    def record_submission(self):
        self.submission_time = time.perf_counter()

    def record_first_token(self):
        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter()

    def record_completion(self, output_ids):
        self.completion_time = time.perf_counter()
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float('nan')

    @property
    def latency(self):
        return self.completion_time - self.submission_time

def main():
    """Main function to run the serving benchmark."""
    parser = argparse.ArgumentParser(description="Serving benchmark for lite_infer.")
    parser.add_argument("--num-requests", type=int, default=128, help="Number of requests to process.")
    parser.add_argument("--request-rate", type=int, default=8, help="Request rate (requests per second).")
    args = parser.parse_args()

    NUM_REQUESTS = args.num_requests
    REQUEST_RATE = args.request_rate

    print(f"\n--- Running benchmark with --num-requests {NUM_REQUESTS} --request-rate {REQUEST_RATE} ---")
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096)
    engine = llm

    # --- Generate random prompts_token_ids ---
    prompts_token_ids = [[randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))] for _ in range(NUM_REQUESTS)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, MAX_OUTPUT_LEN)) for _ in range(NUM_REQUESTS)]

    # --- Generate request arrival times ---
    request_intervals = np.random.poisson(1.0 / REQUEST_RATE, NUM_REQUESTS)
    arrival_times = np.cumsum(request_intervals)

    # --- Benchmark loop ---
    metrics = {}
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []

    with tqdm(total=NUM_REQUESTS, desc="Processing Requests") as pbar:
        while requests_sent < NUM_REQUESTS or not engine.is_finished():
            # --- Send new requests ---
            current_time = time.perf_counter()
            while requests_sent < NUM_REQUESTS and current_time - start_time >= arrival_times[requests_sent]:
            # while requests_sent < NUM_REQUESTS:
                prompt = prompts_token_ids[requests_sent]
                sp = sampling_params[requests_sent]
                
                engine.add_request(prompt, sp)
                
                new_seq = engine.scheduler.waiting[-1]
                seq_id = new_seq.seq_id
                req_metrics = RequestMetrics(seq_id, len(prompt), sp.max_tokens)
                req_metrics.record_submission()
                metrics[seq_id] = req_metrics
                
                requests_sent += 1

            # --- Engine step ---
            if not engine.is_finished():
                finished_outputs, _ = engine.step()

                # Record first token time for all processed sequences
                all_processed_seqs = list(engine.scheduler.running)
                for seq in all_processed_seqs:
                    if seq.seq_id in metrics:
                        metrics[seq.seq_id].record_first_token()

                # for value in finished_outputs.values():
                for seq_id, output_ids in finished_outputs:
                    if seq_id in metrics:
                        metrics[seq_id].record_first_token() # Ensure first token time is recorded
                        metrics[seq_id].record_completion(output_ids) # output token exclude prompt token
                        
                        completed_latencies.append(metrics[seq_id].latency)
                        avg_latency = np.mean(completed_latencies)
                        pbar.set_postfix({"Avg Latency": f"{avg_latency:.2f}s"})
                        pbar.update(1)
            else:
                # If no requests are running or waiting, sleep briefly
                time.sleep(0.01)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # --- Calculate and print metrics ---
    total_input_tokens = sum(m.input_len for m in metrics.values())
    total_output_tokens = sum(m.output_len for m in metrics.values() if m.output_len != -1)
    
    avg_ttft = np.mean([m.ttft for m in metrics.values() if m.first_token_time != -1])
    avg_tpot = np.mean([m.tpot for m in metrics.values() if not np.isnan(m.tpot)])
    avg_latency = np.mean([m.latency for m in metrics.values() if m.completion_time != -1])
    throughput = total_output_tokens / total_time

    total_num_token = total_output_tokens + total_input_tokens
    throughput_total_token = (total_input_tokens + total_output_tokens) / total_time
    throughput_output_token = total_output_tokens / total_time
    throughput_request = requests_sent / total_time

    print("--- Benchmark Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total output token: {total_output_tokens}")
    print(f"实际请求数量: {requests_sent}")
    print(f"请求速率: {REQUEST_RATE} req/s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"首token延迟: {avg_ttft :.2f} s")
    print(f"token生成延迟: {avg_tpot * 1000:.2f} ms/token")
    print(f"平均请求用时: {avg_latency:.2f} s")
    print(f"Total token: {total_num_token} tok, Total output token: {total_output_tokens} tok \n"
          f"{throughput_total_token:.2f} total tok/s, {throughput_output_token:.2f} total output tok/s \n"
          f"{throughput_request:.2f} req/s, Time: {total_time:.2f}s \n"
          f"{llm.prefix_cache_hit_rate[0]:.2f}%, {llm.prefix_cache_hit_rate[1]} cached block, {llm.prefix_cache_hit_rate[2]} cached tok") 
    print("-------------------------\n")

if __name__ == "__main__":
    main()