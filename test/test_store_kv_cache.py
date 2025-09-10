import torch
import matplotlib.pyplot as plt
from lite_infer.layers.attention import store_kvcache


def test1():
    import time
    num_of_times = 560
    num_of_warmup = 10

    B = 1
    N = [512, 1024, 2048, 4096]
    H = 32
    HD = 128
    torch_t = []
    triton_t = []
    for N_ in N:
        # -----------------------triton-----------------------
        key = torch.randn((B * N_, H, HD), dtype=torch.float16).cuda()
        value = torch.randn((B * N_, H, HD), dtype=torch.float16).cuda()
        k_cache = torch.zeros(1, B * N_, H * HD, dtype=torch.float16, device="cuda")
        v_cache = torch.zeros(1, B * N_, H * HD, dtype=torch.float16, device="cuda")
        slot_mapping = torch.arange(0, B * N_, dtype=torch.int32, device="cuda")

        for _ in range(num_of_warmup): # Warm up
            store_kvcache(key, value, k_cache, v_cache, slot_mapping)
        torch.cuda.synchronize()

        t1 = time.time()
        for _ in range(num_of_times):
            store_kvcache(key, value, k_cache, v_cache, slot_mapping)
        torch.cuda.synchronize()
        t2 = time.time()

        # ----------------------torch------------------------
        key = key.view(B * N_, -1)
        value = value.view(B * N_, -1)
        k_cache = k_cache.view(B*N_, -1)
        v_cache = v_cache.view(B*N_, -1)
        for _ in range(num_of_warmup): # Warm up
            k_cache[slot_mapping] = key
            v_cache[slot_mapping] = value
        torch.cuda.synchronize()

        t3 = time.time()
        for _ in range(num_of_times):
            k_cache[slot_mapping] = key
            v_cache[slot_mapping] = value
        torch.cuda.synchronize()
        t4 = time.time()

        triton_t.append((t2-t1)/num_of_times * 1000) # ms
        torch_t.append((t4-t3)/num_of_times * 1000) # ms

        print(f"B={B}, N={N_}, H={H}, D={HD}")
        print("Triton Time cost {:.3f} ms".format(triton_t[-1]))
        print("Torch Time cost {:.3f} ms".format(torch_t[-1]))
        print("key_max ", torch.max(torch.abs(k_cache - key)))
        print("key_mean ", torch.mean(torch.abs(k_cache - key)))
        print("value_max ", torch.max(torch.abs(v_cache - value)))
        print("value_mean ", torch.mean(torch.abs(v_cache - value)))    
        print("--------------------------------------------------------------------")
        assert torch.allclose(key, k_cache, atol=1e-2, rtol=0)
        assert torch.allclose(value, v_cache, atol=1e-2, rtol=0)

    color_ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plt.figure(1, figsize=(10, 6))
    plt.plot(N, torch_t, marker='o', color=color_[0], label="torch")
    plt.plot(N, triton_t, marker='s', color=color_[1], label="triton")
    plt.xlabel("seq_len")
    plt.ylabel("Average execution time (ms)")
    plt.title(f"performance comparison for kv cache store kernel: H={H} Hd={HD}")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/kv_cache_store_kernel_benchamrk.png", dpi=500)



if __name__ == '__main__':
    test1()
