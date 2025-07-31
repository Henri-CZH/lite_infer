import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TORCH_CUDA_ARCH_LIST'] = 'Ampere'

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="swish_lib",
    sources=["swish.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    y: Optional[torch.Tensor] = None,
    warmup: int = 50,
    iters: int = 1000,
    show_all: bool = False,
):
    out = torch.zeros_like(y, dtype=y.dtype).cuda().contiguous()
    # if out is not None:
    #     out.fill_(0)
        # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, y, out)
    else:
        for i in range(warmup):
            out = perf_func(x)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, y, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

@torch.compile
def torch_swish(x, y, out=None):
    if out is None:
        return y * F.silu(x)
    else:
        # torch.sigmoid(x, out=out)
        # out.mul_(x)
        # return out
        res = y * F.silu(x)
        return out.copy_(res)


Ss = [512, 1024, 2048]
Ks = [8192, 11008, 13824, 29568]
L = len(Ks)

SKs = [(S, K) for S in Ss for K in Ks]
res_swish_f16x8 = []
res_torch_f16x8 = []

plt.figure(1)
for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.randn((S, K)).cuda().float().contiguous()
    # run_benchmark(lib.swish_f32, x, "f32", y)
    # run_benchmark(lib.swish_f32x4, x, "f32x4", y)
    # run_benchmark(torch_swish, x, "f32_th", y)
    print("-" * 85)
    x_f16 = x.half().contiguous()
    y_f16 = y.half().contiguous()
    # run_benchmark(lib.swish_f16, x_f16, "f16", y_f16)
    # run_benchmark(lib.swish_f16x2, x_f16, "f16x2", y_f16)
    res_swish_f16x8.append(run_benchmark(lib.swish_f16x8, x_f16, "f16x8", y_f16)[1])
    # run_benchmark(lib.swish_f16x8_pack, x_f16, "f16x8pack", y_f16)
    res_torch_f16x8.append(run_benchmark(torch_swish, x_f16, "f16_th", y_f16)[1])
    print("-" * 85)

color_ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(len(Ss)):
    plt.plot(Ks, res_swish_f16x8[i * L : (i + 1) * L], marker='o', color=color_[i], label="CUDA_SiluAndMul_" + str(Ss[i]))
    plt.plot(Ks, res_torch_f16x8[i * L : (i + 1) * L], marker='s', linestyle='--', color=color_[i], label="torch_SiluAndMul_" + str(Ss[i]))
plt.xlabel("intermediate_size")
plt.ylabel("Average execution time (ms)")
plt.title("torch and CUDA Performance Comparison for SiluAndMul kernel")
plt.legend()
plt.grid(True)
plt.savefig("./swish_kernel_benchamrk.png")


# ------------------------------------SiluAndMul-------------------------------------------------
#                                         S=512, K=8192
# -------------------------------------------------------------------------------------
#          out_f16x8: ['-1.37792969 ', '0.08465576  '], time:0.10860872ms
#         out_f16_th: ['-1.37890625 ', '0.08465576  '], time:0.10778356ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=512, K=11008
# -------------------------------------------------------------------------------------
#          out_f16x8: ['-0.01913452 ', '-0.1472168  '], time:0.14432073ms
#         out_f16_th: ['-0.01911926 ', '-0.1472168  '], time:0.14306736ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=512, K=13824
# -------------------------------------------------------------------------------------
#          out_f16x8: ['0.03970337  ', '-0.37817383 '], time:0.17878866ms
#         out_f16_th: ['0.03970337  ', '-0.37841797 '], time:0.18043184ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=512, K=29568
# -------------------------------------------------------------------------------------
#          out_f16x8: ['-0.01249695 ', '-0.55908203 '], time:0.37460327ms
#         out_f16_th: ['-0.01249695 ', '-0.55908203 '], time:0.37769341ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=1024, K=8192
# -------------------------------------------------------------------------------------
#          out_f16x8: ['0.1583252   ', '0.26074219  '], time:0.21192622ms
#         out_f16_th: ['0.15844727  ', '0.26074219  '], time:0.21007252ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=1024, K=11008
# -------------------------------------------------------------------------------------
#          out_f16x8: ['0.1496582   ', '0.02867126  '], time:0.28034472ms
#         out_f16_th: ['0.14978027  ', '0.02867126  '], time:0.28132367ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=1024, K=13824
# -------------------------------------------------------------------------------------
#          out_f16x8: ['-0.25708008 ', '-0.45361328 '], time:0.35037255ms
#         out_f16_th: ['-0.25708008 ', '-0.45361328 '], time:0.35062480ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=1024, K=29568
# -------------------------------------------------------------------------------------
#          out_f16x8: ['0.03222656  ', '-0.05651855 '], time:0.74144125ms
#         out_f16_th: ['0.03222656  ', '-0.05654907 '], time:0.74080253ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=2048, K=8192
# -------------------------------------------------------------------------------------
#          out_f16x8: ['0.0051918   ', '-0.69384766 '], time:0.41663074ms
#         out_f16_th: ['0.00519562  ', '-0.69433594 '], time:0.41378570ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=2048, K=11008
# -------------------------------------------------------------------------------------
#          out_f16x8: ['-2.93554688 ', '0.94140625  '], time:0.55430746ms
#         out_f16_th: ['-2.93554688 ', '0.94189453  '], time:0.55359721ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=2048, K=13824
# -------------------------------------------------------------------------------------
#          out_f16x8: ['0.0269928   ', '-0.13818359 '], time:0.69592643ms
#         out_f16_th: ['0.0269928   ', '-0.13818359 '], time:0.75060081ms
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#                                         S=2048, K=29568
# -------------------------------------------------------------------------------------
#          out_f16x8: ['-0.07739258 ', '0.06072998  '], time:1.54397202ms
#         out_f16_th: ['-0.07733154 ', '0.06069946  '], time:1.57209587ms
# -------------------------------------------------------------------------------------
