# ite_infer
A light llama-like llm inference framework based on the triton and CUDA kernel

## 特性

- 相比 HF transformers, llama3 1B 和 3B 模型加速比最高达 `1.4` 倍。
- 支持 `llama3`、`Qwen2.5`、`Llava1.5` 模型推理，支持 `top-p` 采样, 支持流式输出。
- 支持 `CUDA graph`，`prefix caching`。
- 支持 `flashattention1`、`flashattention2`、 `flashdecoding`。
- 支持 kv cache 的高效动态管理（`Pagedattnetion`）。
- 支持算子融合，如：逐元素相乘 `*` 和 `silu` 的融合, k v 线性层融合, `skip` 和 `rmsnorm` 融合。
- 部分自定义算子如：`rmsnorm`、`rope`、`softmax`、`逐元素相乘` 等采用高效 `triton` 内核实现。

## GPU Information

cuda 版本以及 torch、triton 版本：

```bash
# nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:19:38_PST_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0

# Python 3.10.12 包版本:
# pip list | grep torch
torch                     2.5.1+cu124
torchaudio                2.5.1+cu124
torchmetrics              1.7.1
torchview                 0.2.7
torchvision               0.20.1+cu124
triton                    3.1.0
```

## 回答准确性验证

llama3.2-1.5B-Instruct 模型流式输出结果测试：

![流式输出](./images/llama3.2_stream_generate.gif)


## benchmark 性能测试

### Llama-3.2-1B 模型性能测试对比

Nvidia `A3000` 硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `1.4x` 倍。batch_size = 16 的提示词，`max_gen_len = 1900` 时，benchmark 性能测试结果:

```bash
lite_llama inference time: 68.0684 s
Transformers inference time: 99.1228 s
lite_llama inference output tokens number: 29892
Transformers inference output tokens number: 30324
lite_llama throughput: 439.15 tokens/s
Transformers throughput: 305.92 tokens/s
lite_llama per token latency: 2.277145 ms/token
Transformers per token latency: 3.268789 ms/token
```

**Performance Results:**
| Inference Engine | 总时间 (s) | 请求速率 (req/s) | Throughput (token/s) | 平均首token延迟 (ms)        | 平均每token延迟 (token/ms)     | 平均请求延迟 (s)  |
|------------------|------------|-----------------|----------------------|----------------------------|-------------------------------|------------------|
|                  |            |                 |                      |                            |                               |                  |
| lite-infer       |    59.13   |        8        |     1128.36          |     15.14                  |      20.03                    |        25.39     |


## 如何使用
`example.py` 程序运行成功后，终端显示界面如下所示，在终端中输入你的问题即可。


