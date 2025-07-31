# ite_infer
A light llama-like llm inference framework based on the triton and CUDA kernel

## 特性

- 相比 vLLM, Qwen3 0.5B 模型加速比最高达 `1.4` 倍。
- 支持 `llama3`、`Qwen2.5`、`Llava1.5` 模型推理，支持 `top-p` 采样, 支持流式输出。
- 支持 `CUDA graph`，`prefix caching`，`张量并行`，`Continuous Batching`。
- 支持 `flashattention1`、`flashattention2`、 `flashdecoding`。
- 支持 kv cache 的高效动态管理（`Pagedattnetion`）。
- 支持算子融合，如：逐元素相乘 `*` 和 `silu` 的融合, k v 线性层融合。
- 部分自定义算子如：`FlashAttention`、`逐元素相乘` 等采用高效 `triton` `CUDA` 内核实现。

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

### Qwen3-0.5 模型性能测试对比

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX A3000 Laptop (6GB)
- Model: Qwen3-0.6B
- Total Requests: 128 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results with CUDA Graphy:**
| Inference Engine | Output Tokens | Time (s) | output token (tokens/s) |  total token (tokens/s)  | Throughput (req/s) |
|------------------|---------------|----------|-------------------------|--------------------------|--------------------|
| vLLM             | 66720         | 87.39    |         769.41          |          1579.37         |      1.46          |
| lite_infer       | 66720         | 58.48    |         1140.90         |          2360.06         |      2.19          |


| Inference Engine | 总时间 (s) | 请求速率 (req/s) | Throughput (token/s) | 平均首token延迟 (s)         | 平均token延迟 (token/ms)     | 平均单个请求用时 (s)  |
|------------------|------------|-----------------|----------------------|----------------------------|-------------------------------|--------------------- |
| lite_infer       |    59.13   |        8        |     1128.36          |     15.14                  |      20.03                    |        25.39         |
| lite_infer       |    58.61   |        128      |     1138.4           |     21.08                  |      20.51                    |        31.57         |


**Performance Results without CUDA Graphy:**
| Inference Engine | Output Tokens | Time (s) | output token (tokens/s) | total token (tokens/s)  | Throughput (req/s) |
|------------------|---------------|----------|-------------------------|-------------------------|--------------------|
| vLLM             | 66720         | 106.64   | 625.67                  |      1294.25            |         1.2        |
| lite_infer       | 66720         | 85.46    | 780.68                  |      1614.91            |         1.5        |


| Inference Engine | 总时间 (s) | 请求速率 (req/s) | Throughput (token/s) | 平均首token延迟 (ms)        | 平均token延迟 (token/ms)     | 平均单个请求用时 (s)  |
|------------------|------------|-----------------|----------------------|----------------------------|-------------------------------|---------------------|
| lite_infer       |    91.29   |        8        |     730.83           |     23.60                  |      29.64                    |        38.84        |
| lite_infer       |    85.87   |        128      |     776.99           |     27.7                   |      28.08                    |        42.16        |


## 如何使用
`example.py` 程序运行成功后，终端显示界面如下所示，在终端中输入你的问题即可。


