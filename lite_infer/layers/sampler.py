import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        执行 Top-p (Nucleus) 采样, 从概率分布中采样下一个词。

        参数：
            logits (torch.Tensor): 逻辑值张量，形状为 `[batch_size, vocab_size]`。
            top_p (torch.Tensor): 累积概率阈值，取值范围在 0 到 1 之间, 形状为 `[batch_size, ]`
            temperatures (torch.Tensor): 温度系数, 形状为 `[batch_size, ]`
        返回：
            torch.Tensor: 采样得到的词索引，形状为 `[batch_size, 1]`。

        说明：
            Top-p 采样算法: 选择概率累积和超过阈值 p 的最小集合，将这些词的概率重新归一化后进行采样。
        """        
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float) # [batch_size, vocab_size]

        # 对概率分布进行降序排序。probs_sort: 排序后的概率值，形状与 probs 相同。probs_idx: 排序后的索引，用于映射回原始词汇表。
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # [batch_size, vocab_size]
        # 计算排序后概率的累积和. 返回的 probs_sum 是累积概率分布。
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # 保留累积概率未超过阈值 p 的词汇的概率，其余词汇的概率被置为 0.0。
        mask = probs_sum - probs_sort > top_p.unsqueeze(dim=1)  # 创建掩码，对于每个位置，计算累积概率（不包括当前词）是否超过阈值 p。
        probs_sort[mask] = 0.0  # 将累积概率超过阈值 p 的词的概率置零.

        # 对剩余的概率重新归一化, 确保总和为 1。
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # 从重新归一化的概率分布中采样下一个词. 返回的 next_token 是采样得到的词在排序后概率分布中的索引。
        next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)

         # 在 probs_idx 的最后一维（dim=-1）中，使用 next_token_sorted_idx 作为索引，提取对应的值。沿着 dim=1（列）进行索引提取
         # torch.gather 函数按照给定的索引张量 index，从输入张量中收集 (获取) 数据，并返回一个与索引张量形状一致的张量。
        sample_tokens = torch.gather(probs_idx, -1, index=next_token_sorted_idx).view(-1) # (batch_size, )
        # epsilon = 1e-10  
        # sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
