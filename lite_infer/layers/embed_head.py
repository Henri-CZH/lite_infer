import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lite_infer.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    # 将输入的词索引转换为对应的词嵌入向量
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings # 词表的大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size # 每个设备负责的词表大小
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank # 当前设备负责的词表部分的起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition # 当前设备负责的词表部分的结束索引
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim)) # (V/tp_size, D)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # param: 当前设备的参数
        # loaded_weight: 完整的权重[V, D]->[V1;V2];     
        param_data = param.data # 当前设备上的参数数据 (V/tp_size, D)
        shard_size = param_data.size(0) # 计算当前 shard 的大小（每个 GPU 负责的部分）
        start_idx = self.tp_rank * shard_size # 计算当前 shard 在当前设备参数中的偏移量, 每个设备总参数量: (V/tp_size, D)
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight) # 将分割后的权重数据复制到当前设备的参数中。

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) # (B,S)->[0,1,4;7,10,11]; vocab_start_idx=2, vocab_end_idx=15->mask=[0,0,1;1,1,1]
            x = mask * (x - self.vocab_start_idx) # 将词条的编号转换为在自己负责的小字典中的编号, (x-vocab_start_idx) = [-2,-1,2;5,8,9]->mask->x = [0,0,2;5,8,9]
        y = F.embedding(x, self.weight) # (B,S)->(B,S,D)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # 将不在当前设备负责范围内的词嵌入向量置为零->(B,1,S)@(B,S,D)->(B,S,D)
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
# 将模型的隐藏状态转换为词汇表上的对数概率
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1 # 累积序列长度Q: [0, seq1_len, seq1_len + eq2_len, ...]
            x = x[last_indices].contiguous() # 取x中每个序列的最后一个token的隐藏状态，并确保张量在内存中是连续的
        logits = F.linear(x, self.weight, self.bias) #(B,S,D)@(V/N)->(B,S,V/N)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None # 在主设备上创建一个列表 all_logits，用于存储各个设备的对数概率
            dist.gather(logits, all_logits, 0) # 将各个设备的对数概率收集到主设备上的all_logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None # 将收集到的对数概率沿着最后一个维度进行拼接: (B,S,V/N)->(B,S,V)
        return logits
