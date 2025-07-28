import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 0) 
        self.input_size_per_partition = input_size # 每个分区的输入大小与原始输入大小相同
        self.output_size_per_partition = divide(output_size, self.tp_size) # 每个设备负责的输出维度大小

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size)) # 创建可训练的权重参数
        self.weight.weight_loader = self.weight_loader # 为权重参数绑定权重加载函数
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # param: 当前设备的参数
        # loaded_weight: 完整的权重W(output_size, input_size)->W=[W0;W1;W2;W3]
        # tp_size=4->GPU0[W0], GPU1[W1], GPU3[W2], GPU3[W3]
        param_data = param.data # 当前设备上的参数数据 , (output_size/tp_size, input_size)
        shard_size = param_data.size(self.tp_dim) # 计算shard_size的大小->output_size_per_partition
        start_idx = self.tp_rank * shard_size # 根据当前设备的排名计算加载权重的起始索引->0, output_size_per_partition, 2*output_size_per_partition, ...
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) # 在 tp_dim 维度上从 start_idx 开始截取 shard_size 大小的片段->GPU1:[0:shard_size,:]; GPU2:[shard_size:2*shard_size,:], ...
        param_data.copy_(loaded_weight) # 将选取的权重复制到当前参数中

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) # (B, S, shard_size)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes # 假设 output_sizes = [4096, 4096]->W0,W1
        super().__init__(input_size, sum(output_sizes), bias=bias) # 支持多个线性层合并为一个大矩阵

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # 将完整权重中的某一个子模块（shard）加载到当前设备的参数中
        # param: 当前设备的参数
        # loaded_weight: 完整的权重[sum(output_sizes), input_size]->[W0;W1]; W0[W00;W01;W02;W03]; W1[W10;W11;W12;W13];
        # loaded_shard_id: 当前要加载的是第几个子模块 W0=0,W1=1
        # tp_size=4->GPU0[W00, W10], GPU1[W01, W11], GPU3[W02;W12], GPU3[W03;W13]
        param_data = param.data # 当前设备上的参数数据 (sum(output_sizes)/tp_size, input_size)
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size  # 计算当前 shard 在当前设备参数中的偏移量, 每个设备总参数量: sum(output_sizes)/tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size # 计算当前 shard 的大小（每个 GPU 负责的部分）
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # 截取出当前 shard 对应的位置
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] # 将完整权重在 tp_dim 维度切分成 tp_size 份, 取出当前设备（tp_rank）的那一份
        param_data.copy_(loaded_weight) # 将当前 shard 的权重复制到参数中
 

class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size # 注意力头的维度大小
        self.total_num_heads = total_num_heads # 总的注意力头的数量
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads # 键值对（K 和 V）的注意力头的数量
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size) # 每个设备处理的注意力头数量
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size) # 每个设备处理 键值对（K 和 V）的注意力头数量
        input_size = hidden_size # 输入的隐藏状态的维度大小
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size # 总的 Q 头数量、K 头数量和 V 头数量乘以每个头的维度大小
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # param: 当前设备的参数
        # loaded_weight: 完整的权重[sum(output_sizes), input_size]->[Q;K;V]; Q[Q0;Q1;Q2;Q3]; K[K0;K1;K2;K3]; V[V0;V1;V2;V3]
        # loaded_shard_id: 当前要加载的是第几个子模块 Q=0,K=1,V=2
        # tp_size=4->GPU0[Q0, K0, V0], GPU1[Q1, K1, V1], GPU3[Q2, K2, V2], GPU3[Q3, K3, V3]
        param_data = param.data # 当前设备上的参数数据 (output_size/tp_size, input_size)
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size # 计算当前 shard 的大小（每个 GPU 负责的部分）
            shard_offset = 0 # 计算当前 shard 在当前设备参数中的偏移量, 每个设备总参数量: (output_size/tp_size, input_size)
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # 从当前设备的参数中选取指定范围的数据
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight) # 将分割后的权重数据复制到当前设备的参数中。


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size) # 每个设备负责的输入维度大小
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # param: 当前设备的参数
        # loaded_weight: 完整的权重[sum(output_sizes), input_size]->W[W0,W1,W2,W3];
        #  # tp_size=4->GPU0[W0], GPU1[W1], GPU3[W2], GPU3[W3]
        param_data = param.data # 当前设备上的参数数据 (output_size/tp_size, input_size)
        shard_size = param_data.size(self.tp_dim) # 计算当前 shard 的大小（每个 GPU 负责的部分）
        start_idx = self.tp_rank * shard_size # 计算当前 shard 在当前设备参数中的偏移量, 每个设备总参数量: (output_size/tp_size, input_size)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight) # 将分割后的权重数据复制到当前设备的参数中。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None) # 只有排名第一的设备才会使用偏置项
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y