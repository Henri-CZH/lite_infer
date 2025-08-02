from copy import copy
from enum import Enum, auto
from itertools import count

from lite_infer.sampling_params import SamplingParams

# SequenceStatus 是一个枚举类，定义了序列的三种状态：WAITING（等待处理）、RUNNING（正在处理）和 FINISHED（处理完成）
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256 # 每个房间可以存储的 token 数量
    counter = count() # 为每个序列生成唯一的 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # 序列的唯一的编号
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids) # 复制输入的 token 列表，避免修改原始列表
        self.last_token = token_ids[-1] # 记录序列的最后一个 token
        self.num_tokens = len(self.token_ids) # 记录序列的 token 总数
        self.num_prompt_tokens = len(token_ids) # 记录输入的提示信息的 token 数量
        self.num_cached_tokens = 0 # 初始时缓存的 token 数量为 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_p = sampling_params.top_p

    def __len__(self):
        return self.num_tokens # 返回序列的token数量

    def __getitem__(self, key):
        return self.token_ids[key] # 通过token编号可以取出对应的token内容

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED # 判断序列是否处理完成

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens # 序列的总长度减去提示信息的长度，得到生成结果的长度

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens] # 获取序列的提示信息的 token 列表

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:] # 获取序列的生成结果的 token 列表

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size # 计算已缓存的内存块数量

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size # 计算序列总共需要的内存块数量

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size # 计算最后一个内存块中 存储的token 数量

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size] # 从一组盒子中取出指定编号的盒子，然后查看盒子里装的token

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 # 在一个长列表的末尾添加一个新的元素，同时更新列表的长度和最后一个元素的值

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token) # 如果生成结果的 token 数量为 0，则保存整个 token 列表；否则，只保存最后一个 token

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1] # 如果生成结果的 token 数量为 0，则将元组的最后一个元素赋值给token 列表
        else:
            self.last_token = state[-1] # 否则，则将元组的最后一个元素赋值给最后一个token
