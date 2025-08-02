from collections import deque
import xxhash
import numpy as np

from lite_infer.engine.sequence import Sequence

# Block 类可以看作是内存中的一个个 “小房间”，
class Block:

    def __init__(self, block_id):
        self.block_id = block_id # 每个房间有一个唯一的编号 block_id
        self.ref_count = 0 # 这个 “房间” 被使用的次数，类似于房间的入住人数
        self.hash = -1 # 这个 “房间” 里内容的一个 “指纹”，用于快速识别内容是否相同
        self.token_ids = [] # “房间” 里实际存放的内容

    # 更新 “房间” 的 “指纹” 和内容
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 将 “房间” 恢复到初始状态
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

# BlockManager 类就像是一个 “酒店管理员”，负责管理所有的 “房间”（内存块）
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0 # 表示酒店里的房间总数
        self.block_size = block_size # 表示每个房间的大小
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 包含所有 “房间” 的列表。
        self.hash_to_block_id: dict[int, int] = dict() # 用于通过 “指纹” 快速找到对应的 “房间编号”，就像通过客人的身份证号找到对应的房间号
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 存储着所有空闲的 “房间编号”
        self.used_block_ids: set[int] = set() # 存储着所有已被使用的 “房间编号”

# 给 “房间” 里的内容生成一个唯一的 “指纹”
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64() # 哈希函数，用于生成哈希值
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little")) # 如果有 prefix，则先将其转换为字节流并更新到哈希函数中
        h.update(np.array(token_ids).tobytes()) # 将 token_ids 转换为字节流并更新到哈希函数中
        return h.intdigest() # 返回生成的哈希值
        # return -1

    # 该方法就像是 “酒店管理员” 将一个空闲的 “房间” 分配给客人
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id] # 进入当前房间
        assert block.ref_count == 0 # 检查这个 “房间” 是否空闲（ref_count == 0）
        block.reset() # 重置为初始状态
        self.free_block_ids.remove(block_id) # 从空闲 “房间” 列表中移除
        self.used_block_ids.add(block_id) # 添加到已使用 “房间” 集合中
        return self.blocks[block_id] # 返回这个 “房间”

    # 该方法是 “酒店管理员” 将一个已退房的 “房间” 标记为空闲
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0 # 检查这个 “房间” 是否空闲（ref_count == 0）
        self.used_block_ids.remove(block_id) # 从已使用 “房间” 集合中移除
        self.free_block_ids.append(block_id) # 添加到空闲 “房间” 列表中

    # 该方法是 “酒店管理员” 检查是否有足够的空闲 “房间” 来满足客人的需求
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks # 如果空闲 “房间” 的数量大于等于客人需要的 “房间” 数量，则返回 True，否则返回 False
    
    # 该方法是 “酒店管理员” 为客人分配一系列 “房间”
    def allocate(self, seq: Sequence):
        assert not seq.block_table # 检查客人是否已经有了 “房间安排表”（seq.block_table），如果有则不进行分配
        h = -1
        cache_miss = False
        # 遍历客人需要的每个 “房间”，计算其 “指纹”
        for i in range(seq.num_blocks):
            token_ids = seq.block(i) # 客人需要当前房间的内容
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # “客人需要房间” 实际存放内容的大小与房间的大小一致，则计算该房间的 “指纹”
            block_id = self.hash_to_block_id.get(h, -1) # 在 “指纹 - 房间编号” 字典中查找对应的 “房间编号
            # block_id = -1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True # 如果在 “指纹 - 房间编号” 字典中找不到对应的 “房间编号”，或者找到的 “房间” 里的内容与客人需要的内容不同，则发生 “缓存未命中”（cache_miss）
            if cache_miss:
                block_id = self.free_block_ids[0] # 如果发生 “缓存未命中”，则从空闲 “房间” 列表中取出一个 “房间” 分配给客人
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size # 更新累计已经缓存的内容大小
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id] # 如果缓存命中，且该房间已被使用，则进入该房间
                    block.ref_count += 1 # 将已有的 “房间” 的使用次数加 1
                else:
                    block = self._allocate_block(block_id) # 如果缓存命中，但该房间未被使用，则分配该房间
            if h != -1:
                block.update(h, token_ids) # 如果 “指纹” 不为 -1，则更新 “房间” 的 “指纹” 和内容
                self.hash_to_block_id[h] = block_id # 并将 “指纹 - 房间编号” 映射添加到字典中
            seq.block_table.append(block_id) # 将分配的 “房间编号” 添加到客人的 “房间安排表” 中

    # 该方法是 “酒店管理员” 为客人办理退房手续
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id] #进入当前房间
            block.ref_count -= 1 # 遍历客人的 “房间安排表”，将每个 “房间” 的使用次数减 1
            if block.ref_count == 0:
                self._deallocate_block(block_id) # 如果某个 “房间” 的使用次数变为 0，则将其标记为空闲
        seq.num_cached_tokens = 0 # 最后将客人的 “已缓存令牌数量” 清零
        seq.block_table.clear() # 并清空 “房间安排表”

    # 该方法是 “酒店管理员” 检查是否有足够的空闲 “房间” 来满足客人追加 “房间” 的需求
    def can_append(self, seq: Sequence) -> bool:
        # 如果空闲 “房间” 的数量大于等于客人是否需要追加 “房间” 的标志，则返回 True
        # len(self.free_block_ids): 统计空房间的数量
        # len(seq) 表示序列 seq 当前的长度，也就是客人存放内容的数量
        # len(seq) % self.block_size == 1时，说明客人存放内容的数量 刚好达到了一个新的房间起始位置，此时需要一个新的房间给后续内容存放
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
   
    # 该方法是 “酒店管理员” 根据客人的需求决定是否追加 “房间”
    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]] # 进入最后一个房间
        # 如果客人的 “存放内容量” 刚好达到一个新的 “房间” 的起始位置 (len(seq) % self.block_size == 1)
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1 # 确保最后一个房间已经被正确初始化，有有效的哈希值
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id) # 则从空闲 “房间” 列表中取出一个 “房间” 分配给客人
            block_table.append(block_id) # 并添加到 “房间安排表” 中
        elif len(seq) % self.block_size == 0: # 如果客人的 “使用量” 刚好填满一个 “房间”
            assert last_block.hash == -1 # 确保最后一个房间还没有被计算哈希值，处于未完成状态
            token_ids = seq.block(seq.num_blocks-1) # 获取序列最后一个房间中的 内容 列表
            # 如果房间列表中有多个房间，获取倒数第二个房间的哈希值作为前缀，就像在给最后一个房间贴指纹时，参考上一个房间的指纹信息
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix) # 计算最后一个 “房间” 的 “指纹”
            last_block.update(h, token_ids) # 根据房间的内容 更新 “房间” 的 “指纹” 信息
            self.hash_to_block_id[h] = last_block.block_id # 并将 “指纹 - 房间编号” 映射添加到字典中
        else:
            assert last_block.hash == -1
