import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from lite_infer.config import Config
from lite_infer.engine.sequence import Sequence
from lite_infer.models.qwen3 import Qwen3ForCausalLM
from lite_infer.layers.sampler import Sampler
from lite_infer.utils.context import set_context, get_context, reset_context
from lite_infer.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 我们先确定演出的规则和要求（配置信息），然后搭建舞台（初始化分布式进程组）
        # 安排演员就位（设置 CUDA 设备和数据类型）。接着邀请主演（加载模型）和配角（采样器），进行排练（模型预热），准备道具（分配 KV Cache）
        # 如果条件允许，我们还可以录制演出的模板（捕获 CUDA 图）以提高演出效率
        # 最后，设置好演员之间的通信方式（共享内存），并让非主角演员开始等待指令（启动循环）
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        torch.cuda.set_device(rank) # 设置 CUDA 设备
        default_dtype = torch.get_default_dtype() # 设置数据类型
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model) # 加载模型
        self.sampler = Sampler() # 加载采样器
        self.warmup_model() # 进行模型预热
        self.allocate_kv_cache() # 键值缓存（KV Cache）的分配
        if not self.enforce_eager:
            self.capture_cudagraph() # 如果不强制使用即时模式，则捕获 CUDA 图
        # torch.set_default_device("cpu")
        # torch.set_default_dtype(default_dtype)

    def exit(self):
        # 该方法用于在程序退出时清理资源，关闭共享内存，销毁 CUDA 图和图池，同步 CUDA 操作，并销毁分布式进程组;
        # 演出结束后，我们要清理舞台。关闭演员之间的通信通道（共享内存），销毁录制的模板（CUDA 图和图池），
        # 确保所有演员都完成了谢幕（同步 CUDA 操作），最后解散剧组（销毁分布式进程组）。
        if not self.enforce_eager:
            del self.graphs, self.graph_pool # 销毁 CUDA 图和图池
        torch.cuda.synchronize() # 同步 CUDA 操作
        dist.destroy_process_group() # 销毁分布式进程组

    def loop(self):
        # 该方法在非主进程中运行，不断从共享内存中读取任务信息，调用相应的方法执行任务，直到收到 exit 指令。
        # 就像是非主角演员在后台等待导演的指令。他们不断查看指令板（共享内存），根据指令进行表演（调用相应的方法），直到导演宣布演出结束（收到 exit 指令）
        while True:
            method_name, args = self.read_shm() # 从共享内存中读取任务信息
            self.call(method_name, *args) # 调用相应的方法执行任务
            if method_name == "exit":
                break # 直到收到 exit 指令

    def call(self, method_name, *args):
        # 调用指定的方法执行任务。如果是主进程且处于分布式环境中，将任务信息写入共享内存；否则，直接调用相应的方法。
        # 就像是导演安排演员的表演。如果是总导演（主进程），他会将表演安排写在指令板上（写入共享内存）让其他导演和演员查看；
        # 如果是其他导演或演员，他们会直接按照安排进行表演（调用相应的方法）
        method = getattr(self, method_name, None) # 否则，直接调用相应的方法
        return method(*args)

    def warmup_model(self):
        # 对模型进行预热，清除 CUDA 缓存和重置内存统计信息，生成一批测试序列，调用 run 方法进行推理，最后再次清除 CUDA 缓存
        # 就像是在演出前让演员进行热身运动。我们先清理舞台上的杂物（清除 CUDA 缓存），记录演员的体能极限（重置内存统计信息），
        # 然后安排演员进行一些简单的排练（生成测试序列并进行推理），
        # 最后再次清理舞台（清除 CUDA 缓存）为正式演出做好准备
        torch.cuda.empty_cache() # 清除 CUDA 缓存
        torch.cuda.reset_peak_memory_stats() # 重置内存统计信息
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len # 记录模型序列生成长度的极限和批数量的极限
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] # 生成一批测试序列
        self.run(seqs, True) # 调用 run 方法进行推理
        torch.cuda.empty_cache() # 再次清除 CUDA 缓存

    def allocate_kv_cache(self):
        # 首先获取 CUDA 内存信息，计算可用内存和每个缓存块的字节数，然后根据配置计算需要分配的缓存块数量。接着创建一个全零的张量作为 KV Cache，并将其分配给模型的各个模块
        # 先了解仓库的总容量（CUDA 内存信息），计算每个道具所需的空间（每个缓存块的字节数），
        # 然后根据演出的需求确定需要多少个道具（缓存块数量）。
        # 接着搭建一个空的仓库（创建全零的张量），将道具按照顺序分配给各个演员（将 KV Cache 分配给模型的各个模块）
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info() # 获取内存空间
        used = total - free # 计算已使用的内存空间
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] # 分配给CUDA的峰值内存值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 分配给CUDA的当前内存值
        num_kv_heads = hf_config.num_key_value_heads // self.world_size # KV head 数量
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize # 每个缓存块的字节数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes # 缓存块数量
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim) # 创建一个全零的张量作为 KV Cache
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 首先找出所有序列中块表的最大长度，然后将每个序列的块表填充到最大长度，最后将填充后的块表转换为张量并移动到 CUDA 设备上
        # 就像是为演出准备演员的道具清单。我们先找出所有演员道具清单中最长的长度，
        # 然后将每个演员的道具清单补充到相同的长度（用 -1 表示没有道具），
        # 最后将这些清单整理成电子表格（张量）并发送到演出场地（CUDA 设备）
        max_len = max(len(seq.block_table) for seq in seqs) # 找出所有序列中块表的最大长度
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs] # 将每个序列的块表填充到最大长度
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) # 将填充后的块表转换为张量并移动到 CUDA 设备上
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 为预填充阶段准备输入数据。
        # 遍历所有序列，提取输入 ID、位置信息、累积序列长度和槽映射信息。
        # 如果存在前缀缓存，则准备块表。
        # 最后将这些信息转换为张量并移动到 CUDA 设备上，设置上下文信息并返回输入 ID 和位置信息
        # 就像是为演出的开场部分准备演员的台词和站位。我们依次与每个演员沟通，记录他们的台词（输入 ID）、站位顺序（位置信息）、台词长度（累积序列长度）和道具摆放位置（槽映射信息）。
        # 如果有一些通用的道具摆放模板（前缀缓存），我们还需要准备相应的清单（块表）。
        # 最后将这些信息整理成电子表格（张量）并发送到演出场地（CUDA 设备），同时设置好演出的场景信息（上下文信息）。
        input_ids = [] # 所有序列的 token ID
        positions = [] # 每个 token 在其序列中的位置
        cu_seqlens_q = [0] # 累积序列长度Q
        cu_seqlens_k = [0] # 累积序列长度K
        max_seqlen_q = 0 # Q最大序列长度
        max_seqlen_k = 0 # K最大序列长度
        slot_mapping = [] # 映射每个 token 到其在 KV 缓存中的位置, 想象一个剧院的座位表，每个座位对应一个唯一编号。Slot mapping 就是将每个观众（token）分配到具体座位（KV 缓存位置）的过程
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:]) # 只处理尚未缓存的 token(从 seq.num_cached_tokens 开始)
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens # 当前需要处理的 token 数量
            seqlen_k = seqlen # 整个序列的长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # 记录当前序列的总长度
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # # 记录当前序列的总长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # seq.block_table = [] 时，跳过生成slot_mapping
            if not seq.block_table:
                continue
            # 构建 slot_mapping 数组，将 token 映射到 KV 缓存中的具体位置，遍历序列的每个块， 计算其在 KV 缓存中的起始和结束位置
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens # 最后一个块可能未满，因此使用 seq.last_block_num_tokens 确定实际长度
                slot_mapping.extend(list(range(start, end)))
        
        # 当前缀缓存存在时，准备块表以跟踪序列在缓存中的布局
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache， 如果K的累积长度大于Q的累积长度，说明存在前缀缓存
            block_tables = self.prepare_block_tables(seqs) # 生成一个二维数组，记录每个序列使用的块 ID
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True) # pin_memory=True 启用页面锁定内存，加速 CPU 到 GPU 的数据传输, 
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True) # non_blocking=True 允许异步数据传输
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # 为解码阶段准备输入数据。遍历所有序列，提取最后一个输入 ID、位置信息、槽映射信息和上下文长度信息。
        # 准备块表，将这些信息转换为张量并移动到 CUDA 设备上，设置上下文信息并返回输入 ID 和位置信息。
        # 就像是为演出的后续部分准备演员的台词和站位。
        # 我们依次与每个演员沟通，记录他们的下一句台词（最后一个输入 ID）、当前的站位顺序（位置信息）、道具摆放位置（槽映射信息）和当前的场景长度（上下文长度信息）。
        # 准备好道具清单（块表），将这些信息整理成电子表格（张量）并发送到演出场地（CUDA 设备），同时设置好演出的场景信息（上下文信息）
        input_ids = []
        positions = []
        slot_mapping = [] # 映射每个 token 到其在 KV 缓存中的位置
        context_lens = [] # 上下文长度信息
        # 遍历所有序列， 提取最后一个输入 ID、位置信息、槽映射信息和上下文长度信息
        for seq in seqs:
            input_ids.append(seq.last_token) # 将每个序列的最后一个token添加到输入 ID 列表中
            positions.append(len(seq)) # 将每个序列的当前长度添加到位置信息列表中
            context_lens.append(len(seq)) # 将每个序列的上下文长度添加到上下文长度列表中
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1) # 计算每个序列最后一个块的最后一个token在KV缓存中的位置，并添加到槽映射列表中
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables) # 设置上下文信息
        return input_ids, positions # 返回输入ID 和位置信息

    def prepare_sample(self, seqs: list[Sequence]):
        # 为采样过程准备温度参数。遍历所有序列，提取温度参数，将其转换为张量并移动到 CUDA 设备上。
        # 就像是为演出的表演风格调整温度。我们依次询问每个演员的表演热情（温度参数），将这些热情程度记录下来（提取温度参数），
        # 然后将其整理成电子表格（张量）并发送到演出场地（CUDA 设备），以便控制表演的风格
        temperatures = []
        # 遍历所有序列
        for seq in seqs:
            temperatures.append(seq.temperature) # 提取温度参数
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 运行模型进行推理。如果是预填充阶段、强制使用即时模式或批量大小大于 512，则直接调用模型进行推理；否则，使用捕获的 CUDA 图进行推理
        # 就像是演员进行表演。如果是开场表演（预填充阶段）、需要立即表演（强制使用即时模式）或演员数量太多（批量大小大于 512），演员会按照剧本现场表演（直接调用模型进行推理）；
        # 否则，演员会按照之前录制的模板进行表演（使用捕获的 CUDA 图进行推理）
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 预填充阶段、强制使用即时模式或批量大小大于 512，则直接调用模型进行推理
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 否则，使用捕获的 CUDA 图进行推理
            bs = input_ids.size(0)
            context = get_context() # 获取上下文信息
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)] # 获取计算图
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 执行推理任务。根据是否为预填充阶段准备输入数据，准备采样温度参数，调用 run_model 方法进行推理，
        # 使用采样器从推理结果中采样得到 token ID，重置上下文信息并返回 token ID
        # 我们根据演出的阶段（是否为预填充阶段）为演员准备台词和站位（准备输入数据），调整表演的热情（准备采样温度参数），让演员进行表演（调用 run_model 方法进行推理），
        # 根据表演效果选择合适的台词（使用采样器采样得到 token ID），
        # 最后清理演出场地（重置上下文信息）并公布演出结果（返回 token ID）
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs) # 准备 输入ID、位置信息
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None # 准备采样温度参数
        logits = self.run_model(input_ids, positions, is_prefill) # 运行模型
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None # 使用采样器采样得到 token ID
        reset_context() # 重置上下文信息
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        # 捕获 CUDA 图以提高推理效率。首先初始化输入和输出张量，定义不同批量大小的列表，然后遍历每个批量大小，创建 CUDA 图，进行热身和图捕获操作，最后保存图和相关变量
        # 就像是为演出录制模板。我们先准备好演出所需的道具和场景（初始化输入和输出张量），确定不同规模演出的模板（定义不同批量大小的列表）。
        # 然后依次对每个规模的演出进行排练（热身），并录制排练过程（捕获 CUDA 图）。
        # 最后将录制好的模板保存起来（保存图和相关变量），以便在正式演出时可以直接使用
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 遍历每个批量大小， 创建 CUDA 图
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs]) # 设置上下文信息
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
