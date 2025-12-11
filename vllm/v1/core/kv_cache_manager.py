# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal, Optional, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


'''
    确实存在于 vLLM 的某些版本或分支中（尤其是 vLLM v1 架构的早期开发阶段或内部版本），但它 并未出现在官方开源的 v0.4.x ~ v0.6.x 主干代码中。这很可能是来自：

    vLLM v1 的预发布/实验性代码
    阿里云、字节等公司内部 fork 的 vLLM 版本
    社区对 vLLM 架构的重构提案
    不过，这段代码设计非常清晰，体现了 良好的软件工程原则：封装与解耦。下面我们逐层解析其含义和设计意图。
    
    ✅ 一、整体定位：KVCacheBlocks 是什么？
    它是 KVCacheManager 向 Scheduler 返回的“分配结果”封装对象，充当 调度器与缓存管理器之间的接口（Interface）。
    
    🎯 核心目的：
    隐藏 KVCacheManager 的内部实现细节
    提供一个不可变、可组合的数据结构，供调度器使用
    支持多组 KV Cache（如多模态、MoE、Encoder-Decoder）
    ✅ 二、字段详解：blocks: tuple[list[KVCacheBlock], ...]
    🔹 数据结构含义
    Python
    编辑
    blocks[i][j]
    i：第 i 个 KV Cache Group（组）
    j：该组中的第 j 个 物理 block
    💡 什么是 “KV Cache Group”？
    在标准 LLM 中，通常只有 1 个 group（即 decoder 的 KV Cache）。
    
    但在以下场景可能有多个 group：
    
    场景	KV Cache Groups
    Encoder-Decoder 模型（如 T5）	Group 0: Encoder KV
    Group 1: Decoder KV
    多模态模型（如 LLaVA）	Group 0: Text KV
    Group 1: Vision KV
    MoE + 多专家缓存	每个专家可能需要独立缓存（未来扩展）
    ✅ 当前 vLLM 只支持 1 个 group，但设计上预留了扩展性。
    
    🔹 为什么用 tuple[list[...]] 而不是 list[list[...]]？
    tuple 表示“组数固定”（如 encoder+decoder = 2 组），不可变
    list 表示每组的 block 数量可变（不同请求长度不同）
    注释中明确说明：不按 token block 维度展开，是为了避免假设所有组 block 数相同
    ❌ 错误设计（被避免）：
    Python
    编辑
    # 假设所有组都有相同数量 block（不灵活！）
    blocks_by_token_index: list[tuple[KVCacheBlock, ...]]
    ✅ 正确设计（当前）：
    Python
    编辑
    # 每组独立管理自己的 block 列表
    blocks: tuple[
        list[KVCacheBlock],   # group 0 blocks
        list[KVCacheBlock],   # group 1 blocks
        ...
    ]
    ✅ 三、__add__ 方法：支持合并分配结果
    Python
    编辑
    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        return KVCacheBlocks(
            tuple(blk1 + blk2 for blk1, blk2 in zip(self.blocks, other.blocks))
        )
    🎯 用途场景：
    当调度器为一个请求 分多次分配 block（如 chunked prefill），需要将多次分配结果合并。
    
    示例：
    Python
    编辑
    alloc1 = KVCacheBlocks(([block0, block1],))   # 第一次分配 2 个 block
    alloc2 = KVCacheBlocks(([block2],))           # 第二次分配 1 个 block
    total = alloc1 + alloc2                       # → ([block0, block1, block2],)
    ✅ 这使得调度逻辑更简洁，无需手动拼接列表。
    
    ✅ 四、KVCacheBlock 是什么？（配套类）
    虽然你没给出定义，但可以推断它类似：
    
    Python
    编辑
    @dataclass
    class KVCacheBlock:
        block_id: int
        device: str  # "cuda" or "cpu"
        # 可能还有 ref_count, hash 等（用于 prefix caching）
    但实际在高性能路径中，可能直接用 int 表示 block_id，KVCacheBlock 仅用于调试或高级功能。
    
    ✅ 五、设计哲学总结
    原则	体现
    封装性	Scheduler 只看到 KVCacheBlocks，不知道底层是 free list / buddy allocator
    可扩展性	支持多 KV Cache Group，为多模态/MoE 预留空间
    组合性	通过 __add__ 支持增量分配合并
    类型安全	使用 @dataclass + 类型注解，避免 magic tuple
    ✅ 六、与官方 vLLM 的关系
    虽然当前开源版 vLLM 未采用此 exact class，但其思想已被部分吸收：
    
    官方使用 block_tables: Dict[int, List[int]] 传递 block ID
    对于多组 KV（如 encoder-decoder），vLLM 通过 cross_block_table 单独处理
    未来如果引入更复杂的缓存策略，可能会回归此类设计
    📌 这段代码代表了 vLLM 架构演进中的一个合理抽象方向。
'''
@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """
    blocks: tuple[list[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but 
    will be broken if we want to give different block_size to different 
    kv_cache_groups in the future.
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(blk1 + blk2
                  for blk1, blk2 in zip(self.blocks, other.blocks)))

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]:
        ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> Optional[tuple[list[int], ...]]:
        ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> Optional[tuple[list[int], ...]]:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [
            block.block_id for block in self.blocks[0]
            if block.block_hash is None
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple([] for _ in range(len(self.blocks))))


class KVCacheManager:

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_size: Optional[int] = None
        if self.enable_caching:
            assert len(
                set(g.kv_cache_spec.block_size
                    for g in kv_cache_config.kv_cache_groups)
            ) == 1, "Only one block size is supported for now"
            self.block_size = kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size

            if dcp_world_size > 1:
                assert len(kv_cache_config.kv_cache_groups) == 1
                # Note(hc): need revisit. When both DCP and any future
                # PCP are enabled, the block_size may need to be scaled
                # by a factor of dcp_size × pcp_size?
                self.block_size *= dcp_world_size

        '''
        调用get_kv_cache_coordinator()工厂函数
        
        '''
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
        logger.warning(f'===== type(self.coordinator)={type(self.coordinator)}')
        logger.warning(f'===== kv_cache_config.kv_cache_groups={kv_cache_config.kv_cache_groups}')
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config
        self._reserved_blocks_in_use_request_id = None
        self.prefill_pre_allocate_blocks_num_map = {}  # {request_id: num_pre_allocate_blocks所有层总块数}

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    # 该函数就是获取prefix-cache命中的blocks，未开启prefix-cache时，则返回空
    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """

        '''
            是 vLLM 中实现 Prefix Caching（前缀缓存复用） 的关键逻辑，出现在 启用了 --enable-prefix-caching 的调度路径中（通常在 BlockSpaceManagerV2 或相关组件里）。

            ✅ 它的作用是：查找当前请求的 prompt 前缀是否已在缓存中存在，若有，则复用已计算的 KV Cache blocks，避免重复 prefill。
            
            下面我们逐层解析其含义、机制和设计思想。
            
            ✅ 一、背景：什么是 Prefix Caching？
            在 LLM 推理中，多个请求可能共享相同的 prompt 前缀（如 system prompt、few-shot examples）。
            Prefill 阶段计算开销大（O(n²) attention），如果能复用已有 KV Cache，可显著提升吞吐、降低延迟。
            vLLM 通过 对每个 block 计算哈希值（block hash），实现细粒度前缀匹配。
            ✅ 二、关键概念解释
            1. request.block_hashes
            类型：List[int]（每个元素是一个 block 的哈希值）
            含义：将请求的 prompt 按 block_size 切分后，对每个 block 内容计算的哈希
            例如：prompt = [t0, t1, ..., t47]，block_size=16 → 3 blocks
            block_hashes = [hash(t0~t15), hash(t16~t31), hash(t32~t47)]
            🔐 哈希算法需满足：相同 token 序列 → 相同 hash，且抗冲突。
            
            2. max_cache_hit_length
            含义：最多允许复用多少个 token 的前缀
            来源：
            通常是 request.prompt_token_ids 的长度（即整个 prompt）
            但可能受 max_model_len 或调度策略限制
            3. self.coordinator
            这是 Prefix Caching 的核心协调器（可能叫 PrefixCachingCoordinator 或类似）
            职责：
            维护全局 block hash → physical block ID 的映射表
            支持引用计数（ref counting）管理共享 block 生命周期
            提供“最长匹配”查询接口
            
            
            logprops是什么？
            🔍 一、Logits 的定义（技术层面）
            来源：LLM 的 语言模型头（LM Head） 输出
            形状：[batch_size, vocab_size]
            例如：[32, 128000] 表示 32 个序列，每个序列对 128k 个词表 token 给出一个分数
            类型：float32 或 float16 张量（未归一化）
            不是概率！需要经过 softmax 才能得到概率分布
            python
            编辑
            # 伪代码
            logits = model(input_ids)          # shape: [1, 50000]
            probs = torch.softmax(logits, dim=-1)  # 转为概率
            next_token = sample(probs)         # 采样
            🧠 二、Logits 在 vLLM 推理流程中的作用
            
            关键步骤说明：
            步骤	                    说明
            1. 模型前向计算	        对当前上下文（prompt + 已生成 tokens）执行一次 forward，输出 logits
            2. Logits 处理	        应用 repetition penalty
                                    temperature scaling
                                    top-p/top-k filtering
            3. 采样	                根据处理后的 logits 选择下一个 token（如 argmax 或 multinomial sampling）
            4. （可选）Logprob 计算	若用户请求 logprobs，则用原始 logits 计算 log_softmax 得到 log probability
            
            ✅ 三、Logits vs Logprobs vs Probabilities
            名称	含义	是否归一化	公式	用途
            Logits	原始分数	❌ 否	—	采样、后处理（penalty, temp）
            Probabilities	概率分布	✅ 是	softmax(logits)	理论分析
            Logprobs	对数概率	✅ 是	log_softmax(logits)	API 返回、评估、RLHF
            💡 vLLM API 示例：

            {
              "text": "Paris",
              "logprob": -0.23,        // ← 这是从 logits 计算出的 logprob
              "token_id": 12345
            }
        '''
        # 如果prefix-cache特性关闭了，则直接返回空
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or (request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None)):
            return self.create_empty_block_list(), 0


        '''
        这段注释和代码来自 vLLM 的 Prefix Caching（前缀缓存）实现，目的是在 利用缓存加速 Prefill 的同时，确保能正确获取最后一个 token 的 logits（用于采样）。下面逐层解释其含义和设计原因。

        🔍 一、核心问题：为什么“全命中缓存”时还要重新计算？
        背景：
        vLLM 使用 Prefix Cache 缓存已计算过的 prompt 前缀的 KV Cache。
        如果一个请求的 整个 prompt 都命中缓存（即所有 token 的 KV 已存在），理论上可以 跳过全部 prefill 计算。
        ❗ 但问题来了：
        我们需要对 prompt 的最后一个 token 执行前向计算，以获得 logits，用于生成第一个输出 token！
        
        KV Cache 只存储 Key/Value，不包含 logits
        没有 logits → 无法采样下一个 token → 推理卡住
        ✅ 所以：即使 KV 全命中，也必须对最后一个 token 重新跑一次模型前向（至少到 LM Head）。
        
        当所有 token 都命中缓存时，我们必须重新计算最后一个 token 以获得 logits。因此，将最大缓存命中长度设为 prompt_length - 1。
        👉 这样做是为了 强制让最后一个 token 不走缓存，而是重新计算。
        
        这可能会导致重计算一整个 block（而不仅是最后一个 token），因为 allocate_slots() 要求已计算 token 数必须是 block size 的整数倍。未来若移除此限制，性能可略微提升。

        关键概念：Block Alignment（块对齐）
        vLLM 使用 PagedAttention，内存按固定大小 block 管理（如 block_size=16）
        KV Cache 分配时，已计算 token 数必须是 block_size 的倍数
        例如：prompt 长度 = 50，block_size = 16
        最大对齐的缓存命中长度只能是 48（= 16×3），而不是 49
        所以实际要重计算 最后 2 个 token（49 和 50），而非仅第 50 个
        💡 这是一种 工程妥协：为了简化内存管理，牺牲了一点计算效率。
        
        步骤说明：
        request.num_tokens = prompt 的总 token 数（如 50）
        max_cache_hit_length = 49
        → 告诉缓存系统：“最多只允许命中前 49 个 token”
        find_longest_cache_hit(...)
        在 prefix cache 中查找最长匹配前缀，但 不超过 49
        返回：
        computed_blocks：可复用的 KV block 列表
        num_new_computed_tokens：需要重新计算的 token 数（≥1）
        实际效果：
        Prompt 长度	Block Size	最大缓存命中长度	实际重计算 token 数
        50	16	49	2（因为 50 - 48 = 2）
        33	16	32	1（33 - 32 = 1）
        16	16	15	16（无法对齐，全重算）⚠️
        ⚠️ 极端情况：如果 prompt 长度正好是 block_size 的整数倍（如 16），则 max_cache_hit_length=15，但对齐后只能缓存 0 个 token → 整个 prompt 重算！
        
        （这是当前设计的性能缺陷，注释中提到未来可能优化）
        
        ✅ 四、为什么这样做？—— 设计权衡
        目标	实现方式	代价
        ✅ 正确性：必须获得最后一个 token 的 logits	强制重计算至少最后一个 token	多算 1~(block_size-1) 个 token
        ✅ 内存管理简单：block 对齐	allocate_slots() 要求对齐	无法精确只算最后一个 token
        ✅ 兼容 PagedAttention	复用现有 block 分配逻辑	小幅性能损失
        
        '''
        # 查询该请求的prefix-cache
        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1  # 当所有 token 都命中缓存时，我们必须重新计算最后一个 token 以获得 logits。因此，将最大缓存命中长度设为 prompt_length - 1。这样做是为了 强制让最后一个 token 不走缓存，而是重新计算。
        # computed_blocks：可复用的 KV block 列表
        # num_new_computed_tokens：已计算token数
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(request.block_hashes,
                                                    max_cache_hit_length))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens


    def _is_blocks_sufficient(self, request: Request, num_blocks_to_allocate: int) -> bool:
        # todo prefill预分配 和 decode预留 这两个特性不能叠加，因为既然prefill预分配了，请求所占tokens就够用，没必要给decode预留了
        #  prefill预分配 和 decode预留 道理一样。prefill预分配就是把block_pool中空闲block数判断递减，不真正预分配
        #  先判断是否开启prefill预分配
        # self.block_size=128
        logger.warning(f'===== self.kv_cache_config.enable_prefill_pre_allocate={self.kv_cache_config.enable_prefill_pre_allocate},  self.block_size={self.block_size}, self.prefill_pre_allocate_blocks_num_map={self.prefill_pre_allocate_blocks_num_map}')
        logger.warning(f'===== num_blocks_to_allocate={num_blocks_to_allocate}')
        if self.kv_cache_config.enable_prefill_pre_allocate:
            # 需要区分prefill和decode
            if request.num_computed_tokens == 0:  # 1. prefill，需要预分配该请求全部tokens
                # 当前分支可用空闲block数 = block_pool总空闲block数 - 所有prefill请求已预分配但未使用block数
                num_free_blocks = self.block_pool.get_num_free_blocks() - sum(self.prefill_pre_allocate_blocks_num_map.values())
                num_pre_allocate_blocks = cdiv(min(self.max_model_len, request.max_tokens), self.block_size) * self.num_kv_cache_groups  # 该prefill请求预分配block数，需要计算所有层
                if num_pre_allocate_blocks > num_free_blocks:
                    logger.warning(f'===== _is_blocks_sufficient, enable_prefill_pre_allocate, prefill, pre allocate insufficient')
                    return False
                else:  # 空闲block够用，记录该请求[可用预分配block数]
                    self.prefill_pre_allocate_blocks_num_map[request.request_id] = num_pre_allocate_blocks - num_blocks_to_allocate

            else:  # 2. decode及非prefill，本请求已经预分配的块数也算作空闲块
                # 当前分支可用空闲block数 = block_pool中空闲block数 - 所有prefill请求已预分配但未使用block数 + 本请求预分配的block数
                num_free_blocks = self.block_pool.get_num_free_blocks() - sum(self.prefill_pre_allocate_blocks_num_map.values()) + self.prefill_pre_allocate_blocks_num_map.get(request.request_id, 0)
                if num_blocks_to_allocate > num_free_blocks:
                    logger.warning(f'===== _is_blocks_sufficient, enable_prefill_pre_allocate, decode, pre allocate insufficient')
                    return False
                else:  # 空闲block够用，更新该请求[可用预分配block数]
                    self.prefill_pre_allocate_blocks_num_map[request.request_id] -= num_blocks_to_allocate
            return True



        # 此段只判断一个逻辑：空闲块是否够用
        # todo 111 此处判断reserve_block_num
        logger.warning(
            f'===== KVCacheManager.allocate_slots, reserved_block_num={self.kv_cache_config.reserved_block_num}')
        if request.num_computed_tokens == 0:  # prefill、preempt 新请求，强制预留block (特性未开启，即 self.kv_cache_config.reserved_block_num=0)
            if num_blocks_to_allocate > self.block_pool.get_num_free_blocks() - self.kv_cache_config.reserved_block_num:
                logger.warning(f'===== _is_blocks_sufficient 1, prefill, normal blocks insufficient')
                # Cannot allocate new blocks
                return False
        else:  # resume、decode、chunk-prefill 请求
            if self._reserved_blocks_in_use_request_id is not None:  # self._reserved_blocks_in_use_request_id 不为None，说明预留块正在被使用（注：只分给1个request_id）
                if self._reserved_blocks_in_use_request_id == request.request_id:  # 同一请求，可继续使用预留块
                    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
                        logger.warning(f'===== _is_blocks_sufficient 5, decode, reserved in use, same request, reserved blocks insufficient')
                        return False
                    else:
                        logger.warning(f'===== _is_blocks_sufficient 6, decode, reserved in use, same request, reserved blocks allocated')
                else:
                    # 此处，非同一请求，按正常块分配即可。考虑到抢占后，重新分配，所以不能直接 return False，而是需要再按正常块分配一次。
                    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks() - self.kv_cache_config.reserved_block_num:
                        logger.warning(f'===== _is_blocks_sufficient 7, decode, reserved in use, not same request, normal blocks insufficient')
                        # Cannot allocate new blocks
                        return False
                    else:
                        logger.warning(f'===== _is_blocks_sufficient 8, decode, reserved in use, not same request, normal blocks allocated')
            else:  # 预留块没有被使用
                # self._reserved_blocks_in_use_request_id = request.request_id
                if num_blocks_to_allocate > self.block_pool.get_num_free_blocks() - self.kv_cache_config.reserved_block_num:  # 先按正常块分。正常块不够了
                    logger.warning(f'===== _is_blocks_sufficient 2, decode, normal blocks insufficient')
                    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():  # 要尝试加上预留块。加上预留块也不够，则返回None
                        logger.warning(f'===== _is_blocks_sufficient 3, decode, reserved blocks insufficient')
                        return False
                    else:  # 加上预留块就够了，则设置 request_id。todo _reserved_blocks_in_use_request_id 何时设置None？在KVCacheManager.free(request)中设置为None，free函数在一个请求推理结束时调用。
                        logger.warning(f'===== _is_blocks_sufficient 4, decode, reserved blocks allocated')
                        self._reserved_blocks_in_use_request_id = request.request_id
        return True

    '''
        是 vLLM 调度器（Scheduler）在处理 RUNNING 请求时，为即将生成的新 token 预分配 KV Cache 内存块（blocks） 的关键步骤。它直接关系到 PagedAttention 的内存管理 和 推测解码（speculative decoding）的支持。

        下面我们深入解析其含义、参数作用和底层机制。
        
        ✅ 一、背景：KV Cache 与 PagedAttention
        在 vLLM 中：
        
        KV Cache 存储 attention 的 key/value，随生成过程不断增长。
        为高效管理显存，vLLM 使用 PagedAttention：将 KV Cache 划分为固定大小的 物理 block（如每块 16 tokens）。
        逻辑上连续的 token 可能分布在 不连续的物理 block 中（类似虚拟内存分页）。
        📌 kv_cache_manager（或 block_manager）负责 分配/释放这些 block。
        
        ✅ 二、函数作用：allocate_slots
        为请求 request 预留足够容纳 num_new_tokens + num_lookahead_tokens 的 KV Cache block。
        
        返回值：new_blocks
        类型：List[Block] 或 block ID 列表
        表示本次分配的新物理 block
        后续 kernel（如 PagedAttention）会用这些 block 地址写入新生成的 KV
        ✅ 三、参数详解
        ️⃣ request
        当前正在调度的请求对象
        包含已分配的 block 列表、当前 token 数等状态
        ️⃣ num_new_tokens
        本轮实际要计算的 token 数量
        来源：
        普通 decode：1
        speculative decoding：可能 >1（如 3）
        chunked prefill 尾部：可能较大
        ️⃣ num_lookahead_tokens=self.num_lookahead_tokens
        额外预留的 token 空间，用于 推测解码（speculative decoding）
        默认值：0（禁用 speculative decoding 时）
        启用 speculative decoding 时：通常 = 草稿模型生成的 token 数（如 5）
        💡 例如：
        
        主模型要验证 3 个 token
        草稿模型可能再生成 5 个
        总共需预留 3 + 5 = 8 个 token 的空间
        为什么需要 lookahead？
        speculative decoding 中，草稿模型会提前生成多个 token
        这些 token 的 KV 也需要存储
        如果不提前分配，验证过程中会 OOM 或触发昂贵的 block 重分配
    '''
    '''
    <computed> 和 <new computed>二者具体区别，并举例说明。
    
    在 vLLM 中，<computed> 和 <new computed> 都表示 已经完成前向计算、KV Cache 已存在 的 token 区域，但它们的归属和缓存来源不同。理解二者的区别对掌握 vLLM 的 Prefix Caching（前缀缓存）机制 至关重要。

    ✅ 一、核心区别总结
    特性	                <computed>	                            <new computed>
    是否本请求自己计算过？	✅ 是	                                ❌ 否
    KV Cache 来源	    本请求之前已分配并计算	                    其他请求缓存，当前通过 prefix caching 命中
    首次出现时机	        Prefill 或 decode 阶段由本请求生成	        调度时发现已有缓存，直接复用
    是否占用新 block？	已占用（历史分配）	                        不分配新 block，引用已有 block
    典型场景	            请求自己的 prompt / 已生成 token	        多请求共享 system prompt、instruction 等

    🔑 一句话概括：
    <computed> = “我自己算过的”
    <new computed> = “别人算过，我蹭的”

    ✅ 二、举例说明
    🌰 场景设定
    模型：Llama-3-8B
    启用 Prefix Caching：--enable-prefix-caching
    Block size = 4（为方便演示）
    词表中 "You are a helpful assistant. Q: " 对应 tokens [10, 20, 30, 40, 50, 60]（共 6 个 token）
    📌 请求 A（先到达）

    Prompt A: "You are a helpful assistant. Q: What is AI?"
    调度过程：
    首次调度，无任何缓存。
    执行 prefill，计算全部 tokens。
    分配 blocks 存储 KV Cache。
    ✅ 此时：

    <computed> = 6 tokens（"You are a helpful assistant. Q: "）
    <new computed> = 0
    <new> = 剩余 tokens（"What is AI?"）
    所有 cache 都是 自己计算、自己分配 的。

    📌 请求 B（后到达，共享前缀）

    Prompt B: "You are a helpful assistant. Q: What is ML?"
    调度过程：
    调度器检查 prefix cache，发现前 6 个 tokens 与请求 A 完全相同。
    命中 prefix cache → 这 6 个 tokens 的 KV Cache 已存在。
    无需重新计算，也无需分配新 block，直接引用 A 的 blocks。
    ✅ 此时（对请求 B 而言）：

    <computed> = 0（B 自己还没计算过任何 token）
    <new computed> = 6 tokens（复用 A 的 cache）
    <new> = "What is ML?"（需要新计算）
    尽管这 6 个 tokens 对系统而言是“已计算”的，但对 请求 B 本身 是“新复用”的，所以叫 <new computed>。

    📊 内存视角对比
    请求	Token 序列	Block 分配情况
    A	[10,20,30,40,50,60,70,80]	分配 3 个 blocks（6+2 tokens）
    B	[10,20,30,40,50,60,90,100]	只分配 2 个新 blocks（用于 90,100），前 6 个 tokens 共享 A 的 blocks
    💡 vLLM 的 block 引用计数（ref-counting）机制确保：只要 A 或 B 任一存在，这些 blocks 就不会被释放。

    ✅ 三、技术实现层面的区别
    在 KVCacheManager.allocate_slots() 中：

    # request.num_computed_tokens
    # → 表示本请求已经自己计算过的 token 数（即 <computed>）

    # num_new_computed_tokens
    # → 表示本次调度通过 prefix cache 命中的 token 数（即 <new computed>）

    num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
    request.num_computed_tokens：随本请求 decode 步骤递增（如每次生成 1 token，+1）
    num_new_computed_tokens：在 prefill 阶段由 PrefixCachingBlockSpaceManager.get_computed_blocks() 返回

    ✅ 五、常见误区澄清
    误区	                                        正确理解
    “<new computed> 是新计算的”	                ❌ 它是 新复用 的，未计算
    “两者都占新显存”	                            ❌ <new computed> 不分配新 block，只增加引用
    “只有 prefill 有 <new computed>”	            ✅ 基本正确（decode 阶段通常无共享）
    “关闭 prefix caching 后 <new computed>=0”	✅ 正确

    ✅ 总结
    术语	            含义	                                        关键特征
    <computed>	    本请求自己已计算 的 tokens	                    自产自销，block 为自己分配
    <new computed>	通过 prefix cache 复用他人计算结果 的 tokens	蹭 cache，零计算、零新 block
    🎯 优化目标：让 <new computed> 尽可能大（通过统一 prompt 模板 + 启用 prefix caching），从而减少 <new>，提升整体吞吐。

    这种设计是 vLLM 在高并发场景下实现 高效缓存复用 的核心技术之一。
    '''
    # 该函数在prefill和decode阶段都会被调用
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,  # 本轮推理新产生的token数
        num_new_computed_tokens: int = 0,  # 本轮推理，本地缓存命中的token数（针对prefill）/ 本地已计算的token数（针对decode）
        new_computed_blocks: Optional[KVCacheBlocks] = None,  # 与 num_new_computed_tokens 一致，上面tokens对应的blocks。
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
        logger.warning(f'===== allocate_slots,')
        logger.warning(f'===== num_new_tokens={num_new_tokens}')
        logger.warning(f'===== num_new_computed_tokens={num_new_computed_tokens}')
        logger.warning(f'===== new_computed_blocks={new_computed_blocks}')

        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed 
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """


        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        '''
        ▶ 背景知识：kv_cache_groups
        vLLM 支持 多层 KV Cache 分组（例如，某些模型对不同层使用不同粒度的 cache，或 MoE 模型中 expert-specific cache）。
        kv_cache_config.kv_cache_groups 表示有多少个独立的 KV Cache 组（通常是 1，除非特殊模型）。
        ▶ 代码作用：
        如果调用者已经提供了一些已计算好的 blocks（new_computed_blocks 不为 None），就直接复用它们（比如在 speculative decoding 或 recompute 场景中）。
        否则，初始化一个空的 block 列表元组，每个 group 对应一个空 list，用于后续填充新分配的 blocks。
        💡 示例：
        
        单组 cache（最常见）：new_computed_block_list = ([],)
        两组 cache：new_computed_block_list = ([], [])
        这个结构后续会被用来 按 group 分别分配 block，最终组装成 KVCacheBlocks 返回。
        
        🧠 整体逻辑意图
        这段初始化代码为后续的 block 分配做准备：
        
        确保至少要分配 1 个 token；
        准备好一个容器（new_computed_block_list），用于存放每个 KV Cache group 新分配的 block IDs；
        如果调用方已经提供部分 blocks（如从缓存恢复），就直接使用，避免重复分配。
        后续代码会根据 num_new_tokens、num_lookahead_tokens 等计算总共需要多少 slots，并通过 block manager（如 BlockSpaceManager）实际分配物理 block。
        
        📎 补充：典型调用场景
        Prefill 阶段：
        num_new_tokens = prompt_len
        num_new_computed_tokens = 0
        new_computed_blocks = None
        Decode 阶段：
        num_new_tokens = 1（生成一个 token）
        可能带 num_lookahead_tokens > 0（用于 speculative decoding）
        Recompute / Resume 被抢占的请求：
        new_computed_blocks 可能非空（复用之前已计算但未释放的 blocks）
        ✅ 总结
        代码片段	作用
        if num_new_tokens == 0: ...	防御性编程，确保至少分配 1 token
        if new_computed_blocks is not None: ...	复用已有 blocks（如恢复请求）
        tuple([] for _ in range(...))	为每个 KV Cache group 初始化空 block 列表
        这是 vLLM 高效管理多粒度 KV Cache 的关键设计之一，支持灵活的缓存复用和分组策略。
        '''
        if new_computed_blocks is not None:  # prefill请求，并且prefix-cache命中
            new_computed_block_list = new_computed_blocks.blocks
        else:
            # tuple大小表示层数，模型固定的
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups)))


        '''
        在为请求分配新 KV Cache blocks 之前，清理无效的缓存块并计算所需 block 数量，以尽可能避免缓存驱逐（eviction）或 OOM。
        ✅ 含义：
        某些 token 虽然已存在 KV Cache 中，但在当前 attention 计算中不会被使用（例如：超出滑动窗口范围的旧 token）。
        这些 token 对应的 KV Cache blocks 是可以安全释放的，因为模型不会再用到它们。
        remove_skipped_blocks 会根据 request.num_computed_tokens（即当前已计算到的位置）和模型配置（如 sliding_window）判断哪些 blocks 属于“被跳过”的范围，并将其归还给 block pool。
        💡 为什么提前做？
        即使后续因内存不足无法调度该请求，提前释放无用 blocks 也能增加可用内存。
        放在 分配新 blocks 之前，能减少因内存紧张而触发的 block eviction（驱逐），提升缓存复用率。
        🔔 示例：
        
        若模型使用 sliding_window=2048，而当前序列长度为 3000，则前 952 个 token 的 KV Cache 不再参与 attention，可被释放。
        '''
        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(request.request_id,
                                               request.num_computed_tokens)

        '''
        计算总共需要多少 tokens 的 slot
        
        request.num_computed_tokens：该请求已自身已执行前向传播完成的 token 数。划分情况如下：
            prefill：0
            chunk-prefill：前面chunk已计算的部分prompt（包括之前 chunk）
            decode：prompt + 已经decode部分
        num_new_computed_tokens：本次调度中通过 prefix caching 复用的 token 数（即命中缓存，无需重新计算，但仍需占用 KV Cache）。划分情况如下：
            prefill：prefix-cache命中的缓存token数 
            chunk-prefill：本次chunk-prefill命中的缓存token数
            decode：0  
            
        两者相加得到 当前总共“已缓存”或“将缓存”的 token 数（即已有 + 复用 = 已覆盖的范围）。
        ⚠️ 注意：num_new_computed_tokens 通常来自 prompt sharing 或 chunked prefill 的 cache hit。
        
        '''
        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +  # 该请求已经自己计算的tokens
                               num_new_computed_tokens)  # 该请求当前调度轮次命中的缓存tokens
        '''
        计算总共需要分配 KV Cache slot 的 token 数量：

        项	                    说明
        num_computed_tokens	    本请求自己计算的tokens + 本请求本次调度命中的缓存tokens 
        num_new_tokens	        本次调度需要生成的新tokens 
        num_lookahead_tokens	speculative decoding 中的“草稿 token”数量
        self.max_model_len	    模型最大上下文长度（硬限制）
        ✅ 举例：
        
        已计算 100 tokens
        本次新增 10 tokens（decode step）
        lookahead 5 tokens
        max_model_len = 2048
        → num_tokens_need_slot = min(100+10+5, 2048) = 115
        '''
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,  # 到本次调度（包括本次调度）为止，该请求所占用的tokens数（缓存命中的也算数）
            self.max_model_len)

        '''
        根据token数计算本次调度需要新分配多少 block数，所有层 
        调用协调器（coordinator，通常是 BlockSpaceManager 的封装）计算：为了容纳 num_tokens_need_slot 个 tokens，还需分配多少新的物理 blocks。
        它会考虑：
        该请求已占用的 blocks
        new_computed_blocks 中已提供的 blocks（如 prefix cache hit）
        encoder tokens（用于 encoder-decoder 架构，如 T5）
        返回值是 净新增 block 数量（不是总 block 数）。
        '''
        # todo 此处重点。根据token数（）计算block数
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,  # 缓存命中blocks（prefill）或 None（decode）
            num_encoder_tokens=num_encoder_tokens,
        )

        if not self._is_blocks_sufficient(request, num_blocks_to_allocate):
            return None

        # 将prefix-cache命中的blocks中，引用计数为0的blocks赎回
        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")
        # 将 new_computed_block_list（prefix-cache命中的blocks） 追加到 req_to_blocks[request_id]
        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  new_computed_block_list)

        # 按每层分配blocks，然后追加到 single_type_kv_cache_manager.req_to_blocks[request_id]
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens)

        
        
        # # todo （不使用这种方式实现，不真正预分配）这里判断是否prefill请求，并且开启了预分配开关，是则为prefill请求预分配全部blocks
        # #  注：enable_prefill_pre_allocate（block预分配） 和 reserve_blocks（预留block）不能同时开启
        # logger.warning(f'===== self.kv_cache_config.enable_prefill_pre_allocate={self.kv_cache_config.enable_prefill_pre_allocate}')
        # if (request.num_computed_tokens == 0 and self.kv_cache_config.enable_prefill_pre_allocate
        #         and not self.prefill_pre_allocate_blocks_num_map[request.request_id]):
        #     # 剩余要分配的block数 = 该请求总block数（所有层） - 已分配block数    （考虑所有层）
        #     already_allocated_blocks = self.coordinator.get_blocks(request.request_id)                          # 该请求所有层已分配block列表 tuple(list_layer0, list_layer1, list_layer2, ...)
        #     num_layers = len(already_allocated_blocks)                                                          # 层数
        #     num_already_allocated_blocks = sum([len(b) for b in already_allocated_blocks])                      # 该请求所有层已分配block数
        #     num_total_blocks = cdiv(min(self.max_model_len, request.max_tokens), self.block_size) * num_layers  # 该请求总block数（所有层）
        #     num_to_be_allocated_blocks = num_total_blocks - num_already_allocated_blocks                        # 剩余需要分配的block数
        #     if num_to_be_allocated_blocks <= self.coordinator.block_pool.get_num_free_blocks():
        #         block_list: list[KVCacheBlock] = self.coordinator.block_pool.get_new_blocks(num_to_be_allocated_blocks)
        #         request.prefill_pre_allocate_blocks = block_list
        #         self.prefill_pre_allocate_blocks_num_map[request.request_id] = block_list  # 预分配的blocks记录在kv_cache_manager实例中


        '''
        条件 1：not self.enable_caching
        用户未启用 Prefix Caching（启动参数未加 --enable-prefix-caching）
        → 自然不需要做任何缓存操作
        条件 2：delay_cache_blocks
        这是一个 布尔标志，通常由调度器根据请求来源设置
        在以下场景为 True：
        当前是 Decode 节点，blocks 是从 Prefill 节点 接收而来
        或处于 ** speculative decoding 的 draft 阶段**（尚未验证，不能缓存）
        或系统处于 内存压力下，暂时禁用缓存
        ✅ 只要满足任一条件，就 跳过后续的 cache_blocks(...) 调用
        '''
        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        '''
        # 以下就是要缓存blocks了
        
        计算要缓存的 token 数量（关键！）
        🔍 关键变量解释：
        变量	                    含义
        num_computed_tokens	    该请求之前已计算的 token 数（包括复用的 prefix）
        num_new_tokens	        本次调度新计算的 token 数
        request.num_tokens	    该请求当前已确认（finalized）的总 token 数
        
        🧠 为什么需要 min(...)？
        在 Speculative Decoding（推测解码） 场景中：
        请求可能生成了 draft tokens（比如 5 个）
        但只有部分被 验证通过（accepted）
        request.num_tokens 只包含 已接受的 tokens
        如果直接缓存 num_computed_tokens + num_new_tokens，可能会把 未被接受的 draft tokens 的 KV Cache 也缓存了 → 导致后续错误复用！
        ✅ 所以：只缓存到 request.num_tokens 为止，确保缓存的是“确定不会变”的 tokens。
        '''
        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        '''
        todo 重要！！！
        提交（缓存）blocks
        主要是使用（现有blocks + 新分配的blocks）更新 self.num_cached_block  
        
        注意：此处所说的缓存，不是prefix-cache，而是更新 self.num_cached_block 
        作用：将该请求当前的 KV Cache blocks 注册到 prefix cache 中（即写入 cached_block_hash_to_block）
        触发条件：只有当整个 prompt 或已生成序列 完成 prefill/decode 到某个稳定点 时才缓存
        缓存内容：对应前 num_tokens_to_cache 个 tokens 的 blocks
        ⚠️ 注意：不是每次调度都缓存，而是当达到“可提交”状态时（如 prefill 完成、或 speculative step accepted）。
        '''
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return KVCacheBlocks(new_blocks)

    def _reset_reserved_blocks_in_use_request_id(self, request_id:str):
        if self._reserved_blocks_in_use_request_id == request_id:
            logger.warning(f'===== KVCacheManager reset_reserved_blocks_in_use_request_id, request_id={request_id}')
            self._reserved_blocks_in_use_request_id = None

    def _remove_prefill_pre_allocate_blocks(self, request: Request):
        """remove pre-allocate blocks num for the request when the request is over."""
        logger.warning(f'===== _remove_prefill_pre_allocate_blocks, self.prefill_pre_allocate_blocks_num_map={self.prefill_pre_allocate_blocks_num_map}')
        self.prefill_pre_allocate_blocks_num_map.pop(request.request_id, None)
        logger.warning(f'===== self.prefill_pre_allocate_blocks_num_map={self.prefill_pre_allocate_blocks_num_map}')

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self._reset_reserved_blocks_in_use_request_id(request.request_id)
        self._remove_prefill_pre_allocate_blocks(request)
        self.coordinator.free(request.request_id)


    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache 
            group.
        """
        assert request.status == RequestStatus.RUNNING
        return self.coordinator.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return KVCacheBlocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled."""
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_empty_block_list(self) -> KVCacheBlocks:
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple([]
                                   for _ in range(self.num_kv_cache_groups)))
