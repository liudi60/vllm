# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal, Optional, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
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

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

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
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
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
        '''
        if (not self.enable_caching
                or (request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None)):
            return self.create_empty_block_list(), 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(request.block_hashes,
                                                    max_cache_hit_length))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens


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
    # 该函数在prefill和decode阶段都会被调用
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
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

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups)))

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(request.request_id,
                                               request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )

        # todo 111 此处判断reserve_block_num
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        '''
        if is_prefill:
            if num_blocks_to_allocate + reserve_block_num > self.block_pool.get_num_free_blocks():
                # Cannot allocate new blocks
                return None
        else:
            if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
                # Cannot allocate new blocks
                return None
            elif num_blocks_to_allocate + reserve_block_num > self.block_pool.get_num_free_blocks():
                reserved_block_avail_ = True
                # 并且，只把reserve_block_num分给1个请求  

        '''

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  new_computed_block_list)

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return KVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
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
