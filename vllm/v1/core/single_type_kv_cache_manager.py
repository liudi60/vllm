# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict

from vllm.utils import cdiv
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_cache_interface import (ChunkedLocalAttentionSpec,
                                        CrossAttentionSpec, FullAttentionSpec,
                                        KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec, SlidingWindowSpec)
from vllm.v1.request import Request

logger = init_logger(__name__)


class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management 
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
        """
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        if self.dcp_world_size > 1:
            self.block_size *= dcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str,
                                        list[KVCacheBlock]] = defaultdict(list)

        '''
        定义了一个字典 self.num_cached_block，其核心作用是：

        为每个正在运行（RUNNING）的请求，记录它当前已成功缓存到 Prefix Cache 中的完整 KV Cache block 数量。
        
        这是实现 增量式前缀缓存（Incremental Prefix Caching） 的关键状态追踪机制。
        
        ✅ 一、逐句解读注释
        1. {req_id: The number of cached blocks for this given request}
        键：请求 ID（str）
        值：该请求已缓存的 block 数量（int）
        2. This is only used to track the RUNNING requests
        只跟踪处于 RUNNING 状态的请求
        一旦请求被 抢占（preempted）或完成（finished），就不再更新或保留此信息
        📌 为什么？
        
        Preempted 请求的 block 可能被释放或重用
        缓存提交（caching）只在请求稳定推进时发生（如 prefill 完成、token 被 accept）
        避免维护无用状态，节省内存
        ✅ 二、核心作用：支持增量缓存（避免重复缓存）
        当一个请求逐步生成 tokens（如 decoding 阶段），它的 KV Cache 会逐渐填满新的 blocks。
        
        vLLM 不会每次重新缓存所有 blocks，而是：
        
        查看 num_cached_block[req_id] → 知道“已经缓存到第几个 block”
        计算当前可缓存的新完整 blocks（例如从第 3 个到第 5 个）
        只缓存新增的部分
        更新 num_cached_block[req_id] = 5
        🔄 示例：
        步骤	总 tokens	完整 blocks	num_cached_block（之前）	缓存范围	更新后值
        Prefill 完成	50	3 (48 tokens)	0	blocks[0:3]	3
        生成 16 新 tokens	66	4 (64 tokens)	3	blocks[3:4]	4
        再生成 16 tokens	82	5 (80 tokens)	4	blocks[4:5]	5
        ✅ 这样确保：每个 block 只被缓存一次，且按需增量提交
        '''
        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: list[KVCacheBlock]) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.

        Returns:
            The number of blocks.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        '''
        self.req_to_blocks 是一个核心数据结构，用于记录每个请求（request）当前占用的物理 KV Cache block 列表。
        Key：request_id（字符串，唯一标识一个请求）
        Value：该请求所有 sequence 当前分配到的 物理 block 对象列表（或 block ID 列表）
        💡 注意：在多序列请求（如 beam search）中，一个 request 可能对应多个 sequences，但 vLLM 目前主要支持单序列，所以通常 1 request → 1 sequence → 1 block list。
        
        ✅ 二、作用详解
        . 跟踪资源归属
        调度器和 block manager 需要知道：
        某个请求占用了哪些 blocks？
        当请求完成或被 abort 时，要释放哪些 blocks？
        . 支持 Prefix Caching
        在 prefix sharing 场景下，多个请求可能共享部分 blocks。
        req_to_blocks 记录的是该请求“逻辑上拥有”的所有 blocks（包括共享的）。
        实际释放时，block manager 会通过 引用计数（ref-counting） 决定是否真正回收物理 block。
        . Chunked Prefill 支持
        一个长请求分多次调度，每次分配新 blocks。
        req_to_blocks[req_id] 会逐步追加新分配的 blocks，形成完整 block 链。
        ✅ 六、注意事项
        线程安全
        
        vLLM 是单线程 event loop，所以 req_to_blocks 无需锁。
        内存泄漏风险
        
        如果请求异常退出未调用 free()，blocks 会泄露 → vLLM 通过 abort_request() 保证清理。
        与 Sequence 对象的关系
        
        在新版 vLLM 中，更推荐通过 Sequence 对象管理 blocks，但 req_to_blocks 仍是高层调度器的便捷视图。
        ✅ 总结
        self.req_to_blocks 是 KV Cache Manager 中用于“按请求 ID 快速查找其所占物理 blocks”的映射表。
        
        用途：资源跟踪、释放、复用
        关键操作：allocate 时 append，free 时 pop + release
        设计意义：实现请求级 KV Cache 生命周期管理，支撑 prefix caching 和 chunked prefill
        它是 vLLM 高效显存管理的基础设施之一。
        '''
        num_new_blocks = (num_required_blocks - len(new_computed_blocks) -
                          len(self.req_to_blocks[request_id]))  # todo self.req_to_blocks[request_id] 该请求所有 sequence 当前分配到的 物理 block 对象列表（或 block ID 列表）

        '''
        ✅ 一、上下文背景
        当一个请求（request）被调度时，vLLM 会尝试：
        
        复用已缓存的 computed blocks（来自其他请求的 prefix cache）
        为剩余部分分配新 blocks
        但这里有个关键问题：
        
        某些“computed blocks”虽然在缓存中，但可能已经被标记为可驱逐（evictable）——即它们当前在 free queue 中且引用计数为 0。
        
        如果直接复用这些 block，不能算作“免费复用”，因为：
        
        它们即将被回收（或已被视为 free）
        要复用它们，就必须将其从 free 状态“提升”回 computed 状态
        这相当于占用了原本可用于新请求的 free block 资源
        因此，vLLM 需要将这类 block 计入资源需求总量，以避免 OOM。
        
        🔹 new_computed_blocks
        这是通过 prefix caching 命中 得到的 block 列表。
        它们逻辑上是“已计算的”，可以复用，但物理状态可能不同。
        🔹 blk.ref_cnt == 0
        引用计数为 0 → 表示当前没有活跃请求在使用这个 block
        这类 block 通常会被放入 free list / eviction queue，准备回收
        🔹 not blk.is_null
        排除空 block（null block 是占位符，无实际存储）
        🔹 sum(...)
        统计：在 new_computed_blocks 中，有多少个 block 处于“可被驱逐”状态
        
        ✅ 三、为什么需要统计 num_evictable_computed_blocks？
        🎯 核心目的：准确评估本次分配对 GPU 显存的实际压力
        考虑两种 computed block：
        
        Block 类型	ref_cnt	是否在 free queue	复用时是否消耗新资源
        活跃共享 block	> 0	❌ 否	❌ 不消耗（纯复用）
        可驱逐 block	= 0	✅ 是	✅ 消耗（需“赎回”）
        💡 可驱逐 block 虽然存在，但系统已将其视为“可用显存”。
        
        如果你复用它，就等于抢回一块本可分配给别人的显存，所以必须计入资源需求。
        
        ✅ 四、后续如何使用这个值？
        通常在计算 总共需要多少 free blocks 时：
        

        # 需要新分配的 blocks（真正全新的）
        num_new_blocks = ceil(num_new_tokens / block_size)
        
        # 可驱逐的 computed blocks 也要“占用”资源
        total_blocks_needed = num_new_blocks + num_evictable_computed_blocks
        
        # 检查是否有足够 free blocks
        if total_blocks_needed > self.block_pool.free_count():
            raise OutOfMemoryError(...)
        这样就能防止因“虚假复用”导致 OOM。
        
        ✅ 五、举例说明
        假设：
        
        GPU 总 blocks: 100
        已用 blocks: 90（其中 80 个活跃，10 个 ref_cnt=0 → 在 free queue）
        free blocks reported: 20（10 真空闲 + 10 可驱逐）
        现在一个新请求到来：
        
        通过 prefix cache 命中 5 个 blocks
        其中 3 个是活跃的（ref_cnt=2）
        2 个是可驱逐的（ref_cnt=0）
        → num_evictable_computed_blocks = 2
        
        即使你“复用”了 5 个 blocks，实际仍需预留 2 个 free slots，因为那 2 个可驱逐 block 必须从 free pool 中“赎回”。
        
        如果不计这 2 个，系统会误以为只用分配 num_new_blocks，可能导致：
        
        分配后 free count < 0
        后续请求 OOM
        ✅ 六、设计哲学
        这体现了 vLLM 的 精细化显存管理思想：
        
        “复用”不等于“零成本” —— 只有真正被多个请求共享的活跃 block才是免费的；
        
        已释放但未回收的 block，复用它等同于分配新 block。
        
        这种机制确保了：
        
        内存预算精确
        OOM 行为可预测
        Prefix Caching 与内存压力平衡
        ✅ 总结
        代码含义	说明
        blk.ref_cnt == 0	该 block 当前无人使用，可被驱逐
        new_computed_blocks	本次通过 prefix cache 命中的 blocks
        num_evictable_computed_blocks	这些命中 block 中，有多少是“名义存在、实际可回收”的
        用途	在资源检查时，将这部分 block 视为“需要占用新资源”，防止 OOM
        🔑 一句话理解：
        
        “能复用的 block，如果已经躺在垃圾桶里（ref_cnt=0），那你捡回来用，也算占地方。”
        
        这是 vLLM 实现高可靠、高密度推理的关键细节之一。
        '''
        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it will be changed from a free block
        # to a computed block when the request is allocated, so we also count
        # it as needed to be allocated.
        num_evictable_computed_blocks = sum(
            blk.ref_cnt == 0 and not blk.is_null
            for blk in new_computed_blocks)
        return num_new_blocks + num_evictable_computed_blocks

    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: list[KVCacheBlock]) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        if request_id not in self.num_cached_block:  # waiting队列中的请求
            # A new request.
            req_blocks = self.req_to_blocks[request_id]
            assert len(req_blocks) == 0
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = len(new_computed_blocks)
        else:  # running队列中的请求
            # A running request. Should not have new computed blocks.
            assert len(new_computed_blocks) == 0

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached 
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block[request.request_id]  # 该请求当前已缓存的 block 数
        num_full_blocks = num_tokens // self.block_size

        # 调用 cache_full_blocks 提交缓存，就是向 block_pool.cached_block_hash_to_block 中插入了1个 kv
        self.block_pool.cache_full_blocks(
            request=request,
            # 此处的 self.req_to_blocks[request.request_id] 该请求的block数在调用allocate_new_blocks函数时，大小已经扩充到了 num_full_blocks，调用 req_blocks.extend(new_blocks) 这一行进行扩充的
            blocks=self.req_to_blocks[request.request_id],  # 该请求已经分配的物理block_ids，包括prefix-cache命中的
            num_cached_blocks=num_cached_blocks,            # 该请求当前已缓存数量（用于跳过）
            num_full_blocks=num_full_blocks,                # 该请求当前应缓存到多少个
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )
        # todo 这里比较关键，更新该请求的已分配的block总数。self.num_cached_block 记录的是每个请求分配的block数量
        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        """
        Get the number of common prefix blocks for all requests in the RUNNING
        state.

        Args:
            request_id: The request ID.
            num_running_requests: The total number of requests in the RUNNING
                state.

        Returns:
            The number of common prefix blocks for all requests in the RUNNING
                state.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than 
        `max_length`. The prefix should be a common prefix hit for all the 
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found, 
        return an empty list. 
        If eagle is enabled, drop the last matched block to force recompute the 
        last block to get the required hidden states for eagle drafting head. 
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.

        Returns:
            A list of cached blocks with skipped blocks replaced by null block
            for each kv cache group in `kv_cache_group_ids`.
            Return a list of length `len(kv_cache_group_ids)`, where the i-th
            element is a list of cached blocks for the i-th kv cache group
            in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """

        raise NotImplementedError

    @abstractmethod
    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and free the 
        blocks. The removed blocks should be replaced by null_block.
        Need to be customized for each attention type.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        raise NotImplementedError


class FullAttentionManager(SingleTypeKVCacheManager):

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, (FullAttentionSpec, ChunkedLocalAttentionSpec)
        ), "FullAttentionManager can only be used for full attention " \
            "and chunked local attention groups"
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids)))
        block_size = kv_cache_spec.block_size
        if dcp_world_size > 1:
            block_size *= dcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        # No need to remove blocks for full attention.
        pass

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        blocks = self.req_to_blocks[request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


class SlidingWindowManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec, block_pool: BlockPool,
                 **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups")
        assert dcp_world_size == 1, "DCP not support sliding window attn now."

        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size)
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            sliding_window_contiguous_blocks += 1

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple([block_pool.null_block] * max_num_blocks
                                for _ in range(len(kv_cache_group_ids)))
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                    block_hashes[i], kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks:]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        # Remove the blocks that are no longer be in the sliding window and
        # skipped during the attention computation.
        last_useful_token = num_computed_tokens - self.sliding_window + 1
        last_useful_block = last_useful_token // self.block_size
        blocks = self.req_to_blocks[request_id]
        removed_blocks: list[KVCacheBlock] = []
        for i in range(last_useful_block - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return 
        0 here for correctness. Need to support cascade attention + sliding 
        window in the future.
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec,
                 block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local 
        window, and set the block as null. Examples:

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in 
        the window(needs lookup), 0th - 7th are not in the window, 
        so they are already marked as computed. We check the complete 
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return 
        [null, null, block 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not 
        in the window, so they are already marked as computed. 
        we return 4 blocks[null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.

        Returns:
            A list of cached blocks
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for " +
            "chunked local attention groups")
        assert use_eagle is False, ("Hybrid KV cache is not supported for " +
                                    "eagle + chunked local attention.")
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (max_length //
                                         kv_cache_spec.attention_chunk_size *
                                         kv_cache_spec.attention_chunk_size)
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (local_attention_start_idx //
                                           kv_cache_spec.block_size)
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids)))
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        # Remove the blocks that are no longer be in the chunked attention
        # window and skipped during the attention computation.

        # [chunk 0][chunk 1]local_attention_start_idx ... current
        # we computed previous number of chunks to get the idx of
        # current chunk window starting offset,
        # e.g. for computed 1024 tokens, the 1024th token (0 indexed)
        # is in the second chunk, there are 1 prev chunk, the start idx
        # is 1024. for 1023, it will be 0.
        num_cached_block = self.num_cached_block.get(request_id, 0)
        local_attention_start_idx = (
            num_computed_tokens
        ) // self.attention_chunk_size * self.attention_chunk_size
        first_useful_block_idx = local_attention_start_idx // self.block_size
        if num_cached_block > 0:
            # Make sure we don't delete the last cached block
            first_useful_block_idx = min(first_useful_block_idx,
                                         num_cached_block - 1)
        # if block size = 128, 0 -> block 0, 1024 (= 128 * 8) ->
        # block 8, 372 (= 128 * 2 + 116) -> block 2
        blocks = self.req_to_blocks[request_id]
        removed_blocks: list[KVCacheBlock] = []
        # we need to keep the last block to get the previous hash key
        for i in range(first_useful_block_idx - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0


class MambaManager(SingleTypeKVCacheManager):

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec,
            MambaSpec), ("MambaManager can only be used for mamba groups")
        assert dcp_world_size == 1, "DCP not support mamba now."
        # Prefix caching is not supported for mamba now. Always return empty
        # list.
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids)))
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        # Each request will always have 1 block at this moment, so no need to
        # remove blocks.
        pass

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        return 0

    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: list[KVCacheBlock]) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.

        Returns:
            The number of blocks
        """

        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (self.kv_cache_spec.block_size *
                           self.kv_cache_spec.num_speculative_blocks)
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = (num_required_blocks - len(new_computed_blocks) -
                          len(self.req_to_blocks[request_id]))
        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it will be changed from a free block
        # to a computed block when the request is allocated, so we also count
        # it as needed to be allocated.
        num_evictable_computed_blocks = sum(
            blk.ref_cnt == 0 and not blk.is_null
            for blk in new_computed_blocks)
        return num_new_blocks + num_evictable_computed_blocks

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> list[KVCacheBlock]:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (self.kv_cache_spec.block_size *
                           self.kv_cache_spec.num_speculative_blocks)
        return super().allocate_new_blocks(request_id, num_tokens)


class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: list[KVCacheBlock]) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        assert len(new_computed_blocks) == 0

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so this method is not relevant.
        raise ValueError("Should not be called as prefix caching is disabled.")

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        # Cross-attention blocks contain request-specific encoder states
        # and are not shared between different requests
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        # Cross-attention does not benefit from prefix caching since:
        # 1. Encoder states are unique per request (different audio/image
        #    inputs)
        # 2. Encoder states are computed once per request, not incrementally
        # 3. No reusable prefix exists between different multimodal inputs
        # Return empty blocks to indicate no cache hits
        raise NotImplementedError(
            "CrossAttentionManager does not support caching")

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        # Cross-attention blocks represent encoder states which are needed
        # for the entire decoding process, so no blocks should be skipped
        pass


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
}


def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec,
                                  **kwargs) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    logger.warning(f'===== get_manager_for_kv_cache_spec, manager_class={manager_class}')
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
