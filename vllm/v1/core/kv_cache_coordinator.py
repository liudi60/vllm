# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager, FullAttentionManager, get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.request import Request
from vllm.logger import init_logger

logger = init_logger(__name__)

class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # ===== KVCacheCoordinator实例化，子类：UnitaryKVCacheCoordinator
        logger.warning(f'===== KVCacheCoordinator实例化，子类：{self.__class__.__name__}')
        '''
            KVCacheConfig(
                num_blocks=5434,
                kv_cache_tensors=[KVCacheTensor(size=1424490496, shared_by=['model.layers.0.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.1.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.2.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.3.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.4.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.5.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.6.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.7.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.8.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.9.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.10.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.11.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.12.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.13.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.14.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.15.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.16.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.17.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.18.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.19.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.20.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.21.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.22.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.23.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.24.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.25.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.26.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.27.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.28.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.29.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.30.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.31.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.32.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.33.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.34.self_attn.attn']),
                    KVCacheTensor(size=1424490496, shared_by=['model.layers.35.self_attn.attn'])],
                kv_cache_groups=[KVCacheGroupSpec(layer_names=['model.layers.0.self_attn.attn',
                                                                'model.layers.1.self_attn.attn',
                                                                'model.layers.2.self_attn.attn',
                                                                'model.layers.3.self_attn.attn',
                                                                'model.layers.4.self_attn.attn',
                                                                'model.layers.5.self_attn.attn',
                                                                'model.layers.6.self_attn.attn',
                                                                'model.layers.7.self_attn.attn',
                                                                'model.layers.8.self_attn.attn',
                                                                'model.layers.9.self_attn.attn',
                                                                'model.layers.10.self_attn.attn',
                                                                'model.layers.11.self_attn.attn',
                                                                'model.layers.12.self_attn.attn',
                                                                'model.layers.13.self_attn.attn',
                                                                'model.layers.14.self_attn.attn',
                                                                'model.layers.15.self_attn.attn',
                                                                'model.layers.16.self_attn.attn',
                                                                'model.layers.17.self_attn.attn',
                                                                'model.layers.18.self_attn.attn',
                                                                'model.layers.19.self_attn.attn',
                                                                'model.layers.20.self_attn.attn',
                                                                'model.layers.21.self_attn.attn',
                                                                'model.layers.22.self_attn.attn',
                                                                'model.layers.23.self_attn.attn',
                                                                'model.layers.24.self_attn.attn',
                                                                'model.layers.25.self_attn.attn',
                                                                'model.layers.26.self_attn.attn',
                                                                'model.layers.27.self_attn.attn',
                                                                'model.layers.28.self_attn.attn',
                                                                'model.layers.29.self_attn.attn',
                                                                'model.layers.30.self_attn.attn',
                                                                'model.layers.31.self_attn.attn',
                                                                'model.layers.32.self_attn.attn',
                                                                'model.layers.33.self_attn.attn',
                                                                'model.layers.34.self_attn.attn',
                                                                'model.layers.35.self_attn.attn'],
                    kv_cache_spec=AscendFullAttentionSpec(block_size=128,
                                                            num_kv_heads=4,
                                                            head_size=128,
                                                            dtype=torch.bfloat16,
                                                            use_mla=False,
                                                            use_sfa=False,
                                                            sliding_window=None,
                                                            attention_chunk_size=None)
                    )]
            )
        '''
        logger.warning(f'===== KVCacheCoordinator构造函数中初创建BlockPool, kv_cache_config={kv_cache_config}')
        self.block_pool = BlockPool(kv_cache_config.num_blocks, enable_caching,
                                    enable_kv_cache_events)

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
            ) for i, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups))

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int,
                                   new_computed_blocks: tuple[
                                       list[KVCacheBlock], ...],
                                   num_encoder_tokens: int) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The number of blocks.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):  # 分层遍历
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_encoder_tokens, [])
            else:
                # 每一层需要新分配的blocks，累加
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_tokens, new_computed_blocks[i])
        return num_blocks_to_allocate

    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.save_new_computed_blocks(request_id,
                                             new_computed_blocks[i])

    def allocate_new_blocks(
            self,
            request_id: str,
            num_tokens: int,
            num_encoder_tokens: int = 0) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id, num_encoder_tokens if isinstance(
                    manager, CrossAttentionManager) else num_tokens)
            for manager in self.single_type_managers)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> list[int]:
        """
        Get the number of common prefix blocks for all requests in the RUNNING
        state for each kv cache group.

        Args:
            request_id: The request ID.
            num_running_requests: The total number of requests in the RUNNING
                state.

        Returns:
            list[int]: The number of common prefix blocks for all requests in
                the RUNNING state for each kv cache group.
        """
        num_blocks_per_group = [
            manager.get_num_common_prefix_blocks(request_id,
                                                 num_running_requests)
            for manager in self.single_type_managers
        ]
        return num_blocks_per_group

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace 
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, num_computed_tokens)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers)

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        pass


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_kv_cache_events: bool,
                 dcp_world_size: int):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         False,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size)
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager))
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 enable_kv_cache_events: bool, dcp_world_size: int):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         enable_caching,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size)
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[
            0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            dcp_world_size=self.dcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of 
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_model_len: int,
                 use_eagle: bool, enable_caching: bool,
                 enable_kv_cache_events: bool, dcp_world_size: int):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         enable_caching,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size)
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Verifies that the model has exactly two types of KV cache groups, and 
        one of them is full attention. Then, split the kv cache groups into full
        attention groups and other groups.
        """
        full_attention_spec: Optional[FullAttentionSpec] = None
        other_spec: Optional[KVCacheSpec] = None
        self.full_attention_group_ids: list[int] = []
        self.other_group_ids: list[int] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                if full_attention_spec is None:
                    full_attention_spec = g.kv_cache_spec
                else:
                    assert full_attention_spec == g.kv_cache_spec, (
                        "HybridKVCacheCoordinator assumes exactly one type of "
                        "full attention groups now.")
                self.full_attention_group_ids.append(i)
            else:
                if other_spec is None:
                    other_spec = g.kv_cache_spec
                else:
                    assert other_spec == g.kv_cache_spec, (
                        "HybridKVCacheCoordinator assumes "
                        "exactly one other type of groups now.")
                self.other_group_ids.append(i)

        assert full_attention_spec is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of full "
            "attention groups now.")
        assert other_spec is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of other "
            "groups now.")

        self.full_attention_manager_cls = FullAttentionManager
        self.other_attention_cls = self.single_type_managers[
            self.other_group_ids[0]].__class__
        self.full_attention_spec = full_attention_spec
        self.other_spec = other_spec
        self.full_attention_block_size = self.full_attention_spec.block_size
        self.other_block_size = self.other_spec.block_size

        if self.enable_caching:
            # this requirement is only needed for the prefix caching logic
            divisible = self.other_block_size % self.full_attention_block_size
            assert divisible == 0, (
                "KVCacheCoordinator assumes the block_size of full "
                "attention layers is divisible by other layers now.")

        if max(self.full_attention_group_ids) < min(self.other_group_ids):
            self.full_attn_first = True
        elif max(self.other_group_ids) < min(self.full_attention_group_ids):
            self.full_attn_first = False
        else:
            raise ValueError(
                "HybridKVCacheCoordinator assumes the full "
                "attention group ids and other attention group ids "
                "do not interleave, either full attention group ids "
                "are before other attention group ids or vice versa."
                "This is for simplifying merging hit_blocks_full_attn and "
                "hit_blocks_other_attn to hit_blocks.")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit for the request.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A list of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """
        # First, find the longest cache hit for full attention.
        hit_blocks_full_attn = (
            self.full_attention_manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=self.full_attention_group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=self.full_attention_spec,
                use_eagle=self.use_eagle,
            ))
        hit_length = len(
            hit_blocks_full_attn[0]) * self.full_attention_block_size

        # Next, find the cache hit for the other attention WITHIN
        # the cache hit of full attention.
        hit_blocks_other_attn = (
            self.other_attention_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=hit_length,
                kv_cache_group_ids=self.other_group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=self.other_spec,
                use_eagle=self.use_eagle,
            ))
        hit_length = len(hit_blocks_other_attn[0]) * self.other_block_size

        # NOTE: the prefix cache hit length must be a multiple of block_size as
        # we don't support partial block cache hit yet. The cache hit length
        # of other attention is ensured to be a multiple of the block size of
        # full attention layers in current implementation, because hit_length is
        # a multiple of other attention's block size, and other attention's
        # block size is a multiple of full attention's block size (verified in
        # `verify_and_split_kv_cache_groups`).
        assert hit_length % self.full_attention_block_size == 0

        # Truncate the full attention cache hit to the length of the
        # cache hit of the other attention.
        for group_hit_blocks in hit_blocks_full_attn:
            del group_hit_blocks[hit_length // self.full_attention_block_size:]

        # Merge the hit blocks of full attention and other attention.
        if self.full_attn_first:
            hit_blocks = hit_blocks_full_attn + hit_blocks_other_attn
        else:
            hit_blocks = hit_blocks_other_attn + hit_blocks_full_attn
        return hit_blocks, hit_length

'''
1
vLLM KVCacheManager中coordinator协调架构设计
本代码图展示了vLLM KVCacheManager中coordinator协调架构的设计原因和实现机制。通过分层设计，coordinator统一管理多种KV缓存类型
（如完整注意力、滑动窗口、交叉注意力等）[1b]，为上层调度器提供统一接口[2a-2d]，同时支持混合注意力机制的前缀缓存优化[3a-3d]。这种
设计使调度器无需感知底层差异[4a-4d]，而各种注意力管理器可以实现特定优化[5a-5d]。


coordinator协调器的初始化架构
KVCacheManager通过工厂模式创建coordinator，实现多种KV缓存类型的统一管理. Hide guide


Motivation
在大规模语言模型推理中，不同层可能使用不同的注意力机制（如完整注意力、滑动窗口、交叉注意力等），每种机制对KV缓存的管理需求差异很大。
如果让上层调度器直接处理这些差异，会导致代码复杂且难以维护。coordinator协调器通过分层设计解决这一问题：它为多种KV缓存类型提供统一
接口，让调度器无需感知底层差异，同时支持混合模型的前缀缓存优化。

Details
工厂模式选择协调器类型
系统根据模型配置自动选择最合适的协调器实现[1b]：

无缓存模式：KVCacheCoordinatorNoPrefixCache，适用于禁用前缀缓存的场景
单组模式：UnitaryKVCacheCoordinator，适用于只使用一种注意力机制的模型
混合模式：HybridKVCacheCoordinator，适用于包含多种注意力机制的复杂模型
共享资源管理
所有协调器都共享同一个BlockPool实例[1d]，确保内存资源的统一管理和高效利用。同时，协调器为每个KV缓存组创建对应的管理器[1c]，如
FullAttentionManager、SlidingWindowManager等，每个管理器专门处理特定注意力机制的缓存逻辑。

统一接口封装
协调器为上层提供统一的操作接口，如free()、allocate_new_blocks()、cache_blocks()等。这些接口内部会遍历所有单类型管理器并执行
相应操作，实现了接口统一、实现多样化的设计目标。

这种架构设计使得vLLM能够灵活支持各种复杂的模型结构，同时保持代码的可维护性和扩展性。


2
协调器统一接口封装
coordinator为不同类型的KV缓存管理器提供统一的操作接口. Hide guide

AI generated guide
Motivation
在大规模语言模型推理中，vLLM需要处理多种不同的注意力机制，如完整注意力、滑动窗口注意力、交叉注意力等。每种机制对KV缓存的管理策略都不同：
完整注意力需要缓存所有token，滑动窗口只需要缓存窗口内的token，交叉注意力则完全不支持前缀缓存。如果让上层调度器直接处理这些差异，会导致
代码复杂且难以维护。coordinator协调器的核心目标就是统一管理多种KV缓存类型，为调度器提供一致的接口，隐藏底层实现的复杂性。

Details
coordinator采用分层协调模式设计。KVCacheManager作为最上层，通过工厂函数get_kv_cache_coordinator()根据配置创建合适的协调器
实例[1b]。协调器内部维护一个single_type_managers数组，每个元素负责管理一种特定类型的KV缓存[1c]，所有管理器共享同一个block_pool
资源池[1c]。

这种设计的核心优势在于接口统一性。当调度器需要释放请求资源时，只需调用coordinator.free()[2a]，协调器会自动遍历所有管理器并执行各自
的释放逻辑[2d]。分配新块时，协调器会根据管理器类型选择合适的参数：交叉注意力管理器使用num_encoder_tokens，其他管理器使用标准num_tokens[2b]。

对于混合注意力模型，HybridKVCacheCoordinator实现了智能协调策略。它首先查找完整注意力的最长缓存命中[3a-3b]，然后在完整注意力范围
内查找其他注意力的缓存[3c]，最后按层顺序合并结果[3d]，确保不同注意力类型之间的缓存一致性。

整个架构让调度器代码保持简洁[4a-4d]，同时允许各种注意力管理器实现特定的优化策略，如滑动窗口的跳过计算[5a]、交叉注意力禁用缓存[5d]等，
实现了高内聚低耦合的设计目标。
'''
def get_kv_cache_coordinator(kv_cache_config: KVCacheConfig,
                             max_model_len: int, use_eagle: bool,
                             enable_caching: bool,
                             enable_kv_cache_events: bool,
                             dcp_world_size: int) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(kv_cache_config,
                                               max_model_len,
                                               use_eagle,
                                               enable_kv_cache_events,
                                               dcp_world_size=dcp_world_size)
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(kv_cache_config,
                                         max_model_len,
                                         use_eagle,
                                         enable_caching,
                                         enable_kv_cache_events,
                                         dcp_world_size=dcp_world_size)
    return HybridKVCacheCoordinator(kv_cache_config,
                                    max_model_len,
                                    use_eagle,
                                    enable_caching,
                                    enable_kv_cache_events,
                                    dcp_world_size=dcp_world_size)
