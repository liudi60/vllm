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

    def _is_blocks_sufficient(self, request: Request, num_blocks_to_allocate: int) -> bool:
        # todo prefill预分配 和 decode预留 这两个特性不能叠加，因为既然prefill预分配了，请求所占tokens就够用，没必要给decode预留了
        #  prefill预分配 和 decode预留 道理一样。prefill预分配就是把block_pool中空闲block数判断递减，不真正预分配
        #  先判断是否开启prefill预分配
        # self.block_size=128
        logger.warning(
            f'===== self.kv_cache_config.enable_prefill_pre_allocate={self.kv_cache_config.enable_prefill_pre_allocate},  self.block_size={self.block_size}, self.prefill_pre_allocate_blocks_num_map={self.prefill_pre_allocate_blocks_num_map}')
        logger.warning(f'===== num_blocks_to_allocate={num_blocks_to_allocate}')
        if self.kv_cache_config.enable_prefill_pre_allocate:
            # 需要区分prefill和decode
            if request.num_computed_tokens == 0:  # 1. prefill，需要预分配该请求全部tokens
                # 当前分支可用空闲block数 = block_pool总空闲block数 - 所有prefill请求已预分配但未使用block数
                num_free_blocks = self.block_pool.get_num_free_blocks() - sum(
                    self.prefill_pre_allocate_blocks_num_map.values())
                num_pre_allocate_blocks = cdiv(min(self.max_model_len, request.max_tokens),
                                               self.block_size) * self.num_kv_cache_groups  # 该prefill请求预分配block数，需要计算所有层
                if num_pre_allocate_blocks > num_free_blocks:
                    logger.warning(
                        f'===== _is_blocks_sufficient, enable_prefill_pre_allocate, prefill, pre allocate insufficient')
                    return False
                else:  # 空闲block够用，记录该请求[可用预分配block数]
                    self.prefill_pre_allocate_blocks_num_map[
                        request.request_id] = num_pre_allocate_blocks - num_blocks_to_allocate

            else:  # 2. decode及非prefill，本请求已经预分配的块数也算作空闲块
                # 当前分支可用空闲block数 = block_pool中空闲block数 - 所有prefill请求已预分配但未使用block数 + 本请求预分配的block数
                num_free_blocks = self.block_pool.get_num_free_blocks() - sum(
                    self.prefill_pre_allocate_blocks_num_map.values()) + self.prefill_pre_allocate_blocks_num_map.get(
                    request.request_id, 0)
                if num_blocks_to_allocate > num_free_blocks:
                    logger.warning(
                        f'===== _is_blocks_sufficient, enable_prefill_pre_allocate, decode, pre allocate insufficient')
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

        if not self._is_blocks_sufficient(request, num_blocks_to_allocate):
            # Cannot allocate new blocks
            return None

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

    def _reset_reserved_blocks_in_use_request_id(self, request_id: str):
        if self._reserved_blocks_in_use_request_id == request_id:
            logger.warning(f'===== KVCacheManager reset_reserved_blocks_in_use_request_id, request_id={request_id}')
            self._reserved_blocks_in_use_request_id = None

    def _remove_prefill_pre_allocate_blocks(self, request: Request):
        """remove pre-allocate blocks num for the request when the request is over."""
        logger.warning(
            f'===== _remove_prefill_pre_allocate_blocks, self.prefill_pre_allocate_blocks_num_map={self.prefill_pre_allocate_blocks_num_map}')
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
