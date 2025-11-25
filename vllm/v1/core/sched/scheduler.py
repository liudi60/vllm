# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        self.dcp_world_size = \
            vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): The scheduler’s block_size must be multiplied
        # by dcp_world_size, since block hashes are computed on the
        # original full token sequence at a granularity of
        # original_block_size × dcp_world_size.
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        elif self.scheduler_config.policy == "sjf":
            self.policy = SchedulingPolicy.SJF
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # waiting队列
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # running队列
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # 创建KVCacheManager
        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        self.peak_split_enabled = False
        self.chunked_prefill_enabled = self.scheduler_config.chunked_prefill_enabled
        if vllm_config.additional_config:
            self.peak_split_enabled = vllm_config.additional_config.get("peak_split_enabled", False) if self.scheduler_config.chunked_prefill_enabled else False
            self.peak_split_factor = vllm_config.additional_config.get("peak_split_factor", 0.5)

        self.chunked_prefill_tail_optimization_factor = vllm_config.additional_config.get("chunked_prefill_tail_optimization_factor", 1)


    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # Peak Split-related
        if self.peak_split_enabled:
            if self.peak_split_factor * (len(self.running) + len(self.waiting)) < self.max_num_running_reqs:
                self.chunked_prefill_enabled = False
                logger.debug(
                    f"peak_split_enabled is {self.peak_split_enabled} and chunked_prefill_enabled is {self.chunked_prefill_enabled}")
            else:
                self.chunked_prefill_enabled = True
                logger.debug(
                    f"peak_split_enabled is {self.peak_split_enabled} and chunked_prefill_enabled is {self.chunked_prefill_enabled}")

        # For logging.
        scheduled_timestamp = time.monotonic()

        # todo 首先处理running队列
        logger.warning(f'===== Scheduler.schedule()中，处理running队列')
        # First, schedule the RUNNING requests.
        '''
            一、上下文：什么是 self.running？
            在 vLLM 中，所有请求被分为三类队列：
            waiting: 刚到达、尚未分配 KV Cache 的请求
            running: 已开始生成、正在处理中 的请求（已有部分输出）
            swapped: 因显存不足被换出到 CPU 的请求
            self.running 是一个 列表（list），包含当前正在 GPU 上执行推理的请求对象（Request 或 SequenceGroup）。
        '''
        req_index = 0  # running队列中现在要处理第几个请求
        '''
            二、两个判断条件的含义
            条件 1：req_index < len(self.running)
            含义：还有未处理的 RUNNING 请求。
            作用：遍历 self.running 列表中的每一个请求。
            注意：vLLM 不会一次性处理所有 running 请求，而是按需调度（见条件2）。
            条件 2：token_budget > 0
            含义：当前还有“token 配额”可用于本次调度批次。
            关键概念：token_budget 是本次调度轮次中允许使用的最大 token 数量，由以下因素决定：
            GPU 显存剩余空间（决定能放多少 KV Cache blocks）
            用户设置的 max_num_batched_tokens
            当前 batch 的总 token 数不能超过模型上下文长度（如 8192）
            💡 token_budget 是动态计算的资源配额，用于防止 OOM 和控制批大小。
        '''
        '''
            三、整体逻辑：为什么这样设计？
            这个 while 循环的目标是：
            
            在不超出显存和 token 限制的前提下，尽可能多地将 RUNNING 请求加入当前推理批次（batch）。
            
            执行流程：
            从 self.running[0] 开始遍历
            对每个请求：
            检查它下一步需要多少 token（通常是 1 个 decode token，但 prefill 可能更多）
            如果 token_budget >= needed_tokens：
            将该请求加入当前 batch
            token_budget -= needed_tokens
            req_index += 1
            否则：
            停止调度（即使后面还有请求，也不处理）
            循环结束 → 继续处理 swapped 或 waiting 队列
            ⚠️ 注意：vLLM 按顺序调度 running 请求，不会跳过某个请求去处理后面的（保证公平性和简单性）。
        '''
        while req_index < len(self.running) and token_budget > 0:

            request = self.running[req_index]  # 单线程处理，EngineCore进程的主线程，没有并发问题

            # logger.warning(
            #     f'===== class Scheduler.schedule(), 处理running队列, len(self.running)={len(self.running)}, req_index={req_index}, request={request}')

            # 计算当前调度轮次中，每个 running 请求还能生成/处理多少新 token（num_new_tokens），并考虑多种优化策略（如分块预填充、模型长度限制、编码器输入等）。
            '''
                ✅ 一、背景：什么是 RUNNING 请求？
                RUNNING 请求是已经开始推理的请求（已完成 prefill 或部分 decode）。
                每次调度时，vLLM 需决定：本次 batch 中，该请求能处理多少新 token？
                对于纯 decode 请求：通常是 1 个 token
                对于 speculative decoding（推测解码）：可能一次处理多个 token
                对于 chunked prefill 的尾部：可能一次处理一个 chunk
            '''

            '''
                # num_tokens_with_spec	    在 speculative decoding 下，目标模型 + 草稿模型总共要验证的 token 数（含已生成的）
                # num_output_placeholders	为未来输出预留的 token 位置数（通常 = num_tokens_with_spec）
                # num_computed_tokens	    已经完成计算的 token 数（包括 prefill 和已 decode 的）
                # 📌 所以 num_new_tokens = 还需要计算的 token 数量
                
                ✅ 举例：
                prompt: 100 tokens（已 prefill）
                已 decode: 5 tokens
                spec decoding 要验证 next 3 tokens
                则 num_tokens_with_spec = 100 + 5 + 3 = 108
                num_computed_tokens = 105
                num_new_tokens = 108 - 105 = 3
            '''
            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            '''
                🔍 背景：什么是 Chunked Prefill？
                当 prompt 很长（如 32k tokens），一次性 prefill 会 OOM 或延迟高。
                vLLM 支持 --enable-chunked-prefill，将 prompt 分成小块（如每块 512 tokens）逐步处理。
                但最后一块（tail）可能很小（如只剩 10 tokens），导致 GPU 利用率低。
                💡 优化策略：
                如果剩余 token 数 num_new_tokens 超过某个阈值（如 long_prefill_token_threshold = 2048）
                且启用了 chunked prefill
                则强制将本次处理量限制为阈值大小（如 2048），避免一次处理过大块
                ⚠️ 注意：这个条件中的 self.chunked_prefill_tail_optimization_factor 通常是 1.0，所以实际比较的是：
                    if long_prefill_token_threshold <= num_new_tokens and chunked_prefill_enabled:
                ✅ 目的：平衡吞吐与延迟，避免大尾块拖慢整个 batch
            '''
            if (0 < self.chunked_prefill_tail_optimization_factor * self.scheduler_config.long_prefill_token_threshold <=
                    num_new_tokens and self.chunked_prefill_enabled):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            '''
                为什么 -1？
                模型最大长度 max_model_len 包括 所有 input + output tokens
                num_computed_tokens 是已计算的 token 数
                所以最多还能生成：max_model_len - num_computed_tokens
                但这里写成 -1，可能是为了预留一个位置给 EOS 或安全边界
                ✅ 防止越界（如 Llama-3-8B 的 max len=8192，不能生成第 8193 个 token）
            '''
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            '''
                5️⃣ 处理编码器输入（Encoder-Decoder 架构）
                适用场景：
                Encoder-Decoder 模型（如 T5、Flan-T5、Whisper）
                这类模型有 独立的 encoder 输入（如 source text），只需计算一次
                逻辑：
                如果该请求有 encoder 输入 且尚未计算
                则调用 _try_schedule_encoder_inputs：
                分配 encoder 计算资源
                可能减少 num_new_tokens（因为 encoder 占用 compute budget）
                返回新的 encoder_compute_budget（类似 token_budget）
                📌 对纯 decoder 模型（如 Llama、Qwen），此分支不执行。
            '''
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_compute_budget
                 ) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_compute_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                logger.warning(f'===== running队列中，当前请求[req_index={req_index}]的num_new_tokens==0，不再生成新tokens了')
                req_index += 1
                continue

            while True:
                # 1、尝试为请求分配物理block
                # todo
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)

                # logger.warning(
                #     f'===== class Scheduler.schedule(), 处理running队列, 分配new_blocks, new_blocks={new_blocks}')

                # 2、物理block分配失败，显存不足
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        # 则触发 抢占（preemption）机制，释放低优先级请求的资源，腾出空间给当前请求
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()  # 默认 FIFO，弹出最后一个（最近加入的）

                    self.kv_cache_manager.free(preempted_req)  # 释放被抢占请求preempted_req的物理block
                    self.encoder_cache_manager.free(preempted_req)  # （用于 encoder-decoder 模型，如 T5）
                    # 标记为 PREEMPTED（被抢占）
                    # num_computed_tokens = 0 表明采用的是 recompute 模式（而非 swap）：
                    # 下次重新调度时，会从头开始计算 prompt（无 KV Cache 缓存）
                    # 如果是 swap 模式，会保留 KV Cache 到 CPU，并记录 num_computed_tokens
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0  # 重置已计算 token 数（recompute 模式）
                    if self.log_stats: # 记录事件（可选），用于性能分析和监控
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)  # 将被抢占请求放回 waiting 队列，prepend_request：插入到 waiting 队列头部，使其下次调度时优先尝试（避免饥饿）。
                    preempted_reqs.append(preempted_req)  # 同时记录到 preempted_reqs 列表，用于后续统计或日志。
                    # 特殊情况：自己被抢占？说明其他running队列中请求都比自己优先级高，此时放弃调度该请求，跳出循环
                    if preempted_req == request: #
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            # 如上，发生抢占时，就会换出请求到waiting队列，等待重计算。然后while循环中重新给req分配block，直到分配成功或者不能抢占为止。
            if not can_schedule:
                break
            assert new_blocks is not None  # 断言确保：只要 can_schedule 为真，new_blocks 一定非空。

            # Schedule the request.
            scheduled_running_reqs.append(request)  # 将请求加入调度列表。scheduled_running_reqs：本轮最终会送入 GPU 计算的 running 请求列表。后续会基于此列表构建 input_tokens、block_tables 等张量。
            # 记录分配的 block 信息
            # req_to_new_blocks：字典，映射 request_id → 新分配的物理 block 列表。
            # 这些 block 会被用于：
            # 更新该请求的 block_table
            # 构建 PagedAttention 所需的物理 block ID 张量
            req_to_new_blocks[request.request_id] = new_blocks
            #  记录本次调度的 token 数
            # num_new_tokens 通常是 1（decode 阶段每次生成 1 个 token）。
            # 但在 prefill 阶段 或 chunked prefill 中可能 >1。
            # 用于后续构建 input 张量或统计。
            num_scheduled_tokens[request.request_id] = num_new_tokens
            #
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras



        # todo 上面先处理running队列，如果running队列为空，则跳过，到此处理waiting队列
        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        logger.warning(f'===== Scheduler.schedule()中，处理waiting队列')
        # Next, schedule the WAITING requests.
        '''
            if not preempted_reqs:
            含义：
            只有当本轮调度中没有发生“抢占（preemption）”时，才允许调度新的 waiting 请求。
            
            为什么？
            Preemption（抢占） 指的是：因为显存不足，vLLM 被迫将某些 running 请求 swap out 到 CPU 或 abort（中止）。
            如果发生了抢占，说明 系统已经处于资源紧张状态。
            此时如果还继续从 waiting 队列拉新请求进来，会加剧资源压力，可能导致：
            更多抢占
            活锁（livelock）：不断 swap in/out，无法推进任何请求
            延迟飙升
            ✅ 设计哲学：先稳定当前 workload，再接纳新请求。
            
        '''
        if not preempted_reqs:  # 没发生抢占，说明资源够用，可以接纳waiting中的请求
            '''
                条件 1：self.waiting
                表示还有未处理的新请求。
                self.waiting 是一个 FIFO 队列（通常是 deque），先进先出。
                条件 2：token_budget > 0
                表示当前调度批次还有 token 配额 可用。
                token_budget 是动态计算的，受以下限制：
                GPU 显存剩余（决定还能分配多少 KV Cache blocks）
                max_num_batched_tokens（用户设置的最大 batch token 数）
                模型最大上下文长度（max_model_len）
                ⚠️ 注意：对于 waiting 请求，首次调度需要处理整个 prompt（prefill），可能消耗大量 tokens（如 1000+），远高于 running 请求的 1 token（decode）。
                
                ✅ 四、为什么这样设计？
                . 防止“大请求饿死小请求” or “小请求阻塞大请求”？
                默认 FIFO 策略下，大请求会阻塞后续所有请求。
                但 vLLM 认为：prompt 长度是用户输入的一部分，不应随意重排（影响语义）。
                如需优化，可通过 --enable-chunked-prefill 分块处理长 prompt。
                . token_budget 保护机制
                即使 waiting 队列非空，也会因 token_budget <= 0 而停止。
                避免一次性拉入太多请求导致 OOM。
                . preempted_reqs 作为“刹车”信号
                一旦系统开始 swap/abort，就暂停接纳新请求，直到 workload 稳定。
            '''
            # 这段代码专门用于 处理 waiting 队列中的请求，尝试将它们加入本轮推理批次（batch）。其核心目标是：在满足资源约束的前提下，尽可能多地从 waiting 队列中调度新请求进行 prefill 计算。
            # waiting 队列：存放刚到达、尚未开始计算的请求（状态为 WAITING）。
            # token_budget：本轮 batch 剩余可处理的 token 数量（由 max_num_batched_tokens 控制）。
            # max_num_running_reqs：running 队列的最大请求数（防上下文切换开销）。
            # scheduled_loras：本轮已调度请求使用的 LoRA ID 集合。
            while self.waiting and token_budget > 0:
                self.waiting.get_statistics()
                logger.warning(f'===== class Scheduler.schedule(), 处理waiting队列, token_budget={token_budget}, self.waiting={self.waiting}')
                if len(self.running) == self.max_num_running_reqs:  # 检查 running 队列是否已满，即使还有 token 预算，也不能超过最大并发请求数。这是为了控制 GPU 上下文切换开销 和 调度复杂度
                    break

                # todo 这里就是从waiting队列中获取请求，waiting队列有多种实现算法：FCFS、Priority、SJF、SJFInHeap
                request = self.waiting.peek_request()  # 查看下一个待调度请求（不弹出），后续根据资源检查结果，决定是否真正弹出并调度。

                # 处理远程 KV Cache 依赖（KVTransfer 场景）
                # 背景：
                # 在 分布式推理 或 KV Cache 共享 场景中，某些请求需等待远程节点发送 KV Cache。
                # 状态为 WAITING_FOR_REMOTE_KVS 表示“正在等远程数据”。
                # 逻辑：
                # 调用 _update_waiting_for_remote_kv() 检查是否已收到数据。
                # 若已就绪 → 改为 WAITING，本次可调度。
                # 若未就绪 → 弹出该请求，放入 skipped_waiting_requests（临时队列），稍后放回 waiting 头部重试。
                # 💡 skipped_waiting_requests.prepend_request(request)：避免饥饿，确保下次优先重试。
                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # 处理结构化输出（FSM 编译中）
                # 背景：
                # 当用户要求 结构化输出（如 JSON Schema、正则表达式），vLLM 需先编译成 有限状态机（FSM）。
                # 编译是异步的，可能尚未完成。
                # 逻辑：
                # 检查 FSM 是否已生成（grammar 是否存在）。
                # 若已完成 → 改为 WAITING，可调度。
                # 若未完成 → 暂时跳过，放入 skipped 队列。
                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # 检查 LoRA 限制
                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # 下面的目标：确定本次 prefill 需要计算多少新 token（num_new_tokens），并检查是否能复用已有 KV。
                num_external_computed_tokens = 0  # 从远程节点匹配到的已计算 token 数（如通过 KVTransfer）
                load_kv_async = False  # 是否需要异步加载远程 KV Cache（此时不分配新计算任务）

                # 场景一：全新请求（num_computed_tokens == 0）
                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # 获取本地已缓存的 block（Prefill 缓存复用）
                    # 作用：检查是否有完全相同的 prompt 已被计算过（例如相同输入多次请求）。
                    # 如果有：
                    # 返回已有的物理 block 列表（new_computed_blocks）
                    # 返回已计算 token 数（num_new_local_computed_tokens）
                    # 否则返回空 block 列表和 0。
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # 获取远程已缓存的 token（KVConnector / KVTransfer）
                    # 背景：在分布式推理或 KV 共享系统中（如 KVTransfer），其他节点可能已计算过该 prompt。
                    # connector 负责与远程节点通信，查询匹配的 token 数。
                    # 返回：
                    # num_external_computed_tokens：远程匹配的 token 数
                    # load_kv_async：是否需要异步拉取远程 KV（此时不能立即计算）
                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                        # 无法确定匹配数
                        # 可能因网络延迟、元数据未同步等，暂时无法判断。
                        # 将请求放入 skipped 队列，稍后重试。
                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # 计算总已计算 token 数
                    # 合并本地 + 远程已计算的部分。
                    # 本次只需计算剩余部分：request.num_tokens - num_computed_tokens
                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    # 场景二：非全新请求（num_computed_tokens > 0）
                    # 被抢占后恢复的请求（KVTransfer 场景）
                    # 在 KVTransfer 等系统中，被抢占的请求可能保留了 num_computed_tokens > 0（因为 KV 被保存/传输）。
                    # 此时不尝试复用本地缓存（因为 block 可能来自远程），直接使用已有值。
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # 正常情况：计算需处理的新 token
                    # request.num_tokens = prompt 长度 + 已生成输出长度（对 resumed 请求很重要）
                    # 减去已计算部分，得到真正需要 prefill 的 token 数
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.chunked_prefill_tail_optimization_factor * self.scheduler_config.long_prefill_token_threshold
                            <= num_new_tokens and self.chunked_prefill_enabled):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # Chunked Prefill 优化：长 prompt 截断
                    # 目的：避免一个超长 prompt（如 32k tokens）阻塞整个 batch。
                    # 如果启用 chunked_prefill，且剩余 token 数超过阈值，则只处理一个 chunk（如 2048 tokens）。
                    # 下次调度继续处理剩余部分。
                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_compute_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_compute_budget)
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens =\
                        self.scheduler_config.max_num_encoder_input_tokens
                else:
                    num_encoder_tokens = 0

                # todo 为request分配blocks
                # request	当前待调度的请求
                # num_new_tokens + num_external_computed_tokens	总共需要预留的 token slot 数
                # • num_new_tokens：本次要计算的新 token
                # • + num_external_computed_tokens：远程已匹配但需占位的 token（即使不计算，也要预留 block）
                # num_new_local_computed_tokens	本地缓存复用的 token 数（用于跳过计算，但 block 已存在）
                # new_computed_blocks	本地复用的 block 列表（来自 prompt caching）
                # num_lookahead_tokens	推测解码（speculative decoding）所需的额外 token 预留
                # delay_cache_blocks=load_kv_async	若为 True，表示 block 已分配但内容稍后异步加载（如 KVTransfer）
                # num_encoder_tokens	encoder-decoder 模型所需的 cross-attention block 数
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                logger.warning(f'===== class Scheduler.schedule(), 分配blocks, new_blocks={new_blocks}')

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()  # todo request从waiting队列弹出，此前只是 peek_request() 查看，block分配成功后，现在才真正移除。
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                # 正常情况：加入 running 队列
                # 请求正式进入运行状态，将参与本轮 GPU 前向计算。
                req_index += 1
                self.running.append(request)  # todo request加入running队列
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)  # 全新 prefill 请求
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)  # 被抢占后恢复的请求，后续构建 batch 时，可能对两类请求做不同处理（如 metrics 统计）
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens  # 扣减 token budget
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

        '''
        这段代码是 vLLM 调度器（Scheduler）在 schedule() 函数的收尾阶段，负责：
        恢复被跳过的请求
        执行调度约束断言（安全检查）
        计算公共前缀（用于 cascade attention 等优化）
        构造最终的 SchedulerOutput 对象
        集成 KV Connector 元数据和事件发布
        清理状态并返回结果
        它是整个调度流程的“出口”，将调度决策封装成结构化输出，供后续的 Worker / Engine Core 使用。
        '''
        # 在调度 waiting 队列时，某些请求因 暂时不可调度（如等待远程 KV、FSM 编译中、LoRA 超限等）被临时移出，放入 skipped_waiting_requests。
        # 这些请求不能丢弃，需在本轮调度结束后放回 waiting 队列头部，确保下次优先重试（避免饥饿）。
        # 为什么是 prepend（插入头部）？
        # 保证公平性：被跳过的请求应比新到达的请求更早被处理。
        # 符合 FCFS 或 Priority 等策略的语义。
        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # 计算公共前缀块数（Common Prefix Blocks）
        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # 构造 SchedulerOutput（核心输出）
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        scheduled_requests = (scheduled_new_reqs + scheduled_running_reqs +
                              scheduled_resumed_reqs)
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(scheduled_requests,
                                     scheduled_spec_decode_tokens))
        # todo 构造SchedulerOutput，包含 running batch 和 waiting batch
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,  # (1) 新请求数据（Prefill），即 prefill batch
            scheduled_cached_reqs=cached_reqs_data,  # (2)缓存请求数据（Decode / Resumed），即 decode batch / resume batch
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[Optional[tuple[list[int], ...]]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = (num_scheduled_tokens[req_id] -
                          len(spec_decode_tokens.get(req_id, ())))
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_tokens_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used.")
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_features[i].identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(
                        request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if not self.encoder_cache_manager.can_allocate(
                    request, i, encoder_compute_budget,
                    num_tokens_to_schedule):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            num_tokens_to_schedule += num_encoder_tokens
            encoder_compute_budget -= num_encoder_tokens
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
        )

    def get_grammar_bitmask(
        self,
        requests: list[Request],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to its index in the batch.
        # This will help us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}
        for i, req in enumerate(requests):
            if req.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[req.request_id] = i

        if not structured_output_request_ids:
            bitmask = None
        else:
            bitmask = self.structured_output_manager.grammar_bitmask(
                self.requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )
        return structured_output_request_ids, bitmask

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None
        kv_connector_stats = (kv_connector_output.kv_connector_stats
                              if kv_connector_output else None)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    ))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats,
                                     kv_connector_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
                draft_token_ids.req_ids,
                draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if not spec_token_ids:
                # NOTE(woosuk): request.spec_token_ids should be updated.
                request.spec_token_ids.clear()
            elif self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids)
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        #
        logger.warning(f'===== scheduler中的工作: 将请求放入waiting队列, self.waiting={self.waiting}')
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
        kv_connector_stats: Optional[KVConnectorStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(num_running_reqs=len(self.running),
                              num_waiting_reqs=len(self.waiting),
                              kv_cache_usage=self.kv_cache_manager.usage,
                              prefix_cache_stats=prefix_cache_stats,
                              spec_decoding_stats=spec_decoding_stats,
                              num_corrupted_reqs=sum(req.is_output_corrupted
                                                     for req in self.running),
                              kv_connector_stats=kv_connector_stats.data
                              if kv_connector_stats else None)

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less than one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in (kv_connector_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (kv_connector_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.warning(
                    "Got finished sending KV transfer for request %s,"
                    "but the request is already freed.", req_id)
            else:
                self._free_blocks(self.requests[req_id])
