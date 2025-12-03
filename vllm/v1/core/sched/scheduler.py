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

        """获取完整的调用栈"""
        import inspect
        import json
        stack = inspect.stack()
        stack_details = []

        for frame_info in stack[1:]:  # 跳过当前函数
            frame, filename, lineno, function, code_line, index = frame_info
            stack_details.append({
                'filename': filename,
                'line_number': lineno,
                'function': function,
                'code_line': code_line
            })
        logger.warning(
            f'===== Scheduler() stack_details={json.dumps(stack_details, indent=4)}')

        '''
        ===== Scheduler() stack_details=[
        {
            "filename": "/home/liudi/vllm/vllm/v1/engine/core.py",
            "line_number": 151,
            "function": "__init__",
            "code_line": [
                "        self.scheduler: SchedulerInterface = Scheduler(\n"
            ]
        },
        {
            "filename": "/home/liudi/vllm/vllm/v1/engine/core.py",
            "line_number": 542,
            "function": "__init__",
            "code_line": [
                "            super().__init__(vllm_config, executor_class, log_stats,\n"
            ]
        },
        {
            "filename": "/home/liudi/vllm/vllm/v1/engine/core.py",
            "line_number": 752,
            "function": "run_engine_core",
            "code_line": [
                "                engine_core = EngineCoreProc(*args, **kwargs)\n"
            ]
        },
        {
            "filename": "/usr/local/python3.11.13/lib/python3.11/multiprocessing/process.py",
            "line_number": 108,
            "function": "run",
            "code_line": [
                "            self._target(*self._args, **self._kwargs)\n"
            ]
        },
        {
            "filename": "/usr/local/python3.11.13/lib/python3.11/multiprocessing/process.py",
            "line_number": 314,
            "function": "_bootstrap",
            "code_line": [
                "                self.run()\n"
            ]
        },
        {
            "filename": "/usr/local/python3.11.13/lib/python3.11/multiprocessing/spawn.py",
            "line_number": 135,
            "function": "_main",
            "code_line": [
                "    return self._bootstrap(parent_sentinel)\n"
            ]
        },
        {
            "filename": "/usr/local/python3.11.13/lib/python3.11/multiprocessing/spawn.py",
            "line_number": 122,
            "function": "spawn_main",
            "code_line": [
                "    exitcode = _main(fd, parent_sentinel)\n"
            ]
        },
        {
            "filename": "<string>",
            "line_number": 1,
            "function": "<module>",
            "code_line": null
        }
    ]
        '''




        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        # kv_cache_config 是在 core.py中 _initialize_kv_caches 函数初始化的
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
        logger.warning(f'===== Scheduler, self.scheduler_config={self.scheduler_config}')
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs  # running队列中的请求数不能超过该限制
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens  # running batch 和 waiting batch 两个batch中加起来请求数不能超过该限制。token_budget 初始化为 max_num_scheduled_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.min_prefill_batch_size = self.scheduler_config.min_prefill_batch_size
        self.prefill_request_batching_timeout_ms = self.scheduler_config.prefill_request_batching_timeout_ms
        self.scheduler_delay_us = self.scheduler_config.scheduler_delay_us
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
        logger.warning(f'===== Scheduler, kv_cache_config={kv_cache_config}')
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


    # 计算一个prefill请求在组batch时需要等待的时间（单位：ms），可以立即组batch时，返回0
    def _compute_prefill_request_pending_delay_ms(self, scheduled_new_reqs:list[Request], req:Request, token_budget:int) -> int:
        logger.warning(f'===== compute_prefill_request_pending_delay_ms, req={req}, \n vllm_config={self.vllm_config}, \n kv_cache_config={self.kv_cache_config}')
        logger.warning(f'===== self.min_prefill_batch_size={self.min_prefill_batch_size}, self.prefill_request_batching_timeout_ms={self.prefill_request_batching_timeout_ms}, self.scheduler_delay_us={self.scheduler_delay_us}')
        # prefill请求包括：running队列中的chunk-prefill请求、waiting队列中的新请求、waiting队列中preempted请求（由于次判断在 not preempted_reqs 下，所以不用考虑此情况）
        # prefill 组batch时，对于每个请求的判断逻辑：
        # 只有当prefill_batch_size没达到min_prefill_batch_size，并且当前请求没有超时，这两个条件下，才会sleep，其他情况均放行

        # 是否sleep？对。ibis中使用的 scheduler_cv_.wait_for(lock, timeout)；
        if (len(scheduled_new_reqs) < self.scheduler_config.min_prefill_batch_size
                and (time.time() - req.arrival_time) * 1000 < self.scheduler_config.prefill_request_batching_timeout_ms):  # 统一用ms计算
            return self.scheduler_config.scheduler_delay_us

        return 0


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
        '''
        二、running 队列：存放哪些请求？
            running 队列保存当前已分配 KV Cache block、正在参与（或即将参与）本轮 GPU 推理的请求。
            
            ✅ 包含以下几类请求：
            请求类型	说明
            1. 正在 Decode 的请求	已完成 prefill，正在逐个生成输出 token（num_computed_tokens > prompt_len）。这是主体。
            2. 刚被调度进来的 Prefill 请求	在 schedule() 中刚从 waiting 移入 running，将在本轮执行 prefill。
            3. 被恢复（Resumed）的请求	之前因抢占被 swap out，现在 swap in 成功，重新加入 running 继续 decode。
            4. 推测解码中的目标/草稿请求	在 Speculative Decoding 模式下，主模型和草稿模型的请求都可能在 running 中协同调度。
            ⚠️ 注意：
            running 中的请求不一定都在本轮被实际计算！
            例如：token budget 用完后，后面排队的 running 请求会“挂起”，留到下一轮。
            请求在 running 中时，必须已分配 KV Cache block（无论在 GPU 还是 CPU swap）。
            🔹 关键特征：request.status == RequestStatus.RUNNING
            
            （但在调度过程中，状态可能临时为 WAITING_FOR_REMOTE_KVS 等，随后转为 RUNNING）
        '''
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
        # 此时的running队列中，没有未处理的prefill请求了，上次调度从waiting队列加入到running队列中的prefill请求，已经推理一轮了，然后再进入当前调度中。
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
            用于计算 当前调度轮次中需要为该请求新分配 KV Cache slots 的 token 数量。这是调度器（Scheduler）在决定如何分配显存资源时的关键逻辑。

            下面逐个解释这三个属性的含义，并说明为什么这样计算。
            
            🔍 一、三个属性详解
            1. request.num_tokens_with_spec
            含义：当前请求在考虑 speculative decoding（推测解码）后的总 token 数。
            组成：
            已输入的 prompt tokens
            已生成的 output tokens
            + 推测解码中“预测”的 future tokens（draft tokens）
            目的：为 speculative decoding 预留额外的 KV Cache 空间。
            💡 如果未启用 speculative decoding，则 num_tokens_with_spec == request.num_tokens（即普通总 token 数）。
            
            2. request.num_output_placeholders
            含义：为未来输出预留的“占位符” token 数量。
            典型值：通常等于 max_tokens（用户指定的最大生成长度）或动态预分配值。
            作用：
            在 prefill 阶段就预先分配 decode 阶段可能需要的 block slots
            避免在 decode 过程中频繁申请显存（提升性能）
            尤其在 PagedAttention + BlockSpaceManager 中用于预分配物理 blocks
            ✅ 这是一种 “预分配”优化策略，减少运行时内存碎片和分配开销。
            
            📌 注意：这些 placeholder 尚未真实生成，只是预留位置。
            
            3. request.num_computed_tokens
            含义：该请求已经完成计算（或已加载 KV Cache）的 token 数量。
            包括：
            Prefill 阶段已处理的 prompt tokens
            Decode 阶段已生成的 output tokens
            （如果启用了 prefix cache）命中的缓存 tokens
            不包括：
            尚未处理的 prompt tokens
            未生成的 output tokens
            speculative decoding 的 draft tokens（除非已验证）
            ✅ 这个值用于追踪请求的进度。
            
            🧮 二、为什么这样计算 num_new_tokens？
            公式：
            num_new_tokens = (num_tokens_with_spec + num_output_placeholders) - num_computed_tokens
            逻辑解释：
            “总共需要的空间” 减去 “已经有的空间” = “还需要新分配的空间”
            
            项	说明
            num_tokens_with_spec	当前已知的所有 token（含 speculative 预测）
            + num_output_placeholders	再加上为未来 output 预留的 slot（保守预分配）
            - num_computed_tokens	减去已经分配并使用的 token 数
            ✅ 结果就是：本次调度需要新申请的 KV Cache slots 对应的 token 数。
            
            📊 三、举个实际例子
            假设一个请求：
            
            Prompt 长度：100 tokens
            用户设置 max_tokens=50（最多生成 50 个 output）
            已生成 10 个 output tokens
            启用了 speculative decoding，当前预测了 5 个 draft tokens
            调度器预分配了全部 50 个 output 的 placeholders
            则：
            
            num_tokens_with_spec = 100 (prompt) + 10 (generated) + 5 (draft) = 115
            num_output_placeholders = 50（预分配的 output slots）
            num_computed_tokens = 100 + 10 = 110（已计算的 prompt + output）
            计算：
            num_new_tokens = (115 + 50) - 110 = 55
            但注意：这 55 包含了：
            
            5 个 draft tokens 的 slots
            40 个未生成 output 的 placeholder slots（50 - 10 已生成）
            💡 实际实现中，num_output_placeholders 可能只指 尚未分配的 placeholder，具体取决于版本。但在多数情况下，它代表总的预分配 output 长度。
            
            ⚠️ 四、注意事项
            num_output_placeholders 并非总是等于 max_tokens
            可能受 --max-model-len 或 block manager 策略限制
            有些版本中，它只表示 本次调度要预分配的数量
            Speculative decoding 是关键触发条件
            若未启用 spec decode，num_tokens_with_spec ≈ num_computed_tokens + 1（下一个 token）
            这个值用于 allocate_slots()
            最终传给 allocate_slots(num_new_tokens=...) 来申请 GPU 显存
            ✅ 五、总结
            属性	含义	是否包含 speculative	是否包含预分配
            num_tokens_with_spec	当前总 token（含 draft）	✅ 是	❌ 否
            num_output_placeholders	为 output 预留的 slot 数	❌ 否	✅ 是
            num_computed_tokens	已计算/加载的 token 数	❌ 否（draft 未验证不算）	❌ 否
            💡 设计目的：
            
            提前为 speculative decoding 和未来 output 分配足够 KV Cache，避免运行时 OOM 或频繁分配，同时精确计算所需新 slots。
            
            这是 vLLM 实现 高性能、支持推测解码、预分配优化 的核心调度逻辑之一。
            '''
            '''
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:46 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=19, request.num_output_placeholders=0, request.num_computed_tokens=18
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:46 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=20, request.num_output_placeholders=0, request.num_computed_tokens=19
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=21, request.num_output_placeholders=0, request.num_computed_tokens=20
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=22, request.num_output_placeholders=0, request.num_computed_tokens=21
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=23, request.num_output_placeholders=0, request.num_computed_tokens=22
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=24, request.num_output_placeholders=0, request.num_computed_tokens=23
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=25, request.num_output_placeholders=0, request.num_computed_tokens=24
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=26, request.num_output_placeholders=0, request.num_computed_tokens=25
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=27, request.num_output_placeholders=0, request.num_computed_tokens=26
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=28, request.num_output_placeholders=0, request.num_computed_tokens=27
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=29, request.num_output_placeholders=0, request.num_computed_tokens=28
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=30, request.num_output_placeholders=0, request.num_computed_tokens=29
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=31, request.num_output_placeholders=0, request.num_computed_tokens=30
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=32, request.num_output_placeholders=0, request.num_computed_tokens=31
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=33, request.num_output_placeholders=0, request.num_computed_tokens=32
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:47 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=34, request.num_output_placeholders=0, request.num_computed_tokens=33
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=35, request.num_output_placeholders=0, request.num_computed_tokens=34
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=36, request.num_output_placeholders=0, request.num_computed_tokens=35
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=37, request.num_output_placeholders=0, request.num_computed_tokens=36
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=38, request.num_output_placeholders=0, request.num_computed_tokens=37
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=39, request.num_output_placeholders=0, request.num_computed_tokens=38
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=40, request.num_output_placeholders=0, request.num_computed_tokens=39
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=41, request.num_output_placeholders=0, request.num_computed_tokens=40
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=42, request.num_output_placeholders=0, request.num_computed_tokens=41
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=43, request.num_output_placeholders=0, request.num_computed_tokens=42
            (EngineCore_DP0 pid=72223) WARNING 11-27 07:33:48 [scheduler.py:436] ===== schedule running, request.num_tokens_with_spec=44, request.num_output_placeholders=0, request.num_computed_tokens=43
            '''
            logger.warning(f'===== schedule running, request.num_tokens_with_spec={request.num_tokens_with_spec}, '
                           f'request.num_output_placeholders={request.num_output_placeholders}, request.num_computed_tokens={request.num_computed_tokens}')
            # request.num_computed_tokens 包含了prefix-cache匹配到的tokens，是在waiting队列处理时的如下代码设置的：
            # num_computed_tokens = (num_new_local_computed_tokens + num_external_computed_tokens)
            num_new_tokens = (request.num_tokens_with_spec +  # 本次调度，该请求推理后总共的token数 - 该请求现已分配的token数
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
            # 开启chunk-prefill后，计算了一部分chunk的请求，会一直在running队列中
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

                # 2、物理block分配失败，显存不足，抢占，放回waiting队列头部，释放blocks
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
        '''
        一、waiting 队列：存放哪些请求？
            waiting 队列保存尚未开始任何 GPU 计算（或尚未进入 running 状态）的请求。
            
            ✅ 包含以下几类请求：
            请求类型	说明
            1. 新到达的 Prefill 请求	用户刚提交的请求，prompt 尚未被处理（即 num_computed_tokens == 0）。这是最常见的类型。
            2. 被抢占（Preempted）后需重新 Prefill 的请求	如果一个请求因显存不足被 swap out 或 recompute preempted（非保留 KV Cache），则下次调度时需从头开始 prefill，会回到 waiting。
            3. 因资源不足被跳过的请求	例如：
            • LoRA adapter 超限
            • token budget 不足
            • 需要异步加载远程 KV（如 KVTransfer）但尚未就绪
            这些请求会被临时放入 skipped_waiting_requests，随后放回 waiting 头部重试。
            4. 推测解码中草稿模型失败后需 fallback 的请求	（较少见）某些 speculative decoding 实现可能将验证失败的请求暂退回到 waiting。
            ❌ 不包含：
            已经部分生成 token 的 decode 请求（除非被完全抢占且无缓存）
            正在运行的请求
            已完成的请求
            🔹 关键特征：request.status == RequestStatus.WAITING
        '''
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
            # 本轮调度中，prefill batch当前需要新的token数
            # # min_prefill_batch_size 限制时是否考虑 chunk-prefill 请求，暂时不考虑
            # num_chunk_prefill_reqs = len([0 for req in scheduled_running_reqs if (req.num_computed_tokens < len(req.prompt_token_ids))])
            while self.waiting and token_budget > 0:
                logger.warning(f'===== class Scheduler.schedule(), 处理waiting队列, token_budget={token_budget}, self.waiting={self.waiting}')
                # 每循环1个请求，running队列都可能新增请求，所以要检查 running 队列是否已达到max_num_running_reqs限制，即使还有 token 预算，也不能超过最大并发请求数。这是为了控制 GPU 上下文切换开销 和 调度复杂度
                if len(self.running) == self.max_num_running_reqs:  # running队列是现在正处于推理阶段的请求数，限制该数量不能超过配置值
                    break

                # todo 这里就是从waiting队列中获取请求，waiting队列有多种实现算法：FCFS、Priority、SJF、SJFInHeap
                request = self.waiting.peek_request()  # 查看下一个待调度请求（不弹出），后续根据资源检查结果，决定是否真正弹出并调度。

                # 达到 max_prefill_batch_size，停止waiting队列调度
                if len(scheduled_new_reqs) > self.scheduler_config.max_prefill_batch_size:
                    logger.warning(f'===== reach max_prefill_batch_size，break waiting_queue schedule')
                    break
                # 判断请求是否可以立即组batch，或者等待固定时间
                delay_us = self._compute_prefill_request_pending_delay_ms(scheduled_new_reqs, request, token_budget)
                if delay_us > 0:
                    logger.warning(f'===== prefill request pending delay, curr_time: {time.time()} s')
                    time.sleep(delay_us / 1_000_000)
                    continue

                '''
                RequestStatus.WAITING_FOR_REMOTE_KVS状态值介绍
                
                出现在调度器（Scheduler）或请求状态管理逻辑中，其目的是 处理分布式推理场景下 KV Cache 跨节点传输的等待状态。这是 vLLM 支持多机（multi-node）推理 的关键机制之一。

                🔍 背景：什么是 “Remote KV Cache”？
                在 单机 vLLM 中，所有请求的 KV Cache 都存储在本地 GPU 显存中。
                
                但在 分布式 vLLM（如使用 Ray 或自定义多机后端） 中：
                
                一个请求的 KV Cache 可能被卸载（offload）到远程节点
                当该请求需要继续生成（decode）时，必须 先从远程节点拉取 KV Cache 回本地
                此时，请求会进入 WAITING_FOR_REMOTE_KVS 状态，暂停调度，直到 KV Cache 传输完成。
                
                ✅ 什么场景会走到这个分支？
                场景 1：多机推理 + KV Cache 卸载（Offloading）
                当 GPU 显存不足时，vLLM 可能将部分 swapped 请求的 KV Cache 存储到 其他机器的 CPU/GPU 内存（而非本地磁盘）
                下次调度该请求时，需先 异步拉取远程 KV Cache
                在拉取完成前，请求状态设为 WAITING_FOR_REMOTE_KVS
                调度器检测到此状态，跳过该请求，不将其加入当前 batch
                📌 这是 分布式 PagedAttention 的扩展行为（目前 vLLM 官方主干尚未完全开源多机版，但企业版或研究分支支持）
                
                场景 2：Speculative Decoding + 辅助模型在远程
                在某些 speculative decoding 架构中，draft model 可能在另一台机器上运行
                验证阶段需要同步 KV 状态，也可能触发远程 KV 等待
                （较少见，属于高级用法）
                
                场景 3：自定义后端或实验性功能
                如果你在使用 vLLM 的 fork 分支（如阿里、NVIDIA 内部版本），可能实现了 跨节点 KV Cache 共享
                此状态用于协调多机间的请求调度
                🧠 为什么需要这个状态？
                问题：如果没有 WAITING_FOR_REMOTE_KVS
                调度器会尝试调度一个 KV Cache 不在本地 的请求
                模型执行时发现 block_tables 指向的物理页不在本地 → 崩溃或错误
                解决方案：
                请求被 swap out 到远程节点时，状态设为 WAITING_FOR_REMOTE_KVS
                后台异步任务开始拉取 KV Cache
                拉取完成后，状态变回 RUNNING 或 SWAPPED
                调度器下次轮询时正常调度
                这保证了 “调度的请求，其 KV Cache 一定可用”。
                
                
                处理远程 KV Cache 依赖（KVTransfer 场景）
                背景：
                在 分布式推理 或 KV Cache 共享 场景中，某些请求需等待远程节点发送 KV Cache。
                状态为 WAITING_FOR_REMOTE_KVS 表示“正在等远程数据”。
                逻辑：
                调用 _update_waiting_for_remote_kv() 检查是否已收到数据。
                若已就绪 → 改为 WAITING，本次可调度。
                若未就绪 → 弹出该请求，放入 skipped_waiting_requests（临时队列），稍后放回 waiting 头部重试。
                💡 skipped_waiting_requests.prepend_request(request)：避免饥饿，确保下次优先重试。
                KVTransfer: skip request if still waiting for remote kvs.
                '''
                # 判断检查请求是否正在等待远程kv-cache传输过来
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self.waiting.pop_request()  # 该请求的远程kv-cache没有传输过来，则将该请求踢出waiting队列
                        skipped_waiting_requests.prepend_request(request)  # 暂时放入 skipped_waiting_requests 这个临时队列中，该队列放的是本轮调度中被忽略的请求，待本轮调度完成后，skipped_waiting_requests中请求重新加到waiting队列中
                        continue

                '''
                RequestStatus.WAITING_FOR_FSM状态介绍 
                这通常表示该请求正在 等待有限状态机（FSM, Finite State Machine）的约束处理完成，尤其与 结构化输出（structured output） 或 词法/语法引导生成（grammar-guided generation） 相关。

                🔍 背景：什么是 FSM 在 LLM 生成中的作用？
                在 vLLM 中，FSM 一般指 基于语法规则（如 JSON Schema、正则表达式、EBNF grammar）构建的有限状态机，用于 约束 LLM 的输出，确保生成内容符合特定格式。
                
                例如：
                强制模型输出合法 JSON
                限制聊天机器人的回复只能是预定义选项
                生成符合 SQL 语法的查询
                vLLM 通过集成 outlines 或自研 FSM 引擎，在 token 采样阶段动态剪枝 logits，只允许生成符合 FSM 当前状态的 token。
                
                ✅ WAITING_FOR_FSM 状态的含义
                当一个请求设置了 结构化输出约束（如 guided_decoding），但在当前调度轮次中：
                
                FSM 尚未构建完成（异步构建中），或
                FSM 状态更新被延迟（例如依赖上一轮生成的 token 来推进状态）
                此时，请求会被置为：RequestStatus.WAITING_FOR_FSM
                调度器（Scheduler）检测到此状态后，会 暂时跳过该请求，不将其加入当前 batch，直到 FSM 准备就绪。
                
                🧩 典型触发场景
                场景 1：首次 Prefill 阶段构建 FSM
                用户发起一个带 guided_json={...} 的请求
                vLLM 需要根据 JSON Schema 异步编译 FSM
                在 FSM 编译完成前，请求状态设为 WAITING_FOR_FSM
                编译完成后，状态恢复为 RUNNING，进入正常调度
                💡 为什么异步？避免阻塞主线程，提升吞吐。
                
                场景 2：Decode 阶段等待 FSM 状态推进
                上一轮生成了一个 token（如 {）
                FSM 需要根据这个 token 计算下一个允许的 token 集合
                如果该计算被延迟（如批量处理 FSM 更新），请求会短暂进入等待状态。
                
                
                处理结构化输出（FSM 编译中）
                背景：
                当用户要求 结构化输出（如 JSON Schema、正则表达式），vLLM 需先编译成 有限状态机（FSM）。
                编译是异步的，可能尚未完成。
                逻辑：
                检查 FSM 是否已生成（grammar 是否存在）。
                若已完成 → 改为 WAITING，可调度。
                若未完成 → 暂时跳过，放入 skipped 队列。
                
                
                ✅ 总结
                问题	                    答案
                WAITING_FOR_FSM 是什么？	请求正在等待 结构化输出 FSM（有限状态机）准备就绪
                为什么需要等待？	        FSM 需要根据 schema 异步构建，或根据上一轮输出更新状态
                什么功能会触发？	        使用 guided_json / guided_regex / guided_grammar 等 引导生成（guided decoding） 功能
                开源版会走到这里吗？	    ✅ 会，如果你启用了 guided decoding（vLLM 已支持 outlines 集成）
                💡 提示：该状态是 临时性、短暂的，正常情况下会在 1~2 个调度周期内恢复为 WAITING。
                
                这种设计使得 vLLM 能在 不牺牲吞吐的前提下，安全地支持复杂的结构化输出约束。
                '''
                # 判断FSM异步构建是否完成
                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()  # FSM异步构建没完成，则当前请求从waiting队列弹出
                        skipped_waiting_requests.prepend_request(request)  # 放入临时队列 skipped_waiting_requests，待本轮调度完成后，skipped_waiting_requests中请求重新加到waiting队列中
                        continue

                '''
                LoRA介绍
                vLLM 的调度器（Scheduler） 中，用于在 启用 LoRA（Low-Rank Adaptation）微调模型 的场景下，控制同时激活的 LoRA 适配器数量不超过限制。
                
                条件拆解：
                子条件	                                                    含义
                self.lora_config	                                        当前引擎启用了 LoRA 支持（即传入了 --enable-lora 等参数）
                request.lora_request	                                    当前请求指定了要使用的 LoRA 适配器（如通过 API 传了 lora_name）
                len(scheduled_loras) == self.lora_config.max_loras	        当前已调度的 LoRA 适配器数量已达上限（例如最多同时加载 2 个 LoRA）
                request.lora_request.lora_int_id not in scheduled_loras	    当前请求所需的 LoRA 不在已调度的集合中
                ✅ 整个 if 成立的含义是：
                
                “这个请求需要一个 LoRA，但该 LoRA 没被加载，且系统已满载（无法再加载新 LoRA）”
                
                🎯 这段代码的作用：跳过无法调度的 LoRA 请求
                当上述条件为真时，调度器会 暂时不调度该请求（通常将其保留在 waiting 队列中），直到：
                
                已加载的某个 LoRA 被释放（对应请求完成）
                有空位可以加载这个新的 LoRA
                这是为了满足 vLLM 的 LoRA 内存管理约束。
                
                🧠 背景知识：vLLM 如何支持多 LoRA？
                vLLM 支持 动态切换多个 LoRA 适配器，但受 GPU 显存限制，不能无限加载。因此引入两个关键配置：
                
                配置项	说明
                max_loras	最多同时加载多少个 LoRA 适配器（默认 1）
                max_cpu_loras	最多缓存多少个 LoRA 在 CPU（可选）
                每个 LoRA 会被分配一个唯一的 lora_int_id（整数 ID）
                调度器维护一个集合 scheduled_loras，记录当前 GPU 上已加载的 LoRA ID
                所有请求共享这 max_loras 个“插槽”
                '''
                # 判断一个带 LoRA 的请求是否因 LoRA 插槽已满且所需 LoRA 未加载 而无法调度
                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()  # 该请求为LoRA请求，因为条件不满足而无法加入batch中，弹出waiting队列
                    skipped_waiting_requests.prepend_request(request)  # 放入临时队列 skipped_waiting_requests，待本轮调度完成后，skipped_waiting_requests中请求重新加到waiting队列中
                    continue

                # 下面的目标：确定本次 prefill 需要计算多少新 token（num_new_tokens），并检查是否能复用已有 KV。
                num_external_computed_tokens = 0  # 从远程节点匹配到的已计算 token 数（如通过 KVTransfer）
                load_kv_async = False  # 是否需要异步加载远程 KV Cache（此时不分配新计算任务）

                # 场景一：全新请求（num_computed_tokens == 0）
                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # 本地prefix-cache
                    # 获取本地已缓存的 block（Prefill 缓存复用：prefix-cache）
                    # 作用：检查是否有完全相同的 prompt 已被计算过（例如相同输入多次请求）。
                    # 如果有：
                    # new_computed_blocks：prefix-cache命中的block列表。后续新allocate出来的blocks，会向后追加。
                    # num_new_local_computed_tokens：已计算的token数。 new_computed_blocks 和 num_new_local_computed_tokens 这二者是一致的：num_new_local_computed_tokens = new_computed_blocks * block_size
                    # 否则返回空 block 列表和 0。
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # 远程prefix-cache
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

                    # 计算总已计算 token 数（即：总cache的token数 = 本地cache的token数 + 远程cache的token数）
                    # 合并本地 + 远程已计算的部分。
                    # 本次只需计算剩余部分：request.num_tokens - num_computed_tokens
                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)

                # '''
                # 在 vLLM 的调度器（Scheduler） 中，waiting 队列里出现 request.num_computed_tokens > 0 的请求，通常表示这是一个 已经被部分处理过、但由于某种原因被中断或暂停，尚未完成 Prefill 阶段的请求。
                # 这类请求 不是全新的请求，而是处于 “部分 Prefill 已完成” 的中间状态。
                #
                # ✅ 什么情况下 waiting 队列中的请求会有 num_computed_tokens > 0？
                # 情况 1️⃣：启用了 Chunked Prefill（分块预填充）
                # 这是最常见的原因。
                #
                # 当 prompt 很长（如 32k tokens），而当前 batch 的 token budget 不足时
                # vLLM 会 只处理 prompt 的一部分（一个 chunk）
                # 处理完后：
                # request.num_computed_tokens += chunk_size
                # 请求 未进入 running，而是 放回 waiting 队列尾部
                # 下次调度继续处理剩余部分
                # 📌 示例：
                #
                # python
                # 编辑
                # request.prompt = "A" * 50000  # 50k tokens
                # scheduler.max_num_batched_tokens = 8192
                # → 第一次调度处理前 8192 tokens → num_computed_tokens = 8192
                # → 请求仍在 waiting 队列，等待下一轮处理 8193~16384...
                #
                # ✅ 这是正常行为，是 vLLM 支持超长 prompt 的关键机制。
                #
                # 情况 2️⃣：Prefill 被抢占（Preemption）
                # 当系统资源紧张（如 GPU 显存不足）
                # 调度器可能 暂停一个正在 Prefill 的请求
                # 将其 KV Cache swap out 到 CPU
                # 请求状态变回 waiting，但保留 num_computed_tokens
                # ⚠️ 注意：这种情况较少见，因为 Prefill 通常优先级高，但极端负载下可能发生。
                #
                # 情况 3️⃣：Prefix Cache 加载 + 部分计算
                # 请求命中部分 prefix cache（如前 1000 tokens）
                # 但剩余部分因资源限制未能一次性完成
                # num_computed_tokens = 1000 + 已计算的新 tokens
                # 🧠 技术细节：num_computed_tokens 的含义
                # 表示 该请求已经成功计算（或加载）的 token 数量
                # 包括：
                # 从 prefix cache 命中的 token
                # 从外部缓存（CPU）加载的 token
                # 本地新计算的 token
                # 不包括 尚未处理的 prompt token
                # 在 SequenceGroup 或 Request 对象中维护，用于：
                #
                # 决定下一次 Prefill 的起始位置
                # 分配正确的 block slots
                # 计算剩余工作量
                # '''
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    # 场景二：非全新请求（request.num_computed_tokens > 0）
                    # 1、被抢占并且换出block（重计算场景下request.num_computed_tokens == 0）：
                    # 被抢占后恢复的请求（KVTransfer 场景，即通过KVTransfer将block swapout远端卡上）
                    # 在 KVTransfer 等系统中，被抢占的请求可能保留了 num_computed_tokens > 0（因为 KV 被保存/传输）。
                    # 此时不尝试复用本地缓存（因为 block 可能来自远程），直接使用已有值。
                    # 2、没有已部分计算的chunk-prefill请求，放在running队列中了

                    '''
                    在启用 chunked prefill 时，new_computed_blocks 被初始化为空（create_empty_block_list()），
                    是因为当前 chunk 的计算结果尚未生成 KV blocks，而之前 chunk 的 blocks 已经通过 request.block_tables 
                    持久化保存，不需要通过 new_computed_blocks 传递。
                    '''
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())  # todo 这里怎么是空的？chunk-prefill之前的算的block不需要记录下来吗？答：本次chunk的计算不需要知道前面chunk算出来的blocks，所以new_computed_blocks是空的，前面算出来的blocks已经存在block_table中了。
                    num_new_local_computed_tokens = 0  # todo 这个是？ 该请求本地已计算的token数。注意：new_computed_blocks 和 num_new_local_computed_tokens 保持一致。
                    num_computed_tokens = request.num_computed_tokens  # 该请求总共已计算的token数。这里 request.num_computed_tokens 就是前面chunk已经算完的token数量

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                '''
                如果当前正在异步加载远程 KV Cache，则本次调度轮次中不再为该请求分配新的计算任务（即不处理新的 tokens）。
                问题	                            答案
                这段代码的作用？	                在异步加载远程 KV Cache 期间，禁止为该请求分配新的计算任务
                为什么设 num_new_tokens = 0？	    避免在 KV 未就绪时访问无效内存，保证推理正确性
                什么场景会触发？	                分布式多机推理 中，KV Cache 存储在远程节点
                开源 vLLM 会走到这里吗？	        ❌ 通常不会（除非启用实验性多机后端）
                对用户的影响？	                    请求会短暂延迟，直到远程 KV 加载完成
                '''
                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0  # todo
                # Number of tokens to be scheduled.
                else:
                    # 正常情况：计算需计算的新 token
                    # request.num_tokens = prompt 长度 + 已生成输出长度（对 resumed 请求很重要）
                    # 减去已计算部分，得到真正需要 prefill 的 token 数
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    # 如果当前请求时新请求，并且开启了chunk-prefill，当 num_new_tokens 超过chunk-prefill的上限值 long_prefill_token_threshold 时，则将 num_new_tokens 设置为上限值 long_prefill_token_threshold
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
                        num_new_tokens > token_budget:  # 如果没开启chunk-prefill， num_new_tokens 比配额 token_budget 大，则跳过当前请求
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)  # 此处为什么取二者较小值，考虑开启chunk-prefill时，num_new_tokens一般会很大，则会截取到token_budget进行计算（显存打满）
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
                # num_new_tokens + num_external_computed_tokens	总共需要预留的 token slot 数 （该请求需要计算的token数 + 该请求需要从远端拉取的token数）
                # • num_new_tokens：本次要计算的新 token
                # • + num_external_computed_tokens：远程已匹配但需占位的 token（即使不计算，也要预留 block，等待异步传输到来，放在这里）
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

                if new_blocks is None:  # block分配失败，说明空间不足，停止组batch
                    # The request cannot be scheduled.
                    break

                # 当前请求的本地blocks分配成功后，更新block_table，拉取远端blocks
                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    # update_state_after_alloc 是 Connector 模块（通常指 PrefixCachingKVCacheManager 或类似组件）的一个方法，职责是：
                    # 将本次分配/加载的 KV blocks 合并到请求的 block table 中，并更新缓存元数据。
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # request从waiting队列弹出，此前只是 peek_request() 查看，block分配成功后，现在才真正移除。
                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                # 如果当前请求需要异步加载远程 KV Cache，则：
                # 不进行实际计算
                # 将其状态设为 WAITING_FOR_REMOTE_KVS
                # 放回 waiting 队列头部（prepend）以便快速重试
                # 跳过本轮调度的后续处理（continue）
                if load_kv_async:  # 条件判断：当前请求是否正在 异步加载远程 KV Cache。load_kv_async 通常由前序逻辑设置，例如：检测到该请求的部分 KV blocks 存在于 CPU 内存（offloaded）；或在 分布式推理 中，KV 存储在其他节点。
                    # 注释说明：此时会 分配 GPU 显存槽位（slots），但 不执行计算，只等待远程数据加载完成。
                    # 注意：虽然注释说 “allocate memory”，但实际分配可能已在之前完成（如通过 allocate_slots 预留 block），此处重点是 状态切换。
                    # 将请求 插入到 waiting 队列的头部（prepend），而不是尾部。
                    # 为什么放头部？
                    # 远程 KV 加载通常是 高优先级阻塞操作
                    # 放头部可让该请求在 下一轮调度中优先被检查
                    # 一旦 KV 加载完成，能立即进入 running 状态，减少延迟
                    # ✅ 这是一种 “快速重试”策略，避免长延迟等待。
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    # 显式设置请求状态为 WAITING_FOR_REMOTE_KVS
                    # 作用：
                    # 调度器后续轮次可根据此状态决定是否检查 KV 是否就绪
                    # 防止重复触发加载逻辑
                    # 便于监控和调试（如日志、metrics）
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    # todo 跳过本轮对该请求的后续处理（如 prefill/decode 计算），因为 KV 尚未就绪，无法安全执行前向计算。因为while循环中，优先判断请求状态是否为 WAITING_FOR_REMOTE_KVS，是则continue继续等待，直到传输完成。
                    continue

                # 正常情况：加入 running 队列
                # 请求正式进入运行状态，将参与本轮 GPU 前向计算。
                req_index += 1
                self.running.append(request)  # todo request加入running队列
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                '''
                在 vLLM 的调度器（Scheduler） 中，处理 waiting 队列时使用两个列表：
                scheduled_new_reqs
                scheduled_resumed_reqs
                是为了 区分两类语义和资源需求不同的请求，从而实现更精细的调度控制、性能优化和状态管理。
                类型	                    含义	                                特点
                scheduled_new_reqs	    全新请求
                                        （首次进入系统，Prefill 尚未开始）	    - 需要完整 Prefill
                                                                            - 无任何 KV Cache
                                                                            - 可能命中 Prefix Cache
                scheduled_resumed_reqs	已部分处理、被中断后恢复的请求
                                        （如被抢占、Chunked Prefill 中间状态）	- 已有部分 KV Cache
                                                                            - 只需继续 Prefill 剩余部分 或 进入 Decode
                                                                            - 状态为 RESUMING
                                                                            
                                                                            
                问题	                            答案
                为什么用两个 list？	            因为 新请求 和 恢复请求 在资源需求、状态、处理逻辑上存在本质差异
                不分开会怎样？	                    逻辑耦合、状态混乱、资源分配错误、性能下降
                resumed_reqs 主要指什么？	        被 抢占（preempted）后恢复 的请求，状态为 RESUMING
                Chunked Prefill 属于哪一类？	    通常仍走 new_reqs 路径（因其未被抢占，只是分块）
                对用户有影响吗？	                ❌ 无感知，但提升了系统稳定性和调度效率             
                
                设计哲学：
                “相同行为归为一类，不同行为分开处理” —— 这是构建高可靠调度系统的基本原则。   
                通过这种分离，vLLM 能够同时高效支持 高吞吐新请求 和 低延迟恢复请求，兼顾性能与鲁棒性。                                                            
                '''
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)  # 全新 prefill 请求，blocks是空的，加入 scheduled_new_reqs（即 prefill_batch）
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)  # todo PREEMPTED状态是上面处理running时设置的，blocks也是空的，和WAITING状态有什么区别吗？          被抢占后恢复的请求，后续构建 batch 时，可能对两类请求做不同处理（如 metrics 统计）
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
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())  # 本轮调度需要新分配的tokens总数。num_scheduled_tokens 中包含 running batch 和 waiting batch 的请求
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        # scheduled_new_reqs：waiting队列中新请求
        # scheduled_resumed_reqs：waiting队列中被抢占的请求
        # scheduled_running_reqs：running队列中组batch的请求
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
            for req in scheduled_new_reqs  # scheduled_new_reqs waiting队列中新请求
        ]

        '''
        _make_cached_request_data 函数的作用是：为当前调度轮次（scheduling iteration）中所有被选中的请求，
        预计算并缓存一批与模型执行（forward）相关的元数据和张量信息，以避免在 model_runner 中重复计算，提升性能。

        ✅ 核心目的
        将调度器（Scheduler）的调度结果 → 转换为 ModelRunner 所需的、可高效执行的中间表示（cached data）
        
        这是 调度器与执行引擎之间的关键桥梁。
        
        🔍 参数详解
        参数	类型	含义
        scheduled_running_reqs	        List[Request]	            当前正在运行的请求（包括 decode 和 chunked prefill）
        scheduled_resumed_reqs	        List[Request]	            从 waiting 队列恢复的请求（如被抢占后 resume 的请求）
                                                                    （注：在较新版本中，可能已合并到 running）
        num_scheduled_tokens	        int                         本轮 batch 中总 token 数（用于分配 KV Cache 等）
        scheduled_spec_decode_tokens	Optional[int]	            如果启用了 推测解码（Speculative Decoding），表示用于验证的 token 数
        req_to_new_blocks	            Dict[Request, List[int]]	每个请求新分配的 PagedAttention block IDs（用于构建 block_tables）
        
        这些数据会被直接传给 ModelRunner.execute_model()，用于构造 CUDA kernel 输入。

        ⚙️ 为什么需要“缓存”？
        避免重复计算
        Position IDs、block tables 等在调度后是确定的，提前算好避免在 GPU 启动前临时计算。
        
        解耦调度器与执行器
        Scheduler 只负责逻辑调度，ModelRunner 只负责执行，中间通过 cached_reqs_data 传递数据。
        
        支持异步/批处理优化
        所有元数据一次性准备好，便于后续张量化（如 torch.tensor(block_tables)）。
        
        该函数通常在 Scheduler.schedule() 的末尾 被调用，然后 LLMEngine 将 cached_reqs_data 传给 ModelRunner。
        
        问：为什么函数 _make_cached_request_data 的参数没有 scheduled_new_reqs ？
        答：因为 scheduled_new_reqs 不需要缓存数据？待确认。
        '''
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        # 本次调度完成，需要送给model进行推理的请求 scheduled_requests，分为三部分：
        # scheduled_new_reqs：waiting队列中的新请求
        # scheduled_resumed_reqs：waiting队列中被抢占的请求
        # scheduled_running_reqs：running队列中组batch的请求
        scheduled_requests = (scheduled_new_reqs + scheduled_running_reqs +
                              scheduled_resumed_reqs)
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(scheduled_requests,
                                     scheduled_spec_decode_tokens))
        # todo 构造SchedulerOutput，包含 running batch 和 waiting batch
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,  # (1) 新请求数据（Prefill），scheduled_new_reqs
            scheduled_cached_reqs=cached_reqs_data,  # (2)缓存请求数据（Decode/chunk-prefill + Resumed），即上面 scheduled_running_reqs + scheduled_resumed_reqs
            num_scheduled_tokens=num_scheduled_tokens,  # （1）和（2）加一起的tokens，按请求记录
            total_num_scheduled_tokens=total_num_scheduled_tokens,  #  （1）和（2）加一起的token数
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
        sampled_token_ids = model_runner_output.sampled_token_ids  # model一轮推理完，得到logits，对logits采样，得到token_id
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
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():  # todo num_scheduled_tokens？
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

            stopped = False  # stopped 表示该请求是否推理结束
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(  # todo 将推理新生成的token_ids追加到request中
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
        stopped = False  # stopped 表示该请求是否推理结束
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # 检查最后的token是否为eos或指定stop_token_ids
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

        # 1、移除请求
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

        # 2、释放请求所占用的资源
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
