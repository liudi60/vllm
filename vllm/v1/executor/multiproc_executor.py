# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, Callable, Optional, Union, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_pp_group, get_tp_group)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import worker_receiver_cache_from_config
from vllm.utils import (_maybe_force_spawn, decorate_logs,
                        get_distributed_init_method, get_loopback_ip,
                        get_mp_context, get_open_port, set_process_title)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.executor.utils import get_and_update_mm_cache
from vllm.v1.outputs import (AsyncModelRunnerOutput, DraftTokenIds,
                             ModelRunnerOutput)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class MultiprocExecutor(Executor):

    supports_pp: bool = True

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). ")

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                             self.world_size,
                                             max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    # todo 创建worker进程  VLLM::Worker_TP0、VLLM::Worker_TP1，每个卡对应1个进程
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                    ))

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        # For pipeline parallel, we use a thread pool for asynchronous
        # execute_model.
        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            # _async_aggregate_workers_output also assumes a single IO thread
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None

    def start_worker_monitor(self):
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers
                             if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, "
                "shutting down executor.", proc_name)
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        Thread(target=monitor_workers,
               daemon=True,
               name="MultiprocWorkerMonitor").start()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            (output, ) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output, ),
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

        # aggregate all workers output to a single output
        if non_block:
            return self.kv_output_aggregator.async_aggregate(
                outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch",
                            unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        outputs = self.collective_rpc("take_draft_token_ids",
                                      unique_reply_rank=self.output_rank)
        return outputs[0]

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue(
                (send_method, args, kwargs, unique_reply_rank))

            workers = (self.workers[unique_reply_rank],
                       ) if unique_reply_rank is not None else self.workers
            responses = []

            def get_response(w: WorkerProcHandle,
                             dequeue_timeout: Optional[float] = None,
                             cancel_event: Optional[threading.Event] = None):
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout, cancel=cancel_event)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause")
                return result

            for w in workers:
                dequeue_timeout = None if deadline is None else (
                    deadline - time.monotonic())

                if self.io_thread_pool is not None:
                    # We must consume worker_response_mq from a single thread.
                    result = self.io_thread_pool.submit(  # type: ignore
                        get_response, w, dequeue_timeout, self.shutdown_event)
                    if not non_block:
                        result = result.result()
                elif not non_block:
                    result = get_response(w, dequeue_timeout,
                                          self.shutdown_event)
                else:
                    raise RuntimeError("non_block can only be used when"
                                       " max_concurrent_batches > 1")
                responses.append(result)

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True

            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, 'workers', None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])

            self.shutdown_event.set()
            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                del self.io_thread_pool

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @cached_property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return self.world_size - self.parallel_config.tensor_parallel_size


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Optional[Connection] = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue  # The worker process writes to this MQ
    death_writer: Optional[Connection] = None

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: MessageQueue) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
    ):
        self.rank = rank
        # ===== WorkerProc.__init__, local_rank=0, rank=0
        # ===== WorkerProc.__init__, local_rank=1, rank=1
        logger.warning(f'===== WorkerProc.__init__, local_rank={local_rank}, rank={rank}')
        '''
        ===== vllm_config=  model='Qwen3-8B-W8A8', 
                            speculative_config=None, 
                            tokenizer='Qwen3-8B-W8A8', 
                            skip_tokenizer_init=False, 
                            tokenizer_mode=auto, 
                            revision=None, 
                            tokenizer_revision=None, 
                            trust_remote_code=True, 
                            dtype=torch.bfloat16, 
                            max_seq_len=22528, 
                            download_dir=None, 
                            load_format=auto, 
                            tensor_parallel_size=2, 
                            pipeline_parallel_size=1, 
                            data_parallel_size=1, 
                            disable_custom_all_reduce=True, 
                            quantization=ascend, 
                            enforce_eager=True, 
                            kv_cache_dtype=auto, 
                            device_config=npu, 
                            structured_outputs_config=StructuredOutputsConfig(backend='auto', 
                            disable_fallback=False, 
                            disable_any_whitespace=False, 
                            disable_additional_properties=False, 
                            reasoning_parser=''), 
                            observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, 
                            otlp_traces_endpoint=None, 
                            collect_detailed_traces=None), 
                            seed=0, 
                            served_model_name=qwen3_moe, 
                            enable_prefix_caching=True, 
                            chunked_prefill_enabled=True, 
                            pooler_config=None, 
                            compilation_config={"level":0,
                                                "debug_dump_path":"",
                                                "cache_dir":"",
                                                "backend":"",
                                                "custom_ops":["all"],
                                                "splitting_ops":null,
                                                "use_inductor":true,
                                                "compile_sizes":[],
                                                "inductor_compile_config":{"enable_auto_functionalized_v2":false},
                                                "inductor_passes":{},
                                                "cudagraph_mode":0,
                                                "use_cudagraph":true,
                                                "cudagraph_num_of_warmups":1,
                                                "cudagraph_capture_sizes":[1],
                                                "cudagraph_copy_inputs":false,
                                                "full_cuda_graph":false,
                                                "use_inductor_graph_partition":false,
                                                "pass_config":{},
                                                "max_capture_size":1,
                                                "local_cache_dir":null}
        '''
        logger.warning(f'===== vllm_config={vllm_config}')
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        wrapper.init_worker(all_kwargs)  # 创建 NPUWorker对象实例
        self.worker = wrapper

        '''
        出现在 vLLM 的多 GPU / 分布式推理架构（特别是 张量并行 Tensor Parallelism 或 RPC 通信场景）中，其核心目的是：

        通过一个已创建的共享内存句柄（input_shm_handle），在当前 Worker 进程中重建一个 MessageQueue 对象，用于高效接收来自调度器（Scheduler）的输出数据（如 token IDs、请求元数据等）。
        
        🔍 逐部分解析
        1. MessageQueue 是什么？
        它是 vLLM 内部实现的一个 基于共享内存（Shared Memory）的进程间通信（IPC）队列。
        用于在 Driver 进程（含 Scheduler） 和 Worker 进程（含模型执行引擎） 之间 零拷贝传递数据。
        典型用途：
        Scheduler → Workers：广播 SchedulerOutput（包含要处理的 requests、block tables、token IDs 等）
        Workers → Driver：返回生成结果（在某些模式下）
        ✅ 目标：避免序列化/反序列化 + 减少 CPU-GPU 同步开销
        
        2. create_from_handle(input_shm_handle, self.worker.rank)
        input_shm_handle：
        是一个 共享内存的“句柄”（handle），通常是一个字符串或整数 ID。
        它由 主进程（Driver）预先创建，并在启动 Worker 时通过参数（如 CLI args、环境变量、pickle 传递）传给每个 Worker。
        该句柄指向一块 所有进程可访问的共享内存区域。
        self.worker.rank：
        表示当前 Worker 的 全局 rank（例如在 TP=4 时，rank 为 0,1,2,3）。
        用于：
        确定从共享内存的哪个“槽位”读取数据（避免冲突）
        实现 单写多读（Single Writer Multiple Readers） 模型
        create_from_handle：
        是一个 工厂方法，不创建新的共享内存，而是 “attach” 到已存在的共享内存段。
        类似于：mmap 一个已知 fd，或打开一个已命名的 POSIX 共享内存对象。
        📌 类比：就像多个进程通过同一个文件名打开 /dev/shm/my_queue。
        '''
        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        '''
        表示 创建一个容量极小（通常为单槽）的进程间消息队列（Message Queue），用于 Worker 向 Driver（或调度器）返回响应结果。

        🔍 逐部分解析
        ✅ MessageQueue(1, 1) 的含义
        虽然 MessageQueue 是 vLLM 内部自定义的 IPC 通信类（非标准库），但从命名和参数可合理推断：
        
        参数	含义（典型设计）
        第一个 1	队列深度（capacity）：最多缓存 1 条消息
        第二个 1	消费者数量（num_readers）：通常为 1（Driver 单线程读取）
        💡 这是一个 SPSC（Single Producer Single Consumer）队列：
        
        Producer：当前 Worker 进程（写入推理结果）
        Consumer：Driver / Scheduler 进程（读取结果）
        🧠 设计目的：为什么需要这个队列？
        在 vLLM 的 多进程架构（如张量并行 TP）中：
        
        Driver 进程：负责 HTTP API、请求调度（Scheduler）
        Worker 进程（每个 GPU 一个）：执行模型前向计算
        它们是 独立进程，无法直接共享 Python 对象，因此需要 IPC 机制传递结果。
        
        ✅ worker_response_mq 就是 Worker → Driver 的“回传通道”
        
        ⚙️ 为什么队列大小是 (1, 1)？
        原因	        说明
        同步通信模式	vLLM 通常采用 同步 step-by-step 推理：
                    Driver 发一批请求 → 等所有 Workers 返回 → 再发下一批
                    因此不需要缓冲多条消息
        避免内存浪费	每个 Worker 都有一个响应队列，若设大容量会浪费共享内存
        简化逻辑	    单槽队列天然具有 背压（backpressure）：
                    若 Driver 未及时读取，Worker 会在 push() 时阻塞，防止生产过快
                    📌 类似于：“每次只允许发一个 reply，等对方收走才能发下一个”
        
        💡 简单说：
        “我（Worker）算完一步，就把结果放进这个‘小盒子’（容量=1），等你（Driver）来拿。你不拿，我就等着——这样我们步调一致，不会乱。”
        '''
        # todo 同步通信模式	vLLM 通常采用 同步 step-by-step 推理：
        #  Driver 发一批请求 → 等所有 Workers 返回 → 再发下一批
        #  因此不需要缓冲多条消息
        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        logger.warning(f'===== self.use_async_scheduling={self.use_async_scheduling}')
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy")
            self.async_output_copy_thread.start()

        # Initialize multimodal receiver cache if needed
        self.mm_receiver_cache = worker_receiver_cache_from_config(
            vllm_config, MULTIMODAL_REGISTRY, shared_worker_lock)

        # Initialize device
        self.worker.init_device()

        # Set process title and log prefix
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel)

        # Load model 加载模型
        self.worker.load_model()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
        }
        # 创建worker进程
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               name=f"VllmWorker-{rank}",
                               daemon=True)

        logger.warning(f'===== WorkerProc.make_worker_process中创建并启动worker进程, proc={proc}')

        # 启动worker进程
        proc.start()
        writer.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:

        e = Exception("WorkerProc initialization failed due to "
                      "an exception in a background process. "
                      "See stack trace for root cause.")

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[Optional[WorkerProcHandle]] = (
            [None] * len(unready_proc_handles))
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    # Extract the message queue handle.
                    worker_response_mq = MessageQueue.create_from_handle(
                        response["handle"], 0)
                    ready_proc_handles[unready_proc_handle.rank] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle, worker_response_mq))

                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()


    '''
    worker线程的执行体
    
    主进程                          Worker 子进程
   │                                │
   │── fork + 传 ready_pipe  ──────▶│
   │                                │
   │                                ├─ 加载模型
   │                                ├─ 初始化 MQ
   │◀── send(READY + MQ handle)  ───┤
   │                                │
   │── wait MQ ready  ─────────────▶│
   │                                ├─ wait MQ ready
   │                                │
   │── send(execute_model)  ───────▶│
   │                                ├─ 执行 forward
   │◀── recv(logits)  ──────────────┤
   │                                │
   │── send(shutdown)  ────────────▶│
   │                                ├─ exit busy_loop
   │                                └─ 进程退出
    '''
    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        logger.warning(f'===== worker_main begin， args={args}')
        '''
        ===== kwargs={'vllm_config': VllmConfig(model_config=ModelConfig(   model='Qwen3-8B-W8A8', 
                                                                            runner='auto', 
                                                                            convert='auto', 
                                                                            task=None, 
                                                                            tokenizer='Qwen3-8B-W8A8', 
                                                                            tokenizer_mode='auto', 
                                                                            trust_remote_code=True, 
                                                                            dtype=torch.bfloat16, 
                                                                            seed=0, 
                                                                            hf_config_path=None, 
                                                                            allowed_local_media_path='', 
                                                                            allowed_media_domains=None, 
                                                                            revision=None, 
                                                                            code_revision=None, 
                                                                            rope_scaling={}, 
                                                                            rope_theta=None, 
                                                                            tokenizer_revision=None, 
                                                                            max_model_len=22528, 
                                                                            spec_target_max_model_len=None, 
                                                                            quantization='ascend', 
                                                                            enforce_eager=True, 
                                                                            max_logprobs=20, 
                                                                            logprobs_mode='raw_logprobs', 
                                                                            disable_sliding_window=False, 
                                                                            disable_cascade_attn=False, 
                                                                            skip_tokenizer_init=False, 
                                                                            enable_prompt_embeds=False, 
                                                                            served_model_name='qwen3_moe', 
                                                                            config_format='auto', 
                                                                            hf_token=None, 
                                                                            hf_overrides={}, 
                                                                            logits_processor_pattern=None, 
                                                                            generation_config='auto', 
                                                                            override_generation_config={}, 
                                                                            enable_sleep_mode=False, 
                                                                            model_impl='auto', 
                                                                            override_attention_dtype=None, 
                                                                            logits_processors=None, 
                                                                            io_processor_plugin=None, 
                                                                            pooler_config=None, 
                                                                            override_pooler_config=None, 
                                                                            multimodal_config=None), 
                                                                            cache_config=CacheConfig(block_size=128, 
                                                                            gpu_memory_utilization=0.9, 
                                                                            swap_space=4.0, 
                                                                            cache_dtype='auto', 
                                                                            is_attention_free=False, 
                                                                            num_gpu_blocks_override=None, 
                                                                            sliding_window=None, 
                                                                            enable_prefix_caching=True, 
                                                                            prefix_caching_hash_algo='sha256', 
                                                                            cpu_offload_gb=0.0, 
                                                                            calculate_kv_scales=False, 
                                                                            cpu_kvcache_space_bytes=None, 
                                                                            mamba_page_size_padded=None, 
                                                                            mamba_cache_dtype='auto', 
                                                                            mamba_ssm_cache_dtype='auto', 
                                                                            num_gpu_blocks=None, 
                                                                            num_cpu_blocks=None, 
                                                                            kv_sharing_fast_prefill=False, 
                                                                            kv_cache_memory_bytes=None), 
                                                                            parallel_config=ParallelConfig( pipeline_parallel_size=1, 
                                                                                                            tensor_parallel_size=2, 
                                                                                                            data_parallel_size=1, 
                                                                                                            data_parallel_size_local=1, 
                                                                                                            data_parallel_rank=0, 
                                                                                                            data_parallel_rank_local=0, 
                                                                                                            data_parallel_master_ip='127.0.0.1', 
                                                                                                            data_parallel_rpc_port=29550, 
                                                                                                            data_parallel_master_port=0, 
                                                                                                            data_parallel_backend='mp', 
                                                                                                            data_parallel_external_lb=False, 
                                                                                                            data_parallel_hybrid_lb=False, 
                                                                                                            enable_expert_parallel=False, 
                                                                                                            enable_eplb=False, 
                                                                                                            eplb_config=EPLBConfig( window_size=1000, 
                                                                                                                                    step_interval=3000, 
                                                                                                                                    num_redundant_experts=0, 
                                                                                                                                    log_balancedness=False), 
                                                                                                            expert_placement_strategy='linear', 
                                                                                                            num_redundant_experts=None, 
                                                                                                            eplb_window_size=None, 
                                                                                                            eplb_step_interval=None, 
                                                                                                            eplb_log_balancedness=None, 
                                                                                                            max_parallel_loading_workers=None, 
                                                                                                            disable_custom_all_reduce=True, 
                                                                                                            enable_dbo=False, 
                                                                                                            dbo_decode_token_threshold=32, 
                                                                                                            dbo_prefill_token_threshold=512, 
                                                                                                            ray_workers_use_nsight=False, 
                                                                                                            ray_runtime_env=None, 
                                                                                                            placement_group=None, 
                                                                                                            distributed_executor_backend='mp', 
                                                                                                            worker_cls='vllm_ascend.worker.worker_v1.NPUWorker', 
                                                                                                            sd_worker_cls='auto', 
                                                                                                            worker_extension_cls='', 
                                                                                                            world_size=2, 
                                                                                                            rank=0, 
                                                                                                            _data_parallel_master_port_list=[], 
                                                                                                            decode_context_parallel_size=1, 
                                                                                                            _api_process_count=1, 
                                                                                                            _api_process_rank=0), 
                                                                            scheduler_config=SchedulerConfig(runner_type='generate', 
                                                                            max_num_batched_tokens=2048, 
                                                                            max_num_seqs=768, 
                                                                            max_model_len=22528, 
                                                                            max_num_partial_prefills=1, 
                                                                            max_long_partial_prefills=1, 
                                                                            long_prefill_token_threshold=0, 
                                                                            num_lookahead_slots=0, 
                                                                            cuda_graph_sizes=[512], 
                                                                            enable_chunked_prefill=True, 
                                                                            is_multimodal_model=False, 
                                                                            max_num_encoder_input_tokens=2048, 
                                                                            encoder_cache_size=2048, 
                                                                            send_delta_data=False, 
                                                                            policy='sjf', 
                                                                            chunked_prefill_enabled=True, 
                                                                            disable_chunked_mm_input=False, 
                                                                            scheduler_cls='vllm.v1.core.sched.scheduler.Scheduler', 
                                                                            disable_hybrid_kv_cache_manager=False, 
                                                                            async_scheduling=False, 
                                                                            max_prefill_batch_size=0, 
                                                                            min_prefill_batch_size=2, 
                                                                            prefill_request_batching_timeout_ms=10000, 
                                                                            scheduler_delay_us=1000000), 
                                                                            device_config=DeviceConfig(device=device(type='npu'), 
                                                                            device_type='npu'), 
                                                                            load_config=LoadConfig(load_format='auto', 
                                                                            download_dir=None, 
                                                                            safetensors_load_strategy='lazy', 
                                                                            model_loader_extra_config={}, 
                                                                            device=None, 
                                                                            ignore_patterns=['original/**/*'], 
                                                                            use_tqdm_on_load=True, 
                                                                            pt_load_map_location='cpu'), 
                                                                            lora_config=None, 
                                                                            speculative_config=None, 
                                                                            structured_outputs_config=StructuredOutputsConfig(backend='auto', 
                                                                            disable_fallback=False, 
                                                                            disable_any_whitespace=False, 
                                                                            disable_additional_properties=False, 
                                                                            reasoning_parser=''), 
                                                                            observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, 
                                                                            otlp_traces_endpoint=None, 
                                                                            collect_detailed_traces=None), 
                                                                            quant_config=AscendQuantConfig: <vllm_ascend.quantization.quant_config.AscendQuantConfig object at 0xffff2df76d10>, 
                                                                            compilation_config={"level":0,
                                                                            "debug_dump_path":"",
                                                                            "cache_dir":"",
                                                                            "backend":"",
                                                                            "custom_ops":["all"],
                                                                            "splitting_ops":null,
                                                                            "use_inductor":true,
                                                                            "compile_sizes":[],
                                                                            "inductor_compile_config":{"enable_auto_functionalized_v2":false},
                                                                            "inductor_passes":{},
                                                                            "cudagraph_mode":0,
                                                                            "use_cudagraph":true,
                                                                            "cudagraph_num_of_warmups":1,
                                                                            "cudagraph_capture_sizes":[1],
                                                                            "cudagraph_copy_inputs":false,
                                                                            "full_cuda_graph":false,
                                                                            "use_inductor_graph_partition":false,
                                                                            "pass_config":{},
                                                                            "max_capture_size":1,
                                                                            "local_cache_dir":null}, 
                                                                            kv_transfer_config=None, 
                                                                            kv_events_config=None, 
                                                                            additional_config={}, 
                                                                            instance_id='db235'), 
                                                                            'local_rank': 0, 
                                                                            'rank': 0, 
                                                                            'distributed_init_method': 'tcp://127.0.0.1:48993', 
                                                                            'input_shm_handle': Handle(local_reader_ranks=[0, 
                                                                            1], 
                                                                            buffer_handle=(2, 
                                                                            16777216, 
                                                                            10, 
                                                                            'psm_d555c7d0'), 
                                                                            local_subscribe_addr='ipc:///tmp/5b9f3a20-dfa1-440b-b93f-6bdcdb2c1fc8', 
                                                                            remote_subscribe_addr=None, 
                                                                            remote_addr_ipv6=False), 
                                                                            'ready_pipe': (<multiprocessing.connection.Connection object at 0xffff2df2e310>, 
                                                                            <multiprocessing.connection.Connection object at 0xffff2df2e1d0>), 
                                                                            'death_pipe': <multiprocessing.connection.Connection object at 0xffff2df2df90>, 
                                                                            'shared_worker_lock': <Lock(owner=None)>}
        '''
        logger.warning(f'===== kwargs={kwargs}')

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        '''
        这段代码来自 vLLM 的多进程（multiprocessing）worker 初始化逻辑，其核心目的是：
        让子进程（worker）能够监听主进程（parent）是否意外退出，并在主进程崩溃时自动终止自己，避免“孤儿 worker”残留。
        这是 健壮的多进程系统中常见的“父进程死亡监控”机制。
        
        '''
        worker = None
        '''
        获取用于同步的管道（Pipe）
        📌 关键概念：multiprocessing.Pipe()
        Pipe() 创建一对 双向连接对象 (conn1, conn2)，用于进程间通信（IPC）。
        数据写入一端，可从另一端读出。
        参数说明：
        变量	作用
        ready_pipe	主进程 ← worker 通知“我已启动就绪”
        - worker 持有 ready_writer（写端）
        - 主进程持有 reader（读端）
        death_pipe	主进程 → worker 通知“我挂了”
        - 主进程持有写端
        - worker 持有 death_pipe（读端）
        💡 注意：death_pipe 是 主进程创建并传给子进程的读端。当主进程退出时，操作系统会自动关闭其所有文件描述符（包括 pipe 写端），导致子进程读端收到 EOF。
        '''
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)
        # 创建 shutdown 事件
        # 一个线程安全的信号量，用于通知主工作线程优雅退出。
        # 初始状态为 False，调用 .set() 后变为 True。
        shutdown_event = threading.Event()
        '''
        启动“父进程死亡监控线程”
        🧠 核心原理：
        正常情况：主进程 alive → death_pipe 写端 open → recv() 永远阻塞。
        主进程崩溃/退出：操作系统关闭其所有 FD → death_pipe 写端关闭 → 子进程 recv() 立即抛出 EOFError。
        子进程响应：捕获异常 → 记录日志 → 设置 shutdown_event → 通知主循环退出。
        ✅ 这是一种 无需轮询、零开销 的父进程存活检测机制。
        '''
        # Start death monitoring thread if death_pipe is provided
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    # This will block until parent process exits (pipe closes)
                    death_pipe.recv()
                except EOFError:
                    # Parent process has exited, terminate this worker
                    logger.info("Parent process exited, terminating worker")
                    # Send signal to self to trigger clean shutdown
                    shutdown_event.set()
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(target=monitor_parent_death,
                                   daemon=True,
                                   name="WorkerDeathMonitor")
            death_monitor.start()

        try:
            '''
            reader 是 ready_pipe 的读端（由主进程持有，用于接收 worker 的就绪信号）
            但 worker 拿到的是写端 ready_writer，读端 reader 对它无用
            关闭无用的 FD（文件描述符），避免资源泄漏
            💡 提醒：ready_pipe = (reader, ready_writer)，其中：
            
            主进程持 reader（读）
            Worker 持 ready_writer（写）
            '''
            reader.close()
            '''
            进入主工作循环处理推理请求
            创建 WorkerProc 实例
                WorkerProc 是 vLLM 中封装 模型加载、KV Cache、RPC 通信 的核心类
            此步骤会：
            加载 HuggingFace 模型（AutoModelForCausalLM.from_pretrained）
                初始化 tokenizer（可选）
                创建 PagedAttention KV Cache（BlockSpaceManager）
                初始化两个关键的消息队列（Message Queue）：
                worker_response_mq：用于向主进程返回结果
                rpc_broadcast_mq：用于接收主进程的广播指令（如 shutdown）
                ⚠️ 这是最耗时的步骤（GPU 显存分配、模型加载）
            '''
            # 创建worker，里面工作主要是：初始化设备、创建NPUWorker实例、创建ModelRunner实例、加载模型权重
            worker = WorkerProc(*args, **kwargs)

            '''
            通知主进程：“我准备好了！”
            📌 为什么需要这个？
            主进程在启动所有 worker 后，会 阻塞等待每个 worker 发送“READY”信号
            只有全部就绪，才开始调度请求，避免 race condition
            📥 发送的内容：
            字段	说明
            "status"	固定值 "READY"，表示初始化成功
            "handle"	消息队列的共享内存句柄（用于跨进程通信）
            主进程用它重建 MQ 的本地代理
            💡 export_handle() 是基于 multiprocessing 的 SharedMemory 或 Queue 的底层机制，允许主进程“连接”到 worker 的响应队列。
            '''
            # Send READY once we know everything is loaded
            ready_writer.send({
                "status":
                WorkerProc.READY_STR,
                "handle":
                worker.worker_response_mq.export_handle(),
            })

            '''
            等待消息队列就绪（关键同步点！）
            ❗ 为什么必须等？
            消息队列（MQ）底层可能使用 共享内存 + 信号量，需要双方都初始化完成才能通信
            如果顺序颠倒（先 close ready_writer 再 wait）会导致死锁！
            主进程可能在收到 READY 后立即尝试发 RPC 指令
            但 worker 的 MQ 还没 ready → 指令丢失或阻塞
            ✅ 注释中强调：
            
            “Must be kept consistent with the Executor”
            
            → 主进程的 MultiprocessingExecutor 必须按相同顺序操作 MQ
            '''
            '''
            在 vLLM 的多进程（multiprocessing）架构 中，worker.rpc_broadcast_mq 和 worker.worker_response_mq 是两个关键的 
            跨进程消息队列（Message Queue, MQ），用于 主进程（driver）与子进程（worker）之间的高效、低延迟通信。

            它们共同构成了 vLLM 多进程执行器（如 MultiprocessingExecutor）的 命令-响应（Request-Reply）通信模型。
            
            🧠 核心作用对比
            消息队列	            方向	            用途	                数据内容
            rpc_broadcast_mq	主进程 → Worker	下发指令（RPC 调用）	execute_model, abort_request, shutdown 等命令 + 参数
            worker_response_mq	Worker → 主进程	返回结果（响应）	    模型输出 logits / 采样 token / 错误信息
            
            💡 可以理解为：
            rpc_broadcast_mq = “老板给工人派活”
            worker_response_mq = “工人干完活交差”
            '''
            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            # 清理 ready_writer
            # 任务已完成（就绪信号已发送），关闭写端，释放资源
            # 设为 None 避免误用
            ready_writer.close()
            ready_writer = None

            '''
            # 进入主工作循环
            🔄 worker_busy_loop 做什么？
            无限循环监听 RPC 指令（来自 rpc_broadcast_mq）
            支持的指令包括：
            "execute_model"：执行前向计算
            "abort_request"：取消请求
            "shutdown"：退出循环
            如果 shutdown_event 被触发（如父进程死亡监控线程调用 .set()），则优雅退出
            💡 这是 worker 的“一生”：从就绪 → 处理请求 → 收到 shutdown → 退出
            '''
            logger.warning(f'===== worker.worker_busy_loop...')
            worker.worker_busy_loop(cancel=shutdown_event)

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_event.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)

    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    def worker_busy_loop(self, cancel: Optional[threading.Event] = None):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            # EngineCore进程向所有worker进程广播"调度生成的batch"
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                cancel=cancel, indefinite=True)
            '''
            ===== method=execute_model
            ===== args=(SchedulerOutput(scheduled_new_reqs=[], 
                        scheduled_cached_reqs=CachedRequestData(
                            req_ids=['chatcmpl-826c652e53d4482586d8ce63664a559f'], 
                            resumed_from_preemption=[False], 
                            new_token_ids=[], 
                            new_block_ids=[None], 
                            num_computed_tokens=[42]), 
                            num_scheduled_tokens={'chatcmpl-826c652e53d4482586d8ce63664a559f': 1}, 
                            total_num_scheduled_tokens=1, 
                            scheduled_spec_decode_tokens={}, 
                            scheduled_encoder_inputs={}, 
                            num_common_prefix_blocks=[1], 
                            finished_req_ids=set(), 
                            free_encoder_mm_hashes=[], 
                            structured_output_request_ids={}, 
                            grammar_bitmask=None, 
                            kv_connector_metadata=None),)
            ===== kwargs={}
            ===== output_rank=0
            '''
            logger.warning(f'===== worker_busy_loop, self.rpc_broadcast_mq.dequeue')
            logger.warning(f'===== method={method}')            # ===== method=get_kv_cache_spec
            logger.warning(f'===== args={args}')                # {}
            logger.warning(f'===== kwargs={kwargs}')            # {}
            logger.warning(f'===== output_rank={output_rank}')  # None
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                # retrieve from shm cache if available
                if self.mm_receiver_cache is not None \
                    and func.__name__ == "execute_model":
                    get_and_update_mm_cache(self.mm_receiver_cache, args)
                # ===== func=<bound method NPUWorker.execute_model of <vllm_ascend.worker.worker_v1.NPUWorker object at 0xffff1384c750>>
                logger.warning(f'===== func={func}')
                output = func(*args, **kwargs)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
                continue

            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        logger.warning(f'===== 设置worker进程名称：{process_name}')
        set_process_title(name=process_name)
        decorate_logs(process_name)


def set_multiprocessing_worker_envs():
    """ Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent 
    process before worker processes are created"""

    _maybe_force_spawn()

    # Configure thread parallelism if OMP_NUM_THREADS isn't set
    #
    # Helps to avoid CPU contention. The default of spawning a thread per
    # core combined with multiprocessing for each GPU can have a negative
    # impact on performance. The contention is amplified when running in a
    # container where CPU limits can cause throttling.
    default_omp_num_threads = 1
    if "OMP_NUM_THREADS" not in os.environ and (
            current_parallelism :=
            torch.get_num_threads()) > default_omp_num_threads:
        logger.warning(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism, default_omp_num_threads)
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        # torch.set_num_threads(n) 是 PyTorch 提供的一个用于控制 CPU 并行计算线程数的函数，主要用于调节底层线性代数库（如 OpenMP、MKL、BLAS 等）在执行 CPU 张量运算时使用的线程数量。
        torch.set_num_threads(default_omp_num_threads)
