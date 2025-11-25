# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
import inspect
import json

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.worker.worker_base import WorkerBase as WorkerBaseV0

logger = init_logger(__name__)


class WorkerBase(WorkerBaseV0):
    """
    Abstract class for v1 worker, mainly define some methods for v1.
    For methods shared by v0 and v1, define them in v0 WorkerBase
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """获取完整的调用栈"""
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
        # logger.warning(f'===== all stack_details={stack_details}')
        logger.warning(f'===== WorkerBase all stack_details={json.dumps(stack_details, indent=4)}')

        '''
        WARNING 11-18 08:56:18 [worker_base.py:46] ===== WorkerBase all stack_details=[
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker_v1.py",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 103,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "__init__",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": [
        WARNING 11-18 08:56:18 [worker_base.py:46]             "        super().__init__(vllm_config=vllm_config,\n"
        WARNING 11-18 08:56:18 [worker_base.py:46]         ]
        WARNING 11-18 08:56:18 [worker_base.py:46]     },
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "/home/liudi/vllm/vllm/worker/worker_base.py",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 248,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "init_worker",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": [
        WARNING 11-18 08:56:18 [worker_base.py:46]             "            self.worker = worker_class(**kwargs)\n"
        WARNING 11-18 08:56:18 [worker_base.py:46]         ]
        WARNING 11-18 08:56:18 [worker_base.py:46]     },
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "/home/liudi/vllm/vllm/v1/executor/multiproc_executor.py",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 406,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "__init__",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": [
        WARNING 11-18 08:56:18 [worker_base.py:46]             "        wrapper.init_worker(all_kwargs)\n"
        WARNING 11-18 08:56:18 [worker_base.py:46]         ]
        WARNING 11-18 08:56:18 [worker_base.py:46]     },
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "/home/liudi/vllm/vllm/v1/executor/multiproc_executor.py",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 578,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "worker_main",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": [
        WARNING 11-18 08:56:18 [worker_base.py:46]             "            worker = WorkerProc(*args, **kwargs)\n"
        WARNING 11-18 08:56:18 [worker_base.py:46]         ]
        WARNING 11-18 08:56:18 [worker_base.py:46]     },
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "/usr/local/python3.11.13/lib/python3.11/multiprocessing/process.py",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 108,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "run",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": [
        WARNING 11-18 08:56:18 [worker_base.py:46]             "            self._target(*self._args, **self._kwargs)\n"
        WARNING 11-18 08:56:18 [worker_base.py:46]         ]
        WARNING 11-18 08:56:18 [worker_base.py:46]     },
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "/usr/local/python3.11.13/lib/python3.11/multiprocessing/process.py",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 314,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "_bootstrap",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": [
        WARNING 11-18 08:56:18 [worker_base.py:46]             "                self.run()\n"
        WARNING 11-18 08:56:18 [worker_base.py:46]         ]
        WARNING 11-18 08:56:18 [worker_base.py:46]     },
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "/usr/local/python3.11.13/lib/python3.11/multiprocessing/spawn.py",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 135,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "_main",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": [
        WARNING 11-18 08:56:18 [worker_base.py:46]             "    return self._bootstrap(parent_sentinel)\n"
        WARNING 11-18 08:56:18 [worker_base.py:46]         ]
        WARNING 11-18 08:56:18 [worker_base.py:46]     },
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "/usr/local/python3.11.13/lib/python3.11/multiprocessing/spawn.py",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 122,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "spawn_main",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": [
        WARNING 11-18 08:56:18 [worker_base.py:46]             "    exitcode = _main(fd, parent_sentinel)\n"
        WARNING 11-18 08:56:18 [worker_base.py:46]         ]
        WARNING 11-18 08:56:18 [worker_base.py:46]     },
        WARNING 11-18 08:56:18 [worker_base.py:46]     {
        WARNING 11-18 08:56:18 [worker_base.py:46]         "filename": "<string>",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "line_number": 1,
        WARNING 11-18 08:56:18 [worker_base.py:46]         "function": "<module>",
        WARNING 11-18 08:56:18 [worker_base.py:46]         "code_line": null
        WARNING 11-18 08:56:18 [worker_base.py:46]     }
        WARNING 11-18 08:56:18 [worker_base.py:46] ]
        '''





        """
        Initialize common worker components.
        
        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
        # Configuration storage
        super().__init__(vllm_config=vllm_config)

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: Optional[torch.device] = None
        # Worker.model_runner
        self.model_runner: Optional[nn.Module] = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup."""
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return
