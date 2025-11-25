# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

import time
import os
import sys
import random


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


from tests.v1.core.utils import create_requests, create_requests_for_sjf
from vllm.v1.core.sched.request_queue import SJFRequestQueue, SJFRequestQueueInHeap


def test_sjf(sjf, sjf_heap, num_requests):
    BLOCK_SIZE = 4
    # num_requests = 10
    num_block_list = [random.randint(1, 1000) for _ in range(num_requests)]
    num_token_list = [ ((block_num // 2 + 1) * BLOCK_SIZE) for block_num in num_block_list ]
    # print(f"num_token_list={num_token_list}")
    max_tokens = max(num_token_list) + 50

    requests = create_requests_for_sjf(num_requests=num_requests,
                               num_token_list=num_token_list,
                               max_tokens=max_tokens,
                               block_size=BLOCK_SIZE)

    # ========================= 测试sjf =========================
    begin = time.time()

    for request in requests:
        sjf.add_request(request)

    for _ in range(len(requests)):
        sjf.pop_request()

    end = time.time()
    cost = end - begin
    print(f'sjf      | num_requests={num_requests} | cost: {cost * 1000 * 1000} us')



    # ========================= 测试sjf_heap =========================
    begin = time.time()

    for request in requests:
        sjf_heap.add_request(request)

    for _ in range(len(requests)):
        sjf_heap.pop_request()

    end = time.time()
    cost_heap = end - begin
    print(f'sjf_heap | num_requests={num_requests} | cost: {cost_heap * 1000 * 1000} us')

    print()




if __name__ == '__main__':
    test_sjf(SJFRequestQueue(), SJFRequestQueueInHeap(), 10)
    test_sjf(SJFRequestQueue(), SJFRequestQueueInHeap(), 100)
    test_sjf(SJFRequestQueue(), SJFRequestQueueInHeap(), 1000)
    test_sjf(SJFRequestQueue(), SJFRequestQueueInHeap(), 3000)
