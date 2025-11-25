import time

import scheduler
from tests.v1.core.utils import create_requests
from vllm.v1.core.sched.request_queue import SJFRequestQueue

sjf = SJFRequestQueue()

create_requests()


print(time.time())





