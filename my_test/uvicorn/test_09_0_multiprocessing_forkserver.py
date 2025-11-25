# 下面是一个 完整的 multiprocessing 使用 forkserver 启动方法的示例，展示了如何：
#
# 设置启动方式为 'forkserver'
# 预加载模块（可选但推荐）
# 主进程初始化重型资源（如大对象、模型）
# 启动 forkserver
# 创建子进程共享该资源
# ✅ 示例目标
# 主进程创建一个“大对象”（模拟加载大模型）
# 子进程通过 forkserver fork 出来后，直接访问该对象（共享内存）
# 验证子进程无需重新创建该对象


# forkserver_example.py
import os
import time
from multiprocessing import Process, set_start_method, set_forkserver_preload
from multiprocessing import forkserver  # 用于 ensure_running


# 模拟一个“重型”模块（比如包含大模型）
class HeavyModel:
    def __init__(self):
        print(f"[{os.getpid()}] Loading heavy model (simulated)...")
        self.data = [0] * 10_000_000  # 模拟大内存占用
        self.version = "v1.0"
        print(f"[{os.getpid()}] Model loaded.")


# 模拟工作函数（子进程执行）
def worker(worker_id, model_ref):
    print()
    time.sleep(worker_id * 3)
    print(f"[{os.getpid()}] Worker {worker_id} started.")
    if worker_id == 0:  # 测试0号子进程修改共享数据后，其他子进程可见？答：错。每个子进程共享主进程的model对象，只是读共享，但是子进程如果写该内存，则会触发COW机制，该内存数据会复制到子进程中，子进程只对自己内部的内存写，其他子进程不可见。
        model_ref.data[0] = 9
    print(f'===== model_ref.data[0]={model_ref.data[0]}')
    # 直接使用主进程创建的模型（共享内存）
    print(f"[{os.getpid()}] Worker {worker_id} sees model version: {model_ref.version}")
    print(f"[{os.getpid()}] Worker {worker_id} data length: {len(model_ref.data)}")
    time.sleep(1000)
    print(f"[{os.getpid()}] Worker {worker_id} done.")


def main():
    # 必须在程序最开始设置 start method（且只能调用一次）
    set_start_method('forkserver')

    # 可选：预加载包含 HeavyModel 的模块
    # 注意：这里我们把 HeavyModel 放在 __main__ 中，实际项目应放在独立模块
    # set_forkserver_preload(["your_module"])  # 如 ["my_model"]

    # 启动 forkserver 前，先创建重型资源
    print(f"[{os.getpid()}] Main process initializing heavy model...")
    model = HeavyModel()
    model.data[0] = 1

    # 显式启动 forkserver（继承当前状态，包括 model）
    print(f"[{os.getpid()}] Starting forkserver...")
    forkserver.ensure_running()

    print(f"[{os.getpid()}] Creating workers...")
    processes = []
    for i in range(3):
        p = Process(target=worker, args=(i, model))
        p.start()
        processes.append(p)

    # 等待子进程结束
    for p in processes:
        p.join()

    print(f"[{os.getpid()}] All workers finished.")


if __name__ == "__main__":
    main()



'''
##################
日志：
root@node-224:/home/liudi/vllm/my_test# python uvicorn/test_09_0_multiprocessing_forkserver.py 
[31471] Main process initializing heavy model...
[31471] Loading heavy model (simulated)...
[31471] Model loaded.
[31471] Starting forkserver...
[31471] Creating workers...

[31474] Worker 0 started.
===== model_ref.data[0]=9
[31474] Worker 0 sees model version: v1.0
[31474] Worker 0 data length: 10000000


[31475] Worker 1 started.
===== model_ref.data[0]=1
[31475] Worker 1 sees model version: v1.0
[31475] Worker 1 data length: 10000000
[31476] Worker 2 started.
===== model_ref.data[0]=1
[31476] Worker 2 sees model version: v1.0
[31476] Worker 2 data length: 10000000




###################
进程树：
(base) root@node-224:~# pstree -apu 2682296
python,2682296 uvicorn/test_09_0_multiprocessing_forkserver.py
  ├─python,2682297 -c from multiprocessing.resource_tracker import main;main(3)
  └─python,2682298 -cfrom multiprocessing.forkserver import main; main(3, 5, ['__main__'], **{'sys_path': ['/home/liudi/vllm/my_test/uvicorn', '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages', '/usr/
      ├─python,2682299 -cfrom multiprocessing.forkserver import main; main(3, 5, ['__main__'], **{'sys_path': ['/home/liudi/vllm/my_test/uvicorn', '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages', '/usr/
      ├─python,2682321 -cfrom multiprocessing.forkserver import main; main(3, 5, ['__main__'], **{'sys_path': ['/home/liudi/vllm/my_test/uvicorn', '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages', '/usr/
      └─python,2682322 -cfrom multiprocessing.forkserver import main; main(3, 5, ['__main__'], **{'sys_path': ['/home/liudi/vllm/my_test/uvicorn', '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages', '/usr/
      

'''

