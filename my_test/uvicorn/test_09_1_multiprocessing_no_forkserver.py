
import os
import time
from multiprocessing import Process


# 模拟一个“重型”模块（比如包含大模型）
class HeavyModel:
    def __init__(self):
        print(f"[{os.getpid()}] Loading heavy model (simulated)...")
        self.data = [0] * 10_000  # 模拟大内存占用
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

    # 主进程创建重型资源
    print(f"[{os.getpid()}] Main process initializing heavy model...")
    model = HeavyModel()
    model.data[0] = 1


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
此处代码是不通过forkserver启动子进程，而是直接启动子进程，看看二者有什么差异：

#########################
日志：
root@node-224:/home/liudi/vllm/my_test# python uvicorn/test_09_1_multiprocessing_no_forkserver.py 
[31477] Main process initializing heavy model...
[31477] Loading heavy model (simulated)...
[31477] Model loaded.
[31477] Creating workers...

[31478] Worker 0 started.
===== model_ref.data[0]=9
[31478] Worker 0 sees model version: v1.0
[31478] Worker 0 data length: 10000


[31479] Worker 1 started.
===== model_ref.data[0]=1
[31479] Worker 1 sees model version: v1.0
[31479] Worker 1 data length: 10000
[31480] Worker 2 started.
===== model_ref.data[0]=1
[31480] Worker 2 sees model version: v1.0
[31480] Worker 2 data length: 10000



#########################
不使用forkserver，查看进程树：
(base) root@node-224:~# ps -ef | grep multiprocess
root     2686218 3214490  0 11:48 pts/2    00:00:00 python uvicorn/test_09_1_multiprocessing_no_forkserver.py
root     2686220 2686218  0 11:48 pts/2    00:00:00 python uvicorn/test_09_1_multiprocessing_no_forkserver.py
root     2686221 2686218  0 11:48 pts/2    00:00:00 python uvicorn/test_09_1_multiprocessing_no_forkserver.py
root     2686222 2686218  0 11:48 pts/2    00:00:00 python uvicorn/test_09_1_multiprocessing_no_forkserver.py
root     2686982 3154020  0 11:50 pts/4    00:00:00 grep --color=auto multiprocess
(base) root@node-224:~# 
(base) root@node-224:~# 
(base) root@node-224:~# pstree -apu 2686218
python,2686218 uvicorn/test_09_1_multiprocessing_no_forkserver.py
  ├─python,2686220 uvicorn/test_09_1_multiprocessing_no_forkserver.py
  ├─python,2686221 uvicorn/test_09_1_multiprocessing_no_forkserver.py
  └─python,2686222 uvicorn/test_09_1_multiprocessing_no_forkserver.py
  
  

#########################
使用forkserver，查看进程树：
(base) root@node-224:~# pstree -apu 2682296
python,2682296 uvicorn/test_09_0_multiprocessing_forkserver.py
  ├─python,2682297 -c from multiprocessing.resource_tracker import main;main(3)
  └─python,2682298 -cfrom multiprocessing.forkserver import main; main(3, 5, ['__main__'], **{'sys_path': ['/home/liudi/vllm/my_test/uvicorn', '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages', '/usr/
      ├─python,2682299 -cfrom multiprocessing.forkserver import main; main(3, 5, ['__main__'], **{'sys_path': ['/home/liudi/vllm/my_test/uvicorn', '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages', '/usr/
      ├─python,2682321 -cfrom multiprocessing.forkserver import main; main(3, 5, ['__main__'], **{'sys_path': ['/home/liudi/vllm/my_test/uvicorn', '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages', '/usr/
      └─python,2682322 -cfrom multiprocessing.forkserver import main; main(3, 5, ['__main__'], **{'sys_path': ['/home/liudi/vllm/my_test/uvicorn', '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages', '/usr/
      



'''