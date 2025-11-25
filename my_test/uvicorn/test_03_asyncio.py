
'''

asyncio 是 Python 标准库中用于编写并发异步 I/O 程序的核心模块，自 Python 3.4 引入，并在 3.5+ 随着 async/await 语法成为主流。它是构建高性能网络服务（如 Web API、爬虫、实时通信）的基础。

🧠 一、核心思想：事件循环 + 协程
asyncio 基于 单线程事件循环（Event Loop） 模型，通过 协程（coroutine） 实现协作式多任务（cooperative multitasking）：

不创建新线程，所有任务在同一线程中交替执行
当一个任务等待 I/O（如网络请求、文件读写）时，主动让出控制权，让其他任务运行
适用于 I/O 密集型 场景（非 CPU 密集型）
✅ 典型应用：FastAPI、aiohttp、数据库异步驱动（asyncpg）、WebSocket 服务等

'''


import asyncio
import time

# 1. 定义协程
async def say_hello(delay, name):
    await asyncio.sleep(delay)  # 模拟异步 I/O
    print(f"Hello, {name}!")


# 启动事件循环（Python ≥ 3.7）
async def main1():
    # == == = 1
    # Hello, Alice!
    # == == = 2
    # Hello, Bob!
    # == == = 3
    # cost
    # time: 3.0037386417388916

    begin = time.time()

    print(f'===== 1')
    await say_hello(1, "Alice")
    print(f'===== 2')
    await say_hello(2, "Bob")
    print(f'===== 3')

    end = time.time()
    print(f'cost time: {end - begin}')



async def main2():
    # == == = 1
    # == == = 2
    # == == = 3
    # Hello, Alice!
    # Hello, Bob!
    # cost
    # time: 2.0015571117401123

    begin = time.time()

    print(f'===== 1')
    future1 = say_hello(1, "Alice")
    print(f'===== 2')
    future2 = say_hello(2, "Bob")
    print(f'===== 3')

    # await asyncio.gather(future1, future2) 这一行相当于如下两行：
    # await future1
    # await future2
    await asyncio.gather(future1, future2)

    end = time.time()
    print(f'cost time: {end - begin}')







# 推荐方式：自动创建并管理事件循环
# asyncio.run(main1())

asyncio.run(main2())


