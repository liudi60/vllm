import asyncio
import time


async def demo_event_loop_behavior():
    """演示事件循环的非阻塞特性"""

    async def task1():
        print(f"[{time.time():.3f}] 任务1开始")
        future = asyncio.Future()

        # 模拟注册事件监听（2秒后完成Future）
        asyncio.get_event_loop().call_later(2, future.set_result, "完成")  # asyncio的event_loop中，每个event对象都有一个next_execute_time字段，下次轮询到该event，检查next_execute_time字段如果早于当前时间，就会执行该event的task，否则忽略该event

        print(f"[{time.time():.3f}] 任务1: 准备await future")
        result = await future  # 不会阻塞事件循环！
        print(f"[{time.time():.3f}] 任务1: await完成，结果: {result}")

    async def task2():
        # 这个任务可以同时运行
        for i in range(5):
            print(f"[{time.time():.3f}] 任务2: 执行第{i + 1}步")
            await asyncio.sleep(0.5)  # 也让出控制权

    # 同时运行两个任务
    await asyncio.gather(task1(), task2())

asyncio.run(demo_event_loop_behavior())


# [1763109795.105] 任务1开始
# [1763109795.105] 任务1: 准备await future
# [1763109795.105] 任务2: 执行第1步
# [1763109795.606] 任务2: 执行第2步
# [1763109796.107] 任务2: 执行第3步
# [1763109796.608] 任务2: 执行第4步
# [1763109797.107] 任务1: await完成，结果: 完成
# [1763109797.109] 任务2: 执行第5步