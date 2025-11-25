import asyncio
import time

# async 与普通函数的区别


# 普通函数
def sync_function():
    time.sleep(1)  # 阻塞调用
    return "同步完成"


# 异步函数
async def async_function():
    await asyncio.sleep(1)  # 非阻塞调用
    return "异步完成"


# 测试性能差异
async def compare_performance():
    start_time = time.time()

    # 同步方式（顺序执行）
    sync_function()
    sync_function()
    sync_time = time.time() - start_time
    print(f"同步执行时间: {sync_time:.2f}秒")

    # 异步方式（并发执行）
    start_time = time.time()
    await asyncio.gather(async_function(), async_function())  # async函数调用f，首先会启动一个新线程t1（或者从线程池获取一个线程），然后将函数f封装为一个task，丢给线程t1，然后返回一个协程类型的对象（类似于java中的future对象）
    async_time = time.time() - start_time
    print(f"异步执行时间: {async_time:.2f}秒")


asyncio.run(compare_performance())

# 同步执行时间: 2.00秒
# 异步执行时间: 1.00秒

'''
async异步函数原理：调用异步函数，就会新启动一个线程来执行。可以这样简单理解，实际实现是类似于linux中的select/poll事件循环。
事件循环工作原理：
1、首先创建一个事件数组 event_array，然后启动一个loop线程不停轮询数组的每个元素。另一方面，用户线程调用异步函数时，会注册一个事件到event_array，则立即返回。然后loop线程发现有事件，则封装一个task，丢给worker线程池进行异步处理。
   这样，用户线程调用异步函数时，只是下发任务后就返回了，具体执行是在后台线程池中执行的，这样也就可以并行执行多个任务了；
   这一整体称为"事件循环"（event_loop）；
   
2、编写两个函数，主函数main_f和异步函数async_f，主函数main_f中调用async_f：
    async def async_f():
        ...异步操作...
        
    async def main_f():
        future = async_f()
        await future
        
    asyncio.run(main_f())
    
3、asyncio.run(main_f())，asyncio.run()的工作就是创建事件循环，然后用户线程执行main_f()，调用synch_f()，即向事件数组中注册一个事件，后台线程执行async_f()中的具体逻辑。
   future = async_f() 中的future类似于事件句柄event，后台worker线程与用户线程就是通过event通信，比如await future，就是用户线程等待worker线程执行完任务。

'''


