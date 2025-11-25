# multiprocess_server.py
import multiprocessing
import uvicorn
from fastapi import FastAPI
import os
import time

app = FastAPI()


@app.get("/")
async def root():
    return {
        "message": "Multi-process Server",
        "worker_pid": os.getpid()
    }


@app.get("/heavy-task")
async def heavy_task():
    """模拟重计算任务"""
    time.sleep(2)  # 模拟耗时操作
    return {
        "status": "completed",
        "worker_pid": os.getpid(),
        "message": "任务完成"
    }


def run_worker(port_offset=0):
    """运行工作进程"""
    port = 8000 + port_offset
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False
    )
    server = uvicorn.Server(config)
    server.run()


def main():
    """多进程启动"""
    workers = 4  # CPU 核心数
    processes = []

    print(f"启动 {workers} 个工作进程...")

    for i in range(workers):
        process = multiprocessing.Process(
            target=run_worker,
            args=(i,)
        )
        processes.append(process)
        process.start()
        print(f"工作进程 {i + 1} 已启动 (PID: {process.pid})")

    # 等待所有进程结束
    for process in processes:
        process.join()


if __name__ == "__main__":
    # 单进程开发模式
    if os.getenv("ENVIRONMENT") == "development":
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    else:
        # 多进程生产模式
        main()