# fastapi_uvloop.py
from fastapi import FastAPI
import uvicorn
import uvloop
import asyncio
import signal
import socket

# 该文件是对api_server.py中服务启动的简化

# # 设置 uvloop
# uvloop.install()

HOST = "0.0.0.0"
PORT = 8080


def create_server_socket(addr: tuple[str, int]) -> socket.socket:
    sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)
    return sock


def setup_server():
    sock_addr = (HOST, PORT)
    sock = create_server_socket(sock_addr)

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    addr, port = sock_addr
    listen_address = f"http://{addr}:{port}"
    return listen_address, sock



def create_app():
    app = FastAPI(title="UVloop FastAPI")

    @app.get("/")
    async def root():
        return {"message": "使用 uvloop 的 FastAPI 应用"}

    @app.get("/heavy-async")
    async def heavy_async_operation():
        """模拟重型异步操作"""
        await asyncio.sleep(0.1)  # 模拟I/O操作
        return {"status": "completed", "operation": "heavy_async"}

    @app.get("/concurrent")
    async def concurrent_operations():
        """并发操作测试"""
        async def mock_db_query(query_id: int):
            await asyncio.sleep(0.05)
            return f"查询结果 {query_id}"

        # 并发执行多个查询
        tasks = [mock_db_query(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        return {"concurrent_results": results}

    return app


async def run_server():
    # 1、创建http应用
    app = create_app()

    # 2、创建监听socket
    listen_address, sock = setup_server()

    # 3、创建http服务对象，使用uvicorn创建
    config = uvicorn.Config(
        app=app,
        host=HOST,
        port=PORT,
        loop="uvloop",  # 明确使用 uvloop
        log_level="debug"
    )

    server = uvicorn.Server(config)
    # server.run()  # 启动服务+将socket.accept()加入event_loop，这个是自动方式，采用下面手动方式

    # 4、server.serve()启动服务，然后手动将socket.accept()加入event_loop
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve(sockets=[sock]))  # server.serve()中是一个while循环，不停把socket.accept()加入event_loop中

    def signal_handler() -> None:
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task  # 挂起当前协程，并放到server_task之后
        return dummy_shutdown()
    except asyncio.CancelledError:
        print("Shutting down FastAPI HTTP server.")
        return server.shutdown()
    finally:
        pass




if __name__ == "__main__":
    uvloop.run(run_server())  # 使用uvloop的事件循环实现，效率比asyncio的默认实现高