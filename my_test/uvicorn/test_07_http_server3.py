import asyncio
import socket
import uvloop
from fastapi import FastAPI
from uvicorn import Config, Server

# 使用 uvloop 作为事件循环策略
uvloop.install()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI + uvloop + custom socket!"}

async def serve_with_custom_socket():
    # 创建一个监听 socket（也可以从外部传入，比如 systemd）
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 8080))
    sock.listen(128)  # backlog
    sock.set_inheritable(True)

    config = Config(
        app=app,
        # 注意：这里不设置 host/port，因为我们用的是 sockets
        loop="none",  # 因为我们已经用了 uvloop.install()
        http="auto",
        lifespan="on",
        access_log=True,
    )
    server = Server(config)

    # 启动服务并传入 socket
    await server.serve(sockets=[sock])

if __name__ == "__main__":
    asyncio.run(serve_with_custom_socket())