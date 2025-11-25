# main_async.py
import asyncio
from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Async Server"}


@app.get("/async-test")
async def async_endpoint():
    await asyncio.sleep(1)  # 模拟异步操作
    return {"message": "异步处理完成"}


async def main():
    """异步启动服务器"""
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )

    server = uvicorn.Server(config)
    await server.serve()  # 启动服务 


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())