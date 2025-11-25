# main.py
from fastapi import FastAPI
import uvicorn

# 创建 FastAPI 应用实例
app = FastAPI(title="My API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
async def create_item(item: dict):
    return {"item": item}

if __name__ == "__main__":
    # 使用 uvicorn.run() 启动服务器，同步方式启动
    uvicorn.run(
        'test_uvicorn:app',
        host="0.0.0.0",
        port=8080,
        reload=True,  # 开发时启用热重载
        log_level="info"
    )


# 测试请求
# curl http://0.0.0.0:8080
# curl http://0.0.0.0:8080/items
# curl http://0.0.0.0:8080/items/3


