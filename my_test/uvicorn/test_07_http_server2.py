# server_with_sockets.py
import asyncio
import socket
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from uvicorn.config import Config
from uvicorn.main import Server
import os
import signal
import logging
from typing import List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UVloopServer")


class SocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = self._create_fastapi_app()
        self.server: Optional[Server] = None
        self.sockets: List[socket.socket] = []

    def _create_fastapi_app(self) -> FastAPI:
        """创建 FastAPI 应用"""
        app = FastAPI(
            title="UVloop Socket Server",
            description="使用 uvloop + sockets 的高性能服务器",
            version="1.0.0"
        )

        # 添加路由
        @app.get("/")
        async def root():
            return {
                "message": "UVloop + Socket Server",
                "status": "running",
                "event_loop": str(type(asyncio.get_event_loop()))
            }

        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

        @app.get("/info")
        async def server_info(request: Request):
            return {
                "host": self.host,
                "port": self.port,
                "client_host": request.client.host if request.client else "unknown",
                "headers": dict(request.headers)
            }

        @app.get("/performance")
        async def performance_test():
            """性能测试端点"""
            start_time = asyncio.get_event_loop().time()

            # 模拟并发异步操作
            async def mock_async_operation(delay: float, operation_id: int):
                await asyncio.sleep(delay)
                return f"操作{operation_id}完成"

            # 创建多个并发任务
            delays = [0.01, 0.02, 0.005, 0.015, 0.01]
            tasks = [mock_async_operation(delay, i) for i, delay in enumerate(delays)]
            results = await asyncio.gather(*tasks)

            processing_time = asyncio.get_event_loop().time() - start_time

            return {
                "results": results,
                "concurrent_tasks": len(tasks),
                "processing_time": f"{processing_time:.3f}秒",
                "efficiency": f"{(sum(delays) / processing_time):.2f}x"
            }

        @app.post("/echo")
        async def echo_data(request: Request):
            """回显接收到的数据"""
            data = await request.json()
            return {
                "received": data,
                "echo": "数据已接收并回显",
                "timestamp": asyncio.get_event_loop().time()
            }

        return app

    def create_sockets(self) -> List[socket.socket]:
        """创建并绑定 sockets"""
        # 设置 socket 选项
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 在 Windows 上设置 SO_REUSEADDR 的不同行为
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass  # 某些系统可能不支持 SO_REUSEPORT

        # 绑定地址和端口
        try:
            sock.bind((self.host, self.port))
            sock.listen(100)  # 设置 backlog
            logger.info(f"Socket 绑定成功: {self.host}:{self.port}")
        except OSError as e:
            logger.error(f"Socket 绑定失败: {e}")
            sock.close()
            raise

        self.sockets = [sock]
        return self.sockets

    def setup_uvloop(self):
        """设置 uvloop 事件循环"""
        # 设置 uvloop 作为事件循环策略
        uvloop.install()
        logger.info("UVloop 事件循环已设置")

        # 获取当前事件循环并进行配置
        loop = asyncio.get_event_loop()

        # 配置事件循环参数
        if hasattr(loop, 'set_debug'):
            loop.set_debug(False)  # 生产环境关闭调试

        logger.info(f"使用事件循环: {type(loop).__name__}")
        return loop

    def setup_signal_handlers(self, loop: asyncio.AbstractEventLoop):
        """设置信号处理器"""

        def shutdown_handler():
            logger.info("收到关闭信号，正在停止服务器...")
            if self.server:
                self.server.should_exit = True

        try:
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop.add_signal_handler(sig, shutdown_handler)
            logger.info("信号处理器设置完成")
        except NotImplementedError:
            logger.warning("当前平台不支持信号处理器")

    async def start_server(self):
        """启动服务器"""
        try:
            # 1. 设置 uvloop
            loop = self.setup_uvloop()

            # 2. 创建 sockets
            sockets = self.create_sockets()

            # 3. 创建 uvicorn 配置
            config = Config(
                app=self.app,
                host=self.host,
                port=self.port,
                loop="uvloop",
                log_level="info",
                access_log=True,
                use_colors=True,
                proxy_headers=True,
                server_header=True,
                date_header=True,
            )

            # 4. 创建并配置服务器
            self.server = Server(config=config)

            # 5. 设置信号处理器
            self.setup_signal_handlers(loop)

            # 6. 启动服务器
            logger.info("启动 uvicorn 服务器...")
            await self.server.serve(sockets=sockets)

        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            raise
        finally:
            await self.cleanup()

    async def cleanup(self):
        """清理资源"""
        logger.info("正在清理资源...")

        # 关闭 sockets
        for sock in self.sockets:
            try:
                sock.close()
            except Exception as e:
                logger.error(f"关闭 socket 时出错: {e}")

        self.sockets.clear()
        logger.info("资源清理完成")

    def print_startup_info(self):
        """打印启动信息"""
        print("=" * 60)
        print("🚀 FastAPI 服务器启动信息")
        print("=" * 60)
        print(f"📡 服务器地址: http://{self.host}:{self.port}")
        print(f"📊 健康检查: http://{self.host}:{self.port}/health")
        print(f"📖 API文档: http://{self.host}:{self.port}/docs")
        print(f"🔧 事件循环: UVloop")
        print(f"💻 进程ID: {os.getpid()}")
        print("=" * 60)


async def main():
    """主函数"""
    # 创建服务器实例
    server = SocketServer(host="0.0.0.0", port=8080)

    # 打印启动信息
    server.print_startup_info()

    try:
        # 启动服务器
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("用户中断服务器")
    except Exception as e:
        logger.error(f"服务器运行错误: {e}")
    finally:
        logger.info("服务器已停止")


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())