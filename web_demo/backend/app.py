import logging
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from .routes import websocket, upload

# 配置日志
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Cascade VAD Web Demo",
    description="一个用于测试Cascade VAD处理库的Web界面",
    version="0.1.0"
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="web_demo/frontend", html=True), name="static")

# 包含路由
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
app.include_router(upload.router, prefix="/api", tags=["File Upload"])

@app.get("/")
async def read_root():
    """根路径，提供一个简单的欢迎信息"""
    return {"message": "欢迎使用Cascade VAD Web演示界面"}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=204)