#!/usr/bin/env python3
"""
Cascade WebSocket VAD 演示服务器

新架构特点:
- 每个客户端一个处理器实例，生命周期与会话绑定。
- 移除了复杂的队列和后台任务，采用直接调用的方式处理音频块。
- 保证了逻辑的同步性和可预测性，修复了此前所有版本的问题。
"""

import base64
import json
import logging
import uuid

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import cascade

# --- 配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cascade VAD 演示")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 数据模型 ---
class VADConfig(BaseModel):
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    speech_pad_ms: int = Field(default=100, ge=0)
    min_silence_duration_ms: int = Field(default=100, ge=0)
    sample_rate: int = Field(default=16000)

# --- 会话管理器 ---
class SessionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.processors: dict[str, cascade.StreamProcessor] = {}
        self.configs: dict[str, VADConfig] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"客户端 {client_id} 已连接")

    async def disconnect(self, client_id: str):
        await self.stop_session(client_id)
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        logger.info(f"客户端 {client_id} 已断开")

    async def start_session(self, client_id: str, config: VADConfig):
        if client_id in self.processors:
            await self.stop_session(client_id)

        # 直接使用前端传入的config字典，Pydantic模型确保了字段的正确性
        processor = cascade.create_processor(**config.model_dump())
        await processor.start()
        self.processors[client_id] = processor
        self.configs[client_id] = config
        logger.info(f"为客户端 {client_id} 启动了新的处理会话")

    async def stop_session(self, client_id: str):
        self.configs.pop(client_id, None)
        processor = self.processors.pop(client_id, None)
        if processor:
            await processor.stop()
            logger.info(f"客户端 {client_id} 的处理会话已停止")

    async def process_chunk(self, client_id: str, chunk: bytes):
        if client_id not in self.processors:
            logger.warning(f"客户端 {client_id} 没有活动的处理器，忽略音频块")
            return

        websocket = self.active_connections.get(client_id)
        if not websocket:
            return

        processor = self.processors[client_id]
        try:
            results: list[cascade.CascadeResult] = await processor.process_chunk(chunk)
            for result in results:
                response_dict = None
                config = self.configs.get(client_id)
                if not config:
                    continue # 如果没有配置信息，则跳过

                # 只处理语音段结果，不再发送单帧结果
                if result.is_speech_segment and result.segment:
                    segment = result.segment
                    audio_data_b64 = None
                    if segment.audio_data and config:
                        # 直接对原始PCM数据进行Base64编码，将WAV转换移到前端
                        audio_data_b64 = base64.b64encode(segment.audio_data).decode('utf-8')

                    response_dict = {
                        "type": "segment",
                        "segment": {
                            "segment_id": segment.segment_id,
                            "start_timestamp_ms": segment.start_timestamp_ms,
                            "end_timestamp_ms": segment.end_timestamp_ms,
                            "duration_ms": segment.duration_ms,
                            "audio_data": audio_data_b64,
                        }
                    }

                if response_dict:
                    await websocket.send_json(response_dict)
        except Exception as e:
            logger.error(f"处理音频块失败 for {client_id}: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})


manager = SessionManager()

# --- WebSocket 端点 ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    if client_id == "new":
        client_id = str(uuid.uuid4())

    await manager.connect(websocket, client_id)
    await websocket.send_json({"type": "connection_ready", "client_id": client_id})

    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "start":
                        config = VADConfig(**data.get("config", {}))
                        await manager.start_session(client_id, config)
                    elif msg_type == "stop":
                        await manager.stop_session(client_id)

                except Exception as e:
                    logger.error(f"处理文本消息失败 from {client_id}: {e}")
                    await websocket.send_json({"type": "error", "message": f"无效消息: {e}"})

            elif "bytes" in message:
                await manager.process_chunk(client_id, message["bytes"])

    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 主动断开")
    except Exception as e:
        logger.error(f"与客户端 {client_id} 通信时发生未知错误: {e}")
    finally:
        await manager.disconnect(client_id)

# --- 根路由和主程序 ---
@app.get("/")
async def get_index():
    return {"message": "Cascade VAD WebSocket API", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
