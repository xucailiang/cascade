import asyncio
import json
import logging
import time
import traceback
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..models import (
    AudioChunkMessage,
    StartRecordingMessage,
)
from ..services.vad_service import VADSession, vad_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session: VADSession | None = None

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message_type = message_data.get('type')

            if message_type == 'start_recording':
                message = StartRecordingMessage(**message_data)
                config = message.config or {
                    'threshold': 0.5, 'chunk_duration_ms': 512, 'overlap_ms': 32,
                    'backend': 'silero', 'sample_rate': 16000, 'channels': 1,
                    'compensation_ms': 0
                }
                session = await vad_service.create_session(session_id, config)
                if session:
                    asyncio.create_task(stream_vad_results(websocket, session))
                    await websocket.send_json({
                        'type': 'status', 'status': 'recording_started',
                        'message': '录音已开始', 'timestamp': int(time.time() * 1000)
                    })
                else:
                    await websocket.send_json({
                        'type': 'error', 'code': 'initialization_failed',
                        'message': 'VAD会话初始化失败'
                    })

            elif message_type == 'audio_chunk':
                if not session:
                    await websocket.send_json({
                        'type': 'error', 'code': 'not_initialized',
                        'message': 'VAD会话未初始化，请先发送start_recording'
                    })
                    continue
                message = AudioChunkMessage(**message_data)
                await session.add_audio_chunk(message.data)

            elif message_type == 'stop_recording':
                await websocket.send_json({
                    'type': 'status', 'status': 'recording_stopped',
                    'message': '录音已停止', 'timestamp': int(time.time() * 1000)
                })
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket连接已断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        await websocket.send_json({
            'type': 'error', 'code': 'processing_error',
            'message': f"处理消息失败: {e}", 'details': traceback.format_exc()
        })
    finally:
        await vad_service.remove_session(session_id)
        logger.info(f"会话已移除: {session_id}")

async def stream_vad_results(websocket: WebSocket, session: VADSession):
    """将VAD结果流式传输到客户端"""
    try:
        async for result in session.process_stream():
            await websocket.send_json({
                'type': 'vad_result',
                'is_speech': result.is_speech, 'probability': result.probability,
                'start_ms': result.start_ms, 'end_ms': result.end_ms,
                'chunk_id': result.chunk_id, 'processing_time_ms': result.processing_time_ms,
                'is_compensated': result.is_compensated, 'original_start_ms': result.original_start_ms
            })
    except Exception as e:
        logger.error(f"流式处理VAD结果时出错: {e}")
