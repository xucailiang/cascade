import asyncio
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, AsyncIterator
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入Cascade模块
from cascade.types import VADConfig as CascadeVADConfig, AudioConfig, AudioChunk as CascadeAudioChunk
from cascade.processor import VADProcessor, VADProcessorConfig
from cascade.backends import create_vad_backend

from ..models import VADResult, PerformanceMetrics

logger = logging.getLogger(__name__)

class VADSession:
    """每个WebSocket连接的VAD会话"""

    def __init__(self, config: Dict[str, Any]):
        self.processor = None
        self.config = config
        self.is_initialized = False
        self.audio_queue = asyncio.Queue()

    async def initialize(self) -> bool:
        """初始化VAD处理器"""
        try:
            vad_config = CascadeVADConfig(
                threshold=self.config.get('threshold', 0.5),
                chunk_duration_ms=self.config.get('chunk_duration_ms', 512),
                overlap_ms=self.config.get('overlap_ms', 32),
                backend=self.config.get('backend', 'silero'),
                compensation_ms=self.config.get('compensation_ms', 0)
            )
            audio_config = AudioConfig(
                sample_rate=self.config.get('sample_rate', 16000),
                channels=self.config.get('channels', 1)
            )
            processor_config = VADProcessorConfig(
                vad_config=vad_config, audio_config=audio_config
            )
            backend = create_vad_backend(vad_config)
            await backend.initialize()
            self.processor = VADProcessor(processor_config)
            await self.processor.initialize(backend)
            self.is_initialized = True
            logger.info("VAD会话初始化成功")
            return True
        except Exception as e:
            logger.error(f"VAD会话初始化失败: {e}")
            return False

    async def audio_chunk_generator(self):
        """从队列中生成音频块"""
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:  # 结束信号
                break
            yield chunk

    async def process_stream(self) -> AsyncIterator[VADResult]:
        """处理音频流"""
        if not self.is_initialized:
            raise RuntimeError("VAD会话未初始化")
        
        async for result in self.processor.process_stream(self.audio_chunk_generator()):
            yield VADResult(
                is_speech=result.is_speech,
                probability=result.probability,
                start_ms=result.start_ms,
                end_ms=result.end_ms,
                chunk_id=result.chunk_id,
                processing_time_ms=0,
                is_compensated=getattr(result, 'is_compensated', False),
                original_start_ms=getattr(result, 'original_start_ms', None)
            )

    async def add_audio_chunk(self, audio_data: List[float]):
        """向队列中添加音频块"""
        await self.audio_queue.put(np.array(audio_data, dtype=np.float32))

    async def close(self):
        """关闭会话"""
        if self.processor:
            await self.audio_queue.put(None) # 发送结束信号
            await self.processor.close()
        logger.info("VAD会话已关闭")

class VADService:
    """VAD处理服务，管理多个会话"""
    
    def __init__(self):
        self.sessions: Dict[str, VADSession] = {}

    async def create_session(self, session_id: str, config: Dict[str, Any]) -> VADSession:
        """创建新的VAD会话"""
        session = VADSession(config)
        if await session.initialize():
            self.sessions[session_id] = session
            return session
        return None

    async def get_session(self, session_id: str) -> Optional[VADSession]:
        """获取VAD会话"""
        return self.sessions.get(session_id)

    async def remove_session(self, session_id: str):
        """移除VAD会话"""
        session = self.sessions.pop(session_id, None)
        if session:
            await session.close()
    
    async def process_file(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理音频文件"""
        session = VADSession(config or {})
        if not await session.initialize():
            raise RuntimeError("无法为文件处理创建VAD会话")
            
        import wave
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            raw_audio = wav_file.readframes(wav_file.getnframes())

            if sample_width == 2:
                audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio_data = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"不支持的位深: {sample_width*8}位")

            if channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)

            if sample_rate != 16000:
                from scipy import signal
                audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                sample_rate = 16000
        
        start_time = time.time()
        
        async def file_chunk_generator():
            chunk_size = int(sample_rate * (session.config.get('chunk_duration_ms', 512) / 1000))
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i+chunk_size]

        results = []
        async for result in session.processor.process_stream(file_chunk_generator()):
             if result.is_speech:
                results.append({
                    'is_speech': result.is_speech,
                    'probability': float(result.probability),
                    'start_ms': result.start_ms,
                    'end_ms': result.end_ms,
                    'is_compensated': getattr(result, 'is_compensated', False),
                    'original_start_ms': getattr(result, 'original_start_ms', None)
                })

        processing_time = time.time() - start_time
        metrics = session.processor.get_performance_metrics()
        await session.close()


        return {
            'duration_sec': len(audio_data) / sample_rate,
            'sample_rate': sample_rate,
            'channels': 1,
            'format': 'wav',
            'audio_data': audio_data.tolist(),
            'results': results,
            'performance': {
                'total_processing_time_ms': processing_time * 1000,
                'realtime_factor': (len(audio_data) / sample_rate) / processing_time,
                'avg_latency_ms': metrics.avg_latency_ms if hasattr(metrics, 'avg_latency_ms') else 0
            }
        }
        
# 创建全局VAD服务实例
vad_service = VADService()