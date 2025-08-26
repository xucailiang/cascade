"""
Cascade处理实例

基于1:1:1绑定架构的单个处理实例，集成VAD状态机和音频处理。
"""

import asyncio
import threading
import time
import logging
from typing import Optional, AsyncIterator, Iterator
from queue import Queue, Empty
import numpy as np

from ..backends.silero import SileroVADBackend
from ..types import VADConfig, AudioChunk
from .types import AudioFrame, CascadeResult, Config, AUDIO_FRAME_SIZE
from .state_machine import VADStateMachine


logger = logging.getLogger(__name__)


class CascadeInstance:
    """
    Cascade处理实例
    
    实现1:1:1绑定架构：
    - 一个线程处理音频数据
    - 一个VAD实例进行语音检测
    - 一个状态机管理语音段收集
    """
    
    def __init__(self, instance_id: str, config: Config):
        """
        初始化Cascade实例
        
        Args:
            instance_id: 实例唯一标识
            config: 配置对象
        """
        self.instance_id = instance_id
        self.config = config
        
        # VAD后端和状态机
        vad_config = VADConfig(threshold=config.vad_threshold)
        self.vad_backend = SileroVADBackend(vad_config)
        self.state_machine = VADStateMachine(instance_id)
        
        # 线程和队列
        self.input_queue: Queue = Queue(maxsize=config.buffer_size_frames)
        self.output_queue: Queue = Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.frame_counter = 0
        
        # 统计信息
        self.total_frames_processed = 0
        self.total_processing_time_ms = 0.0
        self.error_count = 0
        
        logger.info(f"CascadeInstance {instance_id} 初始化完成")
    
    def start(self) -> None:
        """启动处理线程"""
        if self.is_running:
            logger.warning(f"CascadeInstance {self.instance_id} 已在运行")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"cascade-{self.instance_id}",
            daemon=True
        )
        self.worker_thread.start()
        logger.info(f"CascadeInstance {self.instance_id} 启动")
    
    def stop(self) -> None:
        """停止处理线程"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 等待线程结束
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        # 清空队列
        self._clear_queues()
        
        logger.info(f"CascadeInstance {self.instance_id} 停止")
    
    def process_audio_chunk(self, audio_data: bytes) -> None:
        """
        处理音频块
        
        Args:
            audio_data: 音频数据（必须是512样本的倍数）
        """
        if not self.is_running:
            raise RuntimeError(f"CascadeInstance {self.instance_id} 未启动")
        
        # 验证音频数据长度
        expected_bytes = AUDIO_FRAME_SIZE * 2  # 16-bit samples
        if len(audio_data) % expected_bytes != 0:
            logger.warning(f"音频数据长度 {len(audio_data)} 不是 {expected_bytes} 的倍数，丢弃")
            return
        
        # 分割为512样本帧
        for i in range(0, len(audio_data), expected_bytes):
            frame_data = audio_data[i:i + expected_bytes]
            if len(frame_data) == expected_bytes:
                try:
                    self.input_queue.put_nowait(frame_data)
                except:
                    logger.warning(f"CascadeInstance {self.instance_id} 输入队列满，丢弃帧")
    
    def get_results(self) -> Iterator[CascadeResult]:
        """
        获取处理结果
        
        Yields:
            处理结果
        """
        while True:
            try:
                result = self.output_queue.get_nowait()
                yield result
            except Empty:
                break
    
    async def get_results_async(self) -> AsyncIterator[CascadeResult]:
        """
        异步获取处理结果
        
        Yields:
            处理结果
        """
        while self.is_running:
            try:
                # 非阻塞检查
                result = self.output_queue.get_nowait()
                yield result
            except Empty:
                # 短暂等待避免CPU占用过高
                await asyncio.sleep(0.001)
    
    def _worker_loop(self) -> None:
        """工作线程主循环"""
        logger.debug(f"CascadeInstance {self.instance_id} 工作线程启动")
        
        while self.is_running:
            try:
                # 获取音频帧数据
                try:
                    frame_data = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # 处理音频帧
                self._process_frame(frame_data)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"CascadeInstance {self.instance_id} 处理错误: {e}")
        
        logger.debug(f"CascadeInstance {self.instance_id} 工作线程结束")
    
    def _process_frame(self, frame_data: bytes) -> None:
        """
        处理单个音频帧
        
        Args:
            frame_data: 512样本的音频数据
        """
        start_time = time.time()
        
        try:
            # 创建音频帧对象
            self.frame_counter += 1
            timestamp_ms = self.frame_counter * 32.0  # 32ms per frame
            
            # 转换为numpy数组进行VAD检测
            audio_array = np.frombuffer(frame_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 创建AudioChunk用于VAD检测
            audio_chunk = AudioChunk(
                data=audio_array,
                sequence_number=self.frame_counter,
                start_frame=self.frame_counter * AUDIO_FRAME_SIZE,
                chunk_size=AUDIO_FRAME_SIZE,
                timestamp_ms=timestamp_ms,
                sample_rate=16000
            )
            
            # VAD检测
            vad_result = self.vad_backend.process_chunk(audio_chunk)
            
            # 转换VAD结果为字典格式
            vad_dict = None
            if vad_result.original_result:
                vad_dict = vad_result.original_result
            
            # 创建AudioFrame
            frame = AudioFrame(
                frame_id=self.frame_counter,
                audio_data=frame_data,
                timestamp_ms=timestamp_ms,
                vad_result=vad_dict
            )
            
            # 状态机处理
            result = self.state_machine.process_frame(frame)
            
            # 输出结果
            if result:
                self.output_queue.put_nowait(result)
            
            # 更新统计
            processing_time_ms = (time.time() - start_time) * 1000
            self.total_frames_processed += 1
            self.total_processing_time_ms += processing_time_ms
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"CascadeInstance {self.instance_id} 帧处理失败: {e}")
    
    def _clear_queues(self) -> None:
        """清空所有队列"""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except Empty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except Empty:
                break
    
    def reset(self) -> None:
        """重置实例状态"""
        self.state_machine.reset()
        self._clear_queues()
        self.frame_counter = 0
        self.total_frames_processed = 0
        self.total_processing_time_ms = 0.0
        self.error_count = 0
        logger.info(f"CascadeInstance {self.instance_id} 重置")
    
    @property
    def is_busy(self) -> bool:
        """是否正在处理"""
        return not self.input_queue.empty() or not self.output_queue.empty()
    
    @property
    def queue_size(self) -> int:
        """输入队列大小"""
        return self.input_queue.qsize()
    
    @property
    def average_processing_time_ms(self) -> float:
        """平均处理时间"""
        if self.total_frames_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_frames_processed
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.total_frames_processed == 0:
            return 0.0
        return self.error_count / self.total_frames_processed
    
    def __str__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return f"CascadeInstance({self.instance_id}, {status}, frames={self.total_frames_processed})"
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()