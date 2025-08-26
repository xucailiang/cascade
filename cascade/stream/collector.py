"""
语音帧收集器

基于VAD状态机的语音帧收集和管理。
"""

from typing import List, Optional
import time
import logging
from .types import AudioFrame, SpeechSegment, AUDIO_FRAME_SIZE


logger = logging.getLogger(__name__)


class SpeechCollector:
    """
    语音帧收集器
    
    负责收集从VAD检测到start到end之间的所有音频帧，
    并在检测到end时生成完整的语音段。
    """
    
    def __init__(self, segment_id: int):
        """
        初始化语音帧收集器
        
        Args:
            segment_id: 语音段ID
        """
        self.segment_id = segment_id
        self.frames: List[AudioFrame] = []
        self.start_timestamp_ms: Optional[float] = None
        self.start_vad_result: Optional[dict] = None
        self.is_collecting = False
        
        logger.debug(f"SpeechCollector {segment_id} 初始化")
    
    def start_collection(self, start_frame: AudioFrame) -> None:
        """
        开始收集语音帧
        
        Args:
            start_frame: 包含start VAD结果的帧
        """
        if self.is_collecting:
            logger.warning(f"SpeechCollector {self.segment_id} 已在收集中，忽略新的start")
            return
        
        self.frames = [start_frame]
        self.start_timestamp_ms = start_frame.timestamp_ms
        self.start_vad_result = start_frame.vad_result
        self.is_collecting = True
        
        logger.debug(f"SpeechCollector {self.segment_id} 开始收集，时间戳: {self.start_timestamp_ms}ms")
    
    def add_frame(self, frame: AudioFrame) -> None:
        """
        添加音频帧到收集器
        
        Args:
            frame: 要添加的音频帧
        """
        if not self.is_collecting:
            logger.warning(f"SpeechCollector {self.segment_id} 未在收集状态，忽略帧")
            return
        
        self.frames.append(frame)
        logger.debug(f"SpeechCollector {self.segment_id} 添加帧 {frame.frame_id}")
    
    def end_collection(self, end_frame: AudioFrame) -> SpeechSegment:
        """
        结束收集并生成语音段
        
        Args:
            end_frame: 包含end VAD结果的帧
            
        Returns:
            完整的语音段
        """
        if not self.is_collecting:
            raise ValueError(f"SpeechCollector {self.segment_id} 未在收集状态")
        
        if self.start_timestamp_ms is None or self.start_vad_result is None:
            raise ValueError(f"SpeechCollector {self.segment_id} 缺少开始信息")
        
        if end_frame.vad_result is None:
            raise ValueError(f"结束帧缺少VAD结果")
        
        # 添加结束帧
        self.frames.append(end_frame)
        
        # 合并所有音频数据
        audio_data = b''.join(frame.audio_data for frame in self.frames)
        
        # 创建语音段
        segment = SpeechSegment(
            segment_id=self.segment_id,
            audio_data=audio_data,
            start_timestamp_ms=self.start_timestamp_ms,
            end_timestamp_ms=end_frame.timestamp_ms,
            frame_count=len(self.frames),
            start_vad_result=self.start_vad_result,
            end_vad_result=end_frame.vad_result
        )
        
        # 重置状态
        self.is_collecting = False
        logger.info(f"SpeechCollector {self.segment_id} 完成收集: {segment}")
        
        return segment
    
    def reset(self) -> None:
        """重置收集器状态"""
        self.frames.clear()
        self.start_timestamp_ms = None
        self.start_vad_result = None
        self.is_collecting = False
        logger.debug(f"SpeechCollector {self.segment_id} 重置")
    
    @property
    def frame_count(self) -> int:
        """当前收集的帧数"""
        return len(self.frames)
    
    @property
    def duration_ms(self) -> float:
        """当前收集的时长(ms)"""
        if not self.frames or self.start_timestamp_ms is None:
            return 0.0
        return self.frames[-1].timestamp_ms - self.start_timestamp_ms
    
    @property
    def memory_usage_bytes(self) -> int:
        """当前内存使用量(字节)"""
        return sum(len(frame.audio_data) for frame in self.frames)
    
    def __str__(self) -> str:
        status = "collecting" if self.is_collecting else "idle"
        return f"SpeechCollector(id={self.segment_id}, frames={self.frame_count}, {status})"