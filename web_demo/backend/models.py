from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class VADConfig(BaseModel):
    """VAD配置模型"""
    threshold: float = Field(0.5, ge=0.1, le=0.9, description="语音检测阈值")
    chunk_duration_ms: int = Field(512, description="音频块大小(毫秒)")
    overlap_ms: int = Field(32, description="重叠大小(毫秒)")
    workers: int = Field(4, ge=1, le=8, description="工作线程数")
    backend: str = Field("silero", description="VAD后端")
    compensation_ms: int = Field(0, ge=0, le=500, description="延迟补偿时长(毫秒)")

    class Config:
        json_schema_extra = {
            "example": {
                "threshold": 0.5,
                "chunk_duration_ms": 512,
                "overlap_ms": 32,
                "workers": 4,
                "backend": "silero",
                "compensation_ms": 200
            }
        }

class AudioChunk(BaseModel):
    """音频数据块模型"""
    data: List[float] = Field(..., description="音频数据")
    timestamp: int = Field(..., description="客户端时间戳(ms)")
    sequence: int = Field(..., description="序列号")
    sample_rate: int = Field(16000, description="采样率")

class VADResult(BaseModel):
    """VAD检测结果模型"""
    is_speech: bool = Field(..., description="是否为语音")
    probability: float = Field(..., description="置信度")
    start_ms: float = Field(..., description="开始时间(ms)")
    end_ms: Optional[float] = Field(None, description="结束时间(ms)")
    chunk_id: int = Field(..., description="对应的音频块ID")
    processing_time_ms: float = Field(..., description="处理耗时")
    is_compensated: bool = Field(False, description="是否应用了延迟补偿")
    original_start_ms: Optional[float] = Field(None, description="原始开始时间(ms)")

class PerformanceMetrics(BaseModel):
    """性能指标模型"""
    avg_latency_ms: float = Field(..., description="平均延迟")
    max_latency_ms: float = Field(..., description="最大延迟")
    throughput_chunks_per_sec: float = Field(..., description="吞吐量")
    active_threads: int = Field(..., description="活跃线程数")
    buffer_utilization: float = Field(..., description="缓冲区利用率")
    cpu_usage: float = Field(..., description="CPU使用率")
    memory_usage_mb: float = Field(..., description="内存使用")
    timestamp: int = Field(..., description="时间戳")

class WebSocketMessage(BaseModel):
    """WebSocket消息基类"""
    type: str = Field(..., description="消息类型")

class AudioChunkMessage(WebSocketMessage):
    """音频数据块消息"""
    type: str = "audio_chunk"
    data: List[float] = Field(..., description="音频数据")
    timestamp: int = Field(..., description="客户端时间戳(ms)")
    sequence: int = Field(..., description="序列号")
    sample_rate: int = Field(16000, description="采样率")

class ConfigUpdateMessage(WebSocketMessage):
    """配置更新消息"""
    type: str = "config_update"
    config: Dict[str, Any] = Field(..., description="配置参数")

class StartRecordingMessage(WebSocketMessage):
    """开始录音消息"""
    type: str = "start_recording"
    config: Optional[Dict[str, Any]] = Field(None, description="配置参数")

class StopRecordingMessage(WebSocketMessage):
    """停止录音消息"""
    type: str = "stop_recording"

class VADResultMessage(WebSocketMessage):
    """VAD结果消息"""
    type: str = "vad_result"
    is_speech: bool = Field(..., description="是否为语音")
    probability: float = Field(..., description="置信度")
    start_ms: float = Field(..., description="开始时间(ms)")
    end_ms: Optional[float] = Field(None, description="结束时间(ms)")
    chunk_id: int = Field(..., description="对应的音频块ID")
    processing_time_ms: float = Field(..., description="处理耗时")
    is_compensated: bool = Field(False, description="是否应用了延迟补偿")
    original_start_ms: Optional[float] = Field(None, description="原始开始时间(ms)")

class PerformanceMetricsMessage(WebSocketMessage):
    """性能指标消息"""
    type: str = "performance_metrics"
    metrics: Dict[str, Any] = Field(..., description="性能指标")
    timestamp: int = Field(..., description="时间戳")

class StatusMessage(WebSocketMessage):
    """状态消息"""
    type: str = "status"
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    timestamp: int = Field(..., description="时间戳")

class ErrorMessage(WebSocketMessage):
    """错误消息"""
    type: str = "error"
    code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")

class FileUploadResponse(BaseModel):
    """文件上传响应"""
    status: str = Field(..., description="状态")
    file_id: str = Field(..., description="文件ID")
    filename: str = Field(..., description="文件名")
    duration_sec: float = Field(..., description="时长(秒)")
    sample_rate: int = Field(..., description="采样率")
    channels: int = Field(..., description="声道数")
    format: str = Field(..., description="格式")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="VAD结果")
    performance: Optional[Dict[str, Any]] = Field(None, description="性能指标")