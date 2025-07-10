"""
后端配置类型定义

本模块定义了不同VAD后端的配置类型，包括：
- BackendConfig: 后端配置基类
- ONNXConfig: ONNX后端配置
- VLLMConfig: VLLM后端配置
"""


from enum import Enum

from onnxruntime import GraphOptimizationLevel
from pydantic import BaseModel, Field, field_validator, model_validator

from .vad import OptimizationLevel, VADSensitivity


class BackendConfig(BaseModel):
    """VAD后端配置基类"""
    model_path: str | None = Field(
        default=None,
        description="模型文件路径"
    )
    device: str = Field(
        default="cpu",
        description="计算设备"
    )
    optimization_level: OptimizationLevel = Field(
        default=OptimizationLevel.ALL,
        description="优化级别"
    )
    max_batch_size: int = Field(
        default=1,
        description="最大批处理大小",
        ge=1,
        le=64
    )
    warmup_iterations: int = Field(
        default=3,
        description="预热迭代次数",
        ge=0,
        le=10
    )

    class Config:
        extra = "allow"  # 允许后端特定配置


class ONNXConfig(BackendConfig):
    """ONNX后端配置"""
    sample_rate: int = Field(
        default=16000,
        description="音频采样率",
        ge=8000,
        le=48000
    )
    threshold: float = Field(
        default=0.5,
        description="VAD检测阈值",
        ge=0.0,
        le=1.0
    )
    chunk_duration_ms: int = Field(
        default=250,
        description="块时长（毫秒）",
        ge=10,
        le=5000
    )
    providers: list[str] = Field(
        default=["CPUExecutionProvider"],
        description="执行提供者列表"
    )
    intra_op_num_threads: int = Field(
        default=1,
        description="线程内操作线程数",
        ge=1,
        le=16
    )
    inter_op_num_threads: int = Field(
        default=1,
        description="线程间操作线程数",
        ge=1,
        le=16
    )
    execution_mode: str = Field(
        default="sequential",
        description="执行模式"
    )
    graph_optimization_level: GraphOptimizationLevel = Field(
        default=GraphOptimizationLevel.ORT_ENABLE_ALL,
        description="图优化级别"
    )

    @field_validator('graph_optimization_level', mode='before')
    def validate_graph_optimization_level(cls, v):
        """验证并转换图优化级别"""
        if isinstance(v, str):
            level_map = {
                "none": GraphOptimizationLevel.ORT_DISABLE_ALL,
                "basic": GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all": GraphOptimizationLevel.ORT_ENABLE_ALL,
            }
            if v.lower() not in level_map:
                raise ValueError(f"无效的图优化级别: {v}")
            return level_map[v.lower()]
        return v

    @field_validator('providers')
    def validate_providers(cls, v):
        """验证执行提供者"""
        valid_providers = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "OpenVINOExecutionProvider"
        ]
        for provider in v:
            if provider not in valid_providers:
                raise ValueError(f'无效的执行提供者: {provider}')
        return v

    model_config = {
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "examples": [
                {
                    "model_path": "/path/to/model.onnx",
                    "providers": ["CPUExecutionProvider"],
                    "intra_op_num_threads": 1,
                }
            ]
        },
    }


class VLLMConfig(BackendConfig):
    """VLLM后端配置"""
    tensor_parallel_size: int = Field(
        default=1,
        description="张量并行大小",
        ge=1,
        le=8
    )
    max_model_len: int = Field(
        default=2048,
        description="最大模型长度",
        ge=512,
        le=8192
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        description="GPU内存利用率",
        ge=0.1,
        le=1.0
    )
    swap_space: int = Field(
        default=4,
        description="交换空间（GB）",
        ge=0,
        le=32
    )
    dtype: str = Field(
        default="auto",
        description="数据类型"
    )

    @field_validator('dtype')
    def validate_dtype(cls, v):
        """验证数据类型"""
        valid_dtypes = ["auto", "half", "float16", "bfloat16", "float", "float32"]
        if v not in valid_dtypes:
            raise ValueError(f'无效的数据类型: {v}')
        return v

class OverlapStrategy(str, Enum):
    """重叠处理策略"""
    FRONT_PRIORITY = "front_priority"  # 前块优先
    BACK_PRIORITY = "back_priority"    # 后块优先
    MAX_CONFIDENCE = "max_confidence"  # 最高置信度优先


class ProcessorConfig(BaseModel):
    """处理器配置"""
    chunk_duration_ms: int = Field(
        default=250,
        description="块时长（毫秒）",
        ge=10,
        le=5000
    )
    overlap_ms: int = Field(
        default=16,
        description="重叠区域时长（毫秒）",
        ge=0,
        le=100
    )
    overlap_strategy: OverlapStrategy = Field(
        default=OverlapStrategy.FRONT_PRIORITY,
        description="重叠处理策略"
    )
    max_workers: int | None = Field(
        default=None,
        description="最大工作线程数，None表示使用默认值"
    )
    thread_name_prefix: str = Field(
        default="audio-processor",
        description="线程名称前缀"
    )


class VADProcessorConfig(ProcessorConfig):
    """VAD 处理器配置"""
    # 基本配置
    sample_rate: int = Field(
        default=16000,
        description="音频采样率",
        ge=8000,
        le=48000
    )
    sensitivity: VADSensitivity = Field(
        default=VADSensitivity.MEDIUM,
        description="VAD 灵敏度"
    )
    backend_type: str = Field(
        default="onnx",
        description="VAD后端类型，支持'onnx'等"
    )
    model_path: str = Field(
        default="",
        description="VAD模型路径"
    )
    threshold: float = Field(
        default=0.5,
        description="VAD检测阈值",
        ge=0.0,
        le=1.0
    )

    # 线程池配置
    workers: int = Field(
        default=4,
        description="工作线程数",
        ge=1,
        le=32
    )

    # 结果合并配置
    min_speech_duration_ms: int = Field(
        default=100,
        description="最小语音持续时间（毫秒）",
        ge=10,
        le=1000
    )
    min_silence_duration_ms: int = Field(
        default=300,
        description="最小静音持续时间（毫秒）",
        ge=10,
        le=1000
    )
    smoothing_window: int = Field(
        default=3,
        description="平滑窗口大小",
        ge=1,
        le=10
    )

    # 兼容旧版本的配置
    energy_threshold_low: float = Field(
        default=0.01,
        description="低灵敏度能量阈值（已废弃）",
        ge=0.0,
        le=1.0,
        deprecated=True
    )
    energy_threshold_medium: float = Field(
        default=0.005,
        description="中等灵敏度能量阈值（已废弃）",
        ge=0.0,
        le=1.0,
        deprecated=True
    )
    energy_threshold_high: float = Field(
        default=0.002,
        description="高灵敏度能量阈值（已废弃）",
        ge=0.0,
        le=1.0,
        deprecated=True
    )
    speech_pad_ms: int = Field(
        default=30,
        description="语音段前后填充时间（毫秒）",
        ge=0,
        le=500
    )
    normalize_audio: bool = Field(
        default=True,
        description="是否归一化音频"
    )

    @model_validator(mode='after')
    def update_threshold_by_sensitivity(self):
        """根据灵敏度更新阈值"""
        if self.sensitivity == VADSensitivity.LOW:
            self.threshold = 0.7
        elif self.sensitivity == VADSensitivity.MEDIUM:
            self.threshold = 0.5
        elif self.sensitivity == VADSensitivity.HIGH:
            self.threshold = 0.3
        return self
