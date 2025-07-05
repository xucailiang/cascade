"""
后端配置类型定义

本模块定义了不同VAD后端的配置类型，包括：
- BackendConfig: 后端配置基类
- ONNXConfig: ONNX后端配置
- VLLMConfig: VLLM后端配置
"""


from pydantic import BaseModel, Field, validator

from .vad import OptimizationLevel


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
    graph_optimization_level: str = Field(
        default="all",
        description="图优化级别"
    )

    @validator('providers')
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

    class Config:
        schema_extra = {
            "examples": [
                {
                    "model_path": "/path/to/model.onnx",
                    "providers": ["CPUExecutionProvider"],
                    "intra_op_num_threads": 1
                }
            ]
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

    @validator('dtype')
    def validate_dtype(cls, v):
        """验证数据类型"""
        valid_dtypes = ["auto", "half", "float16", "bfloat16", "float", "float32"]
        if v not in valid_dtypes:
            raise ValueError(f'无效的数据类型: {v}')
        return v
