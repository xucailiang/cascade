"""
Cascade VAD后端实现

本模块提供不同VAD算法的后端实现，支持可插拔的架构设计。

支持的后端:
- ONNX: 基于ONNXRuntime的高性能推理后端
- VLLM: 专为大规模语言模型优化的VAD后端

使用方法:
    >>> from cascade.backends import ONNXVADBackend, VADBackend
    >>> from cascade.types import ONNXConfig
    >>> onnx_config = ONNXConfig(model_path="model.onnx")
    >>> backend: VADBackend = ONNXVADBackend(onnx_config)
    >>> await backend.initialize()
"""

from .base import VADBackend
from .onnx import ONNXVADBackend
from .vllm import VLLMVADBackend

__all__ = [
    "VADBackend",
    "ONNXVADBackend",
    "VLLMVADBackend",
]
