"""
Cascade VAD后端实现

本模块提供不同VAD算法的后端实现，采用简洁的工厂模式设计。

支持的后端:
- ONNX: 基于ONNXRuntime的高性能推理后端
- SILERO: 基于Silero-VAD的语音活动检测后端

使用方法:
    >>> from cascade.backends import create_vad_backend
    >>> from cascade.types import VADConfig, VADBackend
    >>>
    >>> # 使用ONNX后端
    >>> config = VADConfig(backend=VADBackend.ONNX)
    >>> backend = create_vad_backend(config)
    >>> await backend.initialize()
    >>>
    >>> # 使用Silero后端
    >>> config = VADConfig(backend=VADBackend.SILERO)
    >>> backend = create_vad_backend(config)
    >>> await backend.initialize()
"""

from cascade.types import CascadeError, ErrorCode, VADConfig
from cascade.types import VADBackend as VADBackendEnum

from .base import VADBackend


def create_vad_backend(config: VADConfig) -> VADBackend:
    """
    根据配置创建VAD后端实例
    
    简单直接的工厂函数，根据配置中的后端类型创建相应的实例。
    避免复杂的注册机制，保持代码简洁。
    
    Args:
        config: VAD配置对象，包含后端类型和相关配置
        
    Returns:
        创建的VAD后端实例
        
    Raises:
        CascadeError: 当不支持指定的后端类型时
        
    Example:
        >>> # 使用ONNX后端
        >>> from cascade.types import VADConfig, VADBackend
        >>> config = VADConfig(backend=VADBackend.ONNX)
        >>> backend = create_vad_backend(config)
        >>>
        >>> # 使用Silero后端
        >>> config = VADConfig(backend=VADBackend.SILERO)
        >>> backend = create_vad_backend(config)
    """
    if config.backend == VADBackendEnum.ONNX:
        # 延迟导入避免循环依赖
        try:
            from .onnx import ONNXVADBackend
            # 创建ONNX特定配置 - 这里暂时使用默认配置
            # 实际的ONNX配置将在onnx.py中处理
            return ONNXVADBackend(config)
        except ImportError as e:
            raise CascadeError(
                f"ONNX后端不可用，请确保已安装onnxruntime: {e}",
                ErrorCode.BACKEND_UNAVAILABLE
            )
    elif config.backend == VADBackendEnum.SILERO:
        # 延迟导入避免循环依赖
        try:
            from .silero import SileroVADBackend
            # 创建Silero后端实例
            return SileroVADBackend(config)
        except ImportError as e:
            raise CascadeError(
                f"Silero后端不可用，请确保已安装依赖: pip install silero-vad 或 torch: {e}",
                ErrorCode.BACKEND_UNAVAILABLE
            )
    else:
        raise CascadeError(
            f"不支持的VAD后端: {config.backend}",
            ErrorCode.BACKEND_UNAVAILABLE,
            context={"supported_backends": [VADBackendEnum.ONNX.value, VADBackendEnum.SILERO.value]}
        )


__all__ = [
    "VADBackend",
    "create_vad_backend",
]
