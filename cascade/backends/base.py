"""
VAD后端抽象基类模块

本模块定义了所有VAD后端实现必须遵循的接口。
"""

from abc import ABC, abstractmethod

from cascade.types.audio import AudioChunk
from cascade.types.config import BackendConfig
from cascade.types.vad import VADResult


class VADBackend(ABC):
    """VAD后端模块的抽象基类"""

    def __init__(self, config: BackendConfig):
        """
        初始化后端。
        
        Args:
            config: 后端的配置对象。
        """
        self.config = config
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        初始化后端，加载模型并准备推理引擎。
        此方法应在主线程中调用，以确保线程安全。
        """
        pass

    @abstractmethod
    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        同步处理单个音频块。
        此方法将在工作线程中被调用，必须是线程安全的。

        Args:
            chunk: 待处理的音频块。

        Returns:
            VAD处理结果。
        """
        pass
    
    @abstractmethod
    def warmup(self) -> None:
        """
        预热模型，消除首次推理的延迟。
        此方法将在每个工作线程中被调用一次。
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭后端，释放所有资源。"""
        pass
        
    @property
    def is_initialized(self) -> bool:
        """返回后端是否已初始化。"""
        return self._is_initialized