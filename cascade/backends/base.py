"""
VAD后端抽象基类

定义所有VAD后端必须遵循的接口协议，确保一致性和可扩展性。
"""

import threading
from abc import ABC, abstractmethod
from typing import Any

from cascade.types import AudioChunk, CascadeError, ErrorCode, VADResult


class VADBackend(ABC):
    """
    VAD后端抽象基类
    
    所有VAD算法实现都必须继承此类并实现其抽象方法。
    提供统一的接口用于：
    - 后端初始化和清理
    - 音频块处理和VAD推理
    - 模型预热和性能优化
    - 线程安全保证
    """

    def __init__(self, config: Any):
        """
        初始化VAD后端
        
        Args:
            config: 后端特定的配置对象
        """
        self._config = config
        self._initialized = False
        self._lock = threading.RLock()  # 递归锁，支持嵌套调用

    @property
    def is_initialized(self) -> bool:
        """检查后端是否已初始化"""
        return self._initialized

    @property
    def config(self) -> Any:
        """获取后端配置"""
        return self._config

    @abstractmethod
    async def initialize(self) -> None:
        """
        异步初始化后端
        
        执行必要的初始化操作，包括：
        - 加载模型文件
        - 设置运行时环境
        - 分配资源
        
        Raises:
            CascadeError: 当初始化失败时
        """
        pass

    @abstractmethod
    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        处理单个音频块并返回VAD检测结果
        
        这是核心的推理方法，必须：
        - 线程安全
        - 高性能（目标：< 5ms P99延迟）
        - 稳定可靠
        
        Args:
            chunk: 音频数据块
            
        Returns:
            VAD检测结果
            
        Raises:
            CascadeError: 当处理失败时
        """
        pass

    @abstractmethod
    def warmup(self, dummy_chunk: AudioChunk) -> None:
        """
        使用虚拟数据预热模型
        
        消除首次推理的冷启动延迟，通常在初始化后调用。
        
        Args:
            dummy_chunk: 用于预热的虚拟音频块
            
        Raises:
            CascadeError: 当预热失败时
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        异步关闭后端并释放资源
        
        执行清理操作，包括：
        - 释放模型资源
        - 清理临时文件
        - 关闭会话连接
        """
        pass

    def _ensure_initialized(self) -> None:
        """
        确保后端已初始化
        
        内部辅助方法，在处理前检查初始化状态。
        
        Raises:
            CascadeError: 如果后端未初始化
        """
        if not self._initialized:
            raise CascadeError(
                "VAD后端未初始化，请先调用 initialize() 方法",
                ErrorCode.INITIALIZATION_FAILED
            )

    def _validate_chunk(self, chunk: AudioChunk) -> None:
        """
        验证输入音频块的有效性
        
        Args:
            chunk: 待验证的音频块
            
        Raises:
            CascadeError: 如果音频块无效
        """
        if chunk is None:
            raise CascadeError(
                "音频块不能为空",
                ErrorCode.INVALID_INPUT
            )

        if chunk.chunk_size <= 0:
            raise CascadeError(
                f"音频块大小无效: {chunk.chunk_size}",
                ErrorCode.INVALID_INPUT
            )

        if chunk.sample_rate <= 0:
            raise CascadeError(
                f"采样率无效: {chunk.sample_rate}",
                ErrorCode.INVALID_INPUT
            )

    def get_backend_info(self) -> dict[str, Any]:
        """
        获取后端信息
        
        Returns:
            包含后端详细信息的字典
        """
        return {
            "backend_type": self.__class__.__name__,
            "initialized": self._initialized,
            "config": self._config.__dict__ if hasattr(self._config, '__dict__') else str(self._config)
        }

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        import asyncio
        try:
            # 如果在async环境中，创建新事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在运行的事件循环中，创建任务
                asyncio.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except RuntimeError:
            # 没有事件循环，创建新的
            asyncio.run(self.close())
