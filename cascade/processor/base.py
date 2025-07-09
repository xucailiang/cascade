"""
音频处理器基础模块

本模块定义了音频处理器的基础接口和通用组件，包括：
- 处理器抽象基类
- 重叠处理策略
- 线程池管理
- 结果合并工具
"""

import abc
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from cascade.buffer.base import AudioBuffer
from cascade.types.audio import AudioConfig


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


class AudioChunk(BaseModel):
    """音频数据块"""
    data: np.ndarray = Field(description="音频数据")
    sequence_number: int = Field(description="序列号", ge=0)
    start_frame: int = Field(description="起始帧位置", ge=0)
    chunk_size: int = Field(description="主要块大小（样本数）", gt=0)
    overlap_size: int = Field(default=0, description="重叠区域大小（样本数）", ge=0)
    timestamp_ms: float = Field(description="时间戳（毫秒）", ge=0.0)
    sample_rate: int = Field(description="采样率", gt=0)
    is_last: bool = Field(default=False, description="是否为最后一块")
    metadata: dict[str, Any] | None = Field(default=None, description="附加元数据")

    class Config:
        arbitrary_types_allowed = True

    def get_total_size(self) -> int:
        """获取总大小（包括重叠）"""
        return self.chunk_size + self.overlap_size

    def get_duration_ms(self) -> float:
        """获取块时长（毫秒）"""
        return self.chunk_size * 1000.0 / self.sample_rate

    def get_end_timestamp_ms(self) -> float:
        """获取结束时间戳"""
        return self.timestamp_ms + self.get_duration_ms()


class ProcessResult(BaseModel):
    """处理结果"""
    chunk: AudioChunk = Field(description="处理的音频块")
    result_data: Any = Field(description="处理结果数据")
    success: bool = Field(default=True, description="处理是否成功")
    error: str | None = Field(default=None, description="错误信息")
    processing_time_ms: float = Field(default=0.0, description="处理时间（毫秒）")
    metadata: dict[str, Any] | None = Field(default=None, description="附加元数据")

    class Config:
        arbitrary_types_allowed = True


class MonitoredThreadPool:
    """可监控的线程池"""
    def __init__(self, max_workers=None, thread_name_prefix=''):
        """
        初始化线程池
        
        Args:
            max_workers: 最大工作线程数，None表示使用默认值
            thread_name_prefix: 线程名称前缀
        """
        # 使用标准库的ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self._lock = threading.RLock()

    def submit(self, fn, *args, **kwargs):
        """
        提交任务到线程池
        
        Args:
            fn: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Future对象
        """
        with self._lock:
            self.active_tasks += 1

        future = self.executor.submit(fn, *args, **kwargs)

        def _done_callback(future):
            with self._lock:
                self.active_tasks -= 1
                if future.exception() is not None:
                    self.failed_tasks += 1
                else:
                    self.completed_tasks += 1

        future.add_done_callback(_done_callback)
        return future

    def shutdown(self, wait=True):
        """
        关闭线程池
        
        Args:
            wait: 是否等待所有任务完成
        """
        self.executor.shutdown(wait=wait)

    def get_stats(self):
        """
        获取线程池统计信息
        
        Returns:
            包含统计信息的字典
        """
        with self._lock:
            return {
                "active_tasks": self.active_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_tasks": self.active_tasks + self.completed_tasks + self.failed_tasks
            }


class AudioProcessor(abc.ABC):
    """音频处理器抽象基类"""

    def __init__(self, config: ProcessorConfig | None = None):
        """
        初始化音频处理器
        
        Args:
            config: 处理器配置，如果为None则使用默认配置
        """
        self.config = config or ProcessorConfig()
        self.thread_pool = MonitoredThreadPool(
            max_workers=self.config.max_workers,
            thread_name_prefix=self.config.thread_name_prefix
        )

    async def process_audio(self, audio_data: np.ndarray, audio_config: AudioConfig) -> list[ProcessResult]:
        """
        处理音频数据
        
        Args:
            audio_data: 音频数据
            audio_config: 音频配置
            
        Returns:
            处理结果列表
            
        Raises:
            ValueError: 当音频数据无效时
        """
        # 使用管道模式处理
        pipeline = [
            self._prepare_audio,     # 准备音频（格式转换等）
            self._split_into_chunks, # 分割为块
            self._process_chunks,    # 处理块
            self._merge_results      # 合并结果
        ]

        # 初始化上下文
        context = {
            "audio_data": audio_data,
            "audio_config": audio_config,
            "processor_config": self.config,
            "chunks": [],
            "results": []
        }

        # 执行管道
        for step in pipeline:
            context = await step(context)

        return context["results"]

    async def process_buffer(self, buffer: AudioBuffer, audio_config: AudioConfig, max_size: int = -1) -> list[ProcessResult]:
        """
        处理音频缓冲区
        
        Args:
            buffer: 音频缓冲区
            audio_config: 音频配置
            max_size: 最大处理样本数，-1表示处理所有可用样本
            
        Returns:
            处理结果列表
            
        Raises:
            ValueError: 当缓冲区无效时
        """
        # 从缓冲区读取数据
        size_to_read = buffer.available() if max_size == -1 else min(max_size, buffer.available())
        audio_data = buffer.read(size_to_read)

        # 处理音频数据
        return await self.process_audio(audio_data, audio_config)

    async def _prepare_audio(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        准备音频数据
        
        Args:
            context: 处理上下文
            
        Returns:
            更新后的上下文
            
        Raises:
            ValueError: 当音频数据无效时
        """
        # 默认实现，子类可以重写
        return context

    async def _split_into_chunks(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        将音频分割为处理块
        
        Args:
            context: 处理上下文
            
        Returns:
            更新后的上下文
            
        Raises:
            ValueError: 当分割失败时
        """
        audio_data = context["audio_data"]
        audio_config = context["audio_config"]
        processor_config = context["processor_config"]

        # 计算块大小和重叠大小（样本数）
        chunk_size = int(processor_config.chunk_duration_ms * audio_config.sample_rate / 1000)
        overlap_size = int(processor_config.overlap_ms * audio_config.sample_rate / 1000)

        # 确保块大小大于重叠大小
        if chunk_size <= overlap_size:
            raise ValueError(f"块大小({chunk_size})必须大于重叠大小({overlap_size})")

        # 计算总样本数和块数
        total_samples = len(audio_data)
        step_size = chunk_size - overlap_size
        
        # 确保至少有一个块（即使音频很短）
        if total_samples == 0:
            num_chunks = 0
            chunks = []
            context["chunks"] = chunks
            return context
        elif total_samples < chunk_size:
            # 短音频只有一个块（严格小于一个块大小）
            num_chunks = 1
            chunk = AudioChunk(
                data=audio_data,
                sequence_number=0,
                start_frame=0,
                chunk_size=total_samples,
                overlap_size=0,
                timestamp_ms=0,
                sample_rate=audio_config.sample_rate,
                is_last=True
            )
            chunks = [chunk]
            context["chunks"] = chunks
            return context
        else:
            # 计算块数
            # 公式: (总样本数 - 重叠样本数 + 步长) // 步长
            overlap_samples = int(processor_config.overlap_ms * audio_config.sample_rate / 1000)
            step_samples = chunk_size - overlap_samples  # 步长 = 块大小 - 重叠大小
            
            # 计算块数
            num_chunks = (total_samples - overlap_samples + step_samples) // step_samples
            
            # 确保至少有一个块
            if num_chunks < 1:
                num_chunks = 1

        # 创建块
        chunks = []
        for i in range(num_chunks):
            # 计算起始位置和结束位置
            start_pos = i * step_size
            end_pos = min(start_pos + chunk_size, total_samples)

            # 创建块
            chunk = AudioChunk(
                data=audio_data[start_pos:end_pos],
                sequence_number=i,
                start_frame=start_pos,
                chunk_size=end_pos - start_pos - (overlap_size if i < num_chunks - 1 else 0),
                overlap_size=overlap_size if i < num_chunks - 1 else 0,
                timestamp_ms=start_pos * 1000.0 / audio_config.sample_rate,
                sample_rate=audio_config.sample_rate,
                is_last=i == num_chunks - 1
            )
            chunks.append(chunk)

        context["chunks"] = chunks
        return context

    async def _process_chunks(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        处理所有块
        
        Args:
            context: 处理上下文
            
        Returns:
            更新后的上下文
            
        Raises:
            RuntimeError: 当处理失败时
        """
        chunks = context["chunks"]
        loop = asyncio.get_running_loop()

        # 使用线程池并行处理
        futures = []
        for chunk in chunks:
            # 将同步处理函数包装为异步任务，使用自定义线程池
            future = loop.run_in_executor(
                None,  # 使用默认执行器
                lambda c=chunk: self.thread_pool.submit(self._process_chunk_sync, c).result()
            )
            futures.append(future)

        # 等待所有任务完成
        context["results"] = await asyncio.gather(*futures)
        return context

    def _process_chunk_sync(self, chunk: AudioChunk) -> ProcessResult:
        """
        同步处理单个块
        
        Args:
            chunk: 音频块
            
        Returns:
            处理结果
            
        Raises:
            Exception: 当处理失败时
        """
        try:
            # 调用子类实现的处理方法
            result_data = self.process_chunk(chunk)

            # 创建处理结果
            result = ProcessResult(
                chunk=chunk,
                result_data=result_data,
                success=True,
                processing_time_ms=0.0  # 实际应用中应该计算处理时间
            )

            return result
        except Exception as e:
            # 创建错误结果
            result = ProcessResult(
                chunk=chunk,
                result_data=None,
                success=False,
                error=str(e),
                processing_time_ms=0.0
            )

            return result

    @abc.abstractmethod
    def process_chunk(self, chunk: AudioChunk) -> Any:
        """
        处理单个音频块
        
        Args:
            chunk: 音频块
            
        Returns:
            处理结果数据
            
        Raises:
            Exception: 当处理失败时
        """
        pass

    async def _merge_results(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        合并处理结果
        
        Args:
            context: 处理上下文
            
        Returns:
            更新后的上下文
            
        Raises:
            ValueError: 当合并失败时
        """
        results = context["results"]
        processor_config = context["processor_config"]

        # 按序列号排序
        results.sort(key=lambda r: r.chunk.sequence_number)

        # 根据重叠策略合并结果
        if processor_config.overlap_strategy == OverlapStrategy.FRONT_PRIORITY:
            # 前块优先策略
            # 默认实现，子类可以重写
            pass
        elif processor_config.overlap_strategy == OverlapStrategy.BACK_PRIORITY:
            # 后块优先策略
            # 子类应该重写此方法实现具体逻辑
            pass
        elif processor_config.overlap_strategy == OverlapStrategy.MAX_CONFIDENCE:
            # 最高置信度优先策略
            # 子类应该重写此方法实现具体逻辑
            pass

        context["results"] = results
        return context

    def shutdown(self, wait: bool = True) -> None:
        """
        关闭处理器
        
        Args:
            wait: 是否等待所有任务完成
        """
        self.thread_pool.shutdown(wait=wait)

    def get_stats(self) -> dict[str, Any]:
        """
        获取处理器统计信息
        
        Returns:
            包含统计信息的字典
        """
        return self.thread_pool.get_stats()
