"""
音频处理管道模块

该模块提供了音频处理管道的实现，允许将多个音频处理器连接在一起，形成处理流水线。
处理管道支持串行和并行处理模式，以及结果的聚合和转换。
"""

import asyncio
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, Field

from cascade.processor.base import AudioProcessor, ProcessResult
from cascade.types.audio import AudioConfig

# 类型变量定义
T = TypeVar('T')
R = TypeVar('R')


class PipelineMode(str, Enum):
    """处理管道模式"""
    SEQUENTIAL = "sequential"  # 串行处理模式
    PARALLEL = "parallel"      # 并行处理模式


class PipelineConfig(BaseModel):
    """处理管道配置"""
    mode: PipelineMode = Field(
        default=PipelineMode.SEQUENTIAL,
        description="处理管道模式，可选串行或并行"
    )
    name: str = Field(
        default="audio_pipeline",
        description="处理管道名称"
    )
    continue_on_error: bool = Field(
        default=False,
        description="是否在处理器出错时继续执行"
    )
    timeout_seconds: float | None = Field(
        default=None,
        description="处理超时时间（秒），None表示无超时"
    )


class ProcessorNode(BaseModel):
    """处理器节点配置"""
    processor: AudioProcessor = Field(
        description="音频处理器实例"
    )
    name: str = Field(
        description="处理器节点名称"
    )
    enabled: bool = Field(
        default=True,
        description="是否启用该处理器"
    )
    input_transform: Callable | None = Field(
        default=None,
        description="输入转换函数，用于在处理前转换输入数据"
    )
    output_transform: Callable[[list[ProcessResult]], list[ProcessResult]] | None = Field(
        default=None,
        description="输出转换函数，用于在处理后转换输出结果"
    )

    class Config:
        arbitrary_types_allowed = True


class PipelineResult(ProcessResult, Generic[T]):
    """管道处理结果"""
    pipeline_name: str = Field(
        description="处理管道名称"
    )
    processor_results: dict[str, list[ProcessResult]] = Field(
        default_factory=dict,
        description="各处理器的处理结果"
    )
    aggregated_result: T | None = Field(
        default=None,
        description="聚合后的结果"
    )


class AudioPipeline(AudioProcessor):
    """
    音频处理管道
    
    将多个音频处理器连接在一起，形成处理流水线。支持串行和并行处理模式，
    以及结果的聚合和转换。
    """

    def __init__(
        self,
        config: PipelineConfig = PipelineConfig(),
        processors: list[ProcessorNode] | None = None,
        result_aggregator: Callable[[dict[str, list[ProcessResult]]], T] | None = None
    ):
        """
        初始化音频处理管道
        
        Args:
            config: 处理管道配置
            processors: 处理器节点列表
            result_aggregator: 结果聚合函数，用于将各处理器的结果聚合为单一结果
        """
        super().__init__()
        self.config = config
        self.processors = processors or []
        self.result_aggregator = result_aggregator
        self.logger = logging.getLogger(f"cascade.pipeline.{config.name}")

    def add_processor(
        self,
        processor: AudioProcessor,
        name: str | None = None,
        enabled: bool = True,
        input_transform: Callable | None = None,
        output_transform: Callable | None = None
    ) -> "AudioPipeline":
        """
        添加处理器到管道
        
        Args:
            processor: 音频处理器实例
            name: 处理器节点名称，如果为None则使用处理器类名
            enabled: 是否启用该处理器
            input_transform: 输入转换函数
            output_transform: 输出转换函数
            
        Returns:
            处理管道实例（用于链式调用）
        """
        if name is None:
            name = processor.__class__.__name__

        # 确保名称唯一
        existing_names = {p.name for p in self.processors}
        if name in existing_names:
            i = 1
            while f"{name}_{i}" in existing_names:
                i += 1
            name = f"{name}_{i}"

        node = ProcessorNode(
            processor=processor,
            name=name,
            enabled=enabled,
            input_transform=input_transform,
            output_transform=output_transform
        )

        self.processors.append(node)
        return self

    def remove_processor(self, name: str) -> bool:
        """
        从管道中移除处理器
        
        Args:
            name: 处理器节点名称
            
        Returns:
            是否成功移除
        """
        for i, node in enumerate(self.processors):
            if node.name == name:
                self.processors.pop(i)
                return True
        return False

    def get_processor(self, name: str) -> ProcessorNode | None:
        """
        获取处理器节点
        
        Args:
            name: 处理器节点名称
            
        Returns:
            处理器节点，如果不存在则返回None
        """
        for node in self.processors:
            if node.name == name:
                return node
        return None

    def enable_processor(self, name: str, enabled: bool = True) -> bool:
        """
        启用或禁用处理器
        
        Args:
            name: 处理器节点名称
            enabled: 是否启用
            
        Returns:
            是否成功设置
        """
        node = self.get_processor(name)
        if node:
            node.enabled = enabled
            return True
        return False

    async def process_audio(
        self,
        audio_data: np.ndarray,
        audio_config: AudioConfig,
        context: dict[str, Any] | None = None
    ) -> list[PipelineResult]:
        """
        处理音频数据
        
        Args:
            audio_data: 音频数据
            audio_config: 音频配置
            context: 处理上下文
            
        Returns:
            处理结果列表
        """
        if context is None:
            context = {}

        # 记录开始处理
        self.logger.debug(f"开始处理管道 {self.config.name}，模式: {self.config.mode.value}")

        # 获取启用的处理器
        enabled_processors = [p for p in self.processors if p.enabled]
        if not enabled_processors:
            self.logger.warning(f"管道 {self.config.name} 中没有启用的处理器")
            return []

        # 根据模式选择处理方法
        if self.config.mode == PipelineMode.SEQUENTIAL:
            processor_results = await self._process_sequential(audio_data, audio_config, context, enabled_processors)
        else:  # PARALLEL
            processor_results = await self._process_parallel(audio_data, audio_config, context, enabled_processors)

        # 聚合结果
        aggregated_result = None
        if self.result_aggregator and processor_results:
            try:
                aggregated_result = self.result_aggregator(processor_results)
            except Exception as e:
                self.logger.error(f"结果聚合失败: {str(e)}")

        # 创建管道结果
        # 从 base.py 中导入 AudioChunk 类
        from cascade.processor.base import AudioChunk

        # 创建一个虚拟的音频块，用于管道结果
        dummy_chunk = AudioChunk(
            data=np.array([]),
            sequence_number=0,
            start_frame=0,
            chunk_size=1,
            timestamp_ms=0.0,
            sample_rate=audio_config.sample_rate
        )

        pipeline_result = PipelineResult(
            pipeline_name=self.config.name,
            processor_results=processor_results,
            aggregated_result=aggregated_result,
            chunk=dummy_chunk,
            result_data=aggregated_result
        )

        return [pipeline_result]

    async def _process_sequential(
        self,
        audio_data: np.ndarray,
        audio_config: AudioConfig,
        context: dict[str, Any],
        processors: list[ProcessorNode]
    ) -> dict[str, list[ProcessResult]]:
        """
        串行处理音频数据
        
        Args:
            audio_data: 音频数据
            audio_config: 音频配置
            context: 处理上下文
            processors: 处理器节点列表
            
        Returns:
            各处理器的处理结果
        """
        results: dict[str, list[ProcessResult]] = {}
        current_audio = audio_data
        current_config = audio_config
        current_context = context.copy()

        for node in processors:
            try:
                # 应用输入转换
                if node.input_transform:
                    try:
                        # 调用输入转换函数
                        transform_args = (current_audio, current_config, current_context)
                        transform_result = node.input_transform(*transform_args[:node.input_transform.__code__.co_argcount])

                        # 解包结果
                        if isinstance(transform_result, tuple):
                            if len(transform_result) >= 1:
                                current_audio = transform_result[0]
                            if len(transform_result) >= 2:
                                current_config = transform_result[1]
                            if len(transform_result) >= 3:
                                current_context = transform_result[2]
                        else:
                            # 如果只返回一个值，假设是音频数据
                            current_audio = transform_result
                    except Exception as e:
                        self.logger.error(f"处理器 {node.name} 输入转换失败: {str(e)}")
                        if not self.config.continue_on_error:
                            raise

                # 处理音频
                processor_task = node.processor.process_audio(
                    current_audio, current_config
                )

                # 添加超时
                if self.config.timeout_seconds:
                    processor_results = await asyncio.wait_for(
                        processor_task,
                        timeout=self.config.timeout_seconds
                    )
                else:
                    processor_results = await processor_task

                # 应用输出转换
                if node.output_transform:
                    try:
                        processor_results = node.output_transform(processor_results)
                    except Exception as e:
                        self.logger.error(f"处理器 {node.name} 输出转换失败: {str(e)}")
                        if not self.config.continue_on_error:
                            raise

                # 保存结果
                results[node.name] = processor_results

            except Exception as e:
                self.logger.error(f"处理器 {node.name} 处理失败: {str(e)}")
                if not self.config.continue_on_error:
                    break

        return results

    async def _process_parallel(
        self,
        audio_data: np.ndarray,
        audio_config: AudioConfig,
        context: dict[str, Any],
        processors: list[ProcessorNode]
    ) -> dict[str, list[ProcessResult]]:
        """
        并行处理音频数据
        
        Args:
            audio_data: 音频数据
            audio_config: 音频配置
            context: 处理上下文
            processors: 处理器节点列表
            
        Returns:
            各处理器的处理结果
        """
        results: dict[str, list[ProcessResult]] = {}
        tasks = []

        # 创建所有处理任务
        for node in processors:
            # 为每个处理器创建独立的上下文副本
            node_context = context.copy()

            # 准备处理函数
            async def process_node(n=node, ctx=node_context):
                try:
                    current_audio = audio_data
                    current_config = audio_config
                    current_context = ctx

                    # 应用输入转换
                    if n.input_transform:
                        try:
                            # 调用输入转换函数
                            transform_args = (current_audio, current_config, current_context)
                            transform_result = n.input_transform(*transform_args[:n.input_transform.__code__.co_argcount])

                            # 解包结果
                            if isinstance(transform_result, tuple):
                                if len(transform_result) >= 1:
                                    current_audio = transform_result[0]
                                if len(transform_result) >= 2:
                                    current_config = transform_result[1]
                                if len(transform_result) >= 3:
                                    current_context = transform_result[2]
                            else:
                                # 如果只返回一个值，假设是音频数据
                                current_audio = transform_result
                        except Exception as e:
                            self.logger.error(f"处理器 {n.name} 输入转换失败: {str(e)}")
                            if not self.config.continue_on_error:
                                raise
                            return n.name, []

                    # 处理音频
                    processor_results = await n.processor.process_audio(
                        current_audio, current_config
                    )

                    # 应用输出转换
                    if n.output_transform:
                        try:
                            processor_results = n.output_transform(processor_results)
                        except Exception as e:
                            self.logger.error(f"处理器 {n.name} 输出转换失败: {str(e)}")
                            if not self.config.continue_on_error:
                                raise

                    return n.name, processor_results

                except Exception as e:
                    self.logger.error(f"处理器 {n.name} 处理失败: {str(e)}")
                    if self.config.continue_on_error:
                        return n.name, []
                    raise

            # 添加到任务列表
            tasks.append(process_node())

        # 等待所有任务完成或超时
        if tasks:
            if self.config.timeout_seconds:
                completed_tasks, _ = await asyncio.wait(
                    tasks,
                    timeout=self.config.timeout_seconds,
                    return_when=asyncio.ALL_COMPLETED
                )

                # 处理完成的任务
                for task in completed_tasks:
                    try:
                        name, processor_results = await task
                        results[name] = processor_results
                    except Exception:
                        # 已在处理函数中记录错误
                        pass
            else:
                # 无超时限制，等待所有任务完成
                completed_results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理结果
                for result in completed_results:
                    if isinstance(result, Exception):
                        # 已在处理函数中记录错误
                        continue

                    if isinstance(result, tuple) and len(result) == 2:
                        name, processor_results = result
                        results[name] = processor_results

        return results


class SequentialPipeline(AudioPipeline):
    """串行处理管道"""

    def __init__(
        self,
        name: str = "sequential_pipeline",
        continue_on_error: bool = False,
        timeout_seconds: float | None = None,
        processors: list[ProcessorNode] | None = None,
        result_aggregator: Callable | None = None
    ):
        """
        初始化串行处理管道
        
        Args:
            name: 管道名称
            continue_on_error: 是否在处理器出错时继续执行
            timeout_seconds: 处理超时时间（秒）
            processors: 处理器节点列表
            result_aggregator: 结果聚合函数
        """
        config = PipelineConfig(
            mode=PipelineMode.SEQUENTIAL,
            name=name,
            continue_on_error=continue_on_error,
            timeout_seconds=timeout_seconds
        )
        super().__init__(config, processors, result_aggregator)


class ParallelPipeline(AudioPipeline):
    """并行处理管道"""

    def __init__(
        self,
        name: str = "parallel_pipeline",
        continue_on_error: bool = False,
        timeout_seconds: float | None = None,
        processors: list[ProcessorNode] | None = None,
        result_aggregator: Callable | None = None
    ):
        """
        初始化并行处理管道
        
        Args:
            name: 管道名称
            continue_on_error: 是否在处理器出错时继续执行
            timeout_seconds: 处理超时时间（秒）
            processors: 处理器节点列表
            result_aggregator: 结果聚合函数
        """
        config = PipelineConfig(
            mode=PipelineMode.PARALLEL,
            name=name,
            continue_on_error=continue_on_error,
            timeout_seconds=timeout_seconds
        )
        super().__init__(config, processors, result_aggregator)
