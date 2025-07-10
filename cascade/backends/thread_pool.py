"""
VAD专用线程池模块

本模块提供VAD专用线程池，负责管理一组工作线程，并将音频处理任务分发给它们。
它与VADBackend协同工作，确保每个线程都有一个独立的VAD模型实例，
从而实现无锁的高性能并行推理。
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from cascade.backends.base import VADBackend
from cascade.types.audio import AudioChunk
from cascade.types.vad import VADConfig, VADResult

# 配置日志
logger = logging.getLogger("cascade.vad.thread_pool")


class VADThreadPool:
    """
    VAD专用线程池，负责管理一组工作线程，并将音频处理任务分发给它们。
    
    本类的核心职责是与VADBackend协同工作，确保每个线程都有一个独立的VAD模型实例。
    它使用线程本地存储机制，为每个工作线程创建并缓存一个专属的模型实例，
    从而实现无锁的高性能并行推理。
    """

    def __init__(self, config: VADConfig, backend_template: VADBackend):
        """
        初始化VAD线程池。
        
        Args:
            config: VAD配置对象，包含workers等参数。
            backend_template: VADBackend的"模板"实例。它自身不持有模型，
                             只持有对这个模板的引用。当每个工作线程执行任务时，
                             会通过线程本地存储创建自己的模型实例。
        """
        self.config = config
        self._backend_template = backend_template
        self.executor = ThreadPoolExecutor(
            max_workers=config.workers,
            thread_name_prefix="vad-worker-"
        )
        self._loop = None  # 将在start方法中初始化
        self._is_started = False
        self._is_closed = False
        
        logger.debug(f"VADThreadPool初始化完成，工作线程数: {config.workers}")

    async def start(self):
        """
        启动线程池，预热每个线程的模型实例。
        
        此方法会向线程池提交N个（N=工作线程数）_warmup_thread任务，
        确保每个工作线程都预先创建并缓存了自己的模型实例。
        
        Raises:
            RuntimeError: 当线程池已经启动或已关闭时抛出。
        """
        if self._is_started:
            raise RuntimeError("VADThreadPool已经启动")
            
        if self._is_closed:
            raise RuntimeError("VADThreadPool已关闭，无法重新启动")
            
        logger.info(f"正在启动VADThreadPool，预热{self.config.workers}个工作线程...")
        
        self._loop = asyncio.get_event_loop()
        
        # 为每个工作线程提交一个预热任务
        warmup_futures = [
            self._loop.run_in_executor(self.executor, self._warmup_thread)
            for _ in range(self.config.workers)
        ]
        
        # 等待所有预热任务完成
        try:
            await asyncio.gather(*warmup_futures)
            self._is_started = True
            logger.info("VADThreadPool启动成功，所有工作线程已预热")
        except Exception as e:
            logger.error(f"VADThreadPool启动失败: {e}")
            # 确保关闭线程池
            await self.close()
            raise

    def _warmup_thread(self):
        """
        预热线程的模型实例。
        
        当每个工作线程执行此方法时，会调用backend_template.warmup()，
        这会触发在该线程本地存储中创建并缓存一个模型实例。
        
        Returns:
            当前线程的名称，用于日志记录。
        """
        thread_name = threading.current_thread().name
        logger.debug(f"正在预热线程 {thread_name} 的VAD模型实例")
        
        try:
            self._backend_template.warmup()
            logger.debug(f"线程 {thread_name} 的VAD模型实例预热成功")
            return thread_name
        except Exception as e:
            logger.error(f"线程 {thread_name} 预热VAD模型失败: {e}")
            raise

    async def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        处理单个音频块。
        
        Args:
            chunk: 待处理的音频块。
            
        Returns:
            VAD处理结果。
            
        Raises:
            RuntimeError: 当线程池未启动或已关闭时抛出。
        """
        if not self._is_started:
            raise RuntimeError("必须先调用start方法启动线程池")
            
        if self._is_closed:
            raise RuntimeError("VADThreadPool已关闭，无法处理音频块")
            
        logger.debug(f"提交音频块 #{chunk.sequence_number} 到线程池处理")
        
        result = await self._loop.run_in_executor(
            self.executor,
            self._process_chunk_sync,
            chunk
        )
        
        logger.debug(f"音频块 #{chunk.sequence_number} 处理完成，结果: {result.is_speech}")
        return result

    def _process_chunk_sync(self, chunk: AudioChunk) -> VADResult:
        """
        同步处理音频块。
        
        此方法在工作线程中执行，使用该线程专属的模型实例进行推理。
        
        Args:
            chunk: 待处理的音频块。
            
        Returns:
            VAD处理结果。
        """
        thread_name = threading.current_thread().name
        logger.debug(f"线程 {thread_name} 开始处理音频块 #{chunk.sequence_number}")
        
        try:
            result = self._backend_template.process_chunk(chunk)
            logger.debug(f"线程 {thread_name} 处理音频块 #{chunk.sequence_number} 完成")
            return result
        except Exception as e:
            logger.error(f"线程 {thread_name} 处理音频块 #{chunk.sequence_number} 失败: {e}")
            raise

    async def close(self):
        """
        关闭线程池，释放所有资源。
        
        此方法会关闭线程池执行器，并调用后端模板的close方法。
        """
        if self._is_closed:
            logger.debug("VADThreadPool已经关闭")
            return
            
        logger.info("正在关闭VADThreadPool...")
        
        # 关闭线程池执行器
        self.executor.shutdown(wait=True)
        
        # 关闭后端模板
        try:
            self._backend_template.close()
        except Exception as e:
            logger.error(f"关闭VAD后端模板失败: {e}")
        
        self._is_closed = True
        self._is_started = False
        logger.info("VADThreadPool已关闭")