"""
流式处理器核心模块

实现异步流式VAD处理编排器，管理多个Cascade实例。
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import weakref
from collections import defaultdict

from .types import Config, CascadeResult, ProcessorStats
from .instance import CascadeInstance


logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    流式处理器
    
    核心功能：
    - 管理多个CascadeInstance实例
    - 提供异步流式处理接口
    - 负载均衡和实例调度
    - 统计信息收集
    """
    
    def __init__(self, config: Config):
        """
        初始化流式处理器
        
        Args:
            config: 处理器配置
        """
        self.config = config
        self.instances: Dict[str, CascadeInstance] = {}
        self.instance_lock = threading.RLock()  # 使用可重入锁
        self.is_running = False
        
        # 线程池管理
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.max_instances,
            thread_name_prefix="cascade-worker"
        )
        
        # 实例负载统计
        self.instance_load: Dict[str, int] = defaultdict(int)
        self.instance_last_used: Dict[str, float] = {}
        
        # 统计信息
        self.total_chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.speech_segments_count = 0
        self.single_frames_count = 0
        self.error_count = 0
        
        # 性能监控
        self.processing_times = []  # 最近100次处理时间
        self.max_processing_times = 100
        
        logger.info(f"StreamProcessor 初始化，最大实例数: {config.max_instances}")
    
    async def start(self) -> None:
        """启动处理器"""
        if self.is_running:
            logger.warning("StreamProcessor 已在运行")
            return
        
        self.is_running = True
        logger.info("StreamProcessor 启动")
    
    async def stop(self) -> None:
        """停止处理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止所有实例
        with self.instance_lock:
            for instance in self.instances.values():
                try:
                    instance.stop()
                except Exception as e:
                    logger.warning(f"停止实例时出错: {e}")
            self.instances.clear()
            self.instance_load.clear()
            self.instance_last_used.clear()
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        logger.info("StreamProcessor 停止")
    
    async def process_stream(
        self, 
        audio_stream: AsyncIterator[bytes],
        stream_id: Optional[str] = None
    ) -> AsyncIterator[CascadeResult]:
        """
        处理音频流
        
        Args:
            audio_stream: 音频数据流
            stream_id: 流标识符，用于实例管理
            
        Yields:
            处理结果
        """
        if not self.is_running:
            raise RuntimeError("StreamProcessor 未启动")
        
        # 获取或创建处理实例
        instance = await self._get_or_create_instance(stream_id)
        
        try:
            # 处理音频流并同时收集结果
            async for audio_chunk in audio_stream:
                if not self.is_running:
                    break
                
                # 提交音频块到实例
                self._update_instance_load(instance.instance_id, 1)
                instance.process_audio_chunk(audio_chunk)
                self.total_chunks_processed += 1
                
                # 收集当前可用的结果
                for result in instance.get_results():
                    # 更新统计
                    if result.is_speech_segment:
                        self.speech_segments_count += 1
                    else:
                        self.single_frames_count += 1
                    self.total_processing_time_ms += result.processing_time_ms
                    
                    yield result
                
                self._update_instance_load(instance.instance_id, -1)
            
            # 等待处理完成并收集剩余结果
            await self._wait_for_completion(instance)
            
            # 收集剩余结果
            async for result in self._collect_remaining_results(instance):
                yield result
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"处理音频流失败: {e}")
            raise
        finally:
            # 清理实例
            if stream_id:
                await self._cleanup_instance(stream_id)
    
    async def process_chunk(self, audio_data: bytes) -> List[CascadeResult]:
        """
        处理单个音频块
        
        Args:
            audio_data: 音频数据
            
        Returns:
            处理结果列表
        """
        if not self.is_running:
            raise RuntimeError("StreamProcessor 未启动")
        
        # 获取可用实例
        instance = await self._get_available_instance()
        
        try:
            # 处理音频块
            start_time = time.time()
            self._update_instance_load(instance.instance_id, 1)
            
            instance.process_audio_chunk(audio_data)
            
            # 智能等待：根据历史处理时间调整等待时间
            wait_time = self._calculate_optimal_wait_time()
            await asyncio.sleep(wait_time)
            
            # 收集结果
            results = list(instance.get_results())
            
            # 记录处理时间
            processing_time = (time.time() - start_time) * 1000
            self._record_processing_time(processing_time)
            self._update_instance_load(instance.instance_id, -1)
            
            # 更新统计
            self.total_chunks_processed += 1
            for result in results:
                if result.is_speech_segment:
                    self.speech_segments_count += 1
                else:
                    self.single_frames_count += 1
                self.total_processing_time_ms += result.processing_time_ms
            
            return results
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"处理音频块失败: {e}")
            raise
    
    async def _get_or_create_instance(self, stream_id: Optional[str]) -> CascadeInstance:
        """获取或创建处理实例"""
        if stream_id is None:
            return await self._get_available_instance()
        
        with self.instance_lock:
            if stream_id in self.instances:
                # 更新使用时间
                self.instance_last_used[stream_id] = time.time()
                return self.instances[stream_id]
            
            # 检查实例数量限制
            if len(self.instances) >= self.config.max_instances:
                # 智能选择实例：优先选择负载最低的
                best_instance = self._select_best_instance()
                if best_instance:
                    return best_instance
                
                # 如果没有合适的实例，清理最久未使用的实例
                await self._cleanup_oldest_instance()
            
            # 创建新实例
            try:
                instance = await self._create_instance(stream_id)
                self.instances[stream_id] = instance
                self.instance_last_used[stream_id] = time.time()
                
                logger.info(f"创建新实例: {stream_id}")
                return instance
            except Exception as e:
                logger.error(f"创建实例失败: {e}")
                # 降级到可用实例
                return await self._get_available_instance()
    
    async def _get_available_instance(self) -> CascadeInstance:
        """获取可用实例"""
        with self.instance_lock:
            if not self.instances:
                # 创建默认实例
                instance_id = f"default-{int(time.time())}"
                try:
                    instance = await self._create_instance(instance_id)
                    self.instances[instance_id] = instance
                    self.instance_last_used[instance_id] = time.time()
                    return instance
                except Exception as e:
                    logger.error(f"创建默认实例失败: {e}")
                    raise RuntimeError("无法创建处理实例") from e
            
            # 智能选择最佳实例
            best_instance = self._select_best_instance()
            if best_instance:
                return best_instance
            
            # 如果所有实例都很忙，返回负载最低的
            return min(
                self.instances.values(),
                key=lambda x: self.instance_load.get(x.instance_id, 0)
            )
    
    async def _create_instance(self, instance_id: str) -> CascadeInstance:
        """创建新的处理实例"""
        instance = CascadeInstance(
            instance_id=instance_id,
            config=self.config
        )
        # 异步初始化VAD后端
        await instance.vad_backend.initialize()
        instance.start()
        return instance
    
    def _select_best_instance(self) -> Optional[CascadeInstance]:
        """选择最佳实例（负载最低且不繁忙）"""
        if not self.instances:
            return None
        
        # 过滤出不繁忙的实例
        available_instances = [
            instance for instance in self.instances.values()
            if not instance.is_busy
        ]
        
        if not available_instances:
            return None
        
        # 选择负载最低的实例
        return min(
            available_instances,
            key=lambda x: self.instance_load.get(x.instance_id, 0)
        )
    
    async def _cleanup_oldest_instance(self) -> None:
        """清理最久未使用的实例"""
        if not self.instance_last_used:
            return
        
        # 找到最久未使用的实例
        oldest_id = min(
            self.instance_last_used.keys(),
            key=lambda x: self.instance_last_used[x]
        )
        
        # 清理实例
        if oldest_id in self.instances:
            instance = self.instances.pop(oldest_id)
            instance.stop()
            self.instance_load.pop(oldest_id, None)
            self.instance_last_used.pop(oldest_id, None)
            logger.info(f"清理最久未使用的实例: {oldest_id}")
    
    async def _wait_for_completion(self, instance: CascadeInstance) -> None:
        """等待实例处理完成"""
        max_wait_time = 0.5  # 最大等待500ms
        check_interval = 0.01  # 每10ms检查一次
        waited = 0.0
        
        while waited < max_wait_time and instance.is_busy:
            await asyncio.sleep(check_interval)
            waited += check_interval
    
    def _calculate_optimal_wait_time(self) -> float:
        """根据历史处理时间计算最优等待时间"""
        if not self.processing_times:
            return 0.05  # 默认50ms
        
        # 计算平均处理时间
        avg_time = sum(self.processing_times) / len(self.processing_times)
        
        # 等待时间为平均处理时间的一半，但不超过100ms，不少于10ms
        wait_time = max(0.01, min(0.1, avg_time / 2000))  # 转换为秒
        return wait_time
    
    def _record_processing_time(self, processing_time_ms: float) -> None:
        """记录处理时间"""
        self.processing_times.append(processing_time_ms)
        
        # 保持最近的处理时间记录
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)
    
    def _update_instance_load(self, instance_id: str, delta: int) -> None:
        """更新实例负载"""
        with self.instance_lock:
            current_load = self.instance_load.get(instance_id, 0)
            new_load = max(0, current_load + delta)
            self.instance_load[instance_id] = new_load
    
    async def _collect_results(self, instance: CascadeInstance) -> AsyncIterator[CascadeResult]:
        """收集实例结果"""
        try:
            async for result in instance.get_results_async():
                # 更新统计
                if result.is_speech_segment:
                    self.speech_segments_count += 1
                else:
                    self.single_frames_count += 1
                self.total_processing_time_ms += result.processing_time_ms
                
                yield result
        except asyncio.CancelledError:
            pass
    
    async def _collect_remaining_results(self, instance: CascadeInstance) -> AsyncIterator[CascadeResult]:
        """收集剩余结果"""
        for result in instance.get_results():
            # 更新统计
            if result.is_speech_segment:
                self.speech_segments_count += 1
            else:
                self.single_frames_count += 1
            self.total_processing_time_ms += result.processing_time_ms
            
            yield result
    
    async def _cleanup_instance(self, stream_id: str) -> None:
        """清理实例"""
        with self.instance_lock:
            if stream_id in self.instances:
                instance = self.instances.pop(stream_id)
                instance.stop()
                logger.info(f"清理实例: {stream_id}")
    
    def get_stats(self) -> ProcessorStats:
        """获取处理器统计信息"""
        # 计算平均处理时间
        avg_processing_time = 0.0
        if self.total_chunks_processed > 0:
            avg_processing_time = self.total_processing_time_ms / self.total_chunks_processed
        
        # 计算语音比例
        total_results = self.speech_segments_count + self.single_frames_count
        speech_ratio = 0.0
        if total_results > 0:
            speech_ratio = self.speech_segments_count / total_results
        
        # 计算吞吐量
        throughput = 0.0
        if self.total_processing_time_ms > 0:
            throughput = self.total_chunks_processed / (self.total_processing_time_ms / 1000.0)
        
        # 计算错误率
        error_rate = 0.0
        if self.total_chunks_processed > 0:
            error_rate = self.error_count / self.total_chunks_processed
        
        # 估算内存使用
        memory_usage_mb = len(self.instances) * 50.0  # 粗略估算每个实例50MB
        
        return ProcessorStats(
            total_chunks_processed=self.total_chunks_processed,
            total_processing_time_ms=self.total_processing_time_ms,
            average_processing_time_ms=avg_processing_time,
            speech_segments=self.speech_segments_count,
            single_frames=self.single_frames_count,
            speech_ratio=speech_ratio,
            throughput_chunks_per_second=throughput,
            memory_usage_mb=memory_usage_mb,
            error_count=self.error_count,
            error_rate=error_rate
        )
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.total_chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.speech_segments_count = 0
        self.single_frames_count = 0
        self.error_count = 0
        logger.info("统计信息已重置")
    
    @property
    def active_instances(self) -> int:
        """活跃实例数"""
        return len(self.instances)
    
    @property
    def is_busy(self) -> bool:
        """是否繁忙"""
        with self.instance_lock:
            return any(instance.is_busy for instance in self.instances.values())
    
    def __str__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return f"StreamProcessor({status}, instances={self.active_instances})"
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()