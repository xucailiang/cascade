"""
VAD结果合并器模块

本模块提供VAD结果合并功能，负责将多个VAD结果合并为连续的语音片段。
它实现了平滑处理、去抖动和最小间隔合并等功能，以提高VAD的准确性和稳定性。
"""

import logging
from typing import List, Optional, Tuple

from cascade.types.vad import VADResult, VADSegment, MergerConfig

# 配置日志
logger = logging.getLogger("cascade.vad.merger")


class ResultMerger:
    """
    VAD结果合并器，负责将多个VAD结果合并为连续的语音片段。
    
    本类实现了以下功能：
    1. 平滑处理：通过滑动窗口平均等方法，减少VAD结果的波动
    2. 去抖动：过滤掉过短的语音/非语音片段
    3. 最小间隔合并：合并间隔过短的语音片段
    """

    def __init__(self, config: MergerConfig):
        """
        初始化VAD结果合并器。
        
        Args:
            config: 合并器配置对象。
        """
        self.config = config
        self._buffer: List[VADResult] = []
        self._segments: List[VADSegment] = []
        self._current_segment: Optional[VADSegment] = None
        
        # 状态标志
        self._is_in_speech = False
        self._last_end_time = 0
        
        logger.debug(f"ResultMerger初始化完成，配置: {config}")

    def add_result(self, result: VADResult) -> None:
        """
        添加单个VAD结果到合并器。
        
        Args:
            result: VAD结果对象。
        """
        logger.debug(f"添加VAD结果: {result}")
        
        # 添加到缓冲区
        self._buffer.append(result)
        
        # 如果缓冲区长度超过窗口大小，则处理最早的结果
        if len(self._buffer) > self.config.window_size:
            self._process_oldest_result()

    def _process_oldest_result(self) -> None:
        """
        处理缓冲区中最早的VAD结果。
        
        此方法实现了平滑处理算法，通过考虑窗口内的多个结果，
        减少VAD结果的波动，提高稳定性。
        """
        if not self._buffer:
            return
            
        # 获取最早的结果
        result = self._buffer.pop(0)
        
        # 应用平滑处理
        smoothed_result = self._apply_smoothing(result)
        
        # 更新语音片段
        self._update_segments(smoothed_result)

    def _apply_smoothing(self, result: VADResult) -> VADResult:
        """
        应用平滑处理算法。
        
        Args:
            result: 原始VAD结果。
            
        Returns:
            平滑处理后的VAD结果。
        """
        # 如果缓冲区为空或窗口大小为1，则不进行平滑处理
        if len(self._buffer) == 0 or self.config.window_size <= 1:
            return result
            
        # 计算窗口内的平均概率
        total_prob = result.probability
        count = 1
        
        for buffered_result in self._buffer:
            total_prob += buffered_result.probability
            count += 1
            
        avg_prob = total_prob / count
        
        # 创建平滑处理后的结果
        smoothed_result = VADResult(
            is_speech=(avg_prob > self.config.threshold),
            probability=avg_prob,
            start_ms=result.start_ms,
            end_ms=result.end_ms,
            chunk_id=result.chunk_id,
            confidence=avg_prob,
        )
        
        logger.debug(f"平滑处理: 原始={result.is_speech}(p={result.probability:.2f}), "
                    f"平滑后={smoothed_result.is_speech}(p={smoothed_result.probability:.2f})")
                    
        return smoothed_result

    def _update_segments(self, result: VADResult) -> None:
        """
        根据VAD结果更新语音片段。
        
        此方法实现了去抖动和最小间隔合并功能，
        过滤掉过短的语音/非语音片段，合并间隔过短的语音片段。
        
        Args:
            result: VAD结果对象。
        """
        # 检查是否需要开始新的语音片段
        if result.is_speech and not self._is_in_speech:
            # 如果当前没有活跃的语音片段，则创建新的语音片段
            self._start_new_segment(result)
            
        # 检查是否需要结束当前语音片段
        elif not result.is_speech and self._is_in_speech:
            # 如果当前有活跃的语音片段，则尝试结束它
            self._try_end_current_segment(result)
            
        # 如果仍在语音片段中，则更新当前片段的结束时间和其他属性
        elif self._is_in_speech and self._current_segment is not None:
            self._current_segment.end_ms = result.end_ms
            self._current_segment.chunk_count += 1
            
            # 更新峰值概率
            if result.probability > self._current_segment.peak_probability:
                self._current_segment.peak_probability = result.probability
                
            # 更新能量统计
            if result.energy_level is not None:
                if self._current_segment.energy_stats is None:
                    self._current_segment.energy_stats = {"avg": result.energy_level, "max": result.energy_level}
                else:
                    # 更新平均能量
                    current_avg = self._current_segment.energy_stats.get("avg", 0)
                    count = self._current_segment.chunk_count - 1  # 减1是因为当前chunk已经加过了
                    new_avg = (current_avg * count + result.energy_level) / self._current_segment.chunk_count
                    self._current_segment.energy_stats["avg"] = new_avg
                    
                    # 更新最大能量
                    current_max = self._current_segment.energy_stats.get("max", 0)
                    if result.energy_level > current_max:
                        self._current_segment.energy_stats["max"] = result.energy_level
            
        # 更新最后处理的时间
        self._last_end_time = result.end_ms

    def _start_new_segment(self, result: VADResult) -> None:
        """
        开始新的语音片段。
        
        Args:
            result: 触发新语音片段的VAD结果。
        """
        # 检查是否需要与前一个片段合并
        if (self._segments and
            result.start_ms - self._segments[-1].end_ms < self.config.min_silence_duration_ms):
            # 如果与前一个片段的间隔小于最小静音持续时间，则合并
            logger.debug(f"合并相邻片段: 间隔={result.start_ms - self._segments[-1].end_ms}ms < "
                        f"{self.config.min_silence_duration_ms}ms")
            self._current_segment = self._segments.pop()
            self._current_segment.end_ms = result.end_ms
            self._current_segment.chunk_count += 1
            
            # 更新峰值概率
            if result.probability > self._current_segment.peak_probability:
                self._current_segment.peak_probability = result.probability
                
            # 更新能量统计
            if result.energy_level is not None:
                if self._current_segment.energy_stats is None:
                    self._current_segment.energy_stats = {"avg": result.energy_level, "max": result.energy_level}
                else:
                    # 更新平均能量
                    current_avg = self._current_segment.energy_stats.get("avg", 0)
                    count = self._current_segment.chunk_count - 1  # 减1是因为当前chunk已经加过了
                    new_avg = (current_avg * count + result.energy_level) / self._current_segment.chunk_count
                    self._current_segment.energy_stats["avg"] = new_avg
                    
                    # 更新最大能量
                    current_max = self._current_segment.energy_stats.get("max", 0)
                    if result.energy_level > current_max:
                        self._current_segment.energy_stats["max"] = result.energy_level
        else:
            # 创建新的语音片段
            self._current_segment = VADSegment(
                start_ms=result.start_ms,
                end_ms=result.end_ms,
                confidence=result.confidence,
                peak_probability=result.probability,
                chunk_count=1,
                energy_stats={"avg": result.energy_level} if result.energy_level is not None else None,
            )
            
        self._is_in_speech = True
        logger.debug(f"开始新语音片段: {self._current_segment}")

    def _try_end_current_segment(self, result: VADResult) -> None:
        """
        尝试结束当前语音片段。
        
        此方法实现了去抖动功能，过滤掉过短的语音片段。
        
        Args:
            result: 触发语音片段结束的VAD结果。
        """
        if self._current_segment is None:
            self._is_in_speech = False
            return
            
        # 计算当前片段的持续时间
        duration = self._current_segment.end_ms - self._current_segment.start_ms
        
        # 检查是否满足最小语音持续时间
        if duration >= self.config.min_speech_duration_ms:
            # 如果持续时间足够长，则保存当前片段
            self._segments.append(self._current_segment)
            logger.debug(f"结束语音片段: {self._current_segment}, 持续时间={duration}ms")
        else:
            # 如果持续时间过短，则丢弃当前片段
            logger.debug(f"丢弃过短语音片段: {self._current_segment}, 持续时间={duration}ms < "
                        f"{self.config.min_speech_duration_ms}ms")
                        
        self._current_segment = None
        self._is_in_speech = False

    def get_segments(self) -> List[VADSegment]:
        """
        获取当前的语音片段列表。
        
        Returns:
            语音片段列表。
        """
        # 返回已完成的片段列表的副本
        return self._segments.copy()

    def flush(self) -> List[VADSegment]:
        """
        刷新缓冲区，处理所有剩余的VAD结果。
        
        Returns:
            最终的语音片段列表。
        """
        logger.debug(f"刷新合并器，处理剩余的{len(self._buffer)}个VAD结果")
        
        # 处理缓冲区中的所有结果
        while self._buffer:
            self._process_oldest_result()
            
        # 如果当前有活跃的语音片段，则结束它
        if self._is_in_speech and self._current_segment is not None:
            # 计算当前片段的持续时间
            duration = self._current_segment.end_ms - self._current_segment.start_ms
            
            # 检查是否满足最小语音持续时间
            if duration >= self.config.min_speech_duration_ms:
                # 如果持续时间足够长，则保存当前片段
                self._segments.append(self._current_segment)
                logger.debug(f"结束最后的语音片段: {self._current_segment}, 持续时间={duration}ms")
            else:
                # 如果持续时间过短，则丢弃当前片段
                logger.debug(f"丢弃过短的最后语音片段: {self._current_segment}, 持续时间={duration}ms < "
                            f"{self.config.min_speech_duration_ms}ms")
                            
            self._current_segment = None
            self._is_in_speech = False
            
        return self._segments.copy()

    def reset(self) -> None:
        """重置合并器状态。"""
        logger.debug("重置合并器状态")
        self._buffer = []
        self._segments = []
        self._current_segment = None
        self._is_in_speech = False
        self._last_end_time = 0