"""
VAD延迟补偿器

实现简化的VAD延迟补偿功能，用于解决VAD模型固有的语音开始检测延迟问题。

设计原则：
- 极简实现：核心逻辑<50行
- 零破坏：不影响现有流程
- 高性能：仅进行时间戳计算
- 流式友好：完全兼容实时处理
"""


from ..types import VADResult


class SimpleDelayCompensator:
    """
    简化的VAD延迟补偿器
    
    通过检测语音开始事件，自动调整时间戳来补偿VAD模型的固有延迟。
    使用简单的阈值交叉检测方法，性能优异且可靠。
    
    补偿原理：
    1. 检测语音开始（从静音到语音的状态转换）
    2. 将开始时间戳向前调整指定的补偿时长
    3. 保留原始时间戳用于调试和分析
    """

    def __init__(self, compensation_ms: int):
        """
        初始化延迟补偿器
        
        Args:
            compensation_ms: 延迟补偿时长（毫秒），0表示禁用补偿
        """
        self.compensation_ms = compensation_ms
        self.enabled = compensation_ms > 0
        self.previous_is_speech = False

    def process_result(self, vad_result: VADResult) -> VADResult:
        """
        处理VAD结果，应用延迟补偿
        
        Args:
            vad_result: 原始VAD检测结果
            
        Returns:
            VADResult: 补偿后的VAD结果
            
        Note:
            如果未启用补偿或不是语音开始，返回原始结果
        """
        if not self.enabled:
            return vad_result

        # 检测语音开始（简单阈值交叉）
        speech_started = (not self.previous_is_speech and
                         vad_result.is_speech)

        # 更新状态
        self.previous_is_speech = vad_result.is_speech

        if speech_started:
            # 语音开始，应用延迟补偿
            compensated_result = vad_result.model_copy()
            compensated_result.original_start_ms = vad_result.start_ms
            compensated_result.start_ms = max(0.0,
                vad_result.start_ms - self.compensation_ms)
            compensated_result.is_compensated = True

            return compensated_result

        # 非语音开始，返回原始结果
        return vad_result

    def reset(self) -> None:
        """
        重置补偿器状态
        
        在处理新的音频流时调用，确保状态清洁。
        """
        self.previous_is_speech = False

    def is_enabled(self) -> bool:
        """检查是否启用延迟补偿"""
        return self.enabled

    def get_compensation_ms(self) -> int:
        """获取当前补偿时长"""
        return self.compensation_ms

    def set_compensation_ms(self, compensation_ms: int) -> None:
        """
        动态调整补偿时长
        
        Args:
            compensation_ms: 新的补偿时长（毫秒）
        """
        self.compensation_ms = compensation_ms
        self.enabled = compensation_ms > 0


def create_delay_compensator(compensation_ms: int | None) -> SimpleDelayCompensator | None:
    """
    创建延迟补偿器的便利函数
    
    Args:
        compensation_ms: 补偿时长（毫秒），0或负数表示不创建补偿器
        
    Returns:
        SimpleDelayCompensator或None: 补偿器实例，如果不需要补偿则返回None
    """
    if compensation_ms is None or compensation_ms <= 0:
        return None

    return SimpleDelayCompensator(compensation_ms)


__all__ = [
    "SimpleDelayCompensator",
    "create_delay_compensator",
]
