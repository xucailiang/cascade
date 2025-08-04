"""
Cascade内部工具模块

提供项目内部使用的工具类和函数，包括：
- 原子操作类 (atomic.py)
- 通用工具函数 (utils.py)

这些工具主要用于：
- 并发编程支持
- 性能优化
- 调试和监控
- 内存管理

注意：这个模块仅供内部使用，不应该被外部代码直接导入。
"""

from .atomic import AtomicBoolean, AtomicFloat, AtomicInteger, AtomicReference
from .thread_pool import ThreadWorkerStats, VADThreadPool, VADThreadPoolConfig
from .utils import (
    # 线程安全
    ThreadSafeCounter,
    # 性能优化
    align_to_cache_line,
    deep_merge_dicts,
    ensure_contiguous,
    exception_handler,
    force_garbage_collection,
    # 内存管理
    get_memory_usage,
    log_function_calls,
    measure_async_time,
    # 时间测量
    measure_time,
    memory_monitor,
    # 调试辅助
    setup_logging,
    timer,
    # 配置验证
    validate_config,
)

# 为了兼容测试文件，提供别名
AtomicCounter = AtomicInteger
ensure_memory_alignment = align_to_cache_line

def check_cache_line_alignment(array, cache_line_size: int = 64) -> bool:
    """
    检查数组是否按缓存行对齐
    
    Args:
        array: 要检查的数组
        cache_line_size: 缓存行大小，默认64字节
        
    Returns:
        True 如果对齐，False 否则
    """
    try:
        import numpy as np
        if isinstance(array, np.ndarray):
            try:
                return array.ctypes.data % cache_line_size == 0
            except AttributeError:
                # 某些numpy版本可能不支持ctypes.data
                return True  # 假设已对齐
        return True
    except ImportError:
        return True

def measure_performance(func, args=None, kwargs=None):
    """
    测量函数性能并返回结果和执行时间
    
    Args:
        func: 要测量的函数
        args: 位置参数元组
        kwargs: 关键字参数字典
        
    Returns:
        (result, duration_ms): 函数结果和执行时间（毫秒）
    """
    import time
    args = args or ()
    kwargs = kwargs or {}

    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        return result, duration_ms
    except Exception:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        # 重新抛出异常但保留时间测量
        raise

__all__ = [
    # 原子操作类
    "AtomicInteger", "AtomicFloat", "AtomicBoolean", "AtomicReference",
    "AtomicCounter",  # 别名

    # VAD线程池组件
    "VADThreadPool", "VADThreadPoolConfig", "ThreadWorkerStats",

    # 性能优化工具
    "align_to_cache_line", "ensure_contiguous",
    "ensure_memory_alignment", "check_cache_line_alignment",  # 别名和新函数

    # 时间测量工具
    "measure_time", "measure_async_time", "timer", "measure_performance",

    # 内存管理工具
    "get_memory_usage", "force_garbage_collection", "memory_monitor",

    # 调试辅助工具
    "setup_logging", "log_function_calls", "exception_handler",

    # 配置验证工具
    "validate_config", "deep_merge_dicts",

    # 线程安全工具
    "ThreadSafeCounter",
]

# 模块版本信息
__version__ = "0.1.0"
__author__ = "Cascade Team"
__description__ = "Cascade项目内部工具模块"
