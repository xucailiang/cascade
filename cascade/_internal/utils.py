"""
内部工具函数模块

提供Cascade项目内部使用的通用工具函数，包括：
- 性能优化工具
- 调试辅助工具
- 内存管理工具
- 时间测量工具

设计原则：
- 高性能实现
- 简洁易用的API
- 完整的错误处理
- 详细的文档说明
"""

import functools
import gc
import logging
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

# 需要使用numpy但避免硬依赖
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# ===== 性能优化工具 =====

def align_to_cache_line(array: Any, cache_line_size: int = 64) -> Any:
    """
    将数组对齐到缓存行
    
    这个函数尝试将数组数据对齐到CPU缓存行边界，
    以提高内存访问性能。
    
    Args:
        array: 输入数组（numpy数组或类似结构）
        cache_line_size: 缓存行大小，默认64字节
        
    Returns:
        对齐后的数组
        
    Raises:
        ImportError: 如果numpy不可用
        ValueError: 如果输入参数无效
    """
    if not HAS_NUMPY:
        raise ImportError("numpy未安装，无法执行数组对齐操作")

    if not isinstance(array, np.ndarray):
        raise ValueError("输入必须是numpy数组")

    if cache_line_size <= 0 or not isinstance(cache_line_size, int):
        raise ValueError("缓存行大小必须是正整数")

    # 计算需要的对齐大小
    element_size = array.itemsize
    total_bytes = array.size * element_size
    aligned_bytes = (total_bytes + cache_line_size - 1) // cache_line_size * cache_line_size
    aligned_size = aligned_bytes // element_size

    # 创建对齐的数组
    aligned_array = np.zeros(aligned_size, dtype=array.dtype)
    aligned_array[:array.size] = array.flatten()

    # 验证对齐（如果可能）
    try:
        if aligned_array.ctypes.data % cache_line_size != 0:
            logging.warning("数组可能未正确对齐到缓存行")
    except AttributeError:
        # 某些numpy版本可能不支持ctypes.data
        pass

    # 返回原始形状的视图
    return aligned_array[:array.size].reshape(array.shape)

def ensure_contiguous(array: Any) -> Any:
    """
    确保数组在内存中是连续的
    
    Args:
        array: 输入数组
        
    Returns:
        连续的数组
    """
    if not HAS_NUMPY:
        return array

    if isinstance(array, np.ndarray) and not array.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(array)
    return array

# ===== 时间测量工具 =====

def measure_time(func: F) -> F:
    """
    测量函数执行时间的装饰器
    
    Args:
        func: 要测量的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"{func.__name__} 执行时间: {execution_time:.2f}ms")

    return wrapper

def measure_async_time(func: F) -> F:
    """
    测量异步函数执行时间的装饰器
    
    Args:
        func: 要测量的异步函数
        
    Returns:
        装饰后的异步函数
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000
            print(f"{func.__name__} 异步执行时间: {execution_time:.2f}ms")

    return wrapper

@contextmanager
def timer(name: str = "操作"):
    """
    上下文管理器形式的计时器
    
    Args:
        name: 操作名称，用于日志输出
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        print(f"{name} 执行时间: {execution_time:.2f}ms")

# ===== 内存管理工具 =====

def get_memory_usage() -> dict[str, float]:
    """
    获取当前进程的内存使用情况
    
    Returns:
        包含内存使用信息的字典
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # 物理内存
            "vms_mb": memory_info.vms / 1024 / 1024,  # 虚拟内存
            "percent": process.memory_percent(),       # 内存使用百分比
        }
    except ImportError:
        # 如果psutil不可用，使用基本的gc统计
        stats = gc.get_stats()
        return {
            "gc_collections": sum(stat["collections"] for stat in stats),
            "gc_collected": sum(stat["collected"] for stat in stats),
            "gc_uncollectable": sum(stat["uncollectable"] for stat in stats),
        }

def force_garbage_collection() -> dict[str, int]:
    """
    强制执行垃圾回收
    
    Returns:
        垃圾回收统计信息
    """
    # 执行完整的垃圾回收
    collected = gc.collect()

    # 获取统计信息
    stats = gc.get_stats()

    return {
        "collected": collected,
        "generation_0": stats[0]["collections"] if stats else 0,
        "generation_1": stats[1]["collections"] if len(stats) > 1 else 0,
        "generation_2": stats[2]["collections"] if len(stats) > 2 else 0,
    }

@contextmanager
def memory_monitor(name: str = "操作"):
    """
    监控代码块的内存使用情况
    
    Args:
        name: 操作名称，用于日志输出
    """
    before = get_memory_usage()
    try:
        yield
    finally:
        after = get_memory_usage()
        if "rss_mb" in before and "rss_mb" in after:
            delta = after["rss_mb"] - before["rss_mb"]
            print(f"{name} 内存变化: {delta:+.2f}MB")

# ===== 调试辅助工具 =====

def setup_logging(
    level: str = "INFO",
    format_string: str | None = None,
    include_thread_info: bool = False
) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        level: 日志级别
        format_string: 自定义格式字符串
        include_thread_info: 是否包含线程信息
        
    Returns:
        配置好的logger实例
    """
    if format_string is None:
        if include_thread_info:
            format_string = "%(asctime)s [%(levelname)s] [%(thread)d] %(name)s: %(message)s"
        else:
            format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    return logging.getLogger("cascade")

def log_function_calls(logger: logging.Logger | None = None):
    """
    记录函数调用的装饰器
    
    Args:
        logger: 可选的logger实例
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_str = ", ".join(repr(arg) for arg in args)
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))

            logger.debug(f"调用 {func.__name__}({all_args})")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} 返回: {result!r}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} 抛出异常: {e}")
                raise

        return wrapper
    return decorator

def exception_handler(
    logger: logging.Logger | None = None,
    reraise: bool = True,
    default_return: Any = None
):
    """
    异常处理装饰器
    
    Args:
        logger: 可选的logger实例
        reraise: 是否重新抛出异常
        default_return: 异常时的默认返回值
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} 发生异常: {e}")
                logger.debug(traceback.format_exc())

                if reraise:
                    raise
                return default_return

        return wrapper
    return decorator

# ===== 配置验证工具 =====

def validate_config(config: dict[str, Any], required_keys: list[str]) -> bool:
    """
    验证配置字典是否包含必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        True 如果配置有效，False 否则
        
    Raises:
        ValueError: 如果配置无效
    """
    if not isinstance(config, dict):
        raise ValueError("配置必须是字典类型")

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"配置缺少必需的键: {missing_keys}")

    return True

def deep_merge_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    深度合并两个字典
    
    Args:
        dict1: 第一个字典
        dict2: 第二个字典（优先级更高）
        
    Returns:
        合并后的字典
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result

# ===== 线程安全工具 =====

class ThreadSafeCounter:
    """线程安全的计数器"""

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> int:
        """增加计数并返回新值"""
        with self._lock:
            self._value += delta
            return self._value

    def decrement(self, delta: int = 1) -> int:
        """减少计数并返回新值"""
        with self._lock:
            self._value -= delta
            return self._value

    def get(self) -> int:
        """获取当前值"""
        with self._lock:
            return self._value

    def reset(self) -> int:
        """重置为0并返回之前的值"""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value

# ===== 导出列表 =====

__all__ = [
    # 性能优化
    "align_to_cache_line", "ensure_contiguous",

    # 时间测量
    "measure_time", "measure_async_time", "timer",

    # 内存管理
    "get_memory_usage", "force_garbage_collection", "memory_monitor",

    # 调试辅助
    "setup_logging", "log_function_calls", "exception_handler",

    # 配置验证
    "validate_config", "deep_merge_dicts",

    # 线程安全
    "ThreadSafeCounter",
]
