"""
内部工具模块

提供Cascade项目内部使用的通用工具函数。
保留utils模块供formats等模块使用。
"""

# 导出utils模块的常用函数
from .utils import (
    align_to_cache_line,
    ensure_contiguous,
    get_memory_usage,
    measure_time,
    timer,
)

__all__ = [
    # 性能优化
    "align_to_cache_line",
    "ensure_contiguous",
    
    # 时间测量
    "measure_time",
    "timer",
    
    # 内存管理
    "get_memory_usage",
]
