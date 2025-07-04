"""
Cascade 内部实现模块

本模块包含项目的内部实现细节，不属于公开API。
外部用户不应直接依赖本模块的内容。

包含的组件:
- Atomic: 原子操作工具
- PerformanceMonitor: 性能监控器
- ThreadPoolManager: 线程池管理器
- InternalUtils: 内部工具函数
"""

# 导入原子操作工具
from .atomic import (
    AtomicValue,
    AtomicCounter,
    AtomicReference,
    AtomicDict,
    AtomicFlag,
    AtomicLock,
    AtomicStampedReference
)

# 导入性能监控工具
from .performance import (
    PerformanceMetric,
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    TimerMetric,
    TimerContext,
    SystemMetrics,
    PerformanceMonitor,
    timed
)

# 导入线程池管理工具
from .thread_pool import (
    TaskPriority,
    TaskStatus,
    TaskStats,
    Task,
    PriorityThreadPoolExecutor,
    ThreadPoolManager
)

# 导入内部工具函数
from .utils import (
    Singleton,
    LazyProperty,
    InternalUtils
)

# 导出的类和函数
__all__ = [
    # 原子操作工具
    "AtomicValue",
    "AtomicCounter",
    "AtomicReference",
    "AtomicDict",
    "AtomicFlag",
    "AtomicLock",
    "AtomicStampedReference",
    
    # 性能监控工具
    "PerformanceMetric",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "TimerMetric",
    "TimerContext",
    "SystemMetrics",
    "PerformanceMonitor",
    "timed",
    
    # 线程池管理工具
    "TaskPriority",
    "TaskStatus",
    "TaskStats",
    "Task",
    "PriorityThreadPoolExecutor",
    "ThreadPoolManager",
    
    # 内部工具函数
    "Singleton",
    "LazyProperty",
    "InternalUtils"
]

# 单例实例
performance_monitor = PerformanceMonitor.get_instance()
thread_pool_manager = ThreadPoolManager.get_instance()