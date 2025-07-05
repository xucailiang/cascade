"""
性能监控工具

本模块提供性能监控和统计功能，用于跟踪系统资源使用情况和性能指标。
"""

import json
import logging
import os
import platform
import statistics
import threading
import time
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

import psutil

from .atomic import AtomicCounter, AtomicDict, AtomicReference, AtomicValue

# 配置日志
logger = logging.getLogger("cascade.performance")


class PerformanceMetric:
    """性能指标基类"""

    def __init__(self, name: str, description: str = ""):
        """
        初始化性能指标
        
        Args:
            name: 指标名称
            description: 指标描述
        """
        self.name = name
        self.description = description
        self.created_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "type": self.__class__.__name__
        }


class CounterMetric(PerformanceMetric):
    """计数器指标"""

    def __init__(self, name: str, description: str = ""):
        """
        初始化计数器指标
        
        Args:
            name: 指标名称
            description: 指标描述
        """
        super().__init__(name, description)
        self.counter = AtomicCounter(0)

    def increment(self, delta: int = 1) -> int:
        """
        增加计数
        
        Args:
            delta: 增加量，默认为1
            
        Returns:
            增加后的计数
        """
        return self.counter.increment(delta)

    def decrement(self, delta: int = 1) -> int:
        """
        减少计数
        
        Args:
            delta: 减少量，默认为1
            
        Returns:
            减少后的计数
        """
        return self.counter.decrement(delta)

    def get(self) -> int:
        """
        获取当前计数
        
        Returns:
            当前计数
        """
        return self.counter.get()

    def reset(self) -> None:
        """重置计数为0"""
        self.counter.reset()

    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        result = super().to_dict()
        result["value"] = self.get()
        return result


class GaugeMetric(PerformanceMetric):
    """仪表盘指标，表示可变的数值"""

    def __init__(self, name: str, description: str = "", initial_value: float = 0.0):
        """
        初始化仪表盘指标
        
        Args:
            name: 指标名称
            description: 指标描述
            initial_value: 初始值
        """
        super().__init__(name, description)
        self.value = AtomicValue(initial_value)

    def set(self, value: float) -> None:
        """
        设置值
        
        Args:
            value: 新值
        """
        self.value.set(value)

    def get(self) -> float:
        """
        获取当前值
        
        Returns:
            当前值
        """
        return self.value.get()

    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        result = super().to_dict()
        result["value"] = self.get()
        return result


class HistogramMetric(PerformanceMetric):
    """直方图指标，用于统计数值分布"""

    def __init__(self, name: str, description: str = ""):
        """
        初始化直方图指标
        
        Args:
            name: 指标名称
            description: 指标描述
        """
        super().__init__(name, description)
        self.values = []
        self._lock = threading.RLock()

    def record(self, value: float) -> None:
        """
        记录值
        
        Args:
            value: 要记录的值
        """
        with self._lock:
            self.values.append(value)

    def count(self) -> int:
        """
        获取记录的值的数量
        
        Returns:
            记录的值的数量
        """
        with self._lock:
            return len(self.values)

    def sum(self) -> float:
        """
        获取记录的值的总和
        
        Returns:
            记录的值的总和
        """
        with self._lock:
            return sum(self.values)

    def average(self) -> float | None:
        """
        获取记录的值的平均值
        
        Returns:
            记录的值的平均值，如果没有记录则返回None
        """
        with self._lock:
            if not self.values:
                return None
            return sum(self.values) / len(self.values)

    def min(self) -> float | None:
        """
        获取记录的值的最小值
        
        Returns:
            记录的值的最小值，如果没有记录则返回None
        """
        with self._lock:
            if not self.values:
                return None
            return min(self.values)

    def max(self) -> float | None:
        """
        获取记录的值的最大值
        
        Returns:
            记录的值的最大值，如果没有记录则返回None
        """
        with self._lock:
            if not self.values:
                return None
            return max(self.values)

    def median(self) -> float | None:
        """
        获取记录的值的中位数
        
        Returns:
            记录的值的中位数，如果没有记录则返回None
        """
        with self._lock:
            if not self.values:
                return None
            return statistics.median(self.values)

    def percentile(self, p: float) -> float | None:
        """
        获取记录的值的百分位数
        
        Args:
            p: 百分位数，范围为[0, 100]
            
        Returns:
            记录的值的百分位数，如果没有记录则返回None
        """
        with self._lock:
            if not self.values:
                return None
            sorted_values = sorted(self.values)
            k = (len(sorted_values) - 1) * p / 100
            f = int(k)
            c = int(k) + 1 if k < len(sorted_values) - 1 else int(k)
            if f == c:
                return sorted_values[f]
            return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def reset(self) -> None:
        """清空记录的值"""
        with self._lock:
            self.values.clear()

    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        result = super().to_dict()
        with self._lock:
            result.update({
                "count": len(self.values),
                "sum": sum(self.values) if self.values else 0,
                "average": sum(self.values) / len(self.values) if self.values else None,
                "min": min(self.values) if self.values else None,
                "max": max(self.values) if self.values else None,
                "median": statistics.median(self.values) if self.values else None,
                "p90": self.percentile(90),
                "p95": self.percentile(95),
                "p99": self.percentile(99)
            })
        return result


class TimerMetric(PerformanceMetric):
    """计时器指标，用于测量操作耗时"""

    def __init__(self, name: str, description: str = ""):
        """
        初始化计时器指标
        
        Args:
            name: 指标名称
            description: 指标描述
        """
        super().__init__(name, description)
        self.histogram = HistogramMetric(f"{name}_histogram", f"Histogram for {name}")
        self.active_timers = AtomicCounter(0)
        self._thread_local = threading.local()

    def start(self) -> None:
        """开始计时"""
        self._thread_local.start_time = time.time()
        self.active_timers.increment()

    def stop(self) -> float:
        """
        停止计时并记录耗时
        
        Returns:
            耗时（秒）
        """
        if not hasattr(self._thread_local, "start_time"):
            raise RuntimeError("计时器未启动")

        elapsed = time.time() - self._thread_local.start_time
        self.histogram.record(elapsed)
        self.active_timers.decrement()
        delattr(self._thread_local, "start_time")
        return elapsed

    def time(self) -> 'TimerContext':
        """
        获取计时器上下文
        
        Returns:
            计时器上下文
        """
        return TimerContext(self)

    def get_active_count(self) -> int:
        """
        获取活动计时器数量
        
        Returns:
            活动计时器数量
        """
        return self.active_timers.get()

    def reset(self) -> None:
        """重置计时器"""
        self.histogram.reset()
        self.active_timers.reset()

    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        result = super().to_dict()
        result.update({
            "active_count": self.get_active_count(),
            "histogram": self.histogram.to_dict()
        })
        return result


class TimerContext:
    """计时器上下文，用于with语句"""

    def __init__(self, timer: TimerMetric):
        """
        初始化计时器上下文
        
        Args:
            timer: 计时器指标
        """
        self.timer = timer

    def __enter__(self) -> 'TimerContext':
        """进入上下文"""
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出上下文"""
        self.timer.stop()


def timed(timer_name: str, description: str = "") -> Callable:
    """
    计时装饰器
    
    Args:
        timer_name: 计时器名称
        description: 计时器描述
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取或创建计时器
            timer = PerformanceMonitor.get_instance().get_or_create_timer(timer_name, description)

            # 开始计时
            timer.start()
            try:
                # 执行函数
                result = func(*args, **kwargs)
                return result
            finally:
                # 停止计时
                elapsed = timer.stop()
                logger.debug(f"函数 {func.__name__} 执行耗时: {elapsed:.6f}秒")
        return wrapper
    return decorator


class SystemMetrics:
    """系统指标收集器"""

    def __init__(self):
        """初始化系统指标收集器"""
        self.process = psutil.Process(os.getpid())

    def collect(self) -> dict[str, Any]:
        """
        收集系统指标
        
        Returns:
            系统指标字典
        """
        # 获取CPU信息
        cpu_percent = psutil.cpu_percent(interval=0.1)
        process_cpu_percent = self.process.cpu_percent(interval=0.1)

        # 获取内存信息
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()

        # 获取磁盘信息
        disk = psutil.disk_usage('/')

        # 获取网络信息
        net_io = psutil.net_io_counters()

        # 获取线程信息
        thread_count = threading.active_count()

        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "logical_count": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            },
            "process": {
                "pid": os.getpid(),
                "cpu_percent": process_cpu_percent,
                "memory": {
                    "rss": process_memory.rss,  # 物理内存
                    "vms": process_memory.vms,  # 虚拟内存
                    "shared": getattr(process_memory, 'shared', 0),  # 共享内存
                    "text": getattr(process_memory, 'text', 0),  # 代码段
                    "data": getattr(process_memory, 'data', 0)  # 数据段
                },
                "threads": thread_count,
                "create_time": datetime.fromtimestamp(self.process.create_time()).isoformat()
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation()
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine()
            }
        }


class PerformanceMonitor:
    """性能监控器"""

    _instance = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> 'PerformanceMonitor':
        """
        获取单例实例
        
        Returns:
            性能监控器实例
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        """初始化性能监控器"""
        self.counters = AtomicDict[CounterMetric]()
        self.gauges = AtomicDict[GaugeMetric]()
        self.histograms = AtomicDict[HistogramMetric]()
        self.timers = AtomicDict[TimerMetric]()

        self.system_metrics = SystemMetrics()
        self.system_metrics_history = []
        self.max_history_size = 100

        self.collection_interval = 60  # 默认收集间隔（秒）
        self.is_collecting = AtomicValue(False)
        self.collection_thread = None
        self.last_collection_time = AtomicReference[datetime](None)

    def get_or_create_counter(self, name: str, description: str = "") -> CounterMetric:
        """
        获取或创建计数器指标
        
        Args:
            name: 指标名称
            description: 指标描述
            
        Returns:
            计数器指标
        """
        counter = self.counters.get(name)
        if counter is None:
            counter = CounterMetric(name, description)
            self.counters.set(name, counter)
        return counter

    def get_or_create_gauge(self, name: str, description: str = "", initial_value: float = 0.0) -> GaugeMetric:
        """
        获取或创建仪表盘指标
        
        Args:
            name: 指标名称
            description: 指标描述
            initial_value: 初始值
            
        Returns:
            仪表盘指标
        """
        gauge = self.gauges.get(name)
        if gauge is None:
            gauge = GaugeMetric(name, description, initial_value)
            self.gauges.set(name, gauge)
        return gauge

    def get_or_create_histogram(self, name: str, description: str = "") -> HistogramMetric:
        """
        获取或创建直方图指标
        
        Args:
            name: 指标名称
            description: 指标描述
            
        Returns:
            直方图指标
        """
        histogram = self.histograms.get(name)
        if histogram is None:
            histogram = HistogramMetric(name, description)
            self.histograms.set(name, histogram)
        return histogram

    def get_or_create_timer(self, name: str, description: str = "") -> TimerMetric:
        """
        获取或创建计时器指标
        
        Args:
            name: 指标名称
            description: 指标描述
            
        Returns:
            计时器指标
        """
        timer = self.timers.get(name)
        if timer is None:
            timer = TimerMetric(name, description)
            self.timers.set(name, timer)
        return timer

    def collect_system_metrics(self) -> dict[str, Any]:
        """
        收集系统指标
        
        Returns:
            系统指标字典
        """
        metrics = self.system_metrics.collect()

        # 保存到历史记录
        self.system_metrics_history.append(metrics)
        if len(self.system_metrics_history) > self.max_history_size:
            self.system_metrics_history.pop(0)

        self.last_collection_time.set(datetime.now())

        return metrics

    def start_collection(self, interval: int = 60) -> None:
        """
        开始定期收集系统指标
        
        Args:
            interval: 收集间隔（秒）
        """
        if self.is_collecting.get():
            logger.warning("性能指标收集已经在运行")
            return

        self.collection_interval = interval
        self.is_collecting.set(True)

        def collection_task():
            while self.is_collecting.get():
                try:
                    self.collect_system_metrics()
                except Exception as e:
                    logger.error(f"收集系统指标时出错: {e}")

                # 等待下一次收集
                time.sleep(self.collection_interval)

        self.collection_thread = threading.Thread(
            target=collection_task,
            name="PerformanceMetricsCollector",
            daemon=True
        )
        self.collection_thread.start()

        logger.info(f"性能指标收集已启动，间隔: {interval}秒")

    def stop_collection(self) -> None:
        """停止定期收集系统指标"""
        if not self.is_collecting.get():
            logger.warning("性能指标收集未运行")
            return

        self.is_collecting.set(False)

        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)

        self.collection_thread = None
        logger.info("性能指标收集已停止")

    def get_all_metrics(self) -> dict[str, Any]:
        """
        获取所有指标
        
        Returns:
            所有指标的字典
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {}
        }

        # 收集计数器指标
        for name, counter in self.counters.items():
            result["counters"][name] = counter.to_dict()

        # 收集仪表盘指标
        for name, gauge in self.gauges.items():
            result["gauges"][name] = gauge.to_dict()

        # 收集直方图指标
        for name, histogram in self.histograms.items():
            result["histograms"][name] = histogram.to_dict()

        # 收集计时器指标
        for name, timer in self.timers.items():
            result["timers"][name] = timer.to_dict()

        # 添加最新的系统指标
        if self.system_metrics_history:
            result["system"] = self.system_metrics_history[-1]
        else:
            result["system"] = self.collect_system_metrics()

        return result

    def reset_all_metrics(self) -> None:
        """重置所有指标"""
        # 重置计数器指标
        for counter in self.counters.values():
            counter.reset()

        # 重置直方图指标
        for histogram in self.histograms.values():
            histogram.reset()

        # 重置计时器指标
        for timer in self.timers.values():
            timer.reset()

        # 清空系统指标历史
        self.system_metrics_history.clear()

        logger.info("所有性能指标已重置")

    def export_metrics_json(self, file_path: str | None = None) -> str:
        """
        导出指标为JSON
        
        Args:
            file_path: 文件路径，如果提供则写入文件
            
        Returns:
            JSON字符串
        """
        metrics = self.get_all_metrics()
        json_str = json.dumps(metrics, indent=2)

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"性能指标已导出到: {file_path}")

        return json_str


# 导出的类和函数
__all__ = [
    "PerformanceMetric",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "TimerMetric",
    "TimerContext",
    "SystemMetrics",
    "PerformanceMonitor",
    "timed"
]
