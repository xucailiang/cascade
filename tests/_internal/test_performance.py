"""
性能监控工具测试
"""

import json
import os
import tempfile
import threading
import time
import unittest

from cascade._internal.performance import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    PerformanceMetric,
    PerformanceMonitor,
    SystemMetrics,
    TimerContext,
    TimerMetric,
    timed,
)


class TestPerformanceMetric(unittest.TestCase):
    """PerformanceMetric测试"""

    def test_init(self):
        """测试初始化"""
        metric = PerformanceMetric("test_metric", "Test metric description")

        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.description, "Test metric description")
        self.assertIsNotNone(metric.created_at)

    def test_to_dict(self):
        """测试转换为字典"""
        metric = PerformanceMetric("test_metric", "Test metric description")

        result = metric.to_dict()
        self.assertEqual(result["name"], "test_metric")
        self.assertEqual(result["description"], "Test metric description")
        self.assertIsNotNone(result["created_at"])
        self.assertEqual(result["type"], "PerformanceMetric")


class TestCounterMetric(unittest.TestCase):
    """CounterMetric测试"""

    def test_init(self):
        """测试初始化"""
        counter = CounterMetric("test_counter", "Test counter description")

        self.assertEqual(counter.name, "test_counter")
        self.assertEqual(counter.description, "Test counter description")
        self.assertEqual(counter.get(), 0)

    def test_increment(self):
        """测试增加计数"""
        counter = CounterMetric("test_counter")

        result = counter.increment()
        self.assertEqual(result, 1)
        self.assertEqual(counter.get(), 1)

        result = counter.increment(5)
        self.assertEqual(result, 6)
        self.assertEqual(counter.get(), 6)

    def test_decrement(self):
        """测试减少计数"""
        counter = CounterMetric("test_counter")
        counter.increment(10)

        result = counter.decrement()
        self.assertEqual(result, 9)
        self.assertEqual(counter.get(), 9)

        result = counter.decrement(5)
        self.assertEqual(result, 4)
        self.assertEqual(counter.get(), 4)

    def test_reset(self):
        """测试重置计数"""
        counter = CounterMetric("test_counter")
        counter.increment(10)

        counter.reset()
        self.assertEqual(counter.get(), 0)

    def test_to_dict(self):
        """测试转换为字典"""
        counter = CounterMetric("test_counter", "Test counter description")
        counter.increment(5)

        result = counter.to_dict()
        self.assertEqual(result["name"], "test_counter")
        self.assertEqual(result["description"], "Test counter description")
        self.assertEqual(result["value"], 5)
        self.assertEqual(result["type"], "CounterMetric")

    def test_thread_safety(self):
        """测试线程安全性"""
        counter = CounterMetric("test_counter")
        iterations = 1000
        thread_count = 10

        def increment():
            for _ in range(iterations):
                counter.increment()

        threads = []
        for _ in range(thread_count):
            thread = threading.Thread(target=increment)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(counter.get(), iterations * thread_count)


class TestGaugeMetric(unittest.TestCase):
    """GaugeMetric测试"""

    def test_init(self):
        """测试初始化"""
        gauge = GaugeMetric("test_gauge", "Test gauge description", 10.5)

        self.assertEqual(gauge.name, "test_gauge")
        self.assertEqual(gauge.description, "Test gauge description")
        self.assertEqual(gauge.get(), 10.5)

    def test_set_get(self):
        """测试设置和获取值"""
        gauge = GaugeMetric("test_gauge")

        gauge.set(42.5)
        self.assertEqual(gauge.get(), 42.5)

    def test_to_dict(self):
        """测试转换为字典"""
        gauge = GaugeMetric("test_gauge", "Test gauge description", 10.5)

        result = gauge.to_dict()
        self.assertEqual(result["name"], "test_gauge")
        self.assertEqual(result["description"], "Test gauge description")
        self.assertEqual(result["value"], 10.5)
        self.assertEqual(result["type"], "GaugeMetric")


class TestHistogramMetric(unittest.TestCase):
    """HistogramMetric测试"""

    def test_init(self):
        """测试初始化"""
        histogram = HistogramMetric("test_histogram", "Test histogram description")

        self.assertEqual(histogram.name, "test_histogram")
        self.assertEqual(histogram.description, "Test histogram description")
        self.assertEqual(histogram.count(), 0)

    def test_record(self):
        """测试记录值"""
        histogram = HistogramMetric("test_histogram")

        histogram.record(10.5)
        histogram.record(20.5)
        histogram.record(30.5)

        self.assertEqual(histogram.count(), 3)

    def test_sum(self):
        """测试求和"""
        histogram = HistogramMetric("test_histogram")

        histogram.record(10.5)
        histogram.record(20.5)
        histogram.record(30.5)

        self.assertEqual(histogram.sum(), 61.5)

    def test_average(self):
        """测试求平均值"""
        histogram = HistogramMetric("test_histogram")

        # 空直方图
        self.assertIsNone(histogram.average())

        histogram.record(10.5)
        histogram.record(20.5)
        histogram.record(30.5)

        self.assertEqual(histogram.average(), 20.5)

    def test_min_max(self):
        """测试最小值和最大值"""
        histogram = HistogramMetric("test_histogram")

        # 空直方图
        self.assertIsNone(histogram.min())
        self.assertIsNone(histogram.max())

        histogram.record(10.5)
        histogram.record(20.5)
        histogram.record(30.5)

        self.assertEqual(histogram.min(), 10.5)
        self.assertEqual(histogram.max(), 30.5)

    def test_median(self):
        """测试中位数"""
        histogram = HistogramMetric("test_histogram")

        # 空直方图
        self.assertIsNone(histogram.median())

        histogram.record(10.5)
        histogram.record(20.5)
        histogram.record(30.5)

        self.assertEqual(histogram.median(), 20.5)

        # 偶数个值
        histogram.record(40.5)
        self.assertEqual(histogram.median(), 25.5)

    def test_percentile(self):
        """测试百分位数"""
        histogram = HistogramMetric("test_histogram")

        # 空直方图
        self.assertIsNone(histogram.percentile(90))

        # 添加100个值：0到99
        for i in range(100):
            histogram.record(i)

        self.assertEqual(histogram.percentile(50), 49.5)  # 中位数
        self.assertEqual(histogram.percentile(90), 89.5)  # 90分位数
        self.assertEqual(histogram.percentile(95), 94.5)  # 95分位数
        self.assertEqual(histogram.percentile(99), 98.5)  # 99分位数

    def test_reset(self):
        """测试重置"""
        histogram = HistogramMetric("test_histogram")

        histogram.record(10.5)
        histogram.record(20.5)

        histogram.reset()
        self.assertEqual(histogram.count(), 0)

    def test_to_dict(self):
        """测试转换为字典"""
        histogram = HistogramMetric("test_histogram", "Test histogram description")

        histogram.record(10.5)
        histogram.record(20.5)
        histogram.record(30.5)

        result = histogram.to_dict()
        self.assertEqual(result["name"], "test_histogram")
        self.assertEqual(result["description"], "Test histogram description")
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["sum"], 61.5)
        self.assertEqual(result["average"], 20.5)
        self.assertEqual(result["min"], 10.5)
        self.assertEqual(result["max"], 30.5)
        self.assertEqual(result["median"], 20.5)
        self.assertEqual(result["type"], "HistogramMetric")

    def test_thread_safety(self):
        """测试线程安全性"""
        histogram = HistogramMetric("test_histogram")
        iterations = 100
        thread_count = 10

        def record_values():
            for i in range(iterations):
                histogram.record(i)

        threads = []
        for _ in range(thread_count):
            thread = threading.Thread(target=record_values)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(histogram.count(), iterations * thread_count)


class TestTimerMetric(unittest.TestCase):
    """TimerMetric测试"""

    def test_init(self):
        """测试初始化"""
        timer = TimerMetric("test_timer", "Test timer description")

        self.assertEqual(timer.name, "test_timer")
        self.assertEqual(timer.description, "Test timer description")
        self.assertEqual(timer.get_active_count(), 0)

    def test_start_stop(self):
        """测试开始和停止计时"""
        timer = TimerMetric("test_timer")

        timer.start()
        self.assertEqual(timer.get_active_count(), 1)

        time.sleep(0.01)  # 短暂延迟

        elapsed = timer.stop()
        self.assertGreater(elapsed, 0)
        self.assertEqual(timer.get_active_count(), 0)
        self.assertEqual(timer.histogram.count(), 1)

    def test_time_context(self):
        """测试计时上下文"""
        timer = TimerMetric("test_timer")

        with timer.time():
            self.assertEqual(timer.get_active_count(), 1)
            time.sleep(0.01)  # 短暂延迟

        self.assertEqual(timer.get_active_count(), 0)
        self.assertEqual(timer.histogram.count(), 1)

    def test_reset(self):
        """测试重置"""
        timer = TimerMetric("test_timer")

        with timer.time():
            time.sleep(0.01)  # 短暂延迟

        timer.reset()
        self.assertEqual(timer.get_active_count(), 0)
        self.assertEqual(timer.histogram.count(), 0)

    def test_to_dict(self):
        """测试转换为字典"""
        timer = TimerMetric("test_timer", "Test timer description")

        with timer.time():
            time.sleep(0.01)  # 短暂延迟

        result = timer.to_dict()
        self.assertEqual(result["name"], "test_timer")
        self.assertEqual(result["description"], "Test timer description")
        self.assertEqual(result["active_count"], 0)
        self.assertEqual(result["histogram"]["count"], 1)
        self.assertEqual(result["type"], "TimerMetric")

    def test_thread_safety(self):
        """测试线程安全性"""
        timer = TimerMetric("test_timer")
        iterations = 10
        thread_count = 10

        def time_operation():
            for _ in range(iterations):
                with timer.time():
                    time.sleep(0.001)  # 短暂延迟

        threads = []
        for _ in range(thread_count):
            thread = threading.Thread(target=time_operation)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(timer.histogram.count(), iterations * thread_count)


class TestTimerContext(unittest.TestCase):
    """TimerContext测试"""

    def test_context_manager(self):
        """测试上下文管理器"""
        timer = TimerMetric("test_timer")

        with TimerContext(timer):
            self.assertEqual(timer.get_active_count(), 1)
            time.sleep(0.01)  # 短暂延迟

        self.assertEqual(timer.get_active_count(), 0)
        self.assertEqual(timer.histogram.count(), 1)


class TestSystemMetrics(unittest.TestCase):
    """SystemMetrics测试"""

    def test_collect(self):
        """测试收集系统指标"""
        metrics = SystemMetrics()

        result = metrics.collect()

        # 验证结果包含预期的字段
        self.assertIn("timestamp", result)
        self.assertIn("system", result)
        self.assertIn("process", result)
        self.assertIn("python", result)
        self.assertIn("platform", result)

        # 验证系统指标
        system = result["system"]
        self.assertIn("cpu", system)
        self.assertIn("memory", system)
        self.assertIn("disk", system)
        self.assertIn("network", system)

        # 验证进程指标
        process = result["process"]
        self.assertIn("pid", process)
        self.assertIn("cpu_percent", process)
        self.assertIn("memory", process)
        self.assertIn("threads", process)

        # 验证Python指标
        python = result["python"]
        self.assertIn("version", python)
        self.assertIn("implementation", python)

        # 验证平台指标
        platform_info = result["platform"]
        self.assertIn("system", platform_info)
        self.assertIn("release", platform_info)
        self.assertIn("version", platform_info)
        self.assertIn("machine", platform_info)


class TestPerformanceMonitor(unittest.TestCase):
    """PerformanceMonitor测试"""

    def setUp(self):
        """测试前准备"""
        # 重置单例实例
        PerformanceMonitor._instance = None
        self.monitor = PerformanceMonitor.get_instance()

    def test_singleton(self):
        """测试单例模式"""
        monitor1 = PerformanceMonitor.get_instance()
        monitor2 = PerformanceMonitor.get_instance()

        self.assertIs(monitor1, monitor2)

    def test_get_or_create_counter(self):
        """测试获取或创建计数器"""
        counter1 = self.monitor.get_or_create_counter("test_counter", "Test counter")
        self.assertIsInstance(counter1, CounterMetric)
        self.assertEqual(counter1.name, "test_counter")
        self.assertEqual(counter1.description, "Test counter")

        # 再次获取相同名称的计数器
        counter2 = self.monitor.get_or_create_counter("test_counter")
        self.assertIs(counter1, counter2)

    def test_get_or_create_gauge(self):
        """测试获取或创建仪表盘"""
        gauge1 = self.monitor.get_or_create_gauge("test_gauge", "Test gauge", 10.5)
        self.assertIsInstance(gauge1, GaugeMetric)
        self.assertEqual(gauge1.name, "test_gauge")
        self.assertEqual(gauge1.description, "Test gauge")
        self.assertEqual(gauge1.get(), 10.5)

        # 再次获取相同名称的仪表盘
        gauge2 = self.monitor.get_or_create_gauge("test_gauge")
        self.assertIs(gauge1, gauge2)

    def test_get_or_create_histogram(self):
        """测试获取或创建直方图"""
        histogram1 = self.monitor.get_or_create_histogram("test_histogram", "Test histogram")
        self.assertIsInstance(histogram1, HistogramMetric)
        self.assertEqual(histogram1.name, "test_histogram")
        self.assertEqual(histogram1.description, "Test histogram")

        # 再次获取相同名称的直方图
        histogram2 = self.monitor.get_or_create_histogram("test_histogram")
        self.assertIs(histogram1, histogram2)

    def test_get_or_create_timer(self):
        """测试获取或创建计时器"""
        timer1 = self.monitor.get_or_create_timer("test_timer", "Test timer")
        self.assertIsInstance(timer1, TimerMetric)
        self.assertEqual(timer1.name, "test_timer")
        self.assertEqual(timer1.description, "Test timer")

        # 再次获取相同名称的计时器
        timer2 = self.monitor.get_or_create_timer("test_timer")
        self.assertIs(timer1, timer2)

    def test_collect_system_metrics(self):
        """测试收集系统指标"""
        result = self.monitor.collect_system_metrics()

        # 验证结果包含预期的字段
        self.assertIn("timestamp", result)
        self.assertIn("system", result)
        self.assertIn("process", result)

        # 验证历史记录
        self.assertEqual(len(self.monitor.system_metrics_history), 1)
        self.assertIs(self.monitor.system_metrics_history[0], result)

        # 验证最后收集时间
        self.assertIsNotNone(self.monitor.last_collection_time.get())

    def test_start_stop_collection(self):
        """测试开始和停止收集"""
        # 开始收集
        self.monitor.start_collection(interval=0.1)
        self.assertTrue(self.monitor.is_collecting.get())
        self.assertIsNotNone(self.monitor.collection_thread)
        self.assertTrue(self.monitor.collection_thread.is_alive())

        # 等待收集一些数据
        time.sleep(0.3)

        # 停止收集
        self.monitor.stop_collection()
        self.assertFalse(self.monitor.is_collecting.get())

        # 验证收集了一些数据
        self.assertGreater(len(self.monitor.system_metrics_history), 0)

    def test_get_all_metrics(self):
        """测试获取所有指标"""
        # 创建一些指标
        counter = self.monitor.get_or_create_counter("test_counter")
        counter.increment(5)

        gauge = self.monitor.get_or_create_gauge("test_gauge")
        gauge.set(10.5)

        histogram = self.monitor.get_or_create_histogram("test_histogram")
        histogram.record(20.5)

        timer = self.monitor.get_or_create_timer("test_timer")
        with timer.time():
            time.sleep(0.01)

        # 获取所有指标
        result = self.monitor.get_all_metrics()

        # 验证结果包含预期的字段
        self.assertIn("timestamp", result)
        self.assertIn("counters", result)
        self.assertIn("gauges", result)
        self.assertIn("histograms", result)
        self.assertIn("timers", result)
        self.assertIn("system", result)

        # 验证指标值
        self.assertEqual(result["counters"]["test_counter"]["value"], 5)
        self.assertEqual(result["gauges"]["test_gauge"]["value"], 10.5)
        self.assertEqual(result["histograms"]["test_histogram"]["count"], 1)
        self.assertEqual(result["timers"]["test_timer"]["histogram"]["count"], 1)

    def test_reset_all_metrics(self):
        """测试重置所有指标"""
        # 创建一些指标
        counter = self.monitor.get_or_create_counter("test_counter")
        counter.increment(5)

        histogram = self.monitor.get_or_create_histogram("test_histogram")
        histogram.record(20.5)

        timer = self.monitor.get_or_create_timer("test_timer")
        with timer.time():
            time.sleep(0.01)

        # 收集系统指标
        self.monitor.collect_system_metrics()

        # 重置所有指标
        self.monitor.reset_all_metrics()

        # 验证指标已重置
        self.assertEqual(counter.get(), 0)
        self.assertEqual(histogram.count(), 0)
        self.assertEqual(timer.histogram.count(), 0)
        self.assertEqual(len(self.monitor.system_metrics_history), 0)

    def test_export_metrics_json(self):
        """测试导出指标为JSON"""
        # 创建一些指标
        counter = self.monitor.get_or_create_counter("test_counter")
        counter.increment(5)

        # 导出为JSON字符串
        json_str = self.monitor.export_metrics_json()

        # 验证JSON字符串
        data = json.loads(json_str)
        self.assertIn("counters", data)
        self.assertIn("test_counter", data["counters"])
        self.assertEqual(data["counters"]["test_counter"]["value"], 5)

        # 导出到文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            json_str = self.monitor.export_metrics_json(temp_path)

            # 验证文件内容
            with open(temp_path) as f:
                file_content = f.read()

            self.assertEqual(file_content, json_str)

            # 验证文件内容是有效的JSON
            data = json.loads(file_content)
            self.assertIn("counters", data)
            self.assertIn("test_counter", data["counters"])
            self.assertEqual(data["counters"]["test_counter"]["value"], 5)
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestTimedDecorator(unittest.TestCase):
    """timed装饰器测试"""

    def setUp(self):
        """测试前准备"""
        # 重置单例实例
        PerformanceMonitor._instance = None
        self.monitor = PerformanceMonitor.get_instance()

    def test_timed_decorator(self):
        """测试timed装饰器"""
        # 定义被装饰的函数
        @timed("test_function", "Test function timing")
        def test_function(sleep_time):
            time.sleep(sleep_time)
            return "result"

        # 调用函数
        result = test_function(0.01)

        # 验证函数返回值
        self.assertEqual(result, "result")

        # 验证计时器
        timer = self.monitor.get_or_create_timer("test_function")
        self.assertEqual(timer.description, "Test function timing")
        self.assertEqual(timer.histogram.count(), 1)
        self.assertGreater(timer.histogram.min(), 0.01)

    def test_timed_decorator_exception(self):
        """测试timed装饰器处理异常"""
        # 定义会抛出异常的被装饰函数
        @timed("test_exception")
        def test_exception():
            time.sleep(0.01)
            raise ValueError("Test exception")

        # 调用函数，应该抛出异常
        with self.assertRaises(ValueError):
            test_exception()

        # 验证计时器仍然记录了时间
        timer = self.monitor.get_or_create_timer("test_exception")
        self.assertEqual(timer.histogram.count(), 1)
        self.assertGreater(timer.histogram.min(), 0.01)


if __name__ == "__main__":
    unittest.main()
