"""
内部模块集成测试

测试_internal模块的组件如何协同工作。
"""

import json
import os
import tempfile
import time
import unittest

from cascade._internal import (
    AtomicCounter,
    AtomicFlag,
    AtomicValue,
    PerformanceMonitor,
    TaskPriority,
    TaskStatus,
    ThreadPoolManager,
    timed,
)


class TestInternalIntegration(unittest.TestCase):
    """内部模块集成测试"""

    def setUp(self):
        """测试前准备"""
        # 重置单例实例
        PerformanceMonitor._instance = None
        ThreadPoolManager._instance = None

        # 获取实例
        self.performance_monitor = PerformanceMonitor.get_instance()
        self.thread_pool_manager = ThreadPoolManager.get_instance()

    def tearDown(self):
        """测试后清理"""
        # 关闭线程池管理器
        if not self.thread_pool_manager.is_shutdown.get():
            self.thread_pool_manager.shutdown(wait=True)

        # 重置性能监控器
        self.performance_monitor.reset_all_metrics()

    def test_thread_pool_with_performance_monitor(self):
        """测试线程池与性能监控器的集成"""
        # 创建计数器指标
        task_counter = self.performance_monitor.get_or_create_counter(
            "test_tasks", "测试任务计数"
        )

        # 创建直方图指标
        task_time_histogram = self.performance_monitor.get_or_create_histogram(
            "test_task_time", "测试任务耗时"
        )

        # 定义任务函数
        def task_func(task_id, sleep_time):
            # 增加任务计数
            task_counter.increment()

            # 记录开始时间
            start_time = time.time()

            # 模拟任务执行
            time.sleep(sleep_time)

            # 记录结束时间和耗时
            end_time = time.time()
            elapsed = end_time - start_time

            # 记录任务耗时
            task_time_histogram.record(elapsed)

            return f"Task {task_id} completed in {elapsed:.3f}s"

        # 提交任务
        tasks = []
        for i in range(10):
            sleep_time = 0.01 * (i + 1)  # 0.01s, 0.02s, ..., 0.1s
            task = self.thread_pool_manager.submit_task(
                task_func,
                args=(i, sleep_time),
                priority=TaskPriority.NORMAL,
                pool_type="default"
            )
            tasks.append(task)

        # 等待所有任务完成
        results = [task.get_result(timeout=2.0) for task in tasks]

        # 验证结果
        self.assertEqual(len(results), 10)
        self.assertTrue(all("completed" in result for result in results))

        # 验证性能指标
        self.assertEqual(task_counter.get(), 10)
        self.assertEqual(task_time_histogram.count(), 10)
        self.assertGreaterEqual(task_time_histogram.min(), 0.01)
        self.assertLessEqual(task_time_histogram.max(), 0.15)  # 允许一些误差

    def test_timed_decorator_with_thread_pool(self):
        """测试timed装饰器与线程池的集成"""
        # 定义被装饰的函数
        @timed("parallel_task", "并行任务计时")
        def parallel_task(count):
            results = []

            def worker(i):
                time.sleep(0.01)
                return i * i

            # 使用线程池执行任务
            tasks = self.thread_pool_manager.submit_tasks(
                funcs=[worker] * count,
                args_list=[(i,) for i in range(count)],
                priority=TaskPriority.HIGH,
                pool_type="default"
            )

            # 获取结果
            for task in tasks:
                results.append(task.get_result(timeout=1.0))

            return results

        # 执行函数
        results = parallel_task(5)

        # 验证结果
        self.assertEqual(results, [0, 1, 4, 9, 16])

        # 验证计时器
        timer = self.performance_monitor.get_or_create_timer("parallel_task")
        self.assertEqual(timer.description, "并行任务计时")
        self.assertEqual(timer.histogram.count(), 1)
        self.assertGreater(timer.histogram.min(), 0.01)

    def test_atomic_operations_with_thread_pool(self):
        """测试原子操作与线程池的集成"""
        # 创建原子计数器
        counter = AtomicCounter(0)

        # 创建原子标志
        completed_flag = AtomicFlag(False)

        # 创建原子值
        result_value = AtomicValue(0)

        # 定义任务函数
        def increment_task(iterations):
            for _ in range(iterations):
                counter.increment()

            # 如果计数达到预期值，设置完成标志
            if counter.get() >= 1000:
                completed_flag.set_true()

                # 设置结果值
                result_value.set(counter.get())

        # 提交任务
        tasks = []
        for _ in range(10):
            task = self.thread_pool_manager.submit_task(
                increment_task,
                args=(100,),
                priority=TaskPriority.NORMAL,
                pool_type="default"
            )
            tasks.append(task)

        # 等待所有任务完成
        for task in tasks:
            task.wait(timeout=1.0)

        # 验证结果
        self.assertEqual(counter.get(), 1000)
        self.assertTrue(completed_flag.get())
        self.assertEqual(result_value.get(), 1000)

    def test_performance_metrics_export_with_thread_pool(self):
        """测试性能指标导出与线程池的集成"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # 启动性能指标收集
            self.performance_monitor.start_collection(interval=0.1)

            # 定义任务函数
            def compute_task(n):
                # 创建计数器
                counter = self.performance_monitor.get_or_create_counter(
                    f"compute_task_{n}", f"计算任务 {n}"
                )
                counter.increment()

                # 创建计时器
                timer = self.performance_monitor.get_or_create_timer(
                    f"compute_time_{n}", f"计算耗时 {n}"
                )

                # 计时执行
                with timer.time():
                    time.sleep(0.01 * n)
                    result = sum(i * i for i in range(1000 * n))

                return result

            # 提交任务
            tasks = []
            for i in range(1, 6):
                task = self.thread_pool_manager.submit_task(
                    compute_task,
                    args=(i,),
                    priority=TaskPriority.NORMAL,
                    pool_type="default"
                )
                tasks.append(task)

            # 等待所有任务完成
            for task in tasks:
                task.wait(timeout=1.0)

            # 等待收集一些系统指标
            time.sleep(0.2)

            # 导出性能指标
            json_str = self.performance_monitor.export_metrics_json(temp_path)

            # 停止性能指标收集
            self.performance_monitor.stop_collection()

            # 验证导出的JSON
            with open(temp_path) as f:
                metrics = json.load(f)

            # 验证指标
            self.assertIn("counters", metrics)
            self.assertIn("timers", metrics)
            self.assertIn("system", metrics)

            # 验证计数器
            for i in range(1, 6):
                counter_name = f"compute_task_{i}"
                self.assertIn(counter_name, metrics["counters"])
                self.assertEqual(metrics["counters"][counter_name]["value"], 1)

            # 验证计时器
            for i in range(1, 6):
                timer_name = f"compute_time_{i}"
                self.assertIn(timer_name, metrics["timers"])
                self.assertEqual(metrics["timers"][timer_name]["histogram"]["count"], 1)

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_thread_pool_stats_with_performance_monitor(self):
        """测试线程池统计信息与性能监控器的集成"""
        # 创建仪表盘指标
        active_tasks_gauge = self.performance_monitor.get_or_create_gauge(
            "active_tasks", "活动任务数"
        )

        # 创建计数器指标
        completed_tasks_counter = self.performance_monitor.get_or_create_counter(
            "completed_tasks", "已完成任务数"
        )

        # 定义任务函数
        def monitored_task(task_id, sleep_time):
            # 更新活动任务数
            stats = self.thread_pool_manager.get_stats()
            active_tasks_gauge.set(stats["active_tasks"])

            # 执行任务
            time.sleep(sleep_time)

            # 更新已完成任务数
            completed_tasks_counter.increment()

            return f"Task {task_id} completed"

        # 提交任务
        tasks = []
        for i in range(5):
            task = self.thread_pool_manager.submit_task(
                monitored_task,
                args=(i, 0.05),
                priority=TaskPriority.NORMAL,
                pool_type="default"
            )
            tasks.append(task)

        # 等待所有任务完成
        for task in tasks:
            task.wait(timeout=1.0)

        # 验证性能指标
        self.assertEqual(completed_tasks_counter.get(), 5)

        # 验证线程池统计信息
        stats = self.thread_pool_manager.get_stats()
        self.assertEqual(stats["submitted_tasks"], 5)
        self.assertEqual(stats["completed_tasks"], 5)
        self.assertEqual(stats["active_tasks"], 0)

    def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 创建计数器指标
        error_counter = self.performance_monitor.get_or_create_counter(
            "error_count", "错误计数"
        )

        # 定义会抛出异常的任务函数
        def failing_task(should_fail):
            if should_fail:
                error_counter.increment()
                raise ValueError("Task failed")
            return "Task succeeded"

        # 提交任务
        success_task = self.thread_pool_manager.submit_task(
            failing_task,
            args=(False,),
            priority=TaskPriority.NORMAL,
            pool_type="default"
        )

        fail_task = self.thread_pool_manager.submit_task(
            failing_task,
            args=(True,),
            priority=TaskPriority.NORMAL,
            pool_type="default"
        )

        # 等待任务完成
        success_task.wait(timeout=1.0)
        fail_task.wait(timeout=1.0)

        # 验证成功任务
        self.assertEqual(success_task.status.get(), TaskStatus.COMPLETED)
        self.assertEqual(success_task.get_result(), "Task succeeded")

        # 验证失败任务
        self.assertEqual(fail_task.status.get(), TaskStatus.FAILED)
        with self.assertRaises(ValueError):
            fail_task.get_result()

        # 验证错误计数
        self.assertEqual(error_counter.get(), 1)

        # 验证线程池统计信息
        stats = self.thread_pool_manager.get_stats()
        self.assertEqual(stats["submitted_tasks"], 2)
        self.assertEqual(stats["completed_tasks"], 1)
        self.assertEqual(stats["failed_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
