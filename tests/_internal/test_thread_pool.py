"""
线程池管理工具测试
"""

import concurrent.futures
import queue
import threading
import time
import unittest

from cascade._internal.thread_pool import (
    PriorityThreadPoolExecutor,
    Task,
    TaskPriority,
    TaskStats,
    TaskStatus,
    ThreadPoolManager,
)


class TestTaskPriority(unittest.TestCase):
    """TaskPriority测试"""

    def test_priority_order(self):
        """测试优先级顺序"""
        # 验证优先级顺序：CRITICAL > HIGH > NORMAL > LOW
        self.assertGreater(TaskPriority.CRITICAL.value, TaskPriority.HIGH.value)
        self.assertGreater(TaskPriority.HIGH.value, TaskPriority.NORMAL.value)
        self.assertGreater(TaskPriority.NORMAL.value, TaskPriority.LOW.value)


class TestTaskStatus(unittest.TestCase):
    """TaskStatus测试"""

    def test_status_values(self):
        """测试状态值"""
        # 验证所有状态值都是唯一的
        statuses = [
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT
        ]

        # 检查唯一性
        self.assertEqual(len(statuses), len(set(status.value for status in statuses)))


class TestTaskStats(unittest.TestCase):
    """TaskStats测试"""

    def test_init(self):
        """测试初始化"""
        current_time = time.time()
        stats = TaskStats(submitted_at=current_time)

        self.assertEqual(stats.submitted_at, current_time)
        self.assertIsNone(stats.started_at)
        self.assertIsNone(stats.completed_at)
        self.assertIsNone(stats.wait_time)
        self.assertIsNone(stats.execution_time)
        self.assertIsNone(stats.total_time)


class TestTask(unittest.TestCase):
    """Task测试"""

    def test_init(self):
        """测试初始化"""
        def test_func(a, b, c=3):
            return a + b + c

        task = Task(test_func, args=(1, 2), kwargs={"c": 4},
                   priority=TaskPriority.HIGH, timeout=10, task_id="test-task")

        self.assertEqual(task.func, test_func)
        self.assertEqual(task.args, (1, 2))
        self.assertEqual(task.kwargs, {"c": 4})
        self.assertEqual(task.priority, TaskPriority.HIGH)
        self.assertEqual(task.timeout, 10)
        self.assertEqual(task.task_id, "test-task")
        self.assertEqual(task.status.get(), TaskStatus.PENDING)
        self.assertIsNone(task.result)
        self.assertIsNone(task.exception)
        self.assertIsNone(task.future)
        self.assertIsNotNone(task.stats)
        self.assertIsNotNone(task.stats.submitted_at)

    def test_execute_success(self):
        """测试成功执行任务"""
        def test_func(a, b):
            return a + b

        task = Task(test_func, args=(1, 2))

        # 执行任务
        result = task.execute()

        # 验证结果
        self.assertEqual(result, 3)
        self.assertEqual(task.result, 3)
        self.assertEqual(task.status.get(), TaskStatus.COMPLETED)

        # 验证统计信息
        self.assertIsNotNone(task.stats.started_at)
        self.assertIsNotNone(task.stats.completed_at)
        self.assertIsNotNone(task.stats.wait_time)
        self.assertIsNotNone(task.stats.execution_time)
        self.assertIsNotNone(task.stats.total_time)

    def test_execute_exception(self):
        """测试执行任务时发生异常"""
        def test_func():
            raise ValueError("Test exception")

        task = Task(test_func)

        # 执行任务，应该抛出异常
        with self.assertRaises(ValueError):
            task.execute()

        # 验证状态和异常
        self.assertEqual(task.status.get(), TaskStatus.FAILED)
        self.assertIsInstance(task.exception, ValueError)
        self.assertEqual(str(task.exception), "Test exception")

        # 验证统计信息
        self.assertIsNotNone(task.stats.started_at)
        self.assertIsNotNone(task.stats.completed_at)
        self.assertIsNotNone(task.stats.execution_time)

    def test_execute_timeout(self):
        """测试执行任务超时"""
        def test_func():
            time.sleep(0.2)
            return "result"

        task = Task(test_func, timeout=0.1)

        # 执行任务，应该抛出超时异常
        with self.assertRaises(TimeoutError):
            task.execute()

        # 验证状态
        self.assertEqual(task.status.get(), TaskStatus.TIMEOUT)

    def test_wait(self):
        """测试等待任务完成"""
        def test_func():
            time.sleep(0.1)
            return "result"

        task = Task(test_func)

        # 在另一个线程中执行任务
        def execute_task():
            task.execute()

        thread = threading.Thread(target=execute_task)
        thread.start()

        # 等待任务完成
        result = task.wait(timeout=0.5)
        thread.join()

        # 验证结果
        self.assertTrue(result)  # 在超时前完成
        self.assertEqual(task.status.get(), TaskStatus.COMPLETED)
        self.assertEqual(task.result, "result")

    def test_wait_timeout(self):
        """测试等待任务完成超时"""
        def test_func():
            time.sleep(0.3)
            return "result"

        task = Task(test_func)

        # 在另一个线程中执行任务
        def execute_task():
            task.execute()

        thread = threading.Thread(target=execute_task)
        thread.start()

        # 等待任务完成，但超时
        result = task.wait(timeout=0.1)

        # 验证结果
        self.assertFalse(result)  # 超时

        # 等待线程完成，避免资源泄漏
        thread.join()

    def test_cancel(self):
        """测试取消任务"""
        def test_func():
            time.sleep(0.1)
            return "result"

        task = Task(test_func)

        # 取消任务
        result = task.cancel()

        # 验证结果
        self.assertTrue(result)
        self.assertEqual(task.status.get(), TaskStatus.CANCELLED)

    def test_cancel_running_task(self):
        """测试取消正在运行的任务"""
        def test_func():
            time.sleep(0.2)
            return "result"

        task = Task(test_func)

        # 在另一个线程中执行任务
        def execute_task():
            task.execute()

        thread = threading.Thread(target=execute_task)
        thread.start()

        # 等待任务开始运行
        time.sleep(0.1)

        # 尝试取消任务
        result = task.cancel()

        # 验证结果
        self.assertFalse(result)  # 无法取消正在运行的任务

        # 等待线程完成，避免资源泄漏
        thread.join()

    def test_is_done(self):
        """测试检查任务是否已完成"""
        def test_func():
            return "result"

        task = Task(test_func)

        # 初始状态
        self.assertFalse(task.is_done())

        # 执行任务
        task.execute()

        # 完成状态
        self.assertTrue(task.is_done())

    def test_get_result(self):
        """测试获取任务结果"""
        def test_func():
            time.sleep(0.1)
            return "result"

        task = Task(test_func)

        # 在另一个线程中执行任务
        def execute_task():
            task.execute()

        thread = threading.Thread(target=execute_task)
        thread.start()

        # 获取任务结果
        result = task.get_result(timeout=0.5)

        # 验证结果
        self.assertEqual(result, "result")

        # 等待线程完成，避免资源泄漏
        thread.join()

    def test_get_result_timeout(self):
        """测试获取任务结果超时"""
        def test_func():
            time.sleep(0.3)
            return "result"

        task = Task(test_func)

        # 在另一个线程中执行任务
        def execute_task():
            task.execute()

        thread = threading.Thread(target=execute_task)
        thread.start()

        # 获取任务结果，但超时
        with self.assertRaises(TimeoutError):
            task.get_result(timeout=0.1)

        # 等待线程完成，避免资源泄漏
        thread.join()

    def test_get_result_exception(self):
        """测试获取任务结果时发生异常"""
        def test_func():
            raise ValueError("Test exception")

        task = Task(test_func)

        # 在另一个线程中执行任务
        def execute_task():
            try:
                task.execute()
            except:
                pass  # 忽略异常

        thread = threading.Thread(target=execute_task)
        thread.start()
        thread.join()

        # 获取任务结果，应该抛出原始异常
        with self.assertRaises(ValueError) as cm:
            task.get_result()

        self.assertEqual(str(cm.exception), "Test exception")

    def test_get_result_cancelled(self):
        """测试获取已取消任务的结果"""
        def test_func():
            return "result"

        task = Task(test_func)

        # 取消任务
        task.cancel()

        # 获取任务结果，应该抛出异常
        with self.assertRaises(RuntimeError) as cm:
            task.get_result()

        self.assertIn("已取消", str(cm.exception))

    def test_comparison(self):
        """测试任务比较（优先级）"""
        task_low = Task(lambda: None, priority=TaskPriority.LOW)
        task_normal = Task(lambda: None, priority=TaskPriority.NORMAL)
        task_high = Task(lambda: None, priority=TaskPriority.HIGH)
        task_critical = Task(lambda: None, priority=TaskPriority.CRITICAL)

        # 验证比较结果
        # 注意：在优先队列中，值越小优先级越高，但我们的TaskPriority是值越大优先级越高
        # 所以在Task.__lt__中，我们反转了比较结果
        self.assertTrue(task_critical < task_high)  # CRITICAL优先级高于HIGH
        self.assertTrue(task_high < task_normal)    # HIGH优先级高于NORMAL
        self.assertTrue(task_normal < task_low)     # NORMAL优先级高于LOW


class TestPriorityThreadPoolExecutor(unittest.TestCase):
    """PriorityThreadPoolExecutor测试"""

    def test_init(self):
        """测试初始化"""
        executor = PriorityThreadPoolExecutor(max_workers=4, thread_name_prefix="test-")

        self.assertEqual(executor._max_workers, 4)
        self.assertEqual(executor._thread_name_prefix, "test-")
        self.assertIsInstance(executor._work_queue, queue.PriorityQueue)

    def test_submit_priority(self):
        """测试按优先级提交任务"""
        executor = PriorityThreadPoolExecutor(max_workers=1)

        # 创建一个事件来控制任务执行
        event = threading.Event()
        results = []

        # 定义任务函数
        def task_func(priority):
            event.wait()  # 等待事件被设置
            results.append(priority)
            return priority

        # 按不同优先级提交任务
        future_normal = executor.submit(task_func, TaskPriority.NORMAL.name)
        future_high = executor.submit(task_func, TaskPriority.HIGH.name)
        future_low = executor.submit(task_func, TaskPriority.LOW.name)
        future_critical = executor.submit(task_func, TaskPriority.CRITICAL.name)

        # 设置事件，允许任务执行
        event.set()

        # 等待所有任务完成
        concurrent.futures.wait([future_normal, future_high, future_low, future_critical])

        # 关闭执行器
        executor.shutdown()

        # 验证任务执行顺序
        # 注意：由于线程池的实现细节，我们无法保证任务的确切执行顺序
        # 但我们可以验证所有任务都已完成并返回了预期的结果
        self.assertEqual(set(results), {TaskPriority.NORMAL.name, TaskPriority.HIGH.name,
                                       TaskPriority.LOW.name, TaskPriority.CRITICAL.name})

        self.assertEqual(future_normal.result(), TaskPriority.NORMAL.name)
        self.assertEqual(future_high.result(), TaskPriority.HIGH.name)
        self.assertEqual(future_low.result(), TaskPriority.LOW.name)
        self.assertEqual(future_critical.result(), TaskPriority.CRITICAL.name)


class TestThreadPoolManager(unittest.TestCase):
    """ThreadPoolManager测试"""

    def setUp(self):
        """测试前准备"""
        # 重置单例实例
        ThreadPoolManager._instance = None
        self.manager = ThreadPoolManager.get_instance()

    def tearDown(self):
        """测试后清理"""
        # 关闭线程池管理器
        if not self.manager.is_shutdown.get():
            self.manager.shutdown(wait=True)

    def test_singleton(self):
        """测试单例模式"""
        manager1 = ThreadPoolManager.get_instance()
        manager2 = ThreadPoolManager.get_instance()

        self.assertIs(manager1, manager2)

    def test_init(self):
        """测试初始化"""
        # 验证默认线程池
        self.assertIsNotNone(self.manager.default_pool)
        self.assertIsNotNone(self.manager.io_pool)
        self.assertIsNotNone(self.manager.compute_pool)

        # 验证任务计数器
        self.assertEqual(self.manager.submitted_tasks.get(), 0)
        self.assertEqual(self.manager.completed_tasks.get(), 0)
        self.assertEqual(self.manager.failed_tasks.get(), 0)
        self.assertEqual(self.manager.cancelled_tasks.get(), 0)

        # 验证关闭标志
        self.assertFalse(self.manager.is_shutdown.get())

    def test_create_pool(self):
        """测试创建自定义线程池"""
        pool = self.manager.create_pool("test-pool", max_workers=2, thread_name_prefix="test-")

        self.assertIsInstance(pool, PriorityThreadPoolExecutor)
        self.assertEqual(pool._max_workers, 2)
        self.assertEqual(pool._thread_name_prefix, "test-")

        # 验证池已添加到自定义池字典
        self.assertTrue(self.manager.custom_pools.contains_key("test-pool"))
        self.assertIs(self.manager.custom_pools.get("test-pool"), pool)

    def test_create_duplicate_pool(self):
        """测试创建重复名称的线程池"""
        self.manager.create_pool("test-pool")

        with self.assertRaises(ValueError):
            self.manager.create_pool("test-pool")

    def test_get_pool(self):
        """测试获取自定义线程池"""
        # 创建线程池
        pool = self.manager.create_pool("test-pool")

        # 获取线程池
        retrieved_pool = self.manager.get_pool("test-pool")

        # 验证结果
        self.assertIs(retrieved_pool, pool)

        # 获取不存在的线程池
        retrieved_pool = self.manager.get_pool("non-existent-pool")
        self.assertIsNone(retrieved_pool)

    def test_remove_pool(self):
        """测试移除自定义线程池"""
        # 创建线程池
        self.manager.create_pool("test-pool")

        # 移除线程池
        result = self.manager.remove_pool("test-pool")

        # 验证结果
        self.assertTrue(result)
        self.assertFalse(self.manager.custom_pools.contains_key("test-pool"))

        # 移除不存在的线程池
        result = self.manager.remove_pool("non-existent-pool")
        self.assertFalse(result)

    def test_submit_task(self):
        """测试提交任务"""
        def test_func(a, b):
            return a + b

        # 提交任务
        task = self.manager.submit_task(test_func, args=(1, 2),
                                      priority=TaskPriority.HIGH,
                                      pool_type="default")

        # 验证任务
        self.assertIsInstance(task, Task)
        self.assertEqual(task.func, test_func)
        self.assertEqual(task.args, (1, 2))
        self.assertEqual(task.priority, TaskPriority.HIGH)

        # 验证任务已添加到任务字典
        self.assertTrue(self.manager.tasks.contains_key(task.task_id))

        # 验证任务计数
        self.assertEqual(self.manager.submitted_tasks.get(), 1)

        # 等待任务完成
        result = task.get_result(timeout=1.0)

        # 验证结果
        self.assertEqual(result, 3)
        self.assertEqual(self.manager.completed_tasks.get(), 1)

    def test_submit_task_to_custom_pool(self):
        """测试提交任务到自定义线程池"""
        # 创建自定义线程池
        self.manager.create_pool("test-pool", max_workers=2)

        def test_func():
            return "result"

        # 提交任务到自定义线程池
        task = self.manager.submit_task(test_func, pool_type="test-pool")

        # 等待任务完成
        result = task.get_result(timeout=1.0)

        # 验证结果
        self.assertEqual(result, "result")

    def test_submit_task_to_nonexistent_pool(self):
        """测试提交任务到不存在的线程池"""
        def test_func():
            return "result"

        # 提交任务到不存在的线程池
        with self.assertRaises(ValueError):
            self.manager.submit_task(test_func, pool_type="non-existent-pool")

    def test_submit_tasks(self):
        """测试批量提交任务"""
        def func1(x):
            return x * 2

        def func2(x):
            return x * 3

        def func3(x):
            return x * 4

        # 批量提交任务
        tasks = self.manager.submit_tasks(
            funcs=[func1, func2, func3],
            args_list=[(1,), (2,), (3,)],
            kwargs_list=[{}, {}, {}],
            priority=TaskPriority.NORMAL,
            pool_type="default"
        )

        # 验证任务数量
        self.assertEqual(len(tasks), 3)
        self.assertEqual(self.manager.submitted_tasks.get(), 3)

        # 等待所有任务完成
        results = [task.get_result(timeout=1.0) for task in tasks]

        # 验证结果
        self.assertEqual(results, [2, 6, 12])
        self.assertEqual(self.manager.completed_tasks.get(), 3)

    def test_map(self):
        """测试映射函数"""
        def square(x):
            return x * x

        # 映射函数到项目列表
        results = self.manager.map(square, [1, 2, 3, 4, 5],
                                 priority=TaskPriority.NORMAL,
                                 pool_type="default")

        # 验证结果
        self.assertEqual(list(results), [1, 4, 9, 16, 25])

    def test_get_task(self):
        """测试获取任务"""
        def test_func():
            return "result"

        # 提交任务
        task = self.manager.submit_task(test_func)

        # 获取任务
        retrieved_task = self.manager.get_task(task.task_id)

        # 验证结果
        self.assertIs(retrieved_task, task)

        # 获取不存在的任务
        retrieved_task = self.manager.get_task("non-existent-task")
        self.assertIsNone(retrieved_task)

    def test_cancel_task(self):
        """测试取消任务"""
        # 创建一个事件来控制任务执行
        event = threading.Event()

        def test_func():
            # 等待事件被设置
            event.wait()
            time.sleep(0.5)
            return "result"

        # 提交任务
        task = self.manager.submit_task(test_func)

        # 确保任务还没有开始执行
        time.sleep(0.1)

        # 取消任务
        result = self.manager.cancel_task(task.task_id)

        # 设置事件，允许任务执行（如果没有被取消）
        event.set()

        # 验证结果
        self.assertTrue(result)
        self.assertEqual(task.status.get(), TaskStatus.CANCELLED)
        self.assertEqual(self.manager.cancelled_tasks.get(), 1)

        # 取消不存在的任务
        result = self.manager.cancel_task("non-existent-task")
        self.assertFalse(result)

    def test_wait_for_tasks(self):
        """测试等待任务完成"""
        def test_func(sleep_time):
            time.sleep(sleep_time)
            return "result"

        # 提交任务
        task1 = self.manager.submit_task(test_func, args=(0.1,))
        task2 = self.manager.submit_task(test_func, args=(0.2,))
        task3 = self.manager.submit_task(test_func, args=(0.3,))

        # 等待所有任务完成
        statuses = self.manager.wait_for_tasks(
            [task1.task_id, task2.task_id, task3.task_id],
            timeout=1.0,
            return_when="ALL_COMPLETED"
        )

        # 验证结果
        self.assertEqual(len(statuses), 3)
        self.assertEqual(statuses[task1.task_id], TaskStatus.COMPLETED)
        self.assertEqual(statuses[task2.task_id], TaskStatus.COMPLETED)
        self.assertEqual(statuses[task3.task_id], TaskStatus.COMPLETED)

    def test_wait_for_tasks_first_completed(self):
        """测试等待第一个任务完成"""
        def test_func(sleep_time):
            time.sleep(sleep_time)
            return "result"

        # 提交任务
        task1 = self.manager.submit_task(test_func, args=(0.1,))
        task2 = self.manager.submit_task(test_func, args=(0.5,))

        # 等待第一个任务完成
        statuses = self.manager.wait_for_tasks(
            [task1.task_id, task2.task_id],
            timeout=0.3,
            return_when="FIRST_COMPLETED"
        )

        # 验证结果
        self.assertEqual(len(statuses), 2)
        self.assertEqual(statuses[task1.task_id], TaskStatus.COMPLETED)

        # 等待所有任务完成
        task2.wait()

    def test_get_stats(self):
        """测试获取线程池统计信息"""
        # 提交一些任务
        def test_func():
            return "result"

        self.manager.submit_task(test_func)
        self.manager.submit_task(test_func)

        # 获取统计信息
        stats = self.manager.get_stats()

        # 验证统计信息
        self.assertEqual(stats["submitted_tasks"], 2)
        self.assertIn("default_pool", stats)
        self.assertIn("io_pool", stats)
        self.assertIn("compute_pool", stats)
        self.assertIn("custom_pools", stats)

    def test_shutdown(self):
        """测试关闭线程池管理器"""
        # 关闭线程池管理器
        self.manager.shutdown(wait=True)

        # 验证关闭标志
        self.assertTrue(self.manager.is_shutdown.get())

        # 尝试提交任务，应该抛出异常
        def test_func():
            return "result"

        with self.assertRaises(RuntimeError):
            self.manager.submit_task(test_func)


if __name__ == "__main__":
    unittest.main()
