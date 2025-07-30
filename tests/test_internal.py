"""
Cascade 内部工具模块单元测试

测试原子操作、性能优化工具和内部实用函数
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

from cascade._internal import (
    AtomicCounter,
    check_cache_line_alignment,
    ensure_memory_alignment,
    measure_performance,
)


class TestAtomicCounter:
    """原子计数器测试"""

    def test_basic_operations(self):
        """测试基本操作"""
        counter = AtomicCounter()

        # 测试初始值
        assert counter.get() == 0

        # 测试设置值
        counter.set(10)
        assert counter.get() == 10

        # 测试递增
        result = counter.increment()
        assert result == 11
        assert counter.get() == 11

        # 测试带步长的递增
        result = counter.increment(5)
        assert result == 16
        assert counter.get() == 16

        # 测试递减
        result = counter.decrement()
        assert result == 15
        assert counter.get() == 15

        # 测试带步长的递减
        result = counter.decrement(3)
        assert result == 12
        assert counter.get() == 12

    def test_compare_and_swap(self):
        """测试比较并交换操作"""
        counter = AtomicCounter(10)

        # 成功的CAS
        success = counter.compare_and_swap(10, 20)
        assert success
        assert counter.get() == 20

        # 失败的CAS
        success = counter.compare_and_swap(10, 30)
        assert not success
        assert counter.get() == 20  # 值不应该改变

    def test_reset(self):
        """测试重置操作"""
        counter = AtomicCounter(100)
        assert counter.get() == 100

        old_value = counter.reset()
        assert old_value == 100
        assert counter.get() == 0

    def test_thread_safety(self):
        """测试线程安全性"""
        counter = AtomicCounter()
        num_threads = 10
        increments_per_thread = 1000
        expected_total = num_threads * increments_per_thread

        def increment_worker():
            """工作线程函数"""
            for _ in range(increments_per_thread):
                counter.increment()

        # 启动多个线程同时进行递增操作
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert counter.get() == expected_total

    def test_concurrent_operations(self):
        """测试并发混合操作"""
        counter = AtomicCounter(1000)

        def mixed_operations():
            """混合操作函数"""
            results = []
            for _ in range(100):
                # 随机执行不同操作
                import random
                op = random.choice(['inc', 'dec', 'set', 'cas'])

                if op == 'inc':
                    results.append(counter.increment())
                elif op == 'dec':
                    results.append(counter.decrement())
                elif op == 'set':
                    counter.set(random.randint(0, 1000))
                else:  # cas
                    old_val = counter.get()
                    counter.compare_and_swap(old_val, random.randint(0, 1000))

            return results

        # 使用线程池执行并发操作
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(mixed_operations) for _ in range(5)]

            # 等待所有任务完成
            for future in as_completed(futures):
                results = future.result()
                assert isinstance(results, list)

        # 确保计数器仍然可以正常工作
        initial_value = counter.get()
        counter.increment()
        assert counter.get() == initial_value + 1


class TestMemoryAlignment:
    """内存对齐测试"""

    def test_ensure_memory_alignment(self):
        """测试内存对齐函数"""
        # 测试不同大小的数组
        sizes = [100, 1000, 10000]

        for size in sizes:
            # 创建未对齐的数组
            unaligned_array = np.random.random(size).astype(np.float32)

            # 确保内存对齐
            aligned_array = ensure_memory_alignment(unaligned_array)

            # 验证数组内容相同
            np.testing.assert_array_equal(unaligned_array, aligned_array)

            # 验证类型保持不变
            assert aligned_array.dtype == unaligned_array.dtype

            # 验证形状保持不变
            assert aligned_array.shape == unaligned_array.shape

    def test_different_dtypes(self):
        """测试不同数据类型的对齐"""
        dtypes = [np.float32, np.float64, np.int16, np.int32]

        for dtype in dtypes:
            array = np.random.random(1000).astype(dtype)
            aligned_array = ensure_memory_alignment(array)

            assert aligned_array.dtype == dtype
            np.testing.assert_array_equal(array, aligned_array)

    def test_multidimensional_arrays(self):
        """测试多维数组对齐"""
        shapes = [(10, 10), (5, 20, 4), (2, 3, 4, 5)]

        for shape in shapes:
            array = np.random.random(shape).astype(np.float32)
            aligned_array = ensure_memory_alignment(array)

            assert aligned_array.shape == shape
            np.testing.assert_array_equal(array, aligned_array)

    def test_cache_line_alignment_check(self):
        """测试缓存行对齐检查"""
        # 创建已对齐的数组
        aligned_array = np.random.random(1000).astype(np.float32)
        aligned_array = ensure_memory_alignment(aligned_array)

        # 检查对齐状态（应该返回结果而不抛出异常）
        is_aligned = check_cache_line_alignment(aligned_array)
        assert isinstance(is_aligned, bool)

        # 测试不同大小的数组
        for size in [64, 128, 256, 512, 1024]:
            test_array = np.zeros(size, dtype=np.float32)
            test_array = ensure_memory_alignment(test_array)
            result = check_cache_line_alignment(test_array)
            assert isinstance(result, bool)


class TestPerformanceMeasurement:
    """性能测量测试"""

    def test_measure_performance_basic(self):
        """测试基本性能测量"""

        def test_function():
            """测试函数"""
            # 模拟一些计算
            time.sleep(0.01)  # 10ms
            return "result"

        # 测量性能
        result, duration_ms = measure_performance(test_function)

        assert result == "result"
        assert duration_ms >= 10.0  # 至少10ms
        assert duration_ms < 50.0   # 不应该超过50ms（考虑系统误差）

    def test_measure_performance_with_args(self):
        """测试带参数的性能测量"""

        def add_numbers(a, b, c=0):
            """测试函数"""
            time.sleep(0.005)  # 5ms
            return a + b + c

        result, duration_ms = measure_performance(
            add_numbers,
            args=(10, 20),
            kwargs={'c': 5}
        )

        assert result == 35
        assert duration_ms >= 5.0

    def test_measure_performance_exception(self):
        """测试异常情况下的性能测量"""

        def failing_function():
            """会抛出异常的函数"""
            time.sleep(0.005)
            raise ValueError("测试异常")

        # 应该重新抛出异常，但仍然测量时间
        with pytest.raises(ValueError, match="测试异常"):
            measure_performance(failing_function)

    def test_measure_performance_no_args(self):
        """测试无参数函数的性能测量"""

        def simple_function():
            """简单函数"""
            return 42

        result, duration_ms = measure_performance(simple_function)

        assert result == 42
        assert duration_ms >= 0
        assert isinstance(duration_ms, float)

    def test_performance_consistency(self):
        """测试性能测量的一致性"""

        def consistent_function():
            """执行时间相对固定的函数"""
            time.sleep(0.01)
            return True

        # 多次测量
        measurements = []
        for _ in range(5):
            _, duration = measure_performance(consistent_function)
            measurements.append(duration)

        # 检查测量结果的一致性（变异系数应该较小）
        import statistics
        mean_duration = statistics.mean(measurements)
        std_duration = statistics.stdev(measurements)

        # 变异系数应该小于0.5（50%）
        coefficient_of_variation = std_duration / mean_duration
        assert coefficient_of_variation < 0.5


class TestModuleIntegration:
    """模块集成测试"""

    def test_atomic_counter_with_performance_measurement(self):
        """测试原子计数器与性能测量的集成"""
        counter = AtomicCounter()

        def increment_1000_times():
            """递增1000次"""
            for _ in range(1000):
                counter.increment()
            return counter.get()

        final_value, duration_ms = measure_performance(increment_1000_times)

        assert final_value == 1000
        assert duration_ms > 0
        assert counter.get() == 1000

    def test_memory_alignment_performance(self):
        """测试内存对齐对性能的影响"""

        # 创建大数组
        size = 100000
        unaligned_array = np.random.random(size).astype(np.float32)
        aligned_array = ensure_memory_alignment(unaligned_array)

        def process_array(arr):
            """处理数组"""
            return np.sum(arr ** 2)

        # 测量未对齐数组的处理时间
        _, unaligned_duration = measure_performance(
            process_array, args=(unaligned_array,)
        )

        # 测量对齐数组的处理时间
        _, aligned_duration = measure_performance(
            process_array, args=(aligned_array,)
        )

        # 两个测量都应该成功
        assert unaligned_duration > 0
        assert aligned_duration > 0

        # 验证结果相同
        unaligned_result = process_array(unaligned_array)
        aligned_result = process_array(aligned_array)
        np.testing.assert_almost_equal(unaligned_result, aligned_result)

    def test_concurrent_memory_operations(self):
        """测试并发内存操作"""
        counter = AtomicCounter()

        def memory_intensive_task():
            """内存密集型任务"""
            # 创建并对齐数组
            array = np.random.random(10000).astype(np.float32)
            aligned_array = ensure_memory_alignment(array)

            # 执行计算
            result = np.sum(aligned_array)

            # 更新计数器
            counter.increment()

            return result

        # 并发执行多个内存密集型任务
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(memory_intensive_task)
                for _ in range(10)
            ]

            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # 验证所有任务都完成了
        assert len(results) == 10
        assert counter.get() == 10

        # 验证所有结果都是有效的数值
        for result in results:
            assert isinstance(result, (int, float, np.number))
            assert not np.isnan(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
