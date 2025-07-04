"""
原子操作工具测试
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from cascade._internal.atomic import (
    AtomicValue,
    AtomicCounter,
    AtomicReference,
    AtomicDict,
    AtomicFlag,
    AtomicLock,
    AtomicStampedReference
)


class TestAtomicValue(unittest.TestCase):
    """AtomicValue测试"""
    
    def test_get_set(self):
        """测试获取和设置值"""
        atomic = AtomicValue(10)
        self.assertEqual(atomic.get(), 10)
        
        atomic.set(20)
        self.assertEqual(atomic.get(), 20)
    
    def test_update(self):
        """测试更新值"""
        atomic = AtomicValue(10)
        
        result = atomic.update(lambda x: x * 2)
        self.assertEqual(result, 20)
        self.assertEqual(atomic.get(), 20)
    
    def test_compare_and_set(self):
        """测试比较并设置值"""
        atomic = AtomicValue(10)
        
        # 成功的比较并设置
        result = atomic.compare_and_set(10, 20)
        self.assertTrue(result)
        self.assertEqual(atomic.get(), 20)
        
        # 失败的比较并设置
        result = atomic.compare_and_set(10, 30)
        self.assertFalse(result)
        self.assertEqual(atomic.get(), 20)
    
    def test_thread_safety(self):
        """测试线程安全性"""
        atomic = AtomicValue(0)
        iterations = 1000
        thread_count = 10
        
        def increment():
            for _ in range(iterations):
                atomic.update(lambda x: x + 1)
        
        threads = []
        for _ in range(thread_count):
            thread = threading.Thread(target=increment)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.assertEqual(atomic.get(), iterations * thread_count)


class TestAtomicCounter(unittest.TestCase):
    """AtomicCounter测试"""
    
    def test_get_set(self):
        """测试获取和设置计数"""
        counter = AtomicCounter(10)
        self.assertEqual(counter.get(), 10)
        
        counter.set(20)
        self.assertEqual(counter.get(), 20)
    
    def test_increment(self):
        """测试增加计数"""
        counter = AtomicCounter(10)
        
        result = counter.increment()
        self.assertEqual(result, 11)
        self.assertEqual(counter.get(), 11)
        
        result = counter.increment(5)
        self.assertEqual(result, 16)
        self.assertEqual(counter.get(), 16)
    
    def test_decrement(self):
        """测试减少计数"""
        counter = AtomicCounter(10)
        
        result = counter.decrement()
        self.assertEqual(result, 9)
        self.assertEqual(counter.get(), 9)
        
        result = counter.decrement(5)
        self.assertEqual(result, 4)
        self.assertEqual(counter.get(), 4)
    
    def test_reset(self):
        """测试重置计数"""
        counter = AtomicCounter(10)
        
        counter.reset()
        self.assertEqual(counter.get(), 0)
    
    def test_thread_safety(self):
        """测试线程安全性"""
        counter = AtomicCounter(0)
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


class TestAtomicReference(unittest.TestCase):
    """AtomicReference测试"""
    
    def test_get_set(self):
        """测试获取和设置引用"""
        obj = {"name": "test"}
        ref = AtomicReference(obj)
        
        self.assertEqual(ref.get(), obj)
        
        new_obj = {"name": "new"}
        ref.set(new_obj)
        self.assertEqual(ref.get(), new_obj)
    
    def test_compare_and_set(self):
        """测试比较并设置引用"""
        obj1 = {"name": "test1"}
        obj2 = {"name": "test2"}
        obj3 = {"name": "test3"}
        
        ref = AtomicReference(obj1)
        
        # 成功的比较并设置
        result = ref.compare_and_set(obj1, obj2)
        self.assertTrue(result)
        self.assertEqual(ref.get(), obj2)
        
        # 失败的比较并设置
        result = ref.compare_and_set(obj1, obj3)
        self.assertFalse(result)
        self.assertEqual(ref.get(), obj2)


class TestAtomicDict(unittest.TestCase):
    """AtomicDict测试"""
    
    def test_get_set(self):
        """测试获取和设置键值对"""
        atomic_dict = AtomicDict({"key1": "value1"})
        
        self.assertEqual(atomic_dict.get("key1"), "value1")
        self.assertEqual(atomic_dict.get("key2"), None)
        self.assertEqual(atomic_dict.get("key2", "default"), "default")
        
        atomic_dict.set("key2", "value2")
        self.assertEqual(atomic_dict.get("key2"), "value2")
    
    def test_remove(self):
        """测试移除键值对"""
        atomic_dict = AtomicDict({"key1": "value1", "key2": "value2"})
        
        value = atomic_dict.remove("key1")
        self.assertEqual(value, "value1")
        self.assertEqual(atomic_dict.get("key1"), None)
        
        value = atomic_dict.remove("key3")
        self.assertEqual(value, None)
    
    def test_contains_key(self):
        """测试是否包含键"""
        atomic_dict = AtomicDict({"key1": "value1"})
        
        self.assertTrue(atomic_dict.contains_key("key1"))
        self.assertFalse(atomic_dict.contains_key("key2"))
    
    def test_clear(self):
        """测试清空字典"""
        atomic_dict = AtomicDict({"key1": "value1", "key2": "value2"})
        
        atomic_dict.clear()
        self.assertEqual(atomic_dict.size(), 0)
        self.assertEqual(atomic_dict.get("key1"), None)
    
    def test_keys_values_items(self):
        """测试获取键、值和键值对"""
        atomic_dict = AtomicDict({"key1": "value1", "key2": "value2"})
        
        self.assertEqual(set(atomic_dict.keys()), {"key1", "key2"})
        self.assertEqual(set(atomic_dict.values()), {"value1", "value2"})
        self.assertEqual(set(atomic_dict.items()), {("key1", "value1"), ("key2", "value2")})
    
    def test_size(self):
        """测试获取字典大小"""
        atomic_dict = AtomicDict({"key1": "value1", "key2": "value2"})
        
        self.assertEqual(atomic_dict.size(), 2)
        
        atomic_dict.set("key3", "value3")
        self.assertEqual(atomic_dict.size(), 3)
        
        atomic_dict.remove("key1")
        self.assertEqual(atomic_dict.size(), 2)
    
    def test_update(self):
        """测试更新字典"""
        atomic_dict = AtomicDict({"key1": "value1"})
        
        atomic_dict.update({"key2": "value2", "key3": "value3"})
        self.assertEqual(atomic_dict.size(), 3)
        self.assertEqual(atomic_dict.get("key2"), "value2")
        self.assertEqual(atomic_dict.get("key3"), "value3")
    
    def test_thread_safety(self):
        """测试线程安全性"""
        atomic_dict = AtomicDict()
        iterations = 100
        thread_count = 10
        
        def add_items():
            for i in range(iterations):
                key = f"thread-{threading.current_thread().name}-{i}"
                atomic_dict.set(key, i)
        
        threads = []
        for i in range(thread_count):
            thread = threading.Thread(target=add_items, name=f"thread-{i}")
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.assertEqual(atomic_dict.size(), iterations * thread_count)


class TestAtomicFlag(unittest.TestCase):
    """AtomicFlag测试"""
    
    def test_get_set(self):
        """测试获取和设置标志"""
        flag = AtomicFlag(False)
        self.assertFalse(flag.get())
        
        flag.set(True)
        self.assertTrue(flag.get())
    
    def test_set_true_false(self):
        """测试设置标志为True和False"""
        flag = AtomicFlag(False)
        
        old_value = flag.set_true()
        self.assertFalse(old_value)
        self.assertTrue(flag.get())
        
        old_value = flag.set_false()
        self.assertTrue(old_value)
        self.assertFalse(flag.get())
    
    def test_compare_and_set(self):
        """测试比较并设置标志"""
        flag = AtomicFlag(False)
        
        # 成功的比较并设置
        result = flag.compare_and_set(False, True)
        self.assertTrue(result)
        self.assertTrue(flag.get())
        
        # 失败的比较并设置
        result = flag.compare_and_set(False, False)
        self.assertFalse(result)
        self.assertTrue(flag.get())


class TestAtomicLock(unittest.TestCase):
    """AtomicLock测试"""
    
    def test_acquire_release(self):
        """测试获取和释放锁"""
        lock = AtomicLock()
        
        # 获取锁
        result = lock.acquire()
        self.assertTrue(result)
        self.assertTrue(lock.is_locked())
        self.assertTrue(lock.is_held_by_current_thread())
        self.assertEqual(lock.get_hold_count(), 1)
        
        # 再次获取锁（可重入）
        result = lock.acquire()
        self.assertTrue(result)
        self.assertEqual(lock.get_hold_count(), 2)
        
        # 释放锁
        lock.release()
        self.assertTrue(lock.is_locked())
        self.assertEqual(lock.get_hold_count(), 1)
        
        # 再次释放锁
        lock.release()
        self.assertFalse(lock.is_locked())
        self.assertEqual(lock.get_hold_count(), 0)
    
    def test_with_statement(self):
        """测试上下文管理器"""
        lock = AtomicLock()
        
        with lock:
            self.assertTrue(lock.is_locked())
            self.assertTrue(lock.is_held_by_current_thread())
            self.assertEqual(lock.get_hold_count(), 1)
        
        self.assertFalse(lock.is_locked())
        self.assertEqual(lock.get_hold_count(), 0)
    
    def test_thread_safety(self):
        """测试线程安全性"""
        lock = AtomicLock()
        counter = 0
        iterations = 1000
        thread_count = 10
        
        def increment():
            nonlocal counter
            for _ in range(iterations):
                with lock:
                    counter += 1
        
        threads = []
        for _ in range(thread_count):
            thread = threading.Thread(target=increment)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.assertEqual(counter, iterations * thread_count)


class TestAtomicStampedReference(unittest.TestCase):
    """AtomicStampedReference测试"""
    
    def test_get_set(self):
        """测试获取和设置引用和版本戳"""
        obj = {"name": "test"}
        ref = AtomicStampedReference(obj, 1)
        
        self.assertEqual(ref.get_reference(), obj)
        self.assertEqual(ref.get_stamp(), 1)
        
        reference, stamp = ref.get()
        self.assertEqual(reference, obj)
        self.assertEqual(stamp, 1)
        
        new_obj = {"name": "new"}
        ref.set(new_obj, 2)
        self.assertEqual(ref.get_reference(), new_obj)
        self.assertEqual(ref.get_stamp(), 2)
    
    def test_compare_and_set(self):
        """测试比较并设置引用和版本戳"""
        obj1 = {"name": "test1"}
        obj2 = {"name": "test2"}
        
        ref = AtomicStampedReference(obj1, 1)
        
        # 成功的比较并设置
        result = ref.compare_and_set(obj1, obj2, 1, 2)
        self.assertTrue(result)
        self.assertEqual(ref.get_reference(), obj2)
        self.assertEqual(ref.get_stamp(), 2)
        
        # 失败的比较并设置（引用不匹配）
        result = ref.compare_and_set(obj1, obj1, 2, 3)
        self.assertFalse(result)
        self.assertEqual(ref.get_reference(), obj2)
        self.assertEqual(ref.get_stamp(), 2)
        
        # 失败的比较并设置（版本戳不匹配）
        result = ref.compare_and_set(obj2, obj1, 1, 3)
        self.assertFalse(result)
        self.assertEqual(ref.get_reference(), obj2)
        self.assertEqual(ref.get_stamp(), 2)
    
    def test_attempt_stamp(self):
        """测试尝试更新版本戳"""
        obj = {"name": "test"}
        ref = AtomicStampedReference(obj, 1)
        
        # 成功的尝试更新版本戳
        result = ref.attempt_stamp(obj, 2)
        self.assertTrue(result)
        self.assertEqual(ref.get_stamp(), 2)
        
        # 失败的尝试更新版本戳
        new_obj = {"name": "new"}
        result = ref.attempt_stamp(new_obj, 3)
        self.assertFalse(result)
        self.assertEqual(ref.get_stamp(), 2)
    
    def test_aba_problem(self):
        """测试ABA问题的解决"""
        # 初始值A
        value_a = "A"
        value_b = "B"
        
        # 创建带版本戳的引用，初始值为A，版本戳为1
        ref = AtomicStampedReference(value_a, 1)
        
        # 线程1：获取初始值和版本戳，然后暂停
        reference1, stamp1 = ref.get()
        self.assertEqual(reference1, value_a)
        self.assertEqual(stamp1, 1)
        
        # 线程2：将值从A改为B，再改回A，版本戳递增
        ref.set(value_b, 2)
        ref.set(value_a, 3)
        
        # 线程1：尝试将值从A改为B，但由于版本戳不匹配，操作失败
        result = ref.compare_and_set(reference1, value_b, stamp1, stamp1 + 1)
        self.assertFalse(result)
        self.assertEqual(ref.get_reference(), value_a)
        self.assertEqual(ref.get_stamp(), 3)


if __name__ == "__main__":
    unittest.main()