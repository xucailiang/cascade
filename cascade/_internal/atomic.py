"""
原子操作工具

本模块提供线程安全的原子操作工具，用于并发环境下的数据访问和修改。
"""

import threading
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar('T')


class AtomicValue(Generic[T]):
    """
    原子值容器
    
    提供线程安全的值访问和修改操作。
    """

    def __init__(self, initial_value: T):
        """
        初始化原子值
        
        Args:
            initial_value: 初始值
        """
        self._value = initial_value
        self._lock = threading.RLock()

    def get(self) -> T:
        """
        获取当前值
        
        Returns:
            当前值
        """
        with self._lock:
            return self._value

    def set(self, new_value: T) -> None:
        """
        设置新值
        
        Args:
            new_value: 新值
        """
        with self._lock:
            self._value = new_value

    def update(self, update_func: Callable[[T], T]) -> T:
        """
        原子更新值
        
        Args:
            update_func: 更新函数，接收当前值并返回新值
            
        Returns:
            更新后的值
        """
        with self._lock:
            new_value = update_func(self._value)
            self._value = new_value
            return new_value

    def compare_and_set(self, expected: T, new_value: T) -> bool:
        """
        比较并设置值
        
        只有当当前值等于预期值时才设置新值。
        
        Args:
            expected: 预期值
            new_value: 新值
            
        Returns:
            是否设置成功
        """
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

    def __str__(self) -> str:
        return f"AtomicValue({self.get()})"

    def __repr__(self) -> str:
        return self.__str__()


class AtomicCounter:
    """
    原子计数器
    
    提供线程安全的计数器操作。
    """

    def __init__(self, initial_value: int = 0):
        """
        初始化计数器
        
        Args:
            initial_value: 初始值，默认为0
        """
        self._value = initial_value
        self._lock = threading.RLock()

    def get(self) -> int:
        """
        获取当前计数
        
        Returns:
            当前计数
        """
        with self._lock:
            return self._value

    def set(self, new_value: int) -> None:
        """
        设置新计数
        
        Args:
            new_value: 新计数值
        """
        with self._lock:
            self._value = new_value

    def increment(self, delta: int = 1) -> int:
        """
        增加计数
        
        Args:
            delta: 增加量，默认为1
            
        Returns:
            增加后的计数
        """
        with self._lock:
            self._value += delta
            return self._value

    def decrement(self, delta: int = 1) -> int:
        """
        减少计数
        
        Args:
            delta: 减少量，默认为1
            
        Returns:
            减少后的计数
        """
        with self._lock:
            self._value -= delta
            return self._value

    def reset(self) -> None:
        """重置计数为0"""
        with self._lock:
            self._value = 0

    def __str__(self) -> str:
        return f"AtomicCounter({self.get()})"

    def __repr__(self) -> str:
        return self.__str__()


class AtomicReference(Generic[T]):
    """
    原子引用
    
    提供线程安全的对象引用访问和修改操作。
    """

    def __init__(self, initial_reference: T | None = None):
        """
        初始化原子引用
        
        Args:
            initial_reference: 初始引用对象，默认为None
        """
        self._reference = initial_reference
        self._lock = threading.RLock()

    def get(self) -> T | None:
        """
        获取当前引用
        
        Returns:
            当前引用对象
        """
        with self._lock:
            return self._reference

    def set(self, new_reference: T | None) -> None:
        """
        设置新引用
        
        Args:
            new_reference: 新引用对象
        """
        with self._lock:
            self._reference = new_reference

    def compare_and_set(self, expected: T | None, new_reference: T | None) -> bool:
        """
        比较并设置引用
        
        只有当当前引用等于预期引用时才设置新引用。
        
        Args:
            expected: 预期引用
            new_reference: 新引用
            
        Returns:
            是否设置成功
        """
        with self._lock:
            if self._reference == expected:
                self._reference = new_reference
                return True
            return False

    def __str__(self) -> str:
        return f"AtomicReference({self.get()})"

    def __repr__(self) -> str:
        return self.__str__()


class AtomicDict(Generic[T]):
    """
    原子字典
    
    提供线程安全的字典操作。
    """

    def __init__(self, initial_dict: dict[str, T] | None = None):
        """
        初始化原子字典
        
        Args:
            initial_dict: 初始字典，默认为空字典
        """
        self._dict = initial_dict or {}
        self._lock = threading.RLock()

    def get(self, key: str, default: T | None = None) -> T | None:
        """
        获取键对应的值
        
        Args:
            key: 键
            default: 默认值，当键不存在时返回
            
        Returns:
            键对应的值，或默认值
        """
        with self._lock:
            return self._dict.get(key, default)

    def set(self, key: str, value: T) -> None:
        """
        设置键值对
        
        Args:
            key: 键
            value: 值
        """
        with self._lock:
            self._dict[key] = value

    def remove(self, key: str) -> T | None:
        """
        移除键值对
        
        Args:
            key: 键
            
        Returns:
            被移除的值，如果键不存在则返回None
        """
        with self._lock:
            return self._dict.pop(key, None)

    def contains_key(self, key: str) -> bool:
        """
        检查是否包含键
        
        Args:
            key: 键
            
        Returns:
            是否包含键
        """
        with self._lock:
            return key in self._dict

    def clear(self) -> None:
        """清空字典"""
        with self._lock:
            self._dict.clear()

    def keys(self) -> list[str]:
        """
        获取所有键
        
        Returns:
            键列表
        """
        with self._lock:
            return list(self._dict.keys())

    def values(self) -> list[T]:
        """
        获取所有值
        
        Returns:
            值列表
        """
        with self._lock:
            return list(self._dict.values())

    def items(self) -> list[tuple[str, T]]:
        """
        获取所有键值对
        
        Returns:
            键值对列表
        """
        with self._lock:
            return list(self._dict.items())

    def size(self) -> int:
        """
        获取字典大小
        
        Returns:
            字典中键值对的数量
        """
        with self._lock:
            return len(self._dict)

    def update(self, other_dict: dict[str, T]) -> None:
        """
        更新字典
        
        Args:
            other_dict: 要合并的字典
        """
        with self._lock:
            self._dict.update(other_dict)

    def __str__(self) -> str:
        return f"AtomicDict({self._dict})"

    def __repr__(self) -> str:
        return self.__str__()


class AtomicFlag:
    """
    原子标志
    
    提供线程安全的布尔标志操作。
    """

    def __init__(self, initial_value: bool = False):
        """
        初始化原子标志
        
        Args:
            initial_value: 初始值，默认为False
        """
        self._value = initial_value
        self._lock = threading.RLock()

    def get(self) -> bool:
        """
        获取当前标志值
        
        Returns:
            当前标志值
        """
        with self._lock:
            return self._value

    def set(self, value: bool) -> None:
        """
        设置标志值
        
        Args:
            value: 新标志值
        """
        with self._lock:
            self._value = value

    def set_true(self) -> bool:
        """
        设置标志为True
        
        Returns:
            设置前的值
        """
        with self._lock:
            old_value = self._value
            self._value = True
            return old_value

    def set_false(self) -> bool:
        """
        设置标志为False
        
        Returns:
            设置前的值
        """
        with self._lock:
            old_value = self._value
            self._value = False
            return old_value

    def compare_and_set(self, expected: bool, new_value: bool) -> bool:
        """
        比较并设置标志
        
        只有当当前值等于预期值时才设置新值。
        
        Args:
            expected: 预期值
            new_value: 新值
            
        Returns:
            是否设置成功
        """
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

    def __str__(self) -> str:
        return f"AtomicFlag({self.get()})"

    def __repr__(self) -> str:
        return self.__str__()


class AtomicLock:
    """
    原子锁
    
    提供可重入的线程锁，支持超时和状态检查。
    """

    def __init__(self):
        """初始化原子锁"""
        self._lock = threading.RLock()
        self._owner = AtomicReference[threading.Thread](None)
        self._count = AtomicCounter(0)

    def acquire(self, timeout: float | None = None) -> bool:
        """
        获取锁
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            是否成功获取锁
        """
        current_thread = threading.current_thread()

        # 检查当前线程是否已经持有锁
        if self._owner.get() == current_thread:
            self._count.increment()
            return True

        # 尝试获取锁
        if self._lock.acquire(timeout=timeout):
            self._owner.set(current_thread)
            self._count.set(1)
            return True

        return False

    def release(self) -> None:
        """
        释放锁
        
        如果当前线程不持有锁，则抛出RuntimeError。
        """
        current_thread = threading.current_thread()

        if self._owner.get() != current_thread:
            raise RuntimeError("尝试释放未持有的锁")

        count = self._count.decrement()
        if count == 0:
            self._owner.set(None)
            self._lock.release()

    def is_locked(self) -> bool:
        """
        检查锁是否被持有
        
        Returns:
            锁是否被持有
        """
        return self._owner.get() is not None

    def is_held_by_current_thread(self) -> bool:
        """
        检查当前线程是否持有锁
        
        Returns:
            当前线程是否持有锁
        """
        return self._owner.get() == threading.current_thread()

    def get_hold_count(self) -> int:
        """
        获取当前线程持有锁的次数
        
        Returns:
            持有次数，如果当前线程不持有锁则返回0
        """
        if self._owner.get() == threading.current_thread():
            return self._count.get()
        return 0

    def __enter__(self) -> 'AtomicLock':
        """上下文管理器入口"""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.release()


class AtomicStampedReference(Generic[T]):
    """
    带版本戳的原子引用
    
    提供线程安全的对象引用访问和修改操作，同时跟踪版本戳以防止ABA问题。
    """

    def __init__(self, initial_reference: T | None = None, initial_stamp: int = 0):
        """
        初始化带版本戳的原子引用
        
        Args:
            initial_reference: 初始引用对象，默认为None
            initial_stamp: 初始版本戳，默认为0
        """
        self._reference = initial_reference
        self._stamp = initial_stamp
        self._lock = threading.RLock()

    def get_reference(self) -> T | None:
        """
        获取当前引用
        
        Returns:
            当前引用对象
        """
        with self._lock:
            return self._reference

    def get_stamp(self) -> int:
        """
        获取当前版本戳
        
        Returns:
            当前版本戳
        """
        with self._lock:
            return self._stamp

    def get(self) -> tuple[T | None, int]:
        """
        获取当前引用和版本戳
        
        Returns:
            当前引用对象和版本戳的元组
        """
        with self._lock:
            return (self._reference, self._stamp)

    def set(self, new_reference: T | None, new_stamp: int) -> None:
        """
        设置新引用和版本戳
        
        Args:
            new_reference: 新引用对象
            new_stamp: 新版本戳
        """
        with self._lock:
            self._reference = new_reference
            self._stamp = new_stamp

    def compare_and_set(self, expected_reference: T | None, new_reference: T | None,
                        expected_stamp: int, new_stamp: int) -> bool:
        """
        比较并设置引用和版本戳
        
        只有当当前引用等于预期引用且当前版本戳等于预期版本戳时才设置新引用和版本戳。
        
        Args:
            expected_reference: 预期引用
            new_reference: 新引用
            expected_stamp: 预期版本戳
            new_stamp: 新版本戳
            
        Returns:
            是否设置成功
        """
        with self._lock:
            if self._reference == expected_reference and self._stamp == expected_stamp:
                self._reference = new_reference
                self._stamp = new_stamp
                return True
            return False

    def attempt_stamp(self, expected_reference: T | None, new_stamp: int) -> bool:
        """
        尝试更新版本戳
        
        只有当当前引用等于预期引用时才更新版本戳。
        
        Args:
            expected_reference: 预期引用
            new_stamp: 新版本戳
            
        Returns:
            是否更新成功
        """
        with self._lock:
            if self._reference == expected_reference:
                self._stamp = new_stamp
                return True
            return False

    def __str__(self) -> str:
        return f"AtomicStampedReference(reference={self._reference}, stamp={self._stamp})"

    def __repr__(self) -> str:
        return self.__str__()


# 导出的类
__all__ = [
    "AtomicValue",
    "AtomicCounter",
    "AtomicReference",
    "AtomicDict",
    "AtomicFlag",
    "AtomicLock",
    "AtomicStampedReference",
]
