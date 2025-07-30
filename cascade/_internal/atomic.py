"""
原子操作工具模块

提供线程安全的原子操作类，用于并发编程中的状态管理。
这些原子类确保在多线程环境下的数据一致性。

设计原则：
- 线程安全优先
- 简洁的API设计
- 高性能实现
- 明确的错误处理
"""

import threading
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar('T')

class AtomicInteger:
    """
    线程安全的原子整数类
    
    提供原子的读取、写入、递增、递减和比较交换操作。
    使用threading.Lock确保操作的原子性。
    """

    def __init__(self, initial_value: int = 0):
        """
        初始化原子整数
        
        Args:
            initial_value: 初始值，默认为0
        """
        if not isinstance(initial_value, int):
            raise TypeError("初始值必须是整数类型")

        self._value = initial_value
        self._lock = threading.Lock()

    def get(self) -> int:
        """
        原子地获取当前值
        
        Returns:
            当前整数值
        """
        with self._lock:
            return self._value

    def set(self, new_value: int) -> None:
        """
        原子地设置新值
        
        Args:
            new_value: 要设置的新值
            
        Raises:
            TypeError: 如果new_value不是整数类型
        """
        if not isinstance(new_value, int):
            raise TypeError("新值必须是整数类型")

        with self._lock:
            self._value = new_value

    def increment(self, delta: int = 1) -> int:
        """
        原子地增加值并返回新值
        
        Args:
            delta: 增加的量，默认为1
            
        Returns:
            增加后的新值
            
        Raises:
            TypeError: 如果delta不是整数类型
        """
        if not isinstance(delta, int):
            raise TypeError("增量必须是整数类型")

        with self._lock:
            self._value += delta
            return self._value

    def decrement(self, delta: int = 1) -> int:
        """
        原子地减少值并返回新值
        
        Args:
            delta: 减少的量，默认为1
            
        Returns:
            减少后的新值
            
        Raises:
            TypeError: 如果delta不是整数类型
        """
        if not isinstance(delta, int):
            raise TypeError("减量必须是整数类型")

        with self._lock:
            self._value -= delta
            return self._value

    def compare_and_set(self, expected: int, update: int) -> bool:
        """
        原子地比较并设置值（CAS操作）
        
        如果当前值等于expected，则将其设置为update。
        这是一个原子操作，要么成功要么失败。
        
        Args:
            expected: 期望的当前值
            update: 要设置的新值
            
        Returns:
            True 如果设置成功，False 如果当前值不等于期望值
            
        Raises:
            TypeError: 如果参数不是整数类型
        """
        if not isinstance(expected, int) or not isinstance(update, int):
            raise TypeError("期望值和更新值都必须是整数类型")

        with self._lock:
            if self._value == expected:
                self._value = update
                return True
            return False

    def get_and_set(self, new_value: int) -> int:
        """
        原子地获取当前值并设置新值
        
        Args:
            new_value: 要设置的新值
            
        Returns:
            之前的值
            
        Raises:
            TypeError: 如果new_value不是整数类型
        """
        if not isinstance(new_value, int):
            raise TypeError("新值必须是整数类型")

        with self._lock:
            old_value = self._value
            self._value = new_value
            return old_value

    def reset(self) -> int:
        """
        重置为0并返回之前的值
        
        Returns:
            重置前的值
        """
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value

    # 为了兼容测试，提供别名
    def compare_and_swap(self, expected: int, update: int) -> bool:
        """compare_and_set的别名"""
        return self.compare_and_set(expected, update)

    def __str__(self) -> str:
        """字符串表示"""
        return str(self.get())

    def __repr__(self) -> str:
        """调试表示"""
        return f"AtomicInteger({self.get()})"

class AtomicFloat:
    """
    线程安全的原子浮点数类
    
    提供原子的读取、写入、递增、递减操作。
    """

    def __init__(self, initial_value: float = 0.0):
        """
        初始化原子浮点数
        
        Args:
            initial_value: 初始值，默认为0.0
        """
        if not isinstance(initial_value, (int, float)):
            raise TypeError("初始值必须是数字类型")

        self._value = float(initial_value)
        self._lock = threading.Lock()

    def get(self) -> float:
        """原子地获取当前值"""
        with self._lock:
            return self._value

    def set(self, new_value: float) -> None:
        """原子地设置新值"""
        if not isinstance(new_value, (int, float)):
            raise TypeError("新值必须是数字类型")

        with self._lock:
            self._value = float(new_value)

    def add(self, delta: float) -> float:
        """原子地增加值并返回新值"""
        if not isinstance(delta, (int, float)):
            raise TypeError("增量必须是数字类型")

        with self._lock:
            self._value += delta
            return self._value

    def compare_and_set(self, expected: float, update: float, tolerance: float = 1e-10) -> bool:
        """
        原子地比较并设置值
        
        由于浮点数精度问题，使用容差进行比较。
        
        Args:
            expected: 期望的当前值
            update: 要设置的新值
            tolerance: 比较容差，默认为1e-10
            
        Returns:
            True 如果设置成功，False 否则
        """
        if not all(isinstance(x, (int, float)) for x in [expected, update, tolerance]):
            raise TypeError("所有参数都必须是数字类型")

        with self._lock:
            if abs(self._value - expected) <= tolerance:
                self._value = float(update)
                return True
            return False

    def __str__(self) -> str:
        return str(self.get())

    def __repr__(self) -> str:
        return f"AtomicFloat({self.get()})"

class AtomicBoolean:
    """
    线程安全的原子布尔类
    
    提供原子的读取、写入、翻转操作。
    """

    def __init__(self, initial_value: bool = False):
        """
        初始化原子布尔值
        
        Args:
            initial_value: 初始值，默认为False
        """
        if not isinstance(initial_value, bool):
            raise TypeError("初始值必须是布尔类型")

        self._value = initial_value
        self._lock = threading.Lock()

    def get(self) -> bool:
        """原子地获取当前值"""
        with self._lock:
            return self._value

    def set(self, new_value: bool) -> None:
        """原子地设置新值"""
        if not isinstance(new_value, bool):
            raise TypeError("新值必须是布尔类型")

        with self._lock:
            self._value = new_value

    def flip(self) -> bool:
        """原子地翻转值并返回新值"""
        with self._lock:
            self._value = not self._value
            return self._value

    def compare_and_set(self, expected: bool, update: bool) -> bool:
        """原子地比较并设置值"""
        if not isinstance(expected, bool) or not isinstance(update, bool):
            raise TypeError("期望值和更新值都必须是布尔类型")

        with self._lock:
            if self._value == expected:
                self._value = update
                return True
            return False

    def __str__(self) -> str:
        return str(self.get())

    def __repr__(self) -> str:
        return f"AtomicBoolean({self.get()})"

class AtomicReference(Generic[T]):
    """
    线程安全的原子引用类
    
    可以存储任何类型的对象引用，提供原子的读取、写入、比较交换操作。
    """

    def __init__(self, initial_value: T | None = None):
        """
        初始化原子引用
        
        Args:
            initial_value: 初始引用值，默认为None
        """
        self._value = initial_value
        self._lock = threading.Lock()

    def get(self) -> T | None:
        """原子地获取当前引用"""
        with self._lock:
            return self._value

    def set(self, new_value: T | None) -> None:
        """原子地设置新引用"""
        with self._lock:
            self._value = new_value

    def compare_and_set(self, expected: T | None, update: T | None) -> bool:
        """
        原子地比较并设置引用
        
        使用 is 操作符进行身份比较，而不是值比较。
        
        Args:
            expected: 期望的当前引用
            update: 要设置的新引用
            
        Returns:
            True 如果设置成功，False 否则
        """
        with self._lock:
            if self._value is expected:
                self._value = update
                return True
            return False

    def get_and_set(self, new_value: T | None) -> T | None:
        """原子地获取当前引用并设置新引用"""
        with self._lock:
            old_value = self._value
            self._value = new_value
            return old_value

    def update_and_get(self, updater: Callable[[T | None], T | None]) -> T | None:
        """
        原子地使用函数更新引用并返回新值
        
        Args:
            updater: 更新函数，接收当前值并返回新值
            
        Returns:
            更新后的新值
        """
        with self._lock:
            self._value = updater(self._value)
            return self._value

    def __str__(self) -> str:
        return str(self.get())

    def __repr__(self) -> str:
        return f"AtomicReference({self.get()!r})"

__all__ = [
    "AtomicInteger", "AtomicFloat", "AtomicBoolean", "AtomicReference"
]
