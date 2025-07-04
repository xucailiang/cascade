"""
内部工具函数

本模块提供各种内部工具函数，用于简化常见操作。
"""

import os
import sys
import time
import json
import logging
import hashlib
import uuid
import tempfile
import shutil
import platform
import subprocess
import importlib
import inspect
import re
import traceback
import functools
import contextlib
import warnings
from typing import (
    Any, Dict, List, Tuple, Set, Optional, Union, Callable, 
    TypeVar, Generic, Iterator, Iterable, Generator, Type, cast
)
from pathlib import Path
from datetime import datetime, timedelta
import threading
import io
import base64
import zlib
import gzip
import pickle
import urllib.request
import urllib.parse
import urllib.error
import socket
import struct
import random
import string
import math
import copy

# 类型变量
T = TypeVar('T')
R = TypeVar('R')

# 配置日志
logger = logging.getLogger("cascade.utils")


class Singleton(type):
    """单例元类"""
    
    _instances = {}
    _lock = threading.RLock()
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            return cls._instances[cls]


class LazyProperty:
    """延迟加载属性装饰器"""
    
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        value = self.func(instance)
        setattr(instance, self.func.__name__, value)
        return value


class InternalUtils:
    """内部工具函数集合"""
    
    @staticmethod
    def setup_logging(
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        log_format: Optional[str] = None,
        date_format: Optional[str] = None,
        console: bool = True,
        file_level: Optional[int] = None,
        console_level: Optional[int] = None,
        capture_warnings: bool = True
    ) -> logging.Logger:
        """
        设置日志
        
        Args:
            level: 全局日志级别
            log_file: 日志文件路径
            log_format: 日志格式
            date_format: 日期格式
            console: 是否输出到控制台
            file_level: 文件日志级别，默认与全局级别相同
            console_level: 控制台日志级别，默认与全局级别相同
            capture_warnings: 是否捕获警告
            
        Returns:
            根日志记录器
        """
        # 设置默认格式
        if log_format is None:
            log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"
        
        # 获取根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(log_format, date_format)
        
        # 添加控制台处理器
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(console_level or level)
            root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(file_level or level)
            root_logger.addHandler(file_handler)
        
        # 捕获警告
        if capture_warnings:
            logging.captureWarnings(True)
        
        return root_logger
    
    @staticmethod
    def get_timestamp() -> float:
        """
        获取当前时间戳
        
        Returns:
            当前时间戳（秒）
        """
        return time.time()
    
    @staticmethod
    def get_iso_timestamp() -> str:
        """
        获取ISO格式的当前时间戳
        
        Returns:
            ISO格式的当前时间戳
        """
        return datetime.now().isoformat()
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        格式化时间
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化后的时间字符串
        """
        if seconds < 0.001:
            return f"{seconds * 1000000:.2f} µs"
        elif seconds < 1:
            return f"{seconds * 1000:.2f} ms"
        elif seconds < 60:
            return f"{seconds:.2f} s"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds %= 60
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            hours = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """
        格式化文件大小
        
        Args:
            size_bytes: 字节数
            
        Returns:
            格式化后的大小字符串
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    @staticmethod
    def generate_uuid() -> str:
        """
        生成UUID
        
        Returns:
            UUID字符串
        """
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_random_string(length: int = 8, include_digits: bool = True, 
                              include_special: bool = False) -> str:
        """
        生成随机字符串
        
        Args:
            length: 字符串长度
            include_digits: 是否包含数字
            include_special: 是否包含特殊字符
            
        Returns:
            随机字符串
        """
        chars = string.ascii_letters
        if include_digits:
            chars += string.digits
        if include_special:
            chars += string.punctuation
        
        return ''.join(random.choice(chars) for _ in range(length))
    
    @staticmethod
    def calculate_md5(data: Union[str, bytes]) -> str:
        """
        计算MD5哈希
        
        Args:
            data: 要计算哈希的数据
            
        Returns:
            MD5哈希字符串
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def calculate_sha256(data: Union[str, bytes]) -> str:
        """
        计算SHA256哈希
        
        Args:
            data: 要计算哈希的数据
            
        Returns:
            SHA256哈希字符串
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def compress_data(data: Union[str, bytes]) -> bytes:
        """
        压缩数据
        
        Args:
            data: 要压缩的数据
            
        Returns:
            压缩后的数据
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return zlib.compress(data)
    
    @staticmethod
    def decompress_data(data: bytes) -> bytes:
        """
        解压数据
        
        Args:
            data: 要解压的数据
            
        Returns:
            解压后的数据
        """
        return zlib.decompress(data)
    
    @staticmethod
    def base64_encode(data: Union[str, bytes]) -> str:
        """
        Base64编码
        
        Args:
            data: 要编码的数据
            
        Returns:
            Base64编码字符串
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def base64_decode(data: str) -> bytes:
        """
        Base64解码
        
        Args:
            data: 要解码的Base64字符串
            
        Returns:
            解码后的数据
        """
        return base64.b64decode(data)
    
    @staticmethod
    def serialize_to_json(obj: Any, indent: Optional[int] = None, 
                         ensure_ascii: bool = False) -> str:
        """
        序列化为JSON
        
        Args:
            obj: 要序列化的对象
            indent: 缩进空格数
            ensure_ascii: 是否确保ASCII编码
            
        Returns:
            JSON字符串
        """
        return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    
    @staticmethod
    def deserialize_from_json(json_str: str) -> Any:
        """
        从JSON反序列化
        
        Args:
            json_str: JSON字符串
            
        Returns:
            反序列化后的对象
        """
        return json.loads(json_str)
    
    @staticmethod
    def serialize_to_pickle(obj: Any) -> bytes:
        """
        序列化为Pickle
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            Pickle字节串
        """
        return pickle.dumps(obj)
    
    @staticmethod
    def deserialize_from_pickle(pickle_bytes: bytes) -> Any:
        """
        从Pickle反序列化
        
        Args:
            pickle_bytes: Pickle字节串
            
        Returns:
            反序列化后的对象
        """
        return pickle.loads(pickle_bytes)
    
    @staticmethod
    def ensure_dir(directory: str) -> None:
        """
        确保目录存在
        
        Args:
            directory: 目录路径
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        获取文件扩展名
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件扩展名（不包含点）
        """
        return os.path.splitext(file_path)[1][1:].lower()
    
    @staticmethod
    def is_file_exists(file_path: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件是否存在
        """
        return os.path.isfile(file_path)
    
    @staticmethod
    def is_dir_exists(directory: str) -> bool:
        """
        检查目录是否存在
        
        Args:
            directory: 目录路径
            
        Returns:
            目录是否存在
        """
        return os.path.isdir(directory)
    
    @staticmethod
    def list_files(directory: str, pattern: Optional[str] = None, 
                  recursive: bool = False) -> List[str]:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件名模式（正则表达式）
            recursive: 是否递归遍历子目录
            
        Returns:
            文件路径列表
        """
        result = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if pattern is None or re.match(pattern, file):
                        result.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and (pattern is None or re.match(pattern, file)):
                    result.append(file_path)
        
        return result
    
    @staticmethod
    def read_file(file_path: str, encoding: str = 'utf-8') -> str:
        """
        读取文本文件
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            
        Returns:
            文件内容
        """
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def read_binary_file(file_path: str) -> bytes:
        """
        读取二进制文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容
        """
        with open(file_path, 'rb') as f:
            return f.read()
    
    @staticmethod
    def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> None:
        """
        写入文本文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 文件编码
        """
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def write_binary_file(file_path: str, content: bytes) -> None:
        """
        写入二进制文件
        
        Args:
            file_path: 文件路径
            content: 文件内容
        """
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'wb') as f:
            f.write(content)
    
    @staticmethod
    def append_file(file_path: str, content: str, encoding: str = 'utf-8') -> None:
        """
        追加文本文件
        
        Args:
            file_path: 文件路径
            content: 要追加的内容
            encoding: 文件编码
        """
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'a', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def copy_file(src_path: str, dst_path: str) -> None:
        """
        复制文件
        
        Args:
            src_path: 源文件路径
            dst_path: 目标文件路径
        """
        # 确保目标目录存在
        dst_dir = os.path.dirname(dst_path)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        shutil.copy2(src_path, dst_path)
    
    @staticmethod
    def move_file(src_path: str, dst_path: str) -> None:
        """
        移动文件
        
        Args:
            src_path: 源文件路径
            dst_path: 目标文件路径
        """
        # 确保目标目录存在
        dst_dir = os.path.dirname(dst_path)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        shutil.move(src_path, dst_path)
    
    @staticmethod
    def delete_file(file_path: str) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功删除
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"删除文件 {file_path} 失败: {e}")
            return False
    
    @staticmethod
    def create_temp_file(suffix: Optional[str] = None, prefix: Optional[str] = None,
                        dir: Optional[str] = None, text: bool = True) -> Tuple[int, str]:
        """
        创建临时文件
        
        Args:
            suffix: 文件后缀
            prefix: 文件前缀
            dir: 临时文件目录
            text: 是否为文本文件
            
        Returns:
            文件描述符和文件路径的元组
        """
        return tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
    
    @staticmethod
    def create_temp_dir(suffix: Optional[str] = None, prefix: Optional[str] = None,
                       dir: Optional[str] = None) -> str:
        """
        创建临时目录
        
        Args:
            suffix: 目录后缀
            prefix: 目录前缀
            dir: 父目录
            
        Returns:
            临时目录路径
        """
        return tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        获取文件大小
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件大小（字节）
        """
        return os.path.getsize(file_path)
    
    @staticmethod
    def get_file_modification_time(file_path: str) -> float:
        """
        获取文件修改时间
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件修改时间戳
        """
        return os.path.getmtime(file_path)
    
    @staticmethod
    def get_file_creation_time(file_path: str) -> float:
        """
        获取文件创建时间
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件创建时间戳
        """
        return os.path.getctime(file_path)
    
    @staticmethod
    def execute_command(command: Union[str, List[str]], shell: bool = False, 
                       cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
                       timeout: Optional[float] = None, encoding: str = 'utf-8') -> Tuple[int, str, str]:
        """
        执行命令
        
        Args:
            command: 命令字符串或参数列表
            shell: 是否使用shell执行
            cwd: 工作目录
            env: 环境变量
            timeout: 超时时间（秒）
            encoding: 输出编码
            
        Returns:
            返回码、标准输出和标准错误的元组
        """
        try:
            process = subprocess.Popen(
                command,
                shell=shell,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                encoding=encoding
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return -1, stdout, stderr
        except Exception as e:
            return -1, "", str(e)
    
    @staticmethod
    def is_module_available(module_name: str) -> bool:
        """
        检查模块是否可用
        
        Args:
            module_name: 模块名称
            
        Returns:
            模块是否可用
        """
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    @staticmethod
    def import_module(module_name: str) -> Any:
        """
        导入模块
        
        Args:
            module_name: 模块名称
            
        Returns:
            导入的模块
        """
        return importlib.import_module(module_name)
    
    @staticmethod
    def get_function_args(func: Callable) -> List[str]:
        """
        获取函数参数名称
        
        Args:
            func: 函数对象
            
        Returns:
            参数名称列表
        """
        return list(inspect.signature(func).parameters.keys())
    
    @staticmethod
    def get_class_methods(cls: Type) -> List[str]:
        """
        获取类的方法名称
        
        Args:
            cls: 类对象
            
        Returns:
            方法名称列表
        """
        return [name for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)]
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "cpu_count": os.cpu_count(),
            "memory_info": {
                "total": None,  # 需要psutil库
                "available": None
            }
        }
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
             exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Callable:
        """
        重试装饰器
        
        Args:
            max_attempts: 最大尝试次数
            delay: 初始延迟时间（秒）
            backoff: 延迟时间的增长因子
            exceptions: 要捕获的异常类型
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attempt = 1
                current_delay = delay
                
                while attempt <= max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts:
                            raise
                        
                        logger.warning(
                            f"尝试 {attempt}/{max_attempts} 失败: {e}, "
                            f"将在 {current_delay:.2f} 秒后重试"
                        )
                        
                        time.sleep(current_delay)
                        attempt += 1
                        current_delay *= backoff
            
            return wrapper
        
        return decorator
    
    @staticmethod
    @contextlib.contextmanager
    def timer(name: Optional[str] = None) -> Generator[None, None, None]:
        """
        计时上下文管理器
        
        Args:
            name: 计时器名称
            
        Yields:
            无
        """
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        
        if name:
            logger.info(f"{name} 耗时: {InternalUtils.format_time(elapsed_time)}")
        else:
            logger.info(f"操作耗时: {InternalUtils.format_time(elapsed_time)}")
    
    @staticmethod
    def chunks(lst: List[T], n: int) -> Generator[List[T], None, None]:
        """
        将列表分块
        
        Args:
            lst: 要分块的列表
            n: 每块的大小
            
        Yields:
            列表分块
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    @staticmethod
    def flatten(lst: List[List[T]]) -> List[T]:
        """
        扁平化嵌套列表
        
        Args:
            lst: 嵌套列表
            
        Returns:
            扁平化后的列表
        """
        return [item for sublist in lst for item in sublist]
    
    @staticmethod
    def group_by(items: List[T], key_func: Callable[[T], Any]) -> Dict[Any, List[T]]:
        """
        按键分组
        
        Args:
            items: 项目列表
            key_func: 键函数
            
        Returns:
            分组字典
        """
        result = {}
        for item in items:
            key = key_func(item)
            if key not in result:
                result[key] = []
            result[key].append(item)
        return result
    
    @staticmethod
    def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
        """
        深度合并字典
        
        Args:
            dict1: 第一个字典
            dict2: 第二个字典
            
        Returns:
            合并后的字典
        """
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = InternalUtils.deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    @staticmethod
    def deep_get(obj: Dict, path: str, default: Any = None, separator: str = '.') -> Any:
        """
        从嵌套字典中获取值
        
        Args:
            obj: 字典
            path: 路径，使用分隔符分隔
            default: 默认值
            separator: 路径分隔符
            
        Returns:
            路径对应的值，如果不存在则返回默认值
        """
        keys = path.split(separator)
        current = obj
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        
        return current
    
    @staticmethod
    def deep_set(obj: Dict, path: str, value: Any, separator: str = '.') -> None:
        """
        设置嵌套字典中的值
        
        Args:
            obj: 字典
            path: 路径，使用分隔符分隔
            value: 要设置的值
            separator: 路径分隔符
        """
        keys = path.split(separator)
        current = obj
        
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    @staticmethod
    def camel_to_snake(name: str) -> str:
        """
        驼峰命名转蛇形命名
        
        Args:
            name: 驼峰命名字符串
            
        Returns:
            蛇形命名字符串
        """
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    @staticmethod
    def snake_to_camel(name: str) -> str:
        """
        蛇形命名转驼峰命名
        
        Args:
            name: 蛇形命名字符串
            
        Returns:
            驼峰命名字符串
        """
        return ''.join(word.title() for word in name.split('_'))
    
    @staticmethod
    def truncate_string(s: str, max_length: int, suffix: str = '...') -> str:
        """
        截断字符串
        
        Args:
            s: 字符串
            max_length: 最大长度
            suffix: 后缀
            
        Returns:
            截断后的字符串
        """
        if len(s) <= max_length:
            return s
        return s[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """
        检查是否为有效的电子邮件地址
        
        Args:
            email: 电子邮件地址
            
        Returns:
            是否有效
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))