"""
内部工具函数测试
"""

import logging
import os
import shutil
import string
import tempfile
import time
import unittest

from cascade._internal.utils import InternalUtils, LazyProperty, Singleton


# 用于pickle测试的全局类
class PickleTestClass:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, PickleTestClass):
            return False
        return self.value == other.value


class TestSingleton(unittest.TestCase):
    """Singleton元类测试"""

    def test_singleton(self):
        """测试单例模式"""
        # 定义使用Singleton元类的类
        class TestClass(metaclass=Singleton):
            def __init__(self, value=None):
                self.value = value

        # 创建实例
        instance1 = TestClass(1)
        instance2 = TestClass(2)

        # 验证是同一个实例
        self.assertIs(instance1, instance2)

        # 验证值没有被第二次初始化覆盖
        self.assertEqual(instance1.value, 1)


class TestLazyProperty(unittest.TestCase):
    """LazyProperty装饰器测试"""

    def test_lazy_property(self):
        """测试延迟加载属性"""
        # 定义使用LazyProperty的类
        class TestClass:
            def __init__(self):
                self.compute_count = 0

            @LazyProperty
            def expensive_property(self):
                self.compute_count += 1
                return "computed_value"

        # 创建实例
        instance = TestClass()

        # 验证初始状态
        self.assertEqual(instance.compute_count, 0)

        # 访问属性
        value1 = instance.expensive_property
        self.assertEqual(value1, "computed_value")
        self.assertEqual(instance.compute_count, 1)

        # 再次访问属性
        value2 = instance.expensive_property
        self.assertEqual(value2, "computed_value")

        # 验证计算函数只被调用一次
        self.assertEqual(instance.compute_count, 1)


class TestInternalUtils(unittest.TestCase):
    """InternalUtils测试"""

    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)

    def test_setup_logging(self):
        """测试设置日志"""
        # 创建临时日志文件
        log_file = os.path.join(self.temp_dir, "test.log")

        # 设置日志
        logger = InternalUtils.setup_logging(
            level=logging.INFO,
            log_file=log_file,
            log_format="%(levelname)s: %(message)s",
            date_format="%Y-%m-%d",
            console=True,
            file_level=logging.DEBUG,
            console_level=logging.WARNING,
            capture_warnings=True
        )

        # 验证日志记录器
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.INFO)

        # 验证处理器
        self.assertEqual(len(logger.handlers), 2)  # 控制台和文件处理器

        # 验证文件处理器
        file_handler = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                file_handler = handler
                break

        self.assertIsNotNone(file_handler)
        self.assertEqual(file_handler.level, logging.DEBUG)

        # 写入日志
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        # 验证日志文件
        with open(log_file) as f:
            content = f.read()
            self.assertIn("DEBUG: Debug message", content)
            self.assertIn("INFO: Info message", content)
            self.assertIn("WARNING: Warning message", content)

    def test_get_timestamp(self):
        """测试获取时间戳"""
        timestamp = InternalUtils.get_timestamp()

        # 验证时间戳是当前时间
        self.assertAlmostEqual(timestamp, time.time(), delta=1.0)

    def test_get_iso_timestamp(self):
        """测试获取ISO格式时间戳"""
        timestamp = InternalUtils.get_iso_timestamp()

        # 验证ISO格式
        self.assertRegex(timestamp, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?")

    def test_format_time(self):
        """测试格式化时间"""
        # 微秒级
        self.assertEqual(InternalUtils.format_time(0.0005), "500.00 µs")

        # 毫秒级
        self.assertEqual(InternalUtils.format_time(0.5), "500.00 ms")

        # 秒级
        self.assertEqual(InternalUtils.format_time(5), "5.00 s")

        # 分钟级
        self.assertEqual(InternalUtils.format_time(65), "1m 5s")

        # 小时级
        self.assertEqual(InternalUtils.format_time(3665), "1h 1m 5s")

    def test_format_size(self):
        """测试格式化文件大小"""
        # 字节级
        self.assertEqual(InternalUtils.format_size(500), "500 B")

        # KB级
        self.assertEqual(InternalUtils.format_size(1500), "1.46 KB")

        # MB级
        self.assertEqual(InternalUtils.format_size(1500000), "1.43 MB")

        # GB级
        self.assertEqual(InternalUtils.format_size(1500000000), "1.40 GB")

    def test_generate_uuid(self):
        """测试生成UUID"""
        uuid1 = InternalUtils.generate_uuid()
        uuid2 = InternalUtils.generate_uuid()

        # 验证UUID格式
        self.assertRegex(uuid1, r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

        # 验证UUID唯一性
        self.assertNotEqual(uuid1, uuid2)

    def test_generate_random_string(self):
        """测试生成随机字符串"""
        # 默认参数
        random_str = InternalUtils.generate_random_string()
        self.assertEqual(len(random_str), 8)
        self.assertTrue(all(c in string.ascii_letters + string.digits for c in random_str))

        # 自定义长度
        random_str = InternalUtils.generate_random_string(length=16)
        self.assertEqual(len(random_str), 16)

        # 不包含数字
        random_str = InternalUtils.generate_random_string(include_digits=False)
        self.assertTrue(all(c in string.ascii_letters for c in random_str))

        # 包含特殊字符
        random_str = InternalUtils.generate_random_string(include_special=True)
        self.assertTrue(any(c in string.punctuation for c in random_str) or
                       all(c in string.ascii_letters + string.digits for c in random_str))

    def test_calculate_md5(self):
        """测试计算MD5哈希"""
        # 字符串输入
        md5_str = InternalUtils.calculate_md5("test")
        self.assertEqual(md5_str, "098f6bcd4621d373cade4e832627b4f6")

        # 字节输入
        md5_bytes = InternalUtils.calculate_md5(b"test")
        self.assertEqual(md5_bytes, "098f6bcd4621d373cade4e832627b4f6")

    def test_calculate_sha256(self):
        """测试计算SHA256哈希"""
        # 字符串输入
        sha256_str = InternalUtils.calculate_sha256("test")
        self.assertEqual(sha256_str, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")

        # 字节输入
        sha256_bytes = InternalUtils.calculate_sha256(b"test")
        self.assertEqual(sha256_bytes, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")

    def test_compress_decompress_data(self):
        """测试压缩和解压数据"""
        # 字符串输入
        original_str = "test" * 100
        compressed = InternalUtils.compress_data(original_str)
        decompressed = InternalUtils.decompress_data(compressed)

        self.assertIsInstance(compressed, bytes)
        self.assertLess(len(compressed), len(original_str))
        self.assertEqual(decompressed.decode("utf-8"), original_str)

        # 字节输入
        original_bytes = b"test" * 100
        compressed = InternalUtils.compress_data(original_bytes)
        decompressed = InternalUtils.decompress_data(compressed)

        self.assertIsInstance(compressed, bytes)
        self.assertLess(len(compressed), len(original_bytes))
        self.assertEqual(decompressed, original_bytes)

    def test_base64_encode_decode(self):
        """测试Base64编码和解码"""
        # 字符串输入
        original_str = "test"
        encoded = InternalUtils.base64_encode(original_str)
        decoded = InternalUtils.base64_decode(encoded)

        self.assertEqual(encoded, "dGVzdA==")
        self.assertEqual(decoded.decode("utf-8"), original_str)

        # 字节输入
        original_bytes = b"test"
        encoded = InternalUtils.base64_encode(original_bytes)
        decoded = InternalUtils.base64_decode(encoded)

        self.assertEqual(encoded, "dGVzdA==")
        self.assertEqual(decoded, original_bytes)

    def test_serialize_deserialize_json(self):
        """测试JSON序列化和反序列化"""
        # 简单对象
        obj = {"name": "test", "value": 123, "nested": {"key": "value"}}

        # 序列化
        json_str = InternalUtils.serialize_to_json(obj)
        self.assertIsInstance(json_str, str)

        # 反序列化
        deserialized = InternalUtils.deserialize_from_json(json_str)
        self.assertEqual(deserialized, obj)

        # 带缩进的序列化
        json_str_indented = InternalUtils.serialize_to_json(obj, indent=2)
        self.assertIn("\n", json_str_indented)

        # 非ASCII字符
        obj_unicode = {"name": "测试"}
        json_str_unicode = InternalUtils.serialize_to_json(obj_unicode)
        self.assertIn("测试", json_str_unicode)

    def test_serialize_deserialize_pickle(self):
        """测试Pickle序列化和反序列化"""
        # 简单对象
        obj = {"name": "test", "value": 123, "nested": {"key": "value"}}

        # 序列化
        pickle_bytes = InternalUtils.serialize_to_pickle(obj)
        self.assertIsInstance(pickle_bytes, bytes)

        # 反序列化
        deserialized = InternalUtils.deserialize_from_pickle(pickle_bytes)
        self.assertEqual(deserialized, obj)

        # 复杂对象（包含自定义类）
        obj_complex = {"obj": PickleTestClass(123)}

        # 序列化
        pickle_bytes = InternalUtils.serialize_to_pickle(obj_complex)

        # 反序列化
        deserialized = InternalUtils.deserialize_from_pickle(pickle_bytes)
        self.assertEqual(deserialized["obj"].value, obj_complex["obj"].value)

    def test_ensure_dir(self):
        """测试确保目录存在"""
        # 创建单级目录
        dir_path = os.path.join(self.temp_dir, "test_dir")
        InternalUtils.ensure_dir(dir_path)
        self.assertTrue(os.path.exists(dir_path))
        self.assertTrue(os.path.isdir(dir_path))

        # 创建多级目录
        nested_dir_path = os.path.join(self.temp_dir, "test_dir", "nested", "dir")
        InternalUtils.ensure_dir(nested_dir_path)
        self.assertTrue(os.path.exists(nested_dir_path))
        self.assertTrue(os.path.isdir(nested_dir_path))

        # 已存在的目录
        InternalUtils.ensure_dir(dir_path)
        self.assertTrue(os.path.exists(dir_path))

    def test_get_file_extension(self):
        """测试获取文件扩展名"""
        self.assertEqual(InternalUtils.get_file_extension("test.txt"), "txt")
        self.assertEqual(InternalUtils.get_file_extension("test.tar.gz"), "gz")
        self.assertEqual(InternalUtils.get_file_extension("test"), "")
        self.assertEqual(InternalUtils.get_file_extension("/path/to/test.py"), "py")

    def test_is_file_exists(self):
        """测试检查文件是否存在"""
        # 创建测试文件
        file_path = os.path.join(self.temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("test")

        # 验证文件存在
        self.assertTrue(InternalUtils.is_file_exists(file_path))

        # 验证文件不存在
        non_existent_file = os.path.join(self.temp_dir, "non_existent.txt")
        self.assertFalse(InternalUtils.is_file_exists(non_existent_file))

    def test_is_dir_exists(self):
        """测试检查目录是否存在"""
        # 创建测试目录
        dir_path = os.path.join(self.temp_dir, "test_dir")
        os.makedirs(dir_path)

        # 验证目录存在
        self.assertTrue(InternalUtils.is_dir_exists(dir_path))

        # 验证目录不存在
        non_existent_dir = os.path.join(self.temp_dir, "non_existent_dir")
        self.assertFalse(InternalUtils.is_dir_exists(non_existent_dir))

    def test_list_files(self):
        """测试列出目录中的文件"""
        # 创建测试目录和文件
        dir_path = os.path.join(self.temp_dir, "test_dir")
        os.makedirs(dir_path)

        file1 = os.path.join(dir_path, "file1.txt")
        file2 = os.path.join(dir_path, "file2.txt")
        file3 = os.path.join(dir_path, "file3.py")

        with open(file1, "w") as f:
            f.write("file1")
        with open(file2, "w") as f:
            f.write("file2")
        with open(file3, "w") as f:
            f.write("file3")

        # 创建子目录和文件
        subdir = os.path.join(dir_path, "subdir")
        os.makedirs(subdir)

        file4 = os.path.join(subdir, "file4.txt")
        with open(file4, "w") as f:
            f.write("file4")

        # 列出所有文件（非递归）
        files = InternalUtils.list_files(dir_path)
        self.assertEqual(len(files), 3)
        self.assertIn(file1, files)
        self.assertIn(file2, files)
        self.assertIn(file3, files)

        # 列出所有文件（递归）
        files = InternalUtils.list_files(dir_path, recursive=True)
        self.assertEqual(len(files), 4)
        self.assertIn(file1, files)
        self.assertIn(file2, files)
        self.assertIn(file3, files)
        self.assertIn(file4, files)

        # 列出匹配模式的文件
        files = InternalUtils.list_files(dir_path, pattern=r".*\.txt")
        self.assertEqual(len(files), 2)
        self.assertIn(file1, files)
        self.assertIn(file2, files)

    def test_read_write_file(self):
        """测试读写文本文件"""
        # 写入文件
        file_path = os.path.join(self.temp_dir, "test.txt")
        content = "测试内容"
        InternalUtils.write_file(file_path, content)

        # 读取文件
        read_content = InternalUtils.read_file(file_path)
        self.assertEqual(read_content, content)

        # 写入嵌套目录中的文件
        nested_file_path = os.path.join(self.temp_dir, "nested", "dir", "test.txt")
        InternalUtils.write_file(nested_file_path, content)

        # 验证文件和目录已创建
        self.assertTrue(os.path.exists(nested_file_path))
        self.assertTrue(os.path.isdir(os.path.dirname(nested_file_path)))

    def test_read_write_binary_file(self):
        """测试读写二进制文件"""
        # 写入文件
        file_path = os.path.join(self.temp_dir, "test.bin")
        content = b"\x00\x01\x02\x03"
        InternalUtils.write_binary_file(file_path, content)

        # 读取文件
        read_content = InternalUtils.read_binary_file(file_path)
        self.assertEqual(read_content, content)

    def test_append_file(self):
        """测试追加文件"""
        # 创建文件
        file_path = os.path.join(self.temp_dir, "test.txt")
        InternalUtils.write_file(file_path, "Line 1\n")

        # 追加内容
        InternalUtils.append_file(file_path, "Line 2\n")
        InternalUtils.append_file(file_path, "Line 3\n")

        # 读取文件
        content = InternalUtils.read_file(file_path)
        self.assertEqual(content, "Line 1\nLine 2\nLine 3\n")

    def test_copy_file(self):
        """测试复制文件"""
        # 创建源文件
        src_path = os.path.join(self.temp_dir, "source.txt")
        InternalUtils.write_file(src_path, "test content")

        # 复制到目标文件
        dst_path = os.path.join(self.temp_dir, "nested", "dest.txt")
        InternalUtils.copy_file(src_path, dst_path)

        # 验证目标文件
        self.assertTrue(os.path.exists(dst_path))
        self.assertEqual(InternalUtils.read_file(dst_path), "test content")

    def test_move_file(self):
        """测试移动文件"""
        # 创建源文件
        src_path = os.path.join(self.temp_dir, "source.txt")
        InternalUtils.write_file(src_path, "test content")

        # 移动到目标文件
        dst_path = os.path.join(self.temp_dir, "nested", "dest.txt")
        InternalUtils.move_file(src_path, dst_path)

        # 验证目标文件
        self.assertTrue(os.path.exists(dst_path))
        self.assertEqual(InternalUtils.read_file(dst_path), "test content")

        # 验证源文件不存在
        self.assertFalse(os.path.exists(src_path))

    def test_delete_file(self):
        """测试删除文件"""
        # 创建文件
        file_path = os.path.join(self.temp_dir, "test.txt")
        InternalUtils.write_file(file_path, "test content")

        # 删除文件
        result = InternalUtils.delete_file(file_path)

        # 验证结果
        self.assertTrue(result)
        self.assertFalse(os.path.exists(file_path))

        # 删除不存在的文件
        result = InternalUtils.delete_file(file_path)
        self.assertFalse(result)

    def test_create_temp_file(self):
        """测试创建临时文件"""
        # 创建临时文件
        fd, path = InternalUtils.create_temp_file(suffix=".txt", prefix="test_")

        try:
            # 验证文件
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".txt"))
            self.assertTrue(os.path.basename(path).startswith("test_"))
        finally:
            # 关闭文件描述符并删除文件
            os.close(fd)
            os.unlink(path)

    def test_create_temp_dir(self):
        """测试创建临时目录"""
        # 创建临时目录
        path = InternalUtils.create_temp_dir(suffix="_dir", prefix="test_")

        try:
            # 验证目录
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.isdir(path))
            self.assertTrue(path.endswith("_dir"))
            self.assertTrue(os.path.basename(path).startswith("test_"))
        finally:
            # 删除目录
            shutil.rmtree(path)

    def test_get_file_size(self):
        """测试获取文件大小"""
        # 创建文件
        file_path = os.path.join(self.temp_dir, "test.txt")
        content = "x" * 1000
        InternalUtils.write_file(file_path, content)

        # 获取文件大小
        size = InternalUtils.get_file_size(file_path)

        # 验证大小
        self.assertEqual(size, 1000)

    def test_get_file_modification_time(self):
        """测试获取文件修改时间"""
        # 创建文件
        file_path = os.path.join(self.temp_dir, "test.txt")
        InternalUtils.write_file(file_path, "test content")

        # 获取修改时间
        mtime = InternalUtils.get_file_modification_time(file_path)

        # 验证时间
        self.assertAlmostEqual(mtime, time.time(), delta=2.0)

    def test_get_file_creation_time(self):
        """测试获取文件创建时间"""
        # 创建文件
        file_path = os.path.join(self.temp_dir, "test.txt")
        InternalUtils.write_file(file_path, "test content")

        # 获取创建时间
        ctime = InternalUtils.get_file_creation_time(file_path)

        # 验证时间
        self.assertAlmostEqual(ctime, time.time(), delta=2.0)

    def test_execute_command(self):
        """测试执行命令"""
        # 执行简单命令
        returncode, stdout, stderr = InternalUtils.execute_command(["echo", "test"])

        # 验证结果
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout.strip(), "test")
        self.assertEqual(stderr, "")

        # 执行带错误的命令
        returncode, stdout, stderr = InternalUtils.execute_command(["ls", "non_existent_file"])

        # 验证结果
        self.assertNotEqual(returncode, 0)
        self.assertEqual(stdout, "")
        self.assertNotEqual(stderr, "")

        # 执行超时命令
        returncode, stdout, stderr = InternalUtils.execute_command(["sleep", "1"], timeout=0.1)

        # 验证结果
        self.assertEqual(returncode, -1)

    def test_is_module_available(self):
        """测试检查模块是否可用"""
        # 可用模块
        self.assertTrue(InternalUtils.is_module_available("os"))
        self.assertTrue(InternalUtils.is_module_available("sys"))

        # 不可用模块
        self.assertFalse(InternalUtils.is_module_available("non_existent_module"))

    def test_import_module(self):
        """测试导入模块"""
        # 导入模块
        os_module = InternalUtils.import_module("os")

        # 验证模块
        self.assertEqual(os_module, os)

        # 导入不存在的模块
        with self.assertRaises(ImportError):
            InternalUtils.import_module("non_existent_module")

    def test_get_function_args(self):
        """测试获取函数参数名称"""
        # 定义测试函数
        def test_func(a, b, c=1, *args, **kwargs):
            pass

        # 获取参数名称
        args = InternalUtils.get_function_args(test_func)

        # 验证参数名称
        self.assertEqual(args, ["a", "b", "c", "args", "kwargs"])

    def test_get_class_methods(self):
        """测试获取类的方法名称"""
        # 定义测试类
        class TestClass:
            def method1(self):
                pass

            def method2(self):
                pass

            @staticmethod
            def static_method():
                pass

        # 获取方法名称
        methods = InternalUtils.get_class_methods(TestClass)

        # 验证方法名称
        self.assertIn("method1", methods)
        self.assertIn("method2", methods)
        self.assertIn("static_method", methods)

    def test_get_system_info(self):
        """测试获取系统信息"""
        # 获取系统信息
        info = InternalUtils.get_system_info()

        # 验证信息
        self.assertIn("platform", info)
        self.assertIn("system", info)
        self.assertIn("release", info)
        self.assertIn("version", info)
        self.assertIn("machine", info)
        self.assertIn("processor", info)
        self.assertIn("python_version", info)
        self.assertIn("python_implementation", info)
        self.assertIn("python_compiler", info)
        self.assertIn("cpu_count", info)
        self.assertIn("memory_info", info)

    def test_retry_decorator(self):
        """测试重试装饰器"""
        # 计数器
        counter = {"count": 0}

        # 定义会失败的函数
        @InternalUtils.retry(max_attempts=3, delay=0.1, backoff=1.0, exceptions=(ValueError,))
        def failing_function():
            counter["count"] += 1
            if counter["count"] < 3:
                raise ValueError("Simulated failure")
            return "success"

        # 调用函数
        result = failing_function()

        # 验证结果
        self.assertEqual(result, "success")
        self.assertEqual(counter["count"], 3)

        # 定义总是失败的函数
        @InternalUtils.retry(max_attempts=3, delay=0.1, backoff=1.0, exceptions=(ValueError,))
        def always_failing_function():
            raise ValueError("Always fails")

        # 调用函数，应该抛出异常
        with self.assertRaises(ValueError):
            always_failing_function()

    def test_timer_context_manager(self):
        """测试计时上下文管理器"""
        # 使用计时器
        with InternalUtils.timer("Test operation"):
            time.sleep(0.1)

        # 无法直接验证日志输出，但可以验证代码执行没有异常

    def test_chunks(self):
        """测试将列表分块"""
        # 测试数据
        data = list(range(10))

        # 分块
        chunks = list(InternalUtils.chunks(data, 3))

        # 验证结果
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], [0, 1, 2])
        self.assertEqual(chunks[1], [3, 4, 5])
        self.assertEqual(chunks[2], [6, 7, 8])
        self.assertEqual(chunks[3], [9])
