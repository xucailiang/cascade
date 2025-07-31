"""
音频环形缓冲区单元测试

测试AudioRingBuffer的核心功能：
- 基本读写操作
- 零拷贝访问
- 线程安全
- 重叠处理
- 错误处理
"""

import pytest
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from cascade.types import AudioConfig, BufferStrategy
from cascade.buffer import AudioRingBuffer
from cascade.types.errors import BufferError, BufferFullError, InsufficientDataError


class TestAudioRingBuffer:
    """音频环形缓冲区测试套件"""

    @pytest.fixture
    def audio_config(self):
        """标准音频配置"""
        return AudioConfig(
            sample_rate=16000,
            channels=1,
            dtype="float32",
            format="wav"
        )

    @pytest.fixture
    def ring_buffer(self, audio_config):
        """创建标准环形缓冲区"""
        return AudioRingBuffer(
            config=audio_config,
            capacity_seconds=2.0,  # 2秒容量
            overflow_strategy=BufferStrategy.BLOCK
        )

    def test_buffer_initialization(self, ring_buffer, audio_config):
        """测试缓冲区初始化"""
        assert ring_buffer.config == audio_config
        assert ring_buffer.capacity_seconds == 2.0
        assert ring_buffer.capacity_samples == 32000  # 2s * 16kHz
        assert ring_buffer.is_empty()
        assert not ring_buffer.is_full()
        assert not ring_buffer.is_closed

    def test_basic_write_read(self, ring_buffer):
        """测试基本写入和读取"""
        # 创建测试数据
        test_data = np.random.randn(1600).astype(np.float32)  # 100ms @ 16kHz
        
        # 写入数据
        success = ring_buffer.write(test_data)
        assert success
        assert ring_buffer.available_samples() == 1600
        assert not ring_buffer.is_empty()

        # 读取数据块
        chunk, available = ring_buffer.get_chunk_with_overlap(
            chunk_size=800,   # 50ms
            overlap_size=160  # 10ms
        )
        
        assert available
        assert chunk is not None
        assert chunk.chunk_size == 800
        assert chunk.overlap_size == 160
        assert len(chunk.data) == 960  # 800 + 160
        
        # 验证数据正确性（前800个样本应该匹配）
        np.testing.assert_array_equal(chunk.data[:800], test_data[:800])

    def test_advance_read_position(self, ring_buffer):
        """测试读取位置前进"""
        # 写入数据
        test_data = np.random.randn(1600).astype(np.float32)
        ring_buffer.write(test_data)
        
        initial_available = ring_buffer.available_samples()
        
        # 获取块但不前进
        chunk, _ = ring_buffer.get_chunk_with_overlap(800, 160)
        assert ring_buffer.available_samples() == initial_available  # 未改变
        
        # 前进读取位置
        ring_buffer.advance_read_position(800)
        assert ring_buffer.available_samples() == initial_available - 800
        
        # 再次读取应该得到后续数据
        chunk2, _ = ring_buffer.get_chunk_with_overlap(800, 0)
        np.testing.assert_array_equal(chunk2.data, test_data[800:1600])

    def test_wrap_around_behavior(self, ring_buffer):
        """测试环形缓冲区的绕环行为"""
        # 填充缓冲区到接近满容量
        chunk_size = 8000  # 500ms
        data_chunks = []
        
        for i in range(3):  # 写入1.5秒数据
            data = np.random.randn(chunk_size).astype(np.float32)
            data_chunks.append(data)
            ring_buffer.write(data)
        
        # 读取一些数据以释放空间
        ring_buffer.get_chunk_with_overlap(8000, 0)
        ring_buffer.advance_read_position(8000)
        
        # 写入新数据，应该绕环
        new_data = np.random.randn(chunk_size).astype(np.float32)
        success = ring_buffer.write(new_data)
        assert success

    def test_zero_copy_vs_copy(self, ring_buffer):
        """测试零拷贝与数据复制的情况"""
        # 写入连续数据
        test_data = np.random.randn(1600).astype(np.float32)
        ring_buffer.write(test_data)
        
        # 读取连续区域（应该是零拷贝）
        chunk, _ = ring_buffer.get_chunk_with_overlap(800, 160)
        assert chunk.metadata['is_continuous']
        
        # 获取性能统计
        stats = ring_buffer.get_performance_stats()
        assert stats['zero_copy_operations'] > 0

    def test_overflow_strategies(self, audio_config):
        """测试不同的溢出策略"""
        # 测试REJECT策略
        buffer_reject = AudioRingBuffer(
            config=audio_config,
            capacity_seconds=0.1,  # 很小的缓冲区
            overflow_strategy=BufferStrategy.REJECT
        )
        
        # 填满缓冲区
        small_data = np.random.randn(800).astype(np.float32)
        buffer_reject.write(small_data)
        buffer_reject.write(small_data)  # 应该填满
        
        # 再次写入应该失败
        success = buffer_reject.write(small_data, blocking=False)
        assert not success
        
        # 测试OVERWRITE策略
        buffer_overwrite = AudioRingBuffer(
            config=audio_config,
            capacity_seconds=0.1,
            overflow_strategy=BufferStrategy.OVERWRITE
        )
        
        # 连续写入应该覆盖旧数据
        for i in range(5):
            data = np.random.randn(800).astype(np.float32)
            success = buffer_overwrite.write(data)
            assert success

    def test_concurrent_access(self, ring_buffer):
        """测试并发访问的线程安全性"""
        test_duration = 1.0  # 测试1秒
        write_interval = 0.01  # 10ms写入间隔
        read_interval = 0.02   # 20ms读取间隔
        
        write_count = 0
        read_count = 0
        errors = []
        
        def writer():
            nonlocal write_count
            end_time = time.time() + test_duration
            while time.time() < end_time:
                try:
                    data = np.random.randn(160).astype(np.float32)  # 10ms
                    if ring_buffer.write(data, blocking=False):
                        write_count += 1
                    time.sleep(write_interval)
                except Exception as e:
                    errors.append(f"Writer error: {e}")
        
        def reader():
            nonlocal read_count
            end_time = time.time() + test_duration
            while time.time() < end_time:
                try:
                    if ring_buffer.available_samples() >= 320:  # 20ms
                        chunk, success = ring_buffer.get_chunk_with_overlap(160, 80)
                        if success:
                            ring_buffer.advance_read_position(160)
                            read_count += 1
                    time.sleep(read_interval)
                except Exception as e:
                    errors.append(f"Reader error: {e}")
        
        # 并发执行
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(writer),
                executor.submit(writer),
                executor.submit(reader),
                executor.submit(reader)
            ]
            
            for future in futures:
                future.result()
        
        # 验证结果
        assert len(errors) == 0, f"并发错误: {errors}"
        assert write_count > 0, "应该有写入操作"
        assert read_count > 0, "应该有读取操作"
        print(f"并发测试: 写入{write_count}次, 读取{read_count}次")

    def test_error_handling(self, ring_buffer):
        """测试错误处理"""
        # 测试无效数据格式
        with pytest.raises(ValueError):
            invalid_data = np.array([[1, 2], [3, 4]], dtype=np.float32)  # 2D数组
            ring_buffer.write(invalid_data)
        
        # 测试错误的数据类型
        with pytest.raises(ValueError):
            wrong_dtype = np.random.randn(100).astype(np.int32)  # 错误类型
            ring_buffer.write(wrong_dtype)
        
        # 测试无效的块参数
        with pytest.raises(ValueError):
            ring_buffer.get_chunk_with_overlap(100, 200)  # overlap > chunk
        
        # 测试在数据不足时前进
        with pytest.raises(InsufficientDataError):
            ring_buffer.advance_read_position(1000)  # 超过可用数据
        
        # 测试关闭后的操作
        ring_buffer.close()
        with pytest.raises(BufferError):
            ring_buffer.write(np.random.randn(100).astype(np.float32))

    def test_performance_monitoring(self, ring_buffer):
        """测试性能监控功能"""
        # 执行一些操作
        for i in range(10):
            data = np.random.randn(160).astype(np.float32)
            ring_buffer.write(data)
            
            if i >= 5:  # 从第6次开始读取
                chunk, _ = ring_buffer.get_chunk_with_overlap(160, 16)
                ring_buffer.advance_read_position(160)
        
        # 检查性能统计
        stats = ring_buffer.get_performance_stats()
        assert stats['write_operations'] == 10
        assert stats['read_operations'] == 5
        assert stats['uptime_seconds'] > 0
        assert 0 <= stats['zero_copy_rate'] <= 1.0
        
        # 检查缓冲区状态
        status = ring_buffer.get_buffer_status()
        assert status.capacity == ring_buffer.capacity_samples
        assert status.status_level in ["normal", "warning", "critical"]

    def test_context_manager(self, audio_config):
        """测试上下文管理器功能"""
        test_data = np.random.randn(100).astype(np.float32)
        
        with AudioRingBuffer(audio_config, 1.0) as buffer:
            assert not buffer.is_closed
            buffer.write(test_data)
            assert buffer.available_samples() == 100
        
        # 退出上下文后应该自动关闭
        assert buffer.is_closed


if __name__ == "__main__":
    # 运行简单的烟雾测试
    config = AudioConfig(sample_rate=16000, channels=1, dtype="float32")
    buffer = AudioRingBuffer(config, 1.0)
    
    # 基本功能测试
    data = np.random.randn(1600).astype(np.float32)
    success = buffer.write(data)
    print(f"写入成功: {success}")
    
    chunk, available = buffer.get_chunk_with_overlap(800, 160)
    print(f"读取成功: {available}, 块大小: {len(chunk.data) if chunk else 0}")
    
    stats = buffer.get_performance_stats()
    print(f"性能统计: {stats}")
    
    buffer.close()
    print("音频缓冲区基本功能测试通过!")