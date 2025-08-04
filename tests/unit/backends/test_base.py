"""
VAD后端抽象基类测试

测试VAD后端抽象基类的基础功能和接口定义。
"""

from typing import Any

import pytest

from cascade.backends.base import VADBackend
from cascade.types import AudioChunk, CascadeError, ErrorCode, VADResult


class MockVADBackend(VADBackend):
    """测试用的Mock VAD后端"""

    def __init__(self, config: Any, should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.process_calls = []
        self.warmup_calls = []

    async def initialize(self) -> None:
        if self.should_fail:
            raise CascadeError("初始化失败", ErrorCode.INITIALIZATION_FAILED)
        self._initialized = True

    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        self._ensure_initialized()
        self._validate_chunk(chunk)

        self.process_calls.append(chunk)

        if self.should_fail:
            raise CascadeError("处理失败", ErrorCode.INFERENCE_FAILED)

        return VADResult(
            is_speech=True,
            probability=0.8,
            start_ms=chunk.timestamp_ms,
            end_ms=chunk.get_end_timestamp_ms(),
            chunk_id=chunk.sequence_number,
            confidence=0.9
        )

    def warmup(self, dummy_chunk: AudioChunk) -> None:
        self._ensure_initialized()
        self.warmup_calls.append(dummy_chunk)

    async def close(self) -> None:
        self._initialized = False


class TestVADBackendBase:
    """VAD后端基类测试"""

    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        return {"test": "config"}

    @pytest.fixture
    def valid_audio_chunk(self):
        """有效的音频块"""
        return AudioChunk(
            data=[0.1, 0.2, 0.3, 0.4],
            sequence_number=1,
            start_frame=0,
            chunk_size=4,
            overlap_size=0,
            timestamp_ms=0.0,
            sample_rate=16000,
            is_last=False
        )

    @pytest.fixture
    def backend(self, mock_config):
        """创建测试后端"""
        return MockVADBackend(mock_config)

    def test_backend_initialization(self, mock_config):
        """测试后端初始化"""
        backend = MockVADBackend(mock_config)

        assert backend.config == mock_config
        assert not backend.is_initialized
        # 检查锁是否是RLock类型（通过属性检查）
        assert hasattr(backend._lock, 'acquire')
        assert hasattr(backend._lock, 'release')
        # 验证可重入锁功能 - 通过功能性测试而非私有属性
        # 同一线程可多次获取锁（RLock特性）
        backend._lock.acquire()
        backend._lock.acquire()  # 如果是RLock，这不会阻塞
        backend._lock.release()
        backend._lock.release()

    @pytest.mark.asyncio
    async def test_async_initialize_success(self, backend):
        """测试异步初始化成功"""
        await backend.initialize()
        assert backend.is_initialized

    @pytest.mark.asyncio
    async def test_async_initialize_failure(self, mock_config):
        """测试异步初始化失败"""
        backend = MockVADBackend(mock_config, should_fail=True)

        with pytest.raises(CascadeError) as exc_info:
            await backend.initialize()

        assert exc_info.value.error_code == ErrorCode.INITIALIZATION_FAILED
        assert not backend.is_initialized

    @pytest.mark.asyncio
    async def test_process_chunk_success(self, backend, valid_audio_chunk):
        """测试音频块处理成功"""
        await backend.initialize()

        result = backend.process_chunk(valid_audio_chunk)

        assert isinstance(result, VADResult)
        assert result.is_speech is True
        assert result.probability == 0.8
        assert result.chunk_id == valid_audio_chunk.sequence_number
        assert len(backend.process_calls) == 1

    @pytest.mark.asyncio
    async def test_process_chunk_not_initialized(self, backend, valid_audio_chunk):
        """测试未初始化时处理音频块"""
        with pytest.raises(CascadeError) as exc_info:
            backend.process_chunk(valid_audio_chunk)

        assert exc_info.value.error_code == ErrorCode.INITIALIZATION_FAILED

    @pytest.mark.asyncio
    async def test_process_chunk_invalid_input(self, backend):
        """测试无效输入处理"""
        await backend.initialize()

        # 测试空块
        with pytest.raises(CascadeError) as exc_info:
            backend.process_chunk(None)
        assert exc_info.value.error_code == ErrorCode.INVALID_INPUT

        # 测试无效块大小 - Pydantic会在创建时验证，这是正确的行为
        # 我们测试 Pydantic ValidationError 被正确抛出
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            AudioChunk(
                data=[],
                sequence_number=1,
                start_frame=0,
                chunk_size=0,  # 无效大小 - Pydantic会阻止创建
                overlap_size=0,
                timestamp_ms=0.0,
                sample_rate=16000
            )
        assert "Input should be greater than 0" in str(exc_info.value)

        # 测试无效采样率 - 同样会被Pydantic阻止
        with pytest.raises(ValidationError) as exc_info:
            AudioChunk(
                data=[0.1, 0.2],
                sequence_number=1,
                start_frame=0,
                chunk_size=2,
                overlap_size=0,
                timestamp_ms=0.0,
                sample_rate=0  # 无效采样率 - Pydantic会阻止创建
            )
        assert "Input should be greater than 0" in str(exc_info.value)

        # 测试后端层面的数据验证 - 使用已有效但边界的数据
        edge_case_chunk = AudioChunk(
            data=[0.0],  # 单个样本
            sequence_number=1,
            start_frame=0,
            chunk_size=1,
            overlap_size=0,
            timestamp_ms=0.0,
            sample_rate=1,  # 最小有效采样率
            is_last=False
        )

        # 这应该能正常处理（虽然是边界情况）
        result = backend.process_chunk(edge_case_chunk)
        assert isinstance(result, VADResult)

    @pytest.mark.asyncio
    async def test_warmup_success(self, backend, valid_audio_chunk):
        """测试模型预热成功"""
        await backend.initialize()

        backend.warmup(valid_audio_chunk)

        assert len(backend.warmup_calls) == 1
        assert backend.warmup_calls[0] == valid_audio_chunk

    @pytest.mark.asyncio
    async def test_warmup_not_initialized(self, backend, valid_audio_chunk):
        """测试未初始化时预热"""
        with pytest.raises(CascadeError):
            backend.warmup(valid_audio_chunk)

    @pytest.mark.asyncio
    async def test_close_backend(self, backend):
        """测试关闭后端"""
        await backend.initialize()
        assert backend.is_initialized

        await backend.close()
        assert not backend.is_initialized

    def test_get_backend_info(self, backend):
        """测试获取后端信息"""
        info = backend.get_backend_info()

        assert isinstance(info, dict)
        assert "backend_type" in info
        assert "initialized" in info
        assert "config" in info
        assert info["backend_type"] == "MockVADBackend"
        assert info["initialized"] is False

    def test_context_manager_protocol(self, mock_config):
        """测试上下文管理器协议"""
        backend = MockVADBackend(mock_config)

        # 测试上下文管理器入口
        with backend as ctx_backend:
            assert ctx_backend is backend

        # close方法应该被调用（虽然是async，但在__exit__中会被处理）

    def test_thread_safety_basic(self, mock_config):
        """测试基本线程安全性"""
        backend = MockVADBackend(mock_config)

        # 验证锁存在
        assert hasattr(backend, '_lock')
        # 验证锁的基本功能
        assert hasattr(backend._lock, 'acquire')
        assert hasattr(backend._lock, 'release')

        # 基本锁操作测试
        with backend._lock:
            # 在锁内执行操作
            info = backend.get_backend_info()
            assert info is not None


class TestVADBackendConcurrency:
    """VAD后端并发测试"""

    @pytest.fixture
    def backend(self):
        """创建测试后端"""
        return MockVADBackend({"concurrent": True})

    @pytest.fixture
    def audio_chunks(self):
        """创建多个音频块用于并发测试"""
        chunks = []
        for i in range(10):
            chunk = AudioChunk(
                data=[0.1 * i, 0.2 * i],
                sequence_number=i,
                start_frame=i * 2,
                chunk_size=2,
                overlap_size=0,
                timestamp_ms=float(i * 100),
                sample_rate=16000
            )
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, backend, audio_chunks):
        """测试并发处理能力"""
        await backend.initialize()

        import concurrent.futures

        results = []
        errors = []

        def process_chunk(chunk):
            try:
                result = backend.process_chunk(chunk)
                results.append(result)
                return result
            except Exception as e:
                errors.append(e)
                raise

        # 使用线程池并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_chunk, chunk)
                for chunk in audio_chunks
            ]

            # 等待所有任务完成
            concurrent.futures.wait(futures)

        # 验证结果
        assert len(errors) == 0, f"并发处理中出现错误: {errors}"
        assert len(results) == len(audio_chunks)
        assert len(backend.process_calls) == len(audio_chunks)

        # 验证所有块都被处理
        processed_ids = {call.sequence_number for call in backend.process_calls}
        expected_ids = {chunk.sequence_number for chunk in audio_chunks}
        assert processed_ids == expected_ids
