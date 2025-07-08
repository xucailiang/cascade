"""
基础处理器测试
"""

import asyncio
from typing import Any

import numpy as np
import pytest

from cascade.processor.base import (
    AudioChunk,
    AudioProcessor,
    OverlapStrategy,
    ProcessorConfig,
    ProcessResult,
)
from cascade.types.audio import AudioConfig, AudioFormat


class SimpleProcessor(AudioProcessor):
    """简单处理器，用于测试基础处理器功能"""

    def __init__(self, config: ProcessorConfig | None = None):
        super().__init__(config)
        self.processed_chunks = []
        self.process_called = False

    def process_chunk(self, chunk: AudioChunk) -> Any:
        """处理单个音频块"""
        self.processed_chunks.append(chunk)
        self.process_called = True
        # 返回块的序列号作为结果
        return {"sequence": chunk.sequence_number, "is_last": chunk.is_last}


class FrontPriorityProcessor(AudioProcessor):
    """前块优先处理器，用于测试重叠策略"""

    def __init__(self):
        config = ProcessorConfig(
            chunk_duration_ms=100,
            overlap_ms=20,
            overlap_strategy=OverlapStrategy.FRONT_PRIORITY
        )
        super().__init__(config)

    def process_chunk(self, chunk: AudioChunk) -> Any:
        """处理单个音频块"""
        # 返回块的序列号和重叠大小
        return {
            "sequence": chunk.sequence_number,
            "overlap_size": chunk.overlap_size
        }


class BackPriorityProcessor(AudioProcessor):
    """后块优先处理器，用于测试重叠策略"""

    def __init__(self):
        config = ProcessorConfig(
            chunk_duration_ms=100,
            overlap_ms=20,
            overlap_strategy=OverlapStrategy.BACK_PRIORITY
        )
        super().__init__(config)

    def process_chunk(self, chunk: AudioChunk) -> Any:
        """处理单个音频块"""
        # 返回块的序列号和重叠大小
        return {
            "sequence": chunk.sequence_number,
            "overlap_size": chunk.overlap_size
        }

    async def _merge_results(self, context: dict[str, Any]) -> dict[str, Any]:
        """重写合并结果方法，实现后块优先策略"""
        results = context["results"]

        # 按序列号排序
        results.sort(key=lambda r: r.chunk.sequence_number)

        # 实现后块优先策略
        # 在实际应用中，这里会有更复杂的逻辑来处理重叠区域
        # 这里只是简单地标记结果为"back_priority"
        for result in results:
            result.metadata = result.metadata or {}
            result.metadata["strategy"] = "back_priority"

        context["results"] = results
        return context


class MaxConfidenceProcessor(AudioProcessor):
    """最高置信度优先处理器，用于测试重叠策略"""

    def __init__(self):
        config = ProcessorConfig(
            chunk_duration_ms=100,
            overlap_ms=20,
            overlap_strategy=OverlapStrategy.MAX_CONFIDENCE
        )
        super().__init__(config)

    def process_chunk(self, chunk: AudioChunk) -> Any:
        """处理单个音频块"""
        # 返回块的序列号、重叠大小和一个模拟的置信度
        confidence = 0.5 + 0.1 * chunk.sequence_number  # 模拟置信度，后面的块置信度更高
        return {
            "sequence": chunk.sequence_number,
            "overlap_size": chunk.overlap_size,
            "confidence": confidence
        }

    async def _merge_results(self, context: dict[str, Any]) -> dict[str, Any]:
        """重写合并结果方法，实现最高置信度优先策略"""
        results = context["results"]

        # 按序列号排序
        results.sort(key=lambda r: r.chunk.sequence_number)

        # 实现最高置信度优先策略
        # 在实际应用中，这里会有更复杂的逻辑来比较重叠区域的置信度
        # 这里只是简单地标记结果为"max_confidence"
        for result in results:
            result.metadata = result.metadata or {}
            result.metadata["strategy"] = "max_confidence"
            # 将置信度添加到元数据中
            if isinstance(result.result_data, dict) and "confidence" in result.result_data:
                result.metadata["confidence"] = result.result_data["confidence"]

        context["results"] = results
        return context


@pytest.fixture
def sample_audio():
    """生成测试用的音频数据"""
    # 创建一个简单的合成音频
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    # 创建一个简单的正弦波
    t = np.linspace(0, duration_sec, num_samples)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波

    return audio, sample_rate


@pytest.fixture
def audio_config(sample_audio):
    """创建音频配置"""
    _, sample_rate = sample_audio
    return AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )


@pytest.mark.asyncio
async def test_audio_chunking(sample_audio, audio_config):
    """测试音频分块功能"""
    audio_data, _ = sample_audio

    # 创建处理器，设置块大小和重叠大小
    config = ProcessorConfig(
        chunk_duration_ms=100,  # 100ms的块
        overlap_ms=20,          # 20ms的重叠
    )
    processor = SimpleProcessor(config)

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) > 0
    assert processor.process_called

    # 验证块的数量和属性
    chunks = processor.processed_chunks
    assert len(chunks) > 0

    # 计算预期的块数
    # 1秒的16kHz音频 = 16000个样本
    # 100ms的块 = 1600个样本
    # 20ms的重叠 = 320个样本
    # 步长 = 1600 - 320 = 1280个样本
    # 块数 = (16000 - 320 + 1280 - 1) // 1280 = 13
    expected_chunks = (len(audio_data) - int(0.02 * audio_config.sample_rate) +
                      int(0.08 * audio_config.sample_rate)) // int(0.08 * audio_config.sample_rate)
    assert len(chunks) == expected_chunks

    # 验证块的属性
    for i, chunk in enumerate(chunks):
        # 验证序列号
        assert chunk.sequence_number == i

        # 验证重叠大小
        if i < len(chunks) - 1:
            assert chunk.overlap_size > 0
        else:
            # 最后一个块没有重叠
            assert chunk.is_last

    # 验证结果数据
    for result in results:
        assert isinstance(result, ProcessResult)
        assert isinstance(result.result_data, dict)
        assert "sequence" in result.result_data
        assert "is_last" in result.result_data


@pytest.mark.asyncio
async def test_overlap_strategies(sample_audio, audio_config):
    """测试不同的重叠策略"""
    audio_data, _ = sample_audio

    # 测试前块优先策略
    front_processor = FrontPriorityProcessor()
    front_results = await front_processor.process_audio(audio_data, audio_config)

    # 测试后块优先策略
    back_processor = BackPriorityProcessor()
    back_results = await back_processor.process_audio(audio_data, audio_config)

    # 测试最高置信度优先策略
    max_conf_processor = MaxConfidenceProcessor()
    max_conf_results = await max_conf_processor.process_audio(audio_data, audio_config)

    # 验证结果数量相同
    assert len(front_results) == len(back_results) == len(max_conf_results)

    # 验证后块优先策略的元数据
    for result in back_results:
        assert result.metadata is not None
        assert "strategy" in result.metadata
        assert result.metadata["strategy"] == "back_priority"

    # 验证最高置信度优先策略的元数据
    for result in max_conf_results:
        assert result.metadata is not None
        assert "strategy" in result.metadata
        assert result.metadata["strategy"] == "max_confidence"
        assert "confidence" in result.metadata


@pytest.mark.asyncio
async def test_processing_pipeline(sample_audio, audio_config):
    """测试处理管道流程"""
    audio_data, _ = sample_audio

    # 创建一个自定义处理器，跟踪管道步骤的执行
    class PipelineTrackingProcessor(AudioProcessor):
        def __init__(self):
            super().__init__()
            self.pipeline_steps = []

        def process_chunk(self, chunk: AudioChunk) -> Any:
            self.pipeline_steps.append("process_chunk")
            return {"processed": True}

        async def _prepare_audio(self, context: dict[str, Any]) -> dict[str, Any]:
            self.pipeline_steps.append("prepare_audio")
            return await super()._prepare_audio(context)

        async def _split_into_chunks(self, context: dict[str, Any]) -> dict[str, Any]:
            self.pipeline_steps.append("split_into_chunks")
            return await super()._split_into_chunks(context)

        async def _process_chunks(self, context: dict[str, Any]) -> dict[str, Any]:
            self.pipeline_steps.append("process_chunks")
            return await super()._process_chunks(context)

        async def _merge_results(self, context: dict[str, Any]) -> dict[str, Any]:
            self.pipeline_steps.append("merge_results")
            return await super()._merge_results(context)

    # 创建处理器
    processor = PipelineTrackingProcessor()

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证管道步骤的执行顺序
    expected_steps = [
        "prepare_audio",
        "split_into_chunks",
        "process_chunks",
        "process_chunk",  # 这一步可能会被调用多次，取决于块的数量
        "merge_results"
    ]

    # 验证前三个步骤和最后一个步骤
    assert processor.pipeline_steps[0] == expected_steps[0]
    assert processor.pipeline_steps[1] == expected_steps[1]
    assert processor.pipeline_steps[2] == expected_steps[2]
    assert processor.pipeline_steps[-1] == expected_steps[-1]

    # 验证中间的process_chunk步骤
    for step in processor.pipeline_steps[3:-1]:
        assert step == "process_chunk"


@pytest.mark.asyncio
async def test_empty_audio(audio_config):
    """测试处理空音频"""
    # 创建空音频
    audio_data = np.array([], dtype=np.float32)

    # 创建处理器
    processor = SimpleProcessor()

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) == 0
    assert not processor.process_called
    assert len(processor.processed_chunks) == 0


@pytest.mark.asyncio
async def test_short_audio(audio_config):
    """测试处理短音频（小于一个块）"""
    # 创建短音频（10ms，小于默认块大小）
    sample_rate = audio_config.sample_rate
    duration_sec = 0.01
    num_samples = int(sample_rate * duration_sec)
    audio_data = np.random.normal(0, 0.1, num_samples)

    # 创建处理器
    processor = SimpleProcessor()

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) == 1
    assert processor.process_called
    assert len(processor.processed_chunks) == 1

    # 验证块属性
    chunk = processor.processed_chunks[0]
    assert chunk.sequence_number == 0
    assert chunk.is_last
    assert chunk.overlap_size == 0
    assert len(chunk.data) == num_samples


@pytest.mark.asyncio
async def test_error_handling():
    """测试错误处理"""
    # 创建一个会抛出错误的处理器
    class ErrorProcessor(AudioProcessor):
        def process_chunk(self, chunk: AudioChunk) -> Any:
            raise ValueError("测试错误")

    # 创建处理器
    processor = ErrorProcessor()

    # 创建简单音频
    sample_rate = 16000
    audio_data = np.random.normal(0, 0.1, sample_rate)  # 1秒的音频
    audio_config = AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) > 0

    # 所有结果应该标记为失败
    for result in results:
        assert not result.success
        assert result.error is not None
        assert "测试错误" in result.error


@pytest.mark.asyncio
async def test_thread_pool():
    """测试线程池管理"""
    # 创建一个使用线程池的处理器
    class ThreadPoolProcessor(AudioProcessor):
        def __init__(self):
            config = ProcessorConfig(
                max_workers=2,
                thread_name_prefix="test-pool"
            )
            super().__init__(config)
            self.processed_count = 0

        def process_chunk(self, chunk: AudioChunk) -> Any:
            # 增加处理计数
            self.processed_count += 1
            # 模拟耗时处理
            import time
            time.sleep(0.01)
            return {"processed": True}

    # 创建处理器
    processor = ThreadPoolProcessor()

    # 创建简单音频（足够长以生成多个块）
    sample_rate = 16000
    audio_data = np.random.normal(0, 0.1, sample_rate * 2)  # 2秒的音频
    audio_config = AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) > 0
    assert processor.processed_count > 0

    # 验证线程池统计
    stats = processor.get_stats()
    assert "completed_tasks" in stats
    assert stats["completed_tasks"] > 0
    assert "failed_tasks" in stats
    assert "active_tasks" in stats

    # 关闭处理器
    processor.shutdown()


if __name__ == "__main__":
    asyncio.run(test_audio_chunking(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_overlap_strategies(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_processing_pipeline(sample_audio(), audio_config(sample_audio())))
    asyncio.run(test_empty_audio(audio_config(sample_audio())))
    asyncio.run(test_short_audio(audio_config(sample_audio())))
    asyncio.run(test_error_handling())
    asyncio.run(test_thread_pool())
    print("所有测试通过！")
