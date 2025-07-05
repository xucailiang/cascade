"""
Pytest配置文件

本文件包含了Pytest的全局配置和共享fixture。
"""

import os
import sys

import numpy as np
import pytest

# 确保cascade包可以被导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# === 通用测试工具 ===

@pytest.fixture
def sample_audio_data():
    """生成用于测试的样本音频数据"""
    # 创建一个1秒、16kHz的正弦波
    sample_rate = 16000
    duration_seconds = 1.0
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    # 生成1kHz的正弦波
    data = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    return data, sample_rate


@pytest.fixture
def sample_vad_config():
    """生成用于测试的VAD配置"""
    from cascade.types import VADConfig

    return VADConfig(
        backend="onnx",
        workers=2,
        threshold=0.5,
        chunk_duration_ms=500,
        overlap_ms=16,
        buffer_capacity_seconds=5,
        processing_mode="streaming",
        optimization_level="all"
    )


@pytest.fixture
def sample_audio_config():
    """生成用于测试的音频配置"""
    from cascade.types import AudioConfig

    return AudioConfig(
        sample_rate=16000,
        format="wav",
        channels=1,
        dtype="float32"
    )


@pytest.fixture
def temp_audio_file(tmp_path, sample_audio_data):
    """创建临时音频文件用于测试"""
    import wave

    data, sample_rate = sample_audio_data
    file_path = tmp_path / "test_audio.wav"

    with wave.open(str(file_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # 将float32转换为int16
        data_int16 = (data * 32767).astype(np.int16)
        wf.writeframes(data_int16.tobytes())

    return file_path


# === 性能测试工具 ===

@pytest.fixture
def benchmark_config():
    """性能基准测试配置"""
    return {
        "iterations": 100,
        "warmup_iterations": 10,
        "timeout_seconds": 60,
        "metrics": ["latency", "throughput", "memory"]
    }


# === 测试标记配置 ===

def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "benchmark: 性能基准测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
    config.addinivalue_line("markers", "audio: 需要音频文件的测试")


# === 测试报告定制 ===

def pytest_html_report_title(report):
    """自定义HTML报告标题"""
    report.title = "Cascade测试报告"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """自定义终端摘要报告"""
    if hasattr(terminalreporter, 'stats') and 'passed' in terminalreporter.stats:
        terminalreporter.write_sep("=", "测试覆盖率摘要")
        try:
            import coverage
            cov = coverage.Coverage()
            cov.load()
            total_coverage = cov.report(show_missing=False)
            terminalreporter.write_line(f"总覆盖率: {total_coverage:.2f}%")

            # 显示各模块覆盖率
            terminalreporter.write_sep("-", "模块覆盖率")
            for module in ["types", "buffer", "formats", "processor", "backends"]:
                module_coverage = cov.report(include=f"cascade/{module}/*", show_missing=False)
                terminalreporter.write_line(f"{module}: {module_coverage:.2f}%")
        except ImportError:
            terminalreporter.write_line("未安装coverage包，无法显示覆盖率信息")
        except Exception as e:
            terminalreporter.write_line(f"获取覆盖率信息失败: {str(e)}")
