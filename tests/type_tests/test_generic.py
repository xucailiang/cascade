"""
通用类型单元测试

测试通用类型的功能和验证规则。
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cascade.types import (
    BufferStatus,
    BufferStrategy,
    ErrorCode,
    ErrorInfo,
    ErrorSeverity,
    LogLevel,
    PerformanceMetrics,
    Status,
    SystemStatus,
)


class TestLogLevel:
    """测试LogLevel枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert LogLevel.DEBUG == "debug"
        assert LogLevel.INFO == "info"
        assert LogLevel.WARNING == "warning"
        assert LogLevel.ERROR == "error"
        assert LogLevel.CRITICAL == "critical"


class TestBufferStrategy:
    """测试BufferStrategy枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert BufferStrategy.BLOCK == "block"
        assert BufferStrategy.OVERWRITE == "overwrite"
        assert BufferStrategy.REJECT == "reject"


class TestErrorCode:
    """测试ErrorCode枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        # 通用错误
        assert ErrorCode.UNKNOWN_ERROR == "E0000"
        assert ErrorCode.INVALID_INPUT == "E0001"
        assert ErrorCode.INVALID_CONFIG == "E0002"
        assert ErrorCode.INITIALIZATION_FAILED == "E0003"

        # 音频相关错误
        assert ErrorCode.UNSUPPORTED_FORMAT == "E1001"
        assert ErrorCode.INVALID_SAMPLE_RATE == "E1002"
        assert ErrorCode.INVALID_CHANNELS == "E1003"
        assert ErrorCode.AUDIO_CORRUPTION == "E1004"

        # 缓冲区错误
        assert ErrorCode.BUFFER_FULL == "E2001"
        assert ErrorCode.BUFFER_EMPTY == "E2002"
        assert ErrorCode.INSUFFICIENT_DATA == "E2003"
        assert ErrorCode.BUFFER_CORRUPTION == "E2004"

        # VAD处理错误
        assert ErrorCode.MODEL_LOAD_FAILED == "E3001"
        assert ErrorCode.INFERENCE_FAILED == "E3002"
        assert ErrorCode.RESULT_VALIDATION_FAILED == "E3003"
        assert ErrorCode.BACKEND_UNAVAILABLE == "E3004"

        # 性能相关错误
        assert ErrorCode.TIMEOUT_ERROR == "E4001"
        assert ErrorCode.MEMORY_ERROR == "E4002"
        assert ErrorCode.THREAD_ERROR == "E4003"
        assert ErrorCode.RESOURCE_EXHAUSTED == "E4004"


class TestErrorSeverity:
    """测试ErrorSeverity枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.CRITICAL == "critical"


class TestStatus:
    """测试Status类"""

    def test_default_values(self):
        """测试默认值"""
        status = Status()
        assert status.code == 0
        assert status.message == "OK"
        assert status.success is True
        assert isinstance(status.timestamp, datetime)
        assert status.details is None

    def test_custom_values(self):
        """测试自定义值"""
        details = {"key": "value"}
        status = Status(
            code=404,
            message="Not Found",
            success=False,
            details=details
        )

        assert status.code == 404
        assert status.message == "Not Found"
        assert status.success is False
        assert status.details == details

    def test_is_ok(self):
        """测试is_ok方法"""
        # 成功状态
        status1 = Status(code=0, success=True)
        assert status1.is_ok() is True

        # 失败状态 - 非零状态码
        status2 = Status(code=500, success=True)
        assert status2.is_ok() is False

        # 失败状态 - 成功标志为False
        status3 = Status(code=0, success=False)
        assert status3.is_ok() is False

    def test_is_error(self):
        """测试is_error方法"""
        # 成功状态
        status1 = Status(code=0, success=True)
        assert status1.is_error() is False

        # 失败状态 - 非零状态码
        status2 = Status(code=500, success=True)
        assert status2.is_error() is True

        # 失败状态 - 成功标志为False
        status3 = Status(code=0, success=False)
        assert status3.is_error() is True

    def test_ok_factory(self):
        """测试ok工厂方法"""
        status = Status.ok(message="Success", details={"key": "value"})
        assert status.code == 0
        assert status.message == "Success"
        assert status.success is True
        assert status.details == {"key": "value"}

    def test_error_factory(self):
        """测试error工厂方法"""
        status = Status.error(code=500, message="Server Error", details={"key": "value"})
        assert status.code == 500
        assert status.message == "Server Error"
        assert status.success is False
        assert status.details == {"key": "value"}


class TestPerformanceMetrics:
    """测试PerformanceMetrics类"""

    def test_basic_properties(self):
        """测试基本属性"""
        metrics = PerformanceMetrics(
            avg_latency_ms=10.5,
            p50_latency_ms=8.2,
            p95_latency_ms=15.3,
            p99_latency_ms=20.1,
            max_latency_ms=25.0,
            throughput_qps=100.0,
            throughput_mbps=5.0,
            error_rate=0.01,
            success_count=990,
            error_count=10,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            active_threads=4,
            queue_depth=10,
            buffer_utilization=0.5,
            zero_copy_rate=0.8,
            cache_hit_rate=0.9,
            collection_duration_seconds=60.0
        )

        assert metrics.avg_latency_ms == 10.5
        assert metrics.p50_latency_ms == 8.2
        assert metrics.p95_latency_ms == 15.3
        assert metrics.p99_latency_ms == 20.1
        assert metrics.max_latency_ms == 25.0
        assert metrics.throughput_qps == 100.0
        assert metrics.throughput_mbps == 5.0
        assert metrics.error_rate == 0.01
        assert metrics.success_count == 990
        assert metrics.error_count == 10
        assert metrics.memory_usage_mb == 256.0
        assert metrics.cpu_usage_percent == 45.0
        assert metrics.active_threads == 4
        assert metrics.queue_depth == 10
        assert metrics.buffer_utilization == 0.5
        assert metrics.zero_copy_rate == 0.8
        assert metrics.cache_hit_rate == 0.9
        assert metrics.collection_duration_seconds == 60.0

    def test_validation(self):
        """测试验证规则"""
        # 有效值
        PerformanceMetrics(
            avg_latency_ms=10.5,
            p50_latency_ms=8.2,
            p95_latency_ms=15.3,
            p99_latency_ms=20.1,
            max_latency_ms=25.0,
            throughput_qps=100.0,
            throughput_mbps=5.0,
            error_rate=0.01,
            success_count=990,
            error_count=10,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            active_threads=4,
            queue_depth=10,
            buffer_utilization=0.5,
            zero_copy_rate=0.8,
            cache_hit_rate=0.9,
            collection_duration_seconds=60.0
        )

        # 无效值 - 负延迟
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                avg_latency_ms=-10.5,
                p50_latency_ms=8.2,
                p95_latency_ms=15.3,
                p99_latency_ms=20.1,
                max_latency_ms=25.0,
                throughput_qps=100.0,
                throughput_mbps=5.0,
                error_rate=0.01,
                success_count=990,
                error_count=10,
                memory_usage_mb=256.0,
                cpu_usage_percent=45.0,
                active_threads=4,
                queue_depth=10,
                buffer_utilization=0.5,
                zero_copy_rate=0.8,
                cache_hit_rate=0.9,
                collection_duration_seconds=60.0
            )

        # 无效值 - 错误率超出范围
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                avg_latency_ms=10.5,
                p50_latency_ms=8.2,
                p95_latency_ms=15.3,
                p99_latency_ms=20.1,
                max_latency_ms=25.0,
                throughput_qps=100.0,
                throughput_mbps=5.0,
                error_rate=1.5,  # 超过1.0
                success_count=990,
                error_count=10,
                memory_usage_mb=256.0,
                cpu_usage_percent=45.0,
                active_threads=4,
                queue_depth=10,
                buffer_utilization=0.5,
                zero_copy_rate=0.8,
                cache_hit_rate=0.9,
                collection_duration_seconds=60.0
            )

    def test_get_total_operations(self):
        """测试获取总操作数"""
        metrics = PerformanceMetrics(
            avg_latency_ms=10.5,
            p50_latency_ms=8.2,
            p95_latency_ms=15.3,
            p99_latency_ms=20.1,
            max_latency_ms=25.0,
            throughput_qps=100.0,
            throughput_mbps=5.0,
            error_rate=0.01,
            success_count=990,
            error_count=10,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            active_threads=4,
            queue_depth=10,
            buffer_utilization=0.5,
            zero_copy_rate=0.8,
            cache_hit_rate=0.9,
            collection_duration_seconds=60.0
        )

        assert metrics.get_total_operations() == 1000  # 990 + 10

    def test_get_success_rate(self):
        """测试获取成功率"""
        metrics = PerformanceMetrics(
            avg_latency_ms=10.5,
            p50_latency_ms=8.2,
            p95_latency_ms=15.3,
            p99_latency_ms=20.1,
            max_latency_ms=25.0,
            throughput_qps=100.0,
            throughput_mbps=5.0,
            error_rate=0.01,
            success_count=990,
            error_count=10,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            active_threads=4,
            queue_depth=10,
            buffer_utilization=0.5,
            zero_copy_rate=0.8,
            cache_hit_rate=0.9,
            collection_duration_seconds=60.0
        )

        assert metrics.get_success_rate() == 0.99  # 990 / 1000

    def test_is_healthy(self):
        """测试是否健康"""
        # 健康状态
        metrics1 = PerformanceMetrics(
            avg_latency_ms=10.5,
            p50_latency_ms=8.2,
            p95_latency_ms=15.3,
            p99_latency_ms=20.1,
            max_latency_ms=25.0,
            throughput_qps=100.0,
            throughput_mbps=5.0,
            error_rate=0.005,
            success_count=995,
            error_count=5,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            active_threads=4,
            queue_depth=10,
            buffer_utilization=0.5,
            zero_copy_rate=0.8,
            cache_hit_rate=0.9,
            collection_duration_seconds=60.0
        )
        assert metrics1.is_healthy() is True

        # 不健康状态 - 错误率过高
        metrics2 = PerformanceMetrics(
            avg_latency_ms=10.5,
            p50_latency_ms=8.2,
            p95_latency_ms=15.3,
            p99_latency_ms=20.1,
            max_latency_ms=25.0,
            throughput_qps=100.0,
            throughput_mbps=5.0,
            error_rate=0.02,  # 超过默认阈值0.01
            success_count=980,
            error_count=20,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            active_threads=4,
            queue_depth=10,
            buffer_utilization=0.5,
            zero_copy_rate=0.8,
            cache_hit_rate=0.9,
            collection_duration_seconds=60.0
        )
        assert metrics2.is_healthy() is False

        # 不健康状态 - 吞吐量过低
        metrics3 = PerformanceMetrics(
            avg_latency_ms=10.5,
            p50_latency_ms=8.2,
            p95_latency_ms=15.3,
            p99_latency_ms=20.1,
            max_latency_ms=25.0,
            throughput_qps=0.5,  # 低于默认阈值1.0
            throughput_mbps=5.0,
            error_rate=0.005,
            success_count=995,
            error_count=5,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            active_threads=4,
            queue_depth=10,
            buffer_utilization=0.5,
            zero_copy_rate=0.8,
            cache_hit_rate=0.9,
            collection_duration_seconds=60.0
        )
        assert metrics3.is_healthy() is False

        # 不健康状态 - 延迟过高
        metrics4 = PerformanceMetrics(
            avg_latency_ms=10.5,
            p50_latency_ms=8.2,
            p95_latency_ms=15.3,
            p99_latency_ms=150.0,  # 超过默认阈值100.0
            max_latency_ms=200.0,
            throughput_qps=100.0,
            throughput_mbps=5.0,
            error_rate=0.005,
            success_count=995,
            error_count=5,
            memory_usage_mb=256.0,
            cpu_usage_percent=45.0,
            active_threads=4,
            queue_depth=10,
            buffer_utilization=0.5,
            zero_copy_rate=0.8,
            cache_hit_rate=0.9,
            collection_duration_seconds=60.0
        )
        assert metrics4.is_healthy() is False

        # 自定义健康标准
        assert metrics4.is_healthy(max_latency_p99=200.0) is True


class TestSystemStatus:
    """测试SystemStatus类"""

    def test_basic_properties(self):
        """测试基本属性"""
        status = SystemStatus(
            status="healthy",
            uptime_seconds=3600.0,
            total_processed_chunks=1000,
            total_audio_duration_seconds=600.0,
            current_load=0.5
        )

        assert status.status == "healthy"
        assert status.uptime_seconds == 3600.0
        assert status.total_processed_chunks == 1000
        assert status.total_audio_duration_seconds == 600.0
        assert status.current_load == 0.5
        assert status.health_issues == []
        assert status.last_error is None
        assert status.performance_summary is None

    def test_is_operational(self):
        """测试是否可操作"""
        # 健康状态
        status1 = SystemStatus(
            status="healthy",
            uptime_seconds=3600.0,
            total_processed_chunks=1000,
            total_audio_duration_seconds=600.0,
            current_load=0.5
        )
        assert status1.is_operational() is True

        # 警告状态
        status2 = SystemStatus(
            status="warning",
            uptime_seconds=3600.0,
            total_processed_chunks=1000,
            total_audio_duration_seconds=600.0,
            current_load=0.8
        )
        assert status2.is_operational() is True

        # 严重状态
        status3 = SystemStatus(
            status="critical",
            uptime_seconds=3600.0,
            total_processed_chunks=1000,
            total_audio_duration_seconds=600.0,
            current_load=0.95
        )
        assert status3.is_operational() is False

        # 未知状态
        status4 = SystemStatus(
            status="unknown",
            uptime_seconds=3600.0,
            total_processed_chunks=1000,
            total_audio_duration_seconds=600.0,
            current_load=0.5
        )
        assert status4.is_operational() is False

    def test_add_health_issue(self):
        """测试添加健康问题"""
        status = SystemStatus(
            status="warning",
            uptime_seconds=3600.0,
            total_processed_chunks=1000,
            total_audio_duration_seconds=600.0,
            current_load=0.8
        )

        # 添加问题
        status.add_health_issue("内存使用率过高")
        assert "内存使用率过高" in status.health_issues

        # 添加重复问题
        status.add_health_issue("内存使用率过高")
        assert status.health_issues.count("内存使用率过高") == 1  # 不应重复添加

        # 添加另一个问题
        status.add_health_issue("CPU使用率过高")
        assert "CPU使用率过高" in status.health_issues
        assert len(status.health_issues) == 2

    def test_clear_health_issues(self):
        """测试清除健康问题"""
        status = SystemStatus(
            status="warning",
            uptime_seconds=3600.0,
            total_processed_chunks=1000,
            total_audio_duration_seconds=600.0,
            current_load=0.8
        )

        # 添加问题
        status.add_health_issue("内存使用率过高")
        status.add_health_issue("CPU使用率过高")
        assert len(status.health_issues) == 2

        # 清除问题
        status.clear_health_issues()
        assert len(status.health_issues) == 0


class TestBufferStatus:
    """测试BufferStatus类"""

    def test_basic_properties(self):
        """测试基本属性"""
        status = BufferStatus(
            capacity=10000,
            available_samples=8000,
            free_samples=2000,
            usage_ratio=0.8,
            status_level="normal",
            write_position=8000,
            read_position=0,
            overflow_count=0,
            underflow_count=0,
            peak_usage=0.9
        )

        assert status.capacity == 10000
        assert status.available_samples == 8000
        assert status.free_samples == 2000
        assert status.usage_ratio == 0.8
        assert status.status_level == "normal"
        assert status.write_position == 8000
        assert status.read_position == 0
        assert status.overflow_count == 0
        assert status.underflow_count == 0
        assert status.peak_usage == 0.9

    def test_validation(self):
        """测试验证规则"""
        # 有效值
        BufferStatus(
            capacity=10000,
            available_samples=8000,
            free_samples=2000,
            usage_ratio=0.8,
            status_level="normal",
            write_position=8000,
            read_position=0,
            overflow_count=0,
            underflow_count=0,
            peak_usage=0.9
        )

        # 无效值 - 可用样本数超过容量
        with pytest.raises(ValidationError):
            BufferStatus(
                capacity=10000,
                available_samples=12000,  # 超过容量
                free_samples=2000,
                usage_ratio=0.8,
                status_level="normal",
                write_position=8000,
                read_position=0,
                overflow_count=0,
                underflow_count=0,
                peak_usage=0.9
            )

    def test_is_healthy(self):
        """测试是否健康"""
        # 健康状态
        status1 = BufferStatus(
            capacity=10000,
            available_samples=8000,
            free_samples=2000,
            usage_ratio=0.8,
            status_level="normal",
            write_position=8000,
            read_position=0,
            overflow_count=0,
            underflow_count=0,
            peak_usage=0.9
        )
        assert status1.is_healthy() is True

        # 警告状态 - 仍然健康
        status2 = BufferStatus(
            capacity=10000,
            available_samples=9500,
            free_samples=500,
            usage_ratio=0.95,
            status_level="warning",
            write_position=9500,
            read_position=0,
            overflow_count=0,
            underflow_count=0,
            peak_usage=0.95
        )
        assert status2.is_healthy() is True

        # 不健康状态 - 严重级别
        status3 = BufferStatus(
            capacity=10000,
            available_samples=9900,
            free_samples=100,
            usage_ratio=0.99,
            status_level="critical",
            write_position=9900,
            read_position=0,
            overflow_count=0,
            underflow_count=0,
            peak_usage=0.99
        )
        assert status3.is_healthy() is False

        # 不健康状态 - 溢出
        status4 = BufferStatus(
            capacity=10000,
            available_samples=8000,
            free_samples=2000,
            usage_ratio=0.8,
            status_level="normal",
            write_position=8000,
            read_position=0,
            overflow_count=1,  # 有溢出
            underflow_count=0,
            peak_usage=1.0
        )
        assert status4.is_healthy() is False


class TestErrorInfo:
    """测试ErrorInfo类"""

    def test_basic_properties(self):
        """测试基本属性"""
        error_info = ErrorInfo(
            error_code=ErrorCode.INVALID_INPUT,
            message="无效的输入参数",
            severity=ErrorSeverity.HIGH,
            timestamp=datetime.now(UTC),
            context={"param": "sample_rate", "value": -1},
            stack_trace="...",
            recovery_suggestions=["检查输入参数", "参考文档"]
        )

        assert error_info.error_code == ErrorCode.INVALID_INPUT
        assert error_info.message == "无效的输入参数"
        assert error_info.severity == ErrorSeverity.HIGH
        assert isinstance(error_info.timestamp, datetime)
        assert error_info.context == {"param": "sample_rate", "value": -1}
        assert error_info.stack_trace == "..."
        assert error_info.recovery_suggestions == ["检查输入参数", "参考文档"]
