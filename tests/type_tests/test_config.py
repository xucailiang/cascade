"""
后端配置类型单元测试

测试后端配置相关类型的功能和验证规则。
"""

import pytest
from pydantic import ValidationError

from cascade.types import BackendConfig, ONNXConfig, OptimizationLevel, VLLMConfig


class TestBackendConfig:
    """测试BackendConfig类"""

    def test_default_values(self):
        """测试默认值"""
        config = BackendConfig()
        assert config.model_path is None
        assert config.device == "cpu"
        assert config.optimization_level == "all"
        assert config.max_batch_size == 1
        assert config.warmup_iterations == 3

    def test_custom_values(self):
        """测试自定义值"""
        config = BackendConfig(
            model_path="/path/to/model",
            device="cuda",
            optimization_level=OptimizationLevel.BASIC,
            max_batch_size=8,
            warmup_iterations=5
        )

        assert config.model_path == "/path/to/model"
        assert config.device == "cuda"
        assert config.optimization_level == "basic"
        assert config.max_batch_size == 8
        assert config.warmup_iterations == 5

    def test_validation(self):
        """测试验证规则"""
        # 有效值
        BackendConfig(
            model_path="/path/to/model",
            device="cuda",
            optimization_level=OptimizationLevel.BASIC,
            max_batch_size=8,
            warmup_iterations=5
        )

        # 无效值 - 批处理大小超出范围
        with pytest.raises(ValidationError):
            BackendConfig(max_batch_size=0)  # 小于最小值1

        with pytest.raises(ValidationError):
            BackendConfig(max_batch_size=65)  # 大于最大值64

        # 无效值 - 预热迭代次数超出范围
        with pytest.raises(ValidationError):
            BackendConfig(warmup_iterations=-1)  # 小于最小值0

        with pytest.raises(ValidationError):
            BackendConfig(warmup_iterations=11)  # 大于最大值10

    def test_extra_fields(self):
        """测试额外字段"""
        # BackendConfig允许额外字段
        config = BackendConfig(
            model_path="/path/to/model",
            custom_field="custom_value"
        )

        assert config.model_path == "/path/to/model"
        assert config.custom_field == "custom_value"  # 额外字段


class TestONNXConfig:
    """测试ONNXConfig类"""

    def test_default_values(self):
        """测试默认值"""
        config = ONNXConfig()
        assert config.model_path is None
        assert config.device == "cpu"
        assert config.optimization_level == "all"
        assert config.max_batch_size == 1
        assert config.warmup_iterations == 3
        assert config.providers == ["CPUExecutionProvider"]
        assert config.intra_op_num_threads == 1
        assert config.inter_op_num_threads == 1
        assert config.execution_mode == "sequential"
        assert config.graph_optimization_level == "all"

    def test_custom_values(self):
        """测试自定义值"""
        config = ONNXConfig(
            model_path="/path/to/model.onnx",
            device="cuda",
            optimization_level=OptimizationLevel.BASIC,
            max_batch_size=8,
            warmup_iterations=5,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            intra_op_num_threads=4,
            inter_op_num_threads=2,
            execution_mode="parallel",
            graph_optimization_level="basic"
        )

        assert config.model_path == "/path/to/model.onnx"
        assert config.device == "cuda"
        assert config.optimization_level == "basic"
        assert config.max_batch_size == 8
        assert config.warmup_iterations == 5
        assert config.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert config.intra_op_num_threads == 4
        assert config.inter_op_num_threads == 2
        assert config.execution_mode == "parallel"
        assert config.graph_optimization_level == "basic"

    def test_providers_validation(self):
        """测试执行提供者验证"""
        # 有效提供者
        ONNXConfig(providers=["CPUExecutionProvider"])
        ONNXConfig(providers=["CUDAExecutionProvider"])
        ONNXConfig(providers=["TensorrtExecutionProvider"])
        ONNXConfig(providers=["OpenVINOExecutionProvider"])
        ONNXConfig(providers=["CPUExecutionProvider", "CUDAExecutionProvider"])

        # 无效提供者
        with pytest.raises(ValidationError):
            ONNXConfig(providers=["InvalidProvider"])

        with pytest.raises(ValidationError):
            ONNXConfig(providers=["CPUExecutionProvider", "InvalidProvider"])

    def test_thread_validation(self):
        """测试线程数验证"""
        # 有效线程数
        ONNXConfig(intra_op_num_threads=1)
        ONNXConfig(intra_op_num_threads=8)
        ONNXConfig(intra_op_num_threads=16)

        ONNXConfig(inter_op_num_threads=1)
        ONNXConfig(inter_op_num_threads=8)
        ONNXConfig(inter_op_num_threads=16)

        # 无效线程数 - 小于最小值
        with pytest.raises(ValidationError):
            ONNXConfig(intra_op_num_threads=0)

        with pytest.raises(ValidationError):
            ONNXConfig(inter_op_num_threads=0)

        # 无效线程数 - 大于最大值
        with pytest.raises(ValidationError):
            ONNXConfig(intra_op_num_threads=17)

        with pytest.raises(ValidationError):
            ONNXConfig(inter_op_num_threads=17)


class TestVLLMConfig:
    """测试VLLMConfig类"""

    def test_default_values(self):
        """测试默认值"""
        config = VLLMConfig()
        assert config.model_path is None
        assert config.device == "cpu"
        assert config.optimization_level == "all"
        assert config.max_batch_size == 1
        assert config.warmup_iterations == 3
        assert config.tensor_parallel_size == 1
        assert config.max_model_len == 2048
        assert config.gpu_memory_utilization == 0.9
        assert config.swap_space == 4
        assert config.dtype == "auto"

    def test_custom_values(self):
        """测试自定义值"""
        config = VLLMConfig(
            model_path="/path/to/model",
            device="cuda",
            optimization_level=OptimizationLevel.BASIC,
            max_batch_size=8,
            warmup_iterations=5,
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            swap_space=8,
            dtype="float16"
        )

        assert config.model_path == "/path/to/model"
        assert config.device == "cuda"
        assert config.optimization_level == "basic"
        assert config.max_batch_size == 8
        assert config.warmup_iterations == 5
        assert config.tensor_parallel_size == 2
        assert config.max_model_len == 4096
        assert config.gpu_memory_utilization == 0.8
        assert config.swap_space == 8
        assert config.dtype == "float16"

    def test_tensor_parallel_validation(self):
        """测试张量并行大小验证"""
        # 有效值
        VLLMConfig(tensor_parallel_size=1)
        VLLMConfig(tensor_parallel_size=4)
        VLLMConfig(tensor_parallel_size=8)

        # 无效值 - 小于最小值
        with pytest.raises(ValidationError):
            VLLMConfig(tensor_parallel_size=0)

        # 无效值 - 大于最大值
        with pytest.raises(ValidationError):
            VLLMConfig(tensor_parallel_size=9)

    def test_model_len_validation(self):
        """测试模型长度验证"""
        # 有效值
        VLLMConfig(max_model_len=512)
        VLLMConfig(max_model_len=2048)
        VLLMConfig(max_model_len=8192)

        # 无效值 - 小于最小值
        with pytest.raises(ValidationError):
            VLLMConfig(max_model_len=511)

        # 无效值 - 大于最大值
        with pytest.raises(ValidationError):
            VLLMConfig(max_model_len=8193)

    def test_memory_utilization_validation(self):
        """测试内存利用率验证"""
        # 有效值
        VLLMConfig(gpu_memory_utilization=0.1)
        VLLMConfig(gpu_memory_utilization=0.5)
        VLLMConfig(gpu_memory_utilization=1.0)

        # 无效值 - 小于最小值
        with pytest.raises(ValidationError):
            VLLMConfig(gpu_memory_utilization=0.09)

        # 无效值 - 大于最大值
        with pytest.raises(ValidationError):
            VLLMConfig(gpu_memory_utilization=1.1)

    def test_swap_space_validation(self):
        """测试交换空间验证"""
        # 有效值
        VLLMConfig(swap_space=0)
        VLLMConfig(swap_space=16)
        VLLMConfig(swap_space=32)

        # 无效值 - 小于最小值
        with pytest.raises(ValidationError):
            VLLMConfig(swap_space=-1)

        # 无效值 - 大于最大值
        with pytest.raises(ValidationError):
            VLLMConfig(swap_space=33)

    def test_dtype_validation(self):
        """测试数据类型验证"""
        # 有效值
        VLLMConfig(dtype="auto")
        VLLMConfig(dtype="half")
        VLLMConfig(dtype="float16")
        VLLMConfig(dtype="bfloat16")
        VLLMConfig(dtype="float")
        VLLMConfig(dtype="float32")

        # 无效值
        with pytest.raises(ValidationError):
            VLLMConfig(dtype="int32")

        with pytest.raises(ValidationError):
            VLLMConfig(dtype="double")
