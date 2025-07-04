"""
Cascade 测试套件

本测试套件包含了对Cascade项目各个模块的测试，确保功能正确性和代码质量。
测试组织结构：
- types/: 类型系统测试
- buffer/: 缓冲区模块测试
- formats/: 格式处理模块测试
- processor/: 处理器模块测试
- backends/: 后端模块测试
- _internal/: 内部工具模块测试
- integration/: 集成测试
- benchmarks/: 性能基准测试

运行测试：
$ pytest                  # 运行所有测试
$ pytest tests/types/     # 运行特定模块测试
$ pytest -m unit          # 运行单元测试
$ pytest -m integration   # 运行集成测试
$ pytest -m benchmark     # 运行性能基准测试
"""