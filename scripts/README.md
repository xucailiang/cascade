# Cascade 项目脚本

本目录包含了Cascade项目的各种实用脚本，用于自动化开发、测试和部署流程。

## 测试脚本

### run_tests.py

这个脚本用于运行Cascade项目的测试套件，支持以下功能：
- 运行所有测试
- 运行特定模块的测试
- 运行特定标记的测试
- 生成覆盖率报告

#### 使用方法

```bash
# 运行所有测试
./scripts/run_tests.py

# 运行特定模块的测试
./scripts/run_tests.py --module types

# 运行特定标记的测试
./scripts/run_tests.py --mark unit

# 不生成覆盖率报告
./scripts/run_tests.py --no-coverage

# 详细输出
./scripts/run_tests.py --verbose
```

#### 参数说明

- `--module`, `-m`: 要测试的模块 (例如: types, buffer, processor)
- `--mark`: 要运行的测试标记 (例如: unit, integration, benchmark)
- `--no-coverage`: 不生成覆盖率报告
- `--verbose`, `-v`: 详细输出

#### 示例

```bash
# 运行types模块的单元测试，并生成覆盖率报告
./scripts/run_tests.py --module types --mark unit

# 运行所有测试，详细输出，不生成覆盖率报告
./scripts/run_tests.py --verbose --no-coverage
```

## 其他脚本

未来将添加更多脚本，包括：

- `build.py`: 构建项目
- `benchmark.py`: 运行性能基准测试
- `docs_gen.py`: 生成文档
- `release.py`: 发布新版本