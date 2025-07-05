#!/usr/bin/env python3
"""
测试运行脚本

本脚本用于运行Cascade项目的测试套件，支持以下功能：
- 运行所有测试
- 运行特定模块的测试
- 运行特定标记的测试
- 生成覆盖率报告
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))


def run_tests(module=None, mark=None, coverage=True, verbose=False):
    """运行测试"""
    # 构建命令
    cmd = ["pytest"]

    # 添加模块
    if module:
        cmd.append(f"tests/{module}")

    # 添加标记
    if mark:
        cmd.append(f"-m {mark}")

    # 添加覆盖率
    if coverage:
        cmd.extend([
            "--cov=cascade",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml"
        ])

    # 添加详细输出
    if verbose:
        cmd.append("-v")

    # 运行命令
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=ROOT_DIR)

    return result.returncode


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行Cascade测试套件")
    parser.add_argument(
        "--module", "-m",
        help="要测试的模块 (例如: types, buffer, processor)",
        default=None
    )
    parser.add_argument(
        "--mark",
        help="要运行的测试标记 (例如: unit, integration, benchmark)",
        default=None
    )
    parser.add_argument(
        "--no-coverage",
        help="不生成覆盖率报告",
        action="store_true"
    )
    parser.add_argument(
        "--verbose", "-v",
        help="详细输出",
        action="store_true"
    )

    args = parser.parse_args()

    # 运行测试
    return run_tests(
        module=args.module,
        mark=args.mark,
        coverage=not args.no_coverage,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())
