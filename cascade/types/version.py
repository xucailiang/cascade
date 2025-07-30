"""
版本和兼容性类型系统

定义Cascade项目中版本管理、兼容性检查相关的类型。
"""

import platform
import re
import sys

from pydantic import BaseModel, Field

# === 版本和兼容性类型 ===

class VersionInfo(BaseModel):
    """版本信息"""
    major: int = Field(description="主版本号", ge=0)
    minor: int = Field(description="次版本号", ge=0)
    patch: int = Field(description="补丁版本号", ge=0)
    pre_release: str | None = Field(default=None, description="预发布标识")
    build_metadata: str | None = Field(default=None, description="构建元数据")

    def __str__(self) -> str:
        """版本字符串表示"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build_metadata:
            version += f"+{self.build_metadata}"
        return version

    def is_compatible_with(self, other: 'VersionInfo') -> bool:
        """检查版本兼容性"""
        # 主版本号相同才兼容
        if self.major != other.major:
            return False
        # 当前版本应该 >= 其他版本
        return (self.minor, self.patch) >= (other.minor, other.patch)

    @classmethod
    def parse(cls, version_string: str) -> 'VersionInfo':
        """解析版本字符串"""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([^+]+))?(?:\+(.+))?$'
        match = re.match(pattern, version_string)

        if not match:
            raise ValueError(f"无效的版本字符串: {version_string}")

        major, minor, patch, pre_release, build_metadata = match.groups()

        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            pre_release=pre_release,
            build_metadata=build_metadata
        )

class CompatibilityInfo(BaseModel):
    """兼容性信息"""
    min_python_version: str = Field(description="最小Python版本")
    supported_platforms: list[str] = Field(description="支持的平台")
    required_dependencies: dict[str, str] = Field(description="必需依赖")
    optional_dependencies: dict[str, str] = Field(default={}, description="可选依赖")
    api_version: VersionInfo = Field(description="API版本")

    def check_python_compatibility(self) -> bool:
        """检查Python版本兼容性"""
        current_version = VersionInfo.parse(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        min_version = VersionInfo.parse(self.min_python_version)
        return current_version.is_compatible_with(min_version)

    def check_platform_compatibility(self) -> bool:
        """检查平台兼容性"""
        current_platform = platform.system().lower()
        return any(current_platform in supported.lower() for supported in self.supported_platforms)

__all__ = [
    "VersionInfo", "CompatibilityInfo"
]
