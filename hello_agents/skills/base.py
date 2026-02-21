"""
技能数据模型

基于 Anthropic Agent Skills 协议，定义技能的结构化表示。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class Skill:
    """技能数据模型

    对应 SKILL.md 解析后的结构化数据。
    遵循 Anthropic Agent Skills 协议。
    """

    name: str
    """技能名称，来自 YAML 元数据"""

    description: str
    """技能描述，用于 LLM 发现与检索"""

    instructions: str
    """SKILL.md 正文内容（Instructions、Examples、Guidelines 等）"""

    path: Path
    """技能目录路径"""

    references: List[str] = field(default_factory=list)
    """附加参考文件列表，如 reference.md"""

    resources: List[str] = field(default_factory=list)
    """resources/ 目录下的资源文件列表"""

    scripts: List[str] = field(default_factory=list)
    """scripts/ 目录下的可执行脚本列表"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """YAML 元数据中的其他字段"""

    def __str__(self) -> str:
        return f"Skill(name={self.name})"
