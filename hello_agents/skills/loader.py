"""
技能加载器

从目录加载 SKILL.md，解析 YAML 元数据与 Markdown 正文。
支持按需加载 reference、resources、scripts。
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple

from .base import Skill


def _parse_yaml_frontmatter(content: str) -> Tuple[dict, str]:
    """解析 YAML 前置内容，返回 (metadata, body)"""
    if not content.strip().startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    yaml_block = parts[1].strip()
    body = parts[2].strip()

    metadata = {}
    for line in yaml_block.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"").strip()
            metadata[key] = value

    return metadata, body


class SkillLoader:
    """技能加载器

    从指定目录加载 SKILL.md，解析为 Skill 对象。
    支持渐进式加载：先加载元数据，再按需加载正文和附属资源。
    """

    SKILL_FILE = "SKILL.md"

    @classmethod
    def load(cls, skill_dir: Path) -> Optional[Skill]:
        """从目录加载单个技能

        Args:
            skill_dir: 技能目录路径，应包含 SKILL.md

        Returns:
            Skill 对象，若 SKILL.md 不存在或解析失败则返回 None
        """
        skill_path = skill_dir / cls.SKILL_FILE
        if not skill_path.exists() or not skill_path.is_file():
            return None

        try:
            content = skill_path.read_text(encoding="utf-8")
        except Exception:
            return None

        metadata, body = _parse_yaml_frontmatter(content)
        name = metadata.get("name", skill_dir.name)
        description = metadata.get("description", "")

        if not description:
            # 尝试从正文第一段提取
            first_line = body.split("\n")[0].strip()
            if first_line and not first_line.startswith("#"):
                description = first_line[:200]

        references = cls._discover_files(skill_dir, ["reference.md", "*.md"], exclude=[cls.SKILL_FILE], root=skill_dir)
        resources = cls._discover_files(skill_dir / "resources", ["*"], root=skill_dir) if (skill_dir / "resources").exists() else []
        scripts = cls._discover_files(skill_dir / "scripts", ["*.py", "*.sh", "*.js"], root=skill_dir) if (skill_dir / "scripts").exists() else []

        return Skill(
            name=name,
            description=description,
            instructions=body,
            path=skill_dir,
            references=references,
            resources=resources,
            scripts=scripts,
            metadata=metadata,
        )

    @classmethod
    def _discover_files(
        cls,
        base: Path,
        patterns: List[str],
        exclude: Optional[List[str]] = None,
        root: Optional[Path] = None,
    ) -> List[str]:
        """发现目录下匹配模式的文件，返回相对于 root 的路径列表"""
        exclude = exclude or []
        root = root or base
        result = []
        if not base.exists() or not base.is_dir():
            return result

        for pattern in patterns:
            for p in base.glob(pattern):
                if p.is_file() and p.name not in exclude:
                    result.append(str(p.relative_to(root)))
        return result

    @classmethod
    def load_resource(cls, skill: Skill, resource_path: str) -> Optional[str]:
        """按需加载技能附属资源

        Args:
            skill: Skill 对象
            resource_path: 相对路径，如 reference.md 或 resources/template.xlsx

        Returns:
            文件内容（文本）或 None
        """
        full_path = skill.path / resource_path
        if not full_path.exists() or not full_path.is_file():
            return None
        try:
            return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
