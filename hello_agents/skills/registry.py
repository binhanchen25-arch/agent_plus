"""
技能注册表

管理技能的加载、注册，并支持将技能注册到 ToolRegistry 供 FunctionCallAgent 使用。
"""

from pathlib import Path
from typing import Optional, Dict, List, TYPE_CHECKING

from .base import Skill
from .loader import SkillLoader
from .skill_tool import SkillTool

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class SkillRegistry:
    """技能注册表

    从目录批量加载技能，并可注册到 ToolRegistry，
    使 FunctionCallAgent 能够通过 function call 按需调用技能。
    """

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def load(self, skill_dir: Path) -> Optional[Skill]:
        """加载单个技能目录

        Args:
            skill_dir: 技能目录路径

        Returns:
            加载成功的 Skill，失败返回 None
        """
        skill = SkillLoader.load(skill_dir)
        if skill:
            self._skills[skill.name] = skill
            print(f"✅ 技能 '{skill.name}' 已加载。")
        return skill

    def load_from_directory(self, base_dir: Path) -> List[Skill]:
        """从目录批量加载技能

        扫描 base_dir 下所有包含 SKILL.md 的子目录。

        Args:
            base_dir: 技能根目录，如 .cursor/skills 或 projects/skills

        Returns:
            成功加载的技能列表
        """
        loaded = []
        if not base_dir.exists() or not base_dir.is_dir():
            print(f"⚠️ 技能目录不存在: {base_dir}")
            return loaded

        for item in base_dir.iterdir():
            if item.is_dir():
                skill = self.load(item)
                if skill:
                    loaded.append(skill)

        return loaded

    def get_skill(self, name: str) -> Optional[Skill]:
        """获取已加载的技能"""
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """列出所有已加载技能名称"""
        return list(self._skills.keys())

    def register_to_tool_registry(self, tool_registry: "ToolRegistry", auto_expand: bool = False) -> int:
        """将技能注册到工具注册表

        每个技能包装为 SkillTool 并注册，供 FunctionCallAgent 使用。

        Args:
            tool_registry: HelloAgents ToolRegistry 实例
            auto_expand: 是否展开（Skill 不展开，保持 False）

        Returns:
            注册的技能数量
        """
        count = 0
        for skill in self._skills.values():
            skill_tool = SkillTool(skill)
            tool_registry.register_tool(skill_tool, auto_expand=auto_expand)
            count += 1
        return count

    def clear(self) -> None:
        """清空已加载技能"""
        self._skills.clear()
        print("🧹 技能注册表已清空。")
