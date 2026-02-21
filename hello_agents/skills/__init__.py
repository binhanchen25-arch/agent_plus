"""
HelloAgents 技能系统

基于 Anthropic Agent Skills 协议，将技能封装为可复用的模块，
通过 SkillTool 继承 Tool，与 FunctionCallAgent 无缝集成。
"""

from .base import Skill
from .loader import SkillLoader
from .registry import SkillRegistry
from .skill_tool import SkillTool

__all__ = [
    "Skill",
    "SkillLoader",
    "SkillRegistry",
    "SkillTool",
]
