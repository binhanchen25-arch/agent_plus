"""
SkillTool - 技能作为工具

继承 Tool 基类，将 Skill 暴露为 FunctionCallAgent 可调用的工具。
实现渐进式披露：LLM 通过 function call 按需获取完整 Instructions。
"""

from typing import Dict, Any, List, Optional

from ..tools.base import Tool, ToolParameter
from .base import Skill
from .loader import SkillLoader


class SkillTool(Tool):
    """技能工具 - 继承 Tool，将技能暴露给 LLM

    层级 1：LLM 看到 name 和 description（通过 function schema）
    层级 2：LLM 发起 function call 时，run() 返回完整 Instructions
    层级 3：可通过 context 参数请求加载附属资源（reference、resources）
    """

    def __init__(self, skill: Skill):
        """初始化技能工具

        Args:
            skill: 已加载的 Skill 对象
        """
        super().__init__(
            name=skill.name,
            description=skill.description or f"技能：{skill.name}"
        )
        self.skill = skill

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数

        提供可选的 context 和 load_resources 参数，支持按需加载。
        """
        return [
            ToolParameter(
                name="context",
                type="string",
                description="当前任务上下文，帮助技能更好地提供指导",
                required=False
            ),
            ToolParameter(
                name="load_resources",
                type="string",
                description="需要加载的附属资源，逗号分隔，如 reference.md,resources/data.json",
                required=False
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行技能工具

        返回完整 Instructions，并按需加载附属资源。
        """
        result_parts = [f"# {self.skill.name}\n\n", self.skill.instructions]

        load_resources = parameters.get("load_resources", "")
        if load_resources:
            for path in [p.strip() for p in load_resources.split(",") if p.strip()]:
                content = SkillLoader.load_resource(self.skill, path)
                if content:
                    result_parts.append(f"\n\n---\n## 资源: {path}\n\n")
                    result_parts.append(content)

        return "".join(result_parts)
