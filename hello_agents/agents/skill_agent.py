"""SkillAgent - 基于技能的智能体

从技能目录加载 Agent Skills，通过 FunctionCallAgent 实现按需调用。
"""

from pathlib import Path
from typing import Optional, Union, List, TYPE_CHECKING

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from .function_call_agent import FunctionCallAgent

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry
    from ..skills.registry import SkillRegistry


class SkillAgent(FunctionCallAgent):
    """基于技能的智能体

    从指定目录加载 Agent Skills（SKILL.md），将技能注册为工具，
    通过 FunctionCallAgent 的 function call 机制实现渐进式披露：
    LLM 按需调用技能，获取完整 Instructions。

    使用示例:
        >>> from hello_agents import SkillAgent, HelloAgentsLLM
        >>> from pathlib import Path
        >>>
        >>> llm = HelloAgentsLLM(model="gpt-4o-mini")
        >>> agent = SkillAgent(
        ...     name="skill-assistant",
        ...     llm=llm,
        ...     skill_dir=Path(".cursor/skills"),
        ... )
        >>> agent.run("如何创建 Cursor 规则？")
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        skill_dir: Optional[Union[str, Path]] = None,
        skill_registry: Optional["SkillRegistry"] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        max_tool_iterations: int = 3,
    ):
        """初始化 SkillAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            skill_dir: 技能目录路径，将从此目录加载所有包含 SKILL.md 的子目录
            skill_registry: 已加载的 SkillRegistry（与 skill_dir 二选一）
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 可选的已有 ToolRegistry，若不提供则自动创建
            max_tool_iterations: 最大工具调用迭代次数
        """
        from ..tools.registry import ToolRegistry
        from ..skills.registry import SkillRegistry

        # 确定 ToolRegistry
        if tool_registry is None:
            tool_registry = ToolRegistry()

        # 加载技能
        registry = skill_registry or SkillRegistry()
        if skill_dir is not None:
            base_dir = Path(skill_dir)
            registry.load_from_directory(base_dir)

        # 将技能注册为工具
        registry.register_to_tool_registry(tool_registry)

        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or _default_skill_system_prompt(),
            config=config,
            tool_registry=tool_registry,
            enable_tool_calling=True,
            max_tool_iterations=max_tool_iterations,
        )
        self.skill_registry = registry

    def list_skills(self) -> List[str]:
        """列出已加载的技能名称"""
        return self.skill_registry.list_skills()


def _default_skill_system_prompt() -> str:
    return (
        "你是一个可靠的 AI 助理，能够通过调用技能获取领域专业知识。"
        "当用户问题涉及特定领域（如编码规范、规则创建、文档处理等）时，"
        "请主动调用相关技能获取详细指导，再基于技能内容给出回答。"
    )
