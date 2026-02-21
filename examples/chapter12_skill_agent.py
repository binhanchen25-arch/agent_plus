"""
第12章示例：SkillAgent - 基于 Agent Skills 的智能体

本示例展示如何：
1. 从技能目录加载 SKILL.md（Anthropic Agent Skills 协议）
2. 使用 SkillAgent 按需调用技能
3. 技能作为 Tool 暴露给 LLM，实现渐进式披露

技能目录结构:
    skill-name/
    ├── SKILL.md          # 主定义（必填）
    ├── reference.md      # 参考材料（可选）
    ├── resources/        # 资源文件（可选）
    └── scripts/         # 可执行脚本（可选）
"""

from pathlib import Path

from dotenv import load_dotenv

from hello_agents import SkillAgent, HelloAgentsLLM


def main() -> None:
    load_dotenv()
    print("=" * 60)
    print("SkillAgent 演示 - 基于 Agent Skills 的智能体")
    print("=" * 60)

    llm = HelloAgentsLLM()

    # 使用内置示例技能目录
    script_dir = Path(__file__).resolve().parent
    skill_dir = script_dir.parent / "hello_agents" / "skills" / "builtin"

    agent = SkillAgent(
        name="skill-assistant",
        llm=llm,
        skill_dir=skill_dir,
    )

    print("\n已加载技能:", agent.list_skills())
    print()

    # 示例问题
    questions = [
        "如何将技能注册到 FunctionCallAgent？",
        "example-skill 这个技能是做什么的？",
    ]

    for q in questions:
        print("用户:", q)
        answer = agent.run(q)
        print("SkillAgent:", answer[:300] + "..." if len(answer) > 300 else answer)
        print()


if __name__ == "__main__":
    main()
