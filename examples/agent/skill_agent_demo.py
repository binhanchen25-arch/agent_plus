"""SkillAgent 示例 - 基于 Agent Skills 的智能体

演示如何从技能目录加载 SKILL.md，并通过 SkillAgent 按需调用技能。
"""

from pathlib import Path

from hello_agents import SkillAgent, HelloAgentsLLM


def main() -> None:
    # 需提前配置 OPENAI_API_KEY
    llm = HelloAgentsLLM(model="gpt-4o-mini")

    # 技能目录：使用内置示例技能，或指定 .cursor/skills 等
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    skill_dir = project_root / "hello_agents" / "skills" / "builtin"

    agent = SkillAgent(
        name="skill-assistant",
        llm=llm,
        skill_dir=skill_dir,
    )

    print("已加载技能:", agent.list_skills())
    print()

    # 提问 - Agent 会按需调用 example-skill 获取指导
    question = "如何将技能注册到 FunctionCallAgent？请简要说明步骤。"
    print("用户:", question)
    print()
    answer = agent.run(question)
    print("SkillAgent:", answer)


if __name__ == "__main__":
    main()
