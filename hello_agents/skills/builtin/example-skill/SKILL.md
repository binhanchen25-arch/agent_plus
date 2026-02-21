---
name: example-skill
description: 示例技能，演示如何编写 Agent Skills。用于创建规则、编码规范或项目约定时使用。
---

# 示例技能

## 快速开始

这是一个符合 Anthropic Agent Skills 协议的示例技能。

## 使用说明

1. 将技能目录放置在 `.cursor/skills/` 或项目指定的技能目录下
2. 使用 `SkillRegistry.load_from_directory()` 加载
3. 通过 `register_to_tool_registry()` 注册到 FunctionCallAgent

## 示例

```python
from pathlib import Path
from hello_agents import SkillRegistry, ToolRegistry, FunctionCallAgent, HelloAgentsLLM

registry = SkillRegistry()
registry.load_from_directory(Path("path/to/skills"))

tool_registry = ToolRegistry()
registry.register_to_tool_registry(tool_registry)

agent = FunctionCallAgent(name="assistant", llm=llm, tool_registry=tool_registry)
```
