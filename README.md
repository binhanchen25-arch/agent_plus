# HelloAgents 增强版
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

基于 [hello-agents](https://github.com/jjyaoao/helloagents) 的增强项目，主要增加了 **Skills（技能）** 和 **Middleware（中间件）** 能力。



#本项目采用Attribution-NonCommercial-ShareAlike 4.0 International 切勿商用，根据开源协议内容，商用需要获得[hello-agents](https://github.com/jjyaoao/helloagents)和作者的授权，如果基于本项目进行二次开发要必须注明原作者、来源、协议链接。使用相同协议开源。


## 特性概览

### 1. Skills 技能系统

基于 Anthropic Agent Skills 协议，支持将技能以结构化方式加载并作为工具暴露给 Agent：

- **Skill 数据模型**：`name`、`description`、`instructions`、`path` 等
- **SkillLoader**：从磁盘读取 `SKILL.md`，解析 YAML 元数据与 Markdown 正文
- **SkillRegistry**：统一管理技能，支持批量加载与注册
- **SkillTool**：将技能封装为 Tool，供 FunctionCallAgent 调用
- **SkillAgent**：内置技能加载与注册的专用 Agent

**技能目录结构：**

```
skill-name/
├── SKILL.md          # 主定义（必填）
├── reference.md      # 参考材料（可选）
├── resources/        # 资源文件（可选）
└── scripts/         # 可执行脚本（可选）
```

### 2. Middleware 中间件系统

在 Agent 执行前后插入可扩展逻辑，支持输入/输出处理、限流、安全检测等。

#### 2.1 基类与执行流程

**AgentMiddleware**（`hello_agents.core.AgentMiddleware`）为抽象基类，Agent 在 `run()` 中会调用：

| 方法 | 调用时机 | 返回值含义 |
|------|----------|------------|
| `execute_before(input_text, **kwargs)` | 调用 LLM 之前 | 返回 `str` 则替换输入，返回 `None` 则不修改 |
| `execute_after(input_text, result, **kwargs)` | 获得 LLM 结果之后 | 返回 `str` 则替换输出，返回 `None` 则不修改 |
| `on_error(input_text, error, **kwargs)` | 发生异常时 | 返回 `str` 可作为降级回复 |
| `is_enabled(**kwargs)` | 每次执行前 | 返回 `False` 可跳过该中间件 |

#### 2.2 两种扩展方式

**方式一：传入回调**

```python
def my_before(input_text: str, **kwargs):
    return input_text.strip()

middleware = AgentMiddleware(before_run=my_before, after_run=my_after)
```

**方式二：继承重写钩子**

```python
class MyMiddleware(AgentMiddleware):
    def on_before_run(self, input_text: str, **kwargs) -> Optional[str]:
        return input_text.strip()  # 修改后返回
        # return None  # 不修改

    def on_after_run(self, input_text: str, result: str, **kwargs) -> Optional[str]:
        return result.upper()  # 可修改输出
```

执行顺序：先执行子类 `on_*` 钩子，再执行传入的 `before_run` / `after_run` 回调。

#### 2.3 内置中间件

| 中间件 | 文件 | 功能 | 参数 |
|--------|------|------|------|
| **LoggingMiddleware** | `MiddleWare/LoggingMiddleware.py` | 打印输入前 50 字符、输出长度 | 无 |
| **InputSanitizeMiddleware** | 同上 | 对输入执行 `strip()` | 无 |
| **RateLimitMiddleware** | 同上 | 限制调用次数，超过后 `is_enabled` 返回 False | `_max_calls=10`（可子类重写） |
| **ToxicBertMiddleware** | `MiddleWare/ToxicBertMiddleware.py` | 毒性检测，超阈值则拦截并返回提示语 | `model_id`、`toxicity_threshold`、`block_message` |

**ToxicBertMiddleware 参数说明：**

- `model_id`：模型 ID，默认 `"unitary/toxic-bert"`
- `toxicity_threshold`：毒性阈值 0~1，默认 `0.5`
- `block_message`：拦截时返回的提示，默认 `"检测到输入内容可能包含不当言论，请修改后重试。"`
- 依赖：`pip install transformers torch`

#### 2.4 自定义中间件示例

```python
from hello_agents.core.AgentMiddleware import AgentMiddleware
from typing import Optional

class PrefixMiddleware(AgentMiddleware):
    """在用户输入前添加前缀"""

    def __init__(self, prefix: str = "[用户]", **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def on_before_run(self, input_text: str, **kwargs) -> Optional[str]:
        return f"{self.prefix} {input_text}"
```

#### 2.5 与 Agent 的集成

`Agent` 基类接受 `middleware` 参数，`ReActAgent` 等子类在 `run()` 中会调用 `execute_before` 处理输入。传入 `middleware=None` 时不使用中间件。

## 项目结构

```
HelloAgents/
├── hello_agents/
│   ├── agents/           # Agent 实现（含 skill_agent）
│   ├── core/             # 核心框架（含 AgentMiddleware）
│   ├── MiddleWare/       # 中间件实现
│   ├── skills/           # 技能系统
│   │   ├── base.py       # Skill 数据模型
│   │   ├── loader.py     # 技能加载器
│   │   ├── registry.py   # 技能注册表
│   │   ├── skill_tool.py # 技能转 Tool
│   │   └── builtin/      # 内置示例技能
│   └── ...
├── examples/             # 示例脚本
│   ├── chapter12_skill_agent.py
│   └── agent/skill_agent_demo.py
└── pyproject.toml
```



## 快速开始

### 使用 SkillAgent

```python
from pathlib import Path
from dotenv import load_dotenv
from hello_agents import SkillAgent, HelloAgentsLLM

load_dotenv()
llm = HelloAgentsLLM()

agent = SkillAgent(
    name="skill-assistant",
    llm=llm,
    skill_dir=Path("hello_agents/skills/builtin"),
)

print(agent.list_skills())
answer = agent.run("如何将技能注册到 FunctionCallAgent？")
```

### 使用 Middleware

```python
from hello_agents import ReActAgent, HelloAgentsLLM
from hello_agents.MiddleWare.LoggingMiddleware import LoggingMiddleware, InputSanitizeMiddleware

llm = HelloAgentsLLM()

# 日志 + 输入清洗
middleware = InputSanitizeMiddleware()  # 或 LoggingMiddleware()

agent = ReActAgent(
    name="assistant",
    llm=llm,
    middleware=middleware,
)

agent.run("你好")
```

**毒性检测（需安装 transformers、torch）：**

```python
from hello_agents.MiddleWare.ToxicBertMiddleware import ToxicBertMiddleware

middleware = ToxicBertMiddleware(
    toxicity_threshold=0.5,
    block_message="内容不合规，请修改后重试。",
)
agent = ReActAgent(name="assistant", llm=llm, middleware=middleware)
```


## 依赖

核心依赖：`openai`、`requests`、`python-dotenv`、`pydantic` 等，详见 `pyproject.toml` 和 `requirements.txt`！
