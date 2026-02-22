# Core 模块源码解析（详细版）

> HelloAgents 核心抽象层完整解读，涵盖 Agent 基类、LLM 客户端、消息模型、配置与异常体系、数据库配置的逐行级分析。

---

## 一、模块概览与定位

### 1.1 模块职责

`core` 是 hello_agents 的**基础抽象层**，提供：

- **Agent 基类**：所有 Agent 实现的统一接口
- **LLM 客户端**：多提供商、统一的模型调用接口
- **消息模型**：对话消息的标准化表示
- **配置管理**：全局与 Agent 级配置
- **异常体系**：框架内统一的错误处理
- **数据库配置**：记忆系统（Qdrant、Neo4j）的连接配置

### 1.2 目录结构

```
core/
├── __init__.py         # 模块导出（通常为空或重导出）
├── agent.py            # Agent 抽象基类
├── llm.py              # HelloAgentsLLM 统一 LLM 客户端
├── message.py          # Message 消息模型
├── config.py           # Config 配置类
├── exceptions.py       # 异常体系
└── database_config.py  # 数据库配置（Qdrant、Neo4j）
```

### 1.3 依赖关系

```
exceptions.py  ←── llm.py（抛出 HelloAgentsException）
     ↑
     └── agent.py 依赖 message.py、llm.py、config.py
```

---

## 二、agent.py —— Agent 基类

### 2.1 源码结构

```python
# 文件: hello_agents/core/agent.py

from abc import ABC, abstractmethod
from typing import Optional
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config

class Agent(ABC):
    """Agent基类"""
    # ...
```

### 2.2 构造函数详解

```python
def __init__(
    self,
    name: str,
    llm: HelloAgentsLLM,
    system_prompt: Optional[str] = None,
    config: Optional[Config] = None
):
    self.name = name
    self.llm = llm
    self.system_prompt = system_prompt
    self.config = config or Config()
    self._history: list[Message] = []
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | str | 是 | Agent 名称，用于日志与标识 |
| `llm` | HelloAgentsLLM | 是 | LLM 客户端，所有 Agent 必须持有 |
| `system_prompt` | Optional[str] | 否 | 系统提示词，定义角色与行为 |
| `config` | Optional[Config] | 否 | 配置对象，未提供时使用 `Config()` | 默认值 |

**实现细节**：

- `config or Config()`：`config` 为 `None` 时创建默认配置
- `_history` 使用下划线前缀，表示内部实现，子类不应直接依赖其结构

### 2.3 抽象方法 `run`

```python
@abstractmethod
def run(self, input_text: str, **kwargs) -> str:
    """运行Agent"""
    pass
```

- **签名**：`input_text` 为用户输入，`**kwargs` 供子类扩展（如 `max_tool_iterations`、`tool_choice`）
- **返回值**：Agent 的回复字符串
- **子类实现**：必须实现此方法，否则无法实例化

### 2.4 历史管理方法

| 方法 | 签名 | 行为 |
|------|------|------|
| `add_message` | `(message: Message) -> None` | 直接 `append` 到 `_history` |
| `clear_history` | `() -> None` | 调用 `_history.clear()` |
| `get_history` | `() -> list[Message]` | 返回 `_history.copy()`，避免外部修改 |

**注意**：`get_history()` 返回副本，调用方修改返回列表不会影响 Agent 内部状态。

### 2.5 字符串表示

```python
def __str__(self) -> str:
    return f"Agent(name={self.name}, provider={self.llm.provider})"

def __repr__(self) -> str:
    return self.__str__()
```

- `__str__` 展示 `name` 和 `llm.provider`，便于调试
- `__repr__` 与 `__str__` 一致，便于在 REPL 中查看

### 2.6 设计要点与扩展指南

- **依赖注入**：`llm`、`config` 由外部注入，便于单元测试和替换实现
- **历史隔离**：`get_history()` 返回副本，保证内部状态不被篡改
- **最小接口**：仅 `run` 为抽象方法，其余为可选能力
- **扩展方式**：继承 `Agent`，实现 `run()`，必要时重写 `add_message`、`clear_history` 等

---

## 三、llm.py —— HelloAgentsLLM

### 3.1 类定义与设计理念

```python
class HelloAgentsLLM:
    """
    为HelloAgents定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。

    设计理念：
    - 参数优先，环境变量兜底
    - 流式响应为默认，提供更好的用户体验
    - 支持多种LLM提供商
    - 统一的调用接口
    """
```

### 3.2 支持的 Provider 完整列表

```python
SUPPORTED_PROVIDERS = Literal[
    "openai",    # OpenAI 官方 API
    "deepseek",  # DeepSeek
    "qwen",      # 阿里通义千问（DashScope）
    "modelscope",# 魔搭 ModelScope
    "kimi",      # 月之暗面 Kimi
    "zhipu",     # 智谱 AI
    "ollama",    # 本地 Ollama
    "vllm",      # vLLM 部署
    "local",     # 通用本地部署
    "auto",      # 自动检测
    "custom",    # 自定义（完全由用户指定）
]
```

### 3.3 构造函数参数完整说明

```python
def __init__(
    self,
    model: Optional[str] = None,           # 模型 ID，缺省从 LLM_MODEL_ID 读取
    api_key: Optional[str] = None,        # API 密钥
    base_url: Optional[str] = None,        # API 基础 URL
    provider: Optional[SUPPORTED_PROVIDERS] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,         # 缺省从 LLM_TIMEOUT 读取，默认 60
    **kwargs
):
```

**环境变量优先级**：参数 > 环境变量。

| 参数 | 环境变量 | 说明 |
|------|----------|------|
| model | `LLM_MODEL_ID` | 模型名称 |
| api_key | 各 provider 专用或 `LLM_API_KEY` | 见下文凭证解析 |
| base_url | 各 provider 专用或 `LLM_BASE_URL` | 见下文 |
| timeout | `LLM_TIMEOUT` | 默认 60 秒 |

### 3.4 初始化流程（完整）

```
__init__()
    │
    ├─ 1. 加载基础参数
    │      model = model or os.getenv("LLM_MODEL_ID")
    │      temperature, max_tokens, timeout（同上逻辑）
    │      self.kwargs = kwargs  # 保留供 invoke 等传递
    │
    ├─ 2. 确定 provider
    │      requested_provider = (provider or "").lower() if provider else None
    │      self.provider = provider or _auto_detect_provider(api_key, base_url)
    │
    ├─ 3. 若 requested_provider == "custom"
    │      → 直接使用 api_key/base_url（或 LLM_API_KEY/LLM_BASE_URL）
    │      否则 → _resolve_credentials(api_key, base_url) 解析
    │
    ├─ 4. 校验
    │      if not self.model: self.model = _get_default_model()
    │      if not all([self.api_key, self.base_url]):
    │          raise HelloAgentsException("API密钥和服务地址必须被提供...")
    │
    └─ 5. _create_client() 创建 OpenAI 客户端
```

### 3.5 自动检测 Provider 逻辑（`_auto_detect_provider`）

**检测顺序**（一旦命中即返回）：

1. **环境变量优先级**（按顺序检查）：
   - `OPENAI_API_KEY` → openai
   - `DEEPSEEK_API_KEY` → deepseek
   - `DASHSCOPE_API_KEY` → qwen
   - `MODELSCOPE_API_KEY` → modelscope
   - `KIMI_API_KEY` 或 `MOONSHOT_API_KEY` → kimi
   - `ZHIPU_API_KEY` 或 `GLM_API_KEY` → zhipu
   - `OLLAMA_API_KEY` 或 `OLLAMA_HOST` → ollama
   - `VLLM_API_KEY` 或 `VLLM_HOST` → vllm

2. **API Key 格式**（`actual_api_key = api_key or os.getenv("LLM_API_KEY")`）：
   - `ms-` 前缀 → modelscope
   - 全小写等于 `"ollama"` → ollama
   - 全小写等于 `"vllm"` → vllm
   - 全小写等于 `"local"` → local
   - `sk-` 开头且长度 > 50 → 继续（需 base_url 辅助）
   - 末尾含 `.` 或后 20 位含 `.` → zhipu

3. **base_url**（`actual_base_url = base_url or os.getenv("LLM_BASE_URL")`）：
   - `api.openai.com` → openai
   - `api.deepseek.com` → deepseek
   - `dashscope.aliyuncs.com` → qwen
   - `api-inference.modelscope.cn` → modelscope
   - `api.moonshot.cn` → kimi
   - `open.bigmodel.cn` → zhipu
   - `localhost` / `127.0.0.1`：
     - `:11434` 或含 `ollama` → ollama
     - `:8000` 且含 `vllm` → vllm
     - `:8080` / `:7860` → local
     - 否则根据 api_key 判断：`ollama` / `vllm` → 对应；否则 → local
   - 含 `:8080` / `:7860` / `:5000` → local

4. **默认**：`auto`

### 3.6 凭证解析（`_resolve_credentials`）完整映射

| provider | api_key 来源 | base_url 默认值 |
|----------|--------------|-----------------|
| openai | `LLM_API_KEY` | https://api.openai.com/v1 |
| deepseek | DEEPSEEK_API_KEY, LLM_API_KEY | https://api.deepseek.com |
| qwen | DASHSCOPE_API_KEY, LLM_API_KEY | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| modelscope | MODELSCOPE_API_KEY, LLM_API_KEY | https://api-inference.modelscope.cn/v1/ |
| kimi | KIMI_API_KEY, MOONSHOT_API_KEY, LLM_API_KEY | https://api.moonshot.cn/v1 |
| zhipu | ZHIPU_API_KEY, GLM_API_KEY, LLM_API_KEY | https://open.bigmodel.cn/api/paas/v4 |
| ollama | OLLAMA_API_KEY, LLM_API_KEY, 或 "ollama" | OLLAMA_HOST, LLM_BASE_URL, 或 http://localhost:11434/v1 |
| vllm | VLLM_API_KEY, LLM_API_KEY, 或 "vllm" | VLLM_HOST, LLM_BASE_URL, 或 http://localhost:8000/v1 |
| local | LLM_API_KEY, 或 "local" | LLM_BASE_URL, 或 http://localhost:8000/v1 |
| custom | api_key 或 LLM_API_KEY | base_url 或 LLM_BASE_URL |
| auto | LLM_API_KEY | LLM_BASE_URL |

### 3.7 默认模型（`_get_default_model`）完整映射

| provider | 默认模型 |
|----------|----------|
| openai | gpt-3.5-turbo |
| deepseek | deepseek-chat |
| qwen | qwen-plus |
| modelscope | Qwen/Qwen2.5-72B-Instruct |
| kimi | moonshot-v1-8k |
| zhipu | glm-4 |
| ollama | llama3.2 |
| vllm | meta-llama/Llama-2-7b-chat-hf |
| local | local-model |
| custom | model 或 gpt-3.5-turbo |
| auto | 根据 base_url 关键词推断（见源码 309–328 行） |

### 3.8 调用方法详解

#### `think(messages, temperature=None) -> Iterator[str]`

```python
def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
```

- **流式调用**：`stream=True`，逐块 `yield` 文本
- **打印**：会打印 "🧠 正在调用..." 和 "✅ 大语言模型响应成功:"，以及逐块内容
- **异常**：捕获后抛出 `HelloAgentsException("LLM调用失败: ...")`
- **temperature**：`None` 时使用 `self.temperature`

#### `invoke(messages, **kwargs) -> str`

```python
def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
```

- **非流式**：`stream=False`，返回完整响应
- **kwargs 透传**：`temperature`、`max_tokens` 等可覆盖，其余传给 `create()`
- **返回值**：`response.choices[0].message.content`，可能为 `None`（需调用方处理）

#### `stream_invoke(messages, **kwargs) -> Iterator[str]`

- 等价于 `think(messages, kwargs.get('temperature'))`，用于向后兼容

### 3.9 内部客户端 `_client`

- 类型：`openai.OpenAI`
- 创建：`OpenAI(api_key=..., base_url=..., timeout=...)`
- 用途：`FunctionCallAgent` 等通过 `self.llm._client` 直接调用 `chat.completions.create(tools=..., tool_choice=...)` 以支持 function calling

### 3.10 使用示例与注意事项

```python
# 方式 1：显式指定 provider
llm = HelloAgentsLLM(provider="openai", model="gpt-4")

# 方式 2：完全依赖环境变量（需设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL_ID）
llm = HelloAgentsLLM()

# 方式 3：custom 模式
llm = HelloAgentsLLM(provider="custom", base_url="https://my-api.com/v1", model="my-model")
```

**注意**：`think` 会向 stdout 打印，生产环境可考虑重定向或修改源码移除 print。

---

## 四、message.py —— 消息模型

### 4.1 MessageRole

```python
MessageRole = Literal["user", "assistant", "system", "tool"]
```

与 OpenAI Chat Completions API 的 role 一致。

### 4.2 Message 类

```python
class Message(BaseModel):
    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| content | str | 消息正文 |
| role | MessageRole | 角色 |
| timestamp | datetime | 默认 `datetime.now()` |
| metadata | Optional[Dict] | 默认 `{}`，可扩展 |

### 4.3 方法

- **`to_dict() -> Dict`**：返回 `{"role": self.role, "content": self.content}`，仅包含 OpenAI 所需字段，不含 `timestamp`、`metadata`
- **`__str__`**：`"[{role}] {content}"`

### 4.4 与 OpenAI 格式的兼容性

- `to_dict()` 可直接用于 `messages` 列表
- 但 OpenAI 的 `tool` 消息需 `tool_call_id`、`name` 等，当前 `Message` 仅覆盖基础 `role`/`content`，复杂 tool 消息需在 Agent 中单独构建

---

## 五、config.py —— 配置管理

### 5.1 Config 类

```python
class Config(BaseModel):
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100
```

### 5.2 环境变量映射（`from_env`）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| DEBUG | "false" | 转为 bool |
| LOG_LEVEL | "INFO" | 日志级别 |
| TEMPERATURE | "0.7" | 转为 float |
| MAX_TOKENS | 无 | 有则转为 int |

### 5.3 `to_dict()`

- 调用 `self.dict()`，返回 Pydantic 字典

### 5.4 使用说明

- `Config` 主要供 Agent 使用，`HelloAgentsLLM` 的配置来自自身参数，不直接使用 `Config`
- `max_history_length` 可用于限制 Agent 历史长度，需在具体 Agent 实现中处理

---

## 六、exceptions.py —— 异常体系

```python
class HelloAgentsException(Exception):
    """HelloAgents基础异常类"""
    pass

class LLMException(HelloAgentsException): pass
class AgentException(HelloAgentsException): pass
class ConfigException(HelloAgentsException): pass
class ToolException(HelloAgentsException): pass
```

**使用建议**：

- 框架内统一抛出 `HelloAgentsException` 及其子类
- 调用方可按 `LLMException`、`ToolException` 等分类捕获

---

## 七、database_config.py —— 数据库配置

### 7.1 QdrantConfig

| 字段 | 类型 | 默认值 | 环境变量 |
|------|------|--------|----------|
| url | Optional[str] | None | QDRANT_URL |
| api_key | Optional[str] | None | QDRANT_API_KEY |
| collection_name | str | hello_agents_vectors | QDRANT_COLLECTION |
| vector_size | int | 384 | QDRANT_VECTOR_SIZE |
| distance | str | cosine | QDRANT_DISTANCE |
| timeout | int | 30 | QDRANT_TIMEOUT |

- **`to_dict()`**：`model_dump(exclude_none=True)`，排除 None 值

### 7.2 Neo4jConfig

| 字段 | 类型 | 默认值 | 环境变量 |
|------|------|--------|----------|
| uri | str | bolt://localhost:7687 | NEO4J_URI |
| username | str | neo4j | NEO4J_USERNAME |
| password | str | hello-agents-password | NEO4J_PASSWORD |
| database | str | neo4j | NEO4J_DATABASE |
| max_connection_lifetime | int | 3600 | NEO4J_MAX_CONNECTION_LIFETIME |
| max_connection_pool_size | int | 50 | NEO4J_MAX_CONNECTION_POOL_SIZE |
| connection_acquisition_timeout | int | 60 | NEO4J_CONNECTION_TIMEOUT |

### 7.3 DatabaseConfig

- 聚合 `qdrant`、`neo4j` 两个子配置
- **`validate_connections()`**：动态导入 `QdrantVectorStore`、`Neo4jGraphStore`，实例化并调用 `health_check()`，返回 `{"qdrant": bool, "neo4j": bool}`

### 7.4 全局接口

- `db_config`：模块加载时执行 `DatabaseConfig.from_env()`
- `get_database_config()`：返回 `db_config`
- `update_database_config(qdrant=..., neo4j=...)`：更新 `db_config` 的子配置

### 7.5 初始化

- 文件开头 `load_dotenv()`，确保环境变量在 `from_env` 前已加载

---

## 八、模块依赖关系图

```
                    exceptions.py
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
  llm.py              agent.py             config.py
    │                     │                     │
    │                     ├─────────────────────┤
    │                     │                     │
    │                     ▼                     │
    │               message.py ◄───────────────┘
    │                     │
    └─────────────────────┘
              （Agent 依赖 llm、message、config）

database_config.py  （独立，供 memory/storage 使用）
```

---

## 九、完整使用示例

```python
from hello_agents.core import HelloAgentsLLM, Config
from hello_agents.core.agent import Agent
from hello_agents.core.message import Message

# 1. 配置与 LLM
config = Config.from_env()
llm = HelloAgentsLLM(provider="openai", model="gpt-4")

# 2. 自定义 Agent
class MyAgent(Agent):
    def run(self, input_text: str, **kwargs) -> str:
        messages = [{"role": "user", "content": input_text}]
        return self.llm.invoke(messages)

# 3. 运行
agent = MyAgent(name="测试", llm=llm, config=config)
reply = agent.run("你好")
agent.add_message(Message("你好", "user"))
agent.add_message(Message(reply, "assistant"))

# 4. 历史
history = agent.get_history()  # 返回副本
agent.clear_history()
```

---

## 十、常见问题与排查

| 问题 | 可能原因 | 排查建议 |
|------|----------|----------|
| API 密钥错误 | 环境变量未设置或 provider 解析错误 | 检查 `LLM_API_KEY` 或对应 provider 的 env |
| 初始化失败 | api_key 或 base_url 为空 | 确认 `all([api_key, base_url])` 为真 |
| 模型未指定 | model 为空且默认推断失败 | 显式传入 `model` 或设置 `LLM_MODEL_ID` |
| think 打印过多 | 内置 print | 生产环境可重定向或修改源码 |

---

*文档基于 hello_agents core 模块源码整理，适用于深度理解与二次开发。*
