# Context 模块源码总结（详细版）

> HelloAgents 上下文工程模块完整解读，实现 GSSC（Gather-Select-Structure-Compress）流水线，用于从多源信息构建结构化上下文。

---

## 一、模块概览

### 1.1 定位与职责

`context` 模块为 Agent 提供**上下文构建能力**，解决以下问题：

- **信息分散**：记忆、RAG、对话历史等分散在不同系统
- **Token 限制**：LLM 有上下文长度上限，需在预算内选择最相关信息
- **结构混乱**：原始信息缺乏统一格式，不利于 LLM 理解
- **优先级不清**：不同信息源的重要性不同，需合理排序与筛选

通过 GSSC 流水线，将多源信息**收集 → 筛选 → 结构化 → 压缩**，输出可直接注入 prompt 的上下文字符串。

### 1.2 目录结构

```
context/
├── __init__.py    # 导出 ContextBuilder、ContextConfig、ContextPacket
└── builder.py     # GSSC 流水线实现（约 355 行）
```

### 1.3 依赖关系

```
builder.py
├── typing (Dict, Any, List, Optional, Tuple)
├── dataclasses (dataclass, field)
├── datetime (datetime)
├── tiktoken (get_encoding)
├── math (exp)
├── ..core.message (Message)
└── ..tools (MemoryTool, RAGTool)
```

---

## 二、核心数据结构详解

### 2.1 ContextPacket（上下文信息包）

**源码位置**：builder.py 第 21–33 行

```python
@dataclass
class ContextPacket:
    """上下文信息包"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0  # 0.0-1.0
    
    def __post_init__(self):
        """自动计算token数"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| content | str | 必填 | 包内容文本 |
| timestamp | datetime | `datetime.now()` | 创建时间，用于新近性计算 |
| metadata | Dict | `{}` | 元数据，核心字段 `type` 见下表 |
| token_count | int | 0 | token 数，`__post_init__` 中自动计算 |
| relevance_score | float | 0.0 | 相关性分数，Select 阶段写入 |

**metadata.type 取值**：

| type | 来源 | 说明 |
|------|------|------|
| instructions | system_instructions | 系统指令 |
| task_state | memory_tool（任务状态查询） | 任务进展、子目标、结论、阻塞 |
| related_memory | memory_tool（用户查询） | 与当前问题相关的记忆 |
| knowledge_base | rag_tool | RAG 检索结果 |
| history | conversation_history | 对话历史 |
| retrieval | additional_packets | 用户自定义检索结果 |
| tool_result | additional_packets | 用户自定义工具结果 |

**`__post_init__` 行为**：仅当 `token_count == 0` 时调用 `count_tokens(content)`，避免重复计算。

### 2.2 ContextConfig（上下文构建配置）

**源码位置**：builder.py 第 36–48 行

```python
@dataclass
class ContextConfig:
    """上下文构建配置"""
    max_tokens: int = 8000           # 总预算
    reserve_ratio: float = 0.15      # 生成余量（10-20%）
    min_relevance: float = 0.3       # 最小相关性阈值
    enable_mmr: bool = True          # 启用最大边际相关性（多样性）
    mmr_lambda: float = 0.7          # MMR平衡参数（0=纯多样性, 1=纯相关性）
    system_prompt_template: str = "" # 系统提示模板
    enable_compression: bool = True   # 启用压缩
    
    def get_available_tokens(self) -> int:
        """获取可用token预算（扣除余量）"""
        return int(self.max_tokens * (1 - self.reserve_ratio))
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_tokens | 8000 | 总 token 预算（含上下文 + 生成） |
| reserve_ratio | 0.15 | 预留 15% 给模型生成，实际可用约 6800 |
| min_relevance | 0.3 | 相关性低于此值的包在 Select 中被过滤 |
| enable_mmr | True | **未实现**，预留 MMR 多样性筛选 |
| mmr_lambda | 0.7 | **未实现**，MMR 平衡参数 |
| system_prompt_template | "" | **未使用**，预留 |
| enable_compression | True | 超预算时是否执行截断 |

**`get_available_tokens()`**：`int(8000 * 0.85) = 6800`，即实际可用于上下文的 token 数。

---

## 三、GSSC 流水线详解

### 3.1 build 主流程（第 81–120 行）

```python
def build(
    self,
    user_query: str,
    conversation_history: Optional[List[Message]] = None,
    system_instructions: Optional[str] = None,
    additional_packets: Optional[List[ContextPacket]] = None
) -> str:
```

**参数说明**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_query | str | 是 | 当前用户问题，用于记忆/RAG 检索与相关性计算 |
| conversation_history | List[Message] | 否 | 对话历史，默认 `[]`，最多取最近 10 条 |
| system_instructions | str | 否 | 系统指令，作为 P0 最高优先级 |
| additional_packets | List[ContextPacket] | 否 | 额外包，直接追加到 Gather 结果 |

**执行顺序**：

```
packets = _gather(...)           # 收集
selected = _select(packets, ...)  # 筛选
structured = _structure(...)      # 结构化
final = _compress(structured)    # 压缩
return final
```

---

### 3.2 Gather 详解（第 122–201 行）

**方法签名**：

```python
def _gather(
    self,
    user_query: str,
    conversation_history: List[Message],
    system_instructions: Optional[str],
    additional_packets: List[ContextPacket]
) -> List[ContextPacket]:
```

**收集顺序与逻辑**：

#### P0：系统指令（第 132–136 行）

```python
if system_instructions:
    packets.append(ContextPacket(
        content=system_instructions,
        metadata={"type": "instructions"}
    ))
```

- 条件：`system_instructions` 非空
- 不调用外部工具，直接包装为 ContextPacket

#### P1a：任务状态记忆（第 139–154 行）

```python
if self.memory_tool:
    try:
        state_results = self.memory_tool.execute(
            "search",
            query="(任务状态 OR 子目标 OR 结论 OR 阻塞)",
            min_importance=0.7,
            limit=5
        )
        if state_results and "未找到" not in state_results:
            packets.append(ContextPacket(
                content=state_results,
                metadata={"type": "task_state", "importance": "high"}
            ))
```

- **条件**：`memory_tool` 非 None
- **调用**：`memory_tool.execute("search", query=..., min_importance=0.7, limit=5)`
- **过滤**：结果为空或含「未找到」时不加入
- **注意**：MemoryTool 标准接口为 `run({"action": "search", ...})`，若无 `execute` 需适配

#### P1b：相关记忆（第 156–165 行）

```python
related_results = self.memory_tool.execute(
    "search",
    query=user_query,
    limit=5
)
if related_results and "未找到" not in related_results:
    packets.append(ContextPacket(
        content=related_results,
        metadata={"type": "related_memory"}
    ))
```

- **query**：直接使用 `user_query`
- **limit**：5 条

#### P2：RAG 检索（第 168–183 行）

```python
if self.rag_tool:
    try:
        rag_results = self.rag_tool.run({
            "action": "search",
            "query": user_query,
            "limit": 5
        })
        if rag_results and "未找到" not in rag_results and "错误" not in rag_results:
            packets.append(ContextPacket(
                content=rag_results,
                metadata={"type": "knowledge_base"}
            ))
```

- **调用**：`rag_tool.run({"action": "search", "query": user_query, "limit": 5})`
- **过滤**：排除「未找到」「错误」

#### P3：对话历史（第 185–195 行）

```python
if conversation_history:
    recent_history = conversation_history[-10:]
    history_text = "\n".join([
        f"[{msg.role}] {msg.content}"
        for msg in recent_history
    ])
    packets.append(ContextPacket(
        content=history_text,
        metadata={"type": "history", "count": len(recent_history)}
    ))
```

- **截断**：只取最近 10 条
- **格式**：`[user] 内容\n[assistant] 内容`

#### 额外包（第 197–198 行）

```python
packets.extend(additional_packets)
```

- 直接追加，不修改 metadata
- 若需参与 Select，应设置 `metadata["type"]` 为 `retrieval` 或 `tool_result`

**异常处理**：memory/rag 异常时 `print` 警告，不中断，对应源不加入 packets。

---

### 3.3 Select 详解（第 203–257 行）

**方法签名**：

```python
def _select(
    self,
    packets: List[ContextPacket],
    user_query: str
) -> List[ContextPacket]:
```

#### 步骤 1：相关性计算（第 208–216 行）

```python
query_tokens = set(user_query.lower().split())
for packet in packets:
    content_tokens = set(packet.content.lower().split())
    if len(query_tokens) > 0:
        overlap = len(query_tokens & content_tokens)
        packet.relevance_score = overlap / len(query_tokens)
    else:
        packet.relevance_score = 0.0
```

- **公式**：`relevance = |query_tokens ∩ content_tokens| / |query_tokens|`
- **分词**：按空格 `split()`，转小写
- **边界**：`user_query` 为空时，`relevance_score = 0`

#### 步骤 2：新近性计算（第 218–223 行）

```python
def recency_score(ts: datetime) -> float:
    delta = max((datetime.now() - ts).total_seconds(), 0)
    tau = 3600  # 1小时时间尺度
    return math.exp(-delta / tau)
```

- **公式**：`recency = exp(-delta / 3600)`，tau 固定 3600 秒
- **范围**：0（很久以前）～ 1（刚创建）

#### 步骤 3：复合分（第 225–229 行）

```python
for p in packets:
    rec = recency_score(p.timestamp)
    score = 0.7 * p.relevance_score + 0.3 * rec
    scored_packets.append((score, p))
```

- **公式**：`score = 0.7 * relevance + 0.3 * recency`
- **排序**：按 score 降序

#### 步骤 4：分离系统指令（第 231–234 行）

```python
system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "instructions"]
remaining = [p for (s, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
             if p.metadata.get("type") != "instructions"]
```

- 系统指令单独保留，不参与排序
- `remaining` 按分数降序

#### 步骤 5：相关性过滤（第 236–237 行）

```python
filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]
```

- 过滤掉 `relevance_score < min_relevance` 的包
- **系统指令不参与此过滤**（已在 system_packets 中）

#### 步骤 6：预算填充（第 239–256 行）

```python
available_tokens = self.config.get_available_tokens()
selected = []
used_tokens = 0

# 先放系统指令
for p in system_packets:
    if used_tokens + p.token_count <= available_tokens:
        selected.append(p)
        used_tokens += p.token_count

# 再按分数放其余
for p in filtered:
    if used_tokens + p.token_count > available_tokens:
        continue
    selected.append(p)
    used_tokens += p.token_count
```

- 系统指令优先，且不占排序位置
- 其余按分数从高到低，在预算内依次加入
- 超预算的包直接跳过

---

### 3.4 Structure 详解（第 259–313 行）

**方法签名**：

```python
def _structure(
    self,
    selected_packets: List[ContextPacket],
    user_query: str,
    system_instructions: Optional[str]
) -> str:
```

**区块顺序与内容**：

| 区块 | 条件 | 内容格式 |
|------|------|----------|
| [Role & Policies] | 存在 type=instructions | `[Role & Policies]\n` + 指令内容 |
| [Task] | 始终 | `[Task]\n用户问题：{user_query}` |
| [State] | 存在 type=task_state | `[State]\n关键进展与未决问题：\n` + 内容 |
| [Evidence] | 存在 type in {related_memory, knowledge_base, retrieval, tool_result} | `[Evidence]\n事实与引用：\n` + 各包内容 |
| [Context] | 存在 type=history | `[Context]\n对话历史与背景：\n` + 内容 |
| [Output] | 始终 | 固定输出格式约束 |

**Output 固定模板**（第 304–309 行）：

```
[Output]
请按以下格式回答：
1. 结论（简洁明确）
2. 依据（列出支撑证据及来源）
3. 风险与假设（如有）
4. 下一步行动建议（如适用）
```

**拼接**：各区块用 `\n\n` 连接。

---

### 3.5 Compress 详解（第 315–342 行）

**方法签名**：

```python
def _compress(self, context: str) -> str:
```

**逻辑**：

1. 若 `enable_compression=False`，直接返回 `context`
2. 计算 `current_tokens = count_tokens(context)`
3. 若 `current_tokens <= available_tokens`，不压缩
4. 否则：
   - 打印警告
   - 按行 `split("\n")`，逐行累加 token
   - 超过 `available_tokens` 时停止，保留已加入的行
   - 用 `"\n".join(compressed_lines)` 返回

**截断特点**：按行截断，可能在某段中间断开，不保证语义完整。

---

## 四、count_tokens 函数（第 345–352 行）

```python
def count_tokens(text: str) -> int:
    """计算文本token数（使用tiktoken）"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4
```

- **编码**：`cl100k_base`（GPT-4/3.5 等）
- **降级**：异常时按 `len(text) // 4` 估算

---

## 五、ContextBuilder 构造函数（第 71–79 行）

```python
def __init__(
    self,
    memory_tool: Optional[MemoryTool] = None,
    rag_tool: Optional[RAGTool] = None,
    config: Optional[ContextConfig] = None
):
    self.memory_tool = memory_tool
    self.rag_tool = rag_tool
    self.config = config or ContextConfig()
    self._encoding = tiktoken.get_encoding("cl100k_base")
```

- `memory_tool`、`rag_tool` 为 None 时，对应 Gather 步骤跳过
- `_encoding` 在 `count_tokens` 中未直接使用（函数内重新 `get_encoding`），可视为冗余

---

## 六、完整使用示例

```python
from hello_agents.context import ContextBuilder, ContextConfig, ContextPacket
from hello_agents.tools import MemoryTool, RAGTool
from hello_agents.core.message import Message

# 1. 初始化工具
memory_tool = MemoryTool(user_id="user1")
rag_tool = RAGTool()

# 2. 自定义配置
config = ContextConfig(
    max_tokens=8000,
    reserve_ratio=0.15,
    min_relevance=0.3,
    enable_compression=True
)

# 3. 创建构建器
builder = ContextBuilder(
    memory_tool=memory_tool,
    rag_tool=rag_tool,
    config=config
)

# 4. 准备输入
history = [
    Message("什么是机器学习？", "user"),
    Message("机器学习是...", "assistant"),
]
additional = [
    ContextPacket("自定义检索结果", metadata={"type": "retrieval"})
]

# 5. 构建上下文
context = builder.build(
    user_query="深度学习和机器学习有什么区别？",
    conversation_history=history,
    system_instructions="你是一个AI技术专家，请基于上下文给出准确回答。",
    additional_packets=additional
)

# 6. 注入 Agent
# agent.system_prompt = context 或作为 user message 的一部分
```

---

## 七、MemoryTool 接口适配

ContextBuilder 使用 `memory_tool.execute("search", query=..., min_importance=..., limit=...)`，而 MemoryTool 标准接口为：

```python
memory_tool.run({
    "action": "search",
    "query": "...",
    "min_importance": 0.7,
    "limit": 5
})
```

**适配方案**：为 MemoryTool 添加 `execute` 方法：

```python
def execute(self, action: str, **kwargs) -> str:
    params = {"action": action, **kwargs}
    return self.run(params)
```

或在 ContextBuilder 中改为 `memory_tool.run({...})`。

---

## 八、边界情况与注意事项

| 情况 | 行为 |
|------|------|
| user_query 为空 | 所有包 relevance_score=0，仅系统指令和 recency 影响排序 |
| packets 为空 | Select 返回空列表，Structure 仅含 [Task] 和 [Output] |
| 全部包被 min_relevance 过滤 | 仅系统指令（若有）进入 selected |
| 系统指令超预算 | 仍会尝试加入，可能占满预算 |
| RAG/Memory 异常 | print 警告，对应源不加入，流程继续 |
| enable_compression=False 且超预算 | 不截断，直接返回超长上下文 |
| count_tokens 异常 | 降级为 len(text)//4，可能偏差较大 |

---

## 九、未实现与可扩展点

1. **MMR**：`enable_mmr`、`mmr_lambda` 未使用，可增加最大边际相关性筛选
2. **system_prompt_template**：未使用，可支持模板变量
3. **recency tau**：当前固定 3600，可暴露到 ContextConfig
4. **历史条数**：最近 10 条写死，可配置
5. **压缩策略**：可引入 LLM 摘要、语义截断等
6. **多轮 Gather**：可支持工具结果、外部 API 等动态源

---

*文档基于 hello_agents context 模块源码整理，适用于深度理解与二次开发。*
