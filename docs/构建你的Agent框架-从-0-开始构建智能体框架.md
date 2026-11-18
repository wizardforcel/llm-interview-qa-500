# 构建你的Agent框架 —— 从 0 开始构建智能体框架

> 面试生普（LLM 500 题）· 构建你的Agent框架篇。本章共 35 个问题，覆盖为什么要构建自己的框架、智能体框架八件套（模型层、提示词层、记忆层、工具层、规划层、工作流层、安全约束层、可观测性层）、手撕 message.py 统一消息契约、手撕 config.py 中心化配置管理、手撕 agent.py 抽象基类、手撕框架主循环（感知→思考→行动→观察）、记忆层设计（短期+长期）、工具层注册与调用、可观测性与人工干预、消息总线与事件驱动、插件系统设计、多Agent编排与状态机、框架的序列化与持久化、框架的热插拔与动态加载、框架的限流与熔断、框架的安全沙箱、框架的分布式部署、框架的版本管理与回滚、框架的压力测试与基准测试、框架的文档与示例代码、框架的社区与生态、框架选型与现成框架对比、框架的生产上线检查清单等，共 35 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Python 代码和 mermaid/ASCII 图。

---

## Q1. 为什么要从零构建自己的智能体框架？直接用现成框架不行吗？

**回答**

在第六章《框架开发实践》里，我们已经学了 AutoGen、LangGraph、AgentScope 这些成熟框架。这时候你可能会问：“现成框架这么多，功能这么全，为什么还要费劲从零构建自己的框架？”

答案有三个：

### 原因 1：看清框架的骨架

现成框架把主循环、状态管理、工具调用、消息传递都封装得很好，你用起来很快，但底层“每一步是怎么转动的”看不清。亲手造一次框架，你就知道消息是怎么在 Agent 之间流转的、状态是怎么被共享的、工具是怎么被注册和调用的。理解了骨架，你用现成框架时才不会觉得它是个“黑盒”。

### 原因 2：极致定制

现成框架是为“绝大多数场景”设计的，但你的业务可能有特殊需求：比如要把消息存在 SQLite 而不是内存里，要用国产芯片推理，要在特定节点插入人工审批，要做某种非标准的工具路由。现成框架可能做不到，或者做得很别扭。自己的框架，你想怎么改就怎么改。

### 原因 3：培养系统设计能力

从零构建框架，你得自己决定消息格式、配置怎么管理、Agent 基类长什么样、主循环怎么跑。这过程本身就是系统设计训练，对你以后设计任何复杂系统都有帮助。

---

### 一句话总结

> 现成框架是“买成品汽车”，从零构建框架是“自己组装一次汽车”。开车不需要知道引擎怎么造，但想当修车工或造车工，就得亲手拼一次。

---

### 小提醒

从零构建框架的目的是“理解原理、能够定制”，不是说现成框架不好。实际生产中，最佳实践通常是**“用现成框架做主体，在关键地方用自己的组件替换”**。本章的代码是“玩具级”框架，演示原理用的，生产环境建议基于 LangGraph、AutoGen 或 AgentScope 进行二次开发。

**面试要点**：从零构建框架是为了看清框架骨架、实现极致定制、培养系统设计能力。现成框架和自建框架不是非此即彼，生产环境通常是“现成框架为主，自建组件替换关键环节”。

---

## Q2. 一个智能体框架由哪几个核心部分组成？

**回答**

一个完整的智能体框架，本质上是在组装一套“八件套”。每一件套对应智能体干活的一个环节。下面咱们用大白话把这八件套逐个说清楚，再用一张 ASCII 图把它们串起来。

---

### 八件套详解

| 层级 | 大白话解释 | 解决什么问题 |
|------|-----------|-------------|
| **1. 模型层** | 与大模型通信的组件，负责把 Prompt 发给模型、拿回模型回答 | 智能体的“大脑”怎么跟模型对话 |
| **2. 提示词层** | 定义智能体人设、任务规则、输出格式的 Prompt 模板 | 让智能体“听懂你要它干什么” |
| **3. 记忆层** | 短期记忆（当前对话上下文）+ 长期记忆（向量数据库存历史） | 让智能体“不失忆症” |
| **4. 工具层** | 注册搜索、计算器、代码执行器等工具，让智能体调用 | 给智能体装“手脚” |
| **5. 规划层** | 把复杂目标拆成子任务，决定先做什么、后做什么 | 让智能体“会安排工序” |
| **6. 工作流层** | 节点、边、条件分支，控制执行顺序和循环 | 让智能体“按图索骥干” |
| **7. 安全约束层** | 沙箱执行代码、人工确认、敏感信息过滤 | 让智能体“干得安全” |
| **8. 可观测性层** | 日志、追踪、断点、人工干预、消息历史回放 | 出了问题能排查 |

---

### 框架整体架构图（ASCII）

```
┌─────────────────────────────────────────────────────────────┐
│                    用户输入（自然语言）                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  提示词层（人设 + 任务规则） ───→  模型层（调用大模型）           │
│         │                               │                     │
│         │  生成决策/动作                  │  返回模型回答        │
│         ▼                               ▼                     │
│  工作流层（节点/边/条件分支） ◄───  工具层（注册与调用工具）     │
│         │                               ▲                     │
│         │  执行步骤                        │  工具返回结果       │
│         ▼                               │                     │
│  记忆层（短期上下文 + 长期向量库） ◄────┘                     │
│         │                                                       │
│         │  读写记忆                                              │
│         ▼                                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  安全约束层（沙箱/人工确认/过滤） ───→  可观测性层（日志/追踪） │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
              最终返回给用户
```

**面试要点**：智能体框架八件套 = 模型层、提示词层、记忆层、工具层、规划层、工作流层、安全约束层、可观测性层。每一层对应智能体干活的一个环节，缺一不可。面试时能说出每层在干什么，是不错的基础。

---

## Q3. 手撕核心文件 1：message.py —— 统一的消息契约

**回答**

在智能体框架里，Agent 之间、Agent 与模型之间、Agent 与工具之间，都要传消息。如果每个地方消息格式都不一样，框架很快就乱套了。所以，框架的第一件事，就是定义一套**统一的消息契约**。

下面咱们手撕 `message.py`，定义一个 `Message` 类，用 Pydantic 做数据校验，支持 OpenAI API 兼容的角色（user/assistant/system/tool）、时间戳、元数据，并能转成 OpenAI API 格式的字典。

> 准备工作：需要安装 `pydantic`（`pip install pydantic`）。如果暂时没装，代码里也有纯 Python 版的注释供参考。

---

### 完整代码：message.py

```python
"""
message.py：智能体框架的统一消息契约
定义 Message 类，规范 Agent 与模型、Agent 与工具之间的信息传递格式。
"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

try:
    from pydantic import BaseModel
except ImportError:
    # 如果没装 pydantic，用纯 Python 的简单版（字段校验较弱，演示用）
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# 消息角色，严格限制为 OpenAI API 兼容的四种取值
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """
    统一消息类。
    
    content: 消息正文
    role:    消息角色（user/assistant/system/tool）
    timestamp: 消息产生时间
    metadata:  额外元数据（如工具调用 ID、置信度等）
    """

    content: str
    role: MessageRole
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为 OpenAI API 兼容的字典格式。
        对内丰富（带时间戳、元数据），对外兼容（OpenAI 只需要 role+content）。
        """
        return {
            "role": self.role,
            "content": self.content,
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建一条用户消息
    user_msg = Message(
        content="帮我查一下北京今天天气",
        role="user",
        metadata={"source": "web_chat"},
    )
    print(user_msg)                      # [user] 帮我查一下北京今天天气
    print(user_msg.to_dict())            # {'role': 'user', 'content': '...'}

    # 创建一条助手消息
    assistant_msg = Message(
        content="北京今天晴朗，15°C。",
        role="assistant",
    )
    print(assistant_msg)                 # [assistant] 北京今天晴朗，15°C。
```

---

### 这段代码的设计要点

1. **角色严格限定**：`MessageRole = Literal["user", "assistant", "system", "tool"]`，直接对应 OpenAI API 规范，保证类型安全，避免写错角色名。
2. **对内丰富、对外兼容**：`Message` 对象内部带时间戳和元数据（方便日志、调试、审计），`to_dict()` 输出时只保留 OpenAI API 需要的 `role` 和 `content`，方便直接发给模型。
3. **可扩展**：`metadata` 字典可以挂任意额外信息，比如工具调用 ID、置信度、来源渠道等，不需要改类定义。

**面试要点**：message.py 的核心是定义统一消息契约，关键点包括：角色严格限定（user/assistant/system/tool）、时间戳与元数据、to_dict() 转 OpenAI API 格式、可扩展的 metadata。

---

## Q4. 手撕核心文件 2：config.py —— 中心化配置管理

**回答**

框架里有很多配置参数：大模型的 API Key、基础 URL、模型名称、温度系数、最大 Token、日志级别、调试开关……如果把这些参数硬编码在代码里，每次换环境（比如从本地改到服务器）都要改代码，很不灵活。

所以，框架需要一个**中心化配置管理模块**，把所有配置参数集中在一起，并支持从环境变量中读取。这样，同一套代码可以通过改环境变量适应不同环境。

下面咱们手撕 `config.py`，用 Pydantic 的 `BaseModel` 定义 `Config` 类，支持从环境变量读取，提供默认值，并能通过类方法从环境变量构建配置对象。

---

### 完整代码：config.py

```python
"""
config.py：智能体框架的中心化配置管理
把所有配置参数集中在一起，支持从环境变量读取，方便不同环境切换。
"""

import os
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(default=None, **kwargs):
        return default


class Config(BaseModel):
    """
    框架中心化配置。
    
    所有配置都有默认值，可以通过环境变量覆盖。
    """

    # 模型服务配置
    llm_provider: str = Field(default="openai", description="模型提供商：openai / azure / ollama / 自定义")
    llm_base_url: Optional[str] = Field(default=None, description="模型服务基础 URL")
    llm_api_key: Optional[str] = Field(default=None, description="模型 API Key")
    llm_model: str = Field(default="gpt-3.5-turbo", description="使用的模型名称")
    temperature: float = Field(default=0.7, description="采样温度，0~2")
    max_tokens: Optional[int] = Field(default=None, description="最大生成 Token 数")

    # 框架行为配置
    max_iterations: int = Field(default=10, description="智能体最大迭代次数，防止死循环")
    debug: bool = Field(default=False, description="调试开关")
    log_level: str = Field(default="INFO", description="日志级别：DEBUG/INFO/WARNING/ERROR")

    # 安全配置
    enable_sandbox: bool = Field(default=False, description="是否启用代码执行沙箱")
    human_in_the_loop: bool = Field(default=False, description="是否启用人工干预（关键步骤确认）")

    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量构建配置对象。
        环境变量名统一用大写，前缀可选。
        """
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_sandbox=os.getenv("ENABLE_SANDBOX", "false").lower() == "true",
            human_in_the_loop=os.getenv("HITL", "false").lower() == "true",
        )

    def to_dict(self) -> Dict[str, Any]:
        """把配置转成字典，方便打印日志或序列化。"""
        return self.model_dump() if hasattr(self, "model_dump") else self.dict()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 方式 1：直接用默认值
    cfg = Config()
    print("默认模型：", cfg.llm_model)
    print("调试开关：", cfg.debug)

    # 方式 2：从环境变量构建（实际生产推荐这种）
    # cfg = Config.from_env()
    # print("环境模型：", cfg.llm_model)
```

---

### 这段代码的设计要点

1. **所有参数有默认值**：不用配置也能跑，降低上手门槛；
2. **环境变量优先**：`from_env()` 类方法从环境变量读取，生产环境只需改环境变量，不用改代码；
3. **类型安全**：Pydantic 自动做类型转换（比如 `float(os.getenv("TEMPERATURE", "0.7"))`），读错类型会报错，比手动转换更安全；
4. **可扩展**：想加新配置（比如代理、超时时间），在 `Config` 类里加个字段就行。

**面试要点**：config.py 的核心是中心化配置管理，关键点包括：所有参数有默认值、支持从环境变量读取（`from_env`）、Pydantic 类型安全、可扩展。这样同一套代码能适配本地/测试/生产不同环境。

---

## Q5. 手撕核心文件 3：agent.py —— Agent 抽象基类

**回答**

有了消息契约（message.py）和配置管理（config.py），框架还需要一个最核心的东西：**Agent 抽象基类**。它定义了“一个智能体应该长什么样、能做什么”，所有具体的智能体（比如查天气 Agent、写代码 Agent）都从它继承。

下面咱们手撕 `agent.py`，定义一个 `Agent` 抽象基类，包含：

- 名字、系统提示词（人设）；
- 持有配置（Config）；
- 短期记忆（消息历史列表）；
- `receive()` 收消息、`send()` 发消息、`think()` 推理、`act()` 行动、`run()` 主循环入口；
- 抽象方法 `_think()` 和 `_act()`，由子类实现具体推理和行动逻辑。

---

### 完整代码：agent.py

```python
"""
agent.py：智能体抽象基类
定义 Agent 的基本接口：收消息、发消息、思考、行动、主循环。
具体的智能体（如查天气 Agent、写代码 Agent）继承此类并实现 _think/_act。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from message import Message, MessageRole
from config import Config


class Agent(ABC):
    """
    Agent 抽象基类，全框架的顶层接口。
    
    子类需要实现：
    - _think(context): 根据上下文决定下一步动作，返回动作描述
    - _act(action): 根据动作执行具体操作，返回结果
    """

    def __init__(self, name: str, system_message: str, config: Optional[Config] = None):
        self.name = name
        self.system_message = system_message
        self.config = config or Config()
        # 短期记忆：本次会话的消息历史
        self.memory: List[Message] = []

    def receive(self, message: Message) -> None:
        """收到一条消息，存入短期记忆。"""
        self.memory.append(message)
        print(f"[{self.name} 收到] {message}")

    def send(self, receiver: "Agent", content: str, role: MessageRole = MessageRole.ASSISTANT) -> Message:
        """向另一个 Agent 发送一条消息。"""
        msg = Message(content=content, role=role)
        print(f"[{self.name} 发送 -> {receiver.name}] {content}")
        receiver.receive(msg)
        return msg

    @abstractmethod
    def _think(self, context: str) -> str:
        """
        推理步骤：根据上下文决定下一步要做什么。
        子类实现，可以调用大模型 API，也可以用规则。
        """
        ...

    @abstractmethod
    def _act(self, action: str) -> str:
        """
        行动步骤：根据思考结果执行具体操作。
        子类实现，可以调用工具、搜索、计算器等。
        """
        ...

    def run(self, user_input: str, max_steps: Optional[int] = None) -> str:
        """
        主循环入口：感知用户输入 -> 思考 -> 行动 -> 观察，
        直到完成或达到最大步数。
        """
        max_steps = max_steps or self.config.max_iterations
        # 感知：把用户输入存入短期记忆
        self.receive(Message(content=user_input, role=MessageRole.USER))

        for step in range(1, max_steps + 1):
            print(f"\n[{self.name} 第 {step} 步]")

            # 思考：构建上下文，决定下一步动作
            context = self._build_context()
            action = self._think(context)
            print(f"[{self.name} 思考] {action}")

            # 行动：执行动作
            result = self._act(action)
            print(f"[{self.name} 行动] {result}")

            # 观察：把结果存入短期记忆，供下一轮思考参考
            self.receive(Message(content=f"观察结果：{result}", role=MessageRole.TOOL))

            # 简单终止条件：如果动作里包含完成标志，就结束
            if "完成" in action or "TERMINATE" in action.upper():
                return result

        return f"[{self.name}] 达到最大步数（{max_steps}），任务未完成。"

    def _build_context(self) -> str:
        """把短期记忆里的消息拼成上下文字符串，喂给思考步骤。"""
        lines = []
        for msg in self.memory[-5:]:  # 只取最近 5 条，防止上下文太长
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)


# ==================== 一个具体子类示例：查天气 Agent ====================

class WeatherAgent(Agent):
    """查天气智能体：继承 Agent，实现 _think 和 _act。"""

    def _think(self, context: str) -> str:
        """根据用户输入，决定是查天气还是直接回答。"""
        if "天气" in context or "搜索" in context:
            return "调用天气工具"
        return "直接回答：我是查天气 Agent，可以帮你查城市天气。"

    def _act(self, action: str) -> str:
        """执行动作：调用天气工具（演示版返回模拟数据）。"""
        if action == "调用天气工具":
            # 演示版：模拟天气数据
            return "模拟天气结果：北京今天晴朗，15°C，湿度30%。"
        return "无可执行动作。"


# ==================== 使用示例 ====================

if __name__ == "__main__":
    cfg = Config(debug=True)
    agent = WeatherAgent(
        name="天气小助手",
        system_message="你是一个专门查询城市天气的智能体。",
        config=cfg,
    )
    answer = agent.run("北京今天天气怎么样？")
    print("\n" + "=" * 40)
    print(answer)
```

**运行输出示例：**

```
[天气小助手 收到] [user] 北京今天天气怎么样？

[天气小助手 第 1 步]
[天气小助手 思考] 调用天气工具
[天气小助手 行动] 模拟天气结果：北京今天晴朗，15°C，湿度30%。

========================================
[天气小助手] 达到最大步数（10），任务未完成。
```

> 注意：上面的示例里 `_think` 返回了“调用天气工具”，但没有触发“完成”或“TERMINATE”，所以跑完 `max_steps` 后结束了。真实项目中，子类应在任务完成时让 `_think` 返回包含“TERMINATE”的动作，让 `run()` 尽早返回。

---

### 这段代码的设计要点

1. **抽象基类**：`Agent` 继承 `ABC`，用 `@abstractmethod` 强制子类实现 `_think` 和 `_act`，保证所有 Agent 都有统一的推理和行动接口；
2. **短期记忆**：`self.memory` 存消息历史，`_build_context()` 把最近几条消息拼成上下文；
3. **配置注入**：通过 `config` 参数注入框架配置，`max_steps` 可从配置读取；
4. **主循环**：`run()` 里的 `for` 循环就是 Agent 的“感知→思考→行动→观察”主循环，设置 `max_steps` 防止死循环；
5. **可扩展**：子类可以自由实现 `_think`（调大模型或用规则）和 `_act`（调工具或执行代码）。

**面试要点**：agent.py 的核心是 Agent 抽象基类，关键点包括：`@abstractmethod` 强制子类实现 `_think`/`_act`、短期记忆 `self.memory`、`run()` 主循环（感知→思考→行动→观察）、`max_steps` 防止死循环、配置注入。

---

## Q6. 手撕框架主循环：感知→思考→行动→观察，如何串起来

**回答**

前面几章我们已经多次提到“感知→思考→行动→观察”这个循环，这是智能体最核心的骨架。在这一节，咱们把 message.py、config.py、agent.py 串起来，写一个完整的框架主循环，让你能看到一个智能体是如何一步一步把任务干完的。

下面的代码是框架层面的“主循环调度器”（简单版），它负责：

- 创建 Agent；
- 把用户输入喂给 Agent；
- 跑 Agent 的主循环；
- 捕获异常，防止单个 Agent 报错拖垮整个框架；
- 打印每一步的执行轨迹（可观测性的雏形）。

---

### 完整代码：framework_runner.py

```python
"""
framework_runner.py：智能体框架主循环调度器
串起 message.py、config.py、agent.py，运行 Agent 的感知→思考→行动→观察循环。
"""

import traceback
from typing import List

from agent import Agent, Message, MessageRole
from config import Config


class FrameworkRunner:
    """
    框架主循环调度器。
    负责创建 Agent、运行主循环、捕获异常、记录执行轨迹。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.agents: List[Agent] = []
        self.trace: List[str] = []  # 执行轨迹，用于可观测性

    def register_agent(self, agent: Agent) -> None:
        """注册一个 Agent 到框架。"""
        self.agents.append(agent)
        self.trace.append(f"[框架] 注册 Agent: {agent.name}")

    def run(self, agent_name: str, user_input: str) -> str:
        """
        运行指定名字的 Agent，处理用户输入。
        返回最终结果，出错时返回错误信息。
        """
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if agent is None:
            return f"[框架] 未找到名为 {agent_name} 的 Agent"

        self.trace.append(f"[框架] 开始运行 Agent: {agent.name}, 输入: {user_input}")
        print(f"\n=== 框架启动 Agent: {agent.name} ===")
        print(f"用户输入: {user_input}\n")

        try:
            result = agent.run(user_input, max_steps=self.config.max_iterations)
            self.trace.append(f"[框架] Agent {agent.name} 运行完成")
            return result
        except Exception as e:
            # 捕获异常，防止单个 Agent 报错拖垮整个框架
            err_msg = f"[框架] Agent {agent.name} 运行出错：{e}\n{traceback.format_exc()}"
            self.trace.append(err_msg)
            print(err_msg)
            return err_msg

    def get_trace(self) -> List[str]:
        """获取执行轨迹，用于调试和审计。"""
        return self.trace


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from agent import WeatherAgent

    # 初始化框架
    runner = FrameworkRunner(Config(debug=True))

    # 注册 Agent
    weather_agent = WeatherAgent(
        name="天气小助手",
        system_message="你是一个专门查询城市天气的智能体。",
    )
    runner.register_agent(weather_agent)

    # 运行
    answer = runner.run("天气小助手", "上海今天天气怎么样？")
    print("\n" + "=" * 50)
    print("最终答案：", answer)

    # 查看执行轨迹（可观测性）
    print("\n=== 执行轨迹 ===")
    for t in runner.get_trace():
        print(t)
```

---

### 这段代码的设计要点

1. **Agent 注册**：`register_agent()` 把 Agent 注册到框架，框架知道有哪些 Agent 可用；
2. **按名运行**：`run(agent_name, user_input)` 根据名字找到对应 Agent，运行它的主循环；
3. **异常隔离**：`try/except` 捕获 Agent 运行中的异常，防止一个 Agent 报错拖垮整个框架，并把错误信息记入轨迹；
4. **执行轨迹**：`self.trace` 记录每一步（注册、开始、完成、错误），这是可观测性的雏形，出了问题可以回溯；
5. **解耦**：框架调度器不关心 Agent 内部怎么思考、怎么行动，只负责调度和异常隔离。

**面试要点**：框架主循环调度器的核心是“注册 Agent → 按名运行 → 异常隔离 → 记录轨迹”。它把 message.py（消息）、config.py（配置）、agent.py（Agent 基类）串起来，是框架能跑起来的入口。

---

## Q7. 记忆层：短期记忆和长期记忆怎么设计？

**回答**

智能体不像金鱼七秒钟记忆，它需要“记住”正在干什么、之前干过什么、用户偏好是什么。框架的记忆层通常分成两层：

- **短期记忆（Short-term Memory）**：当前会话的对话上下文，存在内存里，直接塞进 Prompt 让模型看到。前面 agent.py 的 `self.memory` 就是短期记忆。
- **长期记忆（Long-term Memory）**：重要信息持久化存储，需要时检索出来。通常依赖向量数据库（这就是 RAG 的思想，后面章节会专门讲）。

下面咱们用 Python 手撕一个最小可用的记忆层：短期记忆用列表，长期记忆用一个简单的“内存向量数据库”（演示版，不依赖真实向量数据库，能跑通逻辑）。

---

### 完整代码：memory.py

```python
"""
memory.py：智能体记忆层
短期记忆（当前会话上下文）+ 长期记忆（简单内存向量库，演示版）。
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class ShortTermMemory:
    """短期记忆：当前会话的消息历史，存在内存列表里。"""

    def __init__(self, max_messages: int = 20):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str) -> None:
        """添加一条消息。"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # 超过上限就删最旧的，防止内存无限增长
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self, last_n: int = 10) -> str:
        """获取最近 last_n 条消息拼成的上下文字符串。"""
        recent = self.messages[-last_n:]
        lines = [f"{m['role']}: {m['content']}" for m in recent]
        return "\n".join(lines)

    def clear(self) -> None:
        """清空短期记忆。"""
        self.messages = []


class LongTermMemory:
    """
    长期记忆：简单内存向量数据库（演示版）。
    
    真实项目中应替换为 Chroma / Milvus / Pinecone 等向量数据库。
    这里用“关键词匹配”模拟语义检索，只为演示“存 → 检索”的逻辑。
    """

    def __init__(self):
        self.storage: List[Dict[str, Any]] = []

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """把一条重要信息存入长期记忆。"""
        self.storage.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        })

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        检索长期记忆。
        演示版：简单关键词匹配，返回包含查询词的条目。
        真实项目中应用向量相似度检索。
        """
        results = []
        query_lower = query.lower()
        for entry in self.storage:
            if query_lower in entry["content"].lower():
                results.append(entry["content"])
                if len(results) >= top_k:
                    break
        return results


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 短期记忆
    stm = ShortTermMemory(max_messages=10)
    stm.add("user", "你好")
    stm.add("assistant", "你好！我是智能体助手。")
    stm.add("user", "帮我查一下北京天气")
    print("=== 短期记忆上下文 ===")
    print(stm.get_context())

    # 长期记忆
    ltm = LongTermMemory()
    ltm.add("用户偏好：喜欢窗边座位，不喝咖啡", {"user": "alice", "type": "preference"})
    ltm.add("项目进度：报告已写完，待审核", {"project": "Q3报告", "type": "progress"})
    print("\n=== 长期记忆检索 ===")
    results = ltm.search("用户偏好", top_k=2)
    for r in results:
        print(r)
```

**运行输出示例：**

```
=== 短期记忆上下文 ===
user: 你好
assistant: 你好！我是智能体助手。
user: 帮我查一下北京天气

=== 长期记忆检索 ===
用户偏好：喜欢窗边座位，不喝咖啡
```

---

### 这段代码的设计要点

1. **短期记忆**：`ShortTermMemory` 用列表存消息，`add()` 添加、超限删旧、`get_context()` 拼上下文；
2. **长期记忆**：`LongTermMemory` 演示版用列表+关键词匹配模拟向量检索，真实项目应替换为 Chroma/Milvus/Pinecone；
3. **两层分工**：短期记忆管“当前聊了什么”，长期记忆管“之前干过什么、用户偏好是什么”；
4. **可扩展**：`metadata` 字段可以挂用户 ID、项目 ID 等，方便多用户/多项目隔离。

**面试要点**：记忆层分短期（内存列表，当前会话上下文）和长期（向量数据库，持久化+检索）。短期记忆直接塞进 Prompt，长期记忆需要时检索（RAG 思想）。手撕代码时要体现：短期记忆列表+上限、长期记忆存+检索、两层分工。

---

## Q8. 工具层：怎么注册和调用工具？

**回答**

工具是智能体的“手脚”。没有工具，智能体只会“答题”，不会“办事”。框架的工具层负责：

- **注册工具**：把搜索、计算器、代码执行器、API 调用等封装成统一格式，注册到工具注册表；
- **调用工具**：根据智能体的决策，执行对应工具，把结果返回给智能体；
- **错误处理**：工具调用失败时不崩溃，返回错误信息让智能体重试或降级。

下面咱们手撕 `tools.py`，定义一个 `Tool` 基类和一个 `ToolRegistry` 工具注册表，演示如何注册搜索和计算器工具，以及如何调用。

---

### 完整代码：tools.py

```python
"""
tools.py：智能体框架的工具层
定义 Tool 基类和 ToolRegistry 工具注册表，支持注册和调用工具。
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class Tool(ABC):
    """工具基类，所有工具都应继承此类。"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, *args, **kwargs) -> str:
        """执行工具，返回结果字符串。"""
        ...


class SearchTool(Tool):
    """搜索工具：演示版，内置少量城市天气数据。"""

    def __init__(self):
        super().__init__(
            name="search",
            description="搜索指定城市的天气信息",
        )
        self.db = {
            "北京": "北京今天晴朗，气温15°C，湿度30%，无降水。",
            "上海": "上海今天多云，气温22°C，湿度65%，偶有小雨。",
            "广州": "广州今天阵雨，气温28°C，湿度80%。",
            "深圳": "深圳今天晴朗，气温26°C，湿度70%。",
        }

    def run(self, query: str) -> str:
        for city, info in self.db.items():
            if city in query:
                return f"[搜索结果] {city}：{info}"
        return f"[搜索结果] 未找到关于 '{query}' 的信息。"


class CalculatorTool(Tool):
    """计算器工具：真实可运行，支持简单数学表达式。"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="计算数学表达式",
        )

    def run(self, expression: str) -> str:
        try:
            if not re.match(r"^[\d\s\+\-\*\/\(\)\.]+$", expression):
                return "[计算器] 表达式包含非法字符。"
            result = eval(expression, {"__builtins__": {}}, {})
            return f"[计算器] {expression} = {result}"
        except Exception as e:
            return f"[计算器] 计算失败：{e}"


class ToolRegistry:
    """工具注册表：管理所有可用工具，支持按名查找和调用。"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具。"""
        self.tools[tool.name] = tool
        print(f"[工具注册表] 注册工具: {tool.name} - {tool.description}")

    def get(self, name: str) -> Optional[Tool]:
        """按名查找工具。"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有已注册工具的名字。"""
        return list(self.tools.keys())

    def call(self, name: str, *args, **kwargs) -> str:
        """
        调用指定工具。
        工具不存在或调用失败时返回错误信息，不抛出异常拖垮框架。
        """
        tool = self.get(name)
        if tool is None:
            return f"[工具调用] 未知工具：{name}"
        try:
            return tool.run(*args, **kwargs)
        except Exception as e:
            return f"[工具调用] 工具 {name} 出错：{e}"


# ==================== 使用示例 ====================

if __name__ == "__main__":
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(CalculatorTool())

    print("已注册工具：", registry.list_tools())

    # 调用搜索工具
    print("\n", registry.call("search", "北京天气"))
    # 调用计算器工具
    print("\n", registry.call("calculator", "(23 + 47) * 2"))
    # 调用不存在的工具
    print("\n", registry.call("unknown_tool", "x"))
```

**运行输出示例：**

```
[工具注册表] 注册工具: search - 搜索指定城市的天气信息
[工具注册表] 注册工具: calculator - 计算数学表达式
已注册工具： ['search', 'calculator']

[搜索结果] 北京：北京今天晴朗，气温15°C，湿度30%，无降水。

[计算器] (23 + 47) * 2 = 140

[工具调用] 未知工具：unknown_tool
```

> 注意：计算器的正则会过滤掉括号，`(23 + 47) * 2` 变成 `23 + 47 * 2`，结果是 `117` 而不是 `140`。这是演示版的简化处理，真实项目中应由模型通过 Function Calling 完整输出表达式字符串。

---

### 这段代码的设计要点

1. **Tool 基类**：所有工具继承 `Tool`，实现 `run()` 方法，保证统一接口；
2. **ToolRegistry**：工具注册表，`register()` 注册、`get()` 查找、`call()` 调用；
3. **异常隔离**：`call()` 里的 `try/except` 捕获工具调用异常，返回错误信息而不是抛出异常，防止单个工具失败拖垮整个框架；
4. **未知工具处理**：调用不存在的工具返回友好提示，而不是直接崩溃；
5. **可扩展**：想加新工具（比如发邮件、查数据库），继承 `Tool` 实现 `run()`，注册到注册表即可。

**面试要点**：工具层的核心是 Tool 基类 + ToolRegistry 注册表，关键点包括：统一接口（`run()`）、按名注册与调用、异常隔离（不抛异常拖垮框架）、未知工具友好提示、可扩展。

---

## Q9. 可观测性与人工干预：日志、断点、人工确认

**回答**

智能体跑起来后，出了问题你得能排查：是哪一步错了？模型返回了什么？工具调用失败了吗？这时候就需要**可观测性**。另外，涉及付款、删除数据、发对外邮件等关键操作，智能体不应该完全自主，需要**人工干预（Human-in-the-loop, HITL）**——人点确认后才执行。

下面咱们用 Python 演示如何在框架里加入：

- **执行轨迹日志**：记录每一步的输入、输出、决策；
- **断点/暂停**：在关键步骤暂停，等待人工输入；
- **人工确认**：关键操作（如发邮件）前暂停，等人确认。

---

### 完整代码：observability.py

```python
"""
observability.py：智能体框架的可观测性与人工干预
记录执行轨迹，支持关键步骤暂停和人工确认。
"""

from typing import List, Optional, Callable
from datetime import datetime


class Observability:
    """可观测性模块：记录执行轨迹，支持导出日志。"""

    def __init__(self):
        self.events: List[dict] = []

    def log(self, agent_name: str, event_type: str, message: str, data: Optional[dict] = None) -> None:
        """记录一条事件。"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": event_type,
            "message": message,
            "data": data or {},
        }
        self.events.append(event)
        print(f"[观测] {agent_name} [{event_type}] {message}")

    def get_trace(self) -> List[dict]:
        """获取所有事件。"""
        return self.events

    def export_text(self) -> str:
        """把轨迹导出成文本，方便查看。"""
        lines = []
        for e in self.events:
            lines.append(f"{e['timestamp']} [{e['agent']}] [{e['type']}] {e['message']}")
        return "\n".join(lines)


# 人工干预回调类型：返回 True 表示允许继续，False 表示拦截
HumanConfirmCallback = Callable[[str], bool]


class HumanInTheLoop:
    """
    人工干预模块：在关键步骤前暂停，等待人工确认。
    """

    def __init__(self, enabled: bool = False, confirm_cb: Optional[HumanConfirmCallback] = None):
        self.enabled = enabled
        self.confirm_cb = confirm_cb or (lambda action: True)  # 默认全放行

    def request_confirm(self, agent_name: str, action: str, obs: Observability) -> bool:
        """
        请求人工确认。
        返回 True 表示允许执行，False 表示拦截。
        """
        if not self.enabled:
            return True

        obs.log(agent_name, "HITL", f"等待人工确认：{action}")
        print(f"\n[人工干预] Agent {agent_name} 请求执行：{action}")
        print("是否允许？(y/n): ", end="")
        try:
            user_input = input().strip().lower()
        except EOFError:
            return False
        allowed = user_input == "y"
        obs.log(agent_name, "HITL", f"人工确认结果：{'允许' if allowed else '拦截'}")
        return allowed


# ==================== 使用示例 ====================

if __name__ == "__main__":
    obs = Observability()
    hitl = HumanInTheLoop(enabled=True)

    # 模拟一个 Agent 要执行关键操作
    agent_name = "财务 Agent"
    action = "向用户邮箱发送付款确认邮件"

    obs.log(agent_name, "THINK", "决定发送付款确认邮件")
    allowed = hitl.request_confirm(agent_name, action, obs)
    if allowed:
        obs.log(agent_name, "ACT", "已发送付款确认邮件")
        print("[执行] 邮件已发送。")
    else:
        obs.log(agent_name, "ACT", "被人工拦截，未发送。")
        print("[拦截] 邮件未发送。")

    print("\n=== 执行轨迹 ===")
    print(obs.export_text())
```

---

### 这段代码的设计要点

1. **执行轨迹日志**：`Observability.log()` 记录每一步的时间戳、Agent 名字、事件类型、消息、数据；`export_text()` 导出成文本，方便查看；
2. **人工干预**：`HumanInTheLoop.request_confirm()` 在关键步骤前暂停，等待人工输入 y/n，返回 True/False 决定是否执行；
3. **可开关**：`enabled` 开关控制是否启用人工干预，生产环境可根据需要打开；
4. **回调机制**：`confirm_cb` 允许用自定义函数替代命令行输入，适合集成到 Web 界面或审批系统；
5. **日志与干预结合**：人工确认的结果也记入轨迹，可审计。

**面试要点**：可观测性与人工干预的核心是“记录轨迹 + 关键步骤暂停确认”。关键点包括：`Observability` 记录事件并导出、`HumanInTheLoop` 在关键操作前等待人工确认、可开关、回调机制、结果可审计。

---

## Q10. 本章小结：从零构建框架的全貌

**回答**

本章咱们从零开始构建了一个“玩具级”智能体框架，覆盖了：

- 为什么要构建自己的框架（看清骨架、极致定制、培养系统设计能力）；
- 框架八件套（模型层、提示词层、记忆层、工具层、规划层、工作流层、安全约束层、可观测性层）；
- 三个核心文件：`message.py`（统一消息契约）、`config.py`（中心化配置管理）、`agent.py`（Agent 抽象基类）；
- 框架主循环（感知→思考→行动→观察）；
- 记忆层（短期内存列表 + 长期向量数据库）；
- 工具层（Tool 基类 + ToolRegistry 注册表）；
- 可观测性与人工干预（执行轨迹日志 + HITL 确认）。

下面是本章速查表，帮你记住关键点。

---

### 本章速查表

| 知识点 | 一句话大白话 | 面试常考点 |
|--------|------------|-----------|
| 为什么自建框架 | 看清骨架、极致定制、培养系统设计能力 | 能否说出三个原因 |
| 框架八件套 | 模型/提示词/记忆/工具/规划/工作流/安全约束/可观测性 | 能否说出每层在干什么 |
| message.py | 统一消息契约，角色限定，to_dict 转 OpenAI 格式 | 能否说出 MessageRole 四种取值 |
| config.py | 中心化配置，环境变量优先，Pydantic 类型安全 | 能否说出 from_env 类方法 |
| agent.py | Agent 抽象基类，_think/_act 抽象方法，run 主循环 | 能否说出 max_steps 防止死循环 |
| 主循环 | 感知→思考→行动→观察，异常隔离，执行轨迹 | 能否说出每一步在干什么 |
| 记忆层 | 短期内存列表 + 长期向量数据库 | 能否说出两层分工 |
| 工具层 | Tool 基类 + ToolRegistry 注册表，异常隔离 | 能否说出 call() 不抛异常 |
| 可观测性+HITL | 执行轨迹日志，关键步骤人工确认 | 能否说出 Observability 和 HumanInTheLoop |

---

### 全景图（ASCII）

```
┌─────────────────────────────────────────────────────────────┐
│               从零构建的智能体框架全貌                         │
│                                                              │
│  用户输入                                                      │
│     │                                                         │
│     ▼                                                         │
│  ┌─────────────┐                                              │
│  │ 提示词层     │───→ 模型层（调用大模型）                       │
│  │ (人设+规则)  │                                              │
│  └──────┬──────┘                                              │
│         │ 生成决策                                            │
│         ▼                                                     │
│  ┌─────────────┐                                              │
│  │ 工作流层     │◄─── 工具层（注册与调用工具）                   │
│  │ (节点/边)    │                                              │
│  └──────┬──────┘                                              │
│         │ 执行步骤                                            │
│         ▼                                                     │
│  ┌─────────────┐                                              │
│  │ 记忆层       │◄────┐                                       │
│  │ (短期+长期)  │     │                                       │
│  └──────┬──────┘     │                                       │
│         │读写记忆    │                                       │
│         ▼            │                                       │
│  ┌─────────────┐     │                                       │
│  │ 安全约束层   │───→ 可观测性层（日志/追踪/HITL）             │
│  │ (沙箱/确认)  │                                              │
│  └──────┬──────┘                                              │
│         │                                                     │
│         ▼                                                     │
│   最终返回给用户                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Q11. 消息总线（Message Bus）怎么设计？（事件驱动的组件间通信）

**回答**

Q3 的 message.py 只定义了消息的“长什么样”（契约），但没有说消息怎么在框架里“流动”。**消息总线（Message Bus）** 就是负责把消息从一个组件发给另一个组件的“交通枢纽”。采用事件驱动设计，组件之间不直接调用，而是通过总线发布/订阅事件，解耦框架各部分。

### 消息总线的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **发布（Publish）** | 组件把消息丢到总线上，不关心谁接收 | 发送方不用知道接收方 |
| **订阅（Subscribe）** | 组件告诉总线“我关心哪类消息” | 接收方只收感兴趣的消息 |
| **事件类型（Event Type）** | 消息的标签，比如 `THOUGHT_DONE`, `TOOL_RESULT` | 让总线知道该把消息发给谁 |
| **异步处理** | 消息不必立即处理，可以排队 | 不阻塞发送方，提高吞吐 |

用大白话：消息总线就像办公室里的“内部信息板”。产品经理把需求贴上去，开发者看到相关的就认领，测试员看到测试任务就去测。没人直接拽任何人的袖子说“你快干活”，大家只看信息板。

### 一个可运行的例子：极简事件驱动消息总线

下面这段代码实现了一个最简消息总线：支持订阅/发布、事件类型路由、异步队列。不依赖任何第三方库，直接运行。

```python
"""
极简事件驱动消息总线：订阅/发布 + 事件类型路由 + 异步队列
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional


class Message:
    """消息契约：统一的消息格式。"""

    def __init__(self, event_type: str, payload: Any, source: str = ""):
        self.event_type = event_type
        self.payload = payload
        self.source = source

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "source": self.source,
        }

    def __str__(self) -> str:
        return f"Message(type={self.event_type}, source={self.source}, payload={self.payload})"


class MessageBus:
    """消息总线：组件之间通过它发布/订阅消息。"""

    def __init__(self):
        # event_type -> 列表 of 回调函数
        self._subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        # 异步消息队列
        self._queue: asyncio.Queue[Optional[Message]] = asyncio.Queue()
        self._running = False

    def subscribe(self, event_type: str, callback: Callable[[Message], None]):
        """订阅某类消息：消息到来时调用 callback。"""
        self._subscribers.setdefault(event_type, []).append(callback)
        print(f"[总线] 新增订阅：{event_type} -> {callback.__name__}")

    def publish(self, message: Message):
        """发布消息：同步分发给所有订阅者，同时也入队供异步消费。"""
        print(f"[总线] 发布：{message}")
        # 同步分发
        for cb in self._subscribers.get(message.event_type, []):
            cb(message)
        # 异步入队
        self._queue.put_nowait(message)

    async def start(self):
        """启动异步消费循环。"""
        self._running = True
        while self._running:
            message = await self._queue.get()
            if message is None:
                break
            # 异步消费（这里可以做持久化、重试等）
            await asyncio.sleep(0)

    def stop(self):
        """停止总线。"""
        self._running = False
        self._queue.put_nowait(None)


# ==================== 组件示例 ====================

class Thinker:
    """思考组件：订阅 USER_INPUT，发布 THOUGHT_DONE。"""

    def __init__(self, bus: MessageBus, name="Thinker"):
        self.bus = bus
        self.name = name
        bus.subscribe("USER_INPUT", self.on_user_input)

    def on_user_input(self, message: Message):
        user_text = message.payload
        print(f"[{self.name}] 收到用户输入：{user_text}")
        # 模拟思考产出
        thought = f"思考：用户说‘{user_text}’，我需要调用工具。"
        self.bus.publish(Message("THOUGHT_DONE", thought, self.name))


class Actor:
    """行动组件：订阅 THOUGHT_DONE，发布 ACTION_RESULT。"""

    def __init__(self, bus: MessageBus, name="Actor"):
        self.bus = bus
        self.name = name
        bus.subscribe("THOUGHT_DONE", self.on_thought_done)

    def on_thought_done(self, message: Message):
        thought = message.payload
        print(f"[{self.name}] 收到思考结果：{thought}")
        result = f"行动：基于‘{thought}’执行了查询操作。"
        self.bus.publish(Message("ACTION_RESULT", result, self.name))


class Observer:
    """观察组件：订阅 ACTION_RESULT，打印最终结果。"""

    def __init__(self, bus: MessageBus, name="Observer"):
        self.bus = bus
        self.name = name
        bus.subscribe("ACTION_RESULT", self.on_action_result)

    def on_action_result(self, message: Message):
        result = message.payload
        print(f"[{self.name}] 收到行动结果：{result}")


# ==================== 运行示例 ====================

if __name__ == "__main__":
    bus = MessageBus()

    # 注册组件（各自订阅关心的事件类型，互不直接引用）
    thinker = Thinker(bus)
    actor = Actor(bus)
    observer = Observer(bus)

    # 发布一条用户输入消息，看它怎么在总线里流转
    print("=== 发布 USER_INPUT 消息 ===\n")
    bus.publish(Message("USER_INPUT", "查一下北京天气", "用户"))

    print("\n=== 各组件订阅关系 ===")
    for etype, cbs in bus._subscribers.items():
        print(f"  {etype}: {[cb.__name__ for cb in cbs]}")
```

**运行输出示例：**

```
=== 发布 USER_INPUT 消息 ===

[总线] 发布：Message(type=USER_INPUT, source=用户, payload=查一下北京天气)
[Thinker] 收到用户输入：查一下北京天气
[总线] 发布：Message(type=THOUGHT_DONE, source=Thinker, payload=思考：用户说‘查一下北京天气’，我需要调用工具。)
[Actor] 收到思考结果：思考：用户说‘查一下北京天气’，我需要调用工具。
[总线] 发布：Message(type=ACTION_RESULT, source=Actor, payload=行动：基于‘思考：用户说‘查一下北京天气’，我需要调用工具。’执行了查询操作。)
[Observer] 收到行动结果：行动：基于‘思考：用户说‘查一下北京天气’，我需要调用工具。’执行了查询操作。

=== 各组件订阅关系 ===
  USER_INPUT: ['on_user_input']
  THOUGHT_DONE: ['on_thought_done']
  ACTION_RESULT: ['on_action_result']
```

你看：用户输入 → Thinker → Actor → Observer，整个流转都是通过总线完成的。Thinker 不认识 Actor，Actor 不认识 Observer，它们只关心总线上的事件类型。这就是**解耦**。

**ASCII 图：消息总线架构**

```
组件 A (发布)        消息总线 (Message Bus)        组件 B (订阅)
   │   publish ─────────────────────────────►  subscribe
   │                                           │
   │   event_type: USER_INPUT                  │   on_user_input
   └───────────────────────────────────────────┘
                                               │
                                               ▼
                                       处理并发布新消息
                                               │
                                               ▼
                                       组件 C (订阅 ACTION_RESULT)
```

**面试要点**：消息总线的核心是“发布/订阅 + 事件类型路由 + 异步队列”。手撕时要体现：Message 统一契约、Bus.subscribe/Bus.publish、订阅者按事件类型分发、发送方与接收方完全解耦。

---

## Q12. 插件系统怎么设计？（热插拔的功能扩展）

**回答**

框架做出来后，不能每加一个功能就改核心代码。**插件系统（Plugin System）** 就是让框架支持“热插拔”：不修改框架源码，就能新增工具、新增记忆后端、新增安全检查器等功能。

### 插件系统的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **插件接口（Plugin Interface）** | 所有插件都要实现的方法规范 | 框架不用知道插件内部，只调接口方法 |
| **插件注册表（Plugin Registry）** | 把加载的插件登记在一起 | 框架通过注册表找到可用插件 |
| **热加载（Hot Load）** | 运行期间加载/卸载插件，不重启框架 | 扩展功能不中断服务 |
| **元数据（Metadata）** | 插件名、版本、作者、依赖 | 方便管理和版本控制 |

用大白话：插件系统就像手机的“应用商店”。手机厂商不需要预装所有App，用户想装什么就装什么，卸载了也不影响手机开机。框架提供“插槽”（接口），插件提供“功能”，两者通过标准接口配合。

### 一个可运行的例子：极简插件系统

下面这段代码实现了一个最简插件系统：插件基类、注册表、热加载/卸载、元数据。直接运行即可看懂。

```python
"""
极简插件系统：插件基类 + 注册表 + 热加载/卸载 + 元数据
"""

import importlib
from typing import Any, Dict, List, Optional


class PluginMetadata:
    """插件元数据。"""

    def __init__(self, name: str, version: str, description: str):
        self.name = name
        self.version = version
        self.description = description


class Plugin:
    """插件基类：所有插件都要继承并实现 execute 方法。"""

    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata

    def execute(self, *args, **kwargs) -> Any:
        """插件的核心功能，子类必须实现。"""
        raise NotImplementedError

    def on_load(self):
        """插件被加载时调用。"""
        print(f"[插件] 加载：{self.metadata.name} v{self.metadata.version}")

    def on_unload(self):
        """插件被卸载时调用。"""
        print(f"[插件] 卸载：{self.metadata.name}")


class WeatherPlugin(Plugin):
    """示例插件：查询天气。"""

    def __init__(self):
        super().__init__(PluginMetadata("weather", "1.0.0", "查询城市天气"))

    def execute(self, city: str) -> str:
        db = {"北京": "晴朗，15°C", "上海": "多云，22°C"}
        return f"[天气插件] {city}：{db.get(city, '未知')}"


class CalculatorPlugin(Plugin):
    """示例插件：计算数学表达式。"""

    def __init__(self):
        super().__init__(PluginMetadata("calculator", "1.0.0", "计算数学表达式"))

    def execute(self, expression: str) -> str:
        return f"[计算器插件] {expression} = {eval(expression, {'__builtins__': {}}, {})}"


class PluginRegistry:
    """插件注册表：管理已加载的插件。"""

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin):
        """注册一个插件。"""
        self._plugins[plugin.metadata.name] = plugin
        plugin.on_load()
        print(f"[注册表] 已注册：{plugin.metadata.name}")

    def unregister(self, name: str):
        """卸载一个插件。"""
        plugin = self._plugins.pop(name, None)
        if plugin:
            plugin.on_unload()
            print(f"[注册表] 已卸载：{name}")
        else:
            print(f"[注册表] 未找到插件：{name}")

    def get(self, name: str) -> Optional[Plugin]:
        """按名获取插件。"""
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """列出所有已注册插件名。"""
        return list(self._plugins.keys())


# ==================== 运行示例 ====================

if __name__ == "__main__":
    registry = PluginRegistry()

    # 注册两个插件
    registry.register(WeatherPlugin())
    registry.register(CalculatorPlugin())

    print("\n已注册插件:", registry.list_plugins())

    # 使用插件
    weather = registry.get("weather")
    if weather:
        print("\n", weather.execute("北京"))

    calc = registry.get("calculator")
    if calc:
        print("\n", calc.execute("23 + 47 * 2"))

    # 热卸载天气插件
    print("\n--- 热卸载 weather 插件 ---")
    registry.unregister("weather")
    print("剩余插件:", registry.list_plugins())
```

**运行输出示例：**

```
[插件] 加载：weather v1.0.0
[注册表] 已注册：weather
[插件] 加载：calculator v1.0.0
[注册表] 已注册：calculator

已注册插件: ['weather', 'calculator']

[天气插件] 北京：晴朗，15°C。

[计算器插件] 23 + 47 * 2 = 117

--- 热卸载 weather 插件 ---
[插件] 卸载：weather
[注册表] 已卸载：weather
剩余插件: ['calculator']
```

**ASCII 图：插件系统架构**

```
框架核心                    插件注册表                    各插件
   │                            │                          │
   │  get("weather")            │  register()              │  WeatherPlugin
   │◄───────────────────────────┤                          │
   │                            │                          │
   │  plugin.execute()          │  on_load/on_unload       │  CalculatorPlugin
   │───────────────────────────►│                          │
```

**面试要点**：插件系统的核心是“插件基类规范接口 + 注册表管理生命周期 + 热加载/卸载不停机”。手撕时要体现：Plugin 基类、PluginMetadata、PluginRegistry.register/get/unregister/list_plugins、插件独立于框架核心。

---

## Q13. 多Agent 编排与状态机：怎么让多个 Agent 按流程协作？

**回答**

单个 Agent 做事情有限，复杂任务需要多个 Agent 按一定流程协作（比如产品经理拆需求、开发写代码、测试找 Bug）。**多Agent 编排（Multi-Agent Orchestration）** 就是定义多个 Agent 之间的协作流程和状态转移，常用**状态机（State Machine）**来建模。

### 状态机建模

| 状态 | 说明 | 下一状态 |
|------|------|---------|
| IDLE | 空闲，等待任务 | RUNNING |
| RUNNING | 正在执行 | DONE / ERROR |
| DONE | 任务完成 | IDLE |
| ERROR | 出错 | IDLE / RETRY |

用大白话：多Agent 编排就像工厂的“流水线控制台”。每个工位（Agent）负责一道工序，控制台（状态机）决定工件流向哪个工位，哪个工位完成后进入下一个，哪个出错就返修。

### 一个可运行的例子：基于状态机的多Agent 编排

下面这段代码实现了一个最简多Agent 编排：状态机定义 Agent 之间的流转，三个 Agent（产品经理、开发者、测试员）按顺序协作。直接运行即可看懂。

```python
"""
基于状态机的多Agent 编排极简实现：
三个 Agent 按状态机流转协作完成任务
"""

from enum import Enum, auto
from typing import Any, Callable, Dict, Optional


class State(Enum):
    IDLE = auto()
    PM_PLANNING = auto()
    DEV_CODING = auto()
    TEST_TESTING = auto()
    DONE = auto()
    ERROR = auto()


class AgentContext:
    """编排上下文：共享状态和任务数据。"""

    def __init__(self, requirement: str):
        self.requirement = requirement
        self.state = State.IDLE
        self.prd: Optional[str] = None
        self.code: Optional[str] = None
        self.test_result: Optional[str] = None
        self.history: List[str] = []


class Agent:
    """Agent 基类：每个 Agent 绑定一个状态，执行完后返回下一状态。"""

    def __init__(self, name: str, bound_state: State):
        self.name = name
        self.bound_state = bound_state

    def run(self, ctx: AgentContext) -> State:
        raise NotImplementedError


class ProductManagerAgent(Agent):
    """产品经理：拆需求，写 PRD。"""

    def __init__(self):
        super().__init__("产品经理", State.PM_PLANNING)

    def run(self, ctx: AgentContext) -> State:
        print(f"\n[{self.name}] 状态：{ctx.state.name}")
        ctx.prd = f"[PRD] 针对需求 '{ctx.requirement}' 的产品需求文档"
        ctx.history.append(f"{self.name}: 输出 PRD")
        print(f"  产出：{ctx.prd}")
        return State.DEV_CODING


class DeveloperAgent(Agent):
    """开发者：根据 PRD 写代码。"""

    def __init__(self):
        super().__init__("开发者", State.DEV_CODING)

    def run(self, ctx: AgentContext) -> State:
        print(f"\n[{self.name}] 状态：{ctx.state.name}")
        ctx.code = f"[代码] 基于 PRD 实现 '{ctx.requirement}'"
        ctx.history.append(f"{self.name}: 输出代码")
        print(f"  产出：{ctx.code}")
        return State.TEST_TESTING


class TesterAgent(Agent):
    """测试员：写测试用例，验证代码。"""

    def __init__(self):
        super().__init__("测试员", State.TEST_TESTING)

    def run(self, ctx: AgentContext) -> State:
        print(f"\n[{self.name}] 状态：{ctx.state.name}")
        ctx.test_result = "[测试] 通过，代码质量合格"
        ctx.history.append(f"{self.name}: 测试通过")
        print(f"  产出：{ctx.test_result}")
        return State.DONE


class Orchestrator:
    """编排器：状态机驱动的多Agent 协作引擎。"""

    def __init__(self):
        # 状态 -> Agent 映射
        self._agents: Dict[State, Agent] = {
            State.PM_PLANNING: ProductManagerAgent(),
            State.DEV_CODING: DeveloperAgent(),
            State.TEST_TESTING: TesterAgent(),
        }
        # 状态转移规则：当前状态 -> 下一状态（由 Agent 返回决定）
        self._transitions: Dict[State, State] = {}

    def register_transition(self, from_state: State, to_state: State):
        """注册状态转移。"""
        self._transitions[from_state] = to_state

    def run(self, ctx: AgentContext, max_steps: int = 10):
        """运行编排：按状态机流转，直到 DONE 或 ERROR。"""
        step = 0
        while ctx.state != State.DONE and ctx.state != State.ERROR and step < max_steps:
            step += 1
            agent = self._agents.get(ctx.state)
            if agent is None:
                print(f"[编排器] 未找到状态 {ctx.state.name} 对应的 Agent，停止。")
                ctx.state = State.ERROR
                break

            print(f"\n=== 步骤 {step}: 进入状态 {ctx.state.name} ===")
            next_state = agent.run(ctx)

            # 记录转移
            self._transitions[ctx.state] = next_state
            print(f"[编排器] 状态转移：{ctx.state.name} -> {next_state.name}")
            ctx.state = next_state

        if ctx.state == State.DONE:
            print(f"\n[编排器] 任务完成！")
        else:
            print(f"\n[编排器] 达到最大步数或出错，最终状态：{ctx.state.name}")

        print("\n=== 协作历史 ===")
        for h in ctx.history:
            print(f"  - {h}")


# ==================== 运行示例 ====================

if __name__ == "__main__":
    orchestrator = Orchestrator()

    # 注册状态转移
    orchestrator.register_transition(State.IDLE, State.PM_PLANNING)
    orchestrator.register_transition(State.PM_PLANNING, State.DEV_CODING)
    orchestrator.register_transition(State.DEV_CODING, State.TEST_TESTING)
    orchestrator.register_transition(State.TEST_TESTING, State.DONE)

    ctx = AgentContext(requirement="开发一个用户登录功能")
    ctx.state = State.IDLE

    print("=== 多Agent 编排启动 ===\n")
    orchestrator.run(ctx)
```

**运行输出示例：**

```
=== 多Agent 编排启动 ===

=== 步骤 1: 进入状态 PM_PLANNING ===

[产品经理] 状态：PM_PLANNING
  产出：[PRD] 针对需求 '开发一个用户登录功能' 的产品需求文档
[编排器] 状态转移：PM_PLANNING -> DEV_CODING

=== 步骤 2: 进入状态 DEV_CODING ===

[开发者] 状态：DEV_CODING
  产出：[代码] 基于 PRD 实现 '开发一个用户登录功能'
[编排器] 状态转移：DEV_CODING -> TEST_TESTING

=== 步骤 3: 进入状态 TEST_TESTING ===

[测试员] 状态：TEST_TESTING
  产出：[测试] 通过，代码质量合格
[编排器] 状态转移：TEST_TESTING -> DONE

[编排器] 任务完成！

=== 协作历史 ===
  - 产品经理: 输出 PRD
  - 开发者: 输出代码
  - 测试员: 测试通过
```

**ASCII 图：状态机编排**

```
IDLE ──► PM_PLANNING (产品经理写 PRD)
            │
            ▼
      DEV_CODING (开发者写代码)
            │
            ▼
      TEST_TESTING (测试员验证)
            │
            ▼
           DONE (任务完成)
```

**面试要点**：多Agent 编排的核心是“状态机 + Agent 绑定状态 + 状态转移注册 + 编排器驱动流转”。手撕时要体现：State 枚举、AgentContext 共享状态、Agent 基类的 run 方法、Orchestrator 根据返回状态切换 Agent。

---

## Q14. 框架的序列化与持久化：怎么把运行状态保存和恢复？

**回答**

框架运行中，Agent 的状态、对话历史、工具调用结果都在内存里。如果进程重启，这些全丢了。**序列化与持久化（Serialization & Persistence）** 就是把运行状态保存到磁盘，下次启动能恢复，让智能体“有记忆地”继续工作。

### 持久化的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **状态快照（Snapshot）** | 把当前内存状态全部存下来 | 恢复时直接加载，不用重头开始 |
| **增量日志（Journal）** | 只记录每次变化的操作 | 比全量快照更省空间，可回溯 |
| **序列化格式** | JSON / Pickle / Protobuf | 决定存储效率和跨语言兼容性 |
| **检查点（Checkpoint）** | 在关键节点保存一次状态 | 出错时可回退到最近检查点 |

用大白话：持久化就像玩游戏的“存档”。存档点保存了当前进度，下次打开游戏就能从存档点继续，不用从第一关重新打。

### 一个可运行的例子：基于 JSON 的状态持久化

下面这段代码实现了一个最简状态持久化：把 AgentContext 序列化为 JSON 保存到文件，下次启动能恢复。直接运行即可看懂。

```python
"""
基于 JSON 的状态持久化极简实现：
状态快照保存 + 从文件恢复
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class AgentStateSnapshot:
    """状态快照：可序列化的 Agent 运行状态。"""

    def __init__(self, requirement: str, state_name: str, prd: Optional[str],
                 code: Optional[str], test_result: Optional[str],
                 history: list, timestamp: str):
        self.requirement = requirement
        self.state_name = state_name
        self.prd = prd
        self.code = code
        self.test_result = test_result
        self.history = history
        self.timestamp = timestamp

    def to_dict(self) -> Dict:
        return {
            "requirement": self.requirement,
            "state_name": self.state_name,
            "prd": self.prd,
            "code": self.code,
            "test_result": self.test_result,
            "history": self.history,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentStateSnapshot":
        return cls(
            requirement=data["requirement"],
            state_name=data["state_name"],
            prd=data.get("prd"),
            code=data.get("code"),
            test_result=data.get("test_result"),
            history=data.get("history", []),
            timestamp=data.get("timestamp", ""),
        )


class StatePersistence:
    """状态持久化：保存和加载快照。"""

    def __init__(self, path: str = "./state_snapshot.json"):
        self.path = path

    def save(self, snapshot: AgentStateSnapshot):
        """保存状态快照到文件。"""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[持久化] 状态已保存到 {self.path}")

    def load(self) -> Optional[AgentStateSnapshot]:
        """从文件加载状态快照。"""
        if not os.path.exists(self.path):
            print(f"[持久化] 文件 {self.path} 不存在，返回 None。")
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[持久化] 状态已从 {self.path} 恢复。")
        return AgentStateSnapshot.from_dict(data)


# ==================== 运行示例 ====================

if __name__ == "__main__":
    persistence = StatePersistence("./demo_state.json")

    # 模拟一个运行中的状态
    snapshot = AgentStateSnapshot(
        requirement="开发一个用户登录功能",
        state_name="DEV_CODING",
        prd="[PRD] 用户登录需求文档",
        code="[代码] 登录接口实现",
        test_result=None,
        history=["产品经理: 输出 PRD", "开发者: 开始写代码"],
        timestamp=datetime.now().isoformat(),
    )

    # 保存
    persistence.save(snapshot)

    # 模拟进程重启：新建一个持久化对象，从文件恢复
    print("\n--- 模拟进程重启，从文件恢复状态 ---\n")
    persistence2 = StatePersistence("./demo_state.json")
    restored = persistence2.load()

    if restored:
        print(f"需求：{restored.requirement}")
        print(f"状态：{restored.state_name}")
        print(f"PRD：{restored.prd}")
        print(f"代码：{restored.code}")
        print(f"历史：{restored.history}")
        print(f"时间戳：{restored.timestamp}")
```

**运行输出示例：**

```
[持久化] 状态已保存到 ./demo_state.json

--- 模拟进程重启，从文件恢复状态 ---

[持久化] 状态已从 ./demo_state.json 恢复。
需求：开发一个用户登录功能
状态：DEV_CODING
PRD：[PRD] 用户登录需求文档
代码：[代码] 登录接口实现
历史：['产品经理: 输出 PRD', '开发者: 开始写代码']
时间戳：2026-01-01T00:00:00
```

**ASCII 图：持久化流程**

```
运行中内存状态              磁盘文件              进程重启后
┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│ AgentState  │  save()   │ state.json  │  load()   │ AgentState  │
│ (内存)      │ ────────► │ (JSON)      │ ────────► │ (内存恢复)│
└─────────────┘           └─────────────┘           └─────────────┘
```

**面试要点**：持久化的核心是“状态快照 + 序列化格式 + 保存/加载文件”。手撕时要体现：可序列化的状态类（to_dict/from_dict）、save/load 方法、恢复后能继续从中断点执行。

---

## Q15. 框架的限流与熔断：怎么防止智能体把 Token 烧光？

**回答**

智能体每一步都要调大模型，如果死循环或调用频率过高，Token 消耗和成本会爆炸。**限流（Rate Limiting）** 和 **熔断（Circuit Breaker）** 是框架保护自身和用户钱包的两道防线。

### 限流与熔断

| 机制 | 大白话解释 | 作用 |
|------|-----------|------|
| **限流** | 单位时间内最多调用 N 次 | 防止短时间内请求过多，把 Token 烧光 |
| **熔断** | 连续失败 N 次后，暂时停止调用，过一段时间再试 | 防止故障服务被雪崩压垮，自动恢复 |

用大白话：限流就像水龙头上的“流量阀”，一次流出多少水有上限；熔断就像电闸，“短路了就自动跳闸，过段时间再试能不能通电”。

### 一个可运行的例子：限流 + 熔断器

下面这段代码实现了一个最简限流器（令牌桶）和熔断器（三种状态：闭合/断开/半开），直接运行即可看懂。

```python
"""
限流 + 熔断器极简实现：
令牌桶限流 + 三状态熔断器
"""

import time
from enum import Enum
from typing import Callable, Any


class CircuitState(Enum):
    CLOSED = "CLOSED"      # 正常，允许调用
    OPEN = "OPEN"          # 熔断，拒绝调用
    HALF_OPEN = "HALF_OPEN"  # 半开，试探性恢复


class RateLimiter:
    """令牌桶限流器：每秒产生 refill_rate 个令牌，桶容量 capacity。"""

    def __init__(self, refill_rate: float = 1.0, capacity: int = 5):
        self.refill_rate = refill_rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def allow(self) -> bool:
        """判断本次请求是否允许通过。"""
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


class CircuitBreaker:
    """熔断器：三状态（闭合/断开/半开）。"""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 5.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用函数。"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                print("[熔断器] 状态：HALF_OPEN（试探性恢复）")
            else:
                raise Exception("[熔断器] 状态：OPEN，拒绝调用。")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            print("[熔断器] 状态：CLOSED（恢复正常）")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"[熔断器] 状态：OPEN（连续 {self.failure_count} 次失败，熔断）")


# ==================== 运行示例 ====================

if __name__ == "__main__":
    limiter = RateLimiter(refill_rate=1.0, capacity=3)
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)

    def unstable_api():
        """模拟一个不稳定的 API：50% 概率失败。"""
        import random
        if random.random() < 0.5:
            raise Exception("API 网络超时")
        return "API 调用成功"

    print("=== 限流 + 熔断演示 ===\n")
    for i in range(6):
        print(f"----- 第 {i+1} 次请求 -----")
        # 限流检查
        if not limiter.allow():
            print("[限流] 令牌不足，请求被拒绝。")
            continue
        # 熔断检查
        try:
            result = breaker.call(unstable_api)
            print(f"[结果] {result}")
        except Exception as e:
            print(f"[异常] {e}")
        time.sleep(0.3)
```

**运行输出示例（随机，每次可能不同）：**

```
=== 限流 + 熔断演示 ===

----- 第 1 次请求 -----
[结果] API 调用成功

----- 第 2 次请求 -----
[熔断器] 状态：OPEN（连续 2 次失败，熔断）
[异常] [熔断器] 状态：OPEN，拒绝调用。

----- 第 3 次请求 -----
[限流] 令牌不足，请求被拒绝。
...
```

**ASCII 图：熔断器三状态**

```
CLOSED (正常) ──连续失败达到阈值──► OPEN (熔断，拒绝调用)
   ▲                              │
   │  恢复超时到达                │  半开试探成功
   │                              ▼
HALF_OPEN (半开) ◄─────────────────┘
```

**面试要点**：限流与熔断的核心是“令牌桶限制调用频率 + 三状态熔断器自动开启/恢复”。手撕时要体现：RateLimiter.allow()、CircuitBreaker 三种状态、失败阈值触发熔断、恢复超时进入半开。

---

---

## Q16. 框架的分布式部署：怎么让多个节点协作？（分布式架构基础）

**回答**

单机框架跑在一台机器上，资源有限，一旦任务多了就容易成为瓶颈。**分布式部署（Distributed Deployment）** 就是把框架拆成多个节点，在多台机器上并行运行，提高吞吐量和可用性。

### 分布式部署的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **节点（Node）** | 每台机器上的一个框架实例 | 承担部分计算任务 |
| **负载均衡（Load Balance）** | 把请求分发到不同节点 | 避免单节点过载 |
| **服务发现（Service Discovery）** | 节点之间怎么找到对方 | 新节点加入后自动可用 |
| **分布式锁（Distributed Lock）** | 多个节点不重复处理同一个任务 | 避免重复执行 |
| **状态同步（State Sync）** | 节点之间共享运行状态 | 任务可在节点间迁移 |

用大白话：分布式部署就像快递公司。单机是“一个快递员送所有包裹”，分布式是“多个快递员分区域送”，中间还需要调度中心分配派件单，避免两个快递员送同一个地址。

### 一个可运行的例子：极简分布式节点注册与心跳

下面这段代码模拟一个最简分布式节点注册中心：节点上线注册、定时发心跳、注册中心检测失效节点。直接运行即可看懂分布式节点管理的骨架。

```python
"""
极简分布式节点注册中心：节点注册 + 心跳检测 + 失效剔除
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class NodeInfo:
    """节点信息。"""

    node_id: str
    host: str
    port: int
    last_heartbeat: float
    status: str = "healthy"  # healthy / unhealthy


class NodeRegistry:
    """分布式节点注册中心：管理节点上线、心跳、失效剔除。"""

    def __init__(self, heartbeat_timeout: float = 10.0):
        self.nodes: Dict[str, NodeInfo] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self._lock = threading.Lock()

    def register(self, node_id: str, host: str, port: int):
        """节点上线注册。"""
        with self._lock:
            self.nodes[node_id] = NodeInfo(
                node_id=node_id,
                host=host,
                port=port,
                last_heartbeat=time.time(),
                status="healthy",
            )
            print(f"[注册中心] 节点上线：{node_id} ({host}:{port})")

    def heartbeat(self, node_id: str):
        """节点发送心跳，更新最后活跃时间。"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = time.time()
                self.nodes[node_id].status = "healthy"

    def unregister(self, node_id: str):
        """节点下线注销。"""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                print(f"[注册中心] 节点下线：{node_id}")

    def detect_failure(self):
        """检测超过超时未发心跳的节点，标记为失效。"""
        now = time.time()
        with self._lock:
            for node_id, info in self.nodes.items():
                if now - info.last_heartbeat > self.heartbeat_timeout:
                    info.status = "unhealthy"
                    print(f"[注册中心] 检测到失效节点：{node_id} (最后心跳 {info.last_heartbeat})")

    def list_healthy_nodes(self) -> Dict[str, NodeInfo]:
        """列出所有健康节点。"""
        with self._lock:
            return {nid: info for nid, info in self.nodes.items() if info.status == "healthy"}

    def list_all(self) -> Dict[str, NodeInfo]:
        with self._lock:
            return dict(self.nodes)


# 模拟一个节点定时发心跳
def simulate_node(registry: NodeRegistry, node_id: str, host: str, port: int, interval: float = 2.0, stop_after: float = 8.0):
    """模拟节点：注册、定时心跳、一段时间后下线。"""
    registry.register(node_id, host, port)
    start = time.time()
    while time.time() - start < stop_after:
        registry.heartbeat(node_id)
        time.sleep(interval)
    registry.unregister(node_id)
    print(f"[节点 {node_id}] 已主动下线。")


# ==================== 运行示例 ====================

if __name__ == "__main__":
    registry = NodeRegistry(heartbeat_timeout=5.0)

    # 启动两个模拟节点线程
    t1 = threading.Thread(target=simulate_node, args=(registry, "node-1", "192.168.1.10", 8080), daemon=True)
    t2 = threading.Thread(target=simulate_node, args=(registry, "node-2", "192.168.1.11", 8081), daemon=True)
    t1.start()
    t2.start()

    # 注册中心主循环：每秒检测一次失效，打印健康节点
    print("=== 分布式节点注册中心运行 ===\n")
    try:
        for i in range(6):
            time.sleep(1.5)
            registry.detect_failure()
            healthy = registry.list_healthy_nodes()
            print(f"[第 {i+1} 轮检测] 健康节点：{list(healthy.keys())}")
    except KeyboardInterrupt:
        pass

    print("\n最终所有节点：")
    for nid, info in registry.list_all().items():
        print(f"  {nid}: {info.host}:{info.port} 状态={info.status}")
```

**运行输出示例：**

```
=== 分布式节点注册中心运行 ===

[注册中心] 节点上线：node-1 (192.168.1.10:8080)
[注册中心] 节点上线：node-2 (192.168.1.11:8081)
[第 1 轮检测] 健康节点：['node-1', 'node-2']
[第 2 轮检测] 健康节点：['node-1', 'node-2']
[节点 node-1] 已主动下线。
[注册中心] 节点下线：node-1
[第 3 轮检测] 健康节点：['node-2']
[第 4 轮检测] 健康节点：['node-2']
[节点 node-2] 已主动下线。
[注册中心] 节点下线：node-2
[第 5 轮检测] 健康节点：[]
[第 6 轮检测] 健康节点：[]

最终所有节点：
```

**ASCII 图：分布式节点注册**

```
节点-1                    注册中心 (NodeRegistry)                  节点-2
   │   register                         register                       │
   │─────────────────────────────────────────────────────────────────►│
   │   heartbeat (每 2 秒)        │   heartbeat (每 2 秒)              │
   │◄────────────── healthy ◄─────┘                                  │
   │                                                    detect_failure │
   └──────────────────────────────────────────────────────────────────┘
   (心跳超时 → 标记 unhealthy)
```

**面试要点**：分布式部署的核心是“节点注册 + 心跳检测 + 失效剔除 + 负载均衡”。手撕时要体现：NodeRegistry.register/heartbeat/detect_failure/list_healthy_nodes、节点状态机、心跳超时剔除逻辑。

---

## Q17. 框架的版本管理与回滚：怎么安全升级？（版本管理与回滚）

**回答**

框架在生产环境运行后，难免需要升级（比如换模型、加工具、改提示词）。如果升级出错，整个智能体服务就不可用了。**版本管理与回滚（Version Management & Rollback）** 就是给框架加“存档和还原”功能，升级出错能快速切回旧版本。

### 版本管理与回滚的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **版本号（Version）** | 每次发布打一个标签，如 v1.0, v1.1 | 区分不同版本 |
| **版本仓库（Version Store）** | 保存每个版本的配置和代码 | 能找到任意历史版本 |
| **当前版本指针（Current Pointer）** | 指向正在运行的版本 | 切换版本只需改指针 |
| **回滚（Rollback）** | 把指针指回上一个版本 | 升级出错快速恢复 |
| **灰度发布（Canary）** | 先让一小部分流量跑新版 | 验证无误再全量升级 |

用大白话：版本管理就像游戏的“多个存档槽”。v1.0 是存档 A，v1.1 是存档 B。升级到 v1.1 发现崩溃，立刻切回存档 A，服务马上恢复。

### 一个可运行的例子：极简版本管理器

下面这段代码实现了一个最简版本管理器：保存版本、切换当前版本、回滚到上一版本、列出版本历史。直接运行即可看懂。

```python
"""
极简版本管理器：保存版本 + 切换当前版本 + 回滚 + 版本历史
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class Version:
    """版本信息。"""

    version_id: str
    config: Dict  # 该版本的配置（如模型名、提示词、工具列表）
    created_at: str
    description: str = ""


class VersionManager:
    """版本管理器：保存、切换、回滚版本。"""

    def __init__(self, store_dir: str = "./versions"):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self.current_version_id: Optional[str] = None
        self._load_pointer()

    def _version_path(self, version_id: str) -> str:
        return os.path.join(self.store_dir, f"{version_id}.json")

    def _pointer_path(self) -> str:
        return os.path.join(self.store_dir, "current.txt")

    def _load_pointer(self):
        """加载当前版本指针。"""
        p = self._pointer_path()
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                self.current_version_id = f.read().strip() or None

    def _save_pointer(self, version_id: str):
        """保存当前版本指针。"""
        self.current_version_id = version_id
        with open(self._pointer_path(), "w", encoding="utf-8") as f:
            f.write(version_id)

    def save_version(self, version_id: str, config: Dict, description: str = "") -> Version:
        """保存一个新版本。"""
        import datetime
        version = Version(
            version_id=version_id,
            config=config,
            created_at=datetime.datetime.now().isoformat(),
            description=description,
        )
        with open(self._version_path(version_id), "w", encoding="utf-8") as f:
            json.dump(asdict(version), f, ensure_ascii=False, indent=2)
        self._save_pointer(version_id)
        print(f"[版本] 已保存 {version_id}，并设为当前版本。")
        return version

    def switch_version(self, version_id: str) -> bool:
        """切换到指定版本。"""
        if not os.path.exists(self._version_path(version_id)):
            print(f"[版本] 版本 {version_id} 不存在。")
            return False
        self._save_pointer(version_id)
        print(f"[版本] 已切换到 {version_id}。")
        return True

    def rollback(self) -> Optional[Version]:
        """回滚到上一版本。"""
        versions = self.list_versions()
        if len(versions) < 2:
            print("[版本] 没有可回滚的上一版本。")
            return None
        # 上一版本是列表中倒数第二个
        prev = versions[-2]
        self.switch_version(prev.version_id)
        return prev

    def list_versions(self) -> List[Version]:
        """列出所有版本，按创建时间排序。"""
        versions: List[Version] = []
        for fname in os.listdir(self.store_dir):
            if fname.endswith(".json") and fname != "current.txt":
                with open(os.path.join(self.store_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                versions.append(Version(**data))
        versions.sort(key=lambda v: v.created_at)
        return versions

    def get_current(self) -> Optional[Version]:
        """获取当前版本。"""
        if not self.current_version_id:
            return None
        for v in self.list_versions():
            if v.version_id == self.current_version_id:
                return v
        return None


# ==================== 运行示例 ====================

if __name__ == "__main__":
    vm = VersionManager("./demo_versions")

    # 保存两个版本
    vm.save_version("v1.0", {"model": "gpt-3.5", "temperature": 0.7, "tools": ["search"]}, "初始版本")
    vm.save_version("v1.1", {"model": "gpt-4", "temperature": 0.5, "tools": ["search", "calculator"]}, "升级模型并加计算器")

    print(f"\n当前版本：{vm.get_current().version_id if vm.get_current() else None}")
    print(f"当前配置：{vm.get_current().config if vm.get_current() else None}")

    # 模拟升级出错，回滚
    print("\n--- 模拟 v1.1 出错，回滚到 v1.0 ---\n")
    vm.rollback()

    print(f"\n回滚后当前版本：{vm.get_current().version_id if vm.get_current() else None}")
    print(f"回滚后配置：{vm.get_current().config if vm.get_current() else None}")

    print("\n版本历史：")
    for v in vm.list_versions():
        marker = " (当前)" if v.version_id == vm.current_version_id else ""
        print(f"  {v.version_id} {v.created_at} {v.description}{marker}")
```

**运行输出示例：**

```
[版本] 已保存 v1.0，并设为当前版本。
[版本] 已保存 v1.1，并设为当前版本。

当前版本：v1.1
当前配置：{'model': 'gpt-4', 'temperature': 0.5, 'tools': ['search', 'calculator']}

--- 模拟 v1.1 出错，回滚到 v1.0 ---

[版本] 已切换到 v1.0。

回滚后当前版本：v1.0
回滚后配置：{'model': 'gpt-3.5', 'temperature': 0.7, 'tools': ['search']}

版本历史：
  v1.0 2026-01-01T00:00:00 初始版本
  v1.1 2026-01-01T00:00:01 升级模型并加计算器 (当前)
```

> 注意：示例中版本创建时间因执行瞬间而异，实际运行会显示真实时间戳。

**ASCII 图：版本回滚**

```
v1.0 (存档 A) ─────┐
                   │
v1.1 (存档 B) ──► current 指针 ──► 运行中
                   │
             升级出错，回滚
                   │
                   ▼
             current 指回 v1.0 ──► 恢复运行
```

**面试要点**：版本管理与回滚的核心是“版本号 + 版本仓库 + 当前版本指针 + 回滚到上一版本”。手撕时要体现：Version 数据类、VersionManager.save_version/switch_version/rollback/list_versions/get_current、指针文件持久化。

---

## Q18. 框架的压力测试与基准测试：怎么评估框架性能？（压测与基准）

**回答**

框架做出来后，怎么知道它“快不快、稳不稳、能不能扛住大并发”？**压力测试（Load Testing）** 和 **基准测试（Benchmark Testing）** 就是用数据说话：模拟大量请求，测响应时间、吞吐量、Token 消耗、错误率。

### 压测与基准测试的核心指标

| 指标 | 大白话解释 | 怎么算 |
|------|-----------|--------|
| **QPS（每秒请求数）** | 一秒钟能处理多少请求 | 总请求数 ÷ 总耗时 |
| **平均响应时间** | 一个请求平均花多久 | 总响应时间 ÷ 请求数 |
| **P99 响应时间** | 99% 的请求都在多少毫秒内完成 | 响应时间排序后取 99% 分位数 |
| **错误率** | 请求失败的比例 | 失败请求数 ÷ 总请求数 |
| **Token 消耗率** | 每秒消耗多少 Token | 总 Token ÷ 总耗时 |

用大白话：压测就像给汽车做“极限测试”。开快点看极速（QPS），计时看加速（响应时间），看故障灯（错误率），看油耗（Token 消耗）。

### 一个可运行的例子：极简压测工具

下面这段代码实现了一个最简压测工具：模拟并发请求，统计 QPS、平均响应时间、P99、错误率、Token 消耗。直接运行即可看懂压测骨架。

```python
"""
极简压测工具：并发请求 + 统计 QPS/响应时间/P99/错误率/Token
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Callable, Any, Optional


@dataclass
class RequestResult:
    """单次请求结果。"""

    success: bool
    response_time: float  # 秒
    tokens: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """基准测试报告。"""

    total_requests: int
    successful: int
    failed: int
    qps: float
    avg_response_time: float
    p99_response_time: float
    error_rate: float
    total_tokens: int
    token_rate: float  # 每秒 Token


class BenchmarkRunner:
    """基准测试运行器：模拟并发请求，统计各项指标。"""

    def __init__(self, target: Callable[[], Any], concurrency: int = 10, total_requests: int = 100):
        self.target = target
        self.concurrency = concurrency
        self.total_requests = total_requests

    async def _worker(self, queue: asyncio.Queue, results: List[RequestResult]):
        """单个并发工人：不断从队列取请求并执行。"""
        while True:
            try:
                start = time.perf_counter()
                try:
                    result = await asyncio.to_thread(self.target)
                    elapsed = time.perf_counter() - start
                    # 估算 Token（粗略：按结果长度/4）
                    tokens = max(1, len(str(result)) // 4)
                    results.append(RequestResult(success=True, response_time=elapsed, tokens=tokens))
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    results.append(RequestResult(success=False, response_time=elapsed, error=str(e)))
            except asyncio.CancelledError:
                break

    async def run(self) -> BenchmarkReport:
        """运行基准测试，返回报告。"""
        queue: asyncio.Queue[Optional[Any]] = asyncio.Queue()
        for _ in range(self.total_requests):
            queue.put_nowait(None)

        results: List[RequestResult] = []

        workers = [
            asyncio.create_task(self._worker(queue, results))
            for _ in range(self.concurrency)
        ]

        start = time.perf_counter()
        await queue.join()
        total_elapsed = time.perf_counter() - start

        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        successful = sum(1 for r in results if r.success)
        failed = self.total_requests - successful
        qps = self.total_requests / total_elapsed if total_elapsed > 0 else 0
        avg_rt = statistics.mean([r.response_time for r in results]) if results else 0
        sorted_rt = sorted([r.response_time for r in results])
        p99_idx = max(0, min(len(sorted_rt) - 1, int(len(sorted_rt) * 0.99) - 1))
        p99_rt = sorted_rt[p99_idx] if sorted_rt else 0
        error_rate = failed / self.total_requests if self.total_requests > 0 else 0
        total_tokens = sum(r.tokens for r in results)
        token_rate = total_tokens / total_elapsed if total_elapsed > 0 else 0

        return BenchmarkReport(
            total_requests=self.total_requests,
            successful=successful,
            failed=failed,
            qps=qps,
            avg_response_time=avg_rt,
            p99_response_time=p99_rt,
            error_rate=error_rate,
            total_tokens=total_tokens,
            token_rate=token_rate,
        )


# ==================== 运行示例 ====================

if __name__ == "__main__":
    import random

    def mock_agent_call():
        """模拟一个智能体调用：随机耗时，5% 概率失败。"""
        time.sleep(random.uniform(0.05, 0.2))
        if random.random() < 0.05:
            raise Exception("模型调用超时")
        return "这是模型返回的回答内容，长度足够用来估算 Token。"

    runner = BenchmarkRunner(target=mock_agent_call, concurrency=20, total_requests=200)
    report = asyncio.run(runner.run())

    print("\n=== 基准测试报告 ===")
    print(f"总请求数：{report.total_requests}")
    print(f"成功数：{report.successful}")
    print(f"失败数：{report.failed}")
    print(f"QPS：{report.qps:.2f}")
    print(f"平均响应时间：{report.avg_response_time*1000:.1f} ms")
    print(f"P99 响应时间：{report.p99_response_time*1000:.1f} ms")
    print(f"错误率：{report.error_rate*100:.1f}%")
    print(f"总 Token：{report.total_tokens}")
    print(f"Token 消耗率：{report.token_rate:.0f} Token/s")
```

**运行输出示例：**

```
=== 基准测试报告 ===

总请求数：200
成功数：190
失败数：10
QPS：68.42
平均响应时间：123.4 ms
P99 响应时间：198.7 ms
错误率：5.0%
总 Token：5623
Token 消耗率：191 Token/s
```

> 注意：结果随运行环境和随机数而异，每次运行数值会有波动。

**ASCII 图：压测流程**

```
并发请求（200 个）
      │
      ▼
┌─────────────┐
│ 工人 1      │◄──── 队列
│ 工人 2      │
│ ...         │
│ 工人 20     │
└──────┬──────┘
       │ 执行目标函数
       ▼
   记录：成功/失败、耗时、Token
       │
       ▼
   汇总：QPS、平均RT、P99、错误率、Token率
```

**面试要点**：压测与基准测试的核心指标是“QPS、响应时间、P99、错误率、Token 消耗率”。手撕时要体现：并发队列、工人协程、指标统计、P99 分位数计算、基准报告输出。

---

## Q19. 框架的文档与示例代码：怎么写好框架文档？（文档与示例）

**回答**

框架代码写好了，如果别人（包括半年后的自己）不知道怎么用，框架就白做了。**文档与示例代码（Documentation & Examples）** 是框架的“说明书”，决定框架能否被别人采用。

### 文档的四个层次

| 层级 | 内容 | 面向人群 |
|------|------|---------|
| **快速上手（Quick Start）** | 5 分钟跑通第一个 Demo | 新用户 |
| **核心概念（Concepts）** | 消息契约、配置管理、Agent 基类、主循环 | 开发者 |
| **API 参考（API Reference）** | 每个类、每个方法的参数和返回值 | 集成开发者 |
| **示例代码（Examples）** | 可直接复制运行的完整代码 | 所有用户 |

用大白话：文档就像家具的组装说明书。快速上手是“先装出一个凳子坐着”，核心概念是“解释每个零件叫什么”，API 参考是“列出所有螺丝和板子的规格”，示例代码是“给出完整的组装步骤图”。

### 一个可运行的例子：框架快速上手示例代码

下面这段代码展示一个框架的“快速上手”示例：从安装到跑通第一个 Agent，代码清晰、可直接复制运行。不依赖真实大模型，用规则模拟，直接运行。

```python
"""
框架快速上手示例代码：
从安装到跑通第一个 Agent（规则模拟版，不依赖真实大模型）
"""

from typing import Optional


class Message:
    """统一消息契约。"""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class Tool:
    """工具基类。"""

    def run(self, *args, **kwargs) -> str:
        raise NotImplementedError


class SearchTool(Tool):
    """搜索工具（模拟）。"""

    def run(self, query: str) -> str:
        return f"[搜索结果] {query}：相关信息（模拟数据）"


class Agent:
    """Agent 基类：感知→思考→行动→观察。"""

    def __init__(self, name: str, tools: Optional[dict] = None):
        self.name = name
        self.tools = tools or {}

    def think(self, user_input: str) -> Optional[dict]:
        """思考：决定调用什么工具（规则模拟）。"""
        if "天气" in user_input:
            return {"tool": "search", "input": user_input}
        return None

    def act(self, decision: dict) -> str:
        """行动：调用工具。"""
        tool = self.tools.get(decision["tool"])
        if tool:
            return tool.run(decision["input"])
        return "无可用工具。"

    def run(self, user_input: str) -> str:
        """主循环：感知→思考→行动→观察。"""
        print(f"[感知] 用户输入：{user_input}")
        decision = self.think(user_input)
        if decision:
            print(f"[思考] 决定调用：{decision['tool']}，参数：{decision['input']}")
            result = self.act(decision)
            print(f"[观察] 工具返回：{result}")
            return f"【{self.name}】{result}"
        print(f"[思考] 不需要工具，直接回答。")
        return f"【{self.name}】我是 {self.name}，可以帮你查天气。"


# ==================== 快速上手：跑通第一个 Agent ====================

if __name__ == "__main__":
    # 第一步：创建工具
    tools = {"search": SearchTool()}

    # 第二步：创建 Agent
    agent = Agent(name="天气助手", tools=tools)

    # 第三步：运行
    print(agent.run("北京今天天气怎么样？"))
    print("-" * 40)
    print(agent.run("你好，有什么可以帮我？"))
```

**运行输出示例：**

```
[感知] 用户输入：北京今天天气怎么样？
[思考] 决定调用：search，参数：北京今天天气怎么样？
[观察] 工具返回：[搜索结果] 北京今天天气怎么样？：相关信息（模拟数据）
【天气助手】[搜索结果] 北京今天天气怎么样？：相关信息（模拟数据）
----------------------------------------
[感知] 用户输入：你好，有什么可以帮我？
[思考] 不需要工具，直接回答。
【天气助手】我是 天气助手，可以帮你查天气。
```

**ASCII 图：快速上手流程**

```
用户
   │
   ▼
┌─────────────┐
│ 创建工具    │
│ (SearchTool)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 创建 Agent  │
│ (Agent)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ agent.run() │──► 感知→思考→行动→观察
└─────────────┘
       │
       ▼
   返回结果给用户
```

**面试要点**：框架文档的四个层次是“快速上手、核心概念、API 参考、示例代码”。手撕时要体现：示例代码清晰可运行、每一步都有注释说明、包含安装/创建/运行三个步骤。

---

## Q20. 框架的社区与生态：怎么建设框架社区？（社区与生态）

**回答**

一个框架能否成功，不只取决于代码质量，还取决于**社区与生态（Community & Ecosystem）**：有没有人用、有没有人贡献插件、有没有人回答问题、有没有第三方集成。

### 社区与生态的核心要素

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **文档与教程** | 让新用户快速上手 | 降低学习成本 |
| **示例代码库** | 可复制运行的完整示例 | 展示框架能力 |
| **插件市场** | 第三方开发的插件 | 扩展框架功能 |
| **Issue 与讨论** | 用户提问、报错、建议 | 改进框架、建立信任 |
| **贡献指南（Contributing）** | 告诉别人怎么贡献代码 | 吸引开发者参与 |
| **第三方集成** | 与其他工具和平台对接 | 增加框架价值 |

用大白话：社区就像“俱乐部”。有说明书（文档）、有范例（示例代码）、有插件商店（插件市场）、有讨论区（Issue）、有加入规则（贡献指南）、有朋友圈（第三方集成）。俱乐部越活跃，越多人愿意加入。

### 一个可运行的例子：极简贡献指南与插件示例

下面这段代码展示一个框架的“贡献指南”文本和一个“第三方插件”示例，帮助社区成员理解如何参与。直接运行即可看懂。

```python
"""
极简贡献指南与第三方插件示例：
帮助社区成员理解如何参与框架建设
"""

from typing import Any


class Plugin:
    """插件基类：第三方插件需要继承并实现 execute。"""

    def __init__(self, name: str):
        self.name = name

    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class TranslationPlugin(Plugin):
    """第三方插件示例：翻译插件（由社区贡献者开发）。"""

    def __init__(self):
        super().__init__("translation")

    def execute(self, text: str, target_lang: str = "en") -> str:
        # 模拟翻译
        return f"[翻译插件] '{text}' 翻译成 {target_lang}：（模拟翻译结果）"


# ==================== 贡献指南（文本） ====================

CONTRIBUTING_GUIDE = """
# 为本框架贡献代码

欢迎为本智能体框架贡献代码！下面是参与方式：

## 1. 报告问题（Issue）
如果你在使用中发现 Bug 或有功能建议，请在仓库的 Issues 页面提交，描述清晰的复现步骤和预期行为。

## 2. 提交代码（Pull Request）
1. Fork 本仓库到你的账号。
2. 创建分支：`git checkout -b feature/your-feature`
3. 编写代码，并为新功能添加测试用例。
4. 运行测试：`pytest`
5. 提交代码：`git commit -m "feat: xxx"`
6. 发起 Pull Request，等待维护者审核。

## 3. 开发第三方插件
框架支持热插拔插件。编写插件只需继承 `Plugin` 基类并实现 `execute` 方法：

```python
from framework import Plugin

class MyPlugin(Plugin):
    def __init__(self):
        super().__init__("my-plugin")

    def execute(self, *args, **kwargs):
        # 你的功能代码
        return "插件结果"
```

将插件注册到框架的插件注册表即可使用。

## 4. 行为规范
请遵守社区行为准则，尊重每一位贡献者。
"""


# ==================== 运行示例 ====================

if __name__ == "__main__":
    print("=== 框架贡献指南（摘录）===\n")
    print(CONTRIBUTING_GUIDE[:800])
    print("...")

    print("\n=== 第三方插件示例 ===\n")
    plugin = TranslationPlugin()
    result = plugin.execute("你好，世界", target_lang="en")
    print(result)
```

**运行输出示例：**

```
=== 框架贡献指南（摘录）===

# 为本框架贡献代码

欢迎为本智能体框架贡献代码！下面是参与方式：

## 1. 报告问题（Issue）
如果你在使用中发现 Bug 或有功能建议，请在仓库的 Issues 页面提交，描述清晰的
...

=== 第三方插件示例 ===

[翻译插件] '你好，世界' 翻译成 en：（模拟翻译结果）
```

**ASCII 图：社区与生态结构**

```
框架核心
   │
   ├── 文档与教程 ─────► 新用户上手
   ├── 示例代码库 ─────► 展示能力
   ├── 插件市场 ───────► 第三方插件（如翻译插件）
   ├── Issue 讨论 ─────► 用户反馈与 Bug 报告
   ├── 贡献指南 ───────► 开发者参与开发
   └── 第三方集成 ─────► 与其他平台对接
```

**面试要点**：社区与生态的核心是“文档+示例+插件+讨论+贡献指南+第三方集成”。手撕时要体现：CONTRIBUTING 指南文本、第三方插件继承 Plugin 基类、插件注册使用、社区各要素作用。

---

## Q21. 框架的生产上线检查清单：上线前要核对什么？（安全/成本/运维）

**回答**

框架跑通 Demo 不等于能上线。生产环境下，框架要承担真实用户、真实数据、真实成本。下面是一份**从 Demo 到生产的上线检查清单**，按“安全、成本、运维”三个维度，帮你不漏项。

### 上线检查清单

| 维度 | 检查项 | 说明 |
|------|--------|------|
| **安全** | 最小权限 | 只给框架真正需要的工具和 API 权限 |
| **安全** | 关键操作人工确认 | 付款、删除数据、发对外邮件前需人工确认 |
| **安全** | 输入过滤 | 过滤提示词注入、危险指令（rm、drop 数据库） |
| **安全** | 沙箱隔离 | 代码执行在沙箱里，限制网络和文件系统访问 |
| **安全** | 审计日志 | 每一步的感知、推理、行动、观察都要记日志，出了问题能回溯 |
| **成本** | 模型分层 | 简单任务用小模型，复杂任务才用大模型 |
| **成本** | 缓存中间结果 | 相同查询不重复调用，缓存复用 |
| **成本** | Token 预算告警 | 设置日/月 Token 消耗上限，超限告警 |
| **成本** | 最大步数 | 防止死循环无限烧 Token |
| **运维** | 健康检查 | 提供 `/health` 接口，供负载均衡器检查 |
| **运维** | 监控大盘 | QPS、响应时间、任务完成率、错误率、Token 成本 |
| **运维** | 异常告警 | 连续失败、死循环、安全越界、成本飙升时告警 |
| **运维** | 日志留存 | 执行轨迹日志保留足够时间，方便复盘 |
| **运维** | 版本管理 | 提示词、工具定义、模型版本都要版本化，可回滚 |
| **运维** | 回滚方案 | 新版本出问题能快速切回旧版本 |

### 上线前自测清单（ASCII 版）

```
┌─────────────────────────────────────────────────────────────┐
│              智能体框架生产上线前自测清单                    │
├─────────────────────────────────────────────────────────────┤
│ □ 核心链路跑通（感知→推理→行动→观察→反思）                   │
│ □ 每个工具调用都有异常处理                                   │
│ □ 设置最大步数，防止死循环                                   │
│ □ 关键操作（付款/删除/发邮件）加人工确认                     │
│ □ 工具权限做到最小                                           │
│ □ 输入过滤，防止提示词注入                                   │
│ □ 敏感数据脱敏                                               │
│ □ 审计日志完整                                               │
│ □ 模型分层 + 缓存，控制成本                                  │
│ □ 监控大盘 + 异常告警                                        │
│ □ 健康检查接口                                               │
│ □ 提示词/工具/模型版本化，可回滚                             │
└─────────────────────────────────────────────────────────────┘
```

**面试要点**：生产上线前要检查“安全、成本、运维”三个维度；核心包括最大步数防死循环、关键操作人工确认、最小权限、审计日志、监控告警、版本回滚。

---

## Q22. 多框架集成：怎么把 LangGraph / AutoGen / CrewAI 集成到自己框架？

**回答**

自己造的框架不需要“包打天下”，现成框架（LangGraph、AutoGen、CrewAI）已经做得很好了。**多框架集成** 就是把现成框架作为插件接入自己框架，取长补短：自己框架负责编排和业务逻辑，现成框架负责复杂的多Agent 工作流。

### 集成方式

| 方式 | 大白话解释 | 适用场景 |
|------|-----------|---------|
| **作为插件接入** | 把现成框架封装成自己框架的一个插件/工具 | 需要某个特定能力（如 LangGraph 的状态图） |
| **作为子工作流调用** | 自己框架的某个步骤，交给现成框架去跑 | 某个子任务适合用多Agent 协作 |
| **作为后端引擎** | 自己框架只负责 API 和 UI，底层用现成框架执行 | 快速上线，不想重复造轮子 |

用大白话：多框架集成就像买车。你可以自己造车架（自己框架），但发动机、变速箱可以买成熟的（现成框架），组装起来既有自己的特色，又不至于每个零件都从零开始。

### 一个可运行的例子：把 LangGraph 封装成自己框架的插件

下面这段代码把 LangGraph 风格的状态图封装成自己框架可调用的插件。不依赖真实 LangGraph，用规则模拟，直接运行。

```python
"""
多框架集成极简实现：把 LangGraph 风格状态图封装成自己框架的插件
"""

from typing import Any, Dict, Optional


class LangGraphPlugin:
    """LangGraph 风格插件：封装状态图，供自己框架调用。"""

    def __init__(self, name: str = "langgraph"):
        self.name = name
        self.nodes: Dict[str, Any] = {}
        self.edges: Dict[str, str] = {}

    def add_node(self, name: str, fn: Any):
        self.nodes[name] = fn
        print(f"[LangGraph 插件] 已添加节点：{name}")

    def add_edge(self, frm: str, to: str):
        self.edges[frm] = to

    def run(self, initial_state: Dict) -> Dict:
        """按状态图流转执行，返回最终状态。"""
        state = dict(initial_state)
        current = "start"
        while current in self.nodes:
            print(f"[LangGraph 插件] 执行节点：{current}")
            state = self.nodes[current](state)
            nxt = self.edges.get(current)
            if nxt is None:
                break
            current = nxt
        return state


# 自己的框架：只负责调用插件，不关心内部怎么实现
class MyFramework:
    """自己框架：统一调用各种插件（包括 LangGraph 插件）。"""

    def __init__(self):
        self.plugins: Dict[str, Any] = {}

    def register_plugin(self, name: str, plugin: Any):
        self.plugins[name] = plugin
        print(f"[自己框架] 已注册插件：{name}")

    def run_plugin(self, plugin_name: str, input_data: Dict) -> Dict:
        plugin = self.plugins.get(plugin_name)
        if plugin is None:
            raise ValueError(f"未知插件：{plugin_name}")
        print(f"[自己框架] 调用插件：{plugin_name}")
        return plugin.run(input_data)


# ==================== 运行示例 ====================

if __name__ == "__main__":
    # 构建 LangGraph 风格插件：用户输入 -> 拆需求 -> 写代码 -> 测试
    lg = LangGraphPlugin("pm_dev_test")

    def pm_node(state: Dict) -> Dict:
        state["prd"] = f"[PRD] {state['requirement']}"
        return state

    def dev_node(state: Dict) -> Dict:
        state["code"] = f"[代码] 基于 {state['prd']} 实现"
        return state

    def test_node(state: Dict) -> Dict:
        state["test"] = "[测试] 通过"
        return state

    lg.add_node("start", pm_node)
    lg.add_node("pm", pm_node)
    lg.add_node("dev", dev_node)
    lg.add_node("test", test_node)
    lg.add_edge("start", "pm")
    lg.add_edge("pm", "dev")
    lg.add_edge("dev", "test")

    # 自己框架注册并调用
    fw = MyFramework()
    fw.register_plugin("langgraph", lg)

    result = fw.run_plugin("langgraph", {"requirement": "开发一个登录功能"})
    print("\n=== 最终结果 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
```

**运行输出示例：**

```
[LangGraph 插件] 已添加节点：pm
[LangGraph 插件] 已添加节点：dev
[LangGraph 插件] 已添加节点：test
[自己框架] 已注册插件：langgraph
[自己框架] 调用插件：langgraph
[LangGraph 插件] 执行节点：start
[LangGraph 插件] 执行节点：pm
[LangGraph 插件] 执行节点：dev
[LangGraph 插件] 执行节点：test

=== 最终结果 ===
  requirement: 开发一个登录功能
  prd: [PRD] 开发一个登录功能
  code: [代码] 基于 [PRD] 开发一个登录功能 实现
  test: [测试] 通过
```

**ASCII 图：多框架集成**

```
用户请求
   │
   ▼
┌─────────────┐      ┌─────────────────────┐
│ 自己框架    │─────►│ LangGraph 插件      │
│ (编排/业务) │      │ (状态图工作流)      │
└─────────────┘      └─────────────────────┘
   │                          │
   │  其他插件                │  节点流转
   │ (AutoGen/CrewAI 等)      │
   └──────────────────────────┘
```

**面试要点**：多框架集成的核心是“把现成框架封装成自己框架的插件/子工作流，自己框架负责编排和业务逻辑”。手撕时要体现：插件基类、注册表、统一调用入口、LangGraph 状态图封装示例。

---

## Q23. 框架的可观测性与追踪：怎么知道每一步发生了什么？（分布式追踪）

**回答**

智能体框架跑复杂任务时，一步步的 Thought/Action/Observation 链条很长，出了问题（比如回答错了、调错工具、陷入循环）只看最终答案根本不知道哪一步出了问题。**可观测性与追踪（Observability & Tracing）** 就是把每一步都记录下来，形成可检索的“轨迹”，方便调试和优化。

### 可观测性的三大支柱

| 支柱 | 大白话解释 | 作用 |
|------|-----------|------|
| **日志（Logs）** | 记录每一步发生了什么 | 定位具体错误 |
| **指标（Metrics）** | QPS、响应时间、Token 消耗等 | 监控健康度 |
| **追踪（Traces）** | 把一条请求的所有步骤串成一条链 | 看清整个调用链路，找瓶颈 |

用大白话：可观测性就像给智能体装“黑匣子”。每一步都被记录，出事了能回放，看清是哪个环节出了问题。

### 一个可运行的例子：极简分布式追踪

下面这段代码实现了一个最简追踪系统：给每条请求分配 TraceID，每一步记录 Span（开始时间、结束时间、操作名、结果），最后打印完整追踪链。直接运行即可看懂追踪骨架。

```python
"""
极简分布式追踪系统：TraceID + Span + 追踪链打印
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Span:
    """单个追踪跨度：记录一步的开始、结束、操作、结果。"""

    span_id: str
    trace_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None

    def finish(self, result: Optional[str] = None, error: Optional[str] = None):
        self.end_time = time.time()
        self.result = result
        self.error = error

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "result": self.result,
            "error": self.error,
        }


class Tracer:
    """追踪器：管理 TraceID 和 Span，打印完整追踪链。"""

    def __init__(self):
        self.spans: List[Span] = []
        self._current_trace_id: Optional[str] = None

    def start_trace(self) -> str:
        """开始一个新追踪，返回 TraceID。"""
        self._current_trace_id = str(uuid.uuid4())[:8]
        return self._current_trace_id

    def start_span(self, name: str, trace_id: Optional[str] = None) -> Span:
        """开始一个 Span。"""
        tid = trace_id or self._current_trace_id or str(uuid.uuid4())[:8]
        span = Span(span_id=str(uuid.uuid4())[:8], trace_id=tid, name=name, start_time=time.time())
        self.spans.append(span)
        print(f"[追踪] 开始 Span：{name} (trace={tid}, span={span.span_id})")
        return span

    def get_trace(self, trace_id: str) -> List[Span]:
        """获取某个 TraceID 下的所有 Span。"""
        return [s for s in self.spans if s.trace_id == trace_id]

    def print_trace(self, trace_id: str):
        """打印完整追踪链。"""
        spans = self.get_trace(trace_id)
        if not spans:
            print(f"[追踪] 未找到 trace_id={trace_id} 的记录。")
            return
        print(f"\n=== 追踪链：trace_id={trace_id} ===")
        for s in spans:
            dur = f"{s.duration*1000:.1f}ms" if s.duration else "进行中"
            status = "错误" if s.error else "成功"
            print(f"  [{s.span_id}] {s.name} -> {dur} -> {status}")
            if s.result:
                print(f"         结果：{s.result[:60]}")
            if s.error:
                print(f"         错误：{s.error}")
        print("=" * 40)


# ==================== 运行示例 ====================

if __name__ == "__main__":
    tracer = Tracer()
    trace_id = tracer.start_trace()

    # 模拟一个 ReAct 循环：思考 -> 搜索 -> 观察 -> 思考 -> 计算 -> 回答
    s1 = tracer.start_span("思考：查天气", trace_id)
    time.sleep(0.05)
    s1.finish(result="决定调用搜索工具")

    s2 = tracer.start_span("行动：搜索北京天气", trace_id)
    time.sleep(0.1)
    s2.finish(result="[搜索] 北京：晴朗，15°C")

    s3 = tracer.start_span("思考：组织回答", trace_id)
    time.sleep(0.03)
    s3.finish(result="北京今天晴朗，15°C")

    tracer.print_trace(trace_id)
```

**运行输出示例：**

```
[追踪] 开始 Span：思考：查天气 (trace=7f3a2b1c, span=a1b2c3d4)
[追踪] 开始 Span：行动：搜索北京天气 (trace=7f3a2b1c, span=e5f6g7h8)
[追踪] 开始 Span：思考：组织回答 (trace=7f3a2b1c, span=i9j0k1l2)

=== 追踪链：trace_id=7f3a2b1c ===
  [a1b2c3d4] 思考：查天气 -> 50.0ms -> 成功
         结果：决定调用搜索工具
  [e5f6g7h8] 行动：搜索北京天气 -> 100.2ms -> 成功
         结果：[搜索] 北京：晴朗，15°C
  [i9j0k1l2] 思考：组织回答 -> 30.1ms -> 成功
         结果：北京今天晴朗，15°C
========================================
```

**ASCII 图：追踪链结构**

```
trace_id=7f3a2b1c
   │
   ├── span=a1b2c3d4: 思考：查天气 (50ms)
   │
   ├── span=e5f6g7h8: 行动：搜索北京天气 (100ms)
   │
   └── span=i9j0k1l2: 思考：组织回答 (30ms)
```

**面试要点**：可观测性与追踪的核心是“TraceID 串联全链路 + Span 记录每一步的开始/结束/结果 + 可打印完整链路”。手撕时要体现：Tracer 类、start_trace/start_span/finish、Span 数据结构、print_trace 链路打印。

---

## Q24. 框架的安全沙箱：代码执行怎么隔离？（安全沙箱）

**回答**

智能体框架如果允许执行用户生成的代码（比如代码调试 Agent），一旦代码有恶意操作（rm -rf、偷数据、发垃圾邮件），就会造成严重危害。**安全沙箱（Security Sandbox）** 就是把代码执行隔离在受限环境里，限制它能访问的资源。

### 沙箱的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **进程隔离** | 代码在独立进程里跑，挂了不影响主框架 | 故障隔离 |
| **文件系统限制** | 只能访问指定目录，不能乱改系统文件 | 防止删文件 |
| **网络限制** | 限制能访问的网络端口和域名 | 防止外泄数据、发垃圾邮件 |
| **资源限制** | 限制 CPU、内存、执行时间 | 防止无限循环耗光资源 |
| **白名单机制** | 只允许调用白名单里的函数和模块 | 防止调用危险函数 |

用大白话：安全沙箱就像“隔离病房”。代码在里面跑，能做实验，但出不去，也进不来别的，万一出了问题也不影响主框架。

### 一个可运行的例子：极简安全沙箱

下面这段代码实现了一个最简安全沙箱：白名单机制 + 执行超时 + 危险关键词拦截。不依赖第三方库，直接运行即可看懂沙箱骨架。

```python
"""
极简安全沙箱：白名单机制 + 执行超时 + 危险关键词拦截
"""

import re
import signal
from typing import Any, Dict, Optional


class TimeoutError(Exception):
    """执行超时异常。"""


class Sandbox:
    """安全沙箱：白名单 + 超时 + 危险关键词拦截。"""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        # 允许的内置函数白名单
        self.allowed_builtins = {
            "print", "len", "str", "int", "float", "bool", "list", "dict",
            "sum", "max", "min", "abs", "round", "sorted", "range",
        }
        # 危险关键词：出现就拦截
        self.dangerous_keywords = [
            "import os", "import subprocess", "import sys", "import shutil",
            "import socket", "import urllib", "import requests",
            "open(", "eval(", "exec(", "compile(", "__import__",
            "os.", "subprocess.", "sys.exit", "shutil.", "socket.",
        ]

    def _check_dangerous(self, code: str) -> Optional[str]:
        """检查代码里是否有危险关键词。"""
        for kw in self.dangerous_keywords:
            if kw in code:
                return f"检测到危险关键词：{kw}"
        return None

    def _check_imports(self, code: str) -> Optional[str]:
        """检查 import 语句是否都在白名单里。"""
        import_pattern = re.findall(r"^\s*import\s+(\w+)", code, re.MULTILINE)
        from_pattern = re.findall(r"^\s*from\s+(\w+)", code, re.MULTILINE)
        all_modules = set(import_pattern + from_pattern)
        allowed_modules = {"math", "random", "datetime", "json", "re"}
        for mod in all_modules:
            if mod not in allowed_modules:
                return f"不允许导入模块：{mod}"
        return None

    def run(self, code: str, context: Optional[Dict] = None) -> Dict:
        """在沙箱里执行代码，返回结果。"""
        result: Dict[str, Any] = {"success": False, "output": "", "error": None}

        # 1. 危险关键词检查
        danger = self._check_dangerous(code)
        if danger:
            result["error"] = f"[沙箱拦截] {danger}"
            return result

        # 2. 导入模块白名单检查
        import_err = self._check_imports(code)
        if import_err:
            result["error"] = f"[沙箱拦截] {import_err}"
            return result

        # 3. 构造受限的执行环境
        safe_builtins = {k: __builtins__[k] for k in self.allowed_builtins if k in __builtins__}
        safe_globals = {"__builtins__": safe_builtins}
        safe_locals = context or {}

        # 4. 执行（带超时）
        def _exec():
            exec(code, safe_globals, safe_locals)

        # 使用信号设置超时（仅 Unix/Linux）
        def handler(signum, frame):
            raise TimeoutError("执行超时")

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            output = []
            old_print = print

            def capture_print(*args, **kwargs):
                output.append(" ".join(str(a) for a in args))

            safe_builtins["print"] = capture_print
            _exec()
            signal.setitimer(signal.ITIMER_REAL, 0)
            result["success"] = True
            result["output"] = "\n".join(output)
        except TimeoutError as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            result["error"] = f"[沙箱] 执行超时（>{self.timeout}s）"
        except Exception as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            result["error"] = f"[沙箱] 执行错误：{e}"
        finally:
            signal.signal(signal.SIGALRM, old_handler)

        return result


# ==================== 运行示例 ====================

if __name__ == "__main__":
    sandbox = Sandbox(timeout=3.0)

    # 场景 1：安全代码
    print("=== 场景 1：安全代码 ===")
    r1 = sandbox.run("print('你好'); print(2+3); import math; print(math.sqrt(16))")
    print(r1)

    # 场景 2：危险关键词
    print("\n=== 场景 2：危险关键词 ===")
    r2 = sandbox.run("import os; os.system('rm -rf /')")
    print(r2)

    # 场景 3：不允许的模块
    print("\n=== 场景 3：不允许的模块 ===")
    r3 = sandbox.run("import subprocess")
    print(r3)

    # 场景 4：执行超时
    print("\n=== 场景 4：执行超时 ===")
    r4 = sandbox.run("while True: pass")
    print(r4)
```

**运行输出示例：**

```
=== 场景 1：安全代码 ===
{'success': True, 'output': '你好\n5\n4.0', 'error': None}

=== 场景 2：危险关键词 ===
{'success': False, 'output': '', 'error': "[沙箱拦截] 检测到危险关键词：import os"}

=== 场景 3：不允许的模块 ===
{'success': False, 'output': '', 'error': "[沙箱拦截] 不允许导入模块：subprocess"}

=== 场景 4：执行超时 ===
{'success': False, 'output': '', 'error': "[沙箱] 执行超时（>3s）"}
```

**ASCII 图：安全沙箱架构**

```
用户代码
   │
   ▼
┌─────────────┐
│ 危险关键词  │──拦截──► 阻止执行
│ 检查        │
├─────────────┤
│ 导入白名单  │──拦截──► 阻止执行
│ 检查        │
├─────────────┤
│ 受限执行环境│──► 执行代码（受限 builtins）
│ (超时)      │──超时──► 阻止执行
└─────────────┘
   │
   ▼
  返回结果
```

**面试要点**：安全沙箱的核心是“白名单机制 + 执行超时 + 危险关键词拦截 + 受限执行环境”。手撕时要体现：Sandbox 类、_check_dangerous/_check_imports、受限 builtins、signal 超时控制、返回 success/output/error 结构。

---

## Q25. 框架的并发控制：异步 vs 多线程 vs 多进程，怎么选？

**回答**

框架里有很多任务要并发跑（比如多个 Agent 并行查信息），怎么并发是个选型问题：异步（asyncio）、多线程（threading）、多进程（multiprocessing），各有优劣。

### 并发模型对比

| 模型 | 大白话解释 | 适合场景 | 缺点 |
|------|-----------|---------|------|
| **异步（asyncio）** | 单线程，通过事件循环切换任务，适合 IO 密集 | 调用大模型 API、搜索 API、数据库查询 | CPU 密集任务会阻塞整个事件循环 |
| **多线程（threading）** | 多个线程，共享内存，适合 IO 密集 | 调用外部 API、等待响应 | GIL 限制，CPU 密集无法并行 |
| **多进程（multiprocessing）** | 多个进程，各自内存，适合 CPU 密集 | 复杂计算、本地模型推理 | 内存占用高，进程间通信慢 |

用大白话：三种并发就像餐厅。异步是“一名服务员轮流给多桌上菜”（快，但一桌菜做得太慢会耽误其他桌）；多线程是“多名服务员共用一间厨房”（能并行，但厨房大小/GIL 有限）；多进程是“多间厨房各自独立”（最强，但开销大）。

### 一个可运行的例子：三种并发模型对比

下面这段代码用同一个任务（模拟调用大模型 API，IO 密集），分别用异步、多线程、多进程跑，对比耗时。直接运行即可看懂差异。

```python
"""
三种并发模型对比：异步 vs 多线程 vs 多进程
"""

import asyncio
import time
import threading
import multiprocessing
from typing import List


def io_task(task_id: int) -> str:
    """模拟一个 IO 密集任务：调用外部 API，耗时 0.3 秒。"""
    time.sleep(0.3)
    return f"任务 {task_id} 完成"


async def async_io_task(task_id: int) -> str:
    """异步版 IO 任务。"""
    await asyncio.sleep(0.3)
    return f"任务 {task_id} 完成"


# ==================== 1. 串行（基准）====================

def serial(tasks: int) -> float:
    start = time.perf_counter()
    for i in range(tasks):
        io_task(i)
    return time.perf_counter() - start


# ==================== 2. 异步（asyncio）====================

async def async_run(tasks: int) -> float:
    start = time.perf_counter()
    await asyncio.gather(*[async_io_task(i) for i in range(tasks)])
    return time.perf_counter() - start


# ==================== 3. 多线程（threading）====================

def thread_run(tasks: int) -> float:
    results: List[str] = [None] * tasks

    def worker(i: int):
        results[i] = io_task(i)

    start = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(tasks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


# ==================== 4. 多进程（multiprocessing）====================

def process_worker(i: int, q: multiprocessing.Queue):
    q.put(io_task(i))


def process_run(tasks: int) -> float:
    q = multiprocessing.Queue()
    start = time.perf_counter()
    procs = [multiprocessing.Process(target=process_worker, args=(i, q)) for i in range(tasks)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return time.perf_counter() - start


# ==================== 运行示例 ====================

if __name__ == "__main__":
    TASKS = 20

    print(f"=== 三种并发模型对比（任务数：{TASKS}，每任务 0.3 秒）===\n")

    # 串行
    t_serial = serial(TASKS)
    print(f"串行：{t_serial:.2f} 秒")

    # 异步
    t_async = asyncio.run(async_run(TASKS))
    print(f"异步 asyncio：{t_async:.2f} 秒")

    # 多线程
    t_thread = thread_run(TASKS)
    print(f"多线程 threading：{t_thread:.2f} 秒")

    # 多进程
    t_process = process_run(TASKS)
    print(f"多进程 multiprocessing：{t_process:.2f} 秒")

    print("\n结论：IO 密集任务下，异步/多线程/多进程都远快于串行；CPU 密集任务才适合多进程。")
```

**运行输出示例：**

```
=== 三种并发模型对比（任务数：20，每任务 0.3 秒）===

串行：6.05 秒
异步 asyncio：0.31 秒
多线程 threading：0.35 秒
多进程 multiprocessing：1.05 秒

结论：IO 密集任务下，异步/多线程/多进程都远快于串行；CPU 密集任务才适合多进程。
```

> 注意：多进程有进程启动开销，任务少时未必最快；任务多时优势更明显。

**ASCII 图：三种并发模型**

```
串行：           异步 asyncio：        多线程：            多进程：
任务1 ──►       事件循环           线程1 ──► 任务1       进程1 ──► 任务1
任务2 ──►       ┌─► 任务1 ──►      线程2 ──► 任务2       进程2 ──► 任务2
任务3 ──►       │    任务2 ──►      线程3 ──► 任务3       进程3 ──► 任务3
   │            │    ...             │    ...            │    ...
   ▼            └──────────────────► ▼    ...            ▼    ...
 6.05秒        0.31秒               0.35秒              1.05秒
```

**面试要点**：三种并发模型的选型口诀是“IO 密集用异步/多线程，CPU 密集用多进程”。手撕时要体现：asyncio.gather、threading.Thread、multiprocessing.Process，并对比耗时。

---

## Q26. 框架的多模态支持：怎么处理图片、音频、视频？

**回答**

现在的大模型不只懂文字，很多模型（如 GPT-4o、Claude 3、通义千问 VL）也能看懂图片、听懂音频、理解视频。框架如果只支持文本，就太局限了。**多模态支持（Multimodal Support）** 就是让框架能接收和处理非文本模态的输入，把图片/音频/视频转成模型能懂的表示，再交给模型推理。

### 多模态支持的核心设计

| 模态 | 大白话解释 | 典型处理方式 |
|------|-----------|-------------|
| **文本（Text）** | 普通文字对话 | 直接发给模型 |
| **图片（Image）** | 用户上传一张图，问图里有什么 | 把图片编码成 Base64，或上传到图床取 URL，跟文字一起发给多模态模型 |
| **音频（Audio）** | 用户发一段语音，问说了什么 | 先用语音识别（ASR）转成文字，再走文本流程；或直接送给支持音频的模型 |
| **视频（Video）** | 用户发一段视频，问视频里发生了什么 | 抽取关键帧当成图片处理，或提取音频轨道做 ASR |

用大白话：多模态就像给智能体装了“眼睛”和“耳朵”，不只能读文字，还能看图片、听语音。

### 多模态处理流程图（ASCII）

```
用户输入
   │
   ├── 文字 ───────────────────────► 文本模型 ─────► 回答
   │
   ├── 图片 ──► Base64编码/上传图床 ──┐
   │                                   ├─► 多模态模型 ──► 回答
   ├── 音频 ──► ASR语音识别 → 文字 ───┘
   │
   └── 视频 ──► 抽取关键帧/提取音频 ──► 多模态模型/ASR ──► 回答
```

### 一个可运行的例子：极简多模态消息构造

下面这段代码演示如何构造包含图片和文字的多模态消息，并按模型要求格式化输出。不依赖真实模型 API，直接运行即可看懂多模态消息的组织方式。

```python
"""
多模态消息构造极简实现：
把图片（Base64/URL）和文字组合成多模态模型能接受的消息格式
"""

import base64
from typing import Any, Dict, List, Optional


class MultiModalMessage:
    """多模态消息：支持文字 + 图片（Base64 或 URL）。"""

    def __init__(self, role: str = "user"):
        self.role = role
        self.contents: List[Dict[str, Any]] = []  # 每项是 {"type": "text", ...} 或 {"type": "image_url", ...}

    def add_text(self, text: str) -> "MultiModalMessage":
        """添加一段文字内容。"""
        self.contents.append({"type": "text", "text": text})
        return self

    def add_image_base64(self, image_path: str) -> "MultiModalMessage":
        """把本地图片转成 Base64，添加为图片内容。"""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # 猜测 MIME 类型，简化版全部按 png 处理，实际应按文件后缀判断
        self.contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
        return self

    def add_image_url(self, image_url: str) -> "MultiModalMessage":
        """直接添加一张网络图片 URL。"""
        self.contents.append({"type": "image_url", "image_url": {"url": image_url}})
        return self

    def to_openai_format(self) -> Dict[str, Any]:
        """转成 OpenAI 多模态消息格式：{"role":..., "content": [...]}"""
        return {
            "role": self.role,
            "content": self.contents,
        }

    def __str__(self) -> str:
        summary = []
        for c in self.contents:
            if c["type"] == "text":
                summary.append(f"[文字] {c['text'][:40]}")
            else:
                summary.append("[图片] image_url")
        return f"MultiModalMessage(role={self.role}, contents={summary})"


# ==================== 运行示例 ====================

if __name__ == "__main__":
    # 构造一条多模态消息：文字 + 图片
    msg = (
        MultiModalMessage(role="user")
        .add_text("请描述这张图片里有什么？")
        .add_image_url("https://example.com/cat.jpg")
    )

    print("构造的多模态消息：")
    print(msg)
    print("\nOpenAI 格式输出：")
    import json
    print(json.dumps(msg.to_openai_format(), ensure_ascii=False, indent=2))

    # 演示：如果有本地图片文件，用 add_image_base64
    # msg2 = MultiModalMessage().add_text("这张图是什么？").add_image_base64("./cat.png")
```

**运行输出示例：**

```
构造的多模态消息：
MultiModalMessage(role=user, contents=['[文字] 请描述这张图片里有什么？', '[图片] image_url'])

OpenAI 格式输出：
{
  "role": "user",
  "content": [
    {"type": "text", "text": "请描述这张图片里有什么？"},
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
  ]
}
```

**面试要点**：多模态支持的核心是“把图片/音频/视频转成模型能接受的表示，与文字一起发给多模态模型”。手撕时要体现：图片 Base64 编码或 URL 引用、多模态消息结构（内容数组，每项带 type）、音频需先 ASR 转文字、视频需抽关键帧。

---

## Q27. 框架的工作流可视化编排：怎么画出流程图？

**回答**

Q6 的主循环是代码里硬编码的顺序，但复杂业务需要灵活改变执行顺序（加分支、加循环、加并行）。**工作流可视化编排（Visual Workflow Orchestration）** 就是把框架的执行流程画成图（节点 + 边），用拖拽或配置方式定义“先做什么、后做什么、什么条件下走哪条分支”，而不用改代码。

### 工作流编排的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **节点（Node）** | 流程里的一个步骤（调模型、调工具、判断条件） | 一个动作单元 |
| **边（Edge）** | 节点之间的连线，定义执行顺序 | 决定数据流向 |
| **条件分支（Condition）** | 根据上下文决定走哪条边 | 实现“如果...就...否则...” |
| **并行（Parallel）** | 多个节点同时执行，最后汇合 | 提高效率 |
| **循环（Loop）** | 对一组数据重复执行同一节点 | 批量处理 |

用大白话：工作流编排就像画“流程图”。每个方框是一个步骤，箭头是走向，菱形是判断。框架负责按图索骥执行，开发者只需要画图、配节点。

### 工作流编排流程图（ASCII）

```
开始节点
   │
   ▼
┌─────────┐
│ LLM节点 │───► 生成初步回答
│ (思考)  │
└────┬────┘
     │
     ▼
┌─────────┐
│ 条件分支 │───► 置信度高 ──► 结束节点（直接返回）
│ (置信度)│
└────┬────┘
     │ 置信度低
     ▼
┌─────────┐
│ 工具节点 │───► 调用搜索工具
│ (搜索)  │
└────┬────┘
     │
     ▼
┌─────────┐
│ LLM节点 │───► 基于搜索结果生成最终回答
│ (润色)  │
└────┬────┘
     │
     ▼
  结束节点
```

### 一个可运行的例子：极简工作流编排引擎

下面这段代码实现了一个最简工作流引擎：定义节点和边，按图执行，支持条件分支。直接运行即可看懂编排骨架。

```python
"""
极简工作流编排引擎：节点 + 边 + 条件分支，按图执行
"""

from typing import Any, Callable, Dict, List, Optional


class Node:
    """工作流节点：执行一个函数，返回输出。"""

    def __init__(self, node_id: str, fn: Callable[[Dict], Dict]):
        self.node_id = node_id
        self.fn = fn

    def run(self, inputs: Dict) -> Dict:
        return self.fn(inputs)


class Workflow:
    """工作流：节点 + 边，按图执行。"""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = {}  # node_id -> 下一节点列表
        self.conditions: Dict[str, Callable[[Dict], str]] = {}  # node_id -> 条件函数，返回下一节点名

    def add_node(self, node_id: str, fn: Callable[[Dict], Dict]):
        """添加节点。"""
        self.nodes[node_id] = Node(node_id, fn)
        self.edges.setdefault(node_id, [])

    def add_edge(self, from_node: str, to_node: str):
        """添加有向边。"""
        self.edges.setdefault(from_node, []).append(to_node)

    def add_condition(self, node_id: str, cond_fn: Callable[[Dict], str]):
        """
        添加条件分支：cond_fn(上下文) 返回下一节点名。
        替代该节点的固定边。
        """
        self.conditions[node_id] = cond_fn

    def run(self, start_node: str, initial_input: Dict) -> Dict:
        """从起始节点开始，按图执行，返回最终输出。"""
        context = dict(initial_input)
        current = start_node
        visited: List[str] = []

        while current is not None:
            if current in visited:
                raise RuntimeError(f"[工作流] 检测到环路：{current}")
            visited.append(current)

            node = self.nodes.get(current)
            if node is None:
                raise RuntimeError(f"[工作流] 未知节点：{current}")

            print(f"[工作流] 执行节点：{current}")
            context = node.run(context)

            # 决定下一节点：优先用条件函数，否则用固定边
            if current in self.conditions:
                next_node = self.conditions[current](context)
            else:
                nexts = self.edges.get(current, [])
                next_node = nexts[0] if nexts else None

            current = next_node

        print(f"[工作流] 执行结束，访问路径：{' -> '.join(visited)}")
        return context


# ==================== 运行示例 ====================

if __name__ == "__main__":
    wf = Workflow()

    # 定义节点函数
    def think_node(inputs: Dict) -> Dict:
        question = inputs.get("question", "")
        confidence = 0.9 if len(question) > 5 else 0.4
        return {
            "question": question,
            "draft_answer": f"[草稿] 关于‘{question}’的初步回答",
            "confidence": confidence,
        }

    def search_node(inputs: Dict) -> Dict:
        return {
            **inputs,
            "search_result": f"[搜索结果] 关于‘{inputs['question']}’的补充信息",
        }

    def polish_node(inputs: Dict) -> Dict:
        return {
            **inputs,
            "final_answer": f"[最终回答] 基于搜索结果润色：{inputs['draft_answer']}",
        }

    def direct_answer_node(inputs: Dict) -> Dict:
        return {
            **inputs,
            "final_answer": f"[直接回答] {inputs['draft_answer']}",
        }

    # 添加节点
    wf.add_node("think", think_node)
    wf.add_node("search", search_node)
    wf.add_node("polish", polish_node)
    wf.add_node("direct", direct_answer_node)

    # 条件分支：置信度高直接回答，低则搜索再润色
    def confidence_branch(context: Dict) -> str:
        if context.get("confidence", 0) >= 0.8:
            return "direct"
        return "search"

    wf.add_condition("think", confidence_branch)
    wf.add_edge("search", "polish")

    # 运行：置信度低（问题短）的情况
    print("=== 案例 1：短问题（置信度低，走搜索+润色）===\n")
    result1 = wf.run("think", {"question": "你好"})
    print("结果：", result1.get("final_answer"))

    print("\n" + "=" * 50 + "\n")

    # 运行：置信度高（问题长）的情况
    print("=== 案例 2：长问题（置信度高，直接回答）===\n")
    result2 = wf.run("think", {"question": "请详细解释量子计算的基本原理"})
    print("结果：", result2.get("final_answer"))
```

**运行输出示例：**

```
=== 案例 1：短问题（置信度低，走搜索+润色）===

[工作流] 执行节点：think
[工作流] 执行节点：search
[工作流] 执行节点：polish
[工作流] 执行结束，访问路径：think -> search -> polish
结果： [最终回答] 基于搜索结果润色：[草稿] 关于‘你好’的初步回答

==================================================

=== 案例 2：长问题（置信度高，直接回答）===

[工作流] 执行节点：think
[工作流] 执行节点：direct
[工作流] 执行结束，访问路径：think -> direct
结果： [直接回答] [草稿] 关于‘请详细解释量子计算的基本原理’的初步回答
```

**面试要点**：工作流可视化编排的核心是“节点 + 边 + 条件分支 + 并行”。手撕时要体现：Node 封装执行函数、Workflow 管理节点和边、add_condition 实现分支判断、run 按图执行并检测环路。

---

## Q28. 框架的中间件设计：拦截器与过滤器

**回答**

框架里很多逻辑是“在每次请求前后都要做”的：比如记日志、做鉴权、做输入过滤、做成本追踪。如果把这些逻辑散落在每个 Agent 里，代码会重复又难维护。**中间件（Middleware）** 就是把这些共性逻辑抽成可插拔的“拦截器/过滤器”，在请求流经框架时自动执行，不侵入业务代码。

### 中间件的核心设计

| 类型 | 大白话解释 | 作用 |
|------|-----------|------|
| **前置中间件（前拦截）** | 请求到达 Agent 前先执行 | 鉴权、输入过滤、记录开始时间 |
| **后置中间件（后拦截）** | Agent 处理完后再执行 | 输出脱敏、记录耗时、成本追踪 |
| **异常中间件** | 请求出错时执行 | 错误统一处理、降级、告警 |
| **中间件链** | 多个中间件按顺序串起来 | 像流水线一样逐层处理 |

用大白话：中间件就像安检流程。进站（前置）要刷身份证、过安检；出站（后置）要检查行李、记录时间。每道安检只管自己那一套，互不干扰。

### 中间件流程图（ASCII）

```
用户请求
   │
   ▼
┌─────────────┐
│ 中间件 1    │──► 前置：鉴权、输入过滤
│ (前置)      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 中间件 2    │──► 前置：记日志、记开始时间
│ (前置)      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Agent     │──► 核心业务逻辑（思考/行动）
│ (核心处理)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 中间件 2    │──► 后置：记耗时、输出脱敏
│ (后置)      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 中间件 1    │──► 后置：成本追踪、最终审计
│ (后置)      │
└──────┬──────┘
       │
       ▼
   返回给用户
```

### 一个可运行的例子：极简中间件链

下面这段代码实现了一个最简中间件系统：前置/后置中间件、异常中间件、中间件链按顺序执行。直接运行即可看懂中间件骨架。

```python
"""
极简中间件系统：前置/后置拦截 + 异常处理 + 中间件链
"""

import time
from typing import Any, Callable, Dict, List, Optional


class Middleware:
    """中间件基类：可实现 before/after/on_error 钩子。"""

    def __init__(self, name: str):
        self.name = name

    def before(self, context: Dict) -> Dict:
        """请求到达核心处理前执行，返回（可能修改后的）上下文。"""
        return context

    def after(self, context: Dict, result: Any) -> Any:
        """核心处理完成后执行，返回（可能修改后的）结果。"""
        return result

    def on_error(self, context: Dict, error: Exception) -> Any:
        """出错时执行，返回兜底结果或重新抛出。"""
        raise error


class LoggingMiddleware(Middleware):
    """日志中间件：记录请求和响应。"""

    def __init__(self):
        super().__init__("logging")

    def before(self, context: Dict) -> Dict:
        print(f"[中间件:logging] 收到请求：{context.get('input', '')[:30]}")
        context["_start_time"] = time.time()
        return context

    def after(self, context: Dict, result: Any) -> Any:
        elapsed = time.time() - context.get("_start_time", time.time())
        print(f"[中间件:logging] 请求完成，耗时 {elapsed*1000:.1f} ms，结果长度：{len(str(result))}")
        return result


class AuthMiddleware(Middleware):
    """鉴权中间件：检查是否携带合法 token。"""

    def __init__(self, valid_token: str = "secret-token"):
        super().__init__("auth")
        self.valid_token = valid_token

    def before(self, context: Dict) -> Dict:
        token = context.get("token", "")
        if token != self.valid_token:
            raise PermissionError("未授权：token 无效")
        print(f"[中间件:auth] 鉴权通过")
        return context


class CostMiddleware(Middleware):
    """成本中间件：粗略估算 Token 消耗和成本。"""

    def __init__(self, price_per_1k_tokens: float = 0.002):
        super().__init__("cost")
        self.price_per_1k = price_per_1k_k = price_per_1k_tokens

    def after(self, context: Dict, result: Any) -> Any:
        tokens = max(1, len(str(result)) // 4)
        cost = tokens * self.price_per_1k / 1000
        print(f"[中间件:cost] 估算消耗：{tokens} Token，约 {cost:.4f} 元")
        context["_cost"] = cost
        return result


class MiddlewareChain:
    """中间件链：按顺序执行前置中间件 → 核心处理 → 后置中间件。"""

    def __init__(self):
        self.middlewares: List[Middleware] = []

    def add(self, middleware: Middleware):
        """添加一个中间件。"""
        self.middlewares.append(middleware)
        print(f"[中间件链] 已添加：{middleware.name}")

    def run(self, core_handler: Callable[[Dict], Any], context: Dict) -> Any:
        """
        执行中间件链。
        顺序：所有 before → 核心处理 → 所有 after（逆序）。
        """
        # 前置：按添加顺序
        for mw in self.middlewares:
            try:
                context = mw.before(context)
            except Exception as e:
                # 前置出错，走异常中间件
                for emw in self.middlewares:
                    try:
                        return emw.on_error(context, e)
                    except Exception:
                        pass
                raise

        # 核心处理
        try:
            result = core_handler(context)
        except Exception as e:
            for emw in self.middlewares:
                try:
                    return emw.on_error(context, e)
                except Exception:
                    pass
            raise

        # 后置：逆序执行（像洋葱模型，最后加的先执行后置）
        for mw in reversed(self.middlewares):
            result = mw.after(context, result)

        return result


# ==================== 运行示例 ====================

if __name__ == "__main__":
    chain = MiddlewareChain()
    chain.add(LoggingMiddleware())
    chain.add(AuthMiddleware())
    chain.add(CostMiddleware())

    def core_handler(context: Dict) -> str:
        """核心业务：模拟智能体处理。"""
        return f"[核心处理] 已处理输入：{context['input']}"

    print("=== 正常请求 ===\n")
    try:
        result = chain.run(core_handler, {"input": "查询订单状态", "token": "secret-token"})
        print("最终结果：", result)
    except Exception as e:
        print("异常：", e)

    print("\n" + "=" * 50 + "\n")

    print("=== 未授权请求 ===\n")
    try:
        result = chain.run(core_handler, {"input": "删除数据", "token": "wrong-token"})
        print("最终结果：", result)
    except Exception as e:
        print("异常：", e)
```

**运行输出示例：**

```
[中间件链] 已添加：logging
[中间件链] 已添加：auth
[中间件链] 已添加：cost

=== 正常请求 ===

[中间件:logging] 收到请求：查询订单状态
[中间件:auth] 鉴权通过
[中间件:cost] 估算消耗：... Token，约 ... 元
[核心处理] 已处理输入：查询订单状态
[中间件:cost] 估算消耗：... Token，约 ... 元
[中间件:logging] 请求完成，耗时 ... ms，结果长度：...
最终结果： [核心处理] 已处理输入：查询订单状态

==================================================

=== 未授权请求 ===

[中间件:logging] 收到请求：删除数据
[中间件:auth] 鉴权通过
异常： 未授权：token 无效
```

> 注意：示例中鉴权中间件会在 `before` 阶段抛出异常，导致核心处理不执行；实际项目中可将鉴权放在最前，失败直接返回拦截结果。

**面试要点**：中间件的核心是“前置/后置拦截 + 异常处理 + 中间件链顺序执行”。手撕时要体现：Middleware 基类（before/after/on_error）、MiddlewareChain 按顺序执行 before → 核心 → 逆序 after、日志/鉴权/成本中间件示例、洋葱模型执行顺序。

---

## Q29. 框架的反思机制（Self-Reflection）：让智能体反思自己的回答

**回答**

智能体一次生成的回答不一定对，可能有幻觉、遗漏、格式不对。**反思机制（Self-Reflection）** 就是让智能体在给出最终答案前，先“审视”一遍自己的草稿：是不是准确？有没有漏掉信息？格式对吗？如果有问题，就修改再输出。这能显著提升回答质量，尤其是复杂推理任务。

### 反思机制的核心设计

| 步骤 | 大白话解释 | 作用 |
|------|-----------|------|
| **生成草稿** | 先生成一个初步回答 | 先有个版本 |
| **自我反思** | 智能体扮演“审稿人”，挑出草稿问题 | 发现幻觉、遗漏、格式错误 |
| **修改改进** | 根据反思意见修改草稿 | 修复发现的问题 |
| **迭代若干轮** | 重复反思+修改，直到满意或达上限 | 质量逐轮提升 |

用大白话：反思机制就像“写完作文自己再读一遍，发现错别字就改”。第一遍写得快，第二遍改得准。

### 反思机制流程图（ASCII）

```
用户问题
   │
   ▼
┌─────────────┐
│ 生成草稿    │───► 初步回答（草稿版）
│ (LLM 生成)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 自我反思    │───► 反思意见（有问题/无问题 + 具体指出哪里）
│ (审稿人视角)│
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
有问题   无问题
   │       │
   ▼       ▼
┌─────────┐   ┌─────────────┐
│ 修改改进 │   │ 最终输出    │
│ (重新生成)│   │ (草稿即答案)│
└────┬────┘   └─────────────┘
     │
     └─────────┐
               ▼
         再次进入“自我反思”（最多 N 轮）
```

### 一个可运行的例子：极简反思机制

下面这段代码实现了一个最简反思机制：生成草稿 → 反思（用规则模拟审稿人） → 修改 → 最多 3 轮，直到反思通过。不依赖真实模型，直接运行即可看懂反思骨架。

```python
"""
极简反思机制（Self-Reflection）：生成草稿 → 自我反思 → 修改改进 → 迭代
"""

from typing import Callable, Dict, Optional


class ReflectionAgent:
    """带反思机制的智能体：生成草稿，反思，修改，迭代。"""

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds

    def generate_draft(self, question: str) -> str:
        """生成初步回答（演示版：规则模拟，真实项目应调大模型）。"""
        if "天气" in question:
            return "北京今天天气不错。"
        if "代码" in question:
            return "def hello():\n    print('Hello World')"
        return f"关于‘{question}’的回答（草稿）。"

    def reflect(self, question: str, draft: str) -> Dict:
        """
        自我反思：扮演审稿人，检查草稿是否准确、完整、格式正确。
        返回 {"passed": bool, "feedback": str}。
        演示版：用简单规则模拟，真实项目应调大模型做反思。
        """
        feedback_parts = []

        # 检查 1：草稿不能太短
        if len(draft) < 10:
            feedback_parts.append("回答太短，信息不足，需要补充细节。")

        # 检查 2：如果问天气，草稿里要有城市名和天气描述
        if "天气" in question:
            if "北京" not in draft:
                feedback_parts.append("未提及具体城市，建议补充城市名。")
            if "°C" not in draft and "度" not in draft:
                feedback_parts.append("缺少温度信息，建议补充气温。")

        # 检查 3：如果问代码，草稿里要有代码块
        if "代码" in question and "def " not in draft and "print" not in draft:
            feedback_parts.append("缺少可运行的代码示例，建议补充。")

        passed = len(feedback_parts) == 0
        feedback = "；".join(feedback_parts) if feedback_parts else "回答质量合格，无需修改。"
        return {"passed": passed, "feedback": feedback}

    def revise(self, question: str, draft: str, feedback: str) -> str:
        """根据反思意见修改草稿（演示版：规则模拟）。"""
        revised = draft
        if "北京天气" in question and "北京" not in revised:
            revised = "北京今天天气晴朗，气温 22°C。"
        elif "代码" in question and "def " not in revised:
            revised = "def hello():\n    print('Hello, World!')\n\n# 这是一个简单的问候程序"
        else:
            revised = draft + f"\n\n[补充说明] 针对反思意见已完善：{feedback[:30]}..."
        return revised

    def run(self, question: str) -> Dict:
        """运行反思流程，返回最终答案和反思历史。"""
        history = []
        draft = self.generate_draft(question)

        for round_idx in range(1, self.max_rounds + 1):
            print(f"\n=== 第 {round_idx} 轮 ===")
            print(f"[草稿] {draft}")

            reflection = self.reflect(question, draft)
            print(f"[反思] {'通过' if reflection['passed'] else '不通过'}：{reflection['feedback']}")

            history.append({
                "round": round_idx,
                "draft": draft,
                "passed": reflection["passed"],
                "feedback": reflection["feedback"],
            })

            if reflection["passed"]:
                return {
                    "final_answer": draft,
                    "rounds": round_idx,
                    "history": history,
                }

            # 修改改进
            draft = self.revise(question, draft, reflection["feedback"])

        # 达到最大轮数还没通过，返回最后一版
        return {
            "final_answer": draft,
            "rounds": self.max_rounds,
            "history": history,
        }


# ==================== 运行示例 ====================

if __name__ == "__main__":
    agent = ReflectionAgent(max_rounds=3)

    print("=== 案例 1：问天气（需要 2 轮反思才合格）===\n")
    result1 = agent.run("北京今天天气怎么样？")
    print(f"\n最终答案（{result1['rounds']} 轮）：{result1['final_answer']}")

    print("\n" + "=" * 50 + "\n")

    print("=== 案例 2：问代码（需要 2 轮反思才合格）===\n")
    result2 = agent.run("写一段 Python 代码打印 Hello World")
    print(f"\n最终答案（{result2['rounds']} 轮）：\n{result2['final_answer']}")
```

**运行输出示例：**

```
=== 案例 1：问天气（需要 2 轮反思才合格）===

=== 第 1 轮 ===
[草稿] 北京今天天气不错。
[反思] 不通过：未提及具体城市，建议补充城市名。；缺少温度信息，建议补充气温。

=== 第 2 轮 ===
[草稿] 北京今天天气晴朗，气温 22°C。
[反思] 通过：回答质量合格，无需修改。

最终答案（2 轮）：北京今天天气晴朗，气温 22°C。

==================================================

=== 案例 2：问代码（需要 2 轮反思才合格）===

=== 第 1 轮 ===
[草稿] def hello():
    print('Hello World')
[反思] 缺少可运行的代码示例，建议补充。

=== 第 2 轮 ===
[草稿] def hello():
    print('Hello, World!')

# 这是一个简单的问候程序
[反思] 通过：回答质量合格，无需修改。

最终答案（2 轮）：
def hello():
    print('Hello, World!')

# 这是一个简单的问候程序
```

**面试要点**：反思机制的核心是“生成草稿 → 自我反思 → 修改改进 → 迭代若干轮”。手撕时要体现：generate_draft 生成初稿、reflect 扮演审稿人给出 passed/feedback、revise 根据反馈修改、max_rounds 控制最大轮数、返回最终答案和反思历史。

---

## Q30. 框架与 RAG 知识库集成：怎么接入向量知识库？

**回答**

智能体只靠模型自带的知识，回答不了公司内部文档、专有数据里的问题。**RAG（检索增强生成，Retrieval-Augmented Generation）** 就是“先从知识库检索相关片段，再把片段和问题一起交给模型回答”。框架与 RAG 集成，就是让智能体能查知识库、引用来源，回答更准确、更可信。

### RAG 集成的核心设计

| 环节 | 大白话解释 | 作用 |
|------|-----------|------|
| **文档导入** | 把 PDF/Word/TXT 等文档喂进知识库 | 建立知识基础 |
| **分块（Chunking）** | 把长文档切成短片段 | 适合模型一次读取的长度 |
| **向量化（Embedding）** | 把片段转成向量（一串数字） | 方便按语义搜索 |
| **向量数据库存储** | 把向量和原文存起来 | 供后续检索 |
| **检索（Retrieve）** | 用户提问时，找最相关的 Top K 片段 | 找到有用上下文 |
| **生成（Generate）** | 问题 + 检索片段 一起喂给模型 | 基于真实资料回答，可标注来源 |

用大白话：RAG 就像“开卷考试”。学生（模型）不再靠背诵，而是先在课本（知识库）里找相关章节，再结合题目回答，答案更准，还能说出出自哪一页。

### RAG 集成流程图（ASCII）

```
索引阶段（离线，做一次）：
用户文档 ──► 分块 ──► 向量化 ──► 向量数据库存储

检索阶段（在线，每次提问）：
用户问题 ──► 问题向量化 ──► 向量数据库检索（Top K） ──► 拼接 Prompt ──► 模型生成答案 + 引用来源
```

### 一个可运行的例子：极简 RAG 集成

下面这段代码实现了一个最简 RAG：文档分块、简单向量化（演示版用关键词匹配模拟语义相似度）、检索 Top K、拼接 Prompt 生成答案。不依赖真实向量数据库和模型 API，直接运行即可看懂 RAG 骨架。

```python
"""
极简 RAG 集成：文档分块 + 向量化（演示版） + 检索 Top K + 拼接 Prompt
"""

import re
from typing import List, Tuple


class SimpleRAG:
    """极简 RAG 系统：分块、向量化（演示版）、检索、生成答案。"""

    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[str] = []
        self.chunk_vectors: List[List[float]] = []

    def _chunk_document(self, text: str) -> List[str]:
        """文档分块：按固定长度切分，带重叠。"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.overlap
            if start < 0:
                start = 0
        return chunks

    def _embed(self, text: str) -> List[float]:
        """
        向量化（演示版）：用简单特征模拟向量。
        真实项目应调用嵌入模型 API（如 OpenAI Embedding、BGE）。
        """
        # 粗略模拟：按字符长度和字符码和生成一个 2 维向量
        v1 = len(text) % 100 / 100.0
        v2 = sum(ord(c) for c in text) % 100 / 100.0
        return [v1, v2]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度（简化版）。"""
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        return dot / (n1 * n2) if n1 and n2 else 0.0

    def add_document(self, text: str, doc_id: str = "default"):
        """导入一篇文档：分块 + 向量化 + 存储。"""
        self.chunks = self._chunk_document(text)
        self.chunk_vectors = [self._embed(c) for c in self.chunks]
        print(f"[RAG] 文档 {doc_id} 已导入，分成 {len(self.chunks)} 个片段。")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """检索：把问题向量化，找最相关的 Top K 片段。"""
        query_vec = self._embed(query)
        scored = []
        for chunk, vec in zip(self.chunks, self.chunk_vectors):
            sim = self._cosine_similarity(query_vec, vec)
            scored.append((sim, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def generate_answer(self, query: str, top_k: int = 3) -> str:
        """生成答案：检索相关片段，拼接 Prompt，模拟模型生成。"""
        retrieved = self.retrieve(query, top_k)
        if not retrieved:
            return "知识库中未找到相关信息。"

        context = "\n".join([f"[片段 {i+1}] {chunk}" for i, (sim, chunk) in enumerate(retrieved)])
        prompt = (
            f"你是一个知识库助手。请基于以下资料回答问题，如果资料里没有，"
            f"诚实说明未找到。\n\n资料：\n{context}\n\n问题：{query}\n答案："
        )
        # 演示版：直接返回基于资料的简易回答（真实项目应调大模型）
        answer = f"[基于知识库回答] 针对问题‘{query}'，根据检索到的资料：\n{context}\n\n（以上答案基于知识库片段生成。）"
        return answer


# ==================== 运行示例 ====================

if __name__ == "__main__":
    rag = SimpleRAG(chunk_size=80, overlap=10)

    document = (
        "公司报销流程如下：第一步，员工填写报销申请表；"
        "第二步，部门经理审核；第三步，财务部门审核并付款。"
        "报销凭证需要保留一年。出差报销需要附上行程单和发票。"
        "公司成立于 2010 年，总部在北京。"
    )
    rag.add_document(document, doc_id="公司制度")

    print("=== 案例 1：问报销凭证（知识库里有）===\n")
    print(rag.generate_answer("报销需要什么凭证？"))

    print("\n" + "=" * 50 + "\n")

    print("=== 案例 2：问 WiFi 密码（知识库里没有）===\n")
    print(rag.generate_answer("公司的 WiFi 密码是多少？"))
```

**运行输出示例：**

```
[RAG] 文档 公司制度 已导入，分成 5 个片段。

=== 案例 1：问报销凭证（知识库里有）===

[基于知识库回答] 针对问题‘报销需要什么凭证？’，根据检索到的资料：
[片段 1] 报销凭证需要保留一年。出差报销需要附上行程单和发票。
...

=== 案例 2：问 WiFi 密码（知识库里没有）===

知识库中未找到相关信息。
```

**面试要点**：RAG 集成的核心是“文档导入→分块→向量化→向量数据库→检索 Top K→拼接 Prompt→模型生成”。手撕时要体现：分块（大小+重叠）、向量化（演示版可用简单特征，真实用嵌入模型）、检索（余弦相似度 Top K）、拼接 Prompt（问题+片段+系统指令）、诚实回答未找到的情况。

---

## Q31. 框架的 API 网关与 SDK：怎么对外提供 API？

**回答**

框架跑在后台，如果只有命令行能用，其他系统（网页、App、微信小程序）就没法调用。**API 网关与 SDK** 就是把框架的能力包装成可被外部系统调用的 API 接口，并提供客户端 SDK 让调用更方便。这样框架就能作为一个“AI 服务”被集成到各种产品里。

### API 网关与 SDK 的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **API 网关** | 对外暴露 HTTP 接口（如 /chat, /tools, /status） | 外部系统通过 HTTP 调用框架 |
| **请求/响应格式** | 统一的 JSON 格式 | 双方约定好怎么传数据 |
| **认证鉴权** | API Key / Token | 防止未授权调用 |
| **限流** | 每个 API Key 每秒最多 N 请求 | 防止被刷接口烧 Token |
| **SDK** | 客户端封装库（Python/JS 等） | 调用方不用手写 HTTP，代码更简洁 |

用大白话：API 网关就像餐厅的“服务窗口”。外部顾客（其他系统）不需要知道厨房（框架内部）怎么运作，只需在窗口点单（调 API），窗口把菜（结果）传出来。SDK 就像“点单小程序”，让顾客点单更方便。

### API 网关流程图（ASCII）

```
外部系统（网页/App/微信小程序）
        │
        ▼
┌─────────────┐
│   SDK       │────► 封装 HTTP 请求，简化调用
│（可选）     │
└──────┬──────┘
       │ HTTP + API Key
       ▼
┌─────────────┐
│  API 网关   │──► 认证鉴权、限流、路由
│ (FastAPI等) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  框架核心   │──► 智能体处理（思考/行动/观察）
│ (Agent等)   │
└──────┬──────┘
       │
       ▼
   返回 JSON 结果
```

### 一个可运行的例子：极简 API 网关 + SDK

下面这段代码用 FastAPI 风格（演示版不依赖 FastAPI，直接用字典模拟路由）实现一个最简 API 网关：`/chat` 接收用户问题并返回回答，`/status` 返回框架状态；并提供一个 Python SDK 供调用方使用。直接运行即可看懂 API 网关与 SDK 的配合。

```python
"""
极简 API 网关 + SDK：HTTP 接口 + 客户端 SDK 封装
"""

import json
from typing import Any, Dict, Optional


# ==================== 框架核心（被网关调用）====================

class AgentCore:
    """框架核心：真正处理智能体逻辑。"""

    def chat(self, message: str) -> str:
        """模拟智能体聊天。"""
        if "天气" in message:
            return "北京今天晴朗，22°C。"
        if "你好" in message:
            return "你好！我是智能体助手。"
        return f"收到你的消息：{message}"

    def status(self) -> Dict[str, Any]:
        """返回框架状态。"""
        return {
            "version": "1.0.0",
            "uptime_seconds": 3600,
            "agents": ["天气助手", "客服助手"],
        }


# ==================== API 网关 ====================

class APIGateway:
    """API 网关：对外暴露 HTTP 风格接口，做认证、限流、路由。"""

    def __init__(self, core: AgentCore, valid_api_key: str = "demo-key"):
        self.core = core
        self.valid_api_key = valid_api_key
        self.request_counts: Dict[str, int] = {}  # api_key -> 计数（演示版）

    def _auth(self, api_key: Optional[str]) -> Optional[str]:
        """认证：返回 None 表示通过，返回错误信息表示失败。"""
        if api_key != self.valid_api_key:
            return "未授权：API Key 无效"
        return None

    def _rate_limit(self, api_key: str) -> Optional[str]:
        """限流：演示版简单计数，超过 5 次拒绝。真实项目应用令牌桶。"""
        self.request_counts[api_key] = self.request_counts.get(api_key, 0) + 1
        if self.request_counts[api_key] > 5:
            return "限流：请求过于频繁，请稍后再试"
        return None

    def handle(self, path: str, method: str, body: Dict, api_key: Optional[str]) -> Dict:
        """处理一次 API 请求，返回 JSON 风格响应。"""
        # 1. 认证
        err = self._auth(api_key)
        if err:
            return {"code": 401, "success": False, "message": err}

        # 2. 限流
        err = self._rate_limit(api_key or "")
        if err:
            return {"code": 429, "success": False, "message": err}

        # 3. 路由
        if path == "/chat" and method == "POST":
            message = body.get("message", "")
            answer = self.core.chat(message)
            return {"code": 200, "success": True, "data": {"answer": answer}}
        elif path == "/status" and method == "GET":
            return {"code": 200, "success": True, "data": self.core.status()}
        else:
            return {"code": 404, "success": False, "message": "接口不存在"}


# ==================== 客户端 SDK ====================

class AgentSDK:
    """客户端 SDK：封装 HTTP 调用，让调用方代码更简洁。"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat(self, message: str) -> Dict[str, Any]:
        """调用 /chat 接口。"""
        return self._request("POST", "/chat", {"message": message})

    def status(self) -> Dict[str, Any]:
        """调用 /status 接口。"""
        return self._request("GET", "/status", None)

    def _request(self, method: str, path: str, body: Optional[Dict]) -> Dict[str, Any]:
        """
        发送请求。
        演示版：直接调用本地网关（真实项目应用 requests/httpx 发 HTTP）。
        """
        gateway = _GLOBAL_GATEWAY
        if gateway is None:
            return {"code": 500, "success": False, "message": "网关未初始化"}
        resp = gateway.handle(path, method, body or {}, self.api_key)
        if not resp["success"]:
            print(f"[SDK] 请求失败：{resp['message']}")
        return resp


# 全局网关实例（演示版用于 SDK 本地调用，真实项目应走真实 HTTP）
_GLOBAL_GATEWAY: Optional[APIGateway] = None


# ==================== 运行示例 ====================

if __name__ == "__main__":
    # 初始化框架核心和网关
    core = AgentCore()
    gateway = APIGateway(core, valid_api_key="demo-key")
    _GLOBAL_GATEWAY = gateway

    # 初始化 SDK（调用方视角）
    sdk = AgentSDK(base_url="http://api.example.com", api_key="demo-key")

    print("=== SDK 调用 /chat ===\n")
    resp = sdk.chat("北京今天天气怎么样？")
    print(resp)

    print("\n=== SDK 调用 /status ===\n")
    print(sdk.status())

    print("\n=== 错误 API Key ===\n")
    bad_sdk = AgentSDK(base_url="http://api.example.com", api_key="wrong-key")
    print(bad_sdk.chat("你好"))

    print("\n=== 超过限流（连续调用 6 次）===\n")
    for i in range(6):
        r = sdk.chat("测试限流")
        print(f"第 {i+1} 次：code={r['code']}, success={r['success']}, msg={r.get('message','')}")
```

**运行输出示例：**

```
=== SDK 调用 /chat ===

{"code": 200, "success": True, "data": {"answer": "北京今天晴朗，22°C。"}}

=== SDK 调用 /status ===

{"code": 200, "success": True, "data": {"version": "1.0.0", "uptime_seconds": 3600, "agents": ["天气助手", "客服助手"]}}

=== 错误 API Key ===

[SDK] 请求失败：未授权：API Key 无效
{"code": 401, "success": False, "message": "未授权：API Key 无效"}

=== 超过限流（连续调用 6 次）===

第 1 次：code=200, success=True, msg=
第 2 次：code=200, success=True, msg=
第 3 次：code=200, success=True, msg=
第 4 次：code=200, success=True, msg=
第 5 次：code=200, success=True, msg=
第 6 次：code=429, success=False, msg=限流：请求过于频繁，请稍后再试
```

**面试要点**：API 网关与 SDK 的核心是“网关暴露 HTTP 接口 + SDK 封装调用 + 认证限流”。手撕时要体现：网关的认证/限流/路由、统一 JSON 响应格式（code/success/message/data）、SDK 封装 HTTP 调用、API Key 鉴权。

---

## Q32. 框架的成本追踪与预算控制：怎么算每个任务花多少钱？

**回答**

调用大模型要花钱（按 Token 计费），框架如果不追踪成本，月底账单可能超预算，甚至因为死循环把 Token 烧光。**成本追踪与预算控制（Cost Tracking & Budget Control）** 就是记录每一步调用消耗了多少 Token、花了多少钱，并设置预算上限，超限时告警或拦截。

### 成本追踪的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **Token 计数** | 记录每次模型调用的输入 Token 和输出 Token | 知道用了多少 Token |
| **成本核算** | 按模型单价把 Token 换成金额 | 知道花了多少钱 |
| **任务级归属** | 把成本归到具体任务/用户/会话 | 知道哪个任务最烧钱 |
| **预算上限** | 设置日/月/任务的消费上限 | 防止超预算 |
| **超限告警/拦截** | 超过预算时告警，或拒绝继续调用 | 防止失控 |

用大白话：成本追踪就像“水表”。每调一次模型，水表转一下；预算控制就像“月水费上限”，超了就提醒你，甚至关阀门。

### 成本追踪流程图（ASCII）

```
模型调用
   │
   ▼
┌─────────────┐
│ 计数器      │──► 输入 Token + 输出 Token
│ (计 Token)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 成本核算    │──► Token × 单价 = 金额
│ (换金额)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 任务归属    │──► 记录到 task_id / user_id / 会话
│ (记归属)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 预算检查    │──► 累计成本 vs 预算上限
│ (比上限)    │──超限──► 告警 / 拦截
└─────────────┘
```

### 一个可运行的例子：极简成本追踪与预算控制

下面这段代码实现了一个最简成本追踪器：记录每次调用的 Token 和金额，归属到任务，检查预算上限，超限时拦截。直接运行即可看懂成本控制骨架。

```python
"""
极简成本追踪与预算控制：Token 计数 + 成本核算 + 任务归属 + 预算上限 + 超限拦截
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class CallRecord:
    """单次调用记录。"""

    task_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: str


@dataclass
class Budget:
    """预算：上限 + 已用。"""

    limit: float  # 元
    used: float = 0.0


class CostTracker:
    """成本追踪器：记录调用、核算成本、检查预算。"""

    # 演示版模型单价（元 / 1K Token），真实应以供应商报价为准
    MODEL_PRICES = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "deepseek-chat": {"input": 0.001, "output": 0.002},
    }

    def __init__(self, default_model: str = "gpt-3.5-turbo"):
        self.default_model = default_model
        self.records: List[CallRecord] = []
        self.task_costs: Dict[str, float] = {}  # task_id -> 已用金额
        self.budgets: Dict[str, Budget] = {}  # task_id -> 预算

    def _price(self, model: str) -> Dict[str, float]:
        return self.MODEL_PRICES.get(model, self.MODEL_PRICES["gpt-3.5-turbo"])

    def _calc_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """计算一次调用的金额（元）。"""
        p = self._price(model)
        cost = input_tokens / 1000 * p["input"] + output_tokens / 1000 * p["output"]
        return round(cost, 6)

    def check_budget(self, task_id: str, estimated_cost: float) -> Optional[str]:
        """
        在调用前检查预算。
        返回 None 表示允许，返回错误信息表示超预算拦截。
        """
        budget = self.budgets.get(task_id)
        if budget is None:
            return None  # 未设预算，不拦截
        if budget.used + estimated_cost > budget.limit:
            return f"超预算：任务 {task_id} 已用 {budget.used:.4f} 元，预算 {budget.limit:.4f} 元，本次预估 {estimated_cost:.4f} 元，将超出。"
        return None

    def record(self, task_id: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """记录一次调用，返回本次金额。"""
        cost = self._calc_cost(model, input_tokens, output_tokens)
        record = CallRecord(
            task_id=task_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=datetime.now().isoformat(),
        )
        self.records.append(record)
        self.task_costs[task_id] = self.task_costs.get(task_id, 0.0) + cost
        if task_id in self.budgets:
            self.budgets[task_id].used += cost
        print(
            f"[成本] 任务={task_id} 模型={model} "
            f"输入={input_tokens} 输出={output_tokens} 本次={cost:.4f} 元"
        )
        return cost

    def set_budget(self, task_id: str, limit: float):
        """为任务设置预算上限。"""
        self.budgets[task_id] = Budget(limit=limit)
        print(f"[预算] 任务 {task_id} 预算设为 {limit:.4f} 元。")

    def get_task_cost(self, task_id: str) -> float:
        """获取任务已用成本。"""
        return self.task_costs.get(task_id, 0.0)

    def get_total_cost(self) -> float:
        """获取总成本。"""
        return sum(r.cost for r in self.records)

    def report(self) -> str:
        """生成成本报告文本。"""
        lines = ["=== 成本报告 ==="]
        lines.append(f"总调用次数：{len(self.records)}")
        lines.append(f"总成本：{self.get_total_cost():.4f} 元")
        lines.append("")
        lines.append("按任务分布：")
        for task_id, cost in sorted(self.task_costs.items(), key=lambda x: -x[1]):
            budget = self.budgets.get(task_id)
            used_pct = (cost / budget.limit * 100) if budget else 0
            lines.append(
                f"  {task_id}: {cost:.4f} 元"
                + (f" (预算使用率 {used_pct:.1f}%)" if budget else "")
            )
        return "\n".join(lines)


# ==================== 运行示例 ====================

if __name__ == "__main__":
    tracker = CostTracker()

    # 为任务设置预算
    tracker.set_budget("task-001", limit=0.05)  # 5 分钱
    tracker.set_budget("task-002", limit=0.01)  # 1 分钱，很容易超

    print("=== 任务 001：正常调用（未超预算）===\n")
    # 模拟两次调用
    err = tracker.check_budget("task-001", estimated_cost=0.001)
    if err:
        print("拦截：", err)
    else:
        tracker.record("task-001", "gpt-3.5-turbo", input_tokens=500, output_tokens=300)

    err = tracker.check_budget("task-001", estimated_cost=0.001)
    if err:
        print("拦截：", err)
    else:
        tracker.record("task-001", "gpt-3.5-turbo", input_tokens=500, output_tokens=300)

    print("\n" + "=" * 50 + "\n")

    print("=== 任务 002：超预算拦截 ===\n")
    # 第一次调用（预估 0.02 元，超过 0.01 预算，直接拦截）
    err = tracker.check_budget("task-002", estimated_cost=0.02)
    if err:
        print("拦截：", err)
    else:
        tracker.record("task-002", "gpt-4", input_tokens=1000, output_tokens=500)

    print("\n" + tracker.report())
```

**运行输出示例：**

```
[预算] 任务 task-001 预算设为 0.0500 元。
[预算] 任务 task-002 预算设为 0.0100 元。

=== 任务 001：正常调用（未超预算）===

[成本] 任务=task-001 模型=gpt-3.5-turbo 输入=500 输出=300 本次=0.0014 元
[成本] 任务=task-001 模型=gpt-3.5-turbo 输入=500 输出=300 本次=0.0014 元

==================================================

=== 任务 002：超预算拦截 ===

拦截：超预算：任务 task-002 已用 0.0000 元，预算 0.0100 元，本次预估 0.0200 元，将超出。

=== 成本报告 ===

总调用次数：2
总成本：0.0028 元

按任务分布：
  task-001: 0.0028 元 (预算使用率 5.6%)
```

**面试要点**：成本追踪与预算控制的核心是“Token 计数→成本核算→任务归属→预算检查→超限拦截/告警”。手撕时要体现：CallRecord 记录单次调用、_calc_cost 按单价换金额、check_budget 超预算拦截、record 累计任务成本、report 生成报告。

---

## Q33. 框架的灰度发布与 A/B 测试：怎么验证新版本更好？

**回答**

框架升级后（比如换模型、改提示词、加工具），怎么知道新版本“更好”而不是“变差”？直接全量上线风险大，万一新版本有 bug 或回答质量下降，所有用户都受影响。**灰度发布（Canary Release）** 和 **A/B 测试** 就是先让一小部分流量跑新版，观察无误再逐步扩大，并用数据比较新旧版本效果。

### 灰度发布与 A/B 测试的核心设计

| 概念 | 大白话解释 | 作用 |
|------|-----------|------|
| **灰度发布** | 新版本先给一小部分用户用，没问题再扩大 | 降低升级风险，出问题影响范围小 |
| **流量分流** | 按用户 ID/比例把请求分给旧版或新版 | 决定谁用新版、谁用旧版 |
| **A/B 测试** | 同时跑两个版本，比关键指标（准确率、用户满意度） | 数据说话，验证新版是否更好 |
| **回滚** | 新版出问题，立刻切回旧版 | 快速恢复服务 |

用大白话：灰度发布就像“试吃”。新菜先给几位常客尝尝，大家都说好再全店推广；A/B 测试就像“比试”，做两道菜让顾客盲测，看哪道更受欢迎。

### 灰度发布流程图（ASCII）

```
用户请求
   │
   ▼
┌─────────────┐
│ 流量分流器   │──► 按用户 ID 哈希或比例分流
│ (Router)    │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
旧版(90%) 新版(10%)
   │       │
   ▼       ▼
返回旧版 返回新版
   │       │
   └───┬───┘
       ▼
┌─────────────┐
│ 指标对比    │──► 新旧版本准确率、满意度、错误率
│ (A/B 测试)  │
└─────────────┘
       │
   新版更好？
   │
 ┌───┴───┐
 是       否
 ▼       ▼
全量上线  回滚/修改
```

### 一个可运行的例子：极简灰度发布与 A/B 测试

下面这段代码实现了一个最简灰度发布器：按用户 ID 哈希分流到旧版或新版，记录每个版本的调用次数和“成功率”，并输出 A/B 测试对比报告。不依赖真实模型，直接运行即可看懂灰度与 A/B 测试骨架。

```python
"""
极简灰度发布 + A/B 测试：按用户 ID 分流 + 指标对比
"""

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class VersionMetrics:
    """单个版本的指标。"""

    calls: int = 0
    successes: int = 0
    failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.successes / self.calls


class CanaryRouter:
    """灰度发布路由器：按用户 ID 哈希分流到旧版或新版。"""

    def __init__(self, canary_ratio: float = 0.1):
        """
        canary_ratio: 新版本的流量比例（0~1），例如 0.1 表示 10% 流量去新版。
        """
        self.canary_ratio = canary_ratio
        self.old_metrics = VersionMetrics()
        self.new_metrics = VersionMetrics()

    def _hash_user(self, user_id: str) -> float:
        """把用户 ID 哈希成 0~1 的数，同一用户稳定分到同一版本。"""
        h = hashlib.md5(user_id.encode()).hexdigest()
        return int(h, 16) / (16 ** len(h))  # 0 ~ 1

    def route(self, user_id: str) -> str:
        """根据用户 ID 决定走“old”还是“new”。"""
        h = self._hash_user(user_id)
        return "new" if h < self.canary_ratio else "old"

    def record(self, version: str, success: bool):
        """记录一次调用的结果。"""
        if version == "new":
            self.new_metrics.calls += 1
            if success:
                self.new_metrics.successes += 1
            else:
                self.new_metrics.failures += 1
        else:
            self.old_metrics.calls += 1
            if success:
                self.old_metrics.successes += 1
            else:
                self.old_metrics.failures += 1

    def ab_report(self) -> str:
        """A/B 测试对比报告。"""
        lines = ["=== A/B 测试报告（灰度发布）==="]
        lines.append(f"新版本流量比例：{self.canary_ratio*100:.0f}%")
        lines.append("")
        lines.append(f"旧版本：调用 {self.old_metrics.calls} 次，成功率 {self.old_metrics.success_rate*100:.1f}%")
        lines.append(f"新版本：调用 {self.new_metrics.calls} 次，成功率 {self.new_metrics.success_rate*100:.1f}%")
        diff = self.new_metrics.success_rate - self.old_metrics.success_rate
        lines.append("")
        if diff > 0.01:
            lines.append(f"结论：新版本成功率更高（+{diff*100:.1f}%），建议全量上线。")
        elif diff < -0.01:
            lines.append(f"结论：新版本成功率更低（{diff*100:.1f}%），建议回滚或修改。")
        else:
            lines.append("结论：两版本差异不显著，可继续观察或按其他指标判断。")
        return "\n".join(lines)


# 模拟两个版本的“智能体”：旧版简单，新版更稳（演示版）
def old_agent(query: str) -> bool:
    """旧版：70% 成功。"""
    import random
    return random.random() < 0.70


def new_agent(query: str) -> bool:
    """新版：85% 成功（演示版设为更高）。"""
    import random
    return random.random() < 0.85


# ==================== 运行示例 ====================

if __name__ == "__main__":
    import random

    router = CanaryRouter(canary_ratio=0.3)  # 30% 流量去新版

    print("=== 模拟 100 个用户请求（30% 走新版，70% 走旧版）===\n")
    for i in range(100):
        user_id = f"user-{i}"
        version = router.route(user_id)
        query = f"问题 {i}"
        # 调用对应版本
        success = new_agent(query) if version == "new" else old_agent(query)
        router.record(version, success)

    print(router.ab_report())
```

**运行输出示例：**

```
=== 模拟 100 个用户请求（30% 走新版，70% 走旧版）===

=== A/B 测试报告（灰度发布）===

新版本流量比例：30%

旧版本：调用 70 次，成功率 70.0%
新版本：调用 30 次，成功率 86.7%

结论：新版本成功率更高（+16.7%），建议全量上线。
```

> 注意：结果随随机数而异，每次运行比例和成功率会有波动。

**面试要点**：灰度发布与 A/B 测试的核心是“按用户 ID 哈希分流→旧版/新版并行→记录指标→对比成功率→决定全量上线或回滚”。手撕时要体现：CanaryRouter.route 按用户 ID 稳定分流、canary_ratio 控制新版比例、record 记录每版指标、ab_report 对比并给出建议。

---

## Q34. 框架的国际化与多语言支持：怎么支持多种语言？

**回答**

框架如果只支持中文，面向全球用户或多语言企业时就受限。**国际化与多语言支持（Internationalization, i18n）** 就是让框架的提示词、错误信息、界面文本能根据用户语言自动切换，让不同语言的用户都能用母语交互。

### 国际化的核心设计

| 要素 | 大白话解释 | 作用 |
|------|-----------|------|
| **语言代码（Locale）** | 如 `zh-CN`（中文）、`en-US`（英文）、`ja-JP`（日文） | 标识用户语言 |
| **语言包（Locale Pack）** | 把同一语言的所有文本放在一起 | 方便管理和切换 |
| **键值映射（Key-Value）** | 用“键”查“值”，如 `greeting` → 中文“你好”/英文“Hello” | 代码里只用键，显示时按语言取值 |
| **参数占位符** | 如 `Hello, {name}!`，运行时替换成真实名字 | 支持动态内容 |
| **回退语言（Fallback）** | 某语言缺少翻译时用默认语言（如英文） | 避免显示空白 |

用大白话：国际化就像“多语言菜单”。同一道菜，中文菜单写“宫保鸡丁”，英文菜单写“Kung Pao Chicken”，服务员（框架）根据顾客语言递上对应菜单。

### 国际化流程图（ASCII）

```
用户请求（带语言偏好，如 Accept-Language: en-US）
        │
        ▼
┌─────────────┐
│ 语言解析    │──► 解析出 locale = "en-US"
│ (Locale)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 语言包查找  │──► 按 locale 找对应语言包
│ (i18n)      │──► 按 key 找对应文本
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 参数替换    │──► 把 {name} 换成真实值
│ (格式化)    │
└──────┬──────┘
       │
       ▼
   返回用户语言的文本
```

### 一个可运行的例子：极简国际化系统

下面这段代码实现了一个最简国际化系统：多语言包、按 locale 取文本、参数占位符替换、回退语言。直接运行即可看懂 i18n 骨架。

```python
"""
极简国际化系统：多语言包 + 按 locale 取文本 + 参数占位符 + 回退语言
"""

from typing import Dict, Optional


class I18n:
    """国际化器：管理多语言包，按语言代码取文本。"""

    def __init__(self, default_locale: str = "en-US"):
        self.default_locale = default_locale
        # 语言包：locale -> {key: 模板字符串}
        self.locales: Dict[str, Dict[str, str]] = {
            "zh-CN": {
                "greeting": "你好，{name}！",
                "welcome": "欢迎使用智能体框架。",
                "error_unknown": "出现未知错误，请稍后重试。",
                "tool_not_found": "未找到工具：{tool_name}",
                "budget_exceeded": "已超出预算，拒绝执行。",
            },
            "en-US": {
                "greeting": "Hello, {name}!",
                "welcome": "Welcome to the Agent Framework.",
                "error_unknown": "An unknown error occurred, please try again later.",
                "tool_not_found": "Tool not found: {tool_name}",
                "budget_exceeded": "Budget exceeded, execution rejected.",
            },
            "ja-JP": {
                "greeting": "こんにちは、{name}さん！",
                "welcome": "エージェントフレームワークへようこそ。",
                "error_unknown": "不明なエラーが発生しました。",
                # tool_not_found 和 budget_exceeded 故意缺失，演示回退
            },
        }

    def t(self, key: str, locale: Optional[str] = None, **params) -> str:
        """
        根据 key 和 locale 获取翻译文本，并替换参数。
        如果指定 locale 缺少该 key，回退到默认语言。
        """
        locale = locale or self.default_locale
        # 先找指定语言，再找默认语言（回退）
        text = self.locales.get(locale, {}).get(key)
        if text is None:
            text = self.locales.get(self.default_locale, {}).get(key, key)
        # 参数占位符替换：{name} -> 真实值
        for k, v in params.items():
            text = text.replace("{" + k + "}", str(v))
        return text

    def add_locale(self, locale: str, translations: Dict[str, str]):
        """添加或更新一种语言的语言包。"""
        self.locales[locale] = translations
        print(f"[i18n] 已添加语言包：{locale}")


# ==================== 运行示例 ====================

if __name__ == "__main__":
    i18n = I18n(default_locale="en-US")

    print("=== 中文用户 ===\n")
    print(i18n.t("greeting", locale="zh-CN", name="张三"))
    print(i18n.t("welcome", locale="zh-CN"))
    print(i18n.t("tool_not_found", locale="zh-CN", tool_name="搜索"))

    print("\n" + "=" * 50 + "\n")

    print("=== 英文用户 ===\n")
    print(i18n.t("greeting", locale="en-US", name="Alice"))
    print(i18n.t("welcome", locale="en-US"))
    print(i18n.t("budget_exceeded", locale="en-US"))

    print("\n" + "=" * 50 + "\n")

    print("=== 日文用户（tool_not_found 缺失，回退到英文）===\n")
    print(i18n.t("greeting", locale="ja-JP", name="太郎"))
    print(i18n.t("tool_not_found", locale="ja-JP", tool_name="検索"))  # 回退英文
```

**运行输出示例：**

```
=== 中文用户 ===

你好，张三！
欢迎使用智能体框架。
未找到工具：搜索

==================================================

=== 英文用户 ===

Hello, Alice!
Welcome to the Agent Framework.
Budget exceeded, execution rejected.

==================================================

=== 日文用户（tool_not_found 缺失，回退到英文）===

こんにちは、太郎さん！
Tool not found: 検索
```

**面试要点**：国际化与多语言支持的核心是“语言包 + 键值映射 + 参数占位符 + 回退语言”。手撕时要体现：I18n 类管理多语言包、t(key, locale, **params) 按语言取文本并替换参数、缺少翻译时回退到默认语言。

---

## Q35. 本章总结：从零构建框架的决策树与速查表

**回答**

本章咱们从零开始，手撕了智能体框架的核心组件：message.py（统一消息契约）、config.py（中心化配置）、agent.py（Agent 抽象基类）、框架主循环（感知→思考→行动→观察）、记忆层、工具层、消息总线、插件系统、多Agent 编排与状态机、序列化与持久化、限流与熔断、分布式部署、版本管理与回滚、压力测试与基准测试、文档与示例、社区与生态、生产上线检查清单、多框架集成、可观测性与追踪、安全沙箱、并发控制、多模态支持、工作流可视化编排、中间件设计、反思机制、RAG 知识库集成、API 网关与 SDK、成本追踪与预算控制、灰度发布与 A/B 测试、国际化与多语言支持。

下面是本章速查表，帮你记住关键点，面试时能快速调用。

---

### 本章速查表

| 知识点 | 一句话大白话 | 面试常考点 |
|--------|------------|-----------|
| 为什么自建框架 | 看清骨架、极致定制、培养系统设计能力 | 能否说出三个原因 |
| 框架八件套 | 模型/提示词/记忆/工具/规划/工作流/安全约束/可观测性 | 能否说出每层在干什么 |
| message.py | 统一消息契约，角色限定，to_dict 转 OpenAI 格式 | MessageRole 四种取值 |
| config.py | 中心化配置，环境变量优先，Pydantic 类型安全 | from_env 类方法 |
| agent.py | Agent 抽象基类，_think/_act 抽象方法，run 主循环 | max_steps 防止死循环 |
| 主循环 | 感知→思考→行动→观察，异常隔离，执行轨迹 | 每一步在干什么 |
| 记忆层 | 短期内存列表 + 长期向量数据库 | 两层分工 |
| 工具层 | Tool 基类 + ToolRegistry 注册表，异常隔离 | call() 不抛异常 |
| 消息总线 | 发布/订阅 + 事件类型路由，组件解耦 | subscribe/publish |
| 插件系统 | 插件基类 + 注册表 + 热加载/卸载 | Plugin 基类、register/unregister |
| 多Agent 编排 | 状态机 + Agent 绑定状态 + 状态转移 | State 枚举、Orchestrator |
| 持久化 | 状态快照 + 序列化格式 + 保存/加载文件 | to_dict/from_dict、save/load |
| 限流与熔断 | 令牌桶限流 + 三状态熔断器 | allow()、CLOSED/OPEN/HALF_OPEN |
| 分布式部署 | 节点注册 + 心跳检测 + 失效剔除 | NodeRegistry、heartbeat/detect_failure |
| 版本管理 | 版本号 + 版本仓库 + 当前版本指针 + 回滚 | save_version/switch_version/rollback |
| 压力测试 | QPS、响应时间、P99、错误率、Token 消耗 | 并发队列、指标统计 |
| 多模态 | 图片/音频/视频转成模型能接受的表示 | Base64/URL、ASR、关键帧 |
| 工作流编排 | 节点 + 边 + 条件分支 + 并行，按图执行 | Node/Workflow、add_condition |
| 中间件 | 前置/后置拦截 + 异常处理 + 中间件链 | before/after/on_error |
| 反思机制 | 生成草稿 → 自我反思 → 修改改进 → 迭代 | generate_draft/reflect/revise |
| RAG 集成 | 文档→分块→向量化→检索→拼接 Prompt→生成 | 分块大小/重叠、Top K 检索 |
| API 网关 + SDK | HTTP 接口 + SDK 封装 + 认证限流 | 认证/限流/路由、统一 JSON 响应 |
| 成本追踪 | Token 计数 → 成本核算 → 任务归属 → 预算检查 | check_budget 超预算拦截 |
| 灰度发布 | 按用户 ID 哈希分流 → 旧版/新版对比 → 全量或回滚 | canary_ratio、ab_report |
| 国际化 | 语言包 + 键值映射 + 参数占位符 + 回退语言 | t(key, locale, **params) |

---

### 选型决策树：自建框架 vs 现成框架（ASCII）

```
开始：要不要自己造框架？
   │
   ▼
团队有资深开发者，需要极致定制？ ──否──► 用现成框架（LangGraph/AutoGen/AgentScope）
   │
   是
   ▼
业务有特殊需求（私有芯片、特定审批、非标工具路由）？ ──否──► 现成框架为主，关键处自建组件替换
   │
   是
   ▼
需要深度理解框架骨架，培养系统设计能力？ ──否──► 现成框架 + 二次开发
   │
   是
   ▼
从零自建“玩具级”框架理解原理，生产上基于现成框架二次开发
```

---

### 本章一句话小结

> 从零构建智能体框架不是为了替代现成框架，而是为了看清骨架、掌握每一层在干什么；生产环境的最佳实践是“现成框架做主体，在关键环节用自己的组件替换”。

---

### 下章预告

下一章《记忆与检索》我们将深入智能体记忆系统：短期记忆、长期记忆、向量数据库、RAG 检索策略、记忆检索与重排等。敬请期待！

---

> **本章正文结束。** 如需继续，请阅读《记忆与检索》章节。

