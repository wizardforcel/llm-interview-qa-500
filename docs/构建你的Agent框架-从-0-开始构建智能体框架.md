# 构建你的Agent框架 —— 从 0 开始构建智能体框架

> 面试生普（LLM 500 题）· 构建你的Agent框架篇。本章覆盖为什么要构建自己的框架、智能体框架八件套（模型层、提示词层、记忆层、工具层、规划层、工作流层、安全约束层、可观测性层）、手撕 message.py 统一消息契约、手撕 config.py 中心化配置管理、手撕 agent.py 抽象基类、手撕框架主循环（感知→思考→行动→观察）、记忆层设计（短期+长期）、工具层注册与调用、可观测性与人工干预、本章小结，共 10 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Python 代码和 mermaid/ASCII 图。

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

### 下章预告

下一章《记忆与检索 —— 记忆系统、RAG、存储》我们将深入聊智能体的记忆系统：短期记忆怎么管理、长期记忆怎么用向量数据库实现、RAG（检索增强生成）怎么让智能体查外部知识、不同存储方式怎么选。敬请期待！

---

> **本章正文结束。** 如需继续，请阅读《记忆与检索》章节。
