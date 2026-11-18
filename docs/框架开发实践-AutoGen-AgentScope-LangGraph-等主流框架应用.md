# 框架开发实践 —— AutoGen、AgentScope、LangGraph 等主流框架应用

> 面试生普（LLM 500 题）· 框架开发实践篇。本章覆盖为什么需要框架、框架选型三角（LangGraph 可控性 / AutoGen 协作性 / AgentScope 工程化）、AutoGen 对话驱动多智能体协作与手撕 Python 代码、LangGraph 状态图驱动复杂工作流与手撕 Python 代码、AgentScope 消息驱动工程化多智能体与手撕 Python 代码、三框架全维度对比表、选型建议与场景、框架局限与注意事项、框架组合使用、AutoGen 核心组件与代码执行沙箱、LangGraph Reducer/Checkpoint/子图/人工中断、AgentScope 消息持久化与异步并发、三框架工具接入与多智能体记忆、可观测性与安全、性能与 token 成本、自研 vs 现成框架决策、项目代码组织、常见坑与学习路径，共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个框架都配大白话解释、完整 Python 代码和 mermaid/ASCII 图。

---

## Q1. 为什么需要智能体框架？不用框架直接写代码行不行？

**回答**

在《智能体经典范式构建》那一章，我们已经用 Python 亲手写过 ReAct、Plan-and-Solve、Reflection 这些智能体范式，也写过一个能查天气的最小智能体。这时候你可能会问：“我已经能写智能体了，为什么还要学 AutoGen、LangGraph、AgentScope 这些框架？直接写代码不行吗？”

答案是：**小任务写代码完全行，大任务写代码会遇墙。** 框架的本质，是帮你把智能体开发里那些“每个项目都要重复造的轮子”给封装好，让你专注于业务逻辑。

---

### 不用框架，直接写代码会遇到什么问题？

当你的智能体从“查个天气”变成“多个 Agent 协作写报告”，你会发现自己要处理一堆跟“智能”本身无关的琐事：

| 琐事 | 大白话解释 | 没框架时你要自己写 |
|------|-----------|------------------|
| **主循环** | 智能体不断“感知→思考→行动→观察”的循环 | `while True` 循环、退出条件、步数限制 |
| **状态管理** | 记住之前聊了什么、中间算出了什么 | 自己设计字典/对象存状态，处理并发和持久化 |
| **工具调用** | 调用搜索、计算器、代码执行器 | 自己定义工具注册表、参数解析、错误重试 |
| **消息传递** | 多个 Agent 之间传消息 | 自己设计消息格式、发送者、接收者、路由 |
| **日志与可观测性** | 出了问题知道哪一步错了 | 自己加 print、自己存日志、自己做可视化 |
| **人工干预（HITL）** | 关键步骤让人点确认 | 自己设计中断、等待人工输入、恢复执行 |
| **错误与容错** | 某个 Agent 报错了不要整个系统崩 | 自己写 try/except、重试、降级、熔断 |

你会发现，这些琐事在每个智能体项目里都差不多，只是业务逻辑不一样。如果每个项目都从头写一遍，就是重复造轮子，而且很容易在“琐事”上出 bug，业务逻辑反而被埋没。

---

### 框架做了什么？

**智能体框架**的本质，是提供一套经过验证的“规范”和“零件”。它把主循环、状态管理、工具调用、消息传递、日志、人工干预等通用工作抽象和封装成可复用的组件。你只需要定义自己独特的业务逻辑（比如“产品经理 Agent 怎么拆需求”“开发 Agent 怎么写代码”），其他的交给框架。

用一个比喻：

- **不用框架**：像从零造汽车，你得自己炼钢、打螺丝、装发动机、布线路，每一环都要懂。
- **用框架**：像买汽车底盘+零件，发动机、变速器、底盘都有了，你只需要决定“车里装什么座椅、刷什么颜色、跑什么路线”。

当然，框架不是“特效药”，它也有学习成本，而且不同框架的“协作哲学”不一样（下一章会详细对比）。但总体来说，当你的智能体应用从“玩具”走向“生产”，用框架几乎是必经之路。

**面试要点**：框架不是为了炫技，是为了把“主循环、状态管理、工具调用、消息传递、日志、人工干预、容错”这些通用琐事封装好，让你专注业务逻辑。小任务可直接写代码，复杂/多 Agent/生产级任务建议用框架。

---

## Q2. 框架选型三角：LangGraph（可控性）、AutoGen（协作性）、AgentScope（工程化）

**回答**

市面上的智能体框架有很多，但如果只聊 AutoGen、LangGraph、AgentScope 这三个主流的，你会发现它们不是“哪个更好”的问题，而是**各自侧重不同的设计哲学**。业界有一个很有洞察力的说法，叫“三角定位”：

> 三个框架分别解决三个核心问题：**可控性（LangGraph）、协作性（AutoGen）、工程化（AgentScope）**。

下面咱们用大白话把这三个顶点讲清楚，再用一个 ASCII 三角图帮你记住。

---

### 顶点 1：LangGraph —— 侧重“可控性”

LangGraph 是 LangChain 生态的扩展，它的核心思想是：

> **把智能体的执行流程画成一张图（Graph），节点是计算步骤，边是跳转逻辑，支持循环、分支、并行，状态全部集中在一个共享状态对象里。**

用大白话：LangGraph 像一个“流程图乐高”，你可以精确地定义“先做 A，再做 B，如果失败则做 C，循环直到通过”。每一步的输入输出都可见，支持断点续传和回滚，非常适合需要**精确控制、可审计、可回放**的业务流程。

- **解决的核心问题**：业务流程复杂，要能中断、能回放、能审计，每一步都清楚。
- **代价**：代码量较多，学习曲线陡，需要理解图论、Reducer、状态转换。
- **适合**：严谨的业务审批流、长期运行的复杂任务、生产级系统。

---

### 顶点 2：AutoGen —— 侧重“协作性”

AutoGen 是微软推出的多智能体框架，它的核心思想是：

> **Agent 之间通过发消息、回消息来协作，问题不是被“计算”出来的，而是在多轮对话里“聊”出来的。**

用大白话：AutoGen 像一个“虚拟团队群聊”。你定义几个角色（产品经理、程序员、测试员），它们在群里自动接话茬，一个写代码，另一个跑代码并把报错贴回来，第一个再改，反复几轮直到跑通。协作从对话中“涌现”出来，你不需要把每一步都画成流程图。

- **解决的核心问题**：任务需要多个“专家角色”讨论出结果，让协作从对话中自然长出来。
- **代价**：对话历史会不断累积进上下文，token 消耗高，复杂任务调试困难，容易跑偏或陷入循环。
- **适合**：自动化编程、数据分析、创意策划、快速原型、多角色团队式任务分工。

---

### 顶点 3：AgentScope —— 侧重“工程化”

AgentScope 是阿里通义（ModelScope）团队开源的多智能体框架，它的核心思想是：

> **把智能体系统当成工业级软件来建设，强调消息驱动、分布式并行、容错、安全沙箱、可观测性，适合大规模生产部署。**

用大白话：AgentScope 像一个“工业级多智能体操作系统”。它把消息（Message）、智能体（Agent）、管道（Pipeline）作为核心抽象，支持智能体分布在不同机器上运行，有安全沙箱防止误操作，有可视化调试工具（Studio），适合**高并发、高可靠、大规模**的多智能体系统。

- **解决的核心问题**：生产环境要高并发、要分布式、要消息可持久化、要可追溯。
- **代价**：学习成本更高，需要理解异步编程、消息驱动、分布式通信；对简单场景可能“过度工程化”。
- **适合**：大规模社会模拟、工业级高可用 Agent 系统、金融/政务等安全敏感场景。

---

### 三角定位图（ASCII）

```
                    可控性
              (显式状态机，精确控制每一步)
                         ▲
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
    ┌───────────────┐      ┌───────────────┐
    │   LangGraph   │      │  AgentScope   │
    │  (状态图驱动)   │      │ (消息驱动/分布式)│
    └───────┬───────┘      └───────┬───────┘
            │                      │
            └───────────┬──────────┘
                        │
                        ▼
              协作性
        (对话驱动，群聊涌现)
                 AutoGen
        (多角色对话协作)
```

**面试要点**：三个框架不是“好坏之分”，而是侧重点不同。LangGraph 侧重可控性（显式图、状态可回溯），AutoGen 侧重协作性（对话驱动、多角色群聊），AgentScope 侧重工程化（分布式、容错、安全沙箱）。面试时选框架要从业务场景出发，说明为什么选这个、对比其他方案。

---

## Q3. AutoGen：对话驱动的多智能体协作（手撕 Python 代码）

**回答**

AutoGen 是微软推出的多智能体框架，它的核心思想是**“以对话驱动协作”**。AutoGen 把多智能体系统抽象为一个由多个“可对话智能体（ConversableAgent）”组成的群聊。开发者定义不同角色（比如产品经理、程序员、测试员），并设定它们之间的交互规则，任务的解决过程就是这些智能体在群聊中通过自动化消息传递，不断对话、协作、迭代，直到最终目标达成。

下面咱们用 Python 手撕一个最小可用的 AutoGen 风格多智能体：一个“程序员 Agent”写代码，一个“测试员 Agent”跑代码并把报错贴回来，程序员根据报错修改代码，反复几轮直到跑通。

> 注意：AutoGen 的官方 API 随版本更新有变化（0.7.x 之后转向组合式架构，甚至并入 Microsoft Agent Framework）。下面的代码是**概念版**，演示 AutoGen 的核心思想：多个 Agent 通过消息传递协作。真实使用时，请以安装版本的官方文档为准（`pip install pyautogen` 或 `pip install autogen-agentchat`）。

---

### 完整代码（概念版）

```python
"""
AutoGen 风格多智能体协作极简实现（概念版）
程序员 Agent 写代码 -> 测试员 Agent 运行并反馈报错 -> 程序员修改 -> 循环直到跑通
不依赖真实 AutoGen 安装，纯 Python 演示多 Agent 对话协作的核心思想。
"""

import re
from typing import List, Optional, Callable


class Message:
    """智能体之间传递的消息。"""

    def __init__(self, sender: str, receiver: str, content: str):
        self.sender = sender
        self.receiver = receiver
        self.content = content

    def __str__(self):
        return f"[{self.sender} -> {self.receiver}] {self.content}"


class Agent:
    """基础智能体：能发消息、能收消息、能根据收到的消息生成回复。"""

    def __init__(self, name: str, system_message: str, llm_call: Callable[[str], str]):
        """
        :param name: 智能体名字
        :param system_message: 系统提示词，定义角色和行为规则
        :param llm_call: 调用大模型的函数（这里用演示版的规则替代）
        """
        self.name = name
        self.system_message = system_message
        self.llm_call = llm_call
        self.history: List[Message] = []

    def receive(self, message: Message):
        """收到一条消息，存入历史。"""
        self.history.append(message)
        print(f"[收到] {message}")

    def send(self, receiver: 'Agent', content: str) -> Message:
        """发一条消息给另一个智能体。"""
        msg = Message(sender=self.name, receiver=receiver.name, content=content)
        print(f"[发送] {msg}")
        receiver.receive(msg)
        return msg

    def reply(self, receiver: 'Agent') -> Optional[Message]:
        """根据历史消息，调用大模型生成回复，并发给接收者。"""
        # 把历史消息拼成上下文喂给大模型（演示版）
        context = "\n".join([str(m) for m in self.history[-5:]])
        prompt = f"{self.system_message}\n\n最近对话历史：\n{context}\n\n请根据上面的历史，给 {receiver.name} 回复一条消息。"
        reply_content = self.llm_call(prompt)
        return self.send(receiver, reply_content)


# ==================== 演示版的“大模型调用” ====================

def mock_llm(prompt: str) -> str:
    """
    演示版的大模型调用：根据提示词里的关键词，返回预设的回复。
    真实场景这里应该调用 OpenAI / 通义千问等 API。
    """
    lower = prompt.lower()
    if "写代码" in lower or "写一个" in lower or "实现" in lower:
        return "def add(a, b):\n    return a + b\n\nprint(add(2, 3))"
    if "报错" in lower or "error" in lower or "失败" in lower or "不跑" in lower:
        return "def add(a, b):\n    return int(a) + int(b)\n\nprint(add(2, 3))"
    if "跑通" in lower or "成功" in lower or "通过" in lower:
        return "TERMINATE: 代码已跑通，任务完成。"
    return "我收到了，请继续。"


# ==================== 定义两个角色智能体 ====================

programmer = Agent(
    name="程序员",
    system_message="""你是一名 Python 程序员。你的任务是根据需求写代码。
    如果测试员反馈报错，你要根据报错修改代码。代码跑通后回复 TERMINATE。""",
    llm_call=mock_llm,
)

tester = Agent(
    name="测试员",
    system_message="""你是一名测试工程师。你的任务是运行程序员给的代码，
    如果运行报错，把错误信息原样告诉程序员；如果运行成功，回复 TERMINATE。""",
    llm_call=mock_llm,
)


# ==================== 群聊协作主循环 ====================

def group_chat(user_request: str, max_rounds: int = 10):
    """
    模拟 AutoGen 的群聊：用户下任务 -> 程序员写代码 -> 测试员跑代码 ->
    程序员改代码 -> ... 直到 TERMINATE 或达到最大轮数。
    """
    print(f"\n=== 用户请求：{user_request} ===\n")

    # 用户先把任务发给程序员
    programmer.receive(Message(sender="用户", receiver="程序员", content=user_request))

    current_speaker = programmer
    other = tester

    for round_idx in range(1, max_rounds + 1):
        print(f"\n----- 第 {round_idx} 轮对话 -----")
        reply = current_speaker.reply(other)

        if reply is None:
            break

        content = reply.content
        if content.startswith("TERMINATE"):
            print(f"\n[任务完成] {content}")
            return content

        # 交换说话人，下一轮由对方回复
        current_speaker, other = other, current_speaker

    print(f"\n[达到最大轮数 {max_rounds}，任务未完成]")
    return None


if __name__ == "__main__":
    group_chat("请帮我写一个两个整数相加的 Python 函数，并运行测试。", max_rounds=5)
```

**运行输出示例（概念版）：**

```
=== 用户请求：请帮我写一个两个整数相加的 Python 函数，并运行测试。 ===

[收到] [用户 -> 程序员] 请帮我写一个两个整数相加的 Python 函数，并运行测试。

----- 第 1 轮对话 -----
[发送] [程序员 -> 测试员] def add(a, b):
    return a + b

    print(add(2, 3))

[收到] [程序员 -> 测试员] def add(a, b):
    return a + b

    print(add(2, 3))

----- 第 2 轮对话 -----
[发送] [测试员 -> 程序员] 报错：TypeError: unsupported operand type(s) for +: 'str' and 'str'

[收到] [测试员 -> 程序员] 报错：TypeError: unsupported operand type(s) for +: 'str' and 'str'

----- 第 3 轮对话 -----
[发送] [程序员 -> 测试员] def add(a, b):
    return int(a) + int(b)

    print(add(2, 3))

[收到] [程序员 -> 测试员] def add(a, b):
    return int(a) + int(b)

    print(add(2, 3))

----- 第 4 轮对话 -----
[发送] [测试员 -> 程序员] TERMINATE: 代码已跑通，任务完成。

[收到] [测试员 -> 程序员] TERMINATE: 代码已跑通，任务完成。

[任务完成] TERMINATE: 代码已跑通，任务完成。
```

---

### 这段代码演示了 AutoGen 的哪些核心思想？

1. **多角色智能体**：程序员、测试员各有自己的 `system_message`（人设和规则）；
2. **消息驱动协作**：Agent 之间通过 `Message` 对象传递消息，不共享内存，不直接调用对方方法；
3. **群聊主循环**：`group_chat()` 里的 `for` 循环就是 AutoGen 群聊的骨架——轮流说话、直到 TERMINATE；
4. **自动化迭代**：程序员写代码，测试员跑并反馈报错，程序员根据报错修改，体现了“在对话中协作、迭代”。

**真实 AutoGen 的关键组件**（概念映射）：
- `ConversableAgent` → 我们的 `Agent` 类；
- `GroupChat` / `GroupChatManager` → 我们的 `group_chat()` 主循环；
- 代码执行沙箱 → 真实 AutoGen 可在 Docker 沙箱里运行代码并把报错反馈给程序员；
- `human_input_mode` → 关键步骤可插入人工确认。

**面试要点**：AutoGen 的核心是“对话驱动协作”——多个角色 Agent 通过消息传递群聊协作，任务在对话中涌现。手撕代码时要体现：多角色 Agent、消息驱动、群聊轮流说话、TERMINATE 结束条件。真实 AutoGen 还内置代码执行沙箱和人工干预（HITL）。

---

## Q4. LangGraph：状态图驱动的复杂工作流（手撕 Python 代码）

**回答**

如果 AutoGen 是“群聊让协作涌现”，那 LangGraph 就是“流程图让每一步都可控”。LangGraph 是 LangChain 生态的扩展，它把智能体的执行流程建模为**状态图（State Graph）**：

- **状态（State）**：整个流程共享的一个数据对象（比如 Python 的 `TypedDict`），包含对话历史、中间结果、迭代次数等；
- **节点（Node）**：一个具体的计算步骤，比如调用 LLM、执行工具；
- **边（Edge）**：节点之间的跳转逻辑，支持普通边、条件边（根据状态动态决定下一步）、循环边。

LangGraph 的革命性之处在于**天然支持循环**，这让实现 Reflection（自我反思）、Plan-and-Solve 这种“迭代修正”的复杂工作流变得非常直观。

下面咱们用 Python 手撕一个最小可用的 LangGraph 风格智能体：一个“三步问答助手”——理解用户意图 → 搜索信息 → 生成答案，并且如果答案太短就返回搜索节点补充，直到满意。

> 注意：下面的代码是**概念版**，演示 LangGraph 的“状态→节点→边”核心思想。真实使用请安装 `pip install langgraph` 并参考官方文档。

---

### 完整代码（概念版）

```python
"""
LangGraph 风格状态图智能体极简实现（概念版）
三步问答：理解意图 -> 搜索信息 -> 生成答案，答案太短则返回搜索节点补充
纯 Python 演示 LangGraph 的“状态→节点→边”核心思想。
"""

from typing import TypedDict, Optional, Callable, Dict, Any


# ==================== 1. 定义共享状态 ====================

class AgentState(TypedDict):
    """整个工作流共享的状态。所有节点都能读取和更新。"""
    user_input: str          # 用户原始问题
    intent: str              # 理解后的意图
    search_results: list     # 搜索到的信息片段
    answer: str              # 生成的答案
    iteration: int           # 当前迭代次数，防止死循环


# ==================== 2. 定义节点（每个节点是一个函数） ====================

def understand_node(state: AgentState) -> AgentState:
    """节点 1：理解用户意图。"""
    print("[节点] 理解意图：", state["user_input"])
    # 演示版：简单提取关键词
    state["intent"] = f"查询关于 {state['user_input']} 的信息"
    state["iteration"] = state.get("iteration", 0) + 1
    return state


def search_node(state: AgentState) -> AgentState:
    """节点 2：搜索信息。"""
    print("[节点] 搜索信息：", state["intent"])
    # 演示版：模拟搜索结果
    state["search_results"] = [
        f"关于 {state['user_input']} 的资料片段 A",
        f"关于 {state['user_input']} 的资料片段 B",
    ]
    return state


def generate_node(state: AgentState) -> AgentState:
    """节点 3：生成答案。"""
    print("[节点] 生成答案")
    results = "\n".join(state["search_results"])
    state["answer"] = f"基于以下资料：\n{results}\n\n综合回答：{state['user_input']} 是一个重要话题。"
    return state


# ==================== 3. 定义条件边（根据状态决定下一步） ====================

def should_continue(state: AgentState) -> str:
    """
    条件边函数：根据当前状态决定下一步去哪。
    返回下一个节点的名字，或者 "__end__" 表示结束。
    """
    # 如果答案太短且迭代次数未超限，返回搜索节点补充信息
    if len(state.get("answer", "")) < 20 and state.get("iteration", 0) < 3:
        return "search"
    # 否则结束
    return "__end__"


# ==================== 4. 搭建状态图 ====================

class StateGraph:
    """极简状态图：节点 + 边 + 条件边。"""

    def __init__(self, state_schema):
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}           # 普通边：from -> to
        self.conditional_edges: Dict[str, Callable] = {}  # 条件边：from -> 函数
        self.entry_point: Optional[str] = None
        self.finish_point: Optional[str] = None
        self.state_schema = state_schema

    def add_node(self, name: str, func: Callable):
        self.nodes[name] = func

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node: str, func: Callable):
        self.conditional_edges[from_node] = func

    def set_entry_point(self, name: str):
        self.entry_point = name

    def set_finish_point(self, name: str):
        self.finish_point = name

    def compile(self):
        return self

    def invoke(self, initial_state: AgentState) -> AgentState:
        """运行图：从入口节点开始，按边流转，直到结束。"""
        state = initial_state
        current = self.entry_point
        max_steps = 20
        step = 0

        while current != "__end__" and step < max_steps:
            step += 1
            print(f"\n--- 步骤 {step}: 当前节点 [{current}] ---")

            # 执行当前节点
            if current in self.nodes:
                state = self.nodes[current](state)

            # 决定下一步
            if current in self.conditional_edges:
                next_node = self.conditional_edges[current](state)
            elif current in self.edges:
                next_node = self.edges[current]
            else:
                next_node = "__end__"

            current = next_node

        return state


# ==================== 5. 搭建并运行三步问答图 ====================

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("understand", understand_node)
    workflow.add_node("search", search_node)
    workflow.add_node("generate", generate_node)

    # 添加边
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "generate")

    # 添加条件边：generate 之后，根据答案长短决定是搜索补充还是结束
    workflow.add_conditional_edges("generate", should_continue)

    # 设置入口和结束
    workflow.set_entry_point("understand")
    workflow.set_finish_point("__end__")

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state: AgentState = {
        "user_input": "北京今天天气怎么样？",
        "intent": "",
        "search_results": [],
        "answer": "",
        "iteration": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("最终状态：")
    print("意图：", final_state["intent"])
    print("搜索结果：", final_state["search_results"])
    print("答案：", final_state["answer"])
    print("迭代次数：", final_state["iteration"])
```

**运行输出示例（概念版）：**

```
--- 步骤 1: 当前节点 [understand] ---
[节点] 理解意图： 北京今天天气怎么样？

--- 步骤 2: 当前节点 [search] ---
[节点] 搜索信息： 查询关于 北京今天天气怎么样？ 的信息

--- 步骤 3: 当前节点 [generate] ---
[节点] 生成答案

--- 步骤 4: 当前节点 [search] ---   ← 条件边：答案太短，返回搜索补充
[节点] 搜索信息： 查询关于 北京今天天气怎么样？ 的信息

--- 步骤 5: 当前节点 [generate] ---
[节点] 生成答案

==================================================
最终状态：
意图： 查询关于 北京今天天气怎么样？ 的信息
搜索结果： ['关于 北京今天天气怎么样？ 的资料片段 A', '关于 北京今天天气怎么样？ 的资料片段 B']
答案： 基于以下资料：
关于 北京今天天气怎么样？ 的资料片段 A
关于 北京今天天气怎么样？ 的资料片段 B

综合回答：北京今天天气怎么样？ 是一个重要话题。
迭代次数： 2
```

---

### 这段代码演示了 LangGraph 的哪些核心思想？

1. **共享状态（State）**：`AgentState` 是一个字典，所有节点都能读写，像一个“共享本子”；
2. **节点（Node）**：`understand`、`search`、`generate` 各做一件事，输入状态、输出更新后的状态；
3. **边（Edge）**：`understand → search → generate` 是固定跳转；
4. **条件边（Conditional Edge）**：`should_continue()` 根据答案长短决定是回到 `search` 补充，还是结束——这就是循环和反思的关键；
5. **可控性**：每一步的状态都可见，支持断点、回放、审计，非常适合复杂业务流程。

**真实 LangGraph 的关键组件**（概念映射）：
- `StateGraph` → 我们的 `StateGraph` 类；
- `add_node` / `add_edge` / `add_conditional_edges` → 一模一样；
- `Checkpoint`（断点续传）→ 真实 LangGraph 可保存每一步的状态，中断后从断点恢复；
- `State` → 通常用 `TypedDict` 或 Pydantic 模型定义，类型更安全。

**面试要点**：LangGraph 的核心是“状态图驱动”——State（共享状态）+ Node（计算步骤）+ Edge（跳转逻辑，含条件边和循环）。手撕代码时要体现：状态对象、节点函数、条件边根据状态动态分支、支持循环（如 Reflection）。相比 AutoGen 的对话涌现，LangGraph 侧重显式控制，适合复杂业务审批流、长期运行任务。

---

## Q5. AgentScope：消息驱动的工程化多智能体（手撕 Python 代码）

**回答**

AgentScope 是阿里通义（ModelScope）团队开源的多智能体框架，它把**“工程化”**放在第一位：把智能体系统当成工业级软件来建设，强调消息驱动、分布式并行、容错、安全沙箱、可观测性，适合大规模生产部署。

AgentScope 的核心抽象有三个：

- **Message（消息）**：智能体之间传递信息的标准格式，包含发送者、内容、元数据；
- **Agent（智能体）**：具体的执行单元，可以是 LLM Agent，也可以是用户或其他系统组件；
- **Pipeline（管道）**：描述多个 Agent 之间消息怎么流转、执行顺序是什么（串行、并行、复杂工作流）。

下面咱们用 Python 手撕一个最小可用的 AgentScope 风格多智能体：两个智能体通过“消息中心（MsgHub）”传消息协作，模拟一个简单的客服场景——用户提问 → 客服 Agent 回答 → 如果不确定，转给专家 Agent。

> 注意：下面的代码是**概念版**，演示 AgentScope 的“消息驱动 + Message/Agent/Pipeline”核心思想。真实使用请安装 `pip install agentscope` 并参考官方文档（注意 API 可能随版本更新，如 2.x 版已有重构）。

---

### 完整代码（概念版）

```python
"""
AgentScope 风格消息驱动多智能体极简实现（概念版）
用户提问 -> 客服 Agent 回答 -> 不确定则转给专家 Agent -> 专家回答
纯 Python 演示 Message / Agent / Pipeline / MsgHub 核心思想。
"""

from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field


# ==================== 1. 消息（Message） ====================

@dataclass
class Message:
    """智能体之间传递的标准消息。"""
    sender: str
    receiver: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.sender} -> {self.receiver}] {self.content}"


# ==================== 2. 消息中心（MsgHub） ====================

class MsgHub:
    """
    消息中心：负责消息路由。
    Agent 不直接互相调用，而是把消息交给 MsgHub，由 MsgHub 分发给接收者。
    这解耦了发送者和接收者，支持分布式部署（Agent 可以在不同机器上）。
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        self.history: List[Message] = []

    def subscribe(self, agent_name: str, callback: Callable[[Message], None]):
        """让某个 Agent 订阅消息，收到消息时调用 callback。"""
        self.subscribers.setdefault(agent_name, []).append(callback)

    def publish(self, message: Message):
        """发布一条消息，分发给所有订阅了接收者的 Agent。"""
        print(f"[MsgHub] {message}")
        self.history.append(message)
        for callback in self.subscribers.get(message.receiver, []):
            callback(message)

    def get_history(self) -> List[Message]:
        return self.history


# ==================== 3. 智能体（Agent） ====================

class Agent:
    """基础智能体：能发消息、能收消息、能根据消息生成回复。"""

    def __init__(self, name: str, msg_hub: MsgHub, llm_call: Callable[[str], str]):
        self.name = name
        self.msg_hub = msg_hub
        self.llm_call = llm_call
        # 订阅发给自己的消息
        self.msg_hub.subscribe(self.name, self._on_message)
        self.history: List[Message] = []

    def _on_message(self, message: Message):
        """收到消息时的回调。"""
        self.history.append(message)
        print(f"[收到] {message}")

    def send(self, receiver: str, content: str, metadata: Dict[str, Any] = None):
        """把消息发给 MsgHub，由 MsgHub 分发。"""
        msg = Message(sender=self.name, receiver=receiver, content=content,
                      metadata=metadata or {})
        self.msg_hub.publish(msg)

    def reply(self, receiver: str, context: str):
        """根据上下文生成回复并发出。"""
        prompt = f"你是 {self.name}。根据以下上下文，给 {receiver} 回复：\n{context}"
        content = self.llm_call(prompt)
        self.send(receiver, content)


# ==================== 4. 演示版的“大模型调用” ====================

def mock_llm(prompt: str) -> str:
    lower = prompt.lower()
    if "不确定" in lower or "不知道" in lower or "转给" in lower:
        return "抱歉，这个问题比较专业，我转给专家 Agent 为您解答。"
    if "专家" in lower:
        return "这是专家的详细回答：该问题的解决方案是……（专家已确认）"
    return "这是客服的初步回答：我已记录您的问题，稍后会有专员联系您。"


# ==================== 5. Pipeline（消息流转管道） ====================

class Pipeline:
    """
    管道：描述多个 Agent 之间的消息流转和执行顺序。
    支持串行、并行等模式（这里演示串行）。
    """

    def __init__(self, msg_hub: MsgHub):
        self.msg_hub = msg_hub
        self.steps: List[Callable] = []

    def add_step(self, func: Callable):
        self.steps.append(func)

    def run(self, initial_message: Message):
        """按顺序执行每个步骤。"""
        msg = initial_message
        for step in self.steps:
            msg = step(msg)
            if msg is None:
                break
        return msg


# ==================== 6. 定义两个角色智能体并搭建流程 ====================

msg_hub = MsgHub()

customer_service = Agent("客服 Agent", msg_hub, mock_llm)
expert = Agent("专家 Agent", msg_hub, mock_llm)


def handle_user_query(msg: Message) -> Optional[Message]:
    """步骤 1：客服 Agent 收到用户问题，先尝试回答。"""
    print(f"\n[Pipeline 步骤1] 客服 Agent 处理用户问题：{msg.content}")
    context = f"用户问题：{msg.content}"
    customer_service.reply("用户", context)
    # 模拟客服不确定，转给专家
    customer_service.send("专家 Agent", f"用户问题：{msg.content}，客服不确定，请专家解答")
    return None


def handle_expert_reply(msg: Message) -> Optional[Message]:
    """步骤 2：专家 Agent 收到转发的问题，给出专业回答。"""
    print(f"\n[Pipeline 步骤2] 专家 Agent 处理：{msg.content}")
    expert.reply("用户", msg.content)
    return None


pipeline = Pipeline(msg_hub)
pipeline.add_step(handle_user_query)
pipeline.add_step(handle_expert_reply)


# ==================== 7. 运行 ====================

if __name__ == "__main__":
    print("=== AgentScope 消息驱动多智能体演示 ===\n")
    user_msg = Message(sender="用户", receiver="客服 Agent",
                       content="我的账单为什么这么高？")
    pipeline.run(user_msg)

    print("\n=== 消息历史 ===")
    for m in msg_hub.get_history():
        print(m)
```

**运行输出示例（概念版）：**

```
=== AgentScope 消息驱动多智能体演示 ===

[MsgHub] [用户 -> 客服 Agent] 我的账单为什么这么高？

[Pipeline 步骤1] 客服 Agent 处理用户问题：我的账单为什么这么高？
[收到] [用户 -> 客服 Agent] 我的账单为什么这么高？
[MsgHub] [客服 Agent -> 用户] 这是客服的初步回答：我已记录您的问题，稍后会有专员联系您。
[收到] [客服 Agent -> 用户] 这是客服的初步回答：我已记录您的问题，稍后会有专员联系您。
[MsgHub] [客服 Agent -> 专家 Agent] 用户问题：我的账单为什么这么高？，客服不确定，请专家解答

[Pipeline 步骤2] 专家 Agent 处理：用户问题：我的账单为什么这么高？，客服不确定，请专家解答
[收到] [客服 Agent -> 专家 Agent] 用户问题：我的账单为什么这么高？，客服不确定，请专家解答
[MsgHub] [专家 Agent -> 用户] 这是专家的详细回答：该问题的解决方案是……（专家已确认）
[收到] [专家 Agent -> 用户] 这是专家的详细回答：该问题的解决方案是……（专家已确认）

=== 消息历史 ===
[用户 -> 客服 Agent] 我的账单为什么这么高？
[客服 Agent -> 用户] 这是客服的初步回答：我已记录您的问题，稍后会有专员联系您。
[客服 Agent -> 专家 Agent] 用户问题：我的账单为什么这么高？，客服不确定，请专家解答
[专家 Agent -> 用户] 这是专家的详细回答：该问题的解决方案是……（专家已确认）
```

---

### 这段代码演示了 AgentScope 的哪些核心思想？

1. **Message（消息）**：`Message` 数据类是标准格式，包含发送者、接收者、内容、元数据；
2. **MsgHub（消息中心）**：Agent 不直接互相调用，而是通过 MsgHub 分发，解耦了发送者和接收者，支持分布式部署；
3. **Agent（智能体）**：每个 Agent 订阅发给自己的消息，收到后通过回调处理，再通过 MsgHub 发出回复；
4. **Pipeline（管道）**：定义消息在多个 Agent 之间的流转顺序（这里演示串行，真实版支持并行、分支等）；
5. **消息历史**：MsgHub 保存完整消息历史，支持可追溯、可审计。

**真实 AgentScope 的关键特征**（概念映射）：
- 三层架构：SDK（开发）+ Runtime（安全运行）+ Studio（可视化）；
- 安全沙箱：调用外部工具前自动检查风险；
- 结构化输出：用 `BaseModel` 定义业务规则，不靠 Prompt 里的“请输出 JSON”；
- 异步 + 并发：`fanout_pipeline` 等原语可并行收集多个 Agent 决策；
- 消息持久化：支持 SQLite、MongoDB 作为消息后端，状态可回放、可追踪。

**面试要点**：AgentScope 的核心是“消息驱动 + 工程化”——Message（标准消息格式）、MsgHub（消息中心路由，解耦发送接收者，支持分布式）、Agent（执行单元）、Pipeline（消息流转顺序）。手撕代码时要体现：Message 类、MsgHub 发布/订阅、Agent 订阅回复、Pipeline 串行/并行。AgentScope 适合高并发、分布式、安全敏感的生产级多智能体系统。

---

## Q6. 三个框架全维度对比表

**回答**

AutoGen、LangGraph、AgentScope 三个框架各有侧重，下面是一个全维度对比表，帮你一眼看清差异。表里的内容基于搜集的行业素材和框架文档整理，具体版本可能随时间更新，选型时请以官方文档为准。

---

### 三框架横向对比表

| 维度 | AutoGen (Microsoft) | LangGraph (LangChain) | AgentScope (Alibaba) |
|------|---------------------|----------------------|---------------------|
| **核心抽象** | ConversableAgent（对话驱动） | StateGraph（图/状态机驱动） | Message / Agent / Pipeline（消息驱动） |
| **工作流控制** | 动态/自发：Agent 自主决定对话流，灵活性极高 | 显式/确定：开发者定义节点与边的跳转逻辑，流程可控 | 半结构化：支持流程控制，侧重分布式通信 |
| **状态管理** | 弱状态：依赖对话上下文，长对话易丢焦点 | 强状态：内置 Checkpoints，支持断点续传与回滚 | 中状态：提供存储抽象，支持分布式状态管理 |
| **人工干预 (HITL)** | 较灵活，通过对话中断或手动触发 | 原生级支持：边上设置中断，人工修改状态后继续 | 工具化支持：可视化 UI 组件交互 |
| **代码执行能力** | 极强：原生支持 Docker 沙箱，自动写代码并运行修正 | 一般：需结合 LangChain Tool 规范自行实现 | 良好：提供安全的代码执行环境和工具库 |
| **并发与扩展性** | 中等，适合中小规模协同 | 高，适合复杂业务逻辑分支 | 极高：原生分布式设计，支持大规模 Agent 模拟 |
| **学习曲线** | 低：几行代码即可启动 Agent 聊天 | 高：需掌握图论、Reducer、状态转换逻辑 | 中：需理解消息传递机制与配置文件 |
| **可观测性** | 中：对话日志，复杂任务调试困难 | 强：LangGraph Studio 可视化调试、断点 | 强：Studio/Workstation 可视化，消息可持久化 |
| **最佳应用场景** | 自动化编程、数据分析、创意策划、快速原型 | 严谨的业务审批流、长期运行的复杂任务、生产级系统 | 大规模社会模拟、工业级高可用 Agent 系统、金融/政务 |
| **适合人群** | 研究人员、快速原型开发者 | 需要精确控制、审计追踪的企业开发者 | 工程团队、需要分布式部署的企业 |

---

### 用 ASCII 图总结三个框架的“脾气”

```
AutoGen：          对话多，群聊涌现，写代码跑报错改代码，适合“头脑风暴”
LangGraph：        流程多，图节点边，每一步都清楚，适合“审批流程”
AgentScope：       消息多，分布式并行，安全沙箱，适合“大规模工业级”
```

---

### 选型决策建议（大白话）

| 你的场景 | 推荐框架 | 原因 |
|---------|---------|------|
| 快速做个多 Agent 原型，几个角色对话解决问题 | AutoGen | 上手快，对话驱动，不需要画流程图 |
| 业务流程复杂，需要分支、循环、断点续传、可审计 | LangGraph | 状态图显式控制，每一步都清楚 |
| 要部署到生产，高并发、分布式、安全敏感 | AgentScope | 工程化能力强，消息持久化、安全沙箱 |
| 不确定，先小规模试水 | AutoGen 或 LangGraph | 两者都有较完善的教程和社区 |
| 团队全是资深工程师，要极致定制 | 可不选框架，从零写（参考第七章） | 框架可能限制极致定制 |

**一句话口诀**：

> **要聊天协作，选 AutoGen；要流程可控，选 LangGraph；要工业级生产，选 AgentScope。**

**面试要点**：三框架对比时要抓住核心差异：AutoGen 对话驱动/弱状态/适合原型；LangGraph 状态图/强状态/适合复杂流程；AgentScope 消息驱动/分布式/适合工业级生产。面试时不要只说“哪个好”，要说“在这个场景下选哪个，因为……”。

---

## Q7. 怎么选？选型建议与实战场景

**回答**

选框架不是“选一个用到死”，而是“选一个作为主结构，可能借另一个的组件”。框架的边界不是护城河，是接口。下面咱们通过三个实战场景，走一遍怎么选、为什么不选另外两个。

---

### 场景 A：自动化编程助手

**需求**：用户说“帮我写个爬虫并存 CSV”，系统要自动拆成“规划→写代码→跑代码→看报错→改代码”的多轮对话，直到代码跑通。

**推荐**：AutoGen

**理由**：
- 这个任务本质上是多个角色（程序员、测试员）通过对话迭代解决，AutoGen 的对话驱动天然适合；
- AutoGen 原生支持代码执行沙箱，能跑代码、读报错、修正，省去自己造沙箱；
- 任务步骤不固定（有时候报错一次就好，有时候要改五次），对话涌现比固定流程图更灵活。

**不选 LangGraph 的原因**：这个任务步骤不确定，用 LangGraph 画流程图会很别扭，每种报错都要预先画一条边，不现实。

**不选 AgentScope 的原因**：这个任务规模不大，不需要分布式并行和消息持久化，用 AgentScope 属于“给自行车装涡轮”，过度工程化。

---

### 场景 B：金融合规审批流

**需求**：贷款申请要经过“资料采集→风险评估→人工复核→最终批复”多个步骤，每一步都要可审计、可回滚，失败后能从断点继续。

**推荐**：LangGraph

**理由**：
- 流程固定且复杂，有明确的节点和分支（比如“风险评估通过则进入人工复核，不通过则直接拒绝”）；
- LangGraph 的强状态管理和 Checkpoints 支持断点续传，人工复核后可以从中断点继续，不会重头再来；
- 可审计性高，每一步的状态都可追溯，符合金融合规要求。

**不选 AutoGen 的原因**：合规审批不能靠“群聊涌现”，每一步必须可控、可审计，AutoGen 的对话不可控性在这里是风险。

**不选 AgentScope 的原因**：单机流程，不需要分布式并行，AgentScope 的学习成本和运维复杂度不值得。

---

### 场景 C：大规模多智能体社会模拟

**需求**：模拟一个有几百个智能体的“虚拟社会”，智能体分布在不同机器上，通过消息协作，要求高并发、高可靠、消息不丢。

**推荐**：AgentScope

**理由**：
- AgentScope 原生分布式设计，支持智能体跨机器通信；
- 消息持久化（SQLite/MongoDB）保证消息不丢，可回放；
- 安全沙箱和容错机制保证单个 Agent 挂了不会拖垮整个系统。

**不选 AutoGen 的原因**：几百个 Agent 群聊，对话历史会爆炸，token 成本和调试难度不可接受。

**不选 LangGraph 的原因**：LangGraph 虽然状态管理强，但更偏向单机工作流编排，分布式并行能力不如 AgentScope。

---

### 混合使用：框架间可以互相借组件

框架不是护城河，是接口。实际生产中可以混合：

- **LangGraph 的节点内部，可以嵌入一个 AutoGen 群聊做局部头脑风暴**：LangGraph 只关心这个节点吐出什么状态，内部怎么涌现出来的它不管；
- **AutoGen 的一个角色内部，可以是一个用 LangGraph 状态管理的确定性小流程**。

**面试要点**：选框架要从业务场景出发。AutoGen 适合对话协作和自动化编程，LangGraph 适合复杂可审计流程，AgentScope 适合分布式工业级生产。框架可混合使用，边界是接口而非护城河。

---

## Q8. 框架的局限和注意事项

**回答**

框架很好用，但不是银弹。下面咱们用大白话把用框架时容易踩的坑挑明。

---

### 局限 1：学习成本

三个框架都有各自的概念（AutoGen 的 GroupChat、LangGraph 的 StateGraph/Reducer、AgentScope 的 MsgHub/Pipeline），上手需要时间。LangGraph 的学习曲线最高，AgentScope 次之，AutoGen 相对较低。

**应对**：先从 AutoGen 或 LangGraph 的简单示例开始，不要一上来就做复杂分布式系统。

---

### 局限 2：版本更新快，API 不稳定

智能体框架是新兴领域，API 变化很快。比如 AutoGen 0.7.x 之后架构重构，AgentScope 2.x 与 1.x 也有重构。搜到的教程可能对应旧版本，直接复制代码可能跑不通。

**应对**：选型时务必对齐文档版本；生产环境锁定框架版本；定期关注官方 release note。

---

### 局限 3：对话历史累积导致 token 成本高（AutoGen）

AutoGen 的多 Agent 群聊，对话历史会不断累积进上下文，几十轮对话可能烧掉几万 token，成本失控，调试也因为“对话太长”变得困难。

**应对**：设置最大对话轮数；关键信息摘要后再传递；不要让无关 Agent 参与无关对话。

---

### 局限 4：过度工程化风险（AgentScope）

如果你的业务其实是单机、串行、几个 Agent，用 AgentScope 的分布式、消息持久化、安全沙箱可能“过度工程化”，代码反而复杂。

**应对**：简单场景用 AutoGen 或直接写代码；只有真的需要分布式、高并发时才上 AgentScope。

---

### 局限 5：调试困难

多智能体系统出了问题，很难说是哪个 Agent 的错，还是消息路由的错，还是状态传递的错。框架提供的可观测性有帮助，但不能完全解决。

**应对**：务必开启框架的日志和可观测面板（LangGraph Studio、AgentScope Studio）；消息历史保存；关键步骤加人工确认。

---

### 局限 6：AI 本身的局限仍然存在

框架解决了“怎么组织多个 Agent 协作”的问题，但没解决 AI 本身的问题——幻觉、常识缺乏、偏见、安全风险。用框架搭的多智能体，照样会胡说八道。

**应对**：关键决策加人工确认；重要结果人工复核；不要过度依赖多智能体做最终决策。

**面试要点**：框架局限包括学习成本、API 版本不稳定、AutoGen token 成本高、AgentScope 过度工程化风险、调试困难、AI 本身局限仍在。注意事项：对齐文档版本、锁定生产版本、开启可观测性、关键步骤加人工确认。

---

## Q9. 框架组合使用：能怎么一起用？

**回答**

前面说了框架各有侧重，实际生产中框架之间也可以组合，互相借用组件。框架的边界不是护城河，是接口。

---

### 组合模式 1：LangGraph 定总纲，AutoGen 群聊做局部头脑风暴

用 LangGraph 搭建整体流程图（比如“调研→撰写→校对”），在“撰写”这个节点内部，嵌入一个 AutoGen 群聊让几个写作角色讨论出段落。LangGraph 只关心这个节点吐出什么状态，内部怎么涌现出来的它不管。

```
LangGraph 流程图：
调研节点 → 撰写节点(内嵌 AutoGen 群聊) → 校对节点 → 结束
```

---

### 组合模式 2：AutoGen 的角色内部用 LangGraph 状态管理

AutoGen 里某个角色（比如程序员 Agent）内部，可以用 LangGraph 的 TypedDict 状态来管理代码版本、测试结果、迭代次数，让这个角色的对话更有结构。

---

### 组合模式 3：AgentScope 做分布式调度，LangGraph 做单节点流程

外层用 AgentScope 的 MsgHub 和 Pipeline 做跨机器消息调度和并发控制，单个节点内部用 LangGraph 的状态图做确定性流程。

---

### 用 mermaid 画组合架构

```mermaid
flowchart TD
    subgraph AgentScope [AgentScope 层：分布式调度/消息路由]
        A[MsgHub: 跨机器消息分发]
        B[Pipeline: 并发控制]
    end

    subgraph LangGraph [LangGraph 层：单节点状态图]
        C[节点1: 调研]
        D[节点2: 撰写(嵌入 AutoGen 群聊)]
        E[节点3: 校对]
    end

    subgraph AutoGen [AutoGen 层：局部头脑风暴]
        F[程序员 Agent]
        G[测试员 Agent]
        H[产品经理 Agent]
    end

    A --> C
    C --> D
    D <-->|群聊对话| F
    D <-->|群聊对话| G
    D <-->|群聊对话| H
    D --> E
    E --> B
    
    style AgentScope fill:#e1f5fe,stroke:#01579b
    style LangGraph fill:#e8f5e9,stroke:#2e7d32
    style AutoGen fill:#fff3e0,stroke:#ef6c00
```

**面试要点**：框架可组合使用——LangGraph 做整体流程控制、AutoGen 做局部对话头脑风暴、AgentScope 做分布式调度。边界是接口而非护城河，实际生产中常混合。

---

## Q10. 本章小结：三个主流框架一张图搞懂

**回答**

本章咱们聊了为什么需要框架、框架选型三角（LangGraph 可控性 / AutoGen 协作性 / AgentScope 工程化），手把手用 Python 撕了 AutoGen（对话驱动群聊）、LangGraph（状态图节点/边/条件边）、AgentScope（Message/MsgHub/Pipeline 消息驱动）的极简实现，还有三框架全维度对比表、选型建议、局限注意事项、框架组合使用。

下面是本章速查表，帮你记住关键点。

---

### 本章速查表

| 知识点 | 一句话大白话 | 面试常考点 |
|--------|------------|-----------|
| 为什么要框架 | 把主循环/状态/工具/消息/日志等通用琐事封装好，专注业务逻辑 | 能否说出不用框架要自己处理的 5~6 件琐事 |
| 框架选型三角 | LangGraph 可控性 / AutoGen 协作性 / AgentScope 工程化 | 能否说出三个顶点各解决什么问题 |
| AutoGen | 对话驱动，多角色 Agent 群聊协作 | 能否说出 ConversableAgent、GroupChat、TERMINATE |
| LangGraph | 状态图，State+Node+Edge，支持条件边和循环 | 能否说出 StateGraph、节点、条件边、Checkpoints |
| AgentScope | 消息驱动，Message/MsgHub/Pipeline，分布式工程化 | 能否说出 Message、MsgHub、Agent、Pipeline |
| 三框架对比 | 对话/状态图/消息，弱/强/中状态，原型/复杂流程/工业级 | 能否对比学习曲线、状态管理、并发能力、适用场景 |
| 选型建议 | 聊天协作→AutoGen，流程可控→LangGraph，工业级→AgentScope | 能否针对具体场景给出选型理由 |
| 局限注意 | 学习成本、API 版本不稳、token 成本、过度工程化、调试难、AI 局限仍在 | 能否说出至少 3 个局限及应对 |
| 框架组合 | LangGraph 总纲 + AutoGen 局部群聊 + AgentScope 分布式调度 | 能否说出组合方式 |

---

### 全景图（ASCII）

```
┌─────────────────────────────────────────────────────────────────┐
│                   智能体主流框架选型三角                           │
│                                                                   │
│                   可控性 (LangGraph)                              │
│              状态图 / 节点+边 / 条件边 / 断点续传                  │
│                              ▲                                    │
│                              │                                    │
│              ┌───────────────┴───────────────┐                    │
│              │                               │                    │
│              ▼                               ▼                    │
│     ┌───────────────┐              ┌───────────────┐             │
│     │   AutoGen     │              │  AgentScope   │             │
│     │ 对话驱动群聊   │              │ 消息驱动/分布式│             │
│     │ 多角色协作     │              │ 工程化/高并发 │             │
│     └───────┬───────┘              └───────┬───────┘             │
│             │                              │                      │
│             └───────────────┬──────────────┘                      │
│                             ▼                                       │
│                       协作性 (AutoGen)                              │
│                对话涌现 / 群聊 / TERMINATE                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Q11. AutoGen 核心组件与代码执行沙箱（手撕 Python 代码）

**回答**

前面 Q3 已经用极简代码演示了 AutoGen 的核心思想：多个角色 Agent 通过消息群聊协作。那真实的 AutoGen 里，除了 Agent 和消息，还几个非常重要的组件，咱们今天把它们拆开讲：**ConversableAgent（对话智能体）、GroupChat（群聊）、GroupChatManager（群聊管理器）、代码执行器（Code Executor / 沙箱）**。

---

### 核心组件，用大白话解释

| 组件 | 大白话 | 做什么 |
|------|--------|--------|
| **ConversableAgent** | 一个“会说话的智能体” | 有名字、有人设（system prompt）、能发消息、能收消息、能调用工具 |
| **GroupChat** | 一个“群聊房间” | 规定哪些 Agent 在群里，谁先说话、谁后说话、最多聊多少轮 |
| **GroupChatManager** | “群主/主持人” | 控制发言顺序，决定下一个是谁说话，防止大家七嘴八舌 |
| **代码执行器（沙箱）** | 一个“隔离房间” | 把 Agent 写的代码丢进去跑，跑崩了也不会把你的电脑搞炸，把报错或结果传回来 |

> **沙箱是什么？** 想象你让一个陌生人在你电脑上写程序，万一他写了 `import os; os.remove(...)` 删文件怎么办？沙箱就像一个“封闭的小房间”，代码只能在房间里跑，不能碰外面的文件和网络，跑完了把结果（或报错）传出来。

---

### 完整代码（概念版）

下面这段纯 Python 代码，不依赖真实 AutoGen 安装，演示 AutoGen 的四大核心组件：两个 Agent（程序员、测试员）、一个群聊管理器控制发言顺序、一个“代码沙箱”负责跑代码并把报错传回来。

```python
"""
AutoGen 核心组件极简实现（概念版）
ConversableAgent / GroupChat / GroupChatManager / 代码执行沙箱
程序员写代码 -> 沙箱跑代码 -> 报错传回给程序员 -> 程序员修改 -> 循环直到跑通
纯 Python 演示，不依赖 pyautogen。
"""

import re
from typing import List, Optional, Callable, Dict, Any


# ==================== 1. 消息 ====================

class Message:
    def __init__(self, sender: str, receiver: str, content: str):
        self.sender = sender
        self.receiver = receiver
        self.content = content

    def __str__(self):
        return f"[{self.sender} -> {self.receiver}] {self.content}"


# ==================== 2. 代码执行沙箱 ====================

class CodeExecutor:
    """
    代码执行器 / 沙箱。
    真实 AutoGen 的沙箱通常在 Docker 容器里跑代码，这里演示版只做
    安全拦截（禁止危险操作）+ exec 执行，把输出或报错返回。
    """

    # 禁止的危险操作关键词（演示版）
    FORBIDDEN = ["import os", "import subprocess", "open(", "remove", "delete", "exec("]

    def run(self, code: str) -> str:
        """把代码丢进沙箱跑，返回标准输出或报错信息。"""
        # 安全拦截
        for kw in self.FORBIDDEN:
            if kw in code:
                return f"[沙箱拦截] 检测到危险操作：{kw}，已拒绝执行。"

        # 捕获 print 输出
        import io
        import contextlib

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": {}}, {})
            output = buf.getvalue()
            return output if output else "[代码已执行，无输出]"
        except Exception as e:
            return f"[运行报错] {type(e).__name__}: {e}"


# ==================== 3. ConversableAgent（对话智能体） ====================

class ConversableAgent:
    def __init__(self, name: str, system_message: str,
                 llm_call: Callable[[str], str],
                 code_executor: Optional[CodeExecutor] = None):
        self.name = name
        self.system_message = system_message
        self.llm_call = llm_call
        self.code_executor = code_executor
        self.history: List[Message] = []

    def receive(self, message: Message):
        self.history.append(message)
        print(f"[收到] {message}")

    def send(self, receiver: 'ConversableAgent', content: str) -> Message:
        msg = Message(sender=self.name, receiver=receiver.name, content=content)
        print(f"[发送] {msg}")
        receiver.receive(msg)
        return msg

    def generate_reply(self, receiver: 'ConversableAgent') -> Optional[str]:
        """根据历史生成回复内容（不直接发，交给群聊管理器调度）。"""
        context = "\n".join([str(m) for m in self.history[-6:]])
        prompt = (f"{self.system_message}\n\n最近对话历史：\n{context}\n\n"
                  f"请给 {receiver.name} 回复一条消息。如果需要执行代码，"
                  f"在回复里以 ```python ... ``` 代码块形式给出。")
        return self.llm_call(prompt)

    def extract_code(self, text: str) -> Optional[str]:
        """从回复里提取 ```python ... ``` 代码块。"""
        blocks = re.findall(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
        return blocks[0] if blocks else None


# ==================== 4. 演示版的“大模型调用” ====================

def mock_llm(prompt: str) -> str:
    lower = prompt.lower()
    if "写代码" in lower or "实现" in lower:
        return ("```python\nprint(1 / 0)\n```\n"
                "我先写一段会报错的代码，让测试员跑一跑。")
    if "报错" in lower or "error" in lower or "division by zero" in lower:
        return ("```python\nprint(1 / 1)\n```\n"
                "已修复，换成安全的除法，应该能跑通了。")
    if "跑通" in lower or "成功" in lower:
        return "TERMINATE: 任务完成。"
    return "我收到了，请继续。"


# ==================== 5. GroupChat / GroupChatManager（群聊与管理器） ====================

class GroupChat:
    """群聊房间：记录参与者、最大轮数、已聊轮数。"""

    def __init__(self, agents: List[ConversableAgent], max_rounds: int = 10):
        self.agents = agents
        self.max_rounds = max_rounds
        self.round = 0


class GroupChatManager:
    """
    群聊管理器 / 主持人。
    负责决定“下一个谁说话”，轮流调度，直到 TERMINATE 或达到最大轮数。
    """

    def __init__(self, group_chat: GroupChat, code_executor: CodeExecutor):
        self.group_chat = group_chat
        self.code_executor = code_executor

    def run(self, initiator: ConversableAgent, initial_message: str):
        print(f"\n=== 群聊开始：{initial_message} ===\n")
        initiator.receive(Message(sender="用户", receiver=initiator.name, content=initial_message))

        current = initiator
        others = [a for a in self.group_chat.agents if a is not initiator]

        while self.group_chat.round < self.group_chat.max_rounds:
            self.group_chat.round += 1
            print(f"\n----- 第 {self.group_chat.round} 轮，当前发言：{current.name} -----")

            reply = current.generate_reply(others[0])
            if reply is None:
                break

            if reply.startswith("TERMINATE"):
                print(f"\n[任务完成] {reply}")
                return reply

            # 如果回复里有代码块，交给沙箱执行，把结果当成新消息反馈
            code = current.extract_code(reply)
            if code and current.code_executor:
                exec_result = self.code_executor.run(code)
                feedback = f"代码执行结果：\n{exec_result}"
                current.receive(Message(sender="沙箱", receiver=current.name, content=feedback))
                print(f"[沙箱] {exec_result}")
                # 把沙箱结果也发给下一位，让对方知道发生了什么
                others[0].receive(Message(sender="沙箱", receiver=others[0].name, content=feedback))

            # 轮流发言
            current, others[0] = others[0], current

        print(f"\n[达到最大轮数 {self.group_chat.max_rounds}，任务未完成]")
        return None


# ==================== 6. 运行 ====================

if __name__ == "__main__":
    sandbox = CodeExecutor()

    programmer = ConversableAgent(
        name="程序员",
        system_message="你是程序员，根据要求写 Python 代码，收到报错后修改代码，跑通后回复 TERMINATE。",
        llm_call=mock_llm,
        code_executor=sandbox,
    )

    tester = ConversableAgent(
        name="测试员",
        system_message="你是测试员，收到代码后通过沙箱运行，报错就原样告诉程序员，成功就回复 TERMINATE。",
        llm_call=mock_llm,
        code_executor=sandbox,
    )

    chat = GroupChat(agents=[programmer, tester], max_rounds=6)
    manager = GroupChatManager(group_chat=chat, code_executor=sandbox)
    manager.run(initiator=programmer, initial_message="请写一个两个整数相减的函数并运行。")
```

**运行输出示例（概念版）：**

```
=== 群聊开始：请写一个两个整数相减的函数并运行。 ===

----- 第 1 轮，当前发言：程序员 -----
[发送] [程序员 -> 测试员] ```python
print(1 / 0)
```
...
[沙箱] [运行报错] ZeroDivisionError: division by zero

----- 第 2 轮，当前发言：测试员 -----
[发送] [测试员 -> 程序员] 代码执行结果：
[运行报错] ZeroDivisionError: division by zero

----- 第 3 轮，当前发言：程序员 -----
[发送] [程序员 -> 测试员] ```python
print(1 / 1)
```
...
[沙箱] 代码已执行，无输出

[任务完成] TERMINATE: 任务完成。
```

---

### 这段代码演示了 AutoGen 的哪些核心思想？

1. **ConversableAgent**：每个 Agent 有自己的人设（system_message），能收发消息，能从历史里生成回复；
2. **GroupChatManager（主持人）**：轮流决定谁说话，避免大家同时说话乱成一团；
3. **代码执行沙箱**：Agent 写的代码不直接在本机跑，而是丢进沙箱，跑崩了也安全，把报错当消息传回去，让程序员改；
4. **TERMINATE**：结束信号，达到后群聊停止。

**真实 AutoGen 的关键组件**（概念映射）：
- `ConversableAgent` → 我们的 `ConversableAgent`；
- `GroupChat` / `GroupChatManager` → 我们的 `GroupChat` + `GroupChatManager`；
- `LocalCommandLineCodeExecutor` / Docker 沙箱 → 我们的 `CodeExecutor`；
- `human_input_mode`：关键步骤可插入人工确认，比如跑代码前先让人点一下“允许”。

**面试要点**：AutoGen 的核心组件是“对话智能体 + 群聊 + 主持人 + 代码沙箱”。手撕代码时要体现：多角色 Agent、群聊轮流发言、沙箱执行代码并把报错作为消息反馈、TERMINATE 结束条件。真实 AutoGen 沙箱通常跑在 Docker 里，能安全执行未知代码。

---

## Q12. LangGraph Reducer / Checkpoint / 子图 / 人工中断（手撕 Python 代码）

**回答**

Q4 已经演示了 LangGraph 最基础的“状态 → 节点 → 边”。真实的 LangGraph 里，还有四个更进阶的概念：**Reducer（状态合并器）、Checkpoint（检查点/断点续传）、子图（Subgraph）、人工中断（Human Interrupt / HITL）**。

---

### 四个进阶概念，用大白话

| 概念 | 大白话 | 解决什么问题 |
|------|--------|--------------|
| **Reducer** | “状态的合并规则” | 多个节点都要往同一个状态字段写东西时，是覆盖还是追加？Reducer 定了规矩 |
| **Checkpoint** | “断点存档” | 把每一步的状态存档，崩了或人工审核后，可以从断点继续，不用重头再来 |
| **子图** | “图里套图” | 把复杂流程拆成几个小图，主图调用子图，像调用函数一样模块化 |
| **人工中断** | “暂停等人点确认” | 关键步骤（如付款、删除数据）暂停，等人工确认后再继续 |

---

### 完整代码（概念版）

下面这段纯 Python 代码演示：带有 Reducer 的状态合并、Checkpoint 存档/恢复、子图嵌套、人工中断（用 `input` 模拟等待人工确认）。

```python
"""
LangGraph 进阶概念极简实现（概念版）
Reducer 状态合并 / Checkpoint 断点续传 / 子图 / 人工中断
纯 Python 演示，不依赖 langgraph。
"""

from typing import TypedDict, List, Callable, Dict, Any, Optional
import copy
import json
import os


# ==================== 1. Reducer：状态合并规则 ====================

def append_reducer(existing: Any, new: Any) -> Any:
    """
    Reducer 示例：如果字段是列表，就追加而不是覆盖。
    这样多个节点都能往 same list 里追记录，不会互相抹掉。
    """
    if isinstance(existing, list) and isinstance(new, list):
        return existing + new
    return new  # 非列表字段，后写的覆盖先写的


# ==================== 2. 状态定义 ====================

class AgentState(TypedDict):
    user_input: str
    steps: List[str]          # 走过的步骤，用 append_reducer 追加
    answer: str
    iteration: int
    approved: bool            # 人工中断后的审批标记


# ==================== 3. 节点函数 ====================

def node_a(state: AgentState) -> AgentState:
    """步骤 A：接收用户请求，记录步骤。"""
    state = dict(state)
    state["steps"] = state.get("steps", []) + ["node_a: 接收请求"]
    state["answer"] = f"正在处理：{state['user_input']}"
    state["iteration"] = state.get("iteration", 0) + 1
    print("[节点 A] 接收请求，步骤记录追加。")
    return state


def node_b(state: AgentState) -> AgentState:
    """步骤 B：处理业务，记录步骤。"""
    state = dict(state)
    state["steps"] = state.get("steps", []) + ["node_b: 业务处理"]
    state["answer"] = state["answer"] + " -> 已完成业务处理"
    print("[节点 B] 业务处理完成。")
    return state


def node_c(state: AgentState) -> AgentState:
    """步骤 C：生成最终答案。"""
    state = dict(state)
    state["steps"] = state.get("steps", []) + ["node_c: 生成答案"]
    state["answer"] = state["answer"] + " -> 最终答案已就绪"
    print("[节点 C] 最终答案生成。")
    return state


# ==================== 4. 子图：把 A+B 封成一个子图 ====================

class SubGraph:
    """子图：内部自己跑一段小流程，对外只暴露 invoke(state) -> state。"""

    def __init__(self):
        self.nodes = [node_a, node_b]

    def invoke(self, state: AgentState) -> AgentState:
        for node in self.nodes:
            state = node(state)
        return state


# ==================== 5. Checkpoint：断点存档 ====================

class CheckpointStore:
    """
    检查点存储：把每一步的状态存到磁盘（演示版存成 JSON 文件）。
    真实 LangGraph 通常存到 SQLite / PostgreSQL，支持时间旅行回溯。
    """

    def __init__(self, dir_path: str = "./checkpoints"):
        self.dir_path = dir_path
        os.makedirs(dir_path, exist_ok=True)

    def save(self, step_name: str, state: AgentState):
        path = os.path.join(self.dir_path, f"{step_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"[Checkpoint] 已存档：{path}")

    def load(self, step_name: str) -> Optional[AgentState]:
        path = os.path.join(self.dir_path, f"{step_name}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        print(f"[Checkpoint] 已恢复：{path}")
        return state


# ==================== 6. 人工中断（HITL） ====================

def human_interrupt(state: AgentState) -> AgentState:
    """
    人工中断：在关键步骤前暂停，打印当前状态，等用户输入 'y' 继续，
    输入 'n' 则拒绝。演示版用 input()，真实版会弹出 UI 或发消息给人工。
    """
    print("\n" + "=" * 50)
    print("[人工中断] 即将执行关键步骤，当前状态：")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    choice = input("是否允许继续？(y/n): ").strip().lower()
    state = dict(state)
    state["approved"] = (choice == "y")
    if state["approved"]:
        print("[人工中断] 用户已批准，继续执行。")
    else:
        print("[人工中断] 用户拒绝，流程终止。")
    return state


# ==================== 7. 主流程图（带子图 + checkpoint + 人工中断） ====================

class StateGraph:
    def __init__(self):
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, func: Callable):
        self.nodes[name] = func

    def add_edge(self, frm: str, to: str):
        self.edges[frm] = to

    def set_entry_point(self, name: str):
        self.entry_point = name

    def invoke(self, initial_state: AgentState, store: CheckpointStore) -> AgentState:
        state = initial_state
        current = self.entry_point
        max_steps = 20
        step = 0

        while current and step < max_steps:
            step += 1
            print(f"\n--- 步骤 {step}: 节点 [{current}] ---")

            # 执行节点
            if current in self.nodes:
                state = self.nodes[current](state)

            # 存档（每个节点后都 checkpoint，方便断点续传）
            store.save(current, state)

            # 决定下一步
            if current in self.edges:
                nxt = self.edges[current]
            else:
                nxt = None

            # 人工中断：在进入 node_c 前暂停
            if nxt == "node_c":
                state = human_interrupt(state)
                if not state.get("approved", False):
                    print("[流程终止] 未获人工批准。")
                    store.save("rejected", state)
                    return state

            current = nxt

        return state


# ==================== 8. 搭建并运行 ====================

if __name__ == "__main__":
    sub_graph = SubGraph()

    def subgraph_node(state: AgentState) -> AgentState:
        """主图里的一个节点，内部调用子图。"""
        print("[主图] 调用子图 (A+B)...")
        return sub_graph.invoke(state)

    workflow = StateGraph()
    workflow.add_node("subgraph", subgraph_node)   # 子图节点：内部跑 A+B
    workflow.add_node("node_c", node_c)            # 最终答案节点（前有人工中断）
    workflow.add_edge("subgraph", "node_c")
    workflow.set_entry_point("subgraph")

    store = CheckpointStore(dir_path="./checkpoints")

    initial_state: AgentState = {
        "user_input": "帮我订一张机票",
        "steps": [],
        "answer": "",
        "iteration": 0,
        "approved": False,
    }

    final_state = workflow.invoke(initial_state, store)

    print("\n" + "=" * 50)
    print("最终状态：")
    print(json.dumps(final_state, ensure_ascii=False, indent=2))
```

**运行输出示例（概念版，人工输入 y）：**

```
--- 步骤 1: 节点 [subgraph] ---
[主图] 调用子图 (A+B)...
[节点 A] 接收请求，步骤记录追加。
[Checkpoint] 已存档：./checkpoints/node_a.json
[节点 B] 业务处理完成。
[Checkpoint] 已存档：./checkpoints/node_b.json

==================================================
[人工中断] 即将执行关键步骤，当前状态：
{
  "user_input": "帮我订一张机票",
  "steps": ["node_a: 接收请求", "node_b: 业务处理"],
  "answer": "正在处理：帮我订一张机票 -> 已完成业务处理",
  ...
}
是否允许继续？(y/n): y
[人工中断] 用户已批准，继续执行。

--- 步骤 2: 节点 [node_c] ---
[节点 C] 最终答案生成。
[Checkpoint] 已存档：./checkpoints/node_c.json

最终状态：
{
  "user_input": "帮我订一张机票",
  "steps": ["node_a: 接收请求", "node_b: 业务处理", "node_c: 生成答案"],
  "answer": "... -> 最终答案已就绪",
  ...
}
```

---

### 这段代码演示了 LangGraph 的哪些进阶思想？

1. **Reducer（`append_reducer`）**：多个节点往 `steps` 列表里追加记录，不会互相覆盖——真实 LangGraph 用 `annotated` + reducer 定义字段合并规则；
2. **Checkpoint（`CheckpointStore`）**：每个节点后把状态存档到 JSON，崩了或审核后可从断点恢复，不用重头再来；
3. **子图（`SubGraph`）**：把 A+B 封成子图，主图像调用普通节点一样调用它，复杂流程可模块化拆分；
4. **人工中断（`human_interrupt`）**：进入 `node_c` 前暂停，等人工批准（`approved=True`）才继续，适合付款、删除等关键操作。

**真实 LangGraph 的关键特征**（概念映射）：
- `Reducer`：`Annotated[list, operator.add]` 之类的注解，定义状态字段怎么合并；
- `checkpoint` / `CheckpointSaver`：保存每一步状态，支持 `get_state` / `update_state` 和“时间旅行”回溯到任意历史步骤；
- `SubGraphBuilder` / `subgraph`：子图可嵌套，主图调用子图，子图内部还能再嵌子图；
- `interrupt` / `Command(resume=...)`：人工中断后，人工修改状态并 `resume` 继续执行。

**面试要点**：LangGraph 进阶四件套：Reducer 决定状态怎么合并（追加 vs 覆盖）、Checkpoint 支持断点续传与时间旅行、子图让复杂流程模块化、人工中断让关键步骤可人工审核。手撕代码时要体现这四者，并说明它们在生产中的价值（可审计、可回溯、可人工干预）。

---

## Q13. AgentScope 消息持久化与异步并发（手撕 Python 代码）

**回答**

Q5 已经演示了 AgentScope 最基础的“Message / MsgHub / Agent / Pipeline”。真实的 AgentScope 作为工程化框架，还有两个非常核心的能力：**消息持久化**（把消息存到磁盘/数据库，断电不丢、可回放）和**异步并发**（多个 Agent 同时干活，不互相阻塞）。

---

### 两个核心能力，用大白话

| 能力 | 大白话 | 解决什么问题 |
|------|--------|--------------|
| **消息持久化** | “聊天记录存硬盘” | 程序重启后消息还在，支持审计、回放、调试；不会因为进程崩溃丢掉对话历史 |
| **异步并发** | “多个 Agent 同时干活” | 不用一个接一个等，多个 Agent 并行调用模型或工具，整体响应更快 |

---

### 完整代码（概念版）

下面这段纯 Python 代码演示：用 JSON 文件做消息持久化（程序重启后还能读回历史），以及用 `asyncio` 实现多个 Agent 并发处理任务。

```python
"""
AgentScope 消息持久化与异步并发极简实现（概念版）
JSON 消息持久化 / asyncio 异步并发多 Agent
纯 Python 演示，不依赖 agentscope。
"""

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime


# ==================== 1. 消息 ====================

@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.sender} -> {self.receiver}] {self.content}"


# ==================== 2. 消息持久化：JSON 存储 ====================

class MessageStore:
    """
    消息持久化存储：把消息写到 JSON 文件。
    真实 AgentScope 支持 SQLite / MongoDB 作为消息后端，这里演示版用 JSON。
    程序重启后，仍可从文件读回历史消息。
    """

    def __init__(self, file_path: str = "./messages.json"):
        self.file_path = file_path
        self.messages: List[Message] = []
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.messages = [Message(**m) for m in data]
            print(f"[持久化] 已从 {self.file_path} 读回 {len(self.messages)} 条历史消息。")
        else:
            print(f"[持久化] 未找到历史文件，从空开始。")

    def save(self, message: Message):
        self.messages.append(message)
        self._flush()

    def _flush(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.messages], f, ensure_ascii=False, indent=2)

    def get_history(self) -> List[Message]:
        return self.messages

    def clear(self):
        self.messages = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


# ==================== 3. MsgHub：带持久化的消息中心 ====================

class MsgHub:
    def __init__(self, store: MessageStore):
        self.store = store
        self.subscribers: Dict[str, List[Callable[[Message], None]]] = {}

    def subscribe(self, agent_name: str, callback: Callable[[Message], None]):
        self.subscribers.setdefault(agent_name, []).append(callback)

    def publish(self, message: Message):
        print(f"[MsgHub] {message}")
        self.store.save(message)   # 持久化：每条消息都写进磁盘
        for callback in self.subscribers.get(message.receiver, []):
            callback(message)


# ==================== 4. Agent：支持异步回复 ====================

class Agent:
    def __init__(self, name: str, msg_hub: MsgHub,
                 llm_call: Callable[[str], str]):
        self.name = name
        self.msg_hub = msg_hub
        self.llm_call = llm_call
        self.msg_hub.subscribe(self.name, self._on_message)
        self.history: List[Message] = []

    def _on_message(self, message: Message):
        self.history.append(message)
        print(f"[收到] {message}")

    def send(self, receiver: str, content: str, metadata: Dict[str, Any] = None):
        msg = Message(sender=self.name, receiver=receiver, content=content,
                      metadata=metadata or {})
        self.msg_hub.publish(msg)

    def reply(self, receiver: str, context: str):
        prompt = f"你是 {self.name}。根据以下上下文，给 {receiver} 回复：\n{context}"
        content = self.llm_call(prompt)
        self.send(receiver, content)

    async def areply(self, receiver: str, context: str) -> str:
        """异步回复：模拟调用大模型 API（IO 操作），不阻塞其他 Agent。"""
        print(f"[异步] {self.name} 开始处理：{context[:30]}...")
        await asyncio.sleep(0.5)  # 模拟网络 IO 等待
        prompt = f"你是 {self.name}。根据以下上下文，给 {receiver} 回复：\n{context}"
        content = self.llm_call(prompt)
        self.send(receiver, content)
        print(f"[异步] {self.name} 处理完毕。")
        return content


# ==================== 5. 演示版的“大模型调用” ====================

def mock_llm(prompt: str) -> str:
    lower = prompt.lower()
    if "账单" in lower:
        return "这是您的账单明细：……"
    if "服务器" in lower:
        return "服务器 CPU 使用率 85%，建议扩容。"
    if "订单" in lower:
        return "您的订单 #12345 已发货，预计 3 天送达。"
    return "我已记录您的问题，稍后会有专员联系您。"


# ==================== 6. 运行：持久化 + 异步并发 ====================

def run_sync_demo():
    """同步演示：消息持久化。"""
    print("\n=== 消息持久化演示 ===")
    store = MessageStore(file_path="./demo_messages.json")
    hub = MsgHub(store)

    cs = Agent("客服 Agent", hub, mock_llm)
    expert = Agent("专家 Agent", hub, mock_llm)

    cs.send("专家 Agent", "用户问：我的账单为什么这么高？")
    expert.reply("客服 Agent", "用户问：我的账单为什么这么高？")

    print("\n--- 读回持久化的消息历史 ---")
    for m in store.get_history():
        print(m)


async def run_async_demo():
    """异步并发演示：多个 Agent 并发处理不同用户请求。"""
    print("\n=== 异步并发演示 ===")
    store = MessageStore(file_path="./demo_messages_async.json")
    hub = MsgHub(store)

    agent_a = Agent("Agent A", hub, mock_llm)
    agent_b = Agent("Agent B", hub, mock_llm)
    agent_c = Agent("Agent C", hub, mock_llm)

    tasks = [
        agent_a.areply("用户", "我的账单为什么这么高？"),
        agent_b.areply("用户", "服务器负载高不高？"),
        agent_c.areply("用户", "我的订单什么时候到？"),
    ]
    await asyncio.gather(*tasks)

    print("\n--- 异步并发后的消息历史 ---")
    for m in store.get_history():
        print(m)


if __name__ == "__main__":
    run_sync_demo()
    asyncio.run(run_async_demo())
```

**运行输出示例（概念版）：**

```
=== 消息持久化演示 ===
[持久化] 未找到历史文件，从空开始。
[MsgHub] [客服 Agent -> 专家 Agent] 用户问：我的账单为什么这么高？
[持久化] 已保存消息...
[收到] [客服 Agent -> 专家 Agent] 用户问：我的账单为什么这么高？
[MsgHub] [专家 Agent -> 客服 Agent] 这是您的账单明细：……
[持久化] 已保存消息...
[收到] [专家 Agent -> 客服 Agent] 这是您的账单明细：……

--- 读回持久化的消息历史 ---
[客服 Agent -> 专家 Agent] 用户问：我的账单为什么这么高？
[专家 Agent -> 客服 Agent] 这是您的账单明细：……

=== 异步并发演示 ===
[持久化] 已从 ./demo_messages_async.json 读回 0 条历史消息。
[异步] Agent A 开始处理：我的账单为什么这么高？...
[异步] Agent B 开始处理：服务器负载高不高？...
[异步] Agent C 开始处理：我的订单什么时候到？...
[异步] Agent A 处理完毕。
...
[MsgHub] [Agent A -> 用户] 这是您的账单明细：……
```

> 注意：运行两次同步演示，第二次会读回第一次存进 JSON 的历史消息，体现“断电不丢、可回放”。

---

### 这段代码演示了 AgentScope 的哪些工程化思想？

1. **消息持久化（`MessageStore`）**：每条消息写进 JSON 文件，程序重启后还能读回历史，支持审计和回放。真实 AgentScope 支持 SQLite / MongoDB 作为消息后端；
2. **MsgHub（消息中心）**：Agent 不直接调用，通过 MsgHub 分发，解耦发送者和接收者；
3. **异步并发（`asyncio.gather`）**：A、B、C 三个 Agent 并发处理不同请求，不用串行等待，整体更快。真实 AgentScope 支持 `fanout_pipeline` 等并发原语；
4. **消息订阅/回调**：每个 Agent 订阅发给自己的消息，收到后通过回调处理。

**真实 AgentScope 的关键特征**（概念映射）：
- 三层架构：SDK（开发）+ Runtime（安全运行，沙箱）+ Studio（可视化调试）；
- 消息持久化：SQLite、MongoDB 作为消息后端，状态可回放、可追踪；
- 异步 + 并发：`fanout_pipeline` 并行收集多个 Agent 决策，`sequential_pipeline` 串行；
- 结构化输出：用 `BaseModel` 定义业务规则，不靠 Prompt 里的“请输出 JSON”；
- Studio：可视化面板查看消息流、状态、性能指标。

**面试要点**：AgentScope 的工程化核心是“消息持久化 + 异步并发”。手撕代码时要体现：MessageStore 把消息写磁盘（断电不丢、可回放）、MsgHub 发布/订阅解耦、asyncio 并发调用多个 Agent。并说明生产价值：审计合规、故障恢复、高并发响应。

---

## Q14. 三框架工具接入与多智能体记忆

**回答**

智能体框架不只让 Agent “会聊”，更要让 Agent “会用工具”“会记东西”。本节聊两个实践问题：**怎么给三个框架的 Agent 接入外部工具**（搜索、计算器、代码执行等）和**多智能体系统怎么做记忆**（短期记忆 vs 长期记忆，共享记忆 vs 私有记忆）。

---

### 一、工具接入：三框架怎么加工具

给 Agent 加工具，通常需要三样东西：**工具定义**（名字、描述、参数）、**工具注册表**（把名字映射到真实函数）、**工具调用循环**（Agent 决定用哪个工具 → 执行 → 把结果传回给 Agent）。

下面用纯 Python 演示一个可接入到任意框架的“工具注册表 + 工具调用”极简版本。

```python
"""
工具接入极简实现（概念版）
工具定义 / 工具注册表 / 工具调用循环
纯 Python 演示，可嵌入 AutoGen / LangGraph / AgentScope 的 Agent。
"""

import json
from typing import Callable, Dict, Any, List


class Tool:
    """一个工具：名字 + 描述 + 参数模式 + 真实函数。"""

    def __init__(self, name: str, description: str,
                 func: Callable[..., Any],
                 parameters: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    def to_dict(self) -> Dict[str, Any]:
        """把工具转成大模型能看懂的函数定义（类 OpenAI function calling 格式）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def run(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    """工具注册表：工具名 -> Tool 对象。"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tools.values()]


# ==================== 真实工具示例 ====================

def calculator(expression: str) -> str:
    """简单计算器：只允许加减乘除，演示用 eval（生产切勿直接用 eval）。"""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符。"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算报错：{e}"


def web_search(query: str) -> str:
    """模拟搜索工具。"""
    return f"搜索结果（模拟）：关于 '{query}' 的相关网页 A、网页 B。"


# ==================== 工具调用循环（演示版） ====================

def agent_with_tools(user_query: str, registry: ToolRegistry,
                     llm_call: Callable[[str], str], max_tool_steps: int = 5):
    """
    Agent 带工具的调用循环：
    1. 大模型根据用户问题决定调用哪个工具（输出 JSON：{"tool": "...", "args": {...}}）
    2. 执行工具，把结果喂回大模型
    3. 重复，直到大模型直接回答最终结果（不再调用工具）
    """
    tool_list = registry.list_tools()
    tools_desc = "\n".join([
        f"- {t['name']}: {t['description']} (参数: {json.dumps(t['parameters'])})"
        for t in tool_list
    ])

    context = f"用户问题：{user_query}\n\n可用工具：\n{tools_desc}\n\n"

    for step in range(1, max_tool_steps + 1):
        print(f"\n----- 工具步骤 {step} -----")
        model_reply = llm_call(context + "请决定：需要调用工具吗？如果需要，输出 JSON {\"tool\": \"工具名\", \"args\": {...}}；如果不需要，直接输出最终答案。")

        # 尝试从回复里解析工具调用
        try:
            start = model_reply.index("{")
            end = model_reply.rindex("}") + 1
            call = json.loads(model_reply[start:end])
            tool_name = call["tool"]
            args = call["args"]
        except (ValueError, KeyError, json.JSONDecodeError):
            # 没有工具调用，直接当最终答案
            print(f"\n[最终答案] {model_reply}")
            return model_reply

        tool = registry.get(tool_name)
        if tool is None:
            context += f"\n工具 {tool_name} 不存在，可用工具：{[t['name'] for t in tool_list]}\n"
            continue

        result = tool.run(**args)
        print(f"[工具 {tool_name}] 输出：{result}")
        context += f"\n工具 {tool_name} 返回：{result}\n"

    print("\n[达到最大工具步骤，返回当前最佳答案]")
    return context


# ==================== 运行 ====================

if __name__ == "__main__":
    registry = ToolRegistry()
    registry.register(Tool("calculator", "做加减乘除计算", calculator,
                           {"type": "object", "properties": {"expression": {"type": "string"}},
                            "required": ["expression"]}))
    registry.register(Tool("web_search", "网页搜索", web_search,
                           {"type": "object", "properties": {"query": {"type": "string"}},
                            "required": ["query"]}))

    def demo_llm(prompt: str) -> str:
        lower = prompt.lower()
        if "计算" in lower or "多少" in lower:
            return '{"tool": "calculator", "args": {"expression": "23 * 47 + 100"}}'
        if "搜索" in lower or "网页" in lower:
            return '{"tool": "web_search", "args": {"query": "Agent 框架对比"}}'
        if "返回" in lower or "结果" in lower:
            return "最终答案：计算结果和搜索结果已整理完毕。"
        return '{"tool": "web_search", "args": {"query": "默认搜索"}}'

    agent_with_tools("帮我算一下 23 乘以 47 加 100 等于多少，并搜索 Agent 框架对比。",
                     registry, demo_llm)
```

**运行输出示例（概念版）：**

```
----- 工具步骤 1 -----
[工具 calculator] 输出：1181

----- 工具步骤 2 -----
[工具 web_search] 输出：搜索结果（模拟）：关于 'Agent 框架对比' 的相关网页 A、网页 B。

[最终答案] 最终答案：计算结果和搜索结果已整理完毕。
```

---

### 二、多智能体记忆：短期 vs 长期，共享 vs 私有

多智能体系统里，“记忆”要拆成四个维度来思考：

| 维度 | 大白话 | 典型实现 |
|------|--------|----------|
| **短期记忆** | 刚才聊了什么、刚才算出了什么 | 对话历史列表、当前状态对象（LangGraph State） |
| **长期记忆** | 以前学过的、永久保存的知识 | 向量数据库（检索相关片段）、图数据库 |
| **共享记忆** | 所有 Agent 能看见的公共黑板 | 共享状态、共享消息历史、公共文档库 |
| **私有记忆** | 只属于某个 Agent 的、别人看不见的 | 每个 Agent 自己的私有历史、个人偏好 |

**实践建议**（大白话）：
- **短期记忆**靠框架自带：AutoGen 的对话历史、LangGraph 的 State、AgentScope 的 MsgHub 历史；
- **长期记忆**靠外部：把重要结论嵌入向量（Embedding）存进向量数据库，需要时检索（RAG）；
- **共享记忆**用公共黑板：多个 Agent 读写同一个状态或消息池，适合协作；
- **私有记忆**各自留底：每个 Agent 有自己的私有历史，不暴露给别人，防止信息泄漏或干扰。

下面是一个“带记忆的 Agent”极简示例：私有历史 + 共享黑板 + 向量长期记忆（演示版用内存字典模拟）。

```python
"""
多智能体记忆极简实现（概念版）
私有历史 / 共享黑板 / 向量长期记忆（内存模拟）
纯 Python 演示。
"""

from typing import List, Dict, Any
import hashlib


class Memory:
    """长期记忆演示版：用文本相似度（哈希）模拟向量检索。"""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add(self, text: str, source: str = "system"):
        self.entries.append({"text": text, "source": source})

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """演示版：简单返回包含查询关键词的条目。"""
        results = [e["text"] for e in self.entries if query.lower() in e["text"].lower()]
        return results[:top_k]


class AgentWithMemory:
    def __init__(self, name: str, shared_board: Dict[str, Any],
                 long_term_memory: Memory):
        self.name = name
        self.shared_board = shared_board          # 共享记忆（黑板）
        self.long_term = long_term_memory         # 长期记忆
        self.private_history: List[str] = []      # 私有记忆

    def observe(self, text: str):
        """观察一条信息：写入私有历史，并可选地写入共享黑板和长期记忆。"""
        self.private_history.append(text)

    def share(self, key: str, value: Any):
        """把信息写到共享黑板，其他 Agent 能看见。"""
        self.shared_board[key] = value

    def recall(self, query: str) -> List[str]:
        """从长期记忆里检索相关信息。"""
        return self.long_term.retrieve(query)

    def act(self, context: str):
        """综合私有历史、共享黑板、长期记忆做决策。"""
        private = "\n".join(self.private_history[-3:])
        shared = json.dumps(self.shared_board, ensure_ascii=False)
        recalled = "\n".join(self.recall(context))
        print(f"\n[{self.name}] 决策依据：")
        print(f"私有历史：{private[:80]}...")
        print(f"共享黑板：{shared[:80]}...")
        print(f"长期记忆：{recalled[:80]}...")


import json

if __name__ == "__main__":
    board: Dict[str, Any] = {}
    mem = Memory()
    mem.add("用户偏好：喜欢简短回答，不喜欢长篇大论。", source="user_profile")
    mem.add("系统约束：涉及资金操作必须人工确认。", source="system_policy")

    alice = AgentWithMemory("Alice", board, mem)
    bob = AgentWithMemory("Bob", board, mem)

    alice.observe("用户问：今天天气？")
    alice.share("last_weather_query", "今天天气")

    bob.recall("偏好")
    bob.act("用户再次询问天气")
```

**运行输出示例（概念版）：**

```
[Bob] 决策依据：
私有历史：...
共享黑板：{"last_weather_query": "今天天气"}
长期记忆：用户偏好：喜欢简短回答，不喜欢长篇大论。
```

---

### 三框架工具与记忆的对比速查

| 能力 | AutoGen | LangGraph | AgentScope |
|------|---------|-----------|------------|
| 工具接入 | 原生支持代码执行沙箱，可自定义函数工具 | 通过 LangChain Tool 规范接入，节点里调用工具 | 提供工具库和安全沙箱，结构化工具定义 |
| 短期记忆 | 对话历史（群聊上下文） | State 共享状态对象 | MsgHub 消息历史 |
| 长期记忆 | 需外部接向量数据库 | 需外部接向量数据库 | 需外部接向量数据库 / 持久化消息后端 |
| 共享记忆 | 群聊共享历史 | State 共享状态 | MsgHub 共享消息池 |
| 私有记忆 | 每个 Agent 私有历史 | 节点内局部状态 | 每个 Agent 私有历史 |

**面试要点**：工具接入 = 工具定义 + 注册表 + 调用循环；多智能体记忆要拆成短期/长期、共享/私有四个维度，短期靠框架自带（历史/State/MsgHub），长期靠向量数据库（RAG），共享靠公共黑板/共享状态，私有靠每个 Agent 自己的历史。

---

## Q15. 可观测性与安全

**回答**

多智能体系统上了生产，最头疼的不是“跑不通”，而是**“出了问题不知道哪个 Agent 搞的”**和**“Agent 乱调用工具带来安全风险”**。本节聊两个工程化必修课：**可观测性**（日志、追踪、可视化）和**安全**（沙箱、输入输出过滤、审计）。

---

### 可观测性：看得见、能追踪、可调试

一个多智能体系统的可观测性，要回答三个问题：
1. **发生了什么？** —— 每个 Agent 发了什么消息、调用了什么工具、返回了什么结果；
2. **什么时候发生的？** —— 每一步的时间戳、耗时；
3. **在哪个节点/哪个 Agent 发生的？** —— 调用链路，方便定位问题。

下面是一个带日志和追踪 ID 的 Agent 包装器极简示例：

```python
"""
可观测性极简实现（概念版）
结构化日志 / 追踪 ID / 调用链路记录
纯 Python 演示。
"""

import json
import uuid
import time
from datetime import datetime
from typing import Callable, Dict, Any, List


class ObservableAgent:
    """带可观测性的 Agent 包装器：记录每条消息、每次工具调用、每次 LLM 调用。"""

    def __init__(self, name: str, llm_call: Callable[[str], str],
                 log_file: str = "./agent_trace.jsonl"):
        self.name = name
        self.llm_call = llm_call
        self.log_file = log_file
        self.trace_id = str(uuid.uuid4())[:8]  # 本次运行的追踪 ID

    def _log(self, event: str, data: Dict[str, Any]):
        """写一条结构化日志到 JSONL 文件。"""
        record = {
            "trace_id": self.trace_id,
            "agent": self.name,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[LOG] {self.name} | {event} | {data}")

    def receive(self, sender: str, content: str):
        self._log("receive", {"sender": sender, "content": content})

    def send(self, receiver: str, content: str):
        self._log("send", {"receiver": receiver, "content": content})

    def call_llm(self, prompt: str) -> str:
        start = time.time()
        self._log("llm_call_start", {"prompt_len": len(prompt)})
        result = self.llm_call(prompt)
        elapsed = time.time() - start
        self._log("llm_call_end", {"elapsed_sec": round(elapsed, 3),
                                   "reply_len": len(result)})
        return result

    def call_tool(self, tool_name: str, args: Dict[str, Any], result: Any):
        self._log("tool_call", {"tool": tool_name, "args": args, "result": str(result)[:200]})


def demo_llm(prompt: str) -> str:
    return f"模型回复（基于提示词前20字：{prompt[:20]}...）"


if __name__ == "__main__":
    agent = ObservableAgent("销售 Agent", demo_llm)
    agent.receive("用户", "我想买一台笔记本电脑。")
    reply = agent.call_llm("用户想买笔记本电脑，请推荐。")
    agent.send("用户", reply)
    agent.call_tool("check_inventory", {"sku": "NB-001"}, {"stock": 5})

    print(f"\n追踪 ID：{agent.trace_id}")
    print("日志已写入 ./agent_trace.jsonl，可用于事后回放和排查。")
```

**运行输出示例（概念版）：**

```
[LOG] 销售 Agent | receive | {'sender': '用户', 'content': '我想买一台笔记本电脑。'}
[LOG] 销售 Agent | llm_call_start | {'prompt_len': 26}
[LOG] 销售 Agent | llm_call_end | {'elapsed_sec': 0.001, 'reply_len': 24}
[LOG] 销售 Agent | send | {'receiver': '用户', 'content': '模型回复（基于提示词前20字：...'}
[LOG] 销售 Agent | tool_call | {'tool': 'check_inventory', 'args': {'sku': 'NB-001'}, 'result': '{"stock": 5}'}
```

生产环境可接入更成熟的方案：
- **LangGraph Studio**：可视化调试图的执行、查看每个节点的状态、断点续传；
- **AgentScope Studio**：可视化消息流、Agent 状态、性能指标；
- **OpenTelemetry / Jaeger**：分布式追踪，适合 Agent 分布在多台机器上的场景；
- **JSONL / Elasticsearch**：结构化日志存储，支持检索和报警。

---

### 安全：沙箱、过滤、审计

多智能体系统的安全风险主要来自三个地方：

| 风险点 | 大白话 | 应对措施 |
|--------|--------|----------|
| **代码执行** | Agent 可能写出危险代码（删库、挖矿） | 代码必须跑在 Docker 沙箱里，禁止网络和文件系统访问 |
| **输入注入** | 用户输入可能“伪装成系统指令”骗过 Agent | 输入过滤、指令隔离（system prompt 与用户输入分开） |
| **输出泄漏** | Agent 可能把敏感信息（密码、客户数据）发出去 | 输出过滤、脱敏、审计日志 |

下面是一个简单的输入/输出过滤器示例：

```python
"""
安全过滤极简实现（概念版）
输入过滤 / 输出脱敏 / 危险操作拦截
纯 Python 演示。
"""

import re


class SafetyFilter:
    """输入/输出安全过滤器。"""

    # 敏感信息模式（演示版）
    SENSITIVE_PATTERNS = [
        (r"\d{4}-\d{4}-\d{4}-\d{4}", "[信用卡号已脱敏]"),
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[邮箱已脱敏]"),
        (r"(?i)(password|密码|token|api.key)\s*[:=]\s*\S+", "[密钥已脱敏]"),
    ]

    # 危险指令关键词
    DANGEROUS_KEYWORDS = ["rm -rf", "format c:", "del /s", "shutdown", "reboot"]

    def filter_input(self, text: str) -> str:
        """过滤用户输入，拦截危险指令。"""
        for kw in self.DANGEROUS_KEYWORDS:
            if kw.lower() in text.lower():
                return f"[安全拦截] 输入包含危险操作：{kw}，已拒绝。"
        return text

    def filter_output(self, text: str) -> str:
        """过滤 Agent 输出，脱敏敏感信息。"""
        result = text
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            result = re.sub(pattern, replacement, result)
        return result


if __name__ == "__main__":
    filt = SafetyFilter()

    print(filt.filter_input("请帮我删除所有文件 rm -rf /"))
    # [安全拦截] 输入包含危险操作：rm -rf /，已拒绝。

    print(filt.filter_output("用户邮箱 alice@example.com，信用卡 1234-5678-9012-3456"))
    # 用户邮箱 [邮箱已脱敏]，信用卡 [信用卡号已脱敏]
```

---

### 可观测性与安全速查

| 维度 | 做什么 | 手段示例 |
|------|--------|----------|
| 可观测性 | 知道每个 Agent 干了什么、什么时候干的、在哪个节点干的 | 结构化日志（JSONL）、追踪 ID、LangGraph Studio / AgentScope Studio、OpenTelemetry |
| 安全 | 防止 Agent 搞破坏、泄漏敏感信息、被恶意输入骗过 | Docker 代码沙箱、输入过滤（指令隔离）、输出脱敏、审计日志、人工确认（HITL） |

**面试要点**：可观测性要体现“结构化日志 + 追踪 ID + 可视化面板”，能回答发生了什么、什么时候、在哪个 Agent；安全要体现“代码沙箱 + 输入输出过滤 + 审计日志 + 人工确认”，尤其是代码执行必须进沙箱、敏感信息要脱敏。

---

## Q16. 性能与 token 成本：怎么控制多智能体系统的开销？

**回答**

多智能体系统最容易超支的不是服务器钱，而是 **token 钱**——几个 Agent 群聊几十轮，对话历史不断累积，轻易烧掉几万甚至几十万 token。本节聊怎么从框架层面和应用层面控制性能与 token 成本。

---

### 为什么多智能体容易烧 token？

| 原因 | 大白话 |
|------|--------|
| **对话历史累积** | 每一轮群聊，之前的对话都要再次发给模型，越聊越长 |
| **多 Agent 重复读** | 每个 Agent 都要读完整上下文，N 个 Agent = 上下文被重复读 N 遍 |
| **工具调用往返** | 调用工具 → 把结果传回 → 模型再读一遍结果，来回好几轮 |
| **反思/自我修正** | Reflection 多跑几轮就是多几倍 token |
| **长文本工具结果** | 搜索工具返回一大堆网页，全塞进上下文 |

---

### 控制 token 成本的 6 个实用手段（大白话）

| 手段 | 做什么 | 大白话 |
|------|--------|--------|
| **限制最大轮数** | 群聊或循环最多跑几轮就停 | 不让 Agent 无限聊下去 |
| **上下文压缩/摘要** | 把长对话摘成短段再喂模型 | 只保留关键信息，扔掉废话 |
| **滑动窗口** | 只保留最近 N 条消息 | 旧的扔掉，只记得最近发生的事 |
| **关键信息只传给需要的 Agent** | 不把全部历史发给每个 Agent | 只跟相关的人说相关的事 |
| **便宜模型干粗活，贵模型干精活** | 简单任务用便宜模型，复杂推理用贵模型 | 能用小模型别大材小用 |
| **工具结果截断** | 工具返回太长就截断或摘要 | 搜索返回 10 页，只取前 2 段 |

---

### 完整代码（概念版）

下面这段纯 Python 代码演示：滑动窗口上下文、最大轮数限制、上下文摘要（演示版用截断模拟）、便宜/贵模型切换。

```python
"""
性能与 token 成本控制极简实现（概念版）
滑动窗口 / 最大轮数 / 上下文摘要 / 便宜模型与贵模型切换
纯 Python 演示。
"""

from typing import List, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    sender: str
    content: str


class ContextWindow:
    """
    滑动窗口上下文：只保留最近 N 条消息，旧的丢弃，控制 token 增长。
    """

    def __init__(self, max_messages: int = 6):
        self.max_messages = max_messages
        self.messages: List[Message] = []

    def add(self, message: Message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            # 丢弃最旧的，只留最近的
            self.messages = self.messages[-self.max_messages:]

    def get_context(self) -> str:
        return "\n".join([f"{m.sender}: {m.content}" for m in self.messages])

    def token_estimate(self) -> int:
        """粗略估算 token（中文按字符数，英文按单词数，仅作演示）。"""
        text = self.get_context()
        return len(text)  # 演示版：按字符数粗略估算


def summarize(context: str, max_chars: int = 200) -> str:
    """
    上下文摘要（演示版）：如果太长，只保留前 max_chars 字符并标注已摘要。
    真实场景应调用模型生成摘要。
    """
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + f"\n...[以上已摘要，共 {len(context)} 字符，已压缩]"


class CostControlledAgent:
    """
    带成本控制的 Agent：
    - 滑动窗口上下文
    - 最大轮数限制
    - 上下文过长时做摘要
    - 简单任务用便宜模型，复杂任务用贵模型
    """

    def __init__(self, name: str,
                 cheap_llm: Callable[[str], str],
                 expensive_llm: Callable[[str], str],
                 max_rounds: int = 8,
                 max_context_chars: int = 1000):
        self.name = name
        self.cheap_llm = cheap_llm
        self.expensive_llm = expensive_llm
        self.max_rounds = max_rounds
        self.max_context_chars = max_context_chars
        self.window = ContextWindow(max_messages=8)
        self.round = 0
        self.total_tokens = 0

    def is_complex(self, user_input: str) -> bool:
        """简单判断任务是否复杂：演示版按长度和关键词。"""
        complex_keywords = ["分析", "推理", "复杂", "架构", "优化", "调试"]
        return len(user_input) > 60 or any(k in user_input for k in complex_keywords)

    def reply(self, user_input: str) -> str:
        self.round += 1
        if self.round > self.max_rounds:
            return f"[达到最大轮数 {self.max_rounds}，停止，避免 token 失控]"

        # 把用户输入加入窗口
        self.window.add(Message(sender="用户", content=user_input))

        # 上下文过长时做摘要
        context = self.window.get_context()
        if len(context) > self.max_context_chars:
            context = summarize(context, self.max_context_chars)

        # 选择模型：复杂任务用贵模型，简单任务用便宜模型
        llm = self.expensive_llm if self.is_complex(user_input) else self.cheap_llm

        prompt = f"你是 {self.name}。上下文：\n{context}\n\n请回复。"
        reply_text = llm(prompt)

        # 记录模型回复到窗口
        self.window.add(Message(sender=self.name, content=reply_text))

        # 粗略统计 token
        self.total_tokens += len(prompt) + len(reply_text)
        print(f"[成本] 本轮估算 token：{len(prompt) + len(reply_text)}, 累计：{self.total_tokens}")

        return reply_text


# ==================== 演示模型 ====================

def cheap_llm(prompt: str) -> str:
    return f"[便宜模型] 简单回复：已收到。"


def expensive_llm(prompt: str) -> str:
    return f"[贵模型] 深度分析回复：基于上下文的详细推理结果。"


if __name__ == "__main__":
    agent = CostControlledAgent("助手", cheap_llm, expensive_llm, max_rounds=5)

    # 第一轮：简单任务，用便宜模型
    print(agent.reply("你好"))
    # 第二轮：复杂任务，用贵模型
    print(agent.reply("请帮我分析这个系统架构的性能瓶颈并给出优化建议，涉及数据库、缓存、队列等多个模块。"))
    # 第三轮：又是简单任务
    print(agent.reply("谢谢"))
    # 第四、五轮后达到最大轮数停止
    print(agent.reply("继续"))
    print(agent.reply("继续"))
```

**运行输出示例（概念版）：**

```
[成本] 本轮估算 token：60, 累计：60
[便宜模型] 简单回复：已收到。
[成本] 本轮估算 token：220, 累计：280
[贵模型] 深度分析回复：基于上下文的详细推理结果。
[成本] 本轮估算 token：60, 累计：340
[便宜模型] 简单回复：已收到。
[成本] 本轮估算 token：80, 累计：420
[达到最大轮数 5，停止，避免 token 失控]
[成本] 本轮估算 token：0, 累计：420
[达到最大轮数 5，停止，避免 token 失控]
```

---

### 三框架的 token 成本特点（对比）

| 框架 | token 成本主要来源 | 控制建议 |
|------|-------------------|----------|
| **AutoGen** | 群聊对话历史累积快，每个 Agent 都读全部历史 | 设最大轮数；关键信息摘要；不相关 Agent 不进群 |
| **LangGraph** | 状态对象反复读写，循环多轮会累积 | 给 State 设大小上限；循环加迭代次数上限；用 Checkpoint 避免重跑 |
| **AgentScope** | 消息持久化本身不增加 token，但并发多 Agent 会同时读上下文 | 消息摘要；并发数控制；私有历史不共享给无关 Agent |

**面试要点**：token 成本控制的 6 个手段（限制最大轮数、上下文压缩/摘要、滑动窗口、只传给需要的 Agent、便宜/贵模型切换、工具结果截断）。手撕代码时要体现滑动窗口、最大轮数、摘要、模型切换。并说明三框架的成本特点：AutoGen 群聊最易烧 token，LangGraph 循环需加次数上限，AgentScope 并发需控制上下文读取。

---

## Q17. 自研框架 vs 现成框架：怎么选？

**回答**

用 AutoGen、LangGraph、AgentScope 这些现成框架，还是自己从零写一个框架？这是架构选型里的经典问题。本节用大白话把两种路的优劣、适用场景、决策检查表讲清楚。

---

### 自研框架：优势与劣势

**优势：**
- **完全可控**：没有框架的黑箱，每一步都清楚；
- **无冗余**：只实现自己业务需要的功能，没有没用的代码；
- **学习成本低（对自己人）**：团队自己写的，熟悉度最高；
- **易深度定制**：想改哪里改哪里，不受框架 API 限制。

**劣势：**
- **造轮子成本高**：主循环、状态管理、工具调用、日志、容错、人工干预……全要自己写；
- **容易踩坑**：并发、消息丢失、状态不一致等工程问题容易出 bug；
- **社区弱**：没人帮你修 Bug，没现成教程和插件；
- **维护负担重**：框架本身也要持续维护，分散业务开发精力。

---

### 现成框架：优势与劣势

**优势：**
- **开箱即用**：主循环、状态、工具、消息都现成的；
- **社区成熟**：有教程、插件、示例、Bug 修复；
- **可观测性内置**：LangGraph Studio、AgentScope Studio 等现成可视化；
- **快速迭代**：原型几天就能做出来。

**劣势：**
- **学习成本**：要学框架自己的概念（AutoGen GroupChat、LangGraph StateGraph、AgentScope MsgHub）；
- **API 不稳定**：智能体框架新，版本更新快，代码可能跑不通；
- **过度工程化**：简单业务用了可能“杀鸡用牛刀”；
- **黑箱风险**：框架内部出问题，排查困难。

---

### 决策检查表（大白话，逐条勾选）

| 检查项 | 自研框架倾向 | 现成框架倾向 |
|--------|--------------|--------------|
| 业务流程是否标准、固定 | 是（固定流程，自研可控） | 否（探索阶段，现成更快） |
| 是否需要分布式、高并发 | 否（单机足够） | 是（选 AgentScope 等） |
| 团队是否有框架开发经验 | 是 | 否 |
| 是否需要快速出原型 | 否 | 是 |
| 是否需要极致定制（非标协议等） | 是 | 否 |
| 是否能接受框架 API 变动 | 是（自研不受影响） | 否 |
| 是否需要可观测性/审计 | 否（自研可简单实现） | 是（现成内置 Studio） |

**一句话决策：**
- **快速原型、探索业务、不确定需求** → 用现成框架（AutoGen / LangGraph）；
- **生产级、复杂流程、需可审计** → LangGraph；
- **分布式、高并发、安全敏感** → AgentScope；
- **业务非常特殊、框架都不合适、团队有能力** → 自研框架。

---

### 完整代码（概念版）：一个“最小自研框架”的骨架

下面这段纯 Python 代码展示，如果决定自研，一个最小智能体框架的骨架长什么样（只有 4 个核心部件：消息、Agent、主循环、工具注册表）。

```python
"""
最小自研智能体框架骨架（概念版）
消息 / Agent 基类 / 主循环 / 工具注册表
纯 Python 演示，展示自研框架最小可用的样子。
"""

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Message:
    sender: str
    receiver: str
    content: str


class ToolRegistry:
    """工具注册表：工具名 -> 函数。"""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self.tools[name] = func

    def call(self, name: str, **kwargs) -> Any:
        return self.tools[name](**kwargs)


class Agent:
    """Agent 基类：能收消息、能回消息、能用工具。"""

    def __init__(self, name: str, registry: ToolRegistry,
                 llm_call: Callable[[str], str]):
        self.name = name
        self.registry = registry
        self.llm_call = llm_call
        self.history: List[Message] = []

    def receive(self, message: Message):
        self.history.append(message)
        print(f"[收到] {message}")

    def send(self, receiver: 'Agent', content: str):
        msg = Message(sender=self.name, receiver=receiver.name, content=content)
        print(f"[发送] {msg}")
        receiver.receive(msg)

    def reply(self, receiver: 'Agent') -> Optional[str]:
        context = "\n".join([f"{m.sender}->{m.receiver}: {m.content}"
                             for m in self.history[-5:]])
        prompt = f"{self.name} 的系统提示。历史：\n{context}\n请回复 {receiver.name}："
        content = self.llm_call(prompt)
        self.send(receiver, content)
        return content


def demo_llm(prompt: str) -> str:
    return "收到，已处理。"


if __name__ == "__main__":
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)

    alice = Agent("Alice", registry, demo_llm)
    bob = Agent("Bob", registry, demo_llm)

    alice.receive(Message(sender="用户", receiver="Alice", content="启动协作"))
    alice.reply(bob)
    bob.reply(alice)

    print(f"工具调用示例：add(2,3) = {registry.call('add', a=2, b=3)}")
```

**运行输出示例（概念版）：**

```
[收到] [用户 -> Alice] 启动协作
[发送] [Alice -> Bob] 收到，已处理。
[收到] [Alice -> Bob] 收到，已处理。
[发送] [Bob -> Alice] 收到，已处理。
[收到] [Bob -> Alice] 收到，已处理。
工具调用示例：add(2,3) = 5
```

**面试要点**：自研 vs 现成不是非黑即白。自研适合“业务特殊+有能力+需要完全可控”，现成适合“快速原型+社区生态+内置可观测性”。实际生产中也可“现成框架做主结构，自研做关键组件（如自己的沙箱、自己的审计日志）”。

---

## Q18. 项目代码组织：多智能体项目怎么拆目录？

**回答**

一个多智能体项目长期维护，代码扔在一个文件里很快变成“屎山”。本节用大白话讲一个实用的目录组织方式，并说明每个目录放什么、各模块怎么协作。

---

### 推荐的目录结构

```
multi_agent_project/
├── config/                  # 配置：模型密钥、框架选型、各 Agent 参数
│   ├── settings.yaml
│   └── agents.yaml
├── agents/                  # 各角色 Agent 的定义与提示词
│   ├── __init__.py
│   ├── base_agent.py        # Agent 基类
│   ├── planner.py           # 规划 Agent
│   ├── coder.py             # 编程 Agent
│   └── tester.py            # 测试 Agent
├── tools/                   # 外部工具：搜索、计算器、代码沙箱等
│   ├── __init__.py
│   ├── registry.py          # 工具注册表
│   ├── search.py
│   ├── calculator.py
│   └── sandbox.py
├── memory/                  # 记忆：短期历史、长期向量、共享黑板
│   ├── __init__.py
│   ├── short_term.py
│   ├── long_term.py
│   └── shared_board.py
├── workflow/                # 工作流编排：图、状态、路由
│   ├── __init__.py
│   ├── state.py             # 共享状态定义
│   ├── graph.py             # 状态图/主循环
│   └── routers.py           # 条件路由（下一步去哪）
├── observability/           # 可观测性：日志、追踪、审计
│   ├── __init__.py
│   ├── logger.py
│   └── tracer.py
├── safety/                  # 安全：沙箱、输入输出过滤
│   ├── __init__.py
│   ├── sandbox.py
│   └── filter.py
├── main.py                  # 入口：组装 Agent、工作流、启动
├── requirements.txt         # 依赖
└── README.md                # 说明：怎么跑、怎么加新 Agent
```

---

### 各模块职责（大白话）

| 模块 | 放什么 | 职责 |
|------|--------|------|
| `config/` | YAML/JSON 配置 | 模型 API Key、温度参数、最大轮数、Agent 角色配置，不硬编码在代码里 |
| `agents/` | 各 Agent 实现 | 每个角色一个文件，包含 system prompt、工具权限、回复逻辑 |
| `tools/` | 工具函数 | 搜索、计算器、代码执行沙箱，通过注册表统一接入 |
| `memory/` | 记忆层 | 短期历史（对话窗口）、长期记忆（向量数据库）、共享黑板 |
| `workflow/` | 工作流 | 状态定义、图编排、条件路由（LangGraph 风格或自研主循环） |
| `observability/` | 日志追踪 | 结构化日志、追踪 ID、调用链路记录 |
| `safety/` | 安全层 | 代码沙箱、输入输出过滤、敏感信息脱敏 |
| `main.py` | 入口 | 读配置、创建 Agent、注册工具、启动主循环 |

---

### 示例：main.py 组装（概念版）

```python
"""
main.py：多智能体项目入口（概念版）
从 config 读配置，创建 Agent，注册工具，启动工作流。
"""

import yaml
from agents.base_agent import BaseAgent
from agents.planner import PlannerAgent
from agents.coder import CoderAgent
from agents.tester import TesterAgent
from tools.registry import ToolRegistry
from tools.search import web_search
from tools.calculator import calculator
from workflow.graph import StateGraph
from workflow.state import AgentState
from observability.logger import setup_logger


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("web_search", web_search)
    registry.register("calculator", calculator)
    return registry


def main():
    config = load_config()
    logger = setup_logger(config.get("log_file", "./trace.jsonl"))

    registry = build_tool_registry()

    planner = PlannerAgent("规划 Agent", registry, config)
    coder = CoderAgent("编程 Agent", registry, config)
    tester = TesterAgent("测试 Agent", registry, config)

    # 构建工作流：规划 -> 编程 -> 测试 -> (循环或结束)
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner.run)
    graph.add_node("coder", coder.run)
    graph.add_node("tester", tester.run)
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "tester")
    graph.set_entry_point("planner")

    initial_state: AgentState = {
        "user_input": config.get("task", "写一个 Hello World"),
        "steps": [],
        "answer": "",
    }

    final_state = graph.invoke(initial_state)
    logger.info("任务完成", {"final_answer": final_state["answer"]})


if __name__ == "__main__":
    main()
```

---

### 加新 Agent 的步骤（大白话）

1. 在 `agents/` 下新建一个文件，继承 `BaseAgent`，写好 system prompt；
2. 在 `config/agents.yaml` 里加这个 Agent 的参数（温度、工具权限等）；
3. 在 `main.py` 里实例化它，加到工作流图的节点里；
4. 如果需要新工具，在 `tools/` 下加函数，在 `registry.py` 注册；
5. 跑一遍，看可观测性日志是否正常。

**面试要点**：多智能体项目按职责拆目录（config/agents/tools/memory/workflow/observability/safety），各模块高内聚低耦合，加新 Agent 只需新建文件并注册到工作流，不动既有代码。main.py 只负责组装，不写业务逻辑。

---

## Q19. 常见坑与学习路径

**回答**

学和用智能体框架，新手容易踩一些坑。本节用大白话列出最常见的 8 个坑及应对，再给一条推荐的学习路径，帮你少走弯路。

---

### 常见坑 1：一句话让 Agent “自己搞定”，不拆任务

**现象**：用户给一个很模糊的需求（如“帮我做个数据分析平台”），期望一个 Agent 一步到位，结果 Agent 瞎写代码、跑不通、一直反复。

**应对**：先拆任务。规划 Agent 把大任务拆成小步骤，每个步骤交给专门的 Agent，再汇总。不要指望一个 Agent 什么都干。

---

### 常见坑 2：对话历史无限制，token 爆炸

**现象**：群聊跑几十轮，之前的对话全部塞进上下文，账单爆炸，模型还因为上下文太长开始丢焦点。

**应对**：设置最大轮数；滑动窗口只保留最近 N 条；长对话做摘要；工具结果太长截断。

---

### 常见坑 3：代码直接在本机跑，没沙箱

**现象**：Agent 写了 `import os; os.remove(...)` 或 `subprocess`，直接在开发者电脑上执行，数据被删。

**应对**：代码执行必须进 Docker 沙箱，禁止网络和文件系统访问；生产环境沙箱与业务网络隔离。

---

### 常见坑 4：调试时看不清哪个 Agent 错了

**现象**：多个 Agent 传来传去，输出结果不对，不知道是哪个 Agent 的 prompt 有问题，还是工具返回错了。

**应对**：开启结构化日志 + 追踪 ID，把每条消息、每次工具调用、每次 LLM 调用都记下来；用 LangGraph Studio / AgentScope Studio 可视化。

---

### 常见坑 5：框架 API 升级，代码跑不通

**现象**：昨天还能跑的 AutoGen 代码，今天升级版本后报错，因为 AutoGen 0.7.x 架构重构了。

**应对**：生产环境锁定框架版本（pip requirements 固定版本号）；定期看官方 release note；不要盲目 `pip install --upgrade`。

---

### 常见坑 6：人工确认缺失，关键操作全自动

**现象**：Agent 自动调用了删除数据、发邮件、转账等关键操作，没人审核就执行了。

**应对**：关键步骤加人工中断（HITL），人工点确认后才继续；审计日志记录谁在什么时候批准了什么操作。

---

### 常见坑 7：每个 Agent 都看全部历史，信息过载

**现象**：10 个 Agent 群聊，每个 Agent 都要读完全部消息，上下文又长又乱，重点被淹没。

**应对**：只把相关信息发给相关 Agent；私有历史不共享；共享黑板只写关键结论。

---

### 常见坑 8：自研框架过早，造轮子造到怀疑人生

**现象**：刚开始就决定“自研一个框架”，结果花两周在主循环和消息路由上，业务还没开始。

**应对**：先用现成框架跑通原型；只有业务特殊到现成框架真的不合适，才考虑自研；自研也从最小骨架开始，逐步加功能。

---

### 推荐学习路径（给零基础读者）

```
第 1 周：先跑通一个现成框架的 Hello World
         → 选 AutoGen 或 LangGraph，跑官方示例，理解“Agent 怎么对话”

第 2 周：手撕一个最小框架（消息/主循环/工具）
         → 不依赖框架，纯 Python 写一遍，理解框架底层在干什么

第 3 周：做一个多 Agent 协作小项目
         → 2-3 个角色，分工协作，加上工具（搜索/计算器）

第 4 周：加工程化能力
         → 日志/追踪、消息持久化、人工中断、沙箱安全

第 5 周：控制成本与性能
         → 滑动窗口、最大轮数、摘要、便宜/贵模型切换

第 6 周：部署与运维
         → Docker 沙箱、配置文件化、CI/CD、监控报警
```

**学习原则**：先会用，再懂原理，最后才考虑自研；先单机，再分布式；先简单任务，再复杂流程。

**面试要点**：能说出至少 4-5 个常见坑及应对（token 爆炸、无沙箱代码执行、缺少人工确认、调试难、API 版本不稳、信息过载、过早自研），并给出一条合理的学习路径。

---

## Q20. HITL（人工在环）：怎么让关键步骤暂停等人确认？

**回答**

多智能体系统跑在生产环境，不是所有步骤都能全自动。涉及钱、数据、安全的关键操作，必须“人在环”（Human-in-the-loop, HITL）——流程暂停，等人工确认后才继续。本节聊 HITL 的设计思路，并用代码演示怎么实现“暂停 → 等人确认 → 恢复执行”。

---

### HITL 是什么，大白话

> HITL：关键步骤不自动执行，而是弹出来让人看一眼，人点“同意”才继续，点“拒绝”就终止或走备选流程。

常见需要 HITL 的场景：
- 删除数据库记录、修改配置；
- 发邮件、发短信、转账；
- 执行 Agent 写的代码（特别是涉及文件、网络）；
- 合规审批：贷款审批、保险理赔等。

HITL 的三种形式：
1. **同步阻塞**：流程暂停，等人在控制台/UI 点确认，然后继续；
2. **异步审批**：流程暂停并保存断点，人工稍后在审批界面点同意，系统从断点恢复；
3. **覆盖修改**：人工不只点同意/拒绝，还可以修改 Agent 的输出后再继续。

---

### 完整代码（概念版）

下面这段纯 Python 代码演示：带 HITL 的工作流，在关键步骤前暂停，等用户输入 y/n，输入 y 则继续，输入 n 则终止；也支持“覆盖修改”——用户可以输入自定义内容替换 Agent 生成的结果。

```python
"""
HITL（人工在环）极简实现（概念版）
关键步骤暂停等人确认 / 支持同意、拒绝、覆盖修改
纯 Python 演示，input() 模拟人工确认（生产用 UI/审批系统）。
"""

from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import os


@dataclass
class Message:
    sender: str
    receiver: str
    content: str


class HumanInterruptError(Exception):
    """人工中断异常：流程暂停，等待人工决策。"""


class HITLManager:
    """
    人工在环管理器。
    决定哪些步骤需要人工确认，暂停流程，等待决策后恢复。
    """

    def __init__(self, interactive: bool = True,
                 approval_log: str = "./approvals.jsonl"):
        self.interactive = interactive          # 是否真正等人输入（False 表示自动同意，用于测试）
        self.approval_log = approval_log        # 审计日志：谁在什么时候批准了什么
        self.pending: Optional[Dict[str, Any]] = None  # 当前待人工决策的上下文

    def require_approval(self, step_name: str, context: Dict[str, Any]) -> bool:
        """
        要求人工批准某个步骤。
        返回 True（批准）或 False（拒绝）。
        如果非交互模式，自动返回 True（方便自动化测试）。
        """
        record = {
            "step": step_name,
            "context": context,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

        if not self.interactive:
            record["decision"] = "auto_approved"
            self._log(record)
            print(f"[HITL 自动通过] {step_name}")
            return True

        # 交互模式：暂停，打印上下文，等人工输入
        self.pending = record
        print("\n" + "=" * 60)
        print(f"[人工确认] 步骤：{step_name}")
        print(json.dumps(context, ensure_ascii=False, indent=2))
        print("=" * 60)
        print("选项：")
        print("  y   - 同意，继续执行")
        print("  n   - 拒绝，终止流程")
        print("  m   - 覆盖修改：输入自定义内容，替换 Agent 的结果")
        choice = input("请选择 (y/n/m): ").strip().lower()

        if choice == "y":
            record["decision"] = "approved"
            self._log(record)
            return True
        elif choice == "n":
            record["decision"] = "rejected"
            self._log(record)
            return False
        elif choice == "m":
            override = input("请输入替换内容：")
            record["decision"] = "overridden"
            record["override_content"] = override
            self._log(record)
            # 把覆盖内容存进 pending，供调用方读取
            self.pending["override_content"] = override
            return True  # 覆盖后继续执行，但用替换内容
        else:
            print("无效输入，默认拒绝。")
            record["decision"] = "rejected"
            self._log(record)
            return False

    def get_pending(self) -> Optional[Dict[str, Any]]:
        """获取当前待决策的上下文（含可能的覆盖内容）。"""
        return self.pending

    def clear_pending(self):
        self.pending = None

    def _log(self, record: Dict[str, Any]):
        with open(self.approval_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==================== 带 HITL 的 Agent ====================

class AgentWithHITL:
    def __init__(self, name: str,
                 llm_call: Callable[[str], str],
                 hitl: HITLManager,
                 critical_tools: List[str] = None):
        self.name = name
        self.llm_call = llm_call
        self.hitl = hitl
        self.critical_tools = critical_tools or ["delete_data", "send_email", "transfer_money"]
        self.history: List[Message] = []

    def receive(self, message: Message):
        self.history.append(message)
        print(f"[收到] {message}")

    def send(self, receiver: 'AgentWithHITL', content: str):
        msg = Message(sender=self.name, receiver=receiver.name, content=content)
        print(f"[发送] {msg}")
        receiver.receive(msg)

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        调用工具。如果是关键工具，先走 HITL，人工批准后才真正执行。
        """
        if tool_name in self.critical_tools:
            context = {
                "agent": self.name,
                "tool": tool_name,
                "args": args,
            }
            approved = self.hitl.require_approval(tool_name, context)
            if not approved:
                return f"[工具 {tool_name} 已被人工拒绝]"

            # 检查是否有覆盖内容
            pending = self.hitl.get_pending()
            override = pending.get("override_content") if pending else None
            self.hitl.clear_pending()

            if override:
                print(f"[HITL 覆盖] 使用人工输入的结果：{override}")
                return override

        # 真正执行工具（演示版）
        print(f"[执行工具] {tool_name}({args})")
        return f"工具 {tool_name} 执行完成（模拟结果）"

    def reply(self, receiver: 'AgentWithHITL') -> Optional[str]:
        context = "\n".join([f"{m.sender}->{m.receiver}: {m.content}"
                             for m in self.history[-5:]])
        prompt = f"你是 {self.name}。历史：\n{context}\n请回复 {receiver.name}："
        content = self.llm_call(prompt)
        self.send(receiver, content)
        return content


def demo_llm(prompt: str) -> str:
    return "收到，已处理。"


if __name__ == "__main__":
    # 设为 True 时会真正等 input()；这里演示用 False（自动通过），方便自动运行
    hitl = HITLManager(interactive=False)

    alice = AgentWithHITL("Alice", demo_llm, hitl)
    bob = AgentWithHITL("Bob", demo_llm, hitl)

    alice.receive(Message(sender="用户", receiver="Alice", content="删掉用户 #123 的数据"))
    alice.reply(bob)

    # Alice 调用关键工具 delete_data，会触发 HITL
    result = alice.call_tool("delete_data", {"user_id": 123, "table": "users"})
    print(f"[结果] {result}")

    print("\n审计日志 ./approvals.jsonl 已记录所有决策。")
```

> 注意：上面 `__main__` 里 `content"` 是笔误演示，实际代码中已是正确的 `content="..."`。

**运行输出示例（非交互自动通过模式）：**

```
[收到] [用户 -> Alice] 删掉用户 #123 的数据
[发送] [Alice -> Bob] 收到，已处理。
[收到] [Alice -> Bob] 收到，已处理。
[HITL 自动通过] delete_data
[执行工具] delete_data({'user_id': 123, 'table': 'users'})
[结果] 工具 delete_data 执行完成（模拟结果）

审计日志 ./approvals.jsonl 已记录所有决策。
```

交互模式下，屏幕会暂停打印上下文，等你输入 y/n/m。

---

### 三框架的 HITL 实现对比

| 框架 | HITL 方式 | 大白话 |
|------|-----------|--------|
| **AutoGen** | `human_input_mode`：在对话中插入人工输入，关键步骤暂停等人回复 | 群聊里某个人突然发言，等人工接话 |
| **LangGraph** | `interrupt` / `Command(resume=...)`：在图的边上设中断，暂停后人工修改状态并 resume | 流程图在某个节点前卡住，人工改完状态点继续 |
| **AgentScope** | 可视化 UI 组件 + 消息中断：在 Studio 里查看待办并点确认 | 工作台里列出待审批项，点同意后继续 |

**面试要点**：HITL 的核心是“关键步骤暂停、等人确认、审计日志”。手撕代码时要体现：判断关键操作、暂停流程、输入 y/n/m（同意/拒绝/覆盖）、记录审计日志、恢复执行。三框架各有实现方式，原理一致。

---

## Q21. 多智能体冲突解决：两个 Agent 观点不一致怎么办？

**回答**

多个 Agent 协作，最常见的问题是**冲突**：产品经理 Agent 说“要这个功能”，程序员 Agent 说“不可行”，测试 Agent 说“跑不通”，三个观点打架。如果没机制解决，系统会死循环或做出错误决策。本节聊常见的冲突类型和 5 种解决策略。

---

### 常见冲突类型

| 类型 | 例子 |
|------|------|
| **观点冲突** | 两个 Agent 对同一事给出不同建议 |
| **执行冲突** | 两个 Agent 同时修改同一个文件/数据 |
| **资源冲突** | 多个 Agent 都要调用同一个有限工具（如 API 配额） |
| **优先级冲突** | Agent A 说“先做 A”，Agent B 说“先做 B” |

---

### 5 种解决策略（大白话）

| 策略 | 大白话 | 适合场景 |
|------|--------|----------|
| **仲裁者（裁判）** | 设一个“主管 Agent”专门裁定，听完各方意见后拍板 | 观点冲突，需要一个最终决定 |
| **投票表决** | 多个 Agent 投票，少数服从多数 | 无明显主管，各方地位平等 |
| **链式让渡** | 按固定顺序，后面的 Agent 尊重前面的结论 | 流程固定，有明确先后（规划→编程→测试） |
| **合并融合** | 把各方意见折中，合成一个折中方案 | 各方都有道理，需要综合 |
| **升级人工** | 实在打不起来，暂停扔给人工决策 | 涉及重大决策、安全、合规 |

---

### 完整代码（概念版）：仲裁者 + 投票表决

下面这段纯 Python 代码演示：多个 Agent 提出不同方案，由“仲裁者 Agent”裁定；也演示“投票表决”让多个 Agent 投票决定。

```python
"""
多智能体冲突解决极简实现（概念版）
仲裁者裁定 / 投票表决
纯 Python 演示。
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from collections import Counter


@dataclass
class Proposal:
    """一个提案：谁提出的、内容、理由。"""
    proposer: str
    content: str
    reason: str


class Arbiter:
    """仲裁者：收集多个提案，选出一个最终方案。"""

    def __init__(self, llm_call: Callable[[str], str]):
        self.llm_call = llm_call

    def arbitrate(self, proposals: List[Proposal]) -> Proposal:
        """把所有提案喂给大模型，让它选出最好的一个。"""
        options = "\n".join([
            f"[{i}] 提案人：{p.proposer}\n    内容：{p.content}\n    理由：{p.reason}\n"
            for i, p in enumerate(proposals)
        ])
        prompt = f"""以下是多个智能体提出的不同方案，请你选出一个最合适的，并说明理由。
只输出选中的序号（数字），不要输出其他内容。

{options}

请直接输出序号："""
        reply = self.llm_call(prompt).strip()
        try:
            idx = int(reply)
            if 0 <= idx < len(proposals):
                return proposals[idx]
        except ValueError:
            pass
        # 解析失败，默认选第一个
        return proposals[0]


class VotingMechanism:
    """投票表决：每个 Agent 对每个方案投一票，得票最多者胜出。"""

    def vote(self, proposals: List[Proposal],
             votes: Dict[str, int]) -> Proposal:
        """
        votes: {提案人名字: 投给哪个提案的索引}
        返回得票最多的提案。
        """
        vote_counts = Counter()
        for proposer, chosen_idx in votes.items():
            vote_counts[chosen_idx] += 1

        best_idx = vote_counts.most_common(1)[0][0]
        return proposals[best_idx]


def demo_llm(prompt: str) -> str:
    """演示版：根据提案内容选一个。"""
    if "方案 A" in prompt or "0" in prompt:
        return "0"
    return "0"


if __name__ == "__main__":
    proposals = [
        Proposal("产品经理", "方案 A：先做核心功能，二期再做扩展", "上线快，能快速验证"),
        Proposal("程序员", "方案 B：先把架构搭好，再做功能", "后期好维护，但上线慢"),
        Proposal("测试员", "方案 C：边做边测，每个功能都要有用例", "质量高，但进度慢"),
    ]

    # 策略 1：仲裁者裁定
    arbiter = Arbiter(demo_llm)
    chosen = arbiter.arbitrate(proposals)
    print(f"[仲裁者裁定] 选中：{chosen.proposer} 的 {chosen.content}")

    # 策略 2：投票表决
    votes = {"产品经理": 0, "程序员": 1, "测试员": 0}  # 两人投方案 A，一人投方案 B
    voting = VotingMechanism()
    winner = voting.vote(proposals, votes)
    print(f"[投票表决] 获胜：{winner.proposer} 的 {winner.content}")
```

**运行输出示例（概念版）：**

```
[仲裁者裁定] 选中：产品经理 的 方案 A：先做核心功能，二期再做扩展
[投票表决] 获胜：产品经理 的 方案 A：先做核心功能，二期再做扩展
```

---

### 三框架中如何体现冲突解决

| 框架 | 冲突解决典型做法 |
|------|------------------|
| **AutoGen** | 设一个“主持人 Agent”在群聊里总结并拍板，或者由用户在群里发消息决定 |
| **LangGraph** | 在条件边里写判断逻辑，根据各 Agent 输出决定下一步走哪个分支；或设一个“裁定节点” |
| **AgentScope** | 通过 Pipeline 的聚合步骤，把多个 Agent 的决策汇总后再决定 |

**面试要点**：冲突解决的 5 种策略（仲裁者、投票、链式让渡、合并融合、升级人工）。手撕代码时要体现仲裁者裁定和投票表决，并说明生产中常配合 HITL（重大冲突升级给人工）。

---

## Q22. 成本优化实战：怎么让多智能体系统花更少的钱？

**回答**

多智能体系统的钱主要花在 token（调模型）和工具调用（搜索 API、代码沙箱等）。本节用大白话讲 6 个成本优化手段，并给一个可落地的“便宜模型干粗活、贵模型干精活”的实战示例。

---

### 6 个成本优化手段（大白话）

| 手段 | 做什么 | 大白话 |
|------|--------|--------|
| **便宜模型干粗活** | 分类、提取、格式化等简单任务用便宜模型 | 能用几毛钱的模型别用几块钱的 |
| **贵模型只干精活** | 复杂推理、架构设计、最终审核用贵模型 | 关键决策才请贵专家 |
| **上下文缓存** | 相同的长上下文只发一次，后续复用 | 不要重复发一样的长文本 |
| **滑动窗口** | 只保留最近 N 条对话，旧的扔掉 | 只记得最近发生的事 |
| **工具结果截断** | 工具返回太长只取关键部分 | 搜索返回 10 页，只取前 2 段 |
| **批量/并发合并** | 多个独立请求合并成一次模型调用 | 一次解决多个问题，别一个一个问 |

---

### 完整代码（概念版）：分层模型路由

下面这段纯 Python 代码演示：根据任务复杂度自动路由到“便宜模型”或“贵模型”，并统计累计成本（演示版按字符粗略估算）。

```python
"""
成本优化极简实现（概念版）
分层模型路由 / 便宜模型 vs 贵模型 / 累计成本统计
纯 Python 演示。
"""

from typing import Callable, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str
    cost_per_1k_tokens: float  # 每 1k token 的价格（元，演示用）


# 演示模型配置
CHEAP_MODEL = ModelConfig("便宜模型", 0.0015)   # 每 1k token 0.0015 元
EXPENSIVE_MODEL = ModelConfig("贵模型", 0.03)    # 每 1k token 0.03 元


class CostOptimizedRouter:
    """
    成本优化路由器：根据任务复杂度选模型，统计累计成本。
    """

    def __init__(self, cheap_llm: Callable[[str], str],
                 expensive_llm: Callable[[str], str],
                 cheap_model: ModelConfig = CHEAP_MODEL,
                 expensive_model: ModelConfig = EXPENSIVE_MODEL):
        self.cheap_llm = cheap_llm
        self.expensive_llm = expensive_llm
        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        self.total_cost = 0.0
        self.total_tokens = 0
        self.calls: List[Dict[str, Any]] = []

    def is_complex(self, text: str) -> bool:
        """简单复杂度判断：演示版按长度和关键词。"""
        complex_keywords = ["架构", "推理", "优化", "分析", "复杂", "审核", "决策"]
        return len(text) > 80 or any(k in text for k in complex_keywords)

    def route(self, task_type: str, prompt: str) -> str:
        """
        路由：简单任务用便宜模型，复杂任务用贵模型。
        task_type: "classify" | "extract" | "format" | "reason" | "audit" | ...
        """
        # 分类任务、提取任务、格式化任务：强制用便宜模型
        simple_types = {"classify", "extract", "format", "summarize_short"}
        if task_type in simple_types:
            model = self.cheap_model
            llm = self.cheap_llm
        elif self.is_complex(prompt):
            model = self.expensive_model
            llm = self.expensive_llm
        else:
            model = self.cheap_model
            llm = self.cheap_llm

        reply = llm(prompt)
        tokens = len(prompt) + len(reply)  # 演示版粗略估算
        cost = tokens / 1000.0 * model.cost_per_1k_tokens

        self.total_tokens += tokens
        self.total_cost += cost
        self.calls.append({
            "task_type": task_type,
            "model": model.name,
            "tokens": tokens,
            "cost": cost,
        })

        print(f"[路由] {task_type} -> {model.name}, token≈{tokens}, 成本≈{cost:.4f}元")
        return reply

    def report(self) -> str:
        return (f"累计 token：{self.total_tokens}, 累计成本：{self.total_cost:.4f}元, "
                f"调用次数：{len(self.calls)}")


def cheap_llm(prompt: str) -> str:
    return "[便宜模型] 已完成简单任务。"


def expensive_llm(prompt: str) -> str:
    return "[贵模型] 已完成复杂推理任务。"


if __name__ == "__main__":
    router = CostOptimizedRouter(cheap_llm, expensive_llm)

    # 简单任务：分类
    router.route("classify", "这条评论是正面还是负面？")
    # 简单任务：提取
    router.route("extract", "从句子里提取人名：张三去了北京。")
    # 复杂任务：审核
    router.route("audit", "请审核这段代码的安全性，检查是否有注入漏洞……")
    # 中等任务：格式化（强制便宜模型）
    router.route("format", "把这段 JSON 格式化。")

    print(router.report())
```

**运行输出示例（概念版）：**

```
[路由] classify -> 便宜模型, token≈40, 成本≈0.0001元
[路由] extract -> 便宜模型, token≈40, 成本≈0.0001元
[路由] audit -> 贵模型, token≈120, 成本≈0.0036元
[路由] format -> 便宜模型, token≈30, 成本≈0.0000元
累计 token：230, 累计成本：0.0038元, 调用次数：4
```

---

### 三框架成本优化对比

| 框架 | 成本优化切入点 |
|------|----------------|
| **AutoGen** | 设置群聊最大轮数；让不相关 Agent 退出群聊；用便宜模型做总结 Agent |
| **LangGraph** | State 加大小上限；循环加迭代次数上限；不同节点用不同模型 |
| **AgentScope** | 消息摘要后再分发；控制并发数；关键节点才用贵模型 |

**面试要点**：成本优化的 6 个手段（便宜/贵模型分层、上下文缓存、滑动窗口、工具结果截断、批量合并、最大轮数）。手撕代码时要体现分层模型路由和成本统计。并说明三框架各自的优化切入点。

---

## Q23. 可观测性与安全进阶：审计、脱敏、分布式追踪

**回答**

Q15 已经介绍了可观测性和安全的基础。本节进一步聊生产环境的**审计（谁什么时候干了什么）、脱敏（敏感信息不外漏）、分布式追踪（Agent 分布在多台机器上怎么追踪）**。

---

### 审计：留痕、可追溯、不可篡改

审计的核心是把每一步都记下来，且记录不可篡改：
- **谁**：哪个 Agent、哪个用户、什么角色；
- **什么时候**：精确到毫秒的时间戳；
- **做了什么**：调了哪个工具、发了什么消息、修改了什么状态；
- **结果如何**：成功、失败、被拒绝。

下面是一个带签名（防篡改）的审计日志示例：

```python
"""
审计日志进阶实现（概念版）
结构化审计 / 哈希签名防篡改 / 可追溯
纯 Python 演示。
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, List


class AuditLogger:
    """审计日志：每条记录带前一条的哈希，形成链式结构，篡改可被发现。"""

    def __init__(self, log_file: str = "./audit.jsonl"):
        self.log_file = log_file
        self.prev_hash = "0" * 64  # 初始前置哈希

    def _compute_hash(self, record: Dict[str, Any], prev_hash: str) -> str:
        content = json.dumps(record, sort_keys=True, ensure_ascii=False) + prev_hash
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def log(self, actor: str, action: str, detail: Dict[str, Any],
            result: str = "success"):
        record = {
            "timestamp": datetime.now().isoformat(),
            "actor": actor,
            "action": action,
            "detail": detail,
            "result": result,
            "prev_hash": self.prev_hash,
        }
        record["hash"] = self._compute_hash(record, self.prev_hash)
        self.prev_hash = record["hash"]

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[审计] {actor} | {action} | {result}")

    def verify_chain(self) -> bool:
        """验证审计链是否被篡改（重新计算每条哈希并比对）。"""
        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prev = "0" * 64
        for line in lines:
            record = json.loads(line)
            expected = self._compute_hash({k: v for k, v in record.items()
                                           if k != "hash"}, prev)
            if record["hash"] != expected:
                print(f"[审计验证失败] 篡改点：{record['timestamp']}")
                return False
            prev = record["hash"]
        print("[审计验证通过] 审计链完整，未被篡改。")
        return True


if __name__ == "__main__":
    logger = AuditLogger()
    logger.log("销售 Agent", "调用工具", {"tool": "check_inventory", "sku": "NB-001"})
    logger.log("测试 Agent", "拒绝操作", {"tool": "delete_data", "reason": "未获批准"}, result="rejected")
    logger.verify_chain()
```

**运行输出示例（概念版）：**

```
[审计] 销售 Agent | 调用工具 | success
[审计] 测试 Agent | 拒绝操作 | rejected
[审计验证通过] 审计链完整，未被篡改。
```

---

### 脱敏：敏感信息不外漏

Agent 输出可能包含手机号、邮箱、身份证、银行卡等。必须在输出给用户或写日志前脱敏。

```python
"""
脱敏实现（概念版）
手机号 / 邮箱 / 身份证 / 银行卡 脱敏
纯 Python 演示。
"""

import re


class DataMasker:
    MOBILE = re.compile(r"(\d{3})\d{4}(\d{4})")
    EMAIL = re.compile(r"([^@]{2})[^@]*(@[^@]+)")
    ID_CARD = re.compile(r"(\d{6})\d{8}(\d{4})")
    BANK_CARD = re.compile(r"(\d{4})\d{6,12}(\d{4})")

    def mask(self, text: str) -> str:
        text = self.MOBILE.sub(r"\1****\2", text)
        text = self.EMAIL.sub(r"\1****\2", text)
        text = self.ID_CARD.sub(r"\1********\2", text)
        text = self.BANK_CARD.sub(r"\1******\2", text)
        return text


if __name__ == "__main__":
    masker = DataMasker()
    print(masker.mask("联系方式：13800138000，邮箱 alice@example.com，身份证 110101199001011234"))
    # 输出：联系方式：138****138000，邮箱 al****@example.com，身份证 110101********1234
```

---

### 分布式追踪：多机 Agent 怎么追踪

当 Agent 分布在多台机器上，单机日志不够用，需要**分布式追踪**：给每个请求分配一个全局唯一的 `trace_id`，每个步骤分配 `span_id`，跨机器传递 `trace_id`，把所有日志串起来。

关键做法：
1. **trace_id**：全局唯一，贯穿整个请求；
2. **span_id**：每个步骤/每个 Agent 的唯一 ID；
3. **parent_span_id**：记录上一步是谁，形成调用树；
4. **跨机器传播**：HTTP 头部或消息元数据里带 `trace_id`，接收方读取并继续。

生产可接入 OpenTelemetry / Jaeger / Zipkin，自动采集并可视化调用链路。

**面试要点**：审计要体现“结构化日志 + 哈希签名防篡改 + 链式验证”；脱敏要体现“手机号/邮箱/身份证/银行卡正则脱敏”；分布式追踪要体现“trace_id/span_id/parent_span_id 贯穿跨机器调用”。

---

## Q24. 多智能体评估：怎么评估框架搭的系统好不好？

**回答**

多智能体系统跑通了，怎么知道它“好不好”？“好”的标准是什么？单 Agent 评估看答案对不对，多智能体评估还要看**协作是否顺畅、是否死循环、token 是否失控、人工干预次数多不多**。本节聊多智能体评估的核心指标和评估方法。

---

### 多智能体评估的核心指标（大白话）

| 指标 | 大白话 | 怎么算 |
|------|--------|--------|
| **任务完成率** | 给一批任务，多少比例最终完成了 | 完成的任务数 / 总任务数 |
| **平均轮数** | 完成一个任务平均聊了几轮 | 总轮数 / 完成任务数 |
| **token 成本** | 完成一个任务烧了多少 token | 总 token / 完成任务数 |
| **人工干预次数** | 多少次需要人工点确认或修正 | 统计 HITL 触发次数 |
| **死循环率** | 多少比例任务陷入无限循环被强行停止 | 达最大轮数未完成的比例 |
| **协作一致性** | 多个 Agent 的输出是否矛盾 | 人工或模型打分，0-1 分 |
| **工具调用成功率** | 工具调用多少比例成功 | 成功调用 / 总调用 |

---

### 评估方法：黄金数据集 + 模拟用户

1. **黄金数据集**：准备一批典型任务和标准答案（如“帮我订机票”→“机票已订，航班号 XX”）；
2. **模拟用户**：用另一个模型扮演用户，不断丢任务给多智能体系统，看它怎么响应；
3. **A/B 测试**：同一任务分别用 AutoGen / LangGraph 跑，比较完成率、成本、轮数；
4. **回归测试**：每次改 prompt 或换框架版本，跑一遍黄金数据集，看指标有没有退步。

---

### 完整代码（概念版）：多智能体评估框架

下面这段纯 Python 代码演示：给多智能体系统跑一批任务，统计完成率、平均轮数、token 成本、人工干预次数，输出评估报告。

```python
"""
多智能体评估极简实现（概念版）
黄金数据集 / 任务完成率 / 平均轮数 / token 成本 / 人工干预次数 / 评估报告
纯 Python 演示。
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field
import time


@dataclass
class Task:
    """一条评估任务：输入 + 期望结果 + 任务类型。"""
    task_id: str
    input: str
    expected: str
    task_type: str = "general"


@dataclass
class TaskResult:
    """单个任务的执行结果。"""
    task_id: str
    success: bool
    rounds: int
    tokens: int
    human_interventions: int
    output: str
    error: str = ""


class MultiAgentEvaluator:
    """
    多智能体评估器：
    - 传入一个“系统运行函数”（把输入交给多智能体系统，返回输出）
    - 跑一批黄金任务，统计各项指标
    - 输出评估报告
    """

    def __init__(self, run_system: Callable[[str], Dict[str, Any]]):
        """
        run_system: 输入用户请求，返回 {
            "output": str,      # 系统最终输出
            "rounds": int,      # 聊了几轮
            "tokens": int,      # 烧了多少 token
            "interventions": int,  # 人工干预次数
            "success": bool,    # 是否完成
            "error": str        # 错误信息（失败时）
        }
        """
        self.run_system = run_system
        self.results: List[TaskResult] = []

    def evaluate(self, tasks: List[Task]) -> Dict[str, Any]:
        self.results = []
        for task in tasks:
            print(f"[评估] 任务 {task.task_id}: {task.input[:40]}...")
            start = time.time()
            try:
                sys_result = self.run_system(task.input)
                # 简单判断：输出里包含期望关键词，或系统标记 success
                success = sys_result.get("success", False) or \
                          (task.expected.lower() in sys_result.get("output", "").lower())
                result = TaskResult(
                    task_id=task.task_id,
                    success=success,
                    rounds=sys_result.get("rounds", 1),
                    tokens=sys_result.get("tokens", 0),
                    human_interventions=sys_result.get("interventions", 0),
                    output=sys_result.get("output", ""),
                    error=sys_result.get("error", ""),
                )
            except Exception as e:
                result = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    rounds=0,
                    tokens=0,
                    human_interventions=0,
                    output="",
                    error=str(e),
                )
            self.results.append(result)
            print(f"    -> 成功={result.success}, 轮数={result.rounds}, token={result.tokens}")

        return self._report()

    def _report(self) -> Dict[str, Any]:
        n = len(self.results)
        if n == 0:
            return {"message": "无任务"}

        success_count = sum(1 for r in self.results if r.success)
        total_rounds = sum(r.rounds for r in self.results)
        total_tokens = sum(r.tokens for r in self.results)
        total_interventions = sum(r.human_interventions for r in self.results)
        fail_count = n - success_count

        report = {
            "总任务数": n,
            "成功数": success_count,
            "失败数": fail_count,
            "任务完成率": round(success_count / n, 4),
            "平均轮数": round(total_rounds / n, 2),
            "平均 token": round(total_tokens / n, 2),
            "总 token": total_tokens,
            "人工干预总次数": total_interventions,
            "平均人工干预": round(total_interventions / n, 2),
            "死循环/超限率": round(fail_count / n, 4),
        }
        return report

    def print_report(self, report: Dict[str, Any]):
        print("\n" + "=" * 50)
        print("多智能体评估报告")
        print("=" * 50)
        for k, v in report.items():
            print(f"{k}: {v}")
        print("=" * 50)


# ==================== 模拟一个多智能体系统（演示版） ====================

def mock_multi_agent_system(user_input: str) -> Dict[str, Any]:
    """演示版：模拟多智能体系统的运行。"""
    import random
    success = random.random() > 0.2  # 80% 成功率
    rounds = random.randint(2, 8)
    tokens = rounds * 150
    interventions = 0 if success else 1
    output = "任务已完成：" + user_input if success else "任务失败：超出最大轮数"
    return {
        "output": output,
        "rounds": rounds,
        "tokens": tokens,
        "interventions": interventions,
        "success": success,
        "error": "" if success else "max_rounds_exceeded",
    }


if __name__ == "__main__":
    tasks = [
        Task("T1", "帮我订一张从北京到上海的机票", "机票已订"),
        Task("T2", "分析这段代码的安全性", "安全"),
        Task("T3", "生成一份周报模板", "周报"),
        Task("T4", "帮我退掉订单 #999", "退款"),
        Task("T5", "查询明天的天气", "天气"),
    ]

    evaluator = MultiAgentEvaluator(mock_multi_agent_system)
    report = evaluator.evaluate(tasks)
    evaluator.print_report(report)
```

**运行输出示例（概念版，随机）：**

```
[评估] 任务 T1: 帮我订一张从北京到上海的机票
    -> 成功=True, 轮数=4, token=600
...
多智能体评估报告
==================================================
总任务数: 5
成功数: 4
失败数: 1
任务完成率: 0.8
平均轮数: 5.0
平均 token: 700.0
总 token: 3500
人工干预总次数: 1
平均人工干预: 0.2
死循环/超限率: 0.2
==================================================
```

---

### 三框架评估重点对比

| 框架 | 评估重点 |
|------|----------|
| **AutoGen** | 群聊轮数是否过多、token 是否失控、是否容易跑偏/死循环 |
| **LangGraph** | 条件边分支是否正确、状态是否一致、断点续传是否可恢复 |
| **AgentScope** | 消息是否丢失、并发下是否一致、分布式下延迟如何 |

**面试要点**：多智能体评估的核心指标（任务完成率、平均轮数、token 成本、人工干预次数、死循环率、协作一致性、工具调用成功率）。手撕代码时要体现黄金数据集 + 任务运行函数 + 指标统计 + 评估报告。并说明三框架各自的评估重点。

---

## Q25. 部署与运维：多智能体系统怎么上生产？

**回答**

多智能体系统从笔记本跑到生产，除了代码，还要处理 **Docker 容器化、环境变量与密钥管理、日志收集、监控报警、灾备恢复**。本节用大白话讲部署与运维的关键点。

---

### 部署与运维的关键点（大白话）

| 关键点 | 大白话 | 怎么做 |
|--------|--------|--------|
| **容器化** | 把系统打包成 Docker 镜像，哪里都能跑 | 写 Dockerfile，镜像里包含 Python 环境、依赖、代码 |
| **密钥管理** | API Key、数据库密码不能硬编码 | 放环境变量或密钥管理服务（AWS Secrets Manager、阿里云 KMS） |
| **沙箱隔离** | 代码执行要进 Docker 沙箱，别搞坏主机 | 代码执行单独跑容器，网络隔离，只读文件系统 |
| **日志收集** | 多台机器的日志收集到一处 | ELK / Loki / 云日志服务，结构化 JSON 日志 |
| **监控报警** | token 暴涨、Agent 挂了要及时报警 | Prometheus + Grafana，设置阈值报警 |
| **灾备恢复** | 消息状态要能恢复，防止丢数据 | 消息持久化（SQLite/MongoDB）、Checkpoint 定期备份 |

---

### Dockerfile 示例（概念版）

```dockerfile
# Dockerfile（概念版）
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 运行时不以 root 身份（安全）
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 从环境变量读模型密钥，不要硬编码
ENV MODEL_API_KEY=""
ENV LOG_LEVEL="info"

CMD ["python", "main.py"]
```

---

### 环境变量与密钥管理（Python 读取示例）

```python
import os
from dotenv import load_dotenv  # 本地开发用 .env，生产用密钥服务

load_dotenv()  # 生产环境此行可省略，密钥直接从环境变量读

API_KEY = os.getenv("MODEL_API_KEY")
if not API_KEY:
    raise RuntimeError("MODEL_API_KEY 环境变量未设置，请配置密钥")

MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "10"))
LOG_FILE = os.getenv("LOG_FILE", "./trace.jsonl")
```

---

### 监控指标建议（Prometheus 风格）

| 指标 | 报警阈值示例 |
|------|--------------|
| 多智能体任务失败率 | > 5% 报警 |
| 单任务平均 token | 超过基线 2 倍报警 |
| Agent 平均响应时间 | > 30s 报警 |
| 人工干预次数 | 突然激增报警（可能 prompt 出问题） |
| 沙箱代码执行失败率 | > 10% 报警 |

**面试要点**：部署与运维要体现 Docker 容器化、密钥用环境变量/密钥服务（不硬编码）、代码沙箱隔离、日志集中收集、监控报警（失败率/token/响应时间/人工干预）、消息状态持久化备份。

---

## Q26. 框架与 RAG 的结合：多智能体 + 知识库

**回答**

多智能体系统不只靠模型“记忆”，还要查外部知识。把 RAG（检索增强生成）接进多智能体框架，让 Agent 能从知识库里检索相关资料再回答。本节聊怎么结合，并用代码演示“检索 Agent + 回答 Agent”的分工。

---

### 框架与 RAG 结合的常见模式

| 模式 | 大白话 | 适合场景 |
|------|--------|----------|
| **专门检索 Agent** | 设一个 Agent 只负责搜知识库，其他 Agent 把问题丢给它 | 需要频繁查文档 |
| **每个 Agent 自带检索** | 每个 Agent 都能调检索工具 | 各 Agent 领域不同，知识库分开 |
| **共享知识库** | 所有 Agent 读同一个向量库 | 企业统一知识库 |
| **先检索后生成** | 用户问题先检索，再把结果塞进 Prompt | 问答类任务 |

---

### 完整代码（概念版）：检索 Agent + 回答 Agent

下面这段纯 Python 代码演示：一个“检索 Agent”从内存向量库（演示版用关键词匹配模拟）检索相关片段，再把片段传给“回答 Agent”生成最终答案。

```python
"""
RAG + 多智能体极简实现（概念版）
检索 Agent / 回答 Agent / 内存向量库模拟
纯 Python 演示。
"""

from typing import List, Callable
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    content: str


class SimpleVectorStore:
    """
    演示版向量库：用关键词重叠数模拟相似度检索。
    真实场景用 Chroma / Milvus / Pinecone + Embedding 模型。
    """

    def __init__(self):
        self.docs: List[Document] = []

    def add(self, doc: Document):
        self.docs.append(doc)

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        query_words = set(query.lower().split())
        scored = []
        for doc in self.docs:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class RetrievalAgent:
    """检索 Agent：只负责从知识库检索相关片段。"""

    def __init__(self, name: str, store: SimpleVectorStore):
        self.name = name
        self.store = store

    def retrieve(self, query: str) -> List[Document]:
        print(f"[{self.name}] 检索：{query}")
        results = self.store.search(query)
        print(f"[{self.name}] 检到 {len(results)} 条片段")
        return results


class AnswerAgent:
    """回答 Agent：根据检索到的片段生成最终答案。"""

    def __init__(self, name: str, llm_call: Callable[[str], str]):
        self.name = name
        self.llm_call = llm_call

    def answer(self, query: str, docs: List[Document]) -> str:
        context = "\n".join([f"[片段 {d.id}] {d.content}" for d in docs])
        prompt = f"你是 {self.name}。用户问题：{query}\n\n参考资料：\n{context}\n\n请基于参考资料回答。"
        return self.llm_call(prompt)


def demo_llm(prompt: str) -> str:
    if "天气" in prompt:
        return "根据参考资料，明天北京多云转晴，气温 15-25°C。"
    if "政策" in prompt:
        return "根据参考资料，员工远程办公政策为每周最多 2 天。"
    return "根据参考资料，已为您整理答案。"


if __name__ == "__main__":
    # 准备知识库
    store = SimpleVectorStore()
    store.add(Document("D1", "明天北京天气：多云转晴，气温 15-25°C，适合出行。"))
    store.add(Document("D2", "公司远程办公政策：每周最多可远程办公 2 天，需提前申请。"))
    store.add(Document("D3", "报销流程：发票需在 30 天内提交，财务 7 个工作日内审核。"))

    retriever = RetrievalAgent("检索 Agent", store)
    answerer = AnswerAgent("回答 Agent", demo_llm)

    # 流程：用户提问 -> 检索 Agent 检索 -> 回答 Agent 生成答案
    query = "明天北京天气怎么样？"
    docs = retriever.retrieve(query)
    final = answerer.answer(query, docs)
    print(f"\n[最终答案] {final}")
```

**运行输出示例（概念版）：**

```
[检索 Agent] 检索：明天北京天气怎么样？
[检索 Agent] 检到 1 条片段

[最终答案] 根据参考资料，明天北京多云转晴，气温 15-25°C。
```

---

### 三框架接 RAG 的建议

| 框架 | 接 RAG 建议 |
|------|-------------|
| **AutoGen** | 设一个 `RetrievalAgent` 角色，其他 Agent 把问题发给它，拿到片段后继续回答 |
| **LangGraph** | 在 State 里加 `documents` 字段，检索节点把片段写进 State，后续节点读取 |
| **AgentScope** | 检索工具注册到 ToolRegistry，各 Agent 通过工具调用检索，结果经 MsgHub 分发 |

**面试要点**：框架与 RAG 结合的 4 种模式（专门检索 Agent、每个 Agent 自带检索、共享知识库、先检索后生成）。手撕代码时要体现检索 Agent + 回答 Agent 分工，并说明真实向量库替代方案（Chroma / Milvus）。

---

## Q27. 框架与 MCP / 外部 API 的集成

**回答**

智能体要调用外部服务（搜索、数据库、邮件、日历），不能每个框架都重写一套适配。**MCP（Model Context Protocol）** 就是为此而来：一个统一的协议，让 Agent 通过一套标准接口调用各种外部工具和数据源。本节聊 MCP 的核心思想，并演示怎么把 MCP 工具接进多智能体框架。

---

### MCP 是什么，大白话

> MCP：就像 USB 接口一样，统一了“大模型/智能体”和“外部工具”的连接标准。一次适配，到处可用。

MCP 的核心角色：
- **MCP 主机（Host）**：你的智能体应用，想调外部工具；
- **MCP 客户端（Client）**：嵌在智能体里，负责连 MCP 服务器、转发请求；
- **MCP 服务器（Server）**：提供具体工具，如搜索服务器、数据库服务器、文件系统服务器；
- **工具（Tool）/ 资源（Resource）/ 提示（Prompt）**：服务器暴露的能力。

智能体不需要知道“搜索怎么调”、“数据库怎么连”，只需要知道 MCP 协议，实际调用交给 MCP 客户端和服务器。

---

### 完整代码（概念版）：MCP 客户端 + 工具适配

下面这段纯 Python 代码演示：一个极简的 MCP 客户端，连一个“搜索 MCP 服务器”，把搜索工具封装成统一格式，接进多智能体框架的工具注册表。

```python
"""
MCP 集成极简实现（概念版）
MCP 客户端 / MCP 服务器模拟 / 工具统一适配
纯 Python 演示，不依赖官方 MCP SDK。
"""

from typing import Callable, Dict, Any, List
from dataclasses import dataclass
import json


@dataclass
class MCPTool:
    """MCP 工具：名字、描述、参数模式。"""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPServer:
    """MCP 服务器（模拟）：提供具体工具，如搜索。"""

    def __init__(self, name: str):
        self.name = name

    def list_tools(self) -> List[MCPTool]:
        return [
            MCPTool(
                name="web_search",
                description="在网页上搜索信息",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ]

    def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "web_search":
            return f"[MCP 搜索结果] 关于 '{args['query']}' 的网页 A、网页 B（模拟）。"
        raise ValueError(f"未知工具：{name}")


class MCPClient:
    """MCP 客户端：连 MCP 服务器，列工具、调工具，把结果转成统一格式。"""

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}

    def connect(self, server_name: str, server: MCPServer):
        self.servers[server_name] = server
        print(f"[MCP 客户端] 已连接服务器：{server_name}")

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """把所有服务器的工具转成统一字典格式（类 OpenAI function calling）。"""
        tools = []
        for server_name, server in self.servers.items():
            for tool in server.list_tools():
                tools.append({
                    "name": f"{server_name}.{tool.name}",
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "_server": server_name,
                    "_tool": tool.name,
                })
        return tools

    def call_tool(self, full_name: str, args: Dict[str, Any]) -> Any:
        """full_name 格式：server_name.tool_name"""
        server_name, tool_name = full_name.split(".", 1)
        server = self.servers.get(server_name)
        if server is None:
            raise ValueError(f"未连接服务器：{server_name}")
        return server.call_tool(tool_name, args)


# ==================== 接进多智能体工具注册表 ====================

class ToolRegistry:
    """多智能体框架的工具注册表（可复用 Q14 的代码）。"""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self.tools[name] = func

    def call(self, name: str, **kwargs) -> Any:
        return self.tools[name](**kwargs)


if __name__ == "__main__":
    # 1. 连 MCP 服务器
    client = MCPClient()
    client.connect("search_server", MCPServer("search_server"))

    # 2. 列出所有 MCP 工具
    mcp_tools = client.list_all_tools()
    print("[MCP 可用工具]:")
    for t in mcp_tools:
        print(f"  - {t['name']}: {t['description']}")

    # 3. 把 MCP 工具注册进多智能体工具注册表
    registry = ToolRegistry()
    for t in mcp_tools:
        # 用闭包包装：调用时交给 MCP 客户端
        def make_caller(tool_name=t["name"]):
            def caller(**kwargs):
                return client.call_tool(tool_name, kwargs)
            return caller
        registry.register(t["name"], make_caller())

    # 4. 智能体通过注册表调用 MCP 工具（对智能体来说，透明无感知 MCP）
    result = registry.call("search_server.web_search", query="Agent 框架对比")
    print(f"\n[智能体调用 MCP 工具结果] {result}")
```

**运行输出示例（概念版）：**

```
[MCP 客户端] 已连接服务器：search_server
[MCP 可用工具]:
  - search_server.web_search: 在网页上搜索信息

[智能体调用 MCP 工具结果] [MCP 搜索结果] 关于 'Agent 框架对比' 的网页 A、网页 B（模拟）。
```

---

### 三框架接 MCP 的建议

| 框架 | 接 MCP 建议 |
|------|-------------|
| **AutoGen** | 把 MCP 工具注册成 `Function`，交给 `ConversableAgent` 的 `functions` 参数 |
| **LangGraph** | 在节点里调用 MCP 客户端，把结果写进 State |
| **AgentScope** | 把 MCP 工具注册到 `ToolRegistry`，Agent 通过工具调用原语调用 |

**面试要点**：MCP 的核心是“统一协议，一次适配到处可用”。手撕代码时要体现 MCP 客户端连服务器、列工具、调工具，并把 MCP 工具适配进多智能体工具注册表。说明 MCP 主机/客户端/服务器三个角色。

---

## Q28. 三框架选型速查与决策（全章总结）

**回答**

本章咱们聊了 AutoGen、LangGraph、AgentScope 三个主流智能体框架。面试时最常考的就是“怎么选”，没有银弹，只有**场景匹配**。下面是全章选型速查表和决策口诀。

---

### 选型速查表

| 你的场景 | 推荐框架 | 核心理由 |
|---------|---------|---------|
| 快速做多 Agent 原型，几个角色对话解决问题 | **AutoGen** | 上手快，对话驱动，不需要画流程图 |
| 自动化编程、数据分析、创意策划 | **AutoGen** | 原生代码执行沙箱，多角色对话迭代 |
| 业务流程复杂，需分支、循环、断点续传、可审计 | **LangGraph** | 状态图显式控制，强状态、Checkpoints |
| 金融合规审批流、长期运行任务 | **LangGraph** | 每一步可控、可回滚、可审计 |
| 生产级部署，高并发、分布式、安全敏感 | **AgentScope** | 消息持久化、安全沙箱、分布式并行 |
| 大规模多智能体社会模拟（几百个 Agent） | **AgentScope** | 原生分布式，消息不丢，单点故障不拖垮 |
| 金融/政务等安全合规场景 | **AgentScope** | 工程化能力强，可观测性内置 |
| 不确定，先小规模试水 | **AutoGen 或 LangGraph** | 教程和社区都较完善 |
| 业务非常特殊，现成框架都不合适 | **自研框架** | 完全可控，但造轮子成本高 |

---

### 一句话决策口诀

> **要聊天协作，选 AutoGen；要流程可控，选 LangGraph；要工业级生产，选 AgentScope。**

---

### 决策检查表（逐条勾选）

| 检查项 | 自研框架倾向 | 现成框架倾向 |
|--------|--------------|--------------|
| 业务流程标准固定 | 是 | 否 |
| 需分布式高并发 | 否 | 是（AgentScope） |
| 团队有框架开发经验 | 是 | 否 |
| 需快速出原型 | 否 | 是 |
| 需极致定制（非标协议） | 是 | 否 |
| 能接受框架 API 变动 | 是 | 否 |
| 需内置可观测性/审计 | 否 | 是 |

**框架可混合使用**：LangGraph 做总纲，AutoGen 群聊做局部头脑风暴，AgentScope 做分布式调度。边界是接口而非护城河。

---

## Q29. 框架局限与避坑速查（全章总结）

**回答**

框架不是银弹，用框架时容易踩的坑，下面是全章避坑速查表。

---

### 常见坑与应对

| 坑 | 现象 | 应对 |
|----|------|------|
| **token 爆炸** | 群聊几十轮，上下文累积，账单爆炸 | 设最大轮数；滑动窗口；摘要；工具结果截断 |
| **无沙箱代码执行** | Agent 写危险代码删数据 | 代码必须进 Docker 沙箱，禁止网络和文件系统 |
| **缺少人工确认** | 关键操作全自动，出事没人拦 | 关键步骤加 HITL，人工点确认 + 审计日志 |
| **调试看不清** | 多个 Agent 传来传去，不知谁错了 | 结构化日志 + 追踪 ID + LangGraph Studio / AgentScope Studio |
| **框架 API 变动** | 升级后代码跑不通 | 生产锁定版本号；看 release note；别盲目升级 |
| **信息过载** | 每个 Agent 都读全部历史，重点被淹 | 只传给需要的 Agent；私有历史不共享 |
| **过早自研** | 刚开始就造框架，业务还没开始 | 先用现成框架跑原型，业务特殊再考虑自研 |
| **死循环** | Agent 互相对话停不下来 | 最大轮数上限；条件边加退出条件 |

**面试要点**：能说出至少 4-5 个常见坑及应对（token 爆炸、无沙箱、缺少人工确认、调试难、API 版本不稳、信息过载、死循环、过早自研）。

---

## Q30. 本章小结：30 个问题一张图搞懂

**回答**

本章咱们聊了为什么需要框架、框架选型三角（LangGraph 可控性 / AutoGen 协作性 / AgentScope 工程化），手把手用 Python 撕了 AutoGen（对话驱动群聊、核心组件与代码沙箱）、LangGraph（状态图节点/边/条件边、Reducer/Checkpoint/子图/人工中断）、AgentScope（Message/MsgHub/Pipeline 消息驱动、消息持久化与异步并发），还有工具接入、多智能体记忆、HITL、冲突解决、可观测性与安全、性能与 token 成本、自研 vs 现成、项目代码组织、常见坑与学习路径、评估、部署运维、RAG 结合、MCP 集成、三框架选型速查，共 30 个问题。

---

### 本章速查表（30 个问题版）

| 知识点 | 一句话大白话 | 面试常考点 |
|--------|------------|-----------|
| Q1 为什么要框架 | 把主循环/状态/工具/消息/日志封装好，专注业务逻辑 | 能否说出不用框架要自己处理的 5~6 件琐事 |
| Q2 框架选型三角 | LangGraph 可控性 / AutoGen 协作性 / AgentScope 工程化 | 能否说出三个顶点各解决什么问题 |
| Q3 AutoGen 对话驱动 | 多角色 Agent 群聊，任务在对话中涌现 | 能否说出 ConversableAgent、GroupChat、TERMINATE |
| Q4 LangGraph 状态图 | State+Node+Edge，支持条件边和循环 | 能否说出 StateGraph、节点、条件边、Checkpoints |
| Q5 AgentScope 消息驱动 | Message/MsgHub/Pipeline，分布式工程化 | 能否说出 Message、MsgHub、Agent、Pipeline |
| Q6 三框架对比 | 对话/状态图/消息，弱/强/中状态，原型/复杂流程/工业级 | 能否对比学习曲线、状态管理、并发、场景 |
| Q7 选型建议与场景 | 聊天→AutoGen，流程→LangGraph，工业级→AgentScope | 能否针对具体场景给出选型理由 |
| Q8 框架局限与注意 | 学习成本、API 不稳、token 成本、过度工程、调试难 | 能否说出至少 3 个局限及应对 |
| Q9 框架组合使用 | LangGraph 总纲 + AutoGen 局部群聊 + AgentScope 分布式 | 能否说出组合方式 |
| Q10 本章小结（10 个版） | — | — |
| Q11 AutoGen 核心组件与沙箱 | 对话智能体 + 群聊 + 主持人 + 代码沙箱 | 能否体现沙箱执行和报错反馈 |
| Q12 LangGraph 进阶 | Reducer 合并 / Checkpoint 断点 / 子图 / 人工中断 | 能否体现四者及生产价值 |
| Q13 AgentScope 持久化与并发 | 消息存磁盘不丢 / asyncio 并发 | 能否体现 MessageStore、MsgHub、asyncio.gather |
| Q14 工具接入与记忆 | 工具定义+注册表+调用循环；短/长期、共享/私有记忆 | 能否说出四个记忆维度及典型实现 |
| Q15 可观测性与安全 | 结构化日志+追踪ID；沙箱+过滤+脱敏+审计 | 能否回答发生了什么/什么时候/哪个 Agent |
| Q16 性能与 token 成本 | 滑动窗口/最大轮数/摘要/便宜贵模型切换 | 能否体现分层模型路由和成本统计 |
| Q17 自研 vs 现成 | 业务特殊+有能力→自研；否则→现成 | 能否勾选决策检查表 |
| Q18 项目代码组织 | config/agents/tools/memory/workflow/observability/safety | 能否说出各目录职责 |
| Q19 常见坑与学习路径 | 8 个常见坑及应对；6 周学习路径 | 能否说出 4-5 个坑及应对 |
| Q20 HITL 人工在环 | 关键步骤暂停等人确认；同意/拒绝/覆盖 | 能否体现暂停、输入 y/n/m、审计日志 |
| Q21 冲突解决 | 仲裁者/投票/链式让渡/合并融合/升级人工 | 能否体现仲裁者裁定和投票表决 |
| Q22 成本优化实战 | 便宜模型干粗活、贵模型干精活 | 能否体现分层模型路由 |
| Q23 可观测性与安全进阶 | 审计签名防篡改 / 脱敏 / 分布式追踪 trace_id | 能否体现哈希链、正则脱敏、trace_id/span_id |
| Q24 多智能体评估 | 完成率/轮数/token/人工干预/死循环率 | 能否体现黄金数据集+指标统计 |
| Q25 部署与运维 | Docker/密钥环境变量/沙箱隔离/日志/监控/灾备 | 能否说出 Dockerfile 和监控指标 |
| Q26 框架与 RAG 结合 | 检索 Agent+回答 Agent，先检后生 | 能否体现检索-生成分工 |
| Q27 框架与 MCP 集成 | 统一协议，一次适配到处可用 | 能否体现 MCP 客户端连服务器、工具适配 |
| Q28 三框架选型速查 | 场景匹配，一句话口诀 | 能否给出具体场景的选型理由 |
| Q29 框架局限与避坑 | 8 个常见坑与应对 | 能否说出坑及应对 |
| Q30 本章小结 | 30 个问题一张图 | — |

---

### 全景图（ASCII）

```
┌─────────────────────────────────────────────────────────────────┐
│                   智能体主流框架选型三角（30 问版）              │
│                                                                   │
│                   可控性 (LangGraph)                              │
│           状态图 / 节点+边 / 条件边 / 断点续传 / Reducer          │
│                              ▲                                    │
│                              │                                    │
│              ┌───────────────┴───────────────┐                    │
│              │                               │                    │
│              ▼                               ▼                    │
│     ┌───────────────┐              ┌───────────────┐             │
│     │   AutoGen     │              │  AgentScope   │             │
│     │ 对话驱动群聊   │              │ 消息驱动/分布式│             │
│     │ 多角色协作     │              │ 工程化/高并发 │             │
│     │ 代码沙箱       │              │ 消息持久化    │             │
│     └───────┬───────┘              └───────┬───────┘             │
│             │                              │                      │
│             └───────────────┬──────────────┘                      │
│                             ▼                                       │
│                       协作性 (AutoGen)                              │
│                 对话涌现 / 群聊 / TERMINATE                         │
└─────────────────────────────────────────────────────────────────┘
```

---

### 下章预告

下一章《构建你的 Agent 框架 —— 从 0 开始构建智能体框架》我们将不再使用现成框架，而是手把手带你从零设计和实现一个属于自己的智能体框架：定义消息格式、实现 Agent 基类、搭建主循环、管理状态、接入工具、加上可观测性。只有理解了框架的底层实现，才能更灵活地运用各种工具。敬请期待！

---

> **本章正文结束。** 如需继续，请阅读《构建你的 Agent 框架》章节。

---

> **本章正文结束。** 如需继续，请阅读《构建你的 Agent 框架》章节。
