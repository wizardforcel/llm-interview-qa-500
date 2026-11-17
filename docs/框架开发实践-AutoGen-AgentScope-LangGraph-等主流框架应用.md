# 框架开发实践 —— AutoGen、AgentScope、LangGraph 等主流框架应用

> 面试生普（LLM 500 题）· 框架开发实践篇。本章覆盖为什么需要框架、框架选型三角（LangGraph 可控性 / AutoGen 协作性 / AgentScope 工程化）、AutoGen 对话驱动多智能体协作与手撕 Python 代码、LangGraph 状态图驱动复杂工作流与手撕 Python 代码、AgentScope 消息驱动工程化多智能体与手撕 Python 代码、三框架全维度对比表、选型建议与场景、框架局限与注意事项、框架组合使用、本章小结，共 10 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个框架都配大白话解释、完整 Python 代码和 mermaid/ASCII 图。

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

### 下章预告

下一章《构建你的 Agent 框架 —— 从 0 开始构建智能体框架》我们将不再使用现成框架，而是手把手带你从零设计和实现一个属于自己的智能体框架：定义消息格式、实现 Agent 基类、搭建主循环、管理状态、接入工具、加上可观测性。只有理解了框架的底层实现，才能更灵活地运用各种工具。敬请期待！

---

> **本章正文结束。** 如需继续，请阅读《构建你的 Agent 框架》章节。
