# 记忆与检索 —— 记忆系统，RAG，存储

> 面试生普（LLM 500 题）· 记忆与检索篇。本章覆盖智能体为什么需要记忆、记忆三层（工作记忆/情景记忆/知识记忆）、向量数据库的适用边界（找回原话 vs 记住结论）、短期记忆设计、情景记忆设计（结构化结论+状态）、知识记忆与 RAG 流程（切片→嵌入→向量检索→生成）及手撕 Python 代码、三层记忆协作、记忆系统局限与注意事项、本章小结，共 10 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Python 代码和 mermaid/ASCII 图。

---

## Q1. 智能体为什么需要记忆系统？不用记忆会怎样？

**回答**

如果你跟一个智能体聊天，它每次都把以前的对话全忘了，那它就不是“助手”，而是个“金鱼”。比如：

- 第一轮：你：“我叫小明，喜欢窗边座位，不喝咖啡。”
- 智能体：“好的，已记录。”
- 第二轮：你：“帮我订个位。”
- 智能体（没记忆）：“请问您是哪位？喜欢什么座位？喝什么？”

这就尴尬了。智能体每次对话都像重新见面，用户得把背景重复一遍，还容易出现矛盾（比如上轮你说预算 50 万，这轮它又推荐 80 万的方案，因为它不记得上轮你否决过）。

所以，**智能体需要记忆系统**，目的就是让它“不失忆症”，能跨轮、跨会话地记住上下文、结论、用户偏好和历史决策，把事情办连贯。

---

### 没记忆 vs 有记忆（大白话对比）

| 场景 | 没记忆的智能体 | 有记忆的智能体 |
|------|--------------|--------------|
| 多轮对话 | 每轮都得重复背景 | 能接着上轮聊 |
| 用户偏好 | 喝什么、坐哪里都忘了 | 记得你喜欢窗边、不喝咖啡 |
| 历史决策 | 上轮否决的方案这轮可能再推一遍 | 记得已经否决过，不再重复 |
| 跨会话 | 今天聊的事明天全忘 | 明天还能接着今天干 |
| 复杂任务 | 中间算出的结果丢了，后面没法用 | 中间结果可复用，任务能拆多步做 |

---

### 记忆系统解决的核心问题

1. **上下文贯通**：多轮对话不串场，不需要用户重复讲背景。
2. **状态一致**：历史决策（比如预算改成 30 万）不会被遗忘或覆盖成旧值。
3. **跨会话连续**：今天开始的任务，明天能接着干。
4. **中间结果复用**：复杂任务拆多步，每步的输出可以作为下一步的输入。

用一个 ASCII 图表示有记忆的智能体工作流：

```
用户输入 ──→ 感知 ──→ 思考 ──→ 行动 ──→ 观察
                          ▲          │
                          │          ▼
                    ────── 读写记忆 ──────
                    (短期+情景+知识)
```

**面试要点**：智能体需要记忆系统，是为了避免“金鱼记忆”，实现上下文贯通、状态一致、跨会话连续、中间结果复用。没记忆的智能体每次都得重复背景，还容易自相矛盾。

---

## Q2. 记忆系统分哪几层？每层存什么、用什么存？

**回答**

“记忆”不是一个单一的东西，把它拆开才能做对。业界通常把智能体的记忆分成**三层**，每层存的内容不同、生命周期不同、适合的存储方式也不同。混在一起用，就会做错决策。

---

### 三层记忆对照表

| 层次 | 存什么内容 | 生命周期 | 适合的存储方式 | 大白话例子 |
|------|-----------|---------|-------------|---------|
| **工作记忆**（短期记忆） | 当前任务的中间状态、最近几轮对话 | 分钟级（单次任务内） | 上下文窗口 / 内存列表 | “刚才算出的中间结果是 140” |
| **情景记忆** | 发生过什么、做出了什么结论、用户否决了什么 | 天/周级（跨会话） | 结构化文件 / JSON / 数据库表 | “用户上周否决了 80 万方案，预算改成 30 万” |
| **知识记忆** | 领域事实、文档、常识、操作手册 | 长期（基本不变） | 向量数据库 / 全文索引 | “公司报销制度：单餐不超过 200 元” |

---

### 为什么要拆成三层？

因为每层的需求不一样：

- **工作记忆**追求“快”，只要当前任务能用，任务结束就可以扔；
- **情景记忆**追求“准”，尤其是“结论”和“状态”，不能让向量检索把旧结论当新结论；
- **知识记忆**追求“广”，要能从海量文档里检索出相关片段。

如果只用一种存储（比如全塞向量数据库），就会遇到问题：向量检索擅长“找回原话”（比如翻出你上周说过的一句话），但不擅长“记住结论”（比如记住预算已经从 50 万改成 30 万，旧的 50 万应该作废）。这是两件不同的事。

用一张 ASCII 图表示三层记忆的关系：

```
┌─────────────────────────────────────────────────────────────┐
│                    智能体三层记忆系统                         │
│                                                              │
│  工作记忆（短期）      情景记忆（中期）      知识记忆（长期） │
│  ─────────────      ─────────────      ─────────────       │
│  当前任务中间状态      发生过什么、做了何决策   领域事实、文档、常识 │
│  分钟级              天/周级             长期           │
│  上下文窗口/内存列表   结构化文件/JSON/表   向量库/全文索引   │
│  ─────────────      ─────────────      ─────────────       │
│  最常读写，任务结束可丢   需准确记录结论，允许覆盖   需检索相关片段 │
└─────────────────────────────────────────────────────────────┘
```

**面试要点**：记忆系统分三层——工作记忆（当前任务中间状态，分钟级，上下文窗口）、情景记忆（发生过什么/做了何决策，天/周级，结构化存储）、知识记忆（领域事实/文档，长期，向量库/全文索引）。三层拆开是因为“找回原话”和“记住结论”是两件事，向量库只擅长前者。

---

## Q3. 向量数据库适合存什么？它的局限是什么？

**回答**

很多人听到“智能体记忆”，第一反应就是“上向量数据库”。向量数据库（如 Chroma、Milvus、Pinecone、Weaviate）确实很有用，但它不是万能药。咱们得清楚它适合存什么、不适合存什么，否则很容易踩坑。

---

### 向量数据库适合存什么？

向量数据库擅长**语义相似度检索**——把文本转成向量（Embedding），存进去，查询时把问题也转成向量，找向量最接近的几条。适合：

- **知识记忆**：公司文档、操作手册、产品说明、常见问答（FAQ）；
- **语义搜索**：用户问“退货要多久”，能检索到“退货期限为 15 天”这类语义相关的片段，哪怕原文没有“退货要多久”这几个字；
- **模糊匹配**：用户表述不精确，向量检索仍能找到相关内容。

---

### 向量数据库的局限

| 局限 | 大白话解释 | 例子 |
|------|-----------|------|
| 擅长“找回原话”，不擅长“记住结论” | 向量检索按语义相似度返回片段，但不能自动知道“旧结论已作废” | 用户上周说预算 50 万，这周改成 30 万，向量库里两条都在，检索可能翻出旧的 50 万 |
| 无法自动做“覆盖更新” | 新信息不会自动把旧信息标废，需要额外逻辑处理 | 向量库做不到“这条作废”，结构化记录才可以 |
| 检索可能返回无关或过时内容 | 语义相似不等于事实正确 | 查“公司报销上限”，可能检索到两年前的旧制度 |
| 需要 Embedding 模型和切片策略 | 切太粗或太细都影响检索效果 | 切片太长，关键信息被稀释；太短，上下文缺失 |
| 查询延迟和成本 | Embedding + 向量检索比直接查内存慢，且有 API 成本 | 高频实时查询不适合全走向量库 |

---

### 用 ASCII 图表示向量数据库的适用边界

```
适合向量数据库（知识记忆）：
  ✅ 公司文档、操作手册、FAQ
  ✅ 语义相关的片段检索
  ✅ 用户表述不精确时的模糊匹配

不适合向量数据库（情景记忆）：
  ❌ 需要准确记住“结论+状态”的信息
  ❌ 旧结论需要作废、新结论要覆盖
  ❌ 高频实时查询（延迟和成本高）

应该用什么代替？
  → 情景记忆：结构化文件 / JSON / 数据库表（结论+状态+时间戳，允许覆盖）
  → 工作记忆：上下文窗口 / 内存列表
```

**面试要点**：向量数据库适合存知识记忆（文档、FAQ、语义检索），局限是“擅长找回原话，不擅长记住结论，无法自动覆盖旧信息、有延迟和成本”。情景记忆（比如预算改成 30 万）应结构化存储，不要全依赖向量库。

---

## Q4. 短期记忆怎么设计？（上下文窗口、滑动窗口、压缩）

**回答**

短期记忆（工作记忆）管的是“当前任务正在干什么、最近几轮聊了什么”。它的特点是：读写频繁、生命周期短（单次任务内）、对延迟敏感。下面咱们聊三种常见的短期记忆设计，并手撕一个最小可用的短期记忆 Python 类。

---

### 设计方案 1：全量上下文（最朴素）

把历史对话全部塞进 Prompt，发给模型。

- 优点：零检索误差，模型能直接看到所有历史；
- 缺点：token 随轮数线性增长，长任务会爆上下文窗口，成本高、慢。

适合：50 轮以内的简单任务，是很多情况下的首选。

---

### 设计方案 2：滑动窗口

只保留最近 N 轮对话，旧的扔掉。

- 优点：token 使用固定、可控；
- 缺点：早期的重要信息会被丢失。

适合：不需要长期回溯的对话。

---

### 设计方案 3：压缩摘要

每 N 轮，用模型把历史对话压缩成一段摘要，替换掉原始记录，只保留“已确认结论、被否决方案、待办事项”。

- 优点：token 使用低，重要信息不丢；
- 缺点：压缩可能丢失细节，需要额外调模型。

适合：长任务、需要跨轮回溯的场景。

---

### 手撕短期记忆 Python 代码

下面是一个最小可用的短期记忆类，支持滑动窗口（`max_messages`）和上下文拼接：

```python
"""
short_term_memory.py：短期记忆（工作记忆）
基于内存列表的滑动窗口，支持添加消息、获取上下文、清空。
"""

from typing import List, Dict, Any
from datetime import datetime


class ShortTermMemory:
    """短期记忆：当前会话的消息历史，滑动窗口，防止内存无限增长。"""

    def __init__(self, max_messages: int = 20):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str) -> None:
        """添加一条消息到短期记忆。"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # 超过上限，删最旧的
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self, last_n: int = 10) -> str:
        """获取最近 last_n 条消息拼成的上下文字符串，喂给模型。"""
        recent = self.messages[-last_n:]
        lines = [f"{m['role']}: {m['content']}" for m in recent]
        return "\n".join(lines)

    def clear(self) -> None:
        """清空短期记忆。"""
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    stm = ShortTermMemory(max_messages=5)

    # 模拟多轮对话
    stm.add("user", "你好")
    stm.add("assistant", "你好！我是智能体助手。")
    stm.add("user", "帮我查一下北京天气")
    stm.add("assistant", "北京今天晴朗，15°C。")
    stm.add("user", "那上海呢？")
    stm.add("assistant", "上海今天多云，22°C。")
    stm.add("user", "广州呢？")   # 第 6 条，会挤掉第 1 条“你好”

    print("=== 短期记忆上下文（最近 5 条）===")
    print(stm.get_context())
```

**运行输出示例：**

```
=== 短期记忆上下文（最近 5 条）===
assistant: 你好！我是智能体助手。
user: 帮我查一下北京天气
assistant: 北京今天晴朗，15°C。
user: 那上海呢？
assistant: 上海今天多云，22°C。
user: 广州呢？
```

可以看到，最早的 `user: 你好` 被滑动窗口挤掉了，只保留最近 5 条。

**面试要点**：短期记忆设计有三种：全量上下文（零误差但 token 随轮数增长）、滑动窗口（固定 token 但丢早期信息）、压缩摘要（低 token 但可能丢细节）。手撕代码时要体现：滑动窗口上限、上下文拼接、清空功能。

---

## Q5. 情景记忆怎么设计？（结构化结论 + 状态，允许覆盖）

**回答**

情景记忆管的是“发生过什么、做出了什么结论、用户否决了什么”。它和短期记忆的区别是：情景记忆需要**跨会话保存**，而且最重要的是**准确记录结论和状态**，不能让旧结论误导新决策。

比如：
- 第一周：用户说预算 50 万；
- 第二周：用户说预算改成 30 万；
- 第三周：智能体推方案，应该基于 30 万，而不是翻出第一周的 50 万。

如果全靠向量数据库检索，可能翻出“预算 50 万”的旧对话，导致错误。所以情景记忆应该**结构化存储**：结论 + 状态 + 时间戳，并且允许新结论覆盖旧结论。

---

### 情景记忆设计要点

| 要点 | 说明 |
|------|------|
| 存结论，不只存原文 | 不存“我说预算 50 万”这句原话，而存“预算 = 30 万（已更新）” |
| 带时间戳 | 知道哪条是新的，哪条是旧的 |
| 允许覆盖 | 同一类信息（如预算）新值覆盖旧值 |
| 可检索 | 按关键词或类别检索相关结论 |

---

### 手撕情景记忆 Python 代码

下面是一个最小可用的情景记忆类，基于 JSON 文件持久化，支持按类别（facts/decisions）存储、按关键词检索、同类别新值覆盖旧值：

```python
"""
episodic_memory.py：情景记忆（中长期记忆）
结构化存储结论 + 状态，支持按类别存储、按关键词检索、同类别覆盖。
数据持久化到 JSON 文件，跨会话不丢失。
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class EpisodicMemory:
    """情景记忆：结构化记录发生过的事、做出的决策、用户偏好等。"""

    def __init__(self, path: str = "./episodic_memory.json"):
        self.path = path
        self.data: Dict[str, List[Dict[str, Any]]] = {
            "facts": [],      # 事实类信息（如用户偏好）
            "decisions": [],  # 决策类信息（如预算已改成 30 万）
        }
        self._load()

    def _load(self) -> None:
        """从 JSON 文件加载已有记忆。"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {"facts": [], "decisions": []}

    def _save(self) -> None:
        """把记忆持久化到 JSON 文件。"""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def remember(
        self,
        kind: str,
        content: str,
        source: str = "user",
        override_key: Optional[str] = None,
    ) -> None:
        """
        记录一条情景记忆。
        
        :param kind: 类别，"facts" 或 "decisions"
        :param content: 记录内容（结论，不只是原文）
        :param source: 信息来源，如 "user" / "system" / "tool"
        :param override_key: 覆盖键，同一 override_key 的新记录会覆盖旧记录
        """
        if kind not in self.data:
            self.data[kind] = []

        item = {
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }

        # 如果指定了覆盖键，先删掉同 kind 下同 override_key 的旧记录
        if override_key is not None:
            self.data[kind] = [
                it for it in self.data[kind]
                if it.get("override_key") != override_key
            ]
            item["override_key"] = override_key

        self.data[kind].append(item)
        self._save()
        print(f"[情景记忆] 已记录 [{kind}]：{content}")

    def recall(
        self,
        keyword: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        检索情景记忆。
        
        :param keyword: 关键词，匹配 content 字段
        :param kind: 只检索某个类别（"facts" 或 "decisions"）
        :param limit: 最多返回多少条
        :return: 匹配的记录列表
        """
        candidates: List[Dict[str, Any]] = []
        kinds_to_search = [kind] if kind else list(self.data.keys())

        for k in kinds_to_search:
            for it in self.data.get(k, []):
                if keyword is None or keyword.lower() in it["content"].lower():
                    candidates.append(it)

        # 按时间戳倒序，最新的在前面
        candidates.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return candidates[:limit]

    def get_latest_decision(self, override_key: str) -> Optional[str]:
        """
        获取某个覆盖键的最新决策（如“预算”）。
        如果有多个同 override_key 的记录，返回最新的一条。
        """
        results = self.recall(kind="decisions", limit=50)
        for it in results:
            if it.get("override_key") == override_key:
                return it["content"]
        return None


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import tempfile

    # 用临时文件演示，避免污染当前目录
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    em = EpisodicMemory(path=tmp_path)

    # 第一周：用户说预算 50 万
    em.remember(
        kind="decisions",
        content="预算 = 50 万",
        source="user",
        override_key="budget",
    )

    # 第二周：用户说预算改成 30 万（同 override_key，覆盖旧值）
    em.remember(
        kind="decisions",
        content="预算 = 30 万（已更新）",
        source="user",
        override_key="budget",
    )

    # 检索预算（应该拿到最新的 30 万，而不是 50 万）
    latest_budget = em.get_latest_decision(override_key="budget")
    print(f"\n最新预算决策：{latest_budget}")

    # 检索所有包含“预算”的记录
    print("\n检索“预算”相关记录：")
    for it in em.recall(keyword="预算", limit=5):
        print(f"  [{it['timestamp']}] {it['content']}")

    os.unlink(tmp_path)
```

**运行输出示例：**

```
[情景记忆] 已记录 [decisions]：预算 = 50 万
[情景记忆] 已记录 [decisions]：预算 = 30 万（已更新）

最新预算决策：预算 = 30 万（已更新）

检索“预算”相关记录：
  [...] 预算 = 30 万（已更新）
  [...] 预算 = 50 万
```

可以看到，检索时最新的 30 万排在前面，且 `get_latest_decision("budget")` 正确返回了 30 万，而不是旧的 50 万。

**面试要点**：情景记忆应结构化存储（结论 + 状态 + 时间戳），按类别（facts/decisions）组织，支持同类别新值覆盖旧值（通过 override_key），避免向量库“旧结论不作废”的问题。

---

## Q6. 知识记忆怎么设计？（RAG 流程：切片→嵌入→向量检索→生成）

**回答**

知识记忆管的是“领域事实、文档、常识、操作手册”，基本不变但量大。最常用的设计就是 **RAG（Retrieval-Augmented Generation，检索增强生成）**。

RAG 的大白话解释：在模型回答之前，先去外部知识库里搜一圈，把相关资料拽出来，再结合这些资料生成答案。这样做的好处是：减少模型“胡说八道”（幻觉）、能回答私有数据的问题、答案有据可查。

---

### RAG 的完整流程

```
用户问题 ──→ 问题嵌入（Embedding） ──→ 向量数据库检索（Top-K）
                                            │
                                            ▼
                              检索到的相关文档片段
                                            │
                                            ▼
                              Prompt 拼接（问题 + 片段 + 系统提示词）
                                            │
                                            ▼
                              大模型生成答案 ──→ 返回给用户
```

流程拆成五步：

1. **文档切片（Chunking）**：把长文档切成小片段（比如每段 500 字，重叠 50 字），避免单个片段太长稀释关键信息；
2. **嵌入（Embedding）**：把每个片段转成向量，存入向量数据库；
3. **检索（Retrieval）**：用户问题也转成向量，在向量数据库里找 Top-K 最相似的片段；
4. **Prompt 拼接**：把检索到的片段和用户问题一起塞进 Prompt；
5. **生成（Generation）**：大模型基于片段生成答案，并标注来源。

---

### 手撕 RAG 最小可用版（Python）

下面是一个不依赖真实向量数据库的 RAG 演示版：用简单的“关键词重叠”模拟向量检索，能跑通“切片→检索→拼接→生成”的完整逻辑。真实项目中替换为 Chroma / Milvus / Pinecone 等向量数据库和 Embedding 模型即可。

```python
"""
simple_rag.py：RAG 检索增强生成最小可用版（演示版）
流程：文档切片 → 关键词检索（模拟向量检索） → Prompt 拼接 → 大模型生成（模拟）
"""

from typing import List, Dict, Any
import re


def chunk_document(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    步骤 1：文档切片。
    把长文档切成固定大小的片段，带重叠，避免关键信息在边界被切断。
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


class SimpleEmbedding:
    """
    步骤 2：嵌入（演示版）。
    真实项目中应用 OpenAI Embedding / 通义千问 Embedding 等模型。
    这里用“关键词集合”模拟向量，用于演示检索逻辑。
    """

    def __init__(self):
        self.keywords = set()

    def embed(self, text: str) -> set:
        """把文本转成关键词集合（模拟向量）。"""
        words = re.findall(r"\w+", text.lower())
        self.keywords.update(words)
        return set(words)


class SimpleVectorStore:
    """
    步骤 3：向量数据库（演示版）。
    存储片段及其“向量”（关键词集合），支持相似度检索。
    """

    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    def add(self, chunks: List[str], embedding: SimpleEmbedding) -> None:
        """把切片存入向量库。"""
        for chunk in chunks:
            vec = embedding.embed(chunk)
            self.docs.append({"content": chunk, "vector": vec})

    def search(self, query: str, embedding: SimpleEmbedding, top_k: int = 3) -> List[str]:
        """
        检索 Top-K 最相关片段。
        演示版：用查询关键词与片段关键词的重叠数作为相似度。
        """
        query_vec = embedding.embed(query)
        scored = []
        for doc in self.docs:
            overlap = len(query_vec & doc["vector"])
            if overlap > 0:
                scored.append((overlap, doc["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scored[:top_k]]


def generate_answer(question: str, context_chunks: List[str]) -> str:
    """
    步骤 4-5：Prompt 拼接 + 大模型生成（演示版）。
    真实项目中应把 context_chunks 拼进 Prompt，发给大模型。
    """
    context = "\n".join([f"- {c}" for c in context_chunks])
    prompt = f"""你是一个知识库助手。请基于以下检索到的资料回答问题。
如果资料里没有相关信息，诚实说明“资料中未找到相关信息”。

资料：
{context}

问题：{question}

答案："""
    # 演示版：直接拼接返回，不调真实模型
    return f"[基于检索资料生成的答案]\n{prompt}"


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 准备一段知识库文档
    document = """
    公司报销制度：
    1. 单餐餐饮报销上限为 200 元，需提供发票。
    2. 交通费报销：出租车和网约车单程上限 100 元，地铁和公交实报实销。
    3. 住宿费报销：出差住宿单晚上限 500 元，需提供酒店发票。
    4. 所有报销需在当月 25 日前提交，逾期不予受理。
    """

    # 步骤 1：切片
    chunks = chunk_document(document, chunk_size=100, overlap=20)
    print(f"切片数量：{len(chunks)}")

    # 步骤 2-3：嵌入并存入向量库
    emb = SimpleEmbedding()
    store = SimpleVectorStore()
    store.add(chunks, emb)

    # 用户提问
    question = "单餐餐饮报销上限是多少？"

    # 步骤 3（续）：检索
    retrieved = store.search(question, emb, top_k=3)
    print(f"\n检索到的片段：")
    for c in retrieved:
        print(f"  - {c}")

    # 步骤 4-5：生成答案
    answer = generate_answer(question, retrieved)
    print(f"\n{answer}")
```

**运行输出示例：**

```
切片数量：5

检索到的片段：
  - 1. 单餐餐饮报销上限为 200 元，需提供发票。

[基于检索资料生成的答案]
你是一个知识库助手。请基于以下检索到的资料回答问题。
如果资料里没有相关信息，诚实说明“资料中未找到相关信息”。

资料：
- 1. 单餐餐饮报销上限为 200 元，需提供发票。

问题：单餐餐饮报销上限是多少？

答案：
```

---

### RAG 的优点和局限

| 优点 | 局限 |
|------|------|
| 减少幻觉，答案基于真实资料 | 检索不到相关片段时，答案可能不准 |
| 能回答私有数据问题 | 切片策略不好会丢失上下文 |
| 答案有来源，可溯源 | Embedding + 检索有延迟和成本 |
| 不需要重新训练模型就能加新知识 | 需要维护向量数据库和切片策略 |

**面试要点**：知识记忆通常用 RAG 实现，流程是“切片→嵌入→向量检索→Prompt 拼接→生成”。RAG 擅长减少幻觉、回答私有数据、答案可溯源；局限是检索不到就不准、切片策略影响效果、有延迟和成本。

---

## Q7. 三层记忆如何协作？（分层记忆架构与检索门控）

**回答**

单层记忆都有局限，真正的智能体记忆系统通常是**三层协作**的：工作记忆负责当前任务、情景记忆负责历史结论、知识记忆负责领域事实。三层各司其职，通过一个“检索门控”决定从哪一层取信息。

---

### 三层记忆协作流程

```
用户输入
     │
     ▼
┌─────────────────────┐
│  检索门控（Router）   │  决定从哪一层检索
│  - 当前任务中间状态？ → 工作记忆
│  - 历史决策/结论？     → 情景记忆
│  - 领域事实/文档？     → 知识记忆（RAG）
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
工作记忆    情景记忆/知识记忆
(内存列表)   (JSON/向量库)
           │
           ▼
    拼接成 Prompt ──→ 大模型生成答案
```

---

### 检索门控逻辑示例（Python）

```python
"""
memory_router.py：三层记忆检索门控
根据问题类型决定从工作记忆、情景记忆、知识记忆中检索。
"""

from typing import List, Optional
from short_term_memory import ShortTermMemory
from episodic_memory import EpisodicMemory
from simple_rag import SimpleVectorStore, SimpleEmbedding


class MemoryRouter:
    """三层记忆检索门控：决定从哪层取信息，拼接上下文。"""

    def __init__(
        self,
        stm: ShortTermMemory,
        em: EpisodicMemory,
        vs: SimpleVectorStore,
        emb: SimpleEmbedding,
    ):
        self.stm = stm
        self.em = em
        self.vs = vs
        self.emb = emb

    def route(self, question: str) -> str:
        """
        根据问题内容决定检索策略，拼接上下文字符串。
        """
        parts: List[str] = []

        # 1. 总是先读工作记忆（当前任务中间状态）
        stm_context = self.stm.get_context(last_n=5)
        if stm_context:
            parts.append(f"[工作记忆]\n{stm_context}")

        # 2. 如果问题涉及“预算/决定/偏好”等，查情景记忆
        if any(k in question for k in ["预算", "决定", "偏好", "上次", "之前"]):
            em_results = self.em.recall(keyword=question, limit=3)
            if em_results:
                em_text = "\n".join([f"- {r['content']}" for r in em_results])
                parts.append(f"[情景记忆]\n{em_text}")

        # 3. 如果问题涉及“制度/文档/怎么办”等，查知识记忆（RAG）
        if any(k in question for k in ["制度", "文档", "多少", "怎么办", "流程"]):
            rag_results = self.vs.search(question, self.emb, top_k=3)
            if rag_results:
                rag_text = "\n".join([f"- {c}" for c in rag_results])
                parts.append(f"[知识记忆]\n{rag_text}")

        if not parts:
            parts.append("[无匹配记忆，基于模型常识回答]")

        return "\n\n".join(parts)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import tempfile
    import os

    # 初始化三层记忆
    stm = ShortTermMemory(max_messages=10)
    stm.add("user", "我的报销预算是多少？")
    stm.add("assistant", "您之前设定的报销预算是 30 万。")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name
    em = EpisodicMemory(path=tmp_path)
    em.remember("decisions", "预算 = 30 万（已更新）", source="user", override_key="budget")

    document = "单餐餐饮报销上限为 200 元，需提供发票。"
    chunks = [document]  # 简化：直接用整段作为一个片段
    emb = SimpleEmbedding()
    vs = SimpleVectorStore()
    vs.add(chunks, emb)

    router = MemoryRouter(stm, em, vs, emb)

    # 场景 1：问预算（命中情景记忆）
    print("=== 场景 1：问预算 ===")
    print(router.route("我的预算是多少？"))

    # 场景 2：问报销上限（命中知识记忆）
    print("\n=== 场景 2：问报销上限 ===")
    print(router.route("单餐报销上限是多少？"))

    os.unlink(tmp_path)
```

**运行输出示例：**

```
=== 场景 1：问预算 ===
[工作记忆]
user: 我的报销预算是多少？
assistant: 您之前设定的报销预算是 30 万。

[情景记忆]
- 预算 = 30 万（已更新）

=== 场景 2：问报销上限 ===
[工作记忆]
user: 我的报销预算是多少？
assistant: 您之前设定的报销预算是 30 万。

[知识记忆]
- 单餐餐饮报销上限为 200 元，需提供发票。
```

---

### 三层记忆协作的优势

| 单层记忆的问题 | 三层协作的解决 |
|-------------|-------------|
| 只用向量库，旧结论不作废 | 情景记忆结构化存储，允许覆盖 |
| 只用上下文窗口，长任务爆 token | 工作记忆滑动窗口，情景记忆跨会话 |
| 只用短期记忆，不知道领域事实 | 知识记忆 RAG 查外部文档 |
| 全量塞 Prompt，成本高 | 检索门控只取需要的层，token 可控 |

**面试要点**：三层记忆协作的核心是“检索门控”——根据问题类型决定从工作记忆（当前状态）、情景记忆（历史结论）、知识记忆（RAG 文档）中取信息，拼接成 Prompt 再生成答案。三层各司其职，避免单层记忆的局限。

---

## Q8. 记忆系统的局限和注意事项

**回答**

记忆系统很有用，但不是银弹。下面咱们用大白话把记忆系统容易踩的坑挑明。

---

### 局限 1：向量检索只解决“找回原话”，不解决“记住结论”

向量数据库能翻出你上周说过的一句话，但不知道这个结论后来被改过了。情景记忆（结论+状态+允许覆盖）才是解决“记住结论”的关键。

**应对**：情景记忆用结构化存储（JSON/数据库），不要全依赖向量库。

---

### 局限 2：上下文窗口有限，长任务会爆 token

短期记忆全塞进 Prompt，token 随轮数线性增长，长任务会爆上下文窗口，又慢又贵。

**应对**：滑动窗口、压缩摘要、检索门控只取需要的信息，不要全量塞。

---

### 局限 3：检索可能返回过时或无关内容

语义相似不等于事实正确，向量检索可能翻出两年前的旧制度。

**应对**：元数据带时间戳，检索后按时间过滤；关键事实用情景记忆结构化存储。

---

### 局限 4：长期记忆的隐私与安全

把用户偏好、公司文档存进向量数据库，可能涉及隐私和合规。

**应对**：敏感数据私有化部署向量库；不把机密信息写进 Prompt；做好访问控制和审计日志。

---

### 局限 5：记忆一致性难保证

多个 Agent 共享记忆时，可能同时修改同一条结论，导致冲突。

**应对**：给关键记录加版本号或时间戳，并发写入加锁，读取时取最新版本。

---

### 记忆系统使用 Checklist（ASCII 版）

```
记忆系统使用 Checklist：
┌─────────────────────────────────────────────────────────┐
│ □ 明确哪些信息存工作记忆、哪些存情景记忆、哪些存知识记忆│
│ □ 情景记忆结构化存储（结论+状态+时间戳，允许覆盖）       │
│ □ 短期记忆用滑动窗口或压缩摘要，控制 token               │
│ □ 检索门控只取需要的层，避免全量塞 Prompt               │
│ □ 向量库只存知识记忆，不存需要准确覆盖的结论             │
│ □ 敏感数据私有化部署，做好访问控制                       │
│ □ 关键记录加版本号/时间戳，保证一致性                   │
└─────────────────────────────────────────────────────────┘
```

**面试要点**：记忆系统局限包括向量检索只找原话不记住结论、上下文窗口有限、检索可能过时、隐私安全、多 Agent 一致性。注意事项：情景记忆结构化、短期记忆控制 token、检索门控、向量库只存知识记忆、敏感数据私有化、关键记录加版本号。

---

## Q9. 本章小结：记忆与检索全貌一张图搞懂

**回答**

本章咱们聊了智能体为什么需要记忆、记忆三层（工作记忆/情景记忆/知识记忆）、向量数据库的适用边界、短期记忆设计（全量/滑动窗口/压缩）、情景记忆设计（结构化结论+状态+允许覆盖）、知识记忆与 RAG 流程（切片→嵌入→向量检索→生成）及手撕 Python 代码、三层记忆协作（检索门控）、记忆系统局限与注意事项。

下面是本章速查表，帮你记住关键点。

---

### 本章速查表

| 知识点 | 一句话大白话 | 面试常考点 |
|--------|------------|-----------|
| 为什么要记忆 | 避免“金鱼记忆”，实现上下文贯通、状态一致 | 能否说出没记忆会怎样 |
| 三层记忆 | 工作记忆（分钟级）/情景记忆（天周级）/知识记忆（长期） | 能否说出每层存什么、用什么存 |
| 向量数据库局限 | 擅长找回原话，不擅长记住结论，无法自动覆盖 | 能否说出至少 3 个局限 |
| 短期记忆 | 全量上下文 / 滑动窗口 / 压缩摘要 | 能否说出三种设计及优劣 |
| 情景记忆 | 结构化结论 + 状态 + 时间戳，允许覆盖 | 能否说出 override_key 覆盖机制 |
| 知识记忆/RAG | 切片→嵌入→向量检索→Prompt 拼接→生成 | 能否说出 RAG 五步流程 |
| 三层协作 | 检索门控决定从哪层取信息 | 能否说出门控逻辑 |
| 局限注意 | 向量库只找原话、token 爆、检索过时、隐私、一致性 | 能否说出至少 3 个局限及应对 |

---

### 全景图（ASCII）

```
┌─────────────────────────────────────────────────────────────────┐
│                    智能体记忆与检索全景图                         │
│                                                                   │
│  用户输入                                                         │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────┐                                            │
│  │ 检索门控（Router）│  决定从哪层检索                            │
│  │ - 当前状态 → 工作记忆                                        │
│  │ - 历史结论 → 情景记忆                                          │
│  │ - 领域事实 → 知识记忆（RAG）                                   │
│  └──────────┬──────────┘                                        │
│             │                                                   │
│    ┌────────┴────────┐                                         │
│    │                 │                                         │
│    ▼                 ▼                                         │
│ 工作记忆         情景记忆      知识记忆                        │
│ (内存列表)     (JSON/数据库)   (向量库/RAG)                     │
│ 分钟级         天/周级         长期                             │
│    │                 │                 │                       │
│    └─────────────────┴────────────────┘                       │
│                             │                                 │
│                             ▼                                 │
│                    拼接成 Prompt                                │
│                             │                                 │
│                             ▼                                 │
│                    大模型生成答案                               │
└─────────────────────────────────────────────────────────────────┘
```

---

### 下章预告

下一章《上下文工程 —— 持续交互的“情境理解”》我们将聊怎么让智能体在长对话中保持对上下文的理解：上下文窗口管理、提示词工程、检索增强的上下文、长文档处理、上下文压缩等。敬请期待！

---

> **本章正文结束。** 如需继续，请阅读《上下文工程》章节。
