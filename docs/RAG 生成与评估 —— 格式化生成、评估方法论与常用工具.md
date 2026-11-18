# RAG 生成与评估 —— 格式化生成、评估方法论与常用工具

> 面试生普（LLM 500 题）· RAG 生成与评估篇。本章覆盖 RAG 生成阶段的格式化输出（结构化输出、JSON Mode/Structured Outputs/Function Calling、Output Parser、三层手段、受限解码、应用层校验重试、开源模型方案）以及 RAG 评估方法论（RAG Triad 三个必测指标、检索侧与生成侧指标、RAGAS/TruLens/Langfuse/Promptfoo/DeepEval 工具、LLM-as-Judge、Golden Set 构建、离线与线上评估、常见失败模式与修复、评估阈值与 CI/CD 集成），共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Python 代码和 ASCII 图。

---

## Q1. 什么是格式化生成？RAG 里为什么需要？

**回答**

格式化生成（Structured Output / Formatted Generation）是指让大模型不再只输出一段自由文本，而是输出符合特定结构的数据，比如 JSON、XML、或调用工具的参数。

在 RAG 里，自由文本输出常常不够用。举几个实际场景：

1. **RAG 驱动的电商客服**：用户问“推荐几款适合程序员的键盘”，我们希望模型返回一个包含产品名称、价格、特性、购买链接的 JSON 列表，方便前端直接渲染成商品卡片，而不是一段描述性文字。
2. **自然语言转 API 调用**：用户说“帮我查明天上海到北京的航班”，系统需要解析成结构化请求 `{"departure": "上海", "destination": "北京", "date": "2025-07-18"}`。
3. **结构化答案与引用**：RAG 回答时希望模型输出 `{ "answer": "...", "citations": ["doc_1", "doc_3"] }`，方便前端展示并标注来源。

格式化生成是连接 LLM 的自然语言理解能力和下游程序化逻辑的关键桥梁。没有结构化输出，RAG 系统很难和其他业务系统集成。

> 面试要点：格式化生成＝让 LLM 输出机器可读的结构化数据（JSON/Schema/Function Calling）。RAG 里需要它，因为下游系统（前端渲染、API 调用、数据库写入）需要结构化数据，而不是一段人话。

---

## Q2. 模型自由文本输出 vs 系统需要结构化数据，矛盾在哪？

**回答**

模型的本职是“接龙”——接在用户 prompt 后面，生成最像人话的下一个 token。训练语料里，人写 JSON 时几乎都会在前后加几句解释（“好的，以下是您需要的 JSON 数据：……希望对您有帮助！”），模型照着学，自然也会顺手加上客套话。

但是工程系统需要的是一个可直接解析、可校验、可进数据库的返回值。模型说的是“给人听的话”，系统要的是“能被程序直接接住的数据”。

典型崩溃现场：

```
模型输出：
好的，以下是您需要的 JSON 数据：
{ "name": "张三", "score": 92 }

希望对您有帮助！

程序端：
json.loads(response)  →  当场炸了（前后客套话不是合法 JSON）
```

把模型当成被要求“说说这个项目”的人：只给一张白纸，他能写满半页（背景、进度、注意事项、判断混在一起）；给一张印好格子的表（项目名、负责人、上线日期），他就只能一格一格往里填。**约束不是限制表达，是把输出收进你要的格子里。**

> 面试要点：模型擅长自由文本（“给人听的话”），系统需要结构化数据（“能被程序直接接住的数据”）。两者矛盾在于模型天然爱加客套话和解释，导致直接 json.loads 经常崩。需要通过约束手段把模型输出收进固定格子。

---

## Q3. 让模型输出结构化数据，有哪几层手段？约束强度怎么递增？

**回答**

工程上让模型输出合规结构化数据，有三层递进手段，一层比一层硬：

| 层级 | 手段 | 保证什么 | 优点 | 缺点 |
|------|------|---------|------|------|
| 第一层 | Prompt 提示约束 | 最弱，靠模型“自觉” | 简单，无需特殊 API 支持 | 无保证，经常崩，需重试 |
| 第二层 | JSON Mode / Structured Outputs / 受限解码 | 语法层硬约束，100% 合法 JSON/Schema | 质变，无需反复重试 | 只保证语法，不保证语义（值可能是幻觉） |
| 第三层 | 应用层校验 + 错误反馈重试 | 语义层兜底（业务规则校验） | 兜底业务逻辑错误 | 增加延迟和成本 |

大白话比喻：

- 第一层像“口头约定”：拜托模型只输出 JSON，大部分时候遵守，但没有保证。
- 第二层像“印好格子的表”：模型在采样每个 token 前，先对照语法规则算出“此刻哪些 token 是合法的”，把不合法的候选概率归零，再在允许集合里采样。这是**受限解码（constrained decoding）**，物理层面保证语法合法。
- 第三层像“业务校验员”：拿到 JSON 后，用 Pydantic 等校验分数在 0～100 之间、日期是工作日、枚举值在允许清单里；校验失败时把报错喂回模型重试，告诉它“哪条规则没过”。

> 面试要点：三层手段＝Prompt 提示约束（最弱）→ JSON Mode/Structured Outputs/受限解码（语法层硬约束）→ 应用层校验+重试（语义层兜底）。第二层保证语法合法，不保证内容真实；第三层兜底业务语义。

---

## Q4. JSON Mode（弱约束）是什么？和 Structured Outputs 的区别？

**回答**

JSON Mode 是模型 API 提供的弱约束模式：设置 `response_format = {"type": "json_object"}` 后，模型保证输出是**合法且可解析的 JSON**，但**不保证符合任何特定结构定义**。

比如你要求字段 `name`（字符串）和 `age`（整数），JSON Mode 只能保证返回的是个合法 JSON 对象，但模型仍可能返回 `{ "name": "张三", "age": "三十岁" }`（age 成了字符串），或者多一个字段、少一个字段。

**Structured Outputs / Response Schema（强约束）** 则进一步保证输出**100% 符合你给定的 JSON Schema**（字段、类型、必填项等）。OpenAI 2024 年 8 月的 Structured Outputs 是这个领域的分水岭：API 层面通过受限解码，强制保证 schema 符合率 100%。

| 特性 | JSON Mode | Structured Outputs |
|------|-----------|-------------------|
| 保证合法 JSON | ✅ | ✅ |
| 保证符合指定 Schema | ❌ | ✅ |
| 需要 Schema 输入 | ❌ | ✅ |
| 约束强度 | 弱 | 强 |

> 面试要点：JSON Mode 只保证输出是合法 JSON，不保证字段、类型、必填项；Structured Outputs 进一步保证 100% 符合给定 JSON Schema。生产环境优先用 Structured Outputs，JSON Mode 只适合对结构要求不高、容错高的场景。

---

## Q5. Structured Outputs / Response Schema 是什么？OpenAI 2024-08 为什么是分水岭？

**回答**

Structured Outputs（OpenAI 叫 Response Schema / Structured Outputs API）是模型 API 提供的强约束能力：开发者传入一个 JSON Schema（描述 JSON 应该长什么样：有哪些字段、各是什么类型、哪些必填），模型在 token 采样阶段就被物理限制只能生成符合该语法的 token，**100% 保证输出符合 Schema**。

2024 年 8 月 OpenAI 推出 Structured Outputs 之前，开发者只能靠“Prompt + 解析 + 重试”，工程上很脏，合 schema 率往往只有 70%～90%。2024-08 之后，API 层面直接保证 100% schema 符合，范式从“靠模型自觉”转为“API 层硬约束”。

Anthropic（Tool Use）、Gemini（JSON mode）随后跟进；开源模型则靠 Outlines、Instructor、SGLang CFSM、vLLM guided decoding 补齐。

> 面试要点：Structured Outputs 通过 JSON Schema + 受限解码，API 层保证 100% 输出符合 Schema。OpenAI 2024-08 是分水岭，在此之前靠 Prompt+重试（脏、不稳定），之后 API 层硬约束。Anthropic/Gemini 跟进，开源模型用 Outlines/Instructor/SGLang 补齐。

---

## Q6. Function Calling / Tool Use（参数结构化）是什么？

**回答**

Function Calling（OpenAI 以前叫），Anthropic 叫 Tool Use，是指模型不直接回复文本，而是输出一个“调用某个函数/工具”的结构化请求，包含函数名和按 Schema 组织的参数。

```
用户：“帮我查明天上海到北京的航班”
模型输出（不是人话，是工具调用）：
{
  "name": "search_flights",
  "arguments": { "departure": "上海", "destination": "北京", "date": "2025-07-18" }
}
```

程序收到后，根据函数名去调用真实的航班查询 API，把结果再返回给模型生成最终回答。

Function Calling 本质上是“参数结构化”——Agent 工具的 Schema 就用这个机制定义。和 Structured Outputs 一样，设置 `strict: true` 后，生成的参数能 100% 符合提供的结构定义。

> 面试要点：Function Calling / Tool Use 让模型输出“调用工具+结构化参数”而不是纯文本，是 Agent 调用外部工具的核心机制。`strict: true` 后参数 100% 符合 Schema。与 Structured Outputs 同属强约束能力，区别在于 Function Calling 额外包含“调用哪个工具”的语义。

---

## Q7. 四种结构化输出机制对比（约束强度递增）

**回答**

| 机制 | 名称 | 约束强度 | 实现原理 | 适用场景 |
|------|------|---------|---------|---------|
| 机制 1 | Prompt-based JSON | 最弱 | 靠 prompt 让模型“自觉”只输出 JSON | 小规模、简单 schema、容错高、快速验证想法 |
| 机制 2 | JSON Mode | 弱 | `response_format = {"type": "json_object"}`，保证合法 JSON | 需要合法 JSON 但不严格关心字段/类型 |
| 机制 3 | Structured Outputs / Response Schema | 强 | JSON Schema + 受限解码，API 层保证 100% 符合 Schema | 生产环境、对字段类型必填项有严格要求 |
| 机制 4 | Function Calling / Tool Use | 强 | 模型输出工具调用 + 参数，`strict: true` 保证参数符合 Schema | Agent 调用外部工具、自然语言转 API 调用 |

**选型建议**：生产环境优先用机制 3（Structured Outputs）或机制 4（Function Calling）；机制 2 适合只关心“是个合法 JSON”的场景；机制 1 只适合快速原型验证，不建议上线。

> 面试要点：四种机制按约束强度递增。生产环境选机制 3（Structured Outputs）或机制 4（Function Calling），机制 1（Prompt-based）不建议上线，机制 2（JSON Mode）仅适合对结构要求不高的场景。

---

## Q8. LangChain Output Parser 有哪些？代码示例长什么样？

**回答**

LangChain 提供了 `OutputParser` 组件，用于把模型输出的纯文本字符串解析成预期的结构化数据（如 Python 对象）。常用的有：

- `StrOutputParser`：最基础，直接把输出当字符串返回。
- `JsonOutputParser`：解析包含嵌套结构和列表的复杂 JSON 字符串。
- `PydanticOutputParser`：与 Pydantic 模型结合，实现最严格的格式定义和验证，是生产环境推荐。

`PydanticOutputParser` 的工作原理：把用户定义的 Pydantic 数据模型转成详细的格式指令，注入到 Prompt 里引导模型生成符合该结构的 JSON；再把模型返回的 JSON 字符串安全解析为 Pydantic 对象。

### Python 示例

```python
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

# 1. 定义期望的数据结构
class PersonInfo(BaseModel):
    """用于存储个人信息的数据结构。"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")

# 2. 基于 Pydantic 模型，创建解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. 创建提示模板，注入格式指令
prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. 创建处理链（假定 llm 已初始化）
# chain = prompt | llm | parser

# 5. 执行调用（示例，llm 部分需自行接入）
text = "张三今年30岁，他擅长Python和Go语言。"
# result = chain.invoke({"text": text})
# print(result)  # name='张三' age=30 skills=['Python', 'Go语言']
```

格式指令会自动注入，类似这样：

```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

{"properties": {"age": {"description": "人物年龄", "type": "integer"}, "name": {"description": "人物姓名", "type": "string"}, "skills": {"description": "技能列表", "items": {"type": "string"}, "type": "array"}}, "required": ["age", "name", "skills"], "type": "object"}
```

> 面试要点：LangChain Output Parser 把模型纯文本输出解析成结构化数据。`PydanticOutputParser` 最推荐生产环境使用，它通过 Pydantic 模型自动生成格式指令注入 Prompt，再把返回的 JSON 安全解析为 Pydantic 对象。注意解析器只是“事后解析”，模型仍可能输出非 JSON，生产上建议配合第二层（Structured Outputs）和第三层（应用层校验重试）。

---

## Q9. LangChain `with_structured_output` 有哪几种方法？代码示例？

**回答**

`with_structured_output` 是 LangChain 聊天模型的包装器，把模型输出格式化为匹配给定 Schema 的结构。支持三种 `method`：

| 方法 | 含义 |
|------|------|
| `function_calling` | 用 OpenAI 的工具调用 API（formerly function calling） |
| `json_schema` | 用 OpenAI 的 Structured Outputs API（推荐，严格模式） |
| `json_mode` | 用 OpenAI 的 JSON Mode（只保证合法 JSON，不保证 Schema） |

### Python 示例（OpenAI）

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class AnswerWithJustification(BaseModel):
    """带理由的回答。"""
    answer: str
    justification: str | None = Field(default=None, description="回答的理由")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 推荐：json_schema 方法，strict=True，API 层保证 100% 符合 Schema
structured_model = model.with_structured_output(
    AnswerWithJustification,
    method="json_schema",
    include_raw=False,
    strict=True,
)

result = structured_model.invoke("一磅铁和一磅羽毛哪个重？")
# result 是 AnswerWithJustification 实例，字段已按 Schema 校验
print(result.answer)           # "They weigh the same"
print(result.justification)    # "Both a pound of bricks and a pound of feathers weigh one pound."
```

`include_raw=True` 时返回包含 `raw`（原始消息）、`parsed`（解析后对象）、`parsing_error`（解析错误）的字典，便于调试。

> 面试要点：`with_structured_output` 支持 `function_calling`、`json_schema`、`json_schema` 三种方法。生产推荐 `method="json_schema"` + `strict=True`，API 层保证 100% 符合 Schema。`include_raw=True` 可返回原始消息和解析错误，便于排查。

---

## Q10. Prompt-based JSON 的常见陷阱有哪些？

**回答**

只靠 Prompt 说“只输出 JSON，不要任何其他文字”，模型大部分时候遵守，但没有保证。常见失败模式：

1. **非 JSON**：输出里夹带解释性文本前缀（“Sure, here is...”）、“好的，以下是……”，导致 `json.loads` 直接崩。
2. **JSON 但不符合 Schema**：多字段、少字段、类型错（比如 `age` 应该是整数，模型给了字符串“三十岁”）。
3. **格式偏差**：用单引号、有尾随逗号、被 Markdown 代码块包裹（```json ... ```）。
4. **嵌套结构不完整**：复杂嵌套对象里缺字段或层级错乱。

生产统计：朴素 Prompt 的 JSON 合法率约 70%～90%，合 Schema 率更低；上游解析失败率 5%～30%，对生产系统不可接受。

> 面试要点：Prompt-based JSON 的陷阱包括输出非 JSON、JSON 不符合 Schema、格式偏差（单引号/尾随逗号/Markdown 包裹）、嵌套不完整。合法率 70～90%、解析失败率 5～30%，生产上不能只靠 Prompt，必须配合第二层（Structured Outputs）和第三层（应用层校验重试）。

---

## Q11. 应用层校验 + 重试怎么做？代码示例？

**回答**

第三层手段：拿到模型输出后，用 Pydantic 等校验业务规则（分数在 0～100 之间？日期是工作日？枚举值在允许清单里？）。校验失败时，**不要原样重新问一遍**（模型大概率还输出同样的错误），要把报错一起喂回模型，告诉它“哪条规则没过”，它才知道改哪里。

### Python 示例

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class ScoreOutput(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)  # 分数必须在 0～100 之间

def generate_with_retry(llm_call, prompt: str, max_retries: int = 3):
    last_error = None
    for attempt in range(max_retries):
        raw = llm_call(prompt)  # 调用模型，拿到原始文本
        try:
            # 先做结构化解析（第二层），再做业务校验（第三层）
            import json
            data = json.loads(raw)
            result = ScoreOutput(**data)  # Pydantic 校验业务规则
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            # 把报错喂回模型，构造重试 prompt
            prompt = f"""{prompt}

上一次输出有误，请修正。错误信息：{last_error}
请只输出修正后的 JSON，不要其他文字。"""
    raise RuntimeError(f"结构化输出重试 {max_retries} 次均失败，最后错误：{last_error}")


# 使用示例
# result = generate_with_retry(
#     llm_call=lambda p: my_llm.generate(p),
#     prompt="请给出一个学生姓名和0-100的分数，只输出 JSON。"
# )
```

关键点：重试 prompt 里要包含具体的错误信息，而不是原样重发。

> 面试要点：应用层校验＋重试＝用 Pydantic 校验业务规则，失败时把具体错误喂回模型重试（不要原样重发）。这是语义层兜底，保证输出不仅语法合法，还符合业务逻辑。

---

## Q12. 开源模型怎么做结构化输出？常用方案有哪些？

**回答**

开源模型（如 Llama、Qwen、BGE 等）通常不自带 OpenAI 那样的 Structured Outputs API，需要靠社区方案补齐：

| 方案 | 原理 | 特点 |
|------|------|------|
| **Outlines** | 受限解码（constrained decoding），基于正则/语法约束生成 | 每生成一个 token 前，对照语法规则屏蔽不合法候选，硬保证语法 |
| **Instructor** | Pydantic + 自动重试 | 把 Pydantic 模型转成 Prompt 指令，解析失败自动重试，工程好用 |
| **SGLang CFSM** | 有限状态机约束生成 | 高性能，适合 SGLang 推理引擎 |
| **vLLM guided decoding** | vLLM 原生支持 guided decoding（JSON/正则/结构化） | 推理服务层原生支持，延迟低 |

选型：追求开发效率用 Instructor（Pydantic + 重试）；追求推理性能和硬约束用 Outlines/SGLang/vLLM guided decoding。

> 面试要点：开源模型结构化输出靠 Outlines（受限解码硬约束）、Instructor（Pydantic+重试）、SGLang CFSM、vLLM guided decoding。开发效率优先 Instructor，推理性能优先 Outlines/SGLang/vLLM。

---

## Q13. 流式结构化输出怎么处理？（增量展示 vs 全量消费）

**回答**

流式输出时，模型是 token 一个个吐出来的。对于结构化输出，有两种消费方式：

1. **增量展示**：给用户看“正在生成中的内容”，适合人工阅读场景（比如聊天界面）。但结构化输出增量展示会显示不完整的 JSON（比如还没闭合的大括号），体验较差。
2. **全量消费**：等整个结构化输出生成完，再一次性解析、校验、入库。适合程序下游消费（API 接口、数据库写入）。

生产建议：如果下游是程序消费，**不要流式解析结构化输出**，等全量拿到后再做 `json.loads` 和 Pydantic 校验；如果是给最终用户看，可以流式展示原始文本，但结构化字段只在全量完成后展示。

> 面试要点：流式结构化输出分“增量展示”（给人看，适合聊天界面）和“全量消费”（给程序用，适合 API/数据库）。生产程序下游应等全量完成后再解析校验，避免解析不完整的 JSON。

---

## Q14. 语法合法不等于语义正确，第三层应用层校验的意义是什么？

**回答**

第二层（Structured Outputs / 受限解码）只保证**语法层面**合法：输出百分百是合法 JSON、字段齐全、类型正确。但它不保证**内容**真实：`score` 字段的值可能是模型现编的 999（虽然类型是 int，但在业务上不合理），日期可能是周末、枚举值可能不在允许清单里。

这就是幻觉问题的另一面：模型能按格式输出，但内容可能编造。第三层应用层校验就是在这个基础上，再校验业务语义：

- 分数在 0～100 之间？
- 日期是工作日？
- 枚举值在允许清单里？
- 引用的文档 ID 真的存在于知识库？

校验失败时，把具体错误喂回模型重试，或触发兜底逻辑（告知用户无法生成合规答案）。

> 面试要点：语法合法≠语义正确。第二层只保证 JSON 语法和 Schema 结构，不保证值合理；第三层应用层校验（Pydantic 业务规则＋错误反馈重试）是语义层兜底，防止模型按格式输出幻觉内容。

---

## Q15. RAG 里格式化生成的典型应用场景有哪些？

**回答**

| 场景 | 结构化输出示例 | 作用 |
|------|-------------|------|
| 结构化问答答案 | `{ "answer": "...", "citations": ["doc_1", "doc_3"] }` | 答案和来源分离，前端可标注引用 |
| 电商产品推荐 | `[{"name": "...", "price": "...", "features": [...]}, ...]` | 前端直接渲染商品卡片 |
| 自然语言转 API | `{"departure": "上海", "destination": "北京", "date": "..."}` | 解析成结构化 API 请求，调用后端系统 |
| 信息抽取 | 从文档抽取事件、时间、地点、人物，存入数据库 | 非结构化文档 → 结构化数据库记录 |
| Agent 工具调用 | `{"name": "search_flights", "arguments": {...}}` | Agent 调用外部工具的核心机制 |

> 面试要点：RAG 里格式化生成的典型应用包括结构化问答答案（答案＋引用）、电商产品推荐（前端渲染卡片）、自然语言转 API（结构化请求）、信息抽取（文档→数据库）、Agent 工具调用。

---

## Q16. 为什么 RAG 必须有评估体系？光靠“人工抽检”不行吗？

**回答**

很多团队做 RAG 优化全凭感觉——“加了 Rerank 好像更准了”、“换了 Embedding 模型好像更好了”。但这无法回答面试官和工程的核心问题：

1. **无法量化优化效果**：不知道改动让效果提升了 5% 还是 50%，也不知道在哪些场景提升了、哪些变差了。
2. **无法对比不同方案**：A 方案混合检索，B 方案加重排序，到底哪个更适合当前业务？没有评估只能猜。
3. **无法发现退化**：加了新功能后，老场景效果有没有变差？没有评估发现不了。
4. **无法回归测试**：每次改 chunking、Embedding、Prompt，都要重新跑一遍评估，确认没退步。

只有建立“自动化、可量化、可追溯”的评估体系，RAG 优化才是工程而非盲猜。

> 面试要点：RAG 必须有评估体系，因为“人工抽检”不可扩展、不可量化、不可对比、无法发现退化、无法回归测试。评估体系把“感觉更好了”变成“Recall@K 从 0.70 提升到 0.85”。

---

## Q17. RAG 质量分哪两层？怎么理解“检索质量 × 生成质量”？

**回答**

RAG 质量分为两层相乘：

1. **检索质量**：能不能把相关的文档片段检索出来，排在前面。衡量指标：Recall@K、Precision@K、MRR、NDCG。
2. **生成质量**：拿到检索结果后，大模型能不能基于这些材料生成准确、有依据、不幻觉的答案。衡量指标：Faithfulness（忠实度/扎实性）、Answer Relevancy（答案相关性）、Context Relevance（上下文相关性）。

公式直觉：

```
最终答案质量 ≈ 检索质量 × 生成质量
```

如果检索质量是 0（相关文档都没检出来），那生成质量再高也没用——证据不在上下文里，模型只能胡编。这就是为什么面试中强调“先优化检索，再优化生成”。

ASCII 示意：

```
检索质量 (Recall/Precision/MRR/NDCG)        生成质量 (Faithfulness/Answer Relevancy)
        │                                               │
        ▼                                               ▼
   能不能找到相关片段                              找到后能不能基于片段准确作答
        │                                               │
        └─────────────┬─────────────────────────────────┘
                      ▼
                最终 RAG 答案质量
```

> 面试要点：RAG 质量两层＝检索质量（能不能找到相关片段）× 生成质量（找到后能不能准确作答）。两层相乘，检索为 0 时生成再高也没用。面试回答要分两层讲，别只盯着最终答案。

---

## Q18. RAG Triad 三个必测指标分别是什么？

**回答**

RAG Triad（RAG 三剑客）是评估 RAG 质量最核心的三个指标，至少前三个必测：

| 指标 | 含义 | 不需要标准答案 | 低分意味着 |
|------|------|-------------|-----------|
| **Groundedness / Faithfulness（扎实性/忠实度）** | 回答是否基于召回的材料，有没有幻觉 | ❌ 不需要 | 模型在编造，超出了检索材料 |
| **Context Relevance（上下文相关性）** | 召回的材料对这个问题是不是相关 | ❌ 不需要 | 检索太宽，噪声多 |
| **Answer Relevancy（答案相关性）** | 回答是否真的回答了用户的问题 | ❌ 不需要 | 答非所问，或者只答了部分 |

这三个指标都不需要标准答案（Ground Truth），基于 LLM-as-Judge 即可自动化评估，适合快速迭代。

> 面试要点：RAG Triad 三个必测指标＝Faithfulness（是否基于材料，无幻觉）、Context Relevance（召回材料是否相关）、Answer Relevancy（是否答到点）。三者都不需要标准答案，可用 LLM-as-Judge 自动化评估，是快速迭代的基线。

---

## Q19. Context Recall（上下文召回）和 Citation Accuracy（引用准确性）分别是什么？

**回答**

**Context Recall（上下文召回）**：检索到的材料是否覆盖了回答问题所需的全部关键信息。得分 1.0 表示没有遗漏关键信息；得分接近 0 表示知识库里有关键信息但没被检出来。**需要标准答案（Ground Truth）**才能计算：把标准答案里的每句话，去检查检索到的上下文是否覆盖。

**Citation Accuracy（引用准确性）**：回答里标注的引用出处，是否真的支持该主张。比如模型说“单餐报销不超过 200 元，根据《报销制度》第 3 页”，要核查第 3 页里是否真的有这句话。

这两个指标比 RAG Triad 更难测（需要标准答案或人工核查），适合在有 Golden Set 后做回归测试。

> 面试要点：Context Recall 衡量检索是否覆盖了回答所需全部关键信息（需标准答案）；Citation Accuracy 衡量标注的引用出处是否真的支持主张。两者都比 RAG Triad 更难自动化，适合 Golden Set 回归测试。

---

## Q20. 检索侧评估用哪些指标？Recall@K、Precision@K、MRR、NDCG 分别怎么算？

**回答**

检索侧评估关注：正确答案片段有没有出现在检索结果里，排在什么位置。

| 指标 | 含义 | 直觉 |
|------|------|------|
| **Recall@K** | 前 K 个结果里包含正确答案的比例 | “该找的找到了吗” |
| **Precision@K** | 前 K 个结果里相关内容的比例 | “找到的有多少是相关的” |
| **MRR** | 第一个正确结果排名的倒数平均值 | “找到得够不够靠前” |
| **NDCG@K** | 综合排名位置和相关度等级的排序质量 | “整个排序列表质量如何” |

### Python 示例（Recall@K）

```python
def recall_at_k(retriever, eval_set, k=10):
    """
    eval_set: [(query, set_of_relevant_doc_ids), ...]
    """
    hits = 0
    for query, relevant_ids in eval_set:
        retrieved_ids = set(retriever.search(query, top_k=k))
        if retrieved_ids & relevant_ids:
            hits += 1
    return hits / len(eval_set)


eval_set = [
    ("单餐报销上限是多少？", {"chunk_0"}),
    ("出差住宿怎么报销？", {"chunk_1"}),
]
r = recall_at_k(my_retriever, eval_set, k=5)
print(f"Recall@5 = {r:.2%}")
```

**阈值参考（生产参考）：** Recall@10 ≥ 0.70 及格，0.85 良好，0.95+ 优秀；MRR ≥ 0.50 及格，0.70 良好。

> 面试要点：检索侧核心指标＝Recall@K（覆盖）、Precision@K（纯度）、MRR（首位排名）、NDCG@K（排序质量）。Recall@K 是 RAG 最终答案准确率的上限，低于预期应先优化检索。

---

## Q21. RAGAS 四大核心指标详解？代码示例？

**回答**

RAGAS（Retrieval Augmented Generation Assessment）是目前最流行的 RAG 评估框架，核心思想是“用 LLM 来评估 LLM”。四大核心指标：

| 指标 | 需标准答案 | 含义 |
|------|-----------|------|
| **Faithfulness（忠实度）** | ❌ | 答案是否忠实于上下文，有没有幻觉 |
| **Answer Relevancy（答案相关性）** | ❌ | 答案是否真的回答了问题 |
| **Context Precision（上下文精确度）** | ❌ | 检索材料是否相关，且相关材料是否排在前面 |
| **Context Recall（上下文召回）** | ✅ | 检索材料是否覆盖了标准答案的全部关键信息 |

Faithfulness 的两阶段实现：
- 第一阶段：LLM 把答案拆成原子级事实声明。
- 第二阶段：对每个声明，LLM 判断是否能从上下文找到证据支撑。
- 最终：被支撑的声明数 / 总声明数。

### Python 示例

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

df = {
    "question": ["单餐报销上限是多少？"],
    "answer": ["单餐报销不超过200元，需提供发票。"],
    "contexts": [["单餐报销不超过200元，需提供发票。"]],
    "ground_truth": ["单餐报销上限为200元。"],
}
dataset = Dataset.from_dict(df)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
# 输出示例：{'faithfulness': 1.0, 'answer_relevancy': 1.0, ...}
```

> 面试要点：RAGAS 四大核心指标＝Faithfulness（忠实度，无标准答案）、Answer Relevancy（答案相关性，无标准答案）、Context Precision（上下文精确度，无标准答案）、Context Recall（上下文召回，需标准答案）。Faithfulness 用两阶段 LLM（声明提取→支撑验证）实现。

---

## Q22. 主流 RAG 评估工具对比？怎么选？

**回答**

| 工具 | 定位 | 特点 | 适用场景 |
|------|------|------|---------|
| **RAGAS** | 开源、Python 原生 | 内置 LLM-as-Judge，四大核心指标，最流行 | 自动化离线评估、CI 回归测试 |
| **TruLens** | 开源 | 类似 RAGAS，但 trace/观测能力更强，每次 RAG 调用有完整 span | 需要链路追踪和事后分析 |
| **Langfuse** | LLM 可观测性产品 | 评估是其中一块，和 Prompt 管理一起用 | 生产可观测性 + 评估结合 |
| **Promptfoo** | 配置文件驱动 | 适合 CI 跑，用 YAML 定义测试和断言（llm-rubric） | CI/CD 质量门禁 |
| **DeepEval** | 开源测试套件 | RAG 特定指标 + 通用 LLM 评估，类似 pytest 风格 | 单元测试式评估 |

**选型建议**：快速搭自动化评估选 RAGAS；需要链路追踪选 TruLens；生产环境做可观测性选 Langfuse；CI 质量门禁选 Promptfoo 或 DeepEval。

> 面试要点：主流工具＝RAGAS（开源自动化评估，最流行）、TruLens（trace/观测更强）、Langfuse（可观测性+评估）、Promptfoo（CI 配置文件驱动）、DeepEval（pytest 风格测试套件）。按需求选：快速上手选 RAGAS，CI 门禁选 Promptfoo/DeepEval，生产可观测选 Langfuse。

---

## Q23. 什么是 LLM-as-Judge？靠谱吗？

**回答**

LLM-as-Judge 是指用一个强大的大模型（评委模型）来评估另一个模型的输出质量，而不是靠人工逐条打分。RAGAS 的 Faithfulness、Context Precision 等，底层就是 LLM-as-Judge。

**靠谱吗？** 业界共识是：LLM-as-Judge 的可靠度可接受（与人工判断一致率 85%+），足够作为**自动化基线**和**快速迭代**的参考，但不能完全替代人工抽检，尤其对关键业务场景还需要人工复核。

**避免通胀分的技巧**：评委模型不要和生成模型用同一个。如果 GPT-4o 生成答案又由 GPT-4o 打分，分数会虚高；应该用不同模型或不同尺寸的模型做评委。

> 面试要点：LLM-as-Judge＝用强大模型评估另一个模型输出，可靠度可接受（85%+ 与人工一致），适合做自动化基线和快速迭代，但不能完全替代人工抽检。评委模型应与生成模型不同，避免分数通胀。

---

## Q24. Golden Set（黄金评估集）怎么构建？需要多少条？

**回答**

Golden Set 是评估体系的核心，没有好评估集，工具再多也是玩具。构建步骤：

1. **数量**：至少 50 条才能得出初步结论；生产上线前建议 200+ 条，才能得到稳定、可行动的数字。
2. **分布对齐真实流量**：抓线上 200～500 条真实 query，标注正确答案 + 应被引用的 chunk。
3. **覆盖维度**：问题类型（事实/比较/总结）× 难度（简单/中等/困难）× 语种。
4. **难度分层**：
   - Tier 1：直接事实查询（“产品 A 的价格”），要求 > 95%
   - Tier 2：需多 chunk 合成（“对比 A 和 B”）
   - Tier 3：需推理/多轮（“根据历史数据推测…”），要求 > 70% 就很好
   - Tier 4：反例/越界（“怎么黑进系统？”），应 100% 安全拒答
5. **维护**：每季度抽样新 query 加入；发现生产 bad case → 加入评估集，防止同类问题重复。

> 面试要点：Golden Set 构建＝至少 50 条（生产 200+）→ 抓线上真实 query 并标注正确答案和引用 chunk → 覆盖问题类型×难度×语种 → 分 Tier 1～4 设不同达标标准 → 每季度更新并把生产 bad case 加入。没有好 Golden Set，评估结果不可信。

---

## Q25. 离线评估和线上评估的区别？

**回答**

| 维度 | 离线评估 | 线上评估 |
|------|---------|---------|
| 时机 | 部署前，在测试集上跑 | 部署后，线上流量中抽样 |
| 目的 |  catching 回归，对比方案，质量门禁 | 发现线上只在大规模流量下出现的问题 |
| 数据 | Golden Set（标注好的） | 真实用户 query（部分需人工或 LLM 打分）|
| 指标 | Recall@K、Faithfulness 等完整指标 | 采样的 Faithfulness、答案相关性、幻觉率、用户负反馈率 |
| 频率 | 每次改动前（CI/CD 集成） | 持续监控，告警 |

**两者缺一不可**：离线评估在部署前拦截退化；线上评估捕捉只有真实流量才会暴露的问题（如新 query 类型、分布漂移）。

> 面试要点：离线评估用 Golden Set，部署前跑完整指标做质量门禁；线上评估用真实流量抽样，持续监控采样指标和用户负反馈。离线拦截退化，线上捕捉分布漂移，两者互补。

---

## Q26. 常见 RAG 评估失败模式和修复方法有哪些？

**回答**

| 现象（Symptom） | 根本原因（Root Cause） | 修复方法（Fix） |
|----------------|----------------------|----------------|
| 低 Faithfulness + 高 Context Precision | 模型忽略了检索到的上下文，在胡编 | 加强 Prompt 指令“只基于提供的上下文回答”，加显式约束 |
| 低 Context Recall | 分块把相关信息切太碎，或 Top-K 太小 | 增大 chunk_size、改语义分块、增大 Top-K |
| 低 Context Precision | Embedding 相似度阈值太松，或 Top-K 太大 | 减小 Top-K、提高相似度阈值、加重排序 |
| 高 Context Recall 但低 Faithfulness | 生成模型偏离上下文，用参数化知识覆盖 | 换指令遵循更强的模型，收紧系统 Prompt |
| 低 Answer Relevancy | 检索返回的是相关但不同主题的材料 | 加查询改写/多查询扩展，再检索 |

> 面试要点：常见失败模式和修复＝低 Faithfulness（加 Prompt 约束）→低 Context Recall（增大 chunk/Top-K）→低 Context Precision（减小 Top-K/加重排序）→高 Recall 低 Faithfulness（换强指令遵循模型）→低 Answer Relevancy（加查询改写）。评估发现问题后，要对症下药，别只盯着最终分数。

---

## Q27. 评估阈值怎么设？CI/CD 集成怎么做？

**回答**

**阈值设置**：保守起见，先设较低阈值，证明系统可用后再逐步提高。比如 Faithfulness 先设 0.70，不要一上来就 0.90，否则每次构建都失败，开发者会绕过评估。

示例阈值（可按业务调整）：
- Faithfulness ≥ 0.75
- Answer Relevancy ≥ 0.70
- Context Precision ≥ 0.60
- Context Recall ≥ 0.70

**CI/CD 集成**：在 GitHub Actions 等 CI 中加入评估步骤，使用评估工具 CLI（如 RAGAS/Promptfoo/DeepEval）。退出码 0 表示通过，1 表示失败，直接集成到流水线。可设置 `--exit-on-first-failure` 加快明显回归的构建；全量质量报告则去掉该标志，让所有测试跑完。

**黄金数据集版本控制**：把 Golden Set 和代码放在同一个版本控制里。每次改 chunking 策略或换 Embedding 模型，回归测试会立即告诉你质量是提升还是下降。

> 面试要点：阈值先保守后提高（如 Faithfulness 先 0.70）；CI/CD 集成用评估工具 CLI，退出码 0/1 控制流水线通过与否；Golden Set 与代码同版本控制，每次结构性改动都要跑回归测试。

---

## Q28. 生产环境 RAG 监控指标有哪些？告警阈值参考？

**回答**

线上指标不能只靠离线评估，需要持续抽样监控：

| 指标 | 告警阈值（参考） | 触发时检查什么 |
|------|----------------|---------------|
| Faithfulness（采样） | < 0.75 | 最近文档摄入质量、Prompt 是否被改 |
| Answer Relevancy（采样） | < 0.70 | 查询分布是否发生漂移 |
| 幻觉率 | > 5% | 检索覆盖是否对新 query 类型不足 |
| P95 检索延迟 | > 500 ms | 索引规模、Embedding 模型加载、重排序开销 |
| Context Utilization（上下文利用率） | < 40% | 检回 5 条但答案只引用 1 条，Top-K 太大或重排序无效 |
| 用户负反馈率 | > 10% | 综合以上所有 |

**Context Utilization 值得重视**：检回 5 个 chunk 但答案只引用了 1 个，说明在为无用的检索付费，要么 Top-K 太大，要么重排序过滤不好。

> 面试要点：生产监控指标＝采样 Faithfulness（<0.75 告警）、采样 Answer Relevancy（<0.70）、幻觉率（>5%）、P95 检索延迟（>500ms）、Context Utilization（<40%）、用户负反馈率（>10%）。Context Utilization 少被提及，但能发现为无用检索付费的问题。

---

## Q29. 怎么把“评估发现问题”转化为具体优化动作？

**回答**

评估不是为了出分数，而是为了找优化方向。通用闭环：

```
跑评估 → 找低分指标 → 对照失败模式表定位根本原因 → 采取对应优化 → 重新跑评估确认没退步
```

举例：

- Faithfulness 低 → 检查是否模型忽略上下文 → 加强 Prompt 约束或换强指令遵循模型 → 重跑 Faithfulness。
- Context Recall 低 → 检查分块是否太细或 Top-K 太小 → 增大 chunk_size/Top-K 或改语义分块 → 重跑 Context Recall。
- Context Precision 低 → 检查是否检回太多噪声 → 减小 Top-K、加重排序 → 重跑 Context Precision。

关键原则：每次只改一个变量，重跑完整评估集确认“这个指标上升了，其他指标没退步”。

> 面试要点：评估→定位根本原因→对症优化→重跑评估确认，构成闭环。每次只改一个变量，避免“这个指标上去了，那个指标掉下来”了。

---

## Q30. 本章小结：一个初学者该怎么做 RAG 生成与评估？

**回答**

生成与评估是 RAG 从“能跑”到“好用”的关键。初学者建议按这个路径入门：

1. **格式化生成**：先用 `with_structured_output`（json_schema + strict=True）或 `PydanticOutputParser`，让模型输出结构化 JSON；生产上加第三层应用层校验＋重试，兜底语义错误。
2. **建立评估基线**：跑通 RAGAS，重点测 RAG Triad 三个指标（Faithfulness、Context Relevance、Answer Relevancy），不用标准答案，可快速迭代。
3. **构建 Golden Set**：至少 50 条真实 query，标注正确答案和引用 chunk，按 Tier 1～4 分层设达标标准。
4. **分层评估**：检索侧测 Recall@K/Precision@K/MRR/NDCG，生成侧测 Faithfulness/Answer Relevancy/Context Recall，两层分开看，定位问题。
5. **CI/CD 集成**：把评估加到构建流水线，退出码 0/1 控制是否发布；阈值先保守后提高。
6. **线上监控**：持续抽样监控 Faithfulness、幻觉率、P95 延迟、Context Utilization、用户负反馈率。

一句话总结：**格式化生成让 RAG 结果能被程序接住，评估体系让 RAG 优化有据可依。三层结构化输出手段＋RAG Triad 评估＋Golden Set 回归测试＋CI/CD 质量门禁，是把 RAG 从 Demo 打磨成生产系统的标准工程路径。**

> 面试要点：入门路径＝结构化输出（with_structured_output/PydanticParser＋应用层校验）→ RAGAS 建基线（RAG Triad 三指标）→ Golden Set 构建（50～200+ 条，Tier 分层）→ 分层评估（检索侧＋生成侧分开）→ CI/CD 集成（退出码控制发布）→ 线上监控（采样指标＋用户负反馈）。
