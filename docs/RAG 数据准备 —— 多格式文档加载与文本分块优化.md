# RAG 数据准备 —— 多格式文档加载与文本分块优化

> 面试生普（LLM 500 题）· RAG 数据准备篇。本章覆盖 RAG 数据准备的全流程：多格式文档加载（PDF、Word、Markdown、TXT、HTML/网页、CSV/Excel）、文档加载器选型与元数据注入、文本清洗（去噪、去重、去页眉页脚）、批量加载与目录遍历、错误处理（单文件失败不中断批次）、以及文本分块的五大策略（固定大小、递归、语义、结构化、父子分块）与 chunk_size/chunk_overlap 选型，共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Python 代码和 ASCII 图。

---

## Q1. RAG 的“数据准备”阶段具体做什么？为什么它这么重要？

**回答**

数据准备是 RAG 索引阶段的前半段，负责把杂乱无章的原始文档，变成向量数据库能吃的“干净食材”。它包含三件大事：

1. **加载文档**：把 PDF、Word、Markdown、TXT、网页、CSV 等各种格式的文件读进来，转成纯文本。
2. **清洗文本**：去掉页眉页脚、乱码、重复内容、无关广告等噪声，只留有用信息。
3. **切分文本（分块）**：把长文档切成合适大小的片段（Chunk），为后面的向量化和检索做准备。

用大白话比喻：RAG 像开饭店，数据准备就是“买菜＋洗菜＋切菜”。买错菜（加载错格式）、菜不洗干净（没清洗）、切得太碎或太大（分块不合理），最后做出来的菜（模型回答）都难吃。

ASCII 图示：

```
原始文档                    数据准备                    向量数据库
┌─────────┐                 ┌─────────────┐             ┌─────────┐
│ 产品手册.pdf│─────────────▶│ 加载→清洗→分块 │───────────▶│ [片段A] │
│ 制度.docx  │               │（多格式支持） │             │ [片段B] │
│ FAQ.md    │                 └─────────────┘             │ [片段C] │
└─────────┘                                               └─────────┘
                                              (干净、适中大小、带元数据)
```

> 面试要点：数据准备＝加载（多格式）→清洗（去噪）→分块（切成适当大小）。它是 RAG 的“隐性天花板”：切得太碎语义残缺、切得太大向量被稀释，后面所有环节都在补救。

---

## Q2. RAG 常见的数据源有哪些？分别适合什么场景？

**回答**

RAG 的数据源非常广泛，常见的有：

| 数据源 | 示例格式 | 适合场景 |
|--------|---------|---------|
| 本地文档 | PDF、Word (.docx)、Markdown (.md)、TXT | 公司手册、产品说明书、HR 制度、技术文档 |
| 网页/网站 | HTML、URL | 官网帮助中心、公开文档、博客文章 |
| 结构化数据 | CSV、Excel (.xlsx)、JSON | 产品目录、FAQ 表格、订单记录 |
| 协作平台 | Confluence、Notion、飞书文档、企业微信文档 | 团队协作文档、知识库 |
| 代码仓库 | Git 仓库、代码文件、README | 技术文档、API 说明、代码注释 |
| 数据库 | MySQL、PostgreSQL 等业务表 | 结构化业务数据（常配合 Text2SQL 使用） |

ASCII 示意（多源汇聚）：

```
                    ┌──────────────┐
                    │   RAG 知识库 │
                    │（向量数据库）│
                    └──────┬───────┘
                           ▲
      ┌────────────────────┼────────────────────┐
      │                    │                    │
   本地文档              网页/网站            结构化数据
  (PDF/Word/MD/TXT)      (HTML/URL)           (CSV/Excel/JSON)
      │                    │                    │
      └────────────────────┴────────────────────┘
                  统一加载 → 清洗 → 分块 → 向量化 → 入库
```

> 面试要点：常见数据源包括本地文档（PDF/Word/MD/TXT）、网页（HTML/URL）、结构化数据（CSV/Excel/JSON）、协作平台文档、代码仓库、业务数据库等。不同源用不同的加载器，但都要统一成“纯文本＋元数据”再进入后续流水线。

---

## Q3. 怎么加载 PDF 文档？常用的 Python 库有哪些？

**回答**

PDF 是 RAG 里最常见也最麻烦的格式之一，因为 PDF 里的文字可能是“可选取的文本层”，也可能是“扫描图片”（没法直接选文字）。

常用的 Python 库：

| 库 | 特点 | 适合场景 |
|-----|------|---------|
| pypdf (原 PyPDF2) | 轻量、纯 Python、速度快 | 文本层清晰的普通 PDF |
| PyMuPDF (fitz) | 功能强，可提取文本、图片、页码、坐标 | 需要页码、表格位置等元数据的场景 |
| pdfplumber | 表格提取能力强 | 带表格的 PDF |
| Unstructured | 支持复杂排版、图片、表格，自动识别 | 排版复杂、带图表的 PDF |

### Python 示例（pypdf，最简单入门）

```python
from pypdf import PdfReader

reader = PdfReader("公司报销制度.pdf")
texts = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    texts.append(f"[第{i+1}页] {text}")

full_text = "\n".join(texts)
print(full_text[:500])
```

> 面试要点：PDF 加载常用 pypdf（轻量入门）、PyMuPDF（需页码/坐标）、pdfplumber（表格）、Unstructured（复杂排版）。注意扫描版 PDF 无法直接提取文本，需要先做 OCR（如 Tesseract）再处理。

---

## Q4. 怎么加载 Word (.docx) 文档？

**回答**

Word 文档用 `python-docx` 库加载，可以提取段落文本、表格、标题等。它把文档按“段落”组织，逐段读取并拼接即可。

### Python 示例（python-docx）

```python
from docx import Document

doc = Document("公司薪酬制度.docx")

texts = []
for para in doc.paragraphs:
    if para.text.strip():  # 跳过空段落
        texts.append(para.text)

# 也可以提取表格
for table in doc.tables:
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        texts.append(" | ".join(cells))

full_text = "\n".join(texts)
print(full_text[:500])
```

> 面试要点：Word 文档用 `python-docx` 逐段读取文本，也可遍历表格提取单元格内容。注意 `.doc`（老版本 Word）不被 `python-docx` 支持，需要先转成 `.docx` 或用其他工具。

---

## Q5. 怎么加载 Markdown (.md) 和纯文本 (.txt) 文件？

**回答**

Markdown 和 TXT 是最简单的格式，直接用 Python 内置的 `open()` 读取即可，无需第三方库。Markdown 里的标题符号（#）、列表符号（-）可以保留（给模型提供结构提示），也可以去掉（只保留纯文本）。

### Python 示例

```python
# 读取 Markdown
with open("公司请假制度.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# 读取 TXT
with open("员工手册.txt", "r", encoding="utf-8") as f:
    txt_text = f.read()

print(md_text[:300])
```

> 面试要点：Markdown 和 TXT 直接用 Python `open(encoding="utf-8")` 读取即可，是 RAG 数据准备中最简单、最可靠的格式，建议优先把内部文档沉淀为 Markdown，方便后续处理。

---

## Q6. 怎么加载网页 (HTML) 内容？

**回答**

网页加载分两步：先下载网页 HTML，再把 HTML 里的正文提取出来，去掉导航栏、页脚、广告等噪声。

常用库：`requests`（下载网页）+ `BeautifulSoup`（解析 HTML、提取正文）。

### Python 示例

```python
import requests
from bs4 import BeautifulSoup

url = "https://example.com/help/faq"
resp = requests.get(url, timeout=10)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

# 去掉脚本和样式，避免把 JS/CSS 当正文
for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()

# 提取正文文本
text = soup.get_text(separator="\n", strip=True)

# 去掉空行
lines = [line for line in text.splitlines() if line.strip()]
clean_text = "\n".join(lines)

print(clean_text[:500])
```

> 面试要点：网页加载用 `requests` 下载 + `BeautifulSoup` 提取正文，记得移除 `<script>`、`<style>`、`<nav>`、`<footer>`、`<header>` 等非正文标签，否则噪声会干扰后续检索。

---

## Q7. 怎么加载 CSV 和 Excel 结构化数据？

**回答**

CSV 和 Excel 是结构化数据，每一行是一条记录。RAG 里通常把每一行转成一段描述性文字，再向量化。

- CSV：用 Python 内置 `csv` 或 `pandas` 读取。
- Excel (.xlsx)：用 `pandas`（需要 `openpyxl` 引擎）。

### Python 示例（pandas）

```python
import pandas as pd

# CSV
df_csv = pd.read_csv("产品目录.csv")

# Excel（需 pip install openpyxl）
df_xlsx = pd.read_excel("订单记录.xlsx", engine="openpyxl")

# 把每一行转成一段自然语言描述
descriptions = []
for _, row in df_csv.iterrows():
    desc = f"产品名称：{row['产品名称']}，价格：{row['价格']}元，库存：{row['库存']}件。"
    descriptions.append(desc)

for d in descriptions[:3]:
    print(d)
```

> 面试要点：CSV 用 `pandas.read_csv`，Excel 用 `pandas.read_excel`（需 `openpyxl`）。结构化数据建议按行转成自然语言描述（如“产品名称：…，价格：…”），再参与向量化和检索，元数据可保留原始字段（价格、库存等）用于后续过滤。

---

## Q8. 什么是文档元数据？为什么要注入元数据？

**回答**

元数据是“关于文档的数据”，比如文档名称、格式、所属部门、页码、上传时间、作者等。

注入元数据的价值：

1. **检索追溯**：检索到答案后，可通过元数据快速定位原文（哪个文档、第几页），提升答案可信度。
2. **分类管理**：按文档类型、部门对文档分类，便于后续筛选（比如只搜“人力资源部”的文档）。
3. **上下文补充**：把元数据作为上下文的一部分输入模型，让模型了解背景（比如“这是《报销制度》第 3 页的内容”）。

ASCII 示意（带元数据的片段）：

```
检索到片段：单餐报销不超过 200 元，需提供发票。
元数据：{
  "doc_name": "报销制度.pdf",
  "page": 3,
  "department": "财务部",
  "upload_time": "2026-01-15"
}
                                          ↑
                      模型回答可标注：“根据《报销制度》第3页（财务部）”
```

> 面试要点：元数据＝文档名称、页码、部门、上传时间等。注入元数据后，检索结果可追溯到原文位置，支持按部门/类型筛选，还能作为上下文补充提升回答准确性。

---

## Q9. 用 LangChain 批量加载多种格式的文档并注入元数据，代码长什么样？

**回答**

下面是一个完整示例，批量加载 PDF、Word、Markdown 三种格式的公司制度文档，并为每个文档注入自定义元数据（名称、格式、部门、上传时间）。

先安装依赖：

```bash
pip install langchain pypdf docx2txt unstructured python-markdown
```

完整代码：

```python
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.schema import Document

def load_multiple_documents(doc_info_list: List[Dict]) -> List[Document]:
    """
    批量加载多种格式文档，并注入自定义元数据。
    doc_info_list 每个元素包含：path、format、name、department
    """
    documents = []

    for doc_info in doc_info_list:
        path = doc_info["path"]
        doc_format = doc_info["format"]
        name = doc_info["name"]
        department = doc_info["department"]

        # 1. 根据格式选择对应的加载器
        try:
            if doc_format == "pdf":
                loader = PyPDFLoader(path)
            elif doc_format == "docx":
                loader = Docx2txtLoader(path)
            elif doc_format == "md":
                loader = UnstructuredMarkdownLoader(path)
            else:
                print(f"不支持的文档格式：{doc_format}，跳过 {path}")
                continue

            # 2. 加载文档（每个文档可能生成多个 Document，比如 PDF 每页一个）
            doc_list = loader.load()

            # 3. 注入自定义元数据
            for doc in doc_list:
                doc.metadata.update({
                    "doc_name": name,
                    "doc_format": doc_format,
                    "department": department,
                    "upload_time": "2026-11-18",
                })
                # 统一页码格式（如果有）
                if "page" in doc.metadata:
                    doc.metadata["page"] = f"第{doc.metadata['page']}页"

            documents.extend(doc_list)
            print(f"成功加载：{name}（{path}），共 {len(doc_list)} 页/段")

        except Exception as e:
            # 单文件失败不中断整批处理，记录错误后继续
            print(f"加载 {path} 失败：{str(e)}，已跳过。")
            continue

    return documents


# 使用示例
if __name__ == "__main__":
    docs_info = [
        {"path": "policy.pdf",  "format": "pdf",  "name": "公司考勤制度", "department": "人力资源部"},
        {"path": "policy.docx", "format": "docx", "name": "公司薪酬制度", "department": "财务部"},
        {"path": "policy.md",   "format": "md",   "name": "公司请假制度", "department": "人力资源部"},
    ]

    all_docs = load_multiple_documents(docs_info)
    print(f"\n总共加载文档片段：{len(all_docs)} 个")
    for d in all_docs[:2]:
        print(f"--- {d.metadata['doc_name']} · {d.metadata.get('page','')} ---")
        print(d.page_content[:200])
```

> 面试要点：LangChain 批量加载多格式文档的模式是“根据后缀选加载器 → load() → 统一注入元数据 → 异常捕获（单文件失败不中断批次）”。元数据包括文档名、格式、部门、页码、上传时间，用于后续检索追溯和分类筛选。

---

## Q10. 加载完文档后，为什么还需要清洗文本？常见的清洗操作有哪些？

**回答**

原始文档里常含大量噪声，直接向量化会把噪声也编进去，降低检索精度。常见的清洗操作：

1. **去页眉页脚**：比如每页重复的“公司内部资料 · 保密”、页码。
2. **去空行和多余空白**：连续多个换行、多个空格压缩成一个。
3. **去乱码/乱码字符**：编码问题导致的方框、问号等。
4. **去重复内容**：同一段落被复制粘贴多次。
5. **去广告/导航**：网页里的“相关推荐”“返回顶部”“版权声明”等。

### Python 示例（基础清洗）

```python
import re

def clean_text(text: str) -> str:
    # 去页眉页脚常见标记（示例，按实际文档调整）
    text = re.sub(r"公司内部资料.*?保密", "", text)
    text = re.sub(r"第\s*\d+\s*页", "", text)  # 去页码

    # 去多余空白：连续空格→一个，连续换行→最多两个
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去乱码字符（保留常用中英文和标点）
    text = re.sub(r"[^一-鿿0-9A-Za-z\s，。！？；：、""''（）《》<>\[\]{}]", "", text)

    # 去空行
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

raw = "  公司内部资料 · 保密\n\n\n单餐报销不超过200元，需提供发票。\n\n\n第 3 页  \n"
print(repr(clean_text(raw)))
# 输出: '单餐报销不超过200元，需提供发票。'
```

> 面试要点：清洗是为了去掉页眉页脚、空白、乱码、重复、广告等噪声，避免噪声进入向量空间干扰检索。常用正则表达式做空白压缩、页码去除、非法字符过滤。

---

## Q11. 怎么实现“单文件失败不中断批次”的加载？

**回答**

批量加载时，难免有个别文件损坏、格式不支持、权限不足。如果一个文件报错就把整个程序打断，后面几百个文件都加载不了，生产上不可接受。

做法：把每个文件的加载用 `try/except` 包起来，出错时记录日志并 `continue` 继续处理下一个文件，最后返回成功加载的文档列表。

代码骨架（已在 Q9 示例中体现）：

```python
for doc_info in doc_info_list:
    try:
        loader = ...  # 根据格式选加载器
        doc_list = loader.load()
        documents.extend(doc_list)
    except Exception as e:
        print(f"加载 {path} 失败：{e}，已跳过。")
        continue  # 单文件失败不中断批次
```

> 面试要点：单文件失败不中断批次的核心是 `try/except + continue`，把每个文件独立包裹，出错记录日志后继续处理下一个，保证批次整体可用。

---

## Q12. 什么是文本分块（Chunking）？为什么 RAG 必须分块？

**回答**

文本分块是把长文档切成一个个小片段（Chunk）的过程。RAG 必须分块，原因有三：

1. **Embedding 模型有长度限制**：大多数 Embedding 模型一次只能处理几千个 Token（比如 512、2048），几万字的文档塞不进去。
2. **向量会被稀释**：一篇几万字的文档压成一个向量，里面混杂几十个主题，向量的“中心”不代表任何具体主题，检索很难命中用户想问的那一段。
3. **塞不进 LLM 上下文窗口**：检索出来的片段要拼进 Prompt 送给大模型，如果每片都是几万字，上下文窗口很快撑爆。

ASCII 示意（切太大 vs 切得合适）：

```
切太大（一篇整文档一个向量）        切得合适（每片聚焦一个主题）
┌─────────────────┐              ┌─────┐ ┌─────┐ ┌─────┐
│ 报销制度 + 考勤  │              │ 报销│ │ 考勤│ │ 请假│
│ 制度 + 请假制度  │              │制度A│ │制度B│ │制度C│
│ （几十万字）     │              └─────┘ └─────┘ └─────┘
└─────────────────┘
      │
      ▼
  向量模糊，检索不准
```

> 面试要点：分块是因为 Embedding 模型有长度限制、大文档压成一个向量会语义稀释、且大片段塞不进 LLM 上下文。分块后每片聚焦一个主题，检索更精准。

---

## Q13. 分块太大或太小分别有什么问题？怎么平衡？

**回答**

- **太大**：向量被稀释（多个主题混在一起），检索不准；占用大量上下文窗口，浪费 Token，还可能触发“迷失在中间”（模型忽略中间关键信息）。
- **太小**：语义破碎，一句话被切成两半，两个半句各自都不完整；检索命中半句也答不全问题；碎片太多，索引和存储成本上升。

**平衡建议**：
- 每块大小建议 **300～1000 个 Token**（中文约 300～800 字），按“一个完整知识点”来定。
- 相邻块要有 **重叠（Overlap）**，一般为块大小的 10%～20%，防止关键信息恰好落在切分边界被截断。

> 面试要点：太大语义模糊且费 Token，太小语义破碎且碎片多。建议每块 300～1000 Token，重叠 10%～20%，按“知识单元平均长度”反推，再按实际召回率评测调优。

---

## Q14. 文本分块有哪几种常见策略？各有什么优缺点？

**回答**

常见的分块策略有五种：

| 策略 | 做法 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| 固定大小 | 按固定字符/Token 数切，可加重叠 | 简单快速，块大小可控 | 可能把句子拦腰截断，语义残缺 | 结构未知的纯文本、日志、快速原型 |
| 按分隔符 | 按段落/句子/标点切 | 尊重语义边界 | 块大小不均 | 带段落结构的文档 |
| 递归分块 | 先段落→再句子→再词，逐级切 | 兼顾大小与边界 | 实现略复杂 | 生产最常用，通用默认 |
| 语义分块 | 用 Embedding 找话题边界切 | 语义连贯性最强 | 计算成本高，阈值难定 | 法律、科研、技术支持等高精度场景 |
| 结构化分块 | 顺着标题层级、表格、代码块切 | 保留文档原生结构 | 要求文档本身结构清晰 | 格式规范的技术文档、手册 |
| 父子分块 | 小块检索 + 大块喂模型 | 兼顾“查得准”和“答得全” | 存储双份，开销大 | 生产环境，对召回率和完整性都有要求 |

> 面试要点：五大策略各有优劣。入门可用固定大小或递归分块；生产环境推荐递归分块做 baseline，对召回率要求高的可加语义分块或父子分块；格式规范的文档优先结构化分块。

---

## Q15. 什么是递归分块（Recursive Chunking）？为什么它是生产事实标准？

**回答**

递归分块的核心思想是“尽量在高层级边界切分，切不下去再下沉”：先按段落（`\n\n`）切，段落太长就按行（`\n`）切，行太长就按句子（`。！？`）切，句子还长就按空格/字符切，直到所有块都符合大小限制。

这样做能尽量保留段落、句子等语义完整单位，不会在奇怪的地方（比如一句话中间）把文档拦腰截断，又能根据内容自动调整块的大小。

LangChain 的 `RecursiveCharacterTextSplitter` 就是这个策略的事实标准实现。

ASCII 示意：

```
原文：报销单需先经部门经理审批，再交财务部门。单餐报销不超过200元。出差住宿可报销。

递归分块（优先段落→句子）：
块1：[报销单需先经部门经理审批，再交财务部门。]   ← 按句号边界切
块2：[单餐报销不超过200元。]                            ← 完整句子
块3：[出差住宿可报销。]                                ← 完整句子
```

> 面试要点：递归分块按“段落→行→句子→字符”的优先级逐级切分，优先在语义边界切，兼顾块大小和语义完整性。LangChain `RecursiveCharacterTextSplitter` 是其标准实现，推荐作为生产 baseline。

---

## Q16. chunk_size 和 chunk_overlap 分别是什么？怎么选？

**回答**

- **chunk_size**：每个块的最大大小（字符数或 Token 数）。比如 `chunk_size=500` 表示每块最多 500 个字符。
- **chunk_overlap**：相邻两个块之间重叠的大小。比如 `chunk_overlap=100` 表示块和块之间重复 100 个字符。

### 怎么选？

- **chunk_size**：中文场景建议 300～800 字（或 300～1000 Token），按“一个完整知识点”的平均长度反推。专业术语密集的文档（如法律条文）可小一点；叙述性文章可大一点。
- **chunk_overlap**：一般设为 chunk_size 的 10%～20%，即 50～200 字。重叠主要防止关键信息落在切分边界被截断。overlap 太大会导致大量重复内容被索引，检索时 Top-K 里全是同一段话的不同切片，浪费检索名额和 Token。

> 面试要点：chunk_size 控制每块大小（建议中文 300～800 字），chunk_overlap 控制相邻块重叠（建议 10%～20%）。按知识单元长度反推，再按实际召回率评测调优，不要拍脑袋设 512 就完事。

---

## Q17. 用 Python 实现固定大小分块和递归分块，代码长什么样？

**回答**

下面分别给出两种策略的纯 Python 实现，不依赖 LangChain，方便理解原理。

### 1. 固定大小分块（滑动窗口）

```python
def chunk_fixed(text: str, size: int = 500, overlap: int = 100) -> list:
    """按固定字符数切分，相邻块重叠 overlap 个字符"""
    chunks = []
    start = 0
    step = size - overlap
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += step
    return chunks


text = "公司报销制度：单餐报销不超过200元，需提供发票。出差住宿可报销，标准为每晚不超过500元。员工入职后享有5天带薪年假，需提前申请。"
for i, c in enumerate(chunk_fixed(text, size=40, overlap=10), 1):
    print(f"块{i}: {c!r}")
```

### 2. 递归分块（按段落→句子→字符逐级切）

```python
def chunk_recursive(text: str, size: int = 500, overlap: int = 100) -> list:
    """优先按段落切，段落太长再按句子，句子还长再按字符"""
    separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    def _split(text, seps):
        if len(text) <= size:
            return [text]
        if not seps:
            # 所有分隔符都试过了，强制按字符切
            return chunk_fixed(text, size=size, overlap=overlap)
        sep = seps[0]
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            candidate = (buf + sep + p) if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                # 当前这一段本身超长，递归用下一级分隔符继续切
                sub = _split(p, seps[1:])
                chunks.extend(sub)
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks

    raw_chunks = _split(text, separators)

    # 加重叠：给每个块拼上前一个块的尾部
    final = []
    for i, c in enumerate(raw_chunks):
        if i == 0:
            final.append(c)
        else:
            prev_tail = raw_chunks[i - 1][-overlap:] if overlap else ""
            final.append((prev_tail + c).strip())
    return [c for c in final if c.strip()]


# 使用示例
long_text = """公司报销制度
单餐报销不超过200元，需提供发票。
出差住宿可报销，标准为每晚不超过500元。

员工手册
员工入职后享有5天带薪年假，需提前申请。
办公室空调温度统一设定为26度，请勿私自调节。
"""

for i, c in enumerate(chunk_recursive(long_text, size=60, overlap=15), 1):
    print(f"块{i}: {c!r}")
```

> 面试要点：固定大小分块简单但可能截断句子；递归分块按“段落→句子→字符”优先级逐级切，更能保留语义完整性，推荐作为生产 baseline。两者都要配合 overlap 防边界截断。

---

## Q18. 什么是语义分块（Semantic Chunking）？什么时候用？

**回答**

语义分块不按固定字数切，而是让“内容本身决定在哪里切”：先把文档拆成句子，对每个句子做 Embedding，计算相邻句子的向量相似度。如果相邻两句相似度高，说明话题相近，放在同一个块；如果相似度明显下降，说明话题变了，在这里切开一个新块。

### 优缺点

- 优点：语义连贯性最强，每个块聚焦一个话题，检索最准。
- 缺点：需要对每个句子做 Embedding，计算成本高；相似度阈值难以统一，不同文档可能需要不同阈值；有时会切出极短的碎片块（40 个 Token 左右），反而伤检索。

### 什么时候用？

对检索精度要求很高、且文档是长篇连贯文字的场景，比如法律合同、科研论文、技术支持文档。入门或追求开发效率时，优先用递归分块；语义分块作为进阶优化。

ASCII 示意（语义边界检测）：

```
句子1：报销单需先经部门经理审批。      [嵌入]  ●
句子2：再交财务部门。                    [嵌入]  ●   ← 与句子1相似度高，同一块
句子3：单餐报销不超过200元。              [嵌入]  ●   ← 与句子2相似度高，同一块
                                          ↓ 相似度骤降（话题变了）
句子4：办公室空调温度设定为26度。         [嵌入]  ●   ← 开新块
句子5：请勿私自调节。                     [嵌入]  ●   ← 与句子4相似度高，同一块

语义分块结果：
块A：[句子1 + 句子2 + 句子3]   （报销话题）
块B：[句子4 + 句子5]           （空调话题）
```

> 面试要点：语义分块用 Embedding 算相邻句子相似度，相似度骤降处切开，语义连贯性最强但计算贵、阈值难定。适合法律/科研/技术支持等高精度场景，作为进阶优化而非入门默认。

---

## Q19. 什么是结构化分块？适合什么文档？

**回答**

结构化分块是顺着文档原生的结构来切：按标题层级（H1/H2/H3）、表格、代码块、列表等作为切分边界，一个标题下的内容作为一个块，不跨标题切分。

适合的文档：格式规范、带清晰标题层级的技术文档、产品手册、API 文档、帮助中心文章。

优点：保留文档原生结构，每个块就是一个完整的章节或小节，语义完整，检索后模型得到的上下文也完整。

缺点：要求文档本身结构清晰；如果文档是纯文本、没有标题，就无法用此策略。

ASCII 示意：

```
文档结构：
# 报销制度（H1）
## 单餐报销（H2）
  单餐报销不超过200元，需提供发票。
## 出差住宿（H2）
  出差住宿可报销，每晚不超过500元。

结构化分块结果：
块1：[## 单餐报销 + 单餐报销不超过200元，需提供发票。]
块2：[## 出差住宿 + 出差住宿可报销，每晚不超过500元。]
```

> 面试要点：结构化分块顺着标题层级、表格、代码块切分，每个块是一个完整章节。适合格式规范的技术文档、手册、API 文档；对无结构的纯文本不适用。

---

## Q20. 什么是父子分块（Parent-Child Chunking）？为什么是生产推荐做法？

**回答**

父子分块的核心思想是“小块检索、大块喂模型”：

- **子块（Child）**：切成较小的片段（比如 200～400 字），用于向量化检索，检索精准，容易命中具体答案。
- **父块（Parent）**：子块所属的更大片段（比如整节、整章，1000～2000 字），检索命中子块后，返回其父块作为 Prompt 上下文喂给大模型，保证模型看到完整上下文。

优点：既保住了“检索精准”（小块容易命中细节），又保住了“回答完整”（大块给模型足够上下文）。即使子块检索稍有偏差，返回的父块通常也还包含合理信息，降级优雅。

ASCII 示意：

```
父子分块：
父块（大，喂模型）：[## 退款政策
                    退款时效为7个工作日。VIP用户可加急至24小时。
                    运费不退。特殊商品不支持退款。]
                    ▲
              检索命中子块
子块（小，向量化检索）：[退款时效为7个工作日。VIP用户可加急至24小时。]
```

> 面试要点：父子分块＝小块向量化检索（精准命中），命中后返回其父大块喂模型（完整上下文）。生产环境推荐，兼顾“查得准”和“答得全”，代价是存储父子两份（开销稍大）。

---

## Q21. 父子分块怎么用 Python 实现？给一个可运行示例。

**回答**

下面是一个不依赖 LangChain、纯 Python 实现的父子分块示例：

```python
from typing import List, Tuple

def recursive_split(text: str, size: int, overlap: int = 0,
                    seps=("\n\n", "\n", "。", "！", "？", " ", "")) -> List[str]:
    """递归分块：按分隔符优先级逐级切"""
    if len(text) <= size or not seps:
        return [text] if text.strip() else []
    sep = seps[0]
    parts = text.split(sep)
    chunks, buf = [], ""
    for p in parts:
        candidate = (buf + sep + p) if buf else p
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            chunks.extend(recursive_split(p, size, overlap, seps[1:]))
            buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def parent_child_chunk(docs: List[str],
                       child_size: int = 300,
                       child_overlap: int = 60) -> List[Tuple[str, str]]:
    """
    返回 (子块文本, 父块文本) 配对列表。
    父块即原始文档（可按需进一步按章节切大块）。
    """
    pairs = []
    for doc in docs:
        parent = doc  # 父块：整篇文档（或按章节切的大块）
        children = recursive_split(doc, size=child_size, overlap=child_overlap)
        for child in children:
            pairs.append((child, parent))
    return pairs


# 使用示例
docs = [
    "## 退款政策\n退款时效为7个工作日。VIP用户可加急至24小时。运费不退。特殊商品不支持退款。",
    "## 请假制度\n员工入职后享有5天带薪年假，需提前申请。事假每月不超过3天。",
]

pairs = parent_child_chunk(docs, child_size=40, child_overlap=10)
for i, (child, parent) in enumerate(pairs, 1):
    print(f"配对{i}:")
    print(f"  子块（检索用）：{child!r}")
    print(f"  父块（喂模型）： {parent[:60]!r}...")
```

运行效果示例：

```
配对1:
  子块（检索用）：'退款时效为7个工作日。'
  父块（喂模型）： '## 退款政策...'
```

> 面试要点：父子分块实现思路＝先对每篇文档递归切出小子块（子块向量化检索），记录每个子块所属的父大块（父块喂模型），检索命中子块后返回对应父块作为上下文。代码中 `parent_child_chunk` 返回 `(子块, 父块)` 配对。

---

## Q22. 批量加载整个目录下的文档怎么做？怎么只加载指定后缀？

**回答**

实际工程中文档通常放在一个文件夹里，需要遍历目录批量加载，并且只加载指定后缀（比如只加载 `.pdf`、`.docx`、`.md`），忽略其他文件。

可以用 Python `os.walk` 或 `pathlib.Path.rglob` 遍历目录，按后缀过滤，再交给对应的加载器。

### Python 示例（pathlib + 自定义加载器路由）

```python
from pathlib import Path
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.schema import Document

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}

def load_directory(dir_path: str,
                   extensions=(".pdf", ".docx", ".md"),
                   department: str = "未知") -> List[Document]:
    """递归加载目录下所有指定后缀的文档"""
    root = Path(dir_path)
    documents: List[Document] = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions:
            continue  # 只加载指定后缀

        loader_cls = LOADER_MAP.get(file_path.suffix.lower())
        if loader_cls is None:
            continue

        try:
            doc_list = loader_cls(str(file_path)).load()
            for doc in doc_list:
                doc.metadata.update({
                    "doc_name": file_path.name,
                    "doc_path": str(file_path),
                    "department": department,
                })
            documents.extend(doc_list)
            print(f"已加载：{file_path}（{len(doc_list)} 片段）")
        except Exception as e:
            print(f"跳过 {file_path}：{e}")
            continue

    return documents


# 使用示例
# docs = load_directory("./knowledge_base", extensions=(.pdf, .docx, .md), department="人力资源部")
```

> 面试要点：批量加载目录用 `pathlib.Path.rglob("*")` 递归遍历，按 `suffix` 过滤只加载指定后缀，再交给对应加载器；单文件失败用 `try/except` 记录后继续，保证批次整体可用。

---

## Q23. 加载后的文档怎么去重？去重的粒度是文档级还是块级？

**回答**

同一份文档可能被重复加载（比如两个文件夹里有同名文件，或定时任务重复跑），去重可以避免向量数据库里存冗余片段，节省存储和检索开销。

去重粒度有两种：

1. **文档级去重**：按文档路径或内容哈希（MD5）判断整篇文档是否已加载过，已加载就跳过。适合“增量更新”场景，只处理新文件。
2. **块级去重**：按每个分块的内容哈希判断，重复的块不重复入库。适合“同一片段在不同文档里出现”的场景。

### Python 示例（文档级 MD5 去重）

```python
import hashlib
from pathlib import Path

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# 已处理的文件指纹集合（实际工程中可持久化到数据库/文件）
processed_fingerprints = set()

def should_process(file_path: Path) -> bool:
    fp = file_md5(file_path)
    if fp in processed_fingerprints:
        print(f"跳过重复文件：{file_path}")
        return False
    processed_fingerprints.add(fp)
    return True
```

> 面试要点：去重有文档级（按文件路径/内容哈希，适合增量更新）和块级（按分块内容哈希）两种粒度。文档级去重更常用，可把 MD5 指纹持久化，下次只处理新增或变更的文件。

---

## Q24. RAG 数据准备的增量更新怎么做？文档修改了怎么处理？

**回答**

知识库不是一次性建好的，文档会新增、修改、删除。增量更新的做法：

1. **新增文档**：按文件路径或内容哈希（MD5）判断是否已处理过，未处理则加载、分块、向量化、入库。
2. **修改文档**：比对文件的修改时间（mtime）或内容哈希，若变了，先删掉该文档在向量库里的旧块，再重新加载入库。
3. **删除文档**：从向量库里删掉该文档对应的所有块。

向量数据库通常支持按 `doc_id` 或元数据过滤删除，比如 Chroma 的 `collection.delete(where={"doc_name": "xx.pdf"})`。

ASCII 示意（增量更新流程）：

```
扫描知识库目录
      │
      ▼
┌─────────────┐
│ 文件已存在？  │──否──▶ 加载→分块→向量化→入库
│（按MD5/mtime）│
└──────┬──────┘
       │是
       ▼
┌─────────────┐
│ 内容变了？   │──否──▶ 跳过（无需处理）
└──────┬──────┘
       │是
       ▼
  删旧块 → 重新入库
```

> 面试要点：增量更新用“文件哈希或修改时间”判断新增/修改/删除。新增直接入库；修改先删旧块再重入库；删除按元数据（如文档名）从向量库删除对应块。避免全量重建，节省时间和成本。

---

## Q25. 数据准备做完后，怎么验证切分质量好不好？

**回答**

切分质量直接决定检索上限，验证方法主要有两种：

1. **人工抽检**：随机抽取若干块，看每个块是否语义完整、有没有被拦腰截断、重叠部分是否合理。
2. **召回率评测**：准备一组真实问题（比如 50 个），跑完整 RAG 流程，看问题的答案是否都在检索到的 Top-K 块里。命中率高说明切分好；命中率低说明可能切太碎（语义断）或切太大（向量稀释）。

简单的自动化检查：统计块的长度分布，如果有大量极短块（<50 字）或极长块（>2000 字），说明分块策略需要调整。

> 面试要点：验证切分质量用“人工抽检（语义是否完整）”和“召回率评测（真实问题命中率）”。自动化辅助：统计块长度分布，剔除过短碎片和过长块。

---

## Q26. RAG 数据准备常见的坑有哪些？怎么避免？

**回答**

| 常见坑 | 后果 | 避免方法 |
|--------|------|---------|
| 直接加载扫描版 PDF（无文本层） | 提取为空，检不到任何内容 | 先做 OCR（如 Tesseract）再处理 |
| 建库与查询用不同 Embedding 模型 | 向量空间不统一，检索全错 | 建库查询必须同一模型，版本也要一致 |
| chunk_size 设太大/太小 | 向量稀释或语义破碎，检索不准 | 按知识单元长度反推，300～800 字，评测调优 |
| 忽略 overlap | 关键信息落在切分边界被截断 | overlap 设 10%～20% |
| 噪声不清洗（页眉页脚、广告） | 噪声入库干扰检索 | 加载后做文本清洗 |
| 单文件失败就中断整批 | 一个坏文件卡住几百个好文件 | try/except + continue，单文件失败不中断批次 |
| 不注入元数据 | 检索后无法追溯原文，不可信 | 注入文档名、页码、部门等元数据 |

> 面试要点：数据准备常见坑包括扫描版 PDF 需 OCR、建库查询须同一 Embedding 模型、chunk_size/overlap 要调参、要清洗噪声、单文件失败不中断批次、要注入元数据用于追溯。

---

## Q27. 用一段完整代码串联“加载多格式文档→清洗→分块→注入元数据”，长什么样？

**回答**

下面是一个端到端的最小可运行流水线，把前面的知识点串起来：

```python
import re
import hashlib
from pathlib import Path
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def prepare_knowledge_base(dir_path: str,
                           extensions=(".pdf", ".docx", ".md"),
                           chunk_size: int = 500,
                           chunk_overlap: int = 100,
                           department: str = "未知",
                           processed_fingerprints: set = None) -> List[Document]:
    if processed_fingerprints is None:
        processed_fingerprints = set()

    root = Path(dir_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    all_chunks: List[Document] = []

    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in extensions:
            continue

        # 增量更新：已处理过的文件跳过
        fp = file_md5(file_path)
        if fp in processed_fingerprints:
            continue
        processed_fingerprints.add(fp)

        loader_cls = LOADER_MAP.get(file_path.suffix.lower())
        if loader_cls is None:
            continue

        try:
            docs = loader_cls(str(file_path)).load()
            for doc in docs:
                # 清洗
                doc.page_content = clean_text(doc.page_content)
                if not doc.page_content.strip():
                    continue
                # 注入元数据
                doc.metadata.update({
                    "doc_name": file_path.name,
                    "doc_path": str(file_path),
                    "department": department,
                    "file_md5": fp,
                })
            # 分块
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"已处理：{file_path.name}，分块 {len(chunks)} 个")
        except Exception as e:
            print(f"跳过 {file_path}：{e}")
            continue

    return all_chunks


# 使用示例
# chunks = prepare_knowledge_base("./knowledge_base", department="人力资源部")
# print(f"总共生成分块：{len(chunks)} 个")
```

> 面试要点：这段代码串联了“遍历目录→按后缀选加载器→MD5 增量去重→文本清洗→注入元数据→递归分块”完整流程，可作为 RAG 数据准备的模板，直接套用到生产项目中。

---

## Q28. 什么时候应该把文档先转成 Markdown 再处理？

**回答**

把文档统一转成 Markdown 再处理，适合以下情况：

1. **格式多样的文档集**：PDF、Word、HTML、Excel 都先转 Markdown，后续只用一套 Markdown 处理逻辑，代码简单。
2. **需要保留结构**：Markdown 的标题（#）、列表（-）、表格（\|）能保留文档层级，对结构化分块很友好。
3. **方便版本控制**：Markdown 是纯文本，适合放进 Git 管理，知识库变更容易追溯。

常用工具：`markitdown`（支持 PDF/Word/Excel/PPT/图片/URL 转 Markdown）、`pandoc`（万能格式转换）。

> 面试要点：格式多样、需保留结构、需版本控制时，建议先统一转 Markdown 再处理。工具可用 `markitdown` 或 `pandoc`。转成 Markdown后，后续只需一套 Markdown 加载+分块逻辑，工程复杂度大幅降低。

---

## Q29. 文档里有表格怎么办？直接切块会破坏表格结构吗？

**回答**

直接按字符或段落切块，很可能把表格拦腰截断，一行表格被切到两个块，语义完全不通。处理表格有几种常见做法：

1. **表格转 Markdown 表格**：用 `pdfplumber` 或 `pandas` 把表格提取出来，转成 Markdown 表格形式（`\|` 分隔），作为一个完整块，不拆分。
2. **表格转自然语言描述**：用大模型把表格内容总结成文字，比如“产品A，价格100元，库存50件”，再向量化。
3. **结构化分块时保持表格完整**：如果用结构化分块，把整个表格作为一个块，不跟上下文文字混切。

> 面试要点：表格不宜按字符随意切块。建议先转成 Markdown 表格作为完整块，或转成自然语言描述后再向量化，或在结构化分块时保持表格完整不拆分。

---

## Q30. 本章小结：一个初学者该怎么做 RAG 数据准备？

**回答**

数据准备是 RAG 的“隐性天花板”，做不好后面所有环节都在补救。初学者建议按这个路径入门：

1. **先统一格式**：把 PDF、Word、HTML 等尽量转成 Markdown 或纯文本，后续只用一套逻辑处理。
2. **选对加载器**：PDF 用 pypdf/PyMuPDF，Word 用 python-docx，Markdown/TXT 用 open，网页用 requests + BeautifulSoup，CSV/Excel 用 pandas。
3. **清洗文本**：去页眉页脚、空白、乱码、重复，只留有用信息。
4. **注入元数据**：文档名、页码、部门、上传时间，用于检索追溯和分类筛选。
5. **递归分块做 baseline**：`RecursiveCharacterTextSplitter`，chunk_size 300～800 字，overlap 10%～20%。
6. **增量更新与去重**：用文件 MD5 判断新增/修改/删除，避免全量重建。
7. **验证切分质量**：人工抽检 + 真实问题召回率评测，按效果调参。

一句话总结：**RAG 数据准备不是“随便读个文件就完事”，加载→清洗→分块→元数据→增量更新，每一步都决定了最终问答的准确率。会 Python，花一两天搭建这套流水线，就能把公司文档变成可检索的知识库。**

> 面试要点：入门路径＝统一格式→选加载器→清洗→注入元数据→递归分块（baseline）→增量更新去重→召回率评测调参。每一步都直接影响最终问答准确率。
