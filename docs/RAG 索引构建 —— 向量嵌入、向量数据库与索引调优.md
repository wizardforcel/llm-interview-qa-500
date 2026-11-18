# RAG 索引构建 —— 向量嵌入、向量数据库与索引调优

> 面试生普（LLM 500 题）· RAG 索引构建篇。本章覆盖 RAG 索引阶段的核心：向量嵌入模型原理与选型（BGE/E5/OpenAI、向量维度、查询前缀、归一化、Bi-encoder vs Cross-encoder）、向量数据库选型（FAISS/Chroma/Milvus/Pinecone/Qdrant/pgvector）、ANN 索引算法（Flat/IVF/HNSW/PQ/SQ8/DiskANN/Binary）、HNSW 关键参数（M/efConstruction/efSearch）与索引调优（recall vs latency、量化、元数据过滤、增量更新），共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Python 代码和 ASCII 图。

---

## Q1. 什么是向量嵌入（Embedding）？它在 RAG 里起什么作用？

**回答**

向量嵌入（Embedding）就是把一段文字转成一串数字（向量）的过程。比如：

```
"单餐报销不超过200元"  →  [0.23, -0.45, 0.87, ..., 0.12]   （比如 384 维或 768 维）
```

这串向量就是这段文字的“数学指纹”。Embedding 模型是用海量文本训练出来的，它学到了一个核心规律：**意思相近的句子，转成的向量在数学空间里距离也更近**。

在 RAG 里，Embedding 模型担任“地基”角色：

- **建库时**：把每个文档片段（Chunk）转成向量，存入向量数据库。
- **查询时**：把用户的问题也转成向量，在向量数据库里找距离最近的 Top-K 个片段。

如果 Embedding 做不好，语义空间就乱了，后面的检索、重排序都在“垃圾上做优化”，救不回来。

ASCII 示意（二维简化，实际是几百维）：

```
        向量空间
          ↑
   [报销类]      [空调类]
      ● A        ● D
    ●   B      ●   E
      ● C        ● F
          ↑
          →

A = "单餐报销不超过200元"
B = "报销单上限200元"      → 与 A 很近（同一个簇）
C = "怎么报销发票"         → 与 A 较近
D = "办公室空调26度"
E = "空调温度设定"         → 与 D 很近（另一个簇）
F = "如何调节空调"         → 与 D 较近
```

用户问题向量化后落在向量空间的某个点，离 A/B/C 簇最近，于是检索到关于报销的文档，而不是空调文档。

> 面试要点：Embedding 把文本转成向量，意思相近的文本向量距离更近。它是 RAG 索引阶段的“地基”，决定了检索能不能找到相关片段。

---

## Q2. Embedding 模型常见的有哪些？中文场景怎么选？

**回答**

常用的 Embedding 模型（按初学者友好程度）：

| 模型系列 | 代表模型 | 向量维度 | 特点 | 适合场景 |
|---------|---------|---------|------|---------|
| sentence-transformers | all-MiniLM-L6-v2 | 384 | 开源、轻量、可本地运行 | 入门 demo、英文为主 |
| BGE（智源） | bge-small/ base/ large-zh | 384/ 768/ 1024 | 中英文都强，中文效果好 | **中文场景首选** |
| E5 | multilingual-e5-large | 1024 | 多语言，指令微调 | 多语言/跨语言检索 |
| OpenAI | text-embedding-3-small / large | 1536/ 3072 | 云端 API，效果强，按 Token 付费 | 对效果要求高、不介意成本 |
| Jina | jina-embeddings-v3 | 1024 | 支持 8192 Token 长文本 | 长文档场景 |
| GTE（阿里） | gte-large | 1024 | 中文场景表现好 | 中文场景备选 |

**中文场景怎么选？**

- 入门、本地跑、隐私敏感：选 `BAAI/bge-small-zh-v1.5`（384 维，轻量）或 `bge-base-zh-v1.5`（768 维）。
- 中文企业知识库、追求检索质量：选 `BAAI/bge-large-zh-v1.5`（1024 维）。
- 长文档（超 512 Token）：选支持长上下文的 `bge-m3`（8192 Token）或 `jina-embeddings-v3`。
- 不想部署、预算充足：用 OpenAI `text-embedding-3-small` 云端 API。

> 面试要点：中文场景优先 BGE 系列（small/base/large-zh），长文档选 bge-m3 或 Jina，不想部署可用 OpenAI Embedding。不同模型输出维度不同，建库和查询必须用**同一个模型**，否则向量空间不统一，检索全错。

---

## Q3. 向量维度是什么？维度越高越好吗？

**回答**

向量维度是指 Embedding 模型输出向量的长度，比如 384 维、768 维、1024 维、1536 维、3072 维。每个维度对应语义空间里的一个方向。

常见模型维度对照表：

| 模型 | 维度 |
|------|------|
| all-MiniLM-L6-v2 | 384 |
| bge-small-zh | 384 |
| bge-base-zh | 768 |
| bge-large-zh / OpenAI text-embedding-3-small | 1024 / 1536 |
| OpenAI text-embedding-3-large | 3072 |

**维度越高越好吗？** 不一定。

- 维度高：能表达更细微的语义差异，检索精度可能更高。
- 但代价：存储更大（向量 × 维度 × 4 字节）、计算更慢、内存占用更高。

比如 100 万条向量，384 维占约 1.5 GB，1536 维占约 6 GB，差了 4 倍。

**实用建议**：中文入门先用 384～768 维（bge-small/base），够用且轻量；对检索精度要求极高、资源充足时再上 1024～3072 维。不要一上来就追求最大维度，先按实际召回率评测。

> 面试要点：向量维度是输出向量的长度。越高能表达更细微语义，但存储和计算成本更大。中文入门推荐 384～768 维，按实际召回率评测再决定是否提升。

---

## Q4. 什么是查询前缀（Query Prefix）？忘了加会怎样？

**回答**

部分 Embedding 模型（尤其是 BGE、E5 系列）要求查询（query）和文档（document）在编码前加不同的前缀，因为“查询意图”和“文档内容”在语义空间的分布不一样，加前缀能让模型把它们区分开，提升检索质量。

常见模型的前缀要求：

| 模型系列 | 查询前缀 | 文档前缀 |
|---------|---------|---------|
| BGE (`BAAI/bge-*`) | `Represent this sentence: ` | 无 |
| E5 (`intfloat/e5-*`) | `query: ` | `passage: ` |
| GTE (`Alibaba-NLP/gte-*`) | 无 | 无 |
| Nomic (`nomic-ai/nomic-*`) | `search_query: ` | `search_document: ` |
| all-MiniLM / all-mpnet | 无 | 0 |

### Python 示例（BGE）

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

# BGE：query 加前缀，document 不加
query_emb = model.encode("为这个句子生成表示：单餐报销上限是多少？")
doc_emb = model.encode("单餐报销不超过200元，需提供发票。")
```

**忘了加前缀会怎样？** 查询向量和文档向量分布错位，相似度计算不准，召回率会明显下降（有实测显示掉好几个点）。所以用 BGE/E5 时，务必查文档确认是否需要前缀，且建库与查询必须用同一套前缀规则。

> 面试要点：BGE 需在 query 前加 `Represent this sentence: `，E5 需 `query:` / `passage:`，GTE/all-MiniLM 不需要。建库与查询必须用同一模型、同一前缀规则，否则向量空间不统一，召回率会明显下降。

---

## Q5. 什么是归一化？余弦相似度和点积有什么关系？

**回答**

归一化（L2 归一化）是把向量的长度缩成 1，使其变成单位向量。归一化后，两个向量的**点积就等于余弦相似度**。

公式：

```
余弦相似度 = (A · B) / (|A| × |B|)
```

如果 |A| = |B| = 1（已归一化），则 余弦相似度 = A · B（点积）。

点积计算比余弦相似度快（免去除法和模长计算），而且硬件（GPU/CPU）对点积优化更好。所以很多向量数据库在存储时会对向量做归一化，查询时直接用点积来排序，又快又准。

不同数据库的默认距离度量不一样：

| 数据库 | 默认度量 | 备注 |
|--------|---------|------|
| Chroma | cosine 相似度 | 归一化后等价于点积 |
| FAISS | L2 距离（欧氏距离） | 归一化后 L2 与余弦相似度单调相等 |
| Milvus | 支持 L2 / IP / cosine | 需显式指定 |

> 面试要点：归一化后向量长度为 1，点积即等于余弦相似度。点积计算更快，生产系统常先归一化再用点积排序。不同数据库默认度量不同（Chroma 默认 cosine，FAISS 默认 L2），选型时要注意是否匹配。

---

## Q6. Bi-encoder 和 Cross-encoder 的区别？在 RAG 里分别用在哪？

**回答**

这是理解现代 Embedding 模型的关键概念。

**Bi-encoder（双编码器，用于检索阶段）**

把查询和文档分别编码成两个向量，再算相似度：

```
查询 → Encoder → 向量 q
文档 → Encoder → 向量 d
相似度 = cos(q, d)
```

优点：所有文档向量可以**离线预先算好存起来**，查询时只需要算查询向量，然后在向量数据库里做近邻搜索，速度极快。适合第一阶段召回（Top-K 检索）。

**Cross-encoder（交叉编码器，用于重排序阶段）**

把查询和文档**拼在一起**同时输入模型，模型直接输出一个相关性分数：

```
[查询 + 文档] → Encoder → 相关性分数
```

优点：模型能充分交互查询和文档的细节，判断更精准。缺点：每个查询-文档对都要过一次模型，不能离线预计算，速度慢，只能对检索出来的少数候选（比如 Top-100）做重排序。

### RAG 里的典型分工

```
用户提问
    │
    ▼
Bi-encoder 召回（向量数据库，Top-100，毫秒级）
    │
    ▼
Cross-encoder 重排序（Top-100 → Top-10，秒级，更精准）
    │
    ▼
送 Top-10 给大模型生成答案
```

> 面试要点：Bi-encoder 分别编码查询和文档，可离线预计算向量，用于第一阶段快速召回；Cross-encoder 把查询和文档拼在一起一起编码，更精准但慢，用于第二阶段重排序。生产 RAG 常见“Bi-encoder 召回 + Cross-encoder 重排”组合。

---

## Q7. 怎么用 Python 加载 Embedding 模型并批量向量化文档？

**回答**

下面用 `sentence-transformers` 加载本地 BGE 模型，批量给文档片段生成向量。这是 RAG 索引构建最常见的入门代码。

先安装依赖：

```bash
pip install sentence-transformers numpy
```

完整代码：

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载模型（首次运行会自动下载，需联网）
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")  # 384 维，中文入门推荐

# 文档片段（已分好块）
chunks = [
    "单餐报销不超过200元，需提供发票。",
    "出差住宿可报销，每晚不超过500元。",
    "员工入职后享有5天带薪年假，需提前申请。",
]

# BGE：query 加前缀，document 不加（这里文档不加前缀）
embeddings = model.encode(chunks, normalize_embeddings=True, batch_size=32)

print(f"向量数量：{embeddings.shape[0]}，维度：{embeddings.shape[1]}")
# 输出示例：向量数量：3，维度：384

# 查询时，query 需要加前缀
query = "为这个句子生成表示：单餐报销上限是多少？"
query_emb = model.encode([query], normalize_embeddings=True)[0]

# 计算相似度（已归一化，点积即余弦相似度）
similarities = np.dot(embeddings, query_emb)
best_idx = int(np.argmax(similarities))
print(f"最相关片段：{chunks[best_idx]}，相似度：{similarities[best_idx]:.3f}")
```

关键参数说明：

- `normalize_embeddings=True`：输出归一化向量，点积即余弦相似度，省去除法。
- `batch_size`：批量编码，比一条条编码快 10 倍以上，建议设 32～256。

> 面试要点：用 `sentence-transformers.SentenceTransformer` 加载模型，`encode` 批量向量化。BGE 文档不加前缀、query 加前缀；`normalize_embeddings=True` 后点积即余弦相似度；`batch_size` 批处理能大幅提升向量化速度。

---

## Q8. 什么是向量数据库？为什么 RAG 离不开它？

**回答**

向量数据库专门用来高效存储和检索高维向量。传统数据库（如 MySQL）擅长精确匹配（`WHERE name = '张三'`），而向量数据库擅长**相似度搜索**：给一个查询向量，找距离最近的 Top-K 个文档向量。

如果不用向量数据库，每次查询都要把查询向量与库里所有向量逐一计算距离，复杂度是 O(N)，百万级向量就得数秒，无法满足实时响应。

向量数据库的价值：

1. **持久化存储**：Embedding 算一次，存下来，下次直接用，不用重复计算。
2. **快速检索**：专用索引结构（HNSW、IVF 等），比暴力遍历快 10～100 倍。
3. **增量更新**：随时添加新文档，不用重新计算所有向量。
4. **元数据过滤**：可以按文档名、部门、日期等条件筛选后再检索。

ASCII 示意（有索引 vs 无索引）：

```
无索引（暴力遍历）              有向量索引（HNSW）
┌─────────────┐                ┌─────────────┐
│ 查询向量    │                │ 查询向量    │
│     │       │                │     │       │
│     ▼       │                │     ▼       │
│ 遍历所有向量 │               │ 走 HNSW 图  │
│  O(N) 慢    │───────────────▶│ O(log N) 快 │
└─────────────┘                └─────────────┘
  100万向量需数秒              100万向量毫秒级
```

> 面试要点：向量数据库专门做高维向量相似度搜索，解决 O(N) 暴力遍历太慢的问题。价值在于持久化、快速检索、增量更新、元数据过滤。没有它，RAG 无法实时响应。

---

## Q9. 常见的向量数据库有哪些？初学者该选哪个？

**回答**

常见的向量数据库及其定位：

| 向量数据库 | 定位 | 特点 | 初学者友好度 |
|-----------|------|------|-------------|
| **Chroma** | 轻量级嵌入式向量库 | Python-native，开箱即用，无需部署，支持元数据过滤 | ⭐⭐⭐⭐⭐ 推荐入门 |
| **FAISS** | Meta 开源的检索库（非服务） | 纯 C++，性能极致，支持 GPU，但无持久化 API 服务，需自己包一层 | ⭐⭐⭐⭐ |
| **Milvus** | 分布式生产级向量数据库 | 支持多种索引，存算分离，水平扩展，适合百万～亿级向量 | ⭐⭐⭐（需部署） |
| **Pinecone** | 云端全托管向量数据库 | 免运维，按使用量付费，适合不想自建的团队 | ⭐⭐⭐⭐（需云端账号） |
| **Qdrant** | Rust 编写的开源向量数据库 | 过滤体验好，支持标量量化，部署简单 | ⭐⭐⭐⭐ |
| **pgvector** | PostgreSQL 的向量扩展 | 已有 PostgreSQL 的业务可直接用，无需新增组件 | ⭐⭐⭐⭐ |

**初学者选 Chroma**。它可以直接在 Python 内存模式运行，几行代码就能建立可检索的向量库，非常适合跑第一个 RAG 索引 demo。生产数据量大（百万～亿级）再考虑 Milvus 或云端 Pinecone。

> 面试要点：常见向量数据库有 Chroma、FAISS、Milvus、Pinecone、Qdrant、pgvector。初学者推荐 Chroma（轻量、内存模式、无需部署）；生产大规模选 Milvus；不想运维选 Pinecone；已有 PostgreSQL 可用 pgvector。

---

## Q10. 用 Chroma 搭建一个可检索的向量索引，代码长什么样？

**回答**

下面是一个完整示例，使用 Chroma（内存模式）+ BGE Embedding，把文档片段向量化并存入向量数据库，支持查询检索。

先安装依赖：

```bash
pip install sentence-transformers chromadb numpy
```

完整代码：

```python
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# 1. 加载 Embedding 模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 2. 准备文档片段（已分块、已清洗）
chunks = [
    "单餐报销不超过200元，需提供发票。",
    "出差住宿可报销，每晚不超过500元。",
    "员工入职后享有5天带薪年假，需提前申请。",
    "办公室空调温度统一设定为26度，请勿私自调节。",
]
metadatas = [
    {"doc_name": "报销制度.pdf", "page": 1, "department": "财务部"},
    {"doc_name": "报销制度.pdf", "page": 2, "department": "财务部"},
    {"doc_name": "员工手册.docx", "page": 3, "department": "人力资源部"},
    {"doc_name": "员工手册.docx", "page": 5, "department": "人力资源部"},
]
ids = ["chunk_0", "chunk_1", "chunk_2", "chunk_3"]

# 3. 向量化（文档不加前缀）
embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

# 4. 创建 Chroma 向量数据库（内存模式）
client = chromadb.Client()
collection = client.create_collection(name="company_docs")

# 5. 入库（存向量 + 原始文本 + 元数据）
collection.add(
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
)
print(f"向量库入库完成，共 {collection.count()} 条")

# 6. 查询（query 需要加 BGE 前缀）
query = "为这个句子生成表示：单餐报销上限是多少？"
query_emb = model.encode([query], normalize_embeddings=True).tolist()

results = collection.query(
    query_embeddings=query_emb,
    n_results=2,  # 取 Top-2
)

print("\n检索结果：")
for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    print(f"{i+1}. [{meta['doc_name']} · 第{meta['page']}页] {doc} （距离：{dist:.3f}）")
```

运行效果示例：

```
向量库入库完成，共 4 条

检索结果：
1. [报销制度.pdf · 第1页] 单餐报销不超过200元，需提供发票。（距离：0.12）
2. [报销制度.pdf · 第2页] 出差住宿可报销，每晚不超过500元。（距离：0.45）
```

> 面试要点：Chroma 入门示例＝加载 BGE 模型→文档片段向量化（normalize_embeddings=True）→`chromadb.Client()` 建库→`collection.add()` 存入向量+文本+元数据→`collection.query()` 检索。注意 BGE 的 query 需加前缀，document 不加；建库与查询用同一模型。

---

## Q11. 什么是 ANN（近似最近邻搜索）？为什么不用精确 KNN？

**回答**

ANN（Approximate Nearest Neighbor，近似最近邻搜索）是指在向量空间里找“足够近”的邻居，而不是绝对最近的那个。它牺牲极少量的召回精度（通常损失 <1%），换取查询速度的数量级提升。

精确 KNN 的问题：每次查询都要把查询向量与库里**所有**向量逐一计算距离，复杂度是 O(N)。当向量量达到百万级时，单次查询就需要数秒，无法满足 RAG 的实时响应要求。

ASCII 示意：

```
精确 KNN（暴力遍历）              ANN（近似最近邻）
┌─────────────┐                ┌─────────────┐
│ 查询向量    │                │ 查询向量    │
│     │       │                │     │       │
│     ▼       │                │     ▼       │
│  与所有 N 个 │                │  先查索引   │
│  向量算距离 │                │  再定位邻居 │
│   O(N) 慢   │───────────────▶│  O(log N) 快│
└─────────────┘                └─────────────┘
  100万向量需数秒              100万向量毫秒级
     │                             │
     ▼                             ▼
  召回 100%                     召回 99%+（可接受）
```

> 面试要点：ANN 牺牲极少召回精度（<1%），换取查询速度数量级提升。精确 KNN 是 O(N)，百万级向量太慢；ANN 通常是 O(log N) 左右，毫秒级响应，是生产 RAG 的唯一选择。

---

## Q12. 常见的 ANN 索引算法有哪些？各有什么特点？

**回答**

常见的 ANN 索引算法：

| 算法 | 全称 | 原理 | 特点 | 适用规模 |
|------|------|------|------|---------|
| Flat | — | 暴力扫描，无近似 | 召回 100%，但 O(N) 很慢 | < 10 万向量，仅做 Benchmark |
| IVF | Inverted File Index | 先用 K-means 聚成多个簇，查询时只搜最近的几个簇 | 需训练，适合静态数据，中等精度 | 中等规模 |
| HNSW | Hierarchical Navigable Small World | 分层图结构，类似跳表，从顶层快速下钻 | 查询最快，召回最高，但内存占用大 | 100 万～1 亿（生产常见默认） |
| PQ | Product Quantization | 把向量切成子空间独立量化，压缩存储 | 压缩比高，但有精度损失 | 大规模，配合 IVF/HNSW |
| SQ8 | Scalar Quantization | 每个分量量化到 8 个整数 | 压缩 4×，精度损失小 | 生产常用 |
| Binary | Binary Quantization | 向量量化成 0/1，海明距离搜索 | 压缩 32×，极快，召回略低 | 极致压缩场景，常作粗召回 |
| DiskANN | — | 磁盘友好的图索引 | 适合 10 亿+ 向量，NVMe SSD | 超大规模 |

> 面试要点：Flat 暴力扫描只适合小规模；IVF 聚类加速；HNSW 分层图，生产中最常用（百万～亿级）；PQ/SQ8/Binary 是量化压缩，节省内存但有轻微精度损失，常与 HNSW/IVF 组合；DiskANN 适合 10 亿+ 超大规模。

---

## Q13. 什么是 HNSW？它的三个核心参数（M / efConstruction / efSearch）分别控制什么？

**回答**

HNSW（Hierarchical Navigable Small World，分层可导航小世界图）可以理解成“向量版跳表”：把向量组成多层图，顶层节点少、连边长（像高速公路，负责快速接近目标区域），底层节点多、连边短（像城市道路，负责精细查找）。查询时从顶层入口出发贪心搜索，逐层下钻，最后在底层找到近邻，O(log N) 步就能到达。

三个核心参数：

| 参数 | 含义 | 调大的影响 | 建议值 |
|------|------|-----------|--------|
| M | 每个节点最多连接多少个邻居（图的连通度） | 图越密，召回更稳，但内存和构建时间增加 | 16～64 |
| efConstruction | 建图时的搜索宽度 | 越大建图越慢，但图质量越高，检索更准 | 200～500 |
| efSearch | 查询时的搜索宽度 | 越大查询越慢，但召回率越高 | 100～1000，查询时可动态调整 |

调优口诀：**recall vs latency 的权衡**——想查得准就调大 efSearch 和 M，想查得快就调小；内存受限时控制 M，别超过 128～256 避免内存爆炸。

ASCII 示意（分层图）：

```
        第 3 层（稀疏，长边，快速定位区域）
              ●───────●
             /         \
        第 2 层          
          ●───●───●───●
         /     |     \
    第 1 层（稠密，短边，精细查找）
      ●───●───●───●───●
      查询向量 → 从顶层入口贪心下钻 → 底层找近邻
```

> 面试要点：HNSW 是分层图索引，生产中最常用（百万～亿级）。三个参数：M 控制图连通度（内存与召回的权衡），efConstruction 控制建图质量，efSearch 控制查询时的召回率。调优核心是 recall vs latency 的权衡。

---

## Q14. 向量量化（PQ / SQ8 / Binary）是什么？有什么代价？

**回答**

向量量化是把高维浮点向量压缩成更小的表示，节省存储和计算，代价是轻微的精度损失。

| 量化方式 | 做法 | 压缩比 | 精度损失 | 适用场景 |
|---------|------|--------|---------|---------|
| SQ8（标量量化） | 每个分量量化到 8 个整数（uint8） | 4× | 很小（~1～2% 召回损失） | 生产常用，配合 float 重排 |
| PQ（乘积量化） | 把向量切成子空间，每个子空间独立量化 | 16～48× | 中等 | 亿级向量，内存紧张 |
| Binary（二值量化） | 每个分量量化成 0 或 1，海明距离搜索 | 32× | 较大 | 极致压缩，常作粗召回，再 float 精排 |

生产常用组合：**Binary 粗召回（海明距离极快）→ float32 精排 → Cross-encoder 重排序**。

> 面试要点：量化用空间/速度换少量召回精度。SQ8 压缩 4× 且损失很小，生产常用；PQ 压缩更高但损失更大；Binary 压缩 32×，适合粗召回。量化后通常配合 float 向量重排序，弥补精度损失。

---

## Q15. Chroma、FAISS、Milvus 三个产品怎么选？各适合什么场景？

**回答**

选型决策树：

```
需要生产部署？
├── 否（只是实验/POC）
│   ├── 数据量 < 100 万 → Chroma
│   └── 需要 GPU 加速 / 单机极致性能 → FAISS
└── 是（生产环境）
    ├── 想要全托管、不想运维 → Pinecone
    ├── 已有 PostgreSQL，不想新增组件 → pgvector
    ├── 部署简单、过滤体验好 → Qdrant
    └── 百万～亿级、高并发、多租户、读写分离 → Milvus
```

三个常见产品对比：

| 维度 | Chroma | FAISS | Milvus |
|------|--------|-------|--------|
| 定位 | 轻量级嵌入式向量库 | 检索库（非服务） | 分布式生产级向量数据库 |
| 部署 | 无需部署，Python 内存模式 | 无需部署，但需自己包服务层 | 需部署（分布式） |
| 持久化 | SQLite 原生支持 | 需手动保存/加载索引 | 原生分布式存储 |
| 元数据过滤 | 支持，小规模性能一般 | 需自己包一层 | 支持，标量过滤 + 向量搜索混合查询 |
| 适用规模 | < 100 万向量 | 单机百万～千万级 | 百万～亿级 |
| 生产适用度 | 原型验证、小项目 | 单机高并发 | 大规模生产 |

> 面试要点：入门选 Chroma（无需部署、内存模式）；单机极致性能选 FAISS（需自己包服务层）；大规模生产选 Milvus（分布式、百万～亿级）；不想运维选 Pinecone 云端；已有 PostgreSQL 可用 pgvector。

---

## Q16. 用 FAISS 构建 HNSW 索引并检索，代码长什么样？

**回答**

FAISS 是 Meta 开源的高性能向量检索库，适合单机场景。下面用 FAISS 构建 HNSW 索引并做相似度检索。

先安装依赖：

```bash
pip install faiss-cpu sentence-transformers numpy
```

完整代码：

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 加载 Embedding 模型并向量化文档
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")  # 384 维
chunks = [
    "单餐报销不超过200元，需提供发票。",
    "出差住宿可报销，每晚不超过500元。",
    "员工入职后享有5天带薪年假，需提前申请。",
    "办公室空调温度统一设定为26度，请勿私自调节。",
]
embeddings = model.encode(chunks, normalize_embeddings=True)  # (4, 384)
dim = embeddings.shape[1]

# 2. 构建 HNSW 索引
# IndexHNSWFlat(dim, M)：dim=向量维度，M=每个节点最大连接数
index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200  # 建图质量
index.hnsw.efSearch = 100        # 查询时的召回宽度

# 归一化后用点积（IndexHNSWFlat 默认 L2，归一化向量上 L2 与余弦相似度单调）
index.add(embeddings.astype("float32"))
print(f"FAISS HNSW 索引构建完成，向量数：{index.ntotal}")

# 3. 查询（query 需加 BGE 前缀）
query = "为这个句子生成表示：单餐报销上限是多少？"
query_emb = model.encode([query], normalize_embeddings=True).astype("float32")

D, I = index.search(query_emb, k=2)  # D=距离，I=索引下标

print("\n检索结果：")
for i, idx in enumerate(I[0]):
    print(f"{i+1}. {chunks[idx]} （距离：{D[0][i]:.3f}）")
```

> 面试要点：FAISS `IndexHNSWFlat(dim, M)` 构建 HNSW 索引，`hnsw.efConstruction` 控制建图质量，`hnsw.efSearch` 控制查询召回宽度。归一化后的向量用点积即余弦相似度。FAISS 是库而非服务，生产上需自己包持久化和元数据管理层。

---

## Q17. 向量维度不一致会发生什么？怎么避免？

**回答**

向量维度必须和向量数据库的索引维度**严格一致**。如果建库时用 384 维模型，查询时用 768 维模型（或同一模型不同版本），向量空间完全不同，距离计算无意义，检索结果会全错；很多向量数据库在 add 或 query 时会直接报维度不匹配的错误。

避免方法：

1. **锁定模型版本**：把 Embedding 模型名和版本（如 `BAAI/bge-small-zh-v1.5`）写进配置文件，建库和查询必须用同一个。
2. **存储模型元数据**：在向量数据库的元数据或单独配置里记录 `embedding_model` 和 `dimension`，查询前先校验。
3. **升级模型时全量重建**：如果要换 Embedding 模型，不能只改查询端，必须把所有旧向量删掉，用新模型重新向量化全部文档，再建新索引，原子性切换。

> 面试要点：向量维度不一致会导致检索结果全错甚至报错。避免方法：锁定模型名和版本、存储模型元数据做校验、换模型时全量重建索引并原子切换。

---

## Q18. 元数据过滤在向量数据库里怎么用？为什么重要？

**回答**

元数据过滤是指在向量相似度检索之外，再加上标量条件（如文档名、部门、日期、租户 ID），只在满足条件的向量里做相似度搜索。

例子：只搜“人力资源部”的文档，且只搜 2026 年后上传的。

```python
# Chroma 示例：带元数据过滤的查询
results = collection.query(
    query_embeddings=query_emb,
    n_results=5,
    where={"department": "人力资源部"},  # 标量过滤条件
)
```

**为什么重要？**

1. **权限隔离**：不同租户、不同部门只能检索自己的知识，防止越权查看。
2. **缩小检索范围**：先按元数据过滤，再向量搜索，减少无关干扰，提升精度。
3. **来源追溯与版本管理**：知道每个片段来自哪个文档、哪个版本，方便删除过时文档。

注意：元数据过滤方式有 pre-filter（先过滤后搜索）、in-filter（搜索时一起过滤）、post-filter（搜完后过滤，但可能导致返回少于 Top-K 条）。生产上优先选支持 in-filter 的数据库（如 Milvus、Qdrant），避免 post-filter 返回空页。

> 面试要点：元数据过滤是在向量搜索之外加标量条件（部门、租户、日期等），用于权限隔离、缩小检索范围、来源追溯。生产上优先支持 in-filter 的数据库，避免 post-filter 导致返回结果不足 Top-K。

---

## Q19. 索引调优的核心权衡是什么？怎么衡量好坏？

**回答**

索引调优的核心权衡是 **recall vs latency（召回率 vs 延迟）**：想查得准（召回高）通常要付出更长的查询时间和更大的内存；想查得快就要牺牲少量召回。

常见的衡量指标：

| 指标 | 含义 |
|------|------|
| Recall@K | 检索到的 Top-K 里，真正相关的片段占比（召回率的上限，决定了最终答案的准确率天花板） |
| P50 / P95 / P99 | 查询延迟的分位数（P99 越低越稳） |
| QPS | 每秒查询次数（吞吐量） |
| 内存占用 | 索引占用的内存大小 |

**调优方法论**：

1. 先跑通 baseline（如 HNSW，M=32，efSearch=100）。
2. 用一组真实问题（如 100 个）做召回率评测，统计 Recall@K。
3. 如果 Recall@10 不够高：调大 efSearch、M，或加重排序（Cross-encoder）。
4. 如果延迟太高：调小 efSearch、M，或开启量化（SQ8/Binary）。
5. 持续监控 P99 延迟和 QPS，防止线上性能雪崩。

> 面试要点：索引调优核心是 recall vs latency 权衡。衡量指标：Recall@K（召回率天花板）、P50/P95/P99（延迟）、QPS（吞吐）、内存占用。调优遵循：先跑 baseline→真实问题评测→按 recall 或延迟方向调参→持续监控。

---

## Q20. 怎么评测 RAG 检索阶段的召回率（Recall@K）？

**回答**

召回率评测是衡量索引质量最直接的方法。步骤：

1. **准备评测集**：收集 100 个真实问题，每个问题标注“正确答案应该出现在哪些文档片段里”（相关片段 ID 集合）。
2. **跑检索**：对每个问题，用 RAG 系统检索 Top-K 片段。
3. **计算 Recall@K**：看检索到的片段里是否包含了相关片段。

### Python 示例

```python
def recall_at_k(retriever, eval_set, k=10):
    """
    retriever: 检索器，search(query, top_k=k) 返回检索到的片段 id 集合
    eval_set: [(query, set_of_relevant_doc_ids), ...]
    """
    hits = 0
    for query, relevant_ids in eval_set:
        retrieved_ids = set(retriever.search(query, top_k=k))
        if retrieved_ids & relevant_ids:  # 有交集就算命中
            hits += 1
    return hits / len(eval_set)


# 使用示例
eval_set = [
    ("单餐报销上限是多少？", {"chunk_0", "chunk_1"}),
    ("出差住宿怎么报销？", {"chunk_1"}),
    ("带薪年假几天？", {"chunk_2"}),
]

r = recall_at_k(my_retriever, eval_set, k=5)
print(f"Recall@5 = {r:.2%}")
```

**怎么看结果？**

- Recall@10 达到 0.85～0.95 以上，说明检索基础较好，后续大模型生成有依据。
- 如果 Recall@10 只有 0.5，那不管大模型多强，答案也不会准——证据根本没检索到。这时候应该优化检索（换 Embedding 模型、调分块、加重排序），而不是只改 Prompt。

> 面试要点：Recall@K 衡量检索阶段把“相关片段”检出来的概率，是 RAG 最终答案准确率的上限。评测方法：准备标注了相关片段的真实问题集，跑检索看命中率。如果 Recall@K 低，应优化检索而非只改 Prompt。

---

## Q21. 向量数据库怎么做增量更新？文档修改或删除怎么处理？

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

## Q22. 什么时候应该重建整个向量索引？

**回答**

不是每次知识库变化都要重建整个索引，但在以下情况需要全量重建：

1. **更换 Embedding 模型**：新模型输出的向量维度或分布与旧模型不同，旧向量无法直接用，必须全部重新向量化。
2. **更改向量维度或距离度量**：比如从 cosine 改成 L2，或从 384 维换成 768 维，索引结构本身变了。
3. **索引严重碎片化**：大量增量更新、删除后，索引效率下降，查询延迟变高，需要 compact/rebuild。
4. **更改分块策略（chunk_size / overlap）**：所有文档的切分方式变了，旧块失效，需全部重切重建。

全量重建时建议“建新索引、原子切换”：先构建新索引（新集合），验证无误后再把查询流量切过去，旧索引下线。避免在线系统中断。

> 面试要点：更换 Embedding 模型、更改维度/距离度量、更改分块策略、索引严重碎片化时需要全量重建。生产上采用“建新索引、原子切换”的方式，避免在线中断。

---

## Q23. 向量检索弱于精确匹配的场景有哪些？怎么补救？

**回答**

向量检索擅长语义相似，但对**精确匹配**类内容较弱，比如：

- 产品 SKU、错误代码、函数名、罕见专有名词（这些词分布语义弱，向量容易混）
- 数字、日期、金额等精确值
- 用户明确输入的关键词（如“报销单第 12345 号”）

**补救方法：混合检索（Hybrid Search）**

把“向量检索（语义）”和“关键词检索（BM25 / 倒排索引 / 精确匹配）”的结果融合起来，再重排序。

```
用户提问
    │
    ├────▶ 向量检索（语义相似） ──┐
    │                             │
    └────▶ 关键词检索（BM25/倒排）┼──▶ 结果融合（RRF / 加权） ──▶ Cross-encoder 重排序 ──▶ 大模型
```

常用融合算法：**RRF（Reciprocal Rank Fusion，倒排名融合）**，不需要给两路检索手动调权重，效果稳定。

> 面试要点：向量检索弱于精确匹配（SKU、错误码、数字、精确关键词）。补救方法是混合检索：向量检索 + 关键词检索（BM25/倒排），用 RRF 等算法融合结果，再 Cross-encoder 重排序。

---

## Q24. pgvector 适合什么场景？怎么用？

**回答**

pgvector 是 PostgreSQL 的向量扩展插件，把向量相似度搜索能力直接加到 PostgreSQL 里。

适合场景：

- 公司已经重度依赖 PostgreSQL，不想新增一个独立的向量数据库组件。
- 业务数据和向量数据需要强一致性（同一个事务里写业务记录和向量）。
- 数据量中等（十万～百万级向量），不需要分布式。

使用示例（SQL）：

```sql
-- 启用扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 建表，包含向量列（dimension=384）
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    doc_name TEXT
);

-- 插入向量
INSERT INTO documents (content, embedding)
VALUES ('单餐报销不超过200元', '[0.23, -0.45, ...]');

-- 创建索引（HNSW 或 IVF）
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- 查询最相似的 Top-5
SELECT content, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY distance
LIMIT 5;
```

> 面试要点：pgvector 是 PostgreSQL 向量扩展，适合已有 PostgreSQL 且不想新增组件、数据量中等、需要业务数据与向量强一致性的场景。用 `vector(n)` 列类型，创建 `hnsw` 或 `ivfflat` 索引，用 `<=>`（余弦距离）或 `<->`（L2 距离）算子查询。

---

## Q25. 向量数据库生产环境常见的性能瓶颈有哪些？怎么优化？

**回答**

向量数据库在生产环境常见的性能瓶颈：

| 瓶颈 | 表现 | 优化方法 |
|------|------|---------|
| 查询延迟高 | P99 延迟大，用户等待久 | 调小 efSearch、M；开启量化（SQ8/Binary）；增加实例/副本 |
| 内存占用大 | HNSW 图 + 全量向量占用大量内存 | 开启 PQ/SQ8/Binary 量化；使用 mmap；控制 M ≤ 128～256 |
| 写入抖动 | 批量写入时索引构建导致延迟尖峰 | 分批写入；读写隔离（写入节点与查询节点分离） |
| 元数据过滤慢 | where 条件过滤后检索变慢 | 优化 over-filter 策略；给元数据字段建索引 |
| 索引碎片化 | 频繁增量更新后查询变慢 | 定期重建/ compact 索引 |

> 面试要点：生产环境瓶颈包括查询延迟高、内存占用大、写入抖动、元数据过滤慢、索引碎片化。优化手段：调 efSearch/M、开启量化、分批写入、读写隔离、定期重建索引。

---

## Q26. 用一段完整代码串联“Embedding 向量化 → Chroma 建库 → 带元数据过滤检索 → 评测 Recall@K”，长什么样？

**回答**

下面是一个端到端的最小可运行流水线，把本章的核心知识点串起来：

```python
import hashlib
from pathlib import Path
from typing import List, Set
from sentence_transformers import SentenceTransformer
import chromadb

# 配置
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 384 维
TOP_K = 5

# 1. 加载 Embedding 模型
model = SentenceTransformer(EMBED_MODEL)

# 2. 准备文档片段 + 元数据（实际应从 Q2 的加载流水线获得）
chunks = [
    "单餐报销不超过200元，需提供发票。",
    "出差住宿可报销，每晚不超过500元。",
    "员工入职后享有5天带薪年假，需提前申请。",
    "办公室空调温度统一设定为26度，请勿私自调节。",
]
metadatas = [
    {"doc_name": "报销制度.pdf", "department": "财务部"},
    {"doc_name": "报销制度.pdf", "department": "财务部"},
    {"doc_name": "员工手册.docx", "department": "人力资源部"},
    {"doc_name": "员工手册.docx", "department": "人力资源部"},
]
ids = [f"chunk_{i}" for i in range(len(chunks))]

# 3. 向量化（文档不加前缀，normalize_embeddings=True）
embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

# 4. Chroma 建库（内存模式）
client = chromadb.Client()
collection = client.create_collection(name="company_docs")
collection.add(
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
)
print(f"向量库入库完成，共 {collection.count()} 条")

# 5. 带元数据过滤的检索（只查财务部的文档）
query = "为这个句子生成表示：单餐报销上限是多少？"  # BGE query 前缀
query_emb = model.encode([query], normalize_embeddings=True).tolist()

results = collection.query(
    query_embeddings=query_emb,
    n_results=TOP_K,
    where={"department": "财务部"},  # 元数据过滤
)

retrieved_ids = set(results["ids"][0])
print(f"\n检索到 {len(retrieved_ids)} 条：{retrieved_ids}")

# 6. 评测 Recall@K
def recall_at_k(retrieved_ids: Set[str], relevant_ids: Set[str]) -> float:
    return len(retrieved_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0

# 假设“单餐报销上限”的相关片段是 chunk_0
relevant_ids = {"chunk_0"}
r = recall_at_k(retrieved_ids, relevant_ids)
print(f"Recall@{TOP_K} = {r:.2%}")
```

> 面试要点：这段代码串联了“Embedding 向量化 → Chroma 建库 → 带元数据过滤检索 → Recall@K 评测”完整流程，可作为 RAG 索引构建的模板。关键点：建库与查询同一模型、query 加 BGE 前缀、normalize_embeddings=True、where 做元数据过滤、Recall@K 衡量检索质量上限。

---

## Q27. 索引不能直接在生产上修改参数怎么办？怎么安全调参？

**回答**

向量数据库的索引参数（如 HNSW 的 M、efConstruction）通常是在索引创建时确定的，不能直接在线修改。如果想调参，安全做法是：

1. **建新索引**：用新参数构建一个新的集合/索引，把所有数据重新加载进去。
2. **双写验证**：新旧索引同时运行，用同样的查询集对比召回率和延迟。
3. **原子切换**：验证新索引效果更好后，把查询流量一次性切到新索引，旧索引下线。
4. **灰度发布**：先把一部分流量切到新索引，观察一段时间无异常后再全量切换。

避免直接在线上索引修改参数或全量重建期间停止查询服务，保证 RAG 系统持续可用。

> 面试要点：索引参数一般不可在线修改。安全调参流程：用新参数建新索引→双写验证（对比 Recall 和延迟）→原子切换/灰度发布→下线旧索引。保证线上系统持续可用。

---

## Q28. 向量检索返回的距离分数怎么理解？越小越好吗？

**回答**

距离分数的含义取决于向量数据库用的度量方式：

| 度量 | 含义 | 越小越好？ | 备注 |
|------|------|-----------|------|
| L2 距离（欧氏距离） | 两个向量的直线距离 | ✅ 是 | 归一化向量上与余弦相似度单调反比 |
| 余弦距离 | 1 - 余弦相似度 | ✅ 是 | 值域 0～2，0 表示方向完全一致 |
| 内积（IP） | 两个向量点积 | ❌ 否，越大越好 | 归一化后等于余弦相似度，越大越相似 |

比如 Chroma 默认返回的是“距离”（余弦距离或 L2），值越小表示越相似；如果用 OpenAI Embedding（已归一化）并配置内积，则分数越大越相似。

**注意**：不同数据库、不同度量的分数不可直接比较。Chroma 的 0.12 和 Milvus 的 0.85 不代表同一含义，只能在同一数据库、同一索引内比较相对大小。

> 面试要点：L2 距离和余弦距离越小越相似；内积（IP）越大越相似。不同数据库和度量的分数不可横向比较，只能在同一系统内比较相对大小。

---

## Q29. 向量维度高了会带来什么问题？有什么降维办法？

**回答**

向量维度过高会带来：

1. **存储更大**：向量数 × 维度 × 4 字节，100 万条向量从 384 维到 1536 维，存储增加 4 倍。
2. **计算更慢**：距离计算的乘加次数随维度线性增长，查询延迟上升。
3. **内存压力**：HNSW 图需要把向量和图都留在内存里，高维会撑爆内存。

**降维办法**：

1. **选更小维度的模型**：比如从 bge-large（1024 维）换成 bge-small（384 维），通常对中文检索影响有限，但存储和计算减少 2～3 倍。
2. **Matryoshka Representation Learning（MRL）**：支持把高维向量截断到更短维度而不严重损失性能（如 OpenAI text-embedding-3 系列支持）。
3. **PCA 降维**：对已训练好的向量做主成分分析降维，但通常不如直接选小维度模型效果好。

> 面试要点：维度高带来存储、计算、内存三大压力。降维办法：优先选小维度模型（如 384 维）、使用支持 MRL 的模型可截断维度、或 PCA 降维。不要一上来就用最大维度，按实际召回率评测再决定。

---

## Q30. 本章小结：一个初学者该怎么构建 RAG 向量索引？

**回答**

向量索引是 RAG 检索的核心，做不好直接影响答案准确率。初学者建议按这个路径入门：

1. **选 Embedding 模型**：中文场景优先 BGE 系列（small/base/large-zh），锁定模型名和版本，建库与查询必须同一模型。
2. **注意前缀与归一化**：BGE 的 query 加 `Represent this sentence: ` 前缀，document 不加；开启 `normalize_embeddings=True`，点积即余弦相似度。
3. **选向量数据库**：入门用 Chroma（无需部署、内存模式）；生产大规模用 Milvus；不想运维用 Pinecone；已有 PostgreSQL 用 pgvector。
4. **选 ANN 索引**：HNSW 是生产中最常用的默认选择（百万～亿级），三个参数 M/efConstruction/efSearch 按 recall vs latency 调优。
5. **加元数据与过滤**：注入文档名、页码、部门等元数据，用 where 条件做权限隔离和范围缩小。
6. **评测召回率**：用 Recall@K 衡量检索质量，低于预期时先优化检索（Embedding/分块/重排序），而不是只改 Prompt。
7. **增量更新与安全调参**：用 MD5 判断新增/修改/删除，避免全量重建；调参时建新索引、双写验证、原子切换。

一句话总结：**RAG 向量索引不是“存进去就完事”，模型选型、前缀归一化、数据库选型、HNSW 参数、元数据过滤、召回率评测，每一步都决定了检索能不能找到对的资料。会 Python，花两三天搭建这套索引流水线，就能把公司文档变成可毫秒级检索的向量知识库。**

> 面试要点：入门路径＝选 Embedding 模型（BGE 中文优先）→注意前缀与归一化→选向量数据库（Chroma 入门/Milvus 生产）→HNSW 索引调参→元数据过滤→Recall@K 评测→增量更新与安全调参。每一步都直接影响检索质量。
