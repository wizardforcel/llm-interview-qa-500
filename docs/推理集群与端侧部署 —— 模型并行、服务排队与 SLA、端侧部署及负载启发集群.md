# 推理集群与端侧部署 —— 模型并行、服务排队与 SLA、端侧部署及负载启发集群

> 面试生普（LLM 500 题）· 推理集群与端侧部署篇。本章覆盖 LLM 推理系统落地的工程实践：端侧部署（llama.cpp、MLX、ONNX、Ollama、GGUF 量化、手机/笔记本/边缘设备、端云协同）、单机服务满足 SLA（并发模型、连续批处理、PagedAttention、TTFT/TPOT、前缀亲和、SLA 分级）、推理集群架构（TP/PP/DP/EP 并行策略、P/D 分离、超节点、KV Cache 共享、前缀路由、NIXL、异构集群）、负载启发集群设计（负载均衡、动态路由、弹性伸缩、故障边界、容量规划），共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整代码和 ASCII 图。

---

## Q1. 什么是端侧部署？为什么要把大模型跑在手机、笔记本等端设备上？

**回答**

端侧部署（Edge Deployment）是指把大语言模型（LLM）直接部署在用户本地设备上运行，而不是依赖云端服务器。端设备包括手机、平板、笔记本电脑、Apple Silicon Mac、嵌入式盒子、汽车域控制器等。

**为什么要端侧部署？** 有几个核心原因：

1. **隐私保护**：用户的对话内容、文档、笔记本地处理，不需要上传到云端，避免敏感信息外泄。
2. **离线可用**：没有网络也能跑，适合飞机、高铁、偏远地区等无网络环境。
3. **零推理成本**：不需要付 API 调用费，也不需要租云服务器，只有设备电费和硬件折旧成本。
4. **低延迟**：本地推理不需要网络往返，对于短对话、代码补全、语音转写等场景，响应更快。

当然，端侧也有局限：模型大小受限于设备内存（通常只能跑 1B～35B 的小模型，量化后）、吞吐量远低于数据中心 GPU（通常几十 tok/s 而非每秒上百 token）、发热和功耗较高。

常见端侧运行栈：

| 栈 | 平台 | 格式 | 特点 |
|------|-----------|------|------|
| llama.cpp | Win/Mac/Linux/移动 | GGUF | 社区最大，CPU+GPU 层卸载，最通用 |
| MLX | Apple Silicon | MLX/safetensors | 统一内存，苹果芯片优化 |
| ONNX Runtime | 跨平台 | ONNX | 企业集成、DirectML/CoreML |
| Ollama | 桌面 | 封装 llama.cpp | 一键拉模型，开发者友好 |
| MLC-LLM | 移动/Web | WASM/MLC | 浏览器和手机端原生运行 |

ASCII 图示（端侧 vs 云端）：

```
云端推理：                    端侧推理：
┌─────────┐                  ┌─────────┐
│ 用户设备  │                  │ 用户设备  │
│  (手机/   │                  │  (手机/   │
│  笔记本)  │                  │  笔记本)  │
└────┬─────┘                  └────┬─────┘
     │ 网络                      │  本地推理
     ▼                           ▼
┌─────────────┐           ┌─────────────┐
│ 云端模型服务  │           │ 本地模型    │
│  (GPU 集群)   │           │ (llama.cpp/ │
│               │           │  MLX/Ollama)│
└─────────────┘           └─────────────┘
   数据外传，         本地处理，无需网络
   有延迟，           隐私更好
   有费用             零 API 费用
```

> 面试要点：端侧部署是把大模型跑在用户本地设备（手机、笔记本、嵌入式）上，优点是隐私好、离线可用、零 API 费用、低延迟；缺点是模型受限于设备内存、吞吐量低、发热功耗高。常见运行栈包括 llama.cpp、MLX、ONNX Runtime、Ollama、MLC-LLM。

---

## Q2. 端侧常用的模型格式有哪些？GGUF 是什么？

**回答**

端侧模型通常不直接用原始 PyTorch 权重，而是转换成专门的推理格式。最常见的是 **GGUF**（llama.cpp 的统一格式），它把模型权重、分词器、元数据（上下文长度、旋转嵌入参数等）打包在一个文件里，方便加载。

常见端侧模型格式：

| 格式 | 适用运行时 | 特点 |
|------|-----------|------|
| GGUF | llama.cpp、Ollama、LM Studio | 单文件，包含权重+分词器+元数据，社区最大 |
| MLX | Apple Silicon (Apple Silicon Mac) | 苹果统一内存架构优化，Mac 上运行快 |
| ONNX | ONNX Runtime、DirectML、CoreML | 跨平台，企业集成，适合嵌入式 |
| TFLite | Android/iOS | Google 官方移动端推理框架 |
| ExecuTorch | 移动端 | Meta 推出，面向边缘设备 |

以 **llama.cpp** 为例，它使用 GGUF 格式，支持 CPU 纯运行，也可以把部分层卸载到 GPU（Metal/CUDA/Vulkan），是目前端侧部署最通用的方案。

典型 7B 模型（Q4_K_M 量化，约 4.1 GB 权重）的显存/内存规划：

```
7B 模型 Q4_K_M：
├─ 权重：约 4.1 GB
├─ KV Cache（4K 上下文）：约几百 MB
├─ 运行时开销：约 1 GB
└─ 总计：约 5～6 GB，笔记本（8GB/16GB 内存）可跑
```

> 面试要点：端侧常用模型格式有 GGUF（llama.cpp/Ollama）、MLX（Apple Silicon）、ONNX（跨平台）、TFLite（移动端）。GGUF 是目前社区最大的端侧模型格式，单文件包含权重和元数据，方便部署。7B 模型量化后约 4～5 GB 权重，笔记本（8GB/16GB 内存）通常可跑。

---

## Q3. 怎么在笔记本/手机上跑第一个本地大模型？

**回答**

以 **llama.cpp + Ollama** 为例，这是目前端侧部署最简单的路径：

**步骤 1：安装 Ollama**
- Mac/Windows：从 ollama.com 下载桌面应用，一键安装。
- Linux：`curl -fsSL https://ollama.com/install.sh | sh`

**步骤 2：拉取并运行模型**
```bash
# 拉取并运行 3B 小模型（适合 8GB 内存笔记本）
ollama run gemma3:1b

# 或者启动服务端（OpenAI 兼容 API）
ollama serve
# 然后另开终端：
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:1b", "messages": [{"role": "user", "content": "你好"}]}'
```

**步骤 3：在 Python 中调用**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")
resp = client.chat.completions.create(
    model="gemma3:1b",
    messages=[{"role": "user", "content": "用一句话解释什么是大模型"}]
)
print(resp.choices[0].message.content)
```

对于 **Apple Silicon Mac**，也可以直接用 **MLX**：
```bash
pip install mlx-lm
python -m mlx_lm.generate --model mlx-community/gemma-3-1b-4bit
```

对于 **Android/iOS 手机**，可以用 **PocketPal**、**Mammoth**、**Enchanted** 等 App，直接导入 GGUF 格式模型文件，完全离线运行。

> 面试要点：端侧部署最简单的入门是 Ollama（一键安装，OpenAI 兼容 API）。Apple Silicon 用户也可直接用 MLX。手机端可用 PocketPal 等 App 导入 GGUF 模型。关键是选对量化档（Q4_K_M 是质量与大小的平衡点），7B 模型约 4～5 GB 权重，8GB 内存笔记本通常可跑。

---

## Q4. 端侧模型的量化怎么选？Q4_K_M 是什么意思？

**回答**

量化是指把模型权重从高精度（如 FP32 4 字节）压缩到低精度（如 INT4 0.5 字节），从而大幅减小模型体积，让大模型能跑在内存有限的端设备上。

常见 GGUF 量化档（以 7B 模型为例）：

| 量化档 | 每参数字节 | 7B 模型大小 | 质量损失 | 适用场景 |
|---------|-----------|------------|----------|---------|
| Q8_0 | 8 bit | ~7.2 GB | 极佳 | 质量优先，内存 16GB+ |
| Q5_K_M | 5 bit | ~4.8 GB | 很好 | 内存有余，追求质量 |
| **Q4_K_M** | **4 bit** | **~4.1 GB** | **好** | **推荐默认，平衡最佳** |
| Q3_K_M | 3 bit | ~3.3 GB | 可用 | 极端内存受限 |
| Q2_K | 2 bit | ~2.7 GB | 差 | 仅限研究/实验 |

**Q4_K_M** 是目前社区最推荐的默认档：大约 3.7x 压缩， perplexity 只增加一点点，绝大多数对话、代码、摘要场景下用户几乎察觉不出质量差异，是“实用性与质量的甜点”。

Python 示例（估算不同量化下 7B 模型占用）：

```python
def estimate_model_size_gib(params_b, bits_per_param):
    """估算模型权重占用（GiB），不含 KV Cache 和运行时开销"""
    total_bytes = params_b * 10**9 * (bits_per_param / 8)
    return total_bytes / (1024**3)

# 7B 模型不同量化
for bits, name in [(16, "FP16/BF16"), (8, "INT8"), (4, "INT4/Q4_K_M"), (3, "Q3_K_M"), (2, "Q2_K")]:
    size = estimate_model_size_gib(7, bits)
    print(f"{name:>8} ({bits} bit): {size:.1f} GB")
```

输出示例：
```
FP16/BF16: 14.0 GB
INT8: 7.0 GB
INT4/Q4_K_M: 3.5 GB
Q3_K_M: 2.6 GB
Q2_K: 1.8 GB
```

注意：实际运行还需要加上 KV Cache 和运行时开销（通常再 +1～2 GB），所以 7B Q4_K_M 在 8GB 内存笔记本上通常可跑，但上下文长度不能太长。

> 面试要点：量化格式常见 FP16(2B)、INT8(1B)、INT4(0.5B)。Q4_K_M 是社区推荐的默认档，约 3.7x 压缩，质量损失小。7B 模型 Q4_K_M 约 4.1 GB 权重，加上 KV Cache 和运行时，8GB 内存笔记本通常可跑，但上下文长度建议控制在 2K～4K。

---

## Q5. 端侧部署的评估检查清单有哪些？

**回答**

把模型部署到端设备前，建议按以下清单逐项核对：

| 检查项 | 说明 |
|---------|---------|
| 模型版本与量化档 | 确认具体模型名、参数量、量化等级（如 Qwen3-4B Q4_K_M） |
| 运行时与版本 | llama.cpp/MLX/Ollama 版本，与模型是否兼容 |
| 硬件环境 | CPU/GPU/NPU 型号、内存容量、是否有独立显存 |
| 上下文长度 | 实际测试最大可加载的上下文长度，避免 OOM |
| 吞吐量 | 测量 tok/s（每秒生成 token 数），是否满足交互体验 |
| 首 token 延迟 (TTFT) | 测量从发送请求到首个 token 的时间 |
| 内存峰值 | 运行期间的峰值内存/显存占用，确保不 OOM |
| 温度与功耗 | 长时间运行后设备发热是否导致频率降频 |
| 离线能力 | 断网后是否仍能正常推理 |
| 隐私与合规 | 模型文件来源是否可信，是否包含敏感数据 |

Python 示例（端侧推理基准脚本骨架）：

```python
import time
from llama_cpp import Llama

llm = Llama(model_path="model.q4_k_m.gguf", n_ctx=4096, n_threads=8)

prompt = "请写一段关于春天的文字。"
tokens = []

start = time.perf_counter()
for chunk in llm(prompt, max_tokens=200, stream=True):
    tokens.append(chunk)
    if chunk["choices"][0].get("text"):
        first_token_time = time.perf_counter() - start
        break

total_time = time.perf_counter() - start
print(f"首 token 延迟：{first_token_time:.3f} 秒")
print(f"总耗时：{total_time:.3f} 秒")
print(f"生成 token 数：{len(tokens)}")
print(f"吞吐量：{len(tokens)/total_time:.2f} tok/s")
```

> 面试要点：端侧部署评估清单包括模型版本与量化档、运行时与硬件环境、上下文长度、吞吐量、TTFT、内存峰值、温度功耗、离线能力、隐私合规。建议建立可回归的评测脚本，记录首 token 延迟和总吞吐量。

---

## Q6. 端侧与云端怎么混合使用？（混合云边架构）

**回答**

端侧模型虽然好，但受限于设备算力，复杂推理（长文档分析、复杂代码、多步推理）仍不如云端大模型。因此，生产环境通常采用**端云混合架构**：

- **本地端侧**：处理高频、隐私敏感、低延迟要求高的任务（如代码补全、笔记摘要、语音转写、简单问答）。
- **云端大模型**：处理复杂推理、长文档、多模态、专业领域知识等端侧跑不动的任务。

**路由策略示例：**

```
用户请求
    │
    ▼
┌─────────────────┐
│ 路由决策（边缘网关/本地客户端）│
├─────────────────┤
│  简单/隐私/离线  │────▶ 端侧模型（llama.cpp/MLX/Ollama）│
│  复杂/多模态/长文档 │────▶ 云端模型（API 调用）          │
└─────────────────┘
```

代码示例（简单的路由逻辑）：

```python
def route_request(user_input, local_llm, cloud_llm, max_local_tokens=512):
    """
    简单的路由逻辑：短文本本地处理，长文本/复杂推理上云。
    """
    # 简单启发式：按输入长度和关键词判断
    if len(user_input) < max_local_tokens and not any(
        kw in user_input for kw in ["分析", "代码", "数学", "逻辑", "长文档"]
    ):
        return local_llm.chat(user_input)  # 端侧处理
    else:
        return cloud_llm.chat(user_input)  # 云端处理
```

**混合架构的优势：**
- 隐私敏感的数据留在本地，只有非敏感的复杂推理上云。
- 成本可控：只对少量复杂请求调用云端 API。
- 可用性：云端不可用时，端侧模型仍可提供基础服务（降级）。

> 面试要点：端云混合架构将请求按复杂度和隐私要求分流：简单、隐私、低延迟场景走端侧；复杂、多模态、长文档场景走云端。路由可基于输入长度、关键词、用户偏好等简单规则实现，兼顾成本、隐私和体验。

---

## Q7. 单机推理服务的 QPS 和并发怎么理解？

**回答**

单机推理服务的核心指标是 **QPS（Queries Per Second，每秒查询数）** 和 **并发数（同时处理的请求数）**。

LLM 推理不是传统 Web 服务那样可以随便并发的，因为每个请求都要占用显存（KV Cache）和算力。如果并发过高，会导致：
- 显存溢出（OOM）
- 排队等待，TTFT 和 TPOT 大幅上升
- 甚至整个服务卡死

**估算单机并发能力的简化方法：**

假设模型 7B，Q4_K_M 量化，单卡 8GB 显存：
- 权重：约 4 GB
- 剩余给 KV Cache：约 3 GB
- 每 token KV Cache 占用约 0.5～1 KB（取决于模型结构）
- 假设序列长度 2048，每请求 KV Cache 约 1～2 MB
- 则单卡可并发约 1500～3000 个 token 的并发请求

更精确的估算需要结合具体模型和批次大小进行实测。

ASCII 图示（单机并发与延迟关系）：

```
并发请求数
   ▲
   │      / 吞吐量上升
   │     /
   │    / 
   │   /   饱和区（排队延迟飙升）
   │  /────────
   │ /
   │/
   └────────────────▶ 并发数
   
   建议工作区间：并发适中，排队延迟仍在可接受范围
```

> 面试要点：单机推理服务的 QPS 和并发受显存（KV Cache 容量）和算力限制。不能无限增加并发，否则会导致 OOM 或排队延迟飙升。需要根据模型大小、量化等级、序列长度估算单卡可承载的并发请求数，并通过压测确定最佳并发配置。

---

## Q7(续). 什么是连续批处理（Continuous Batching）？为什么对单机服务很重要？

**回答**

传统的**静态批处理**（Static Batching）要求所有请求一起到达、一起处理、一起输出。这意味着短请求必须等长请求处理完（头阻塞），GPU 利用率低，TTFT 高。

**连续批处理（Continuous Batching）**允许请求独立进出：一个请求的 Decode 完成后立即输出，新请求可以立即填补空出的位置。GPU 始终有任务可做，利用率更高。

优势：
1. **短请求不再被长请求阻塞**，TTFT 大幅降低。
2. **GPU 利用率更高**，因为批次始终保持满载。
3. **整体吞吐量提升**，尤其适合交互式对话场景（用户问答穿插，长短不一）。

vLLM、TensorRT-LLM、SGLang 等现代推理框架都默认采用连续批处理。

ASCII 图示（静态批处理 vs 连续批处理）：

```
静态批处理（所有请求同步开始、同步结束）：
请求 A（长）：[==========]
请求 B（短）：[==]        ← 必须等 A 完成
请求 C（短）：[==]        ← 必须等 A 完成
总时间：取决于最长请求 A

连续批处理（独立进出）：
请求 A（长）：[==========]
请求 B（短）：  [==]      ← A 还在跑时，B 已经开始
请求 C（短）：     [==]    ← B 完成后 C 立即填补
总时间：接近最长请求 A，但中间请求不再阻塞
```

> 面试要点：连续批处理允许请求独立进出，避免短请求被长请求阻塞，GPU 利用率更高，TTFT 和整体吞吐都更好。现代推理框架（vLLM、TensorRT-LLM、SGLang）均默认采用连续批处理。

---

## Q8. 什么是 TTFT 和 TPOT？在单机服务中怎么测量？

**回答**

- **TTFT（Time-To-First-Token）**：从用户发送请求到收到**第一个输出 token** 的时间。包含排队等待 + Prefill 计算（把整个 Prompt 编码进 KV Cache）。对于长 Prompt，TTFT 通常较高。
- **TPOT（Time-Per-Output-Token）**：从第一个 token 到后续每个 token 的平均间隔时间。对应 Decode 阶段，受内存带宽限制（每次生成一个 token 都要把整个 KV Cache 从显存读一遍）。

Python 示例（测量 TTFT 和 TPOT）：

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

start = time.perf_counter()
first_token_time = None
token_count = 0

stream = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "请写一篇关于春天的文章，不少于 300 字。"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        if first_token_time is None:
            first_token_time = time.perf_counter() - start
        token_count += 1

total_time = time.perf_counter() - start
ttft = first_token_time
tpot = (total_time - first_token_time) / max(token_count - 1, 1)

print(f"TTFT: {ttft:.3f} 秒")
print(f"TPOT: {tpot:.3f} 秒/token")
print(f"总 token 数：{token_count}")
```

> 面试要点：TTFT 是请求到首个 token 的时间（受 Prefill 影响）；TPOT 是后续每个 token 的间隔（受 Decode 内存带宽限制）。通过流式调用并记录每个 token 的到达时间，可以分别计算 TTFT 和 TPOT。

---

## Q9. 什么是 PagedAttention？它解决了什么问题？

**回答**

PagedAttention 是 vLLM 提出的 KV Cache 管理技术，借鉴操作系统虚拟内存分页的思想。

**传统问题**：传统推理引擎为每个请求预先分配连续的、最大可能长度的 KV Cache 空间。这导致：
- 大量内存碎片（分配了但没用完）。
- 无法充分利用显存，并发上限低。

**PagedAttention 的做法：**
把 KV Cache 分成固定大小的“页”（blocks，通常 16 或 32 个 token），按需分配，不要求连续。不同请求的 KV Cache 可以散落在不同物理页框中，通过页表映射逻辑地址。

**优势：**
1. 消除外部碎片，显存利用率大幅提高。
2. 支持前缀缓存（Prefix Caching）：多个请求共享相同前缀的 KV Cache 页，避免重复计算。
3. 支持动态换入换出（Swapping）：把不常用的 KV Cache 页换到 CPU 内存，腾出 GPU 显存。

> 面试要点：PagedAttention 把 KV Cache 分成固定页按需分配，消除外部碎片，支持前缀缓存共享和动态换入换出，显著提高显存利用率和并发能力，是 vLLM 等现代推理框架的核心技术。

---

## Q10. 单机服务的 SLA 指标有哪些？怎么设定？

**回答**

SLA（Service Level Agreement，服务等级协议）是对服务质量的承诺。推理系统常见 SLA 指标：

| 指标 | 含义 | 典型阈值（示例） |
|------|------|-----------------|
| TTFT P99 | 首 token 时间 99 分位 | < 2 秒（短对话） |
| TPOT P99 | 单 token 生成时间 99 分位 | < 100 ms |
| 错误率 | 请求失败率 | < 0.1% |
| 吞吐量 | 每秒生成 token 数 | 根据模型和卡数而定 |
| 可用性 | 服务正常运行时间 | > 99.9% |

**设定 SLA 的原则：**
- 基于真实压测数据，而非拍脑袋。
- 区分不同场景（实时对话 vs 离线批处理），设不同 SLO。
- 留有余量，不要把指标设在极限值附近。

> 面试要点：推理系统 SLA 常见指标包括 TTFT P99、TPOT P99、错误率、吞吐量、可用性。设定 SLA 应基于真实压测数据，区分不同业务场景，并留有余量，避免指标设在极限值附近导致频繁违约。

---

## Q10. 单机推理服务的常见瓶颈有哪些？怎么定位？

**回答**

单机推理服务常见瓶颈：

| 瓶颈 | 表现 | 定位方法 |
|------|------|---------|
| 显存不足 | OOM，无法加载模型或批次受限 | 监控显存占用，计算 权重+KV Cache+激活值 总和 |
| 内存带宽 | Decode 阶段 TPOT 高，GPU 利用率低 | Roofline 图看是否在斜线附近，rocprofv3 看内存带宽利用率 |
| 计算不足 | Prefill 阶段延迟高，GPU 计算饱和 | Roofline 图看是否在水平线附近，MFU 是否很高 |
| 排队阻塞 | 高并发下 P99 延迟飙升 | 监控队列长度、排队时间，对比吞吐量曲线 |

定位流程：先看端到端延迟是否达标 → 若不达标，拆分为 Prefill/Decode 阶段 → Prefill 慢看计算带宽（计算密集型）→ Decode 慢看内存带宽（存储密集型）→ 并发高慢看排队模型。

> 面试要点：单机推理服务瓶颈包括显存不足、内存带宽、计算不足、排队阻塞。定位遵循：端到端延迟 → 拆分 Prefill/Decode → Roofline 判瓶颈类型 → rocprofv3 看硬件计数器 → 排队模型看并发能力。

---

## Q11. 什么是前缀路由（Prefix Routing）？它和 KV 亲和有什么关系？

**回答**

前缀路由（Prefix Routing）是指将具有相同前缀（如同一段系统提示、同一文档上下文）的请求路由到同一个推理实例，以便复用该实例已缓存的 KV Cache 前缀部分。

KV 亲和（KV Affinity）是前缀路由的一种具体表现：把同一会话（同一用户的多轮对话）路由到同一个实例，复用该会话的 KV Cache。

**为什么重要？**
- 避免重复计算 Prefill：同一会话的多轮对话共享前文，不需要每次都从头重算。
- 降低 TTFT：命中前缀缓存的请求，首 token 返回更快。
- 提高整体吞吐：减少重复计算，释放算力给其他请求。

> 面试要点：前缀路由把具有相同前缀（如同一会话的历史上下文）的请求路由到同一个实例，复用已缓存的 KV Cache 前缀，避免重复 Prefill 计算，降低 TTFT，提升系统吞吐。

---

## Q12. 端侧模型和云端模型怎么选型？有什么区别？

**回答**

| 维度 | 端侧部署 | 云端部署 |
|------|---------|---------|
| 隐私 | 数据不出设备，最佳 | 数据上传云端，有泄露风险 |
| 网络依赖 | 离线可用 | 必须联网 |
| 延迟 | 低（无网络往返） | 较高（有网络往返） |
| 模型大小 | 受限于设备内存（通常 1B～35B 量化） | 无上限（可跑 70B 甚至更大） |
| 推理成本 | 零 API 费用，仅设备电费 | 按 Token 付费，成本随使用量增加 |
| 更新 | 需要重新下载模型文件 | 云端更新，用户无感知 |

**选型建议：**
- **隐私敏感、离线场景、低延迟要求**：优先端侧部署。
- **复杂推理、多模态、长文档、高吞吐**：优先云端部署。
- **生产环境**：常用端云混合架构，简单请求走端侧，复杂请求走云端。

> 面试要点：端侧部署适合隐私敏感、离线、低延迟场景，但模型大小受限于设备内存；云端部署模型无上限，适合复杂推理，但需联网并按 Token 付费。生产环境常用端云混合，简单请求端侧，复杂请求云端。

---

## Q13. 端侧推理的常见瓶颈有哪些？怎么优化？

**回答**

端侧推理常见瓶颈：

| 瓶颈 | 表现 | 优化方法 |
|------|------|---------|
| 显存/内存不足 | 模型加载失败，OOM | 量化（INT4/Q4_K_M）、减少上下文长度、使用更小模型 |
| 推理速度慢 | 每秒生成 token 数低（<5 tok/s） | 选轻量模型、减少上下文长度、使用 GPU 加速（Metal/CUDA/Vulkan） |
| 发热功耗高 | 设备发热、电池耗电快 | 限制并发、降低量化精度、限制最大 token 数 |
| 首 token 延迟高 | 模型加载慢、首次推理慢 | 预加载模型、保持进程常驻、使用较小模型 |

> 面试要点：端侧推理瓶颈包括显存/内存不足、推理速度慢、发热功耗高、首 token 延迟高。优化方法：量化（选 Q4_K_M）、减少上下文长度、选轻量模型、启用 GPU/NPU 加速、预加载模型。

---

## Q14. 什么是前缀缓存（Prefix Caching）？它在推理集群中怎么用？

**回答**

前缀缓存是指把多个请求共享的公共前缀（如同一段系统提示、同一文档上下文、同一会话的历史对话）的 KV Cache 缓存下来，后续请求直接复用，避免重复计算 Prefill。

**优势：**
- 降低 TTFT：命中前缀缓存的请求，无需重新计算前文。
- 提高吞吐：减少重复计算，释放算力给其他请求。
- 适合场景：多轮对话、文档问答（多个用户问同一文档的不同问题）、代码助手（同一代码库的多个问题）。

**实现方式：**
- 推理引擎（如 vLLM）内置前缀缓存支持，基于请求的哈希值（如模型指纹 + 前缀文本哈希）作为缓存键。
- 路由层可以将具有相同前缀的请求调度到同一个实例，提高缓存命中率。

> 面试要点：前缀缓存把多个请求共享的公共前缀（如系统提示、会话历史）的 KV Cache 缓存起来，后续请求直接复用，避免重复 Prefill 计算，降低 TTFT，提高系统吞吐。适合多轮对话、文档问答等场景。

---

## Q15. 什么是 P/D 分离（Prefill/Decode 分离）？为什么要做？

**回答**

P/D 分离（Prefill/Decode Decoupling）是指把 Prefill 阶段（计算密集型）和 Decode 阶段（内存带宽受限）部署到不同的推理实例上。

**为什么要分离？**
- Prefill 需要大量计算（矩阵乘法），适合算力强的实例。
- Decode 需要大量内存带宽（每次读整个 KV Cache），适合带宽高的实例。
- 如果混部，Prefill 的大批量计算会挤占 Decode 的内存带宽，导致 Decode TPOT 上升；反之亦然。

**架构示意：**

```
用户请求
    │
    ▼
┌─────────────┐
│ 路由调度层    │─────────────┐
└──────┬──────┘         │
     │                   │
     ▼                   ▼
┌─────────┐     ┌─────────────┐
│ Prefill   │     │ Decode      │
│ 池（计算型）│     │ 池（带宽型）   │
└─────────┘     └─────────────┘
     │                   │
     └───────┬───────────┘
             │
             ▼
        返回最终答案给用户
```

P/D 分离后，Prefill 池和 Decode 池可以独立扩缩容，根据负载特征独立调整资源，避免相互干扰。

> 面试要点：P/D 分离把计算密集型的 Prefill 阶段和内存带宽受限的 Decode 阶段部署到不同实例，避免两者互相抢占资源。适合混合负载场景，能显著提升整体吞吐和延迟稳定性。

---

## Q16. 推理集群的常见并行策略有哪些？各适合什么场景？

**回答**

| 策略 | 全称 | 切分维度 | 适合场景 | 通信特点 |
|------|------|---------|---------|---------|
| TP | Tensor Parallelism | 模型内层切分（矩阵行/列） | 单节点内多卡，模型放不下单卡 | 高频通信，适合 NVLink |
| PP | Pipeline Parallelism | 模型层切分（前 N 层/后 N 层） | 跨节点，模型太大放不下单节点 | 低频通信，有气泡 |
| DP | Data Parallelism | 数据切分（不同请求） | 并发请求多，模型可完整放入单卡 | 无模型参数通信，仅梯度/结果同步 |
| EP | Expert Parallelism | MoE 专家切分 | MoE 模型（如 DeepSeek-MoE） | 专家间 All-to-All 通信 |

**选型建议：**
- **单节点内多卡**：优先 TP，NVLink 带宽高，TP 的高频通信开销可接受。
- **多节点**：优先 PP，跨节点 TP 的高频通信开销太大；PP 的低频通信更适合跨节点网络。
- **超大模型（放不下单节点）**：必须用 PP 把模型切分到多节点，再在节点内用 TP 填满每个节点的所有卡。
- **MoE 模型**：可加 EP 将不同专家切分到不同卡上。

> 面试要点：并行策略包括 TP（模型内层切分，单节点内多卡）、PP（模型层切分，跨节点）、DP（数据切分，提高并发）、EP（MoE 专家切分）。选型遵循：单节点优先 TP，多节点优先 PP，超大模型用 PP+TP 混合，MoE 模型可加 EP。

---

## Q17. 什么是超节点（Super Node）？它解决了什么问题？

**回答**

超节点是指将数十到数百张 GPU 卡通过高速互联（如 NVLink、InfiniBand）在物理上组成一个超大规模的计算域，在逻辑上表现为“一张超大 GPU”。

**超节点的核心价值：**
1. **统一内存编址**：跨卡的 GPU 显存可以像本地内存一样直接寻址，模型参数和 KV Cache 可以跨卡共享，无需频繁拷贝。
2. **高带宽低时延**：节点内显卡间通信带宽极高（如 NVLink 数百 GB/s），时延极低（亚毫秒级），适合 TP 和 MoE 的高频跨卡通信。
3. **减少跨节点通信**：将原本需要跨机房通信的计算压缩到同一物理机柜内，大幅降低网络开销。

**适用场景：**
- 大模型训练和推理（单卡显存放不下模型）。
- MoE 模型（专家并行需要大量跨卡通信）。
- 长上下文推理（KV Cache 很大，需要跨卡共享）。

超节点不是万能药，它适合计算密集型、通信密集型的大规模模型；对于简单的小模型推理，单机多卡或多个独立小集群可能更经济。

> 面试要点：超节点将数十到数百张 GPU 通过高速互联组成逻辑上的“大 GPU”，统一内存编址、高带宽低时延，适合大模型训练推理和 MoE 模型。它不是所有场景的默认选择，中小模型通常单机多卡即可，不需要上超节点。

---

## Q18. 推理集群的负载均衡策略有哪些？怎么选？

**回答**

推理集群的负载均衡策略包括：

| 策略 | 原理 | 适用场景 | 缺点 |
|------|------|---------|------|
| 轮询（Round Robin） | 请求依次分发到各实例 | 请求耗时相似、实例规格一致 | 忽略实例当前负载，可能导致某些实例过载 |
| 最少连接（Least Connections） | 发给当前活跃请求数最少的实例 | 请求处理时间差异大 | 忽略请求复杂度差异 |
| 加权轮询（Weighted Round Robin） | 根据实例性能设置权重 | 实例规格不一致 | 权重需手动调整，不够动态 |
| 最少延迟（Least Latency） | 发给历史延迟最低的实例 | 对延迟敏感的实时服务 | 历史延迟不代表未来，可能振荡 |
| KV 亲和/前缀路由 | 相同前缀的请求发到同一实例 | 多轮对话、文档问答 | 可能导致某些实例负载过高，需配合容量保护 |

**生产环境建议：**
- 优先使用 **KV 亲和/前缀路由**，提高缓存命中率，降低 TTFT。
- 在亲和路由基础上，加上 **容量保护**（某个实例请求过多时溢出到其他实例），避免热点实例过载。
- 使用 **自动伸缩**（Autoscaling），根据 QPS 和排队长度动态增减实例。

> 面试要点：推理集群负载均衡策略包括轮询、最少连接、加权轮询、最少延迟、KV 亲和/前缀路由。生产环境优先 KV 亲和（复用缓存，降低 TTFT），并配合容量保护和自动伸缩，避免单个实例过载。

---

## Q19. 什么是弹性伸缩（Autoscaling）？在推理集群中怎么实现？

**回答**

弹性伸缩是指根据实时负载自动增减推理实例数量，以在保证服务质量（SLA）的同时控制成本。

**触发指标：**
- 排队长度（等待处理的请求数）
- P99 延迟（尾部延迟）
- GPU 利用率
- KV Cache 占用率

**实现方式：**
1. **基于指标的触发**：当 P99 延迟超过阈值或排队长度超过设定值时，自动启动新实例；当负载降低后，自动下线空闲实例。
2. **基于调度计划**：根据已知的流量模式（如工作日白天高峰、夜间低谷）预先扩缩容。
3. **混合模式**：基础容量用固定实例，突发流量用弹性伸缩。

**实施注意事项：**
- 实例启动有预热时间（模型加载需要时间），提前扩容，不要等到完全饱和才扩容。
- 缩容时要等待正在处理的请求完成，避免中断用户请求。
- 设置冷却时间（Cooldown），避免频繁扩缩容导致抖动。

> 面试要点：弹性伸缩根据实时负载自动增减推理实例，触发指标包括排队长度、P99 延迟、GPU 利用率、KV Cache 占用率。实施时注意模型加载预热时间、缩容时等待请求完成、设置冷却时间避免抖动。

---

## Q20. 推理系统的故障边界有哪些？怎么保证高可用？

**回答**

推理系统的故障边界包括：

| 故障类型 | 表现 | 应对策略 |
|---------|------|---------|
| 单个推理实例崩溃 | 该实例上的所有请求失败 | 部署多个副本，请求路由到健康实例 |
| 单个 GPU 卡故障 | 该卡上的所有模型实例不可用 | 跨多个物理机部署副本，避免单点故障 |
| 模型加载失败 | 实例启动但无法服务 | 健康检查，自动重启或下线该实例 |
| 网络分区 | 集群内部分实例无法通信 | 多机房部署，跨地域负载均衡 |
| 数据中心宕机 | 整个集群不可用 | 多地域部署，灾备切换 |

**高可用设计原则：**
- **无单点故障**：关键组件（推理实例、路由器、负载均衡器）都有冗余。
- **快速故障检测**：健康检查（Health Check）定期探测实例状态，失败后自动下线。
- **优雅降级**：当部分实例不可用时，系统仍能服务剩余请求，而不是整体崩溃。
- **备份与恢复**：模型权重、配置文件定期备份，支持快速重新部署。

> 面试要点：推理系统故障边界包括单实例崩溃、单卡故障、模型加载失败、网络分区、数据中心宕机。高可用设计遵循：无单点故障、快速故障检测、优雅降级、备份恢复。

---

## Q21. 用 Python 实现一个简单的基于负载的推理请求路由器，代码长什么样？

**回答**

下面是一个基于 Python 的轻量级推理请求路由器示例，根据各实例的当前并发数和延迟动态选择目标实例：

```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import aiohttp

@dataclass
class InstanceInfo:
    instance_id: str
    url: str
    active_requests: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0
    last_check: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests else 0.0

    @property
    def score(self) -> float:
        """越低越好：结合当前并发和平均延迟"""
        return self.active_requests * 10 + self.avg_latency_ms

class LoadBalancedRouter:
    def __init__(self):
        self.instances: Dict[str, InstanceInfo] = {}

    def register(self, instance_id: str, url: str):
        self.instances[instance_id] = InstanceInfo(instance_id, url)

    def unregister(self, instance_id: str):
        self.instances.pop(instance_id, None)

    def select(self) -> Optional[InstanceInfo]:
        if not self.instances:
            return None
        # 选择 score 最小的实例（负载最轻且延迟最低）
        return min(self.instances.values(), key=lambda inst: inst.score)

    async def route(self, request_data: dict) -> dict:
        instance = self.select()
        if instance is None:
            raise RuntimeError("No available instance")

        start = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{instance.url}/v1/chat/completions",
                                        json=request_data, timeout=30) as resp:
                    result = await resp.json()
            latency_ms = (time.perf_counter() - start) * 1000
            instance.active_requests -= 1
            instance.total_requests += 1
            instance.total_latency_ms += latency_ms
            return result
        except Exception as e:
            instance.active_requests -= 1
            raise

    async def acquire(self, instance_id: str):
        self.instances[instance_id].active_requests += 1

    def release(self, instance_id: str):
        if instance_id in self.instances:
            self.instances[instance_id].active_requests = max(
                0, self.instances[instance_id].active_requests - 1
            )

# 使用示例
router = LoadBalancedRouter()
router.register("inst-1", "http://instance-1:8000")
router.register("inst-2", "http://instance-2:8000")

# 模拟发送请求
async def main():
    await router.acquire("inst-1")
    result = await router.route({
        "model": "qwen3:4b",
        "messages": [{"role": "user", "content": "你好"}]
    })
    print(result)

asyncio.run(main())
```

> 面试要点：简单负载均衡路由器根据每个实例的当前并发数和平均延迟动态选择目标实例，score 越小越优先。实际生产中还需加入健康检查、熔断、重试和 Prometheus 指标上报。

---

## Q22. 端侧模型和云端模型如何混合部署？混合架构的优势是什么？

**回答**

端云混合架构是指将简单、隐私敏感、低延迟要求高的请求在端侧（手机、笔记本）本地处理，将复杂、多模态、长文档的请求发送到云端大模型处理。

**混合架构示意：**

```
用户请求
    │
    ▼
┌─────────────────┐
│ 路由决策层        │─────────────┐
│ (本地客户端/网关) │         │
└──────┬────────┘         │
     │                   │
     ▼                   ▼
┌─────────┐      ┌─────────────────┐
│ 端侧模型   │      │ 云端大模型      │
│ (llama.cpp/ │      │ (API 调用)    │
│  MLX/Ollama)│      │               │
└─────────┘      └─────────────────┘
```

**混合架构的优势：**
1. **隐私保护**：敏感数据留在本地，只有非敏感的复杂推理上云。
2. **成本可控**：只对少量复杂请求调用云端 API，大量简单请求在端侧免费处理。
3. **可用性**：云端不可用时，端侧模型仍可提供基础服务（降级）。
4. **体验平衡**：简单请求在端侧秒级响应，复杂请求在云端获得高质量答案。

**路由逻辑示例（简化版）：**

```python
def route_request(user_input: str, local_llm, cloud_llm,
                  max_local_tokens: int = 512):
    """
    简单路由逻辑：
    - 输入短且无复杂关键词 → 端侧处理
    - 输入长或包含复杂关键词 → 云端处理
    """
    complex_keywords = ["代码", "数学", "逻辑", "长文档", "分析", "总结"]
    
    is_simple = (
        len(user_input) < max_local_tokens and
        not any(kw in user_input for kw in complex_keywords)
    )
    
    if is_simple:
        return local_llm.chat(user_input)  # 端侧
    else:
        return cloud_llm.chat(user_input)  # 云端
```

> 面试要点：端云混合架构按请求复杂度和隐私要求分流：简单、隐私、低延迟场景走端侧；复杂、多模态、长文档场景走云端。优势是隐私好、成本可控、云端不可用时端侧仍可降级服务、体验平衡。

---

## Q23. 推理系统的常见评估工具有哪些？怎么选？

**回答**

| 工具 | 定位 | 特点 | 适用场景 |
|------|------|------|---------|
| vLLM | 开源推理框架 | PagedAttention、连续批处理、高吞吐 | 生产环境推理服务 |
| TensorRT-LLM | NVIDIA 官方推理框架 | 极致单卡性能、算子融合 | NVIDIA GPU 上追求极致性能 |
| SGLang | 开源推理框架 | 编程灵活、RadixAttention 前缀缓存 | 研究和自定义推理流程 |
| LMDeploy | 昆仑多模态推理框架 | 多模态支持、昆腾硬件优化 | 昆腾硬件生态 |
| llama.cpp | 端侧推理框架 | 纯 C/C++，无 Python 依赖 | 笔记本、嵌入式设备 |
| MLC-LLM | 移动端推理框架 | 浏览器和手机端原生运行 | iOS/Android 端侧部署 |

**选型建议：**
- 生产环境高吞吐：vLLM 或 TensorRT-LLM。
- 端侧部署：llama.cpp（笔记本）、MLX（Apple Silicon）、MLC-LLM（移动端）。
- 研究和自定义：SGLang（编程灵活）。

> 面试要点：推理系统常见评估工具有 vLLM（生产高吞吐）、TensorRT-LLM（NVIDIA 极致性能）、SGLang（研究灵活）、llama.cpp（端侧通用）、MLX（Apple Silicon）、MLC-LLM（移动端）。按硬件平台和使用场景选型。

---

## Q24. 推理系统的常见故障有哪些？怎么排查？

**回答**

| 常见故障 | 表现 | 排查方法 |
|---------|------|---------|
| 模型加载失败 | 启动报错，无法启动服务 | 检查模型文件完整性、显存是否足够、运行时版本是否匹配 |
| 显存溢出（OOM） | 请求失败，服务崩溃 | 减少批次大小、降低上下文长度、使用量化模型、增加显存 |
| TTFT 过高 | 用户等待首 token 时间长 | 检查模型大小、上下文长度、是否启用了前缀缓存、前缀命中率 |
| TPOT 过高 | 生成速度慢，用户体验差 | 检查显存带宽是否饱和、是否开启了 KV Cache 量化、是否开启了多卡 TP 导致通信开销 |
| P99 延迟飙升 | 高并发下少数请求极慢 | 检查排队模型、是否有单个长请求阻塞队列、是否启用了连续批处理 |
| 错误率高 | 请求返回错误或空响应 | 检查日志、模型版本是否匹配、提示词模板是否正确 |

> 面试要点：推理系统常见故障包括模型加载失败、显存溢出、TTFT 过高、TPOT 过高、P99 延迟飙升、错误率高。排查遵循：查看日志 → 检查显存和带宽 → 查看排队和并发 → 验证模型和配置 → 压测复现。

---

## Q25. 推理集群的容量规划怎么做？

**回答**

容量规划需要回答：已知模型、请求量、延迟要求，需要多少卡、多少机器？

**核心计算公式：**

1. **单卡显存需求：**
   ```
   显存 = 权重大小 + KV Cache(并发请求数 × 序列长度) + 激活值 + 系统开销
   ```

2. **单卡并发能力：**
   ```
   单卡最大并发 = (可用显存 - 权重显存 - 系统预留) / (每请求 KV Cache 占用)
   ```

3. **所需卡数：**
   ```
   所需卡数 = 总并发请求数 / 单卡最大并发
   ```

4. **所需机器数：**
   ```
   所需机器数 = 所需卡数 / 单卡 GPU 数
   ```

**Python 示例（容量规划计算器）：**

```python
def plan_inference_cluster(
    num_params_b: float,      # 参数量 (B)
    seq_len: int,             # 序列长度
    batch_size: int,          # 目标并发批次
    weight_bytes: float = 2,  # FP16=2, INT8=1, INT4=0.5
    kv_bytes_per_token: float = 2 * 8192 * 2 / 1024**3,  # 粗略估算
    gpu_mem_gib: float = 80,  # 单卡显存 (如 H100 80GB)
    system_overhead: float = 0.1,
    gpus_per_node: int = 8
) -> dict:
    # 1. 权重显存
    weight_gib = num_params_b * 10**9 * weight_bytes / (1024**3)
    
    # 2. 单请求 KV Cache 显存（简化估算）
    kv_per_req_gib = (2 * 80 * 128 * seq_len * kv_bytes) / (1024**3)
    
    # 3. 单卡可容纳的并发请求数
    available_gib = gpu_mem_gib * (1 - system_overhead) - weight_gib
    max_concurrency = int(available_gib / kv_per_req_gib)
    
    # 4. 所需卡数
    num_gpus = int(batch_size / max_concurrency) + 1
    num_nodes = int(num_gpus / gpus_per_node) + 1
    
    return {
        "weight_gib": round(weight_gib, 2),
        "kv_per_req_gib": round(kv_per_req_gib, 4),
        "max_concurrency_per_gpu": max_concurrency,
        "num_gpus": num_gpus,
        "num_nodes": num_nodes
    }

# 示例：70B 模型，FP16，序列长度 4096，目标并发 64，单卡 80GB，每卡 8 张
plan = plan_inference_cluster(
    num_params_b=70,
    seq_len=4096,
    batch_size=64,
    weight_bytes=2,
    gpu_mem_gib=80,
    gpus_per_node=8
)
print(plan)
```

> 面试要点：容量规划需要算清权重显存、KV Cache 显存、激活值显存，算出单卡能承载多少并发请求，再据此算出需要多少卡、多少机器。要留有系统开销余量（通常 10%），不要压到极限。

---

## Q26. 端侧模型和云端模型如何混合部署？混合架构的优势是什么？

**回答**

端云混合架构是指将简单、隐私敏感、低延迟要求高的请求在端侧（手机、笔记本）本地处理，将复杂、多模态、长文档的请求发送到云端大模型处理。

**混合架构示意：**

```
用户请求
    │
    ▼
┌─────────────────┐
│ 路由决策层        │─────────────┐
│ (本地客户端/网关) │         │
└──────┬────────┘         │
     │                   │
     ▼                   ▼
┌─────────┐      ┌─────────────────┐
│ 端侧模型   │      │ 云端大模型      │
│ (llama.cpp/ │      │ (API 调用)    │
│  MLX/Ollama)│      │               │
└─────────┘      └─────────────────┘
```

**混合架构的优势：**
1. **隐私保护**：敏感数据留在本地，只有非敏感的复杂推理上云。
2. **成本可控**：只对少量复杂请求调用云端 API，大量简单请求在端侧免费处理。
3. **可用性**：云端不可用时，端侧模型仍可提供基础服务（降级）。
4. **体验平衡**：简单请求在端侧秒级响应，复杂请求在云端获得高质量答案。

**路由逻辑示例（简化版）：**

```python
def route_request(user_input: str, local_llm, cloud_llm,
                  max_local_tokens: int = 512):
    """
    简单路由逻辑：
    - 输入短且无复杂关键词 → 端侧处理
    - 输入长或包含复杂关键词 → 云端处理
    """
    complex_keywords = ["代码", "数学", "逻辑", "长文档", "分析", "总结"]
    
    is_simple = (
        len(user_input) < max_local_tokens and
        not any(kw in user_input for kw in complex_keywords)
    )
    
    if is_simple:
        return local_llm.chat(user_input)  # 端侧
    else:
        return cloud_llm.chat(user_input)  # 云端
```

> 面试要点：端云混合架构按请求复杂度和隐私要求分流：简单、隐私、低延迟场景走端侧；复杂、多模态、长文档场景走云端。优势是隐私好、成本可控、云端不可用时端侧仍可降级服务、体验平衡。

---

## Q26(续). 推理系统的常见评估工具有哪些？怎么选？

**回答**

| 工具 | 定位 | 特点 | 适用场景 |
|------|------|------|---------|
| vLLM | 开源推理框架 | PagedAttention、连续批处理、高吞吐 | 生产环境推理服务 |
| TensorRT-LLM | NVIDIA 官方推理框架 | 极致单卡性能、算子融合 | NVIDIA GPU 上追求极致性能 |
| SGLang | 开源推理框架 | 编程灵活、RadixAttention 前缀缓存 | 研究和自定义推理流程 |
| LMDeploy | 昆仑多模态推理框架 | 多模态支持、昆腾硬件优化 | 昆腾硬件生态 |
| llama.cpp | 端侧推理框架 | 纯 C/C++，无 Python 依赖 | 笔记本、嵌入式设备 |
| MLC-LLM | 移动端推理框架 | 浏览器和手机端原生运行 | iOS/Android 端侧部署 |

**选型建议：**
- 生产环境高吞吐：vLLM 或 TensorRT-LLM。
- 端侧部署：llama.cpp（笔记本）、MLX（Apple Silicon）、MLC-LLM（移动端）。
- 研究和自定义：SGLang（编程灵活）。

> 面试要点：推理系统常见评估工具有 vLLM（生产高吞吐）、TensorRT-LLM（NVIDIA 极致性能）、SGLang（研究灵活）、llama.cpp（端侧通用）、MLX（Apple Silicon）、MLC-LLM（移动端）。按硬件平台和使用场景选型。

---

## Q27. 推理系统的常见故障有哪些？怎么排查？

**回答**

| 常见故障 | 表现 | 排查方法 |
|---------|------|---------|
| 模型加载失败 | 启动报错，无法启动服务 | 检查模型文件完整性、显存是否足够、运行时版本是否匹配 |
| 显存溢出（OOM） | 请求失败，服务崩溃 | 减少批次大小、降低上下文长度、使用量化模型、增加显存 |
| TTFT 过高 | 用户等待首 token 时间长 | 检查模型大小、上下文长度、是否启用了前缀缓存、前缀命中率 |
| TPOT 过高 | 生成速度慢，用户体验差 | 检查显存带宽是否饱和、是否开启了 KV Cache 量化、是否开启了多卡 TP 导致通信开销 |
| P99 延迟飙升 | 高并发下少数请求极慢 | 检查排队模型、是否有单个长请求阻塞队列、是否启用了连续批处理 |
| 错误率高 | 请求返回错误或空响应 | 检查日志、模型版本是否匹配、提示词模板是否正确 |

> 面试要点：推理系统常见故障包括模型加载失败、显存溢出、TTFT 过高、TPOT 过高、P99 延迟飙升、错误率高。排查遵循：查看日志 → 检查显存和带宽 → 查看排队和并发 → 验证模型和配置 → 压测复现。

---

## Q28. 推理集群的容量规划怎么做？

**回答**

容量规划需要回答：已知模型、请求量、延迟要求，需要多少卡、多少机器？

**核心计算公式：**

1. **单卡显存需求：**
   ```
   显存 = 权重大小 + KV Cache(并发请求数 × 序列长度) + 激活值 + 系统开销
   ```

2. **单卡并发能力：**
   ```
   单卡最大并发 = (可用显存 - 权重显存 - 系统预留) / (每请求 KV Cache 占用)
   ```

3. **所需卡数：**
   ```
   所需卡数 = 总并发请求数 / 单卡最大并发
   ```

4. **所需机器数：**
   ```
   所需机器数 = 所需卡数 / 单卡 GPU 数
   ```

**Python 示例（容量规划计算器）：**

```python
def plan_inference_cluster(
    num_params_b: float,      # 参数量 (B)
    seq_len: int,             # 序列长度
    batch_size: int,          # 目标并发批次
    weight_bytes: float = 2,  # FP16=2, INT8=1, INT4=0.5
    kv_bytes_per_token: float = 2 * 8192 * 2 / 1024**3,  # 粗略估算
    gpu_mem_gib: float = 80,  # 单卡显存 (如 H100 80GB)
    system_overhead: float = 0.1,  # 10% 系统开销
    gpus_per_node: int = 8   # 单卡 GPU 数
) -> dict:
    # 1. 权重显存
    weight_gib = num_params_b * 10**9 * weight_bytes / (1024**3)
    
    # 2. KV Cache 显存（粗略估算）
    kv_per_req_gib = (2 * 80 * 128 * seq_len * kv_bytes) / (1024**3)
    
    # 3. 单卡可容纳的并发请求数
    available_gib = gpu_mem_gib * (1 - system_overhead) - weight_gib
    max_concurrency = int(available_gib / kv_per_req_gib)
    
    # 4. 所需卡数
    num_gpus = int(batch_size / max_concurrency) + 1
    num_nodes = int(num_gpus / gpus_per_node) + 1
    
    return {
        "weight_gib": round(weight_gib, 2),
        "kv_per_req_gib": round(kv_per_req_gib, 4),
        "max_concurrency_per_gpu": max_concurrency,
        "num_gpus": num_gpus,
        "num_nodes": num_nodes
    }

# 示例：70B 模型，FP16，序列长度 4096，目标并发 64，单卡 80GB，每卡 8 张
plan = plan_inference_cluster(
    num_params_b=70,
    seq_len=4096,
    batch_size=64,
    weight_bytes=2,
    gpu_mem_gib=80,
    gpus_per_node=8
)
print(plan)
```

> 面试要点：容量规划需要算清权重显存、KV Cache 显存、激活值显存，算出单卡能承载多少并发请求，再据此计算需要多少卡、多少机器。要留有系统开销余量（通常 10%），不要压到极限。

---

## Q29. 推理系统的写在最后：生产落地的关键建议

**回答**

把 LLM 推理系统从“跑通”到“好用”，需要注意以下几点：

1. **先算 Roofline 上限，再定目标**：不要盲目追求最高吞吐，先算出模型在目标硬件上的理论性能天花板，设定合理的 SLA 目标。
2. **先测单机，再上集群**：单机性能不达标，多卡堆叠也救不了。先在单机上把基础性能（TTFT、TPOT、显存占用）调优到位，再扩展到集群。
3. **KV Cache 是显存大头**：对于长对话和大批次，KV Cache 往往比模型权重占用更多显存，必须做 PagedAttention、前缀缓存、量化来控制。
4. **前缀亲和和连续批处理是必备**：多轮对话场景下，不做前缀缓存会浪费大量重复计算；不做连续批处理会导致短请求被长请求阻塞。
5. **监控和告警是底线**：生产环境必须监控 TTFT、TPOT、P99 延迟、显存占用、错误率，并设置告警，否则故障发生时无法快速定位。
6. **端云混合是实用选择**：不是所有请求都需要云端大模型，简单请求走端侧可以节省成本、降低延迟、保护隐私。

一句话总结：**推理系统落地的关键不是“跑通”，而是“有据可依”：有模型七元组、有 Roofline 上限、有真机 Profile 数据、有容量规划计算、有生产环境监控，才能做出稳定、可扩展、成本可控的推理服务。**

> 面试要点：本章小结＝先算 Roofline 上限 → 单机先跑通 → KV Cache 是大头（PagedAttention、前缀缓存）→ 前缀亲和和连续批处理是必备 → 监控和告警是底线 → 端云混合是实用选择。推理系统落地要有数据支撑，不要拍脑袋。
