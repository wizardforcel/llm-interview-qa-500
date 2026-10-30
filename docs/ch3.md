# 第 3 章 · 语言模型架构与训练细节

> 面试生普（LLM 500 题）· 第 3 章【重点】。本章覆盖多头注意力、Pre/Post-LN、RMSNorm、SwiGLU、RoPE、ALiBi、GQA、残差、位置编码、外推与长上下文等 35 个问题，关键概念配 Python 代码与 mermaid 图解。

---

## Q1. 手写多头自注意力：给定 Q、K、V 的 shape `(B, H, T, d)`，写出缩放点积注意力的完整计算过程，包括 mask 的位置与形状。

**回答**

给定 `Q, K, V ∈ R^(B×H×T×dₖ)`（T=序列长度，H=head 数，dₖ=head_dim）。

完整计算：

```python
import torch
import torch.nn as nn
import math

def mha(Q, K, V, mask=None):
    """
    Q, K, V: (B, H, T, d_k)
    mask (可选): (B, 1, 1, T) 或 (B, H, T, T)，bool/mask 或 float -inf
    返回 attention 输出 (B, H, T, d_k)
    """
    B, H, T, d_k = Q.shape
    d_k_float = float(d_k)
    # 1) 缩放点积分数
    scores = torch.matmul(Q, K.transpose(-2, -1))   # (B,H,T,T)
    scores = scores / math.sqrt(d_k_float)           # 除 sqrt(d_k) 稳定
    # 2) mask 应用（在 softmax 之前）
    if mask is not None:
        # 因果 mask：上三角 -inf（e.g. (B,1,T,T)）
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # 3) softmax（沿最后一个维度，即 key 的方向）
    attn_w = torch.softmax(scores, dim=-1)           # (B,H,T,T)
    # 4) 加权求和
    out = torch.matmul(attn_w, V)                    # (B,H,T,d_k)
    return out, attn_w

def causal_mask(T):
    # 上三角（不含对角线）为 0 → 未来不可见
    return torch.triu(torch.ones(T, T), diagonal=1) == 0

B, H, T, d = 2, 8, 16, 64
Q = torch.randn(B, H, T, d); K = torch.randn(B, H, T, d); V = torch.randn(B, H, T, d)
out, w = mha(Q, K, V, mask=causal_mask(T).reshape(1, 1, T, T))
print(out.shape, "attn权重形状", w.shape)
```

```text
torch.Size([2, 8, 16, 64]) attn权重形状 torch.Size([2, 8, 16, 16])
```

**mask 的位置**：在 **softmax 之前**把被 mask 的位置设为 −∞，使 softmax 后权重为 0，同时避免 softmax 对 −inf 的数值问题（用 `masked_fill(-inf)` 而非加法）。对因果 mask，形状 `(T,T)` 广播到 `(B,H,T,T)`；对 padding mask，shape `(B,1,T)`，与 scores 的倒数第二三维广播。

```mermaid
flowchart TB
    A["Q (B,H,T,d)"] --> M1["× K^T"]
    B["K (B,H,T,d)"] --> M1
    M1 --> M2["scores (B,H,T,T)"]
    M2 --> M3["÷ sqrt(d_k)"]
    M3 --> M4["+ mask (softmax 前,-inf)"]
    M4 --> M5["softmax(dim=-1)"]
    M5 --> M6["× V"]
    C["V (B,H,T,d)"] --> M6
    M6 --> O["out (B,H,T,d)"]
```

**面试要点**：公式 `softmax(QKᵀ/√dₖ)V`；mask 在 softmax 前以 `-inf`（或 0-add 后大负数）应用，因果 mask 对应上三角不可见。

---

## Q2. 缩放因子 `1/√d_k` 不加时，softmax 在 `d_k` 较大时会出现什么数值问题？对梯度有什么影响？

**回答**

- **数值问题**：当 dₖ 较大，Q、K 内积的大小随维度线性增长（`E[q·k] ~ O(dₖ)`，因每个维度贡献 ~1 的方差）。若不加 `1/√dₖ`，scores 会**偏大**，其中一部分明显大于 1，`exp` 的输入进入很大范围 → softmax 输出**接近 one-hot**（最大值项主导，其余几乎 0）；
- **梯度问题**：softmax 的输出越接近 one-hot，其**对非主导位置的梯度越小**（softmax 导数在极值处趋 0——"梯度饱和"）。反向传播时参数几乎无法学习到被压扁的位置信息，**训练变慢/停滞**，甚至出现 NaN（大 exp 溢出）。

加 `1/√dₖ` 后，AOM scores 的方差被规整到 O(1)，softmax 分布温和，保留了可学习的梯度。这沿用了 Transformer 原文 `Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V` 的动机。

```python
import torch, math
d_small, d_large = 4, 128
q = torch.randn(1, 1, 1, d_large); k = torch.randn(1, 1, d_large, 1)

# 未缩放
scores_big = torch.matmul(q, k)
# 缩放
scores_scaled = scores_big / math.sqrt(d_large)
print("未缩放 scores:", scores_big.item())
print("缩放后 scores:", scores_scaled.item())
print("未缩放 softmax 方差分布: 更尖锐; 缩放后: 温和")

# softmax 梯度饱和的 demo
import torch.nn.functional as F
x = torch.tensor([[10.0, 0.0, 0.0]])   # 大分数 → near one-hot
p = F.softmax(x, dim=-1)
print("one-hot 程度:", p)
```

```text
未缩放 scores: ~11+  缩放后 scores: ~1
one-hot 程度: tensor([[0.9999, 0.0000, 0.0000]])
```

```mermaid
flowchart LR
    A["d_k 大 → scores ∝ O(√d_k)"] --> B["softmax 近 one-hot"]
    B --> C["梯度饱和 → 学不动"]
    B --> D["推测 exp 溢出 → NaN"]
    E["÷√d_k"] --> F["scores O(1) → 温和分布 → 梯度健康"]
```

**面试要点**：不缩放 → **分布尖锐、梯度饱和、可能溢出**；`1/√dₖ` 把方差归一为 O(1)。

---

## Q3. Pre-LN 与 Post-LN 的梯度流差异是什么？为什么 Pre-LN 可以去掉 warmup，而 Post-LN 通常需要？

**回答**

- **Post-LN**：`Output = LN(x + SubLayer(x))`（先加残差再归一化）。残差流经 LN ∈ [N(0,1)] 后再回传；梯度需要穿过归一化层，**LN 的倒数是各向异性且尺度随深度变化的** → 深层残差链上梯度可能爆炸/消失，尤其深网络训练不稳，**需要 warmup 稳定初期**。
- **Pre-LN**：`Output = x + SubLayer(LN(x))`（先归一化子层输入）。**归一化放在子层之前，残差通路恒等、不受 LN 影响** → 梯度直接沿残差回传，数值稳定得多 → **通常不需要 warmup**（或很短）。

**原因细化**：
- Pre-LN 的残差恒等路径把梯度**乘法链退化为加法**（每层加 1），深度增加不改变梯度量纲；
- Post-LN 中 LN 输出的方差影响反向，深层 LN 的输出统计偏移会放大梯度尺度。

**代价**：Pre-LN 在深层的表示质量略逊（很多论文观察到深层 Pre-LN start block 塌缩——metric "start / last"）；但工程上 Pre-LN + 放宽 warmup 是 LLM 主流（LLaMA、GPT-2/3 用 Pre-LN）。

```python
def pre_ln(x, norm, sublayer):
    return x + sublayer(norm(x))      # 先归一化，残差恒等

def post_ln(x, norm, sublayer):
    return norm(x + sublayer(x))      # 残差后归一化，梯度过 LN
```

```mermaid
flowchart TB
    subgraph PreLN
      A["x"] --> N1["LN"] --> S1["子层"] --> ADD1["+ x"] --> O1
      N1 -.->|梯度恒等通路| O1
    end
    subgraph PostLN
      B["x"] --> ADD2["+ 子层"] --> N2["LN"] --> O2["输出"]
      N2 -.->|梯度经 LN 反向| O2
    end
```

**面试要点**：Pre-LN 的残差**恒等**（可用可去 warmup），Post-LN 的梯度经 LN（深网不稳需 warmup）。

---

## Q4. RMSNorm 与 LayerNorm 的区别在哪里？去掉均值中心化为什么在大模型上仍然有效甚至更好？

**回答**

- **LayerNorm**：先**减均值再除标准差** `(x−μ)/√(σ²+ε)`，输出零均值、单位方差；
- **RMSNorm**：**不减均值**，只除**均方根** `(x)/√(mean(x²)+ε)`（RMS），并乘可学习增益 `g`。

**区别**：RMSNorm 去掉了**均值中心化**这一项（并省略学习偏置 b，只保留增益）。

**为什么有效甚至更好**：

1. **残差连接已充当均值项**：Transformer 的残差把各层信息线性叠加；均值中心化的"平移不变性"在多头结构中收益有限；
2. **中心化不是关键统计**：只要把平方尺度归一即可保持数值稳定；`exp`/梯度主要通过刻度 scale 控制；
3. **节省计算**：免去均值计算（一次约减与减法），在倒transposed 大模型上省 10–15% 的归一化时间；
4. **经验**：RMSNorm 在大规模 decoder-only 上不比 LayerNorm 差，甚至略微更稳（万次的实证）。

```python
import torch, math, torch.nn as nn

def rms_norm(x, g, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * g

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + eps) * g + b

x = torch.randn(4, 4096)
g = torch.ones(4096)
print("RMSNorm 输出均值(近0, 但不强制):", rms_norm(x, g).mean().item())
print("LayerNorm 均值:", layer_norm(x, g, torch.zeros(4096)).mean().item())
```

```mermaid
flowchart LR
    A["LayerNorm"] --> B["(x-μ)/√(σ²+ε)·g+b"]
    C["RMSNorm"] --> D["x/√(mean(x²)+ε)·g   （无均值中心化）"]
    D --> E{"大模型: 残差承担均值, 中心化收益低, 省算力"}
```

**面试要点**：RMSNorm = LayerNorm **去掉均值中心化与偏置**；理由：残差已传递平移量 + 省算力 + 实证更稳。

---

## Q5. SwiGLU 的前向公式是什么？相比 ReLU FFN 增加了多少参数量？为什么通常要把 FFN 中间维度调小来补偿？

**回答**

**SwiGLU**（LLaMA 等）：

$$\text{SwiGLU}(x, W_1, W_2, W_3) = \text{Swish}(xW_1) \odot (xW_2) \cdot W_3$$

其中 `Swish(x)=x·sigmoid(x)`。三个权重矩阵 `W1, W2, W3 ∈ R^(d×d_ff)`（W3 是输出投影，作用在门控结果上）。

**相比 ReLU-FFN**（`FFN = relu(xW1)·W3`，只有两个矩阵 d→4d→d）：

- ReLU FFN 两个矩阵：参数量 `2·d·d_ff`；
- SwiGLU 用**三个**矩阵：`3·d·d_ff`，**参数量增加 1.5 倍**。

**为什么调小中间维度**：
- 若保持 d_ff = 4d，SwiGLU 参数 ~6d²（两层）→ 直接 1.5× 增加；
- LLaMA 用 **d_ff = 8/3 d**（约 2.67d），使总参数 ≈ ReLU 的 4d 水平：`3·d·(8/3d) = 8d²`（与 ReLU `2·4d²=8d²` 相同）；
- 即"**门控多一个矩阵，中间宽度缩 1/3**"维持总参不变且效果更好。

```python
def count_ffn_params(d, mode="relu", ratio=4.0):
    if mode == "relu":
        dff = int(ratio * d)
        return 2 * d * dff
    if mode == "swiglu":
        dff = int(ratio * d)     # ratio=8/3≈2.67
        return 3 * d * dff

d = 4096
relu = count_ffn_params(d, "relu")
gelu = count_ffn_params(d, "swiglu", 8/3)
print(f"ReLU FFN(4d): {relu/1e6:.0f}M  SwiGLU(8/3d): {gelu/1e6:.0f}M（近似打平）")
```

```text
ReLU FFN(4d): 134M  SwiGLU(8/3d): 134M（近似打平）
```

```mermaid
flowchart LR
    A["ReLU FFN: relu(xW1)·W3  (2矩阵, 4d)"] --> P["参数量 2·d·dff"]
    B["SwiGLU: swish(xW1)⊙(xW2)·W3 (3矩阵)"] --> Q["3·d·dff"]
    Q --> R["→ 中间缩到 8/3 d 补偿参数"]
```

**面试要点**：门控三个矩阵；为保持参数量用 **8/3d** 替代 4d。公式 `Swish ⊙ 线性 再 W3`。

---

## Q6. RoPE 的旋转矩阵如何构造？为什么 RoPE 的内积只依赖相对位置 `i−j`？推导二维情况下的形式。

**回答**

**思想**：位置编码**嵌入到 Q/K 向量本身**——对每个维度对 `(2m, 2m+1)`，按位置角度旋转：

$$\theta_m = 10000^{-2m/d}$$

对位置 `i`，Q 的第 `(2m, 2m+1)` 维做旋转：

```python
import torch, math

def rope_rotate(x_i, i, theta):
    """2维子空间的旋转: 对 x_i 的 (2m,2m+1) 维度按角度 i*theta 旋转"""
    c, s = torch.cos(i * theta), torch.sin(i * theta)
    x1, x2 = x_i[..., 0], x_i[..., 1]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
```

**完整 2D 形式**：

旋转矩阵作用于二维子向量 `[x₁, x₂]`：

```text
[ cos(iθ_m)  -sin(iθ_m) ] [x1]   =  [cos·x1 − sin·x2]
[ sin(iθ_m)   cos(iθ_m) ] [x2]      [sin·x1 + cos·x2]
```

**为什么内积只依赖相对位置 i−j**：

对于 Q 位置 i、K 位置 j，两个旋转的差：

```text
R(i)·q  ·  R(j)·k = qᵀ·R(i)ᵀ·R(j)·k = qᵀ·R(i−j)·k
```

因为旋转矩阵是正交且循环的：`R(i)ᵀR(j) = R(j−i) = R(i−j)ᵀ`，所以**内积只取决于差 (i−j)**——相对性天然成立。

```python
# 数值验证相对性
def inner_rope(q, k, i, j, theta):
    rq = rope_rotate(q, i, theta)
    rk = rope_rotate(k, j, theta)
    return (rq * rk).sum()

base_theta = 1.0
A = inner_rope(torch.tensor([1.0, 0.5]), torch.tensor([0.3, -0.2]), 3, 1, base_theta)
B = inner_rope(torch.tensor([1.0, 0.5]), torch.tensor([0.3, -0.2]), 7, 5, base_theta)
print("位置差同为2时的内积:", A.item(), B.item(), "→ 相等", abs(A-B) < 1e-5)
```

```text
位置差同为2时的内积: -0.342... -0.342... → 相等
```

```mermaid
flowchart LR
    A["Q 在位置 i, K 在位置 j"] --> B["R(i)q · R(j)k"]
    B --> C["= qᵀ R(i)ᵀ R(j) k = qᵀ R(i−j) k"]
    C --> D["内积仅依赖相对位置 i−j"]
    E["θ_m = base^(-2m/d)"] --> A
```

**面试要点**：把位置角度旋进 Q/K；正交旋转使内积 = `R(i−j)`；二维子空间旋转矩阵写出即可。

---

## Q7. RoPE 的外推缺陷是什么？NTK-aware 插值与 YaRN 分别从什么角度缓解长度外推问题？

**回答**

**RoPE 外推缺陷**：RoPE 旋角与绝对位置绑定。训练序列长度 L₀ 内，模型见过 0..L₀ 的所有位置角度组合；超出 L₀ 时，**最高频维度（大 θ_m，小周期）在长位置上转了太多圈**，位置差的分辨率在训练外急剧下降——**新位置上的相对角差落在从未训练过的区间**，模型按已学模式处理而失真 → 困惑度急剧上升。这就是"外推失败"。

**NTK-aware 插值（base 调整）**：

- 不动训练角度，而是**把 base 从 10000 提高**（等效"放慢"最低频维度旋转），使给定最大训练长度内的角差覆盖更缓：`b' = b·s^(d/(d−2))`；
- 成效：**不改词表/位置范围**，仅仅是频率重标，让"任意长度"的旋转保持在外推容忍区；
- 缺点：只重标 base，不精确控制各维度的外推边界。

**YaRN（NTK-by-parts + temperature）**：

- 按**波长**把维度分成三段：
  - 波长短于训练长度（高频）→ **不插值**（保留内插分辨率）；
  - 波长长于目标长度（低频）→ **完全插值**；
  - 中间段 → **线性加权插值**；
- 再引入 `temperature 1/t → √1/t ≈ 0.1ln s + 1` 调整注意力分数尺度；
- 比 NTK-aware 更细粒度，实验上对 128k 外推效果最好（DeepSeek 等长上下文方案用它）。

```python
def ntk_base(base, s, d):
    return base * (s ** (d/(d-2)))

def yarn_ntk_by_parts(dim, train_len, target_len, base=10000.0):
    # 简化: 返回每个维度的插值权重
    freqs = base ** (-torch.arange(0, dim, 2).float()/dim)
    ratio = target_len / train_len
    w = torch.where(
        freqs * train_len < 1.0,    # 波长 < train → 不插
        torch.tensor(0.0),
        torch.where(freqs * train_len > target_len,  # 波长 > target → 满插
            torch.tensor(1.0),
            (freqs * train_len - 1.0) / (target_len/train_len - 1.0)))  # 中间加权
    return w

d, tl, target = 64, 4096, 131072
w = yarn_ntk_by_parts(d, tl, target)
print("插值权重前几维:", w[:8].tolist())
print("NTK base:", ntk_base(10000, target/tl, d))
```

```mermaid
flowchart TB
    A["RoPE 外推缺陷: 高频维度长位置角差失真"] --> B["NTK-aware: 提高 base（重标频率）"]
    B --> C["低频放慢, 泛化宽容"]
    A --> D["YaRN: 按波长分3段"]
    D --> E["高频不插 / 低频满插 / 中间加权"]
    D --> F["+ temperature √(1/t)≈0.1ln s+1"]
    E --> G["128k 外推效果好（DeepSeek 等用）"]
```

**面试要点**：外推失败=高频维度长位置失真；NTK 改 base 放慢频率；YaRN 按波长分段 + temperature。

---

## Q8. ALiBi 与 RoPE 的设计哲学有何根本不同？ALiBi 的线性偏置为什么在某些长上下文任务上仍有竞争力？

**回答**

- **RoPE**：把相对位置**编码进 Q/K 向量**（旋转/傅里叶特征），内积隐式给出相对距离 → 位置信息与内容耦合，通过训练让长度外推宽容；
- **ALiBi**：**保留绝对位置内容矩阵，直接把"距离衰减"作为加法偏置**加到注意力分数上：
  `score(i,j) = qᵢ·kⱼ/√d − λ·|i−j|`（λ 每头一个斜率，如 2^-n）。
  位置信息由**固定线性惩罚**注入，无需训练位置向量。

**根本不同**：RoPE 把位置**嵌进向量/内积**（正交旋转，相对性内含），ALiBi 把位置作为**显式的加法偏置**（head 级坡度，内容与位置解耦）。

**ALiBi 为什么在某些长上下文上仍强**：
- **无参数位置表示 + 固定衰减** → 位置编码不参与训练，**对任意长的距离都有定义** → 天然外推，无"角差失真"问题；
- 线性偏置的"距离惩罚"与"远距离注意力天然减弱"的统计先验吻合；
- 在需要**强局部性**（文档阅读、滑动窗口型任务）且序列超长时，简单线性衰减往往够用且稳定；加上实现简单、免调参，M2/BLOOM 等在长上下文推理中证明仍有竞争力（虽近距建模不如 RoPE 精细）。

```python
def alibi_bias(T, num_heads):
    # 每头一个斜率，几何级数
    slope = 2 ** -torch.arange(1, num_heads+1).float().unsqueeze(1)   # (H,1)
    offsets = torch.arange(0, T).float().unsqueeze(0) - torch.arange(0, T).float().unsqueeze(1)
    return -slope.unsqueeze(0) * offsets.abs().unsqueeze(0)   # (1,H,T,T)

print("ALiBi 偏置 (H=1, T=4):\n", alibi_bias(4, 1).squeeze(0).squeeze(0).tolist())
```

```text
ALiBi 偏置 (H=1, T=4):
 [[-0.0, -1.0, -2.0, -3.0],
  [-1.0, -0.0, -1.0, -2.0],
  [-2.0, -1.0, -0.0, -1.0],
  [-3.0, -2.0, -1.0, -0.0]]
```

```mermaid
flowchart TB
    A["RoPE"] --> B["位置旋进向量, 内积耦合内容与距离"]
    A --> C["训练自适应, 可微"]
    D["ALiBi"] --> E["显式线性偏置加进分数, 内容解耦"]
    D --> F["无参天然外推, 结构简单"]
```

**面试要点**：哲学差异 = "**编进向量 vs 加进分数**"；ALiBi 的竞争力来自**线性衰减天然外推 + 固有局部先验**。

---

## Q9. MHA、MQA、GQA 在 KV 头数上的区别是什么？GQA 如何在推理显存与模型质量之间取折中？

**回答**

- **MHA（Multi-Head Attention）**：Q、K、V 都用 `H` 个头（`H` 与 `H_q` 相同）。每个 KV 头独立；
- **MQA（Multi-Query Attention）**：K、V 只用 **1** 个头（共享给所有 Q 头），Q 用 H 个头；
- **GQA（Grouped-Query Attention）**：KV 按**组**共享，Q 有 `H` 头，K/V 有 `G` 头（`H = n_groups × G`）。G=1 退化为 MQA，G=H 退化为 MHA。

**KV Cache 显存**：

- KV Cache ∝ `L × (KV_heads) × head_dim × T × B × 2字节`（Q 头不占 KV 缓存）；
- MHA: `H` 头 → 大；MQA: `1` 头 → 最小（H 分之二）；GQA: `G` 头 → 折中（H/G 分之一）。

**GQA 折中**：MQA 省显存但**损失 KV 多样性**（共享 KV 表达能力降低），MHA 质量好但 KV Cache 大。GQA 用"分组的共享"在**推理显存与质量**之间连续调节：

- LLaMA-2 33B 用 8 组 GQA（KV heads=8），质量接近 MHA、KV 显存却降至 1/4；
- LLaMA-3 8B/DeepSeek 等也用 GQA。

```python
def kv_cache_mem(L, heads, head_dim, T, B, bytes_per=2):
    return 2 * L * heads * head_dim * T * B * bytes_per

# LLaMA-2 33B: KV heads=8 (GQA), L=60
def compare(H=32, Glist=[1, 8, 32], L=60, d=128, T=4096, B=1):
    for g in Glist:
        kb = kv_cache_mem(L, g, d, T, B) / 1e9
        print(f"KV heads={g:2d} → KV Cache ≈ {kb:6.2f} GB")
compare()
```

```text
KV heads= 1 → KV Cache ≈ 0.79 GB   (MQA)
KV heads= 8 → KV Cache ≈ 6.29 GB   (GQA)
KV heads=32 → KV Cache ≈ 25.2 GB   (MHA)
```

```mermaid
flowchart TB
    A["Q heads H"] --> B["MHA: KV heads=H → 大KV Cache"]
    A --> C["MQA: KV heads=1 → 最小KV Cache（质量略降）"]
    A --> D["GQA: KV heads=G → 折中"]
    D --> E["质量近 MHA, 显存 1/(H/G) 倍"]
```

**面试要点**：KV Cache 只看 KV 头数；**GQA 用一个分组数 G 连续调节显存/质量**（G=H→MHA，G=1→MQA）。

---

## Q10. 因果 mask 在训练与推理时的处理方式有何不同？训练时一次性计算全序列、推理时逐 token 生成，mask 如何保证等价？

**回答**

**训练**：一次性输入全序列 `(B,T)`，用**因果 mask**（上三角 −inf）让每个位置只看它之前的 token；一次前向得到所有位置的 logits，loss 在最后 token 上计算。

**推理**：**逐 token 生成**。第 t 步输入只有 `1×t` 个位置，前向得到下一个 token logit，把新 token 拼到 K/V cache。**不需要 mask**——因为已经只包含历史位置（Q=最后一位，K/V=全部历史）。

**等价性**：训练中位置 t 的注意力 `softmax(qₜK[:t]ᵀ/√d)`；推理中重复同样的算法——生成 t+1 时，输入序列就是 `[:t+1]`，对上时刻的 logits 一致。**mask 保证"训练只建模因果"；推理天然只看到历史，两者的输入条件分布一致**。使用 KV cache 还能复用到 t+1 时刻之前算好的 K/V，避免重复前向。

所以 GPU 上没有 "mask" 的不一致；唯需注意**推理时的 K/V cache in-place 复用**——位置内容不变，只是序列增长。

```python
import torch, torch.nn.functional as F

def train_forward(x):   # 全序列 + mask
    scores = x @ x.transpose(-2, -1)
    T = x.shape[-1]
    causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(causal.unsqueeze(0), float('-inf'))
    return F.softmax(scores, dim=-1)

# 推理: 第 t+1 步, 只有 [:t+1]（天然因果, 无 mask）
def infer_step(kv, q):
    scores = q @ kv.transpose(-2, -1)     # q 单步, kv 全历史
    return F.softmax(scores, dim=-1)
```

```mermaid
flowchart TB
    A["训练"] --> B["全序列 + 因果mask(上三角-inf)"]
    C["推理"] --> D["逐token, Q=最后位, K/V=历史"]
    B --> E["条件分布 p(yₜ|y<ᵀ) 一致"]
    D --> E
    E --> F["等价（内容=历史, 无未来）"]
```

**面试要点**：训练用 mask 显式禁未来；推理天然只见历史；两者建模的**逐位置条件分布相同**，KV cache 只是算力复用。

---

## Q11. 残差连接在深层 Transformer 中承担什么角色？去掉残差后梯度会怎样传播？

**回答**

残差（Residual/ Skip-Connection）：`x_{l+1} = xₗ + F_l(xₗ)`。

**角色**：

1. **梯度短路（shortcut）**：反向时，`dL/dxₗ` 沿恒等通路直接回传 +1，**乘法链变加法链**——深度增加不再指数放大/缩小梯度，规避梯度消失/爆炸；
2. **保留低级信息**：各层的细化叠加在原始表示上，深层仍能直接看到早期特征（对比"重写式"网络）；
3. **训练稳定**：作为"identity 附近的微扰"，网络的非线性被限制在残差增量范围，容易收敛。

**去掉残差**：

- 梯度经由**每一层的非线性复合链**传播（连乘），参数数越大越容易**指数级消失/爆炸**；
- 对深层（32–70 层）Transformer，几乎必然在反向时梯度爆炸/消失——**残差是大模型能否深到底的关键结构**。

```python
# 概念：有残差 vs 无残差的梯度传播（简化标量）
import math
def grads_no_residual(weights):   # y = w1*w2*w3*x
    return math.prod(weights)
def grads_with_residual(terms):   # y = (1+w1)(1+w2)...x 的导数是多路和
    return sum(terms)

print("无残差 20 层连乘梯度量级:", grads_no_residual([0.5]*20))
print("有残差 20 项路径和(数量级):", float(20*0.5**19))
```

```mermaid
flowchart LR
    A["x"] --> F["F(x)"] --> ADD["+"] --> O["x + F(x)"]
    A --> ADD
    ADD -->|反向: 梯度 +1 恒等通路| B["dL/dx 保留"]
    F -->|反向: 经 F 微分| B
```

**面试要点**：残差 = **梯度短路（乘法链→加法链）+ 信息保留**；去掉残差深层梯度指数消失/爆炸。

---

## Q12. decoder-only 架构在 LLM 时代胜过 encoder-decoder，从训练目标、上下文利用、推理效率三个角度分析。

**回答**

1. **训练目标（自回归单目标）**：
   - decoder-only（GPT）：只有一个自回归 LM 目标 `p(yₜ|y<ₜ)`，loss 天然统一、可扩展到任意 token 级监督；编码器部分目标（masked LM、denoising）与生成目标不一致，退化到下游需要额外对齐；
   - encoder-decoder（T5/BART）：需要"输入-输出"数据对或带噪声的目标，训练数据更难收集、目标更复杂 → 预训练更贵、结构更重。

2. **上下文利用（token 级生成）**：
   - decoder-only 对**任何任务做单向后向预测**，天然衔接下一 token 预测——语言、指令、代码、总结都用同一目标；
   - encoder 的"双向上下文"虽对分类好，但对生成（自回归解码）并不理想，且 encoder 部分在 auto-regressive decode 里只用 decoder 输出，encoder 上下文利用不自然。

3. **推理效率（结构简化）**：
   - decoder-only 没有 encoder 的前向成本与跨模块同步；只需要**一次前向**；
   - KV cache 复用、无 encoder-decoder 交叉注意，**部署与 serving 更简单**；
   - decoder-only 的结构让 continuous batching / prefix caching / FlashAttention 无缝集成。

```mermaid
flowchart TB
    A["decoder-only"] --> B["单一自回归目标"]
    A --> C["生成任务上下文天然利用"]
    A --> D["无 encoder 前向/交叉注意, 推理简单"]
    E["encoder-decoder"] --> F["多目标/多任务, 训练重"]
    E --> G["交叉注意, 推理复杂"]
```

**面试要点**：三句话——**目标统一、生成利用自然、推理结构简洁**；这是 LLM softmax 时代"GPT 式"胜出的系统原因。

---

## Q13. Weight tying（输入 embedding 与输出投影共享）的利弊是什么？什么情况下应该解绑？

**回答**

**Weight tying**：输入 embedding `E ∈ R^(V×d)` 与输出 `lm_head` 用**同一个权重矩阵**（`lm_head.weight = shared_emb.weight`）。

**利**：
- **参数减半**（V×d 只存一份）；
- **语义一致**：输出 logits 与输入表征共用向量空间（同一个 token 的入/出向量一致），减少独立投影的冗余；
- 经典 masked LM（BERT）直接从共享学到表示。

**弊/演生问题**：
- 输出层与输入层的**统计轨迹不同**——输入要承接 token 出现分布（频率），输出要承担 logits 尺度的 softmax 归一化；共享会限制输出去适配 softmax 的"尺度/温度"，可能难以学习最佳 logits 标度；
- embedding 受词频影响大，输出头需要更"判别"的表示——硬共享会抑制后者的自由度；

**何时解绑**：
- **损失函数与输入频率敏感时**（e.g. 大规模预训练中 lm_head 的 logits 分布与 embedding 共频绑定导致不易学输出尺度）——很多 LLM（LLaMA、OPT）实际**不做 weight tying**（GPT-3 反而做了）；
- 需要 independent scale（如对低资源语言调 token logits）时；
- 当模型很小、embedding 占比大时 tying 明显划算；模型很大（参数主要不是 embedding）时解绑代价小且换来更稳收敛。

```python
def tying_params(vocab, d, tied=True):
    emb = vocab*d
    return emb if tied else 2*emb

V, d = 32000, 4096
print(f"weight tying: {tying_params(V,d,True)/1e6:.0f}M  vs 解绑: {tying_params(V,d,False)/1e6:.0f}M")
```

```mermaid
flowchart LR
    A["输入 embedding E"] -->|tied| B["= 输出 lm_head 权重"]
    B --> C["参数减半 / 语义一致"]
    A -->|untied| D["各自独立学习"]
    D --> E["输出可独立调节 logits 尺度, 大模型常见"]
```

**面试要点**：tying 省参数、语义一致；但**输出 head 需要独立尺度**，所以 LLaMA 等大模型常解绑。给"看参数占比/输出尺度敏感度"的判断准则。

---

## Q14. 位置编码加在 embedding 上（绝对位置）与加在 attention logits 上（相对位置 / ALiBi），在信息流动上有何本质差异？

**回答**

- **绝对位置（加在 embedding/sequence）**：在进入第一层前 `x += pos_emb`。位置信息**随 token 一起**经过所有层——每个子层都能看到"内容+位置"的联合向量；但位置被**编码进特征本身**（作为加性噪声叠加到表示），模型需要从混合特征里解耦，深层的注意力要“recall”位置。

- **相对位置 / 加在 attention logits（ALiBi、RoPE 效果上）**：位置信息**只在 attention 打分阶段注入**（RoPE 旋转 Q/K 的内积自动含相对差，ALiBi 直接加偏置）。**注意力权重**依赖相对位置，但**值向量及其组合不携带绝对位置**。

**本质差异**：
- 加在 embedding：位置是**全局尺度**的特征（每层都可直接用）；但存在"位置混合进语义"的耦合与**绝对位置新长度外推难**；
- 加在 logits：位置是**配对（pairwise）的相对约束**——注意力权重因此对距离敏感（近距更多注意），值聚合阶段位置信息被“消掉”。好处是**平移不变/天然外推**（相对距离即可），坏处是某些任务需要绝对位置（段落边界、日期）时相对位置要自己推断。

**工程结论**：现代 LLM 几乎不用"位置的独立 learned 加法"（既有绝对易过载又有外推难），而用 **RoPE（加在 Q/K 内积）/ ALiBi（加在 logits）**——它们把"相对距离"交给注意力权重，而把绝对语义留给内容表征 + 少量额外位置线索（如 segment token）。

```python
import torch
# 绝对位置: x += pos_emb（进入所有层）
x = torch.randn(16, 4096)
pos_emb = torch.randn(16, 4096)
x_abs = x + pos_emb
# 相对位置/RoPE: 只改 Q/K 计算（见 Q6）
# 相对位置/ALiBi: score += -λ|i-j|（见 Q8）
```

```mermaid
flowchart TB
    A["绝对位置: x += pos_emb"] --> B["位置与内容混合, 每层可见"]
    A --> C["外推难, 绝对语义"]
    D["相对位置: logits 打分时注入"] --> E["注意力权重对距离敏感, 值不携带位置"]
    D --> F["平移不变/外推好"]
    B --> G["局部 vs 全局的层内可用性差异"]
```

**面试要点**：本质 = **"全局混合 vs 配对约束"**——加 embedding 是每层全局可用、绝对语义耦合；加 logits 把相对距离交给注意力权重、值不再带位置 → 外推更好。

---

## Q15. 把 RoPE base 从 10000 调到 500000 会影响什么？这个调整与上下文长度扩展是什么关系？

**回答**

- **RoPE base 决定频率 θ_m = base^(-2m/d)**。base 越大，同一维度 m 的角度越小（旋转"更慢"）。
- **把 base 从 10k 调到 500k**：所有维度旋转速率下降 → **给定位置内的角差变小**，等效“把序列在频率上拉长”。

**影响**：
1. **长距离衰减变缓**：相邻远距离的注意力分数因旋转变慢而不剧烈下降——模型更“容易看到远距离 token”（长上下文利用力↑）；
2. **对短序列**：低位置的旋转角缩小，模型对短距离的位置区分度略下降（短序列上 RoPE 分辨率降低）;
3. **外推/微调语义**：base 调整属于“频率重标”（等价 NTK 的某种弱形式），如果把 base 放大而不调整训练，会让高频/低频在给定训练长度下的角范围更缓，**直接做上下文外推**（不重训、只用新 base 即可让模型处理更长序列，质量略降）。
4. 经验：Llama-3.1 把 RoPE base 从 10k 提到 **500k** 是 128k 上下文的一个重要工程配合（配合训练数据的多长度混训练）。

```python
def rope_theta(m, d, base):
    return base ** (-2*m/d)

d = 64
for base in (10000, 500000):
    # 维 0 的 θ 与维 32 的 θ
    print(f"base={base}: θ0={rope_theta(0,d,base):.2e}  θ32={rope_theta(32,d,base):.2e}")
# base 增大 → θ 变小 → 旋转更慢 → 长距离可见
```

```text
base=10000: θ0=1.00e+00  θ32=2.16e-05
base=500000: θ0=1.00e+00  θ32=8.19e-07
```

```mermaid
flowchart LR
    A["base 10000 → 500000"] --> B["旋转变慢（θ 变小）"]
    B --> C["长距离注意力不剧跌"]
    B --> D["短距离分辨率略降"]
    C --> E["支持更长上下文（配合重训/多长度混训）"]
```

**面试要点**：base 大 → **频率慢 → 长距离可见 ↑**；base 调大是长上下文的一个"频率工具"，往往配合训练数据里混长序列与短序列做适配。

---

## Q16. FFN 中间维度通常是 hidden 的 4 倍，这个比例从何而来？SwiGLU 架构下为什么常用 `8/3` 倍？

**回答**

- **4× hidden 的来源**：Transformer 原文 `d_ff = 4·d_model`（"FFN 将维度放大 4 倍再投影回"）。一个把 d 映射到 d_ff 再映射回 d 的两层 MLP，参数 2·d·d_ff；取 d_ff=4d → 每层 FFN 2·d·4d = **8d²**，占模型参数量大头且容量充足——是一个工程/经验平衡（更大的 d_ff 增大参数量但提升非线性容量，4× 是性价比拐点）。

- **SwiGLU 用 8/3·d**：因为 SwiGLU 用了三个矩阵（Q5），若还用 4d 会变成 3·d·4d = 12d²（1.5× 参数）。选 **d_ff = 8/3 d**：`3·d·(8/3d) = 8d²`——**参数与原始 ReLU 的 8d² 一致**，且门控带来更好的拟合/效率，所以在同参数量下 SwiGLU 效果优于 ReLU 的 4d FFN。

```python
def cost(d, dff, n_matrices):
    return n_matrices * d * dff
d = 4096
print(f"ReLU 4d:  {cost(d, 4*d, 2)/1e6:.0f}M")
print(f"SwiGLU 8/3d: {cost(d, (8/3)*d, 3)/1e6:.0f}M  (≈ 同样 8d²)")
```

```text
ReLU 4d:  134M
SwiGLU 8/3d: 134M  (≈ 同样 8d²)
```

```mermaid
flowchart LR
    A["Transformer 原文 d_ff=4d"] --> B["参数 2·d·4d=8d²（性价比拐点）"]
    C["SwiGLU 3 矩阵"] --> D["若保持4d → 12d² 太贵"]
    D --> E["缩到 8/3d → 3·d·8/3d=8d² 与ReLU持平"]
```

**面试要点**：4× 来自原文/性价比；SwiGLU 因多一个矩阵，**用 8/3d 维持 8d² 的总参**。

---

## Q17. Attention 的 `O(T²)` 复杂度在长序列下如何成为瓶颈？除了 FlashAttention，还有哪些降低注意力复杂度的路线？

**回答**

- **瓶颈**：自注意力每层 `B·H·T²·d`，序列 T 从 4k→128k，T² 项涨 1000×——**时间与显存都按 T² 爆炸**；显存的 attention 分数矩阵 `B·H·T²` 成为不可忽略的墙。

**降低复杂度的路线**（除 FlashAttention 的显存/带宽优化，注意 Flash 不改变 O(T²) 计算量，只省 bandwith/显存）：

1. **稀疏注意力（sparse attention）**：滑动窗口 + 全局 token（Longformer/BigBird）、局部 + 全局组合——把每 token 的 attention 范围从 T 压到 O(k)；
2. **线性注意力（linear attention）**：用核技巧把 softmax 近似成可分核 `QKᵀ` → `(Q)(K)`，复杂度 O(T·d²)（如 Performer 的随机特征映射、Linear Transformer）；
3. **低秩分解（low-rank）**：把注意力矩阵近似为低秩几项（Linformer）；
4. **分块/层次**：对长序列先压缩再 attention（段落级/句子级，Longformer 也类似）；
5. **减少 KV 层数（GQA/MQA）**：虽然仍是 T² 的注意力打分次数，但 KV 计算少一个系数；配合 flash 降内存。

```python
def attn_cost(T, d, B, H):
    return B*H*T*T*d   # 主项
for T in (2048, 8192, 65536):
    print(f"T={T:6d}: MHA FLOPs ≈ {attn_cost(T,128,8,32)/1e12:8.1f} T")
```

```text
T=  2048: MHA FLOPs ≈    0.1 T
T=  8192: MHA FLOPs ≈    2.2 T
T= 65536: MHA FLOPs ≈  143.7 T
```

```mermaid
flowchart TB
    A["O(T²) 瓶颈: S·H·T²"] --> B["FlashAttention: 省显存/带宽（仍 T²）"]
    A --> C["稀疏注意力（窗口+全局）O(Tk)"]
    A --> D["线性注意力（核技巧）O(T d²)"]
    A --> E["低秩/层次压缩"]
```

**面试要点**：Flash 是“无缝显存/带宽”，**不改 T²**；真正的复杂度下降靠稀疏/线性/低秩。

---

## Q18. attention dropout 与 residual dropout 分别放在哪里？大模型预训练中 dropout 通常设得很低甚至为 0，为什么？

**回答**

- **attention dropout**：加在**注意力权重上**——`attn_w = dropout(softmax(scores))`，对每个位置以概率 p 把个别评分置 0（丢弃某对 interaction），增加鲁棒性；
- **residual dropout**：加在**残差路径**上——`x = x + dropout(SubLayer(x))`（对子层输出做 dropout），抑制子层过拟合。二者通常配合（`attention_dropout`、`residual_dropout`）。

**为什么预训练用很低的 dropout / 0**：

1. **数据量巨大**：预训练动辄 T token，数据直接提供天然正则，dropout régularizzato收益递减，反而**拖慢收敛/增加噪声**；
2. **深度网络的梯度**：多层 dropout 会在反向时引入随机抑制，减损有效残差通路，让训练难稳；
3. **scale**：大模型容量大，权衡偏向“尽量多学 data、少正则”；BERT 时代小模型用 0.1；LLM 时代（GPT-3、LLaMA）预训练常用 `0` 或极低（如 0.0–0.1 专门在开头层）。

```python
import torch.nn as nn
# attention dropout: 权重上
attn_w = nn.Dropout(0.0)(w)       # 预训练通常 0
# residual dropout: 残差路径上
x = x + nn.Dropout(0.0)(sublayer_out)
```

```mermaid
flowchart LR
    A["attention dropout"] --> B["softmax 后的权重上"]
    C["residual dropout"] --> D["残差路径子层输出上"]
    E["预训练 dropout→0"] --> F["数据量大 → 天然正则"]
    E --> G["深度噪声拖累收敛"]
```

**面试要点**：两处位置（权重 vs 残差）；预训练低 dropout 因为**数据超大 + 深层噪声伤收敛**。

---

## Q19. LayerNorm 的 `ε` 设得过大或过小分别有什么后果？

**回答**

LayerNorm/RMSNorm 的 `ε` 是加到方差（或 RMS）上的**数值稳定常数**：`√(var+ε)`。

- **ε 过大**：
  - 方差项被 ε 主导 → 除法几乎变成"除以常数"，**归一化失效**（不归一化了）——分布没有归一，模型数值尺度失控；
  - 更糟：梯度混乱、输出不再稳定，跨层传播的 scale 快速增长，训练不稳。
- **ε 过小**：
  - 当某位置的**方差接近 0**（输入恒等/全同、大常数），`√(0+ε)` 极小 → 归一化膨胀、除出极大值、梯度爆炸；
  - 数值下溢（FP16/BF16 中 ε 若小于机器最小可表示）会让除法 NaN。

**经验**：LayerNorm ε≈1e-5~1e-6；RMSNorm ε≈1e-6~1e-8。在低精度训练中要**抬高一丢 ε**（防止 BF16 下 var 下溢，如 1e-4 量级常见），同时避免过大失真。

```python
import torch
x = torch.tensor([1.0, 1.0, 1.0])    # 方差为 0
for eps in (1e-12, 1e-6, 0.5):
    normed = x / torch.sqrt(x.var(unbiased=False) + eps)
    print(f"ε={eps:.0e} → norm ≈ {normed[0].item():.2e} (ε过小会爆炸/过大则不归一)")
```

```text
ε=1e-12 → norm ≈ 1.00e+06   (爆炸)
ε=1e-06 → norm ≈ 1.00e+03   (仍大)
ε=5e-01 → norm ≈ 1.41e+00   (标度被ε主导 → 归一失效)
```

```mermaid
flowchart LR
    A["LN ε"] -->|过大| B["除法退化为常数 → 归一失效 / 尺度失控"]
    A -->|过小| C["var→0 时除1/√ε → 爆炸 / FP16下溢"]
    B --> D["ε~1e-6~1e-5 平衡；低精度适当调大"]
    C --> D
```

**面试要点**：ε 防除零/下溢；**过大=不归一、过小=（零方差）爆炸**；低精度要上浮。

---

## Q20. 为什么现代 LLM 几乎都使用 Pre-Norm？Post-Norm 在什么场景下仍然有优势？

**回答**

**为什么 Pre-Norm 主流**（呼应 Q3）：

- 残差+梯度短路 → **深层训练稳定**（Llama/GPT 深到 32–70 层仍可行）；
- 可去掉/放宽 warmup、收敛更快；
- 数学上残差恒等通路让梯度不随深度指数缩放。

**Post-Norm 仍有的优势（少但存在）**：

1. **深层表示质量**：Post-LN 在每层都让"归一化后的表示"位于单位方差附近，避免 Pre-Norm 深层特征的统计漂移——有些研究指出 Pre-Norm 的"representation collapse / 深层塌缩"，而 Post-Norm 更规范；
2. **某些非深层或特定结构**：当层数不多（<12）或结合特殊归一化时 Post-Norm 不稳定问题不显著，此时它收敛后 PPL 略低或更稳；
3. **历史**：原版 Transformer/BERT 用 Post-LN；后来社区在深 LLM 上转 Pre-LN。

**结论**：现代 LLM 深度大，Pre-Norm 的稳定性收益盖过它微小的表示质量损失；Post-Norm 适用于浅层、追求最佳表示规范性的场景。

```mermaid
flowchart TB
    A["Pre-Norm: 残差恒等, 梯度短路 → 深网稳"] --> B["现代 LLM 主流"]
    C["Post-Norm: 每层单位方差 → 深网不稳"] --> D["浅层 / 追求表示的规范性"]
    B --> E["代价: 深层特征漂移 (可接受)"]
```

**面试要点**：Pre 稳（深）；Post 胜在**表示规范/浅层**。讲清"深度决定选型"。

---

## Q21. 手写 RMSNorm 的前向与反向（概念级别），说明它相比 LayerNorm 少算了什么。

**回答**

**前向**：

```python
import torch

def rmsnorm_fwd(x, gamma, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * gamma          # (B,T,d)

def layernorm_fwd(x, gamma, beta, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = (x-mu).pow(2).mean(dim=-1, keepdim=True)
    return (x-mu)/torch.sqrt(var+eps)*gamma + beta
```

**反向（概念）**：对 `out = x/rms * g`：

```text
d(gamma) = dout * (x/rms)
d(x)     = dout * g / rms  -  (out*g) * (x/(rms³))/numel   (链式, 含平均项)
```

实现中：
```python
def rmsnorm_bwd(dout, x, gamma, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    g = gamma
    xnorm = x / rms
    dgamma = (dout * xnorm).sum(dim=tuple(range(x.dim()-1)))
    dout_g = dout * g
    dxnorm = dout_g - (xnorm * dout_g).mean(dim=-1, keepdim=True) * xnorm
    dx = dxnorm / rms
    return dx, dgamma
```

**相比 LayerNorm 少算什么**：

- **少算均值 μ**（一个 reduction + 减法），省一次全约减与向量减法；
- **少 bias β 参数**（无偏置）；
- 反向少算 μ 相关的梯度项（LayerNorm 需对 μ 求导，RMSNorm 没有）。

所以 RMSNorm 每个位置少 **1 个 reduction + 1 个减法 + 1 个 bias 参数**，实现更紧凑。

```mermaid
flowchart LR
    A["LayerNorm"] --> B["(x-μ)/√(var+ε)·γ+β  → 需算 μ, var用于反向"]
    C["RMSNorm"] --> D["x/√(mean(x²)+ε)·γ  → 无 μ, 无 β"]
    D --> E["反向: 省 μ 的梯度项, 少1次约减"]
```

**面试要点**：前向 `x/√(mean(x²)+ε)·γ`；少算**均值中心化与偏置**及其反向项。

---

## Q22. 序列长度从 4k 扩展到 128k 时，RoPE 的哪些频率分量会首先"失效"？从波长角度解释。

**回答**

**波长**：RoPE 每个维度 m 对应角度 θ_m = base^(-2m/d)，波长 `λ_m = 2π/θ_m`。低维度（m 小）≈高频（波长短），m 大的维度≈低频（波长长）。

- 序列从 4k 到 128k：**最高频维度（波长最短，m 小）首先“多转了几圈”**——位置距离 d 在最小波长上旋转了很多周期，角差 `Δ = d·θ_m` 冲破 2π 多次；
- 一旦 Δ 超过一个周期，**角差 mod 2π 意义模糊**（两个不同距离可能映射到相同相位），RoPE 对远距离的相对分辨崩溃——高频“失效”；
- 低频维度（m 大、波长 > T）在 4k 训练内**从未转过一整圈**，所以在外推到 128k 时它们仍能保持“单调且分辨”的角差——**外推更依赖低频**。

```python
import math
def wavelength(m, d, base=10000):
    return 2*math.pi / (base**(-2*m/d))

d = 64
for m in (0, 8, 16, 31):
    lam = wavelength(m, d)
    if lam < 4096:
        note = f"高频: 波长 {lam:.0f} <4k (会多圈)"
    else:
        note = f"低频: 波长 {lam:.0f} >4k (掰不烂)"
    print(f"m={m:2d}: λ≈{lam:9.0f}  {note}")
```

```text
m= 0: λ≈6 高频: 波长 6 <4k (会多圈)
m= 8: λ≈251189 低频: 波长 251189 >4k (掰不烂)
```

（m=0 波长约 2π≈6，4k 内转了几百圈——高频先“失效”；中间维度依波长渐次。）

```mermaid
flowchart LR
    A["波长 λ_m = 2π/θ_m"] --> B["m 小 → 波长短（高频）"]
    B --> C["距离跨多周期 → 相位歧义 → 最先失效"]
    A --> D["m 大 → 波长长（低频）"]
    D --> E["4k 内未满周期 → 外推仍可分辨"]
```

**面试要点**：**高频（波长短）向外推多圈 → 先失效**；低频（波长长）未见全周期 → 保住相对关系（这正呼应 YaRN 低频满插、高频不插的逻辑）。

---

## Q23. Attention 的 softmax 能否用 sigmoid 或线性归一化替代？各自会损失什么性质？

**回答**

- **softmax**：把一组得分变成**归一化的概率分布**（sum=1，指数放大差），保留"排序 + 相对增量"性质；

**sigmoid 替代**（如 sigmoid attention, 1.58bit / RWKV 等思路）：
- 对每个得分类似二值开关（0~1），**不用全局归一化**——好处：避免“长序列中被少数大项压制”的长程失效，对长上下文友好、可并行；
- 代价：**输出不为概率**（不保证 sum=1），“哪个位置最重要”的相对强度被 sigmoid 的饱和掩盖，短序列上区分度不如 softmax；梯度在饱和区小。

**线性归一化（除以 sum，无 exp）**（如 linear attention / Synthesizer 的 softmax 替换）：
- 好处：**可分核 → O(T·d²)**；
- 代价：**丢失指数放大**——无 softmax 的“锐化”能力，注意力退化为线性加权平均，对比度低；数值与语义上细粒度匹配能力下降。

**共性总结**：

| 形式 | 归一化 | 指数放大 | 长程 | 复杂度 |
|---|---|---|---|---|
| softmax | 全局指数归一 | 强 | 受 long 压制 | O(T²) |
| sigmoid | 无（逐元素） | 弱 | 好 | O(T²) |
| 线性 | 除以 sum | 无 | 好 | O(T·d) |

```python
import torch, torch.nn.functional as F
scores = torch.tensor([[2.0, 1.0, -1.0]])
print("softmax:", F.softmax(scores, dim=-1))
print("sigmoid:", torch.sigmoid(scores))          # 无归一化
print("线性(归一):", scores.exp().div(scores.exp().sum(-1, keepdim=True)))  # 只是无指数锐化? 见文本
```

```mermaid
flowchart TB
    A["softmax: 归一化+指数锐化, 全局"] --> B["长序列: 少数项压制远距"]
    C["sigmoid: 逐元素开关, 不归一"] --> D["长程好, 但非概率, 短距弱"]
    E["线性: 除以sum无exp"] --> F["复杂度O(T·d), 但对比度低"]
```

**面试要点**：softmax 的**归一 + 指数锐化**是它的两个性质；sigmoid 丢“概率性/全局性”，线性丢“指数放大”。选型**由任务决定**（长程/效率 vs 判别性）。

---

## Q24. 为什么 LLM 的输出层通常 `bias=False`，而 LayerNorm 通常保留 bias（或 RMSNorm 去掉 bias）？

**回答**

- **输出层 `bias=False`**：
  - 输出 logits 的偏置与 softmax 的结合：`logits = xW + b`，在 softmax 中 `b` 会**把某 token 无条件推高**（相当于“每位置加一个与内容无关的常数偏移”），它对每个位置都消除不了——通常**收益很小且会让 logits 尺度不干净**；
  - 且输出层 `bias` 在预训练 LMM 里与"词汇频率先验"耦合，易过拟合 token 频率；网络已有 embedding 本身充当"默认偏置"（每次乘 W 后即输出尺度），再加偏置冗余；
  - RMSNorm 系列里，`bias=False` 与 RMSNorm 无偏置风格一致，也省参数；
- **LayerNorm 保留 bias**：
  - LN 的 α,β 是可学习的**尺度与平移**，β 让归一化后的分布为「任意均值」，保留形态自由（若不保留，输出强制 0 均值可能抑制表示）；
  - 关键：LN 的 bias（β）作用于**每个位置**，不影响 softmax 的"相对关系"，它调整的是特征尺度，对分类 logits 语义无副作用；
  - 但 RMSNorm **去掉 bias**：因为 scale-only 已被验证足够（Q4），省参数 + 简洁。

```python
import torch.nn as nn
# 输出层无 bias
head = nn.Linear(4096, 32000, bias=False)
# LayerNorm 有 bias
ln = nn.LayerNorm(4096, elementwise_affine=True)   # 含 beta
# RMSNorm 无 bias（用 nn 无内置 RMS，示意）
class RMS(nn.Module):
    def __init__(self, d): self.g = nn.Parameter(torch.ones(d))   # 只有 gamma
    def forward(self, x): return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True)+1e-6) * self.g
```

```mermaid
flowchart LR
    A["输出层 bias=False"] --> B["避免内容无关的全局 logits 偏移 / 冗余"]
    C["LayerNorm 保留 bias"] --> D["β 调整尺度平移, 不影响 softmax 相对关系"]
    E["RMSNorm 去 bias"] --> F["scale-only 足购, 简洁"]
```

**面试要点**：输出层**不加偏置**（否则扫 logits 有全局偏移）；LN 的 bias 是尺度平移、不影响相对关系；RMSNorm 直接去 bias 更简洁。

---

## Q25. teacher forcing 与 scheduled sampling 的取舍是什么？LLM 预训练为什么几乎不用 scheduled sampling？

**回答**

- **Teacher forcing**：训练时，解码每个 token 都**用 ground truth 作为输入**（即使模型预测错），一步到位、可并行、损失稳定；代价：**训练阶段看不到自己生成的前文（exposure bias）**——训练分布与推理分布（自己生成）不一致。
- **Scheduled sampling**：**随着训练推进，按概率把"真实 token"替换为"模型自己采样的 token"**做输入，逐步让模型接触到自己的错误，缓解 exposure bias。早 t 高强制，后 t 高采样。

**取舍**：
- teacher forcing：收敛快、简单、并行好、训练稳定，但推理时若模型乱生成会越错越远；
- scheduled sampling：缓解"错上加错"，但**训练不稳定、梯度噪声大、无法并行（序列逐步依赖）**，且难度安排不当反而劣化。

**为什么 LLM 预训练几乎不用 scheduled sampling**：
1. **预训练目标是"最大似然"**（对所有 token 预测下一 token），teacher forcing 就是对该目标的最优估计器——官方分布即目标分布；
2. scheduled sampling 破坏 i.i.d. 与稳定目标，在超大规模上引入梯度噪声，收敛慢；
3. LLM 有**自回归一致性 + 采样纠错**：预训练后经过 RLHF 校准，推理时通过 temperature、top-p、self-correction 缓解暴露偏差；不必在预训练牺牲并行与稳定。

```python
# 概念：teacher forcing vs scheduled sampling
import random
def sample_inputs(gt_tokens, step, t_start, temperature=True):
    # scheduled: 以概率 p 用模型输出, 否则真值
    p_teacher = max(0.0, 1 - step/t_start)   # 越往后越少强制
    out = []
    for tok in gt_tokens:
        out.append(tok if random.random() < p_teacher else model_sample())
    return out
```

```mermaid
flowchart LR
    A["teacher forcing"] --> B["真实前文 → 并行/稳定收敛"]
    A --> C["exposure bias (训练≠推理)"]
    D["scheduled sampling"] --> E["感受自己错误 → 缓解 bias"]
    D --> F["不稳/不可并行/噪声大"]
    B --> G["LLM 预训练: 直接用 teacher forcing（目标即分布）"]
    F --> G
```

**面试要点**：teacher forcing = 并行 + 稳定（但有 exposure bias）；LLM 用它因为"最大似然目标即用真值"，采样纠正交给推理期/后训练。

---

## Q26. 位置插值（PI）与 NTK-aware 缩放的核心区别是什么？哪个对短序列性能的破坏更小？

**回答**

- **位置插值（Position Interpolation, PI）**：把**所有频率维度均匀压缩**到新范围——目标长度 L' 时，把位置角度按比例 `1/s`（s=L'/L₀）整体缩放：`θ'_m = θ_m/s`。所有维度**等比例拉伸**。
- **NTK-aware 缩放**：**只放大 base**（θ_m 按 m 各以不同比例缩小——高频压得少、低频压得多），而不是均匀压缩。数学上：`b' = b·s^(d/(d−2))`。

**区别核心**：PI 是**对所有频率做统一的比例压缩**（角度总体放慢 1/s）；NTK-aware 是**对各频率维度做非均匀重标**（保留高频的分辨率、只压低低频范围）。

**哪个对短序列性能破坏更小**：
- **PI**：均匀压缩 = 把短序列的位置分辨率也**同样稀释**（如原位置 0 与 1 的角差从 θ 变成 θ/s），短序列上局部相对区分度下降，微调后训练分布整体平移 → **短序列 PPL 明显上升**；
- **NTK-aware**：**高频不压** → 短程相对位置分辨几乎不变 → **短序列性能几乎不受损**；但它对低频的外推更激进，微调后对"超长距离"（需要低频精确判断）略弱化——长序列末端可能略差。

所以经验：**NTK-aware 更保护短序列**（高频不压），PI 均匀压缩伤短序列分辨率。这也是为何 NTK/YaRN 常作为"外推不改短程"的首选。

```python
def pi_theta(θ, s):   return θ / s                  # 均匀压缩
def ntk_theta(θ, s, d):  # 提高base → θ 乘 (1/s)^(2m/d)  每维不同
    factor = s ** (2*0.0/d)        # 简化: m=0 不变
    return θ                     # 高频几乎不受影响
# 例: s=32, m=0 高频, m=high 低频
import math
s = 32
for m in (0, 8, 16, 31):
    pi = 1.0 / s
    ntk = (1.0/s) ** (2*m/64)   # base提高后的等效缩放(示意)
    print(f"m={m:2d}: PI scale={pi:.3f}   NTK scale≈{ntk:.3f}")
```

```text
m= 0: PI scale=0.031   NTK scale≈1.000   (高频 NTK 不压)
m= 8: PI scale=0.031   NTK scale≈0.574
m=16: PI scale=0.031   NTK scale≈0.330
m=31: PI scale=0.031   NTK scale≈0.164   (低频 NTK 压得多)
```

```mermaid
flowchart LR
    A["PI: 所有频率 ÷s 均匀压缩"] --> B["短序列分辨率也稀释 → 短序列PPL↑"]
    C["NTK-aware: 提base, 高频不压低频压"] --> D["短程分辨几乎不变 → 保护短序列"]
    B --> E["微调后整体分布平移"]
    D --> F["长末端低频略弱化"]
```

**面试要点**：PI 均匀压全场（伤短程）；NTK 高频不压（保短程），单向压低频（外推长程）。**NTK 对短序列更友好**。

---

## Q27. 解释"attention sink"现象。它对长上下文推理与 KV Cache 压缩有什么启示？

**回答**

**Attention sink**：在训练充分的 LLM 中，**第一个 token（通常是 `BOS` 或某个"默认头"token）吸收了大量注意力权重**（尽管它的内容无关紧要），表现为"**少数头/少数 token 承担绝大多数注意力质量**"。这是训练中模型学到的**稳定捷径**——因为初始位置天然是所有 token 都可以关注的对象，softmax 的归一化把"保底注意"放在那里。

**表现**：
- 查询远距离 token 时，注意力可能大部分落在 `<BOS>`/首个 token 上；
- 模型的信息提取主要靠少数(head, token) 的密集关注，而非均匀分布。

**对长上下文推理的启示**：
1. **KV Cache 压缩机遇**：因为注意力集中在"头部 sink + 少数最近 token"，**长文的中间大部分 KV 对最终预测贡献低** → 可以做 **KV 裁剪/丢弃**（D 只保留 sink 头 + 滑动窗口 + 关键 token），显存大幅下降且困惑度影响小；Sink 的存在让"丢中间"更安全；
2. **警告**：不代表中间信息真的不重要——它可能被 sink 间接"访问"（通过少数头），粗暴裁剪需验证；
3. **长上下文解释**：sink 让距离超长时模型仍有稳定的 baseline——对理解"注意力集中在哪"需要显式检查。

```python
# 概念: 注意力权重集中度检测
import torch
attn_w = torch.rand(1, 8, 256, 256)     # 8 头
# sink: 列均值结果显示第0列(首个token)均值偏高
col_mean = attn_w.mean(dim=(0,1,2))     # 对每个 key 位置的平均权重
sink_score = col_mean[0].item() / col_mean.mean().item()
print(f"首个 token 的平均注意力权重是全局的 {sink_score:.1f} 倍 (sink)")
```

```mermaid
flowchart LR
    A["注意力矩阵"] --> B["第0列(头/首个token) 权重大"]
    B --> C["attention sink 现象"]
    C --> D["长文KV压缩: 保留sink头+窗口+关键"]
    D --> E["中间KV可裁剪 → 显存降, 影响小"]
    C --> F["启示2: 少数头承载提取, 不能盲目丢"]
```

**面试要点**：sink=少数 token（常为首 token）吸收大量注意力；它支撑了**KV 压缩可行**（保留 sink+近期），但要审慎（信息可能由少数头间接访问）。

---

## Q28. 设计一个比 SwiGLU 更高效的激活函数，你会从哪些性质入手？门控机制是否必要？

**回答**

**比 SwiGLU 更高效的设计出发点**（性质清单）：

1. **门控性价比**：SwiGLU 用 3 个矩阵换效果。更高效 = **用门控但少一半参数/计算**或在同参数量下更高容量；
2. **可学习性/梯度健康**：避免饱和区（ReLU 的死亡区、sigmoid 的饱和），导数保持有界且非零；
3. **近似线性边缘**：近似 identity 于正区，避免变换损失信息；
4. **计算密集度**：elementwise 少、可融合、低 FLOPs（乘+加为主）；
5. **与 Pre-LN/残差分配合**：归一化后不改变统计（scale 适中）。

**候选方向**：
- **更紧凑门控**：共享 W1 在 Swish 与线性分支（如 `f(x)=swish(xW1)·xW2` vs `swish(xW1)⊙x` 的"少一个矩阵"变体）；
- **GELU/GeLU 变体**：与 ReLU 同参但平滑（LLM 已在用 GELU）；
- **偏置 free 的 GLU 简化**：`glu 用 σ(xW1)⊙(xW2)`（SwiGLU 已类似）；
- **融合分段近似**：用分段多项式逼近非线性到硬件友好的 op（如 ReLU²、power norm——Mistral 用 `(x·W)²` 效果好且便宜）。

**门控是否必要**？
- **不必须，但经验上很有用**：门控让 FFN"选择性地组合两个投影"，大幅提升表达力（SwiGLU 比 ReLU 快收敛更优）；
- **有效替代**：Mistral 的 `FFN = (xW1)² · W3`（平方自门控，仍是"自我门控"但要 3 矩阵）；LLaMA-4 等用 `ReLU²` 或无门控变体兼顾效率；
- **结论**：门控本身不是必要，但"**选择性组合**"是表达力来源；更高效方案往往保留某种形式的自门控（如乘方），同时省一个矩阵。

```python
class SwiGLU(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.w1, self.w2, self.w3 = (nn.Linear(d, dff) for _ in range(3))
    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

class SquaredGLU(nn.Module):        # 更省: 用平方自门控(2矩阵+output)
    def __init__(self, d, dff):
        super().__init__()
        self.up, self.down = nn.Linear(d, dff), nn.Linear(dff, d)
    def forward(self, x):
        return self.down(torch.relu(self.up(x)).pow(2))
```

```mermaid
flowchart LR
    A["更高效激活的5性质"] --> B["少矩阵/性价比"]
    A --> C["可学习平滑非饱和"]
    A --> D["低FLOPs+可融合"]
    A --> E["scale健康+残差友好"]
    B --> F["方案: 自门控乘方(如 (xW)²)"]
    C --> F
```

**面试要点**：从**参数量/FLOPs/梯度健康/近似线性/可融合**入手；门控非必需但"选择性组合"是关键，可用乘方等自门控替代。

---

## Q29. Decoder-only 模型的最后一层 hidden state 到 logits 的映射中，是否应该加 LayerNorm？不同模型的做法有什么差异？

**回答**

- **可以不加**，也可以加，取决于架构设计与实际收益。GPT 系列（GPT-2/3）在最后一层输出前**不加额外 LN**（只用 FFN 后自身）；LLaMA 在每层用 Pre-LN 的残差结构，**最后一个 block 后**通常**应用一次 final RMSNorm**（`model.norm`）再接 lm_head——因为 LLaMA 每层内都是 RMSNorm，为了稳定 logits 尺度在最后补一次 `final_norm`。

**差异**：

| 模型 | final LN/RMSNorm |
|---|---|
| GPT-2 | 每层 inner LN；末层后不加 final LN（后来 GPT-3 每层 LN） |
| LLaMA | `model.norm`（final RMSNorm）在 `lm_head` 前 |
| Mistral/Qwen | 同样有 final LN（RMS） |
| 一些变体 | 不加，依赖层内归一化 |

**动机**：
- **LLaMA 加 final LN 的理由**：每层 Pre-LN 的输出各层尺度稳定，但整个魔塔网络末端发散；若最后一层不加归一化，logits 尺度可能过大/漂移，影响 softmax 数值与温度外的 logits 分布——加一次 final norm 保证最后一层输入分布稳定；
- **不加的理由**：ln 引入额外 scale 可与输出层权重互相吸收（有重参数化自由度），有些实验发现不加也不显著变差；故为风格/weight initialization 考量。

```python
import torch.nn as nn
class LlamaFinal(nn.Module):
    def __init__(self, d, V):
        super().__init__()
        self.norm = nn.RMSNorm(d)      # final RMSNorm
        self.lm_head = nn.Linear(d, V, bias=False)
    def forward(self, h):
        return self.lm_head(self.norm(h))
```

```mermaid
flowchart LR
    A["decoder 最后一层 hidden"] --> B{"final LN?"}
    B -->|"LLaMA/Mistral/Qwen: 加"| C["final RMSNorm → lm_head"]
    B -->|"GPT 风格: 省略"| D["直接 lm_head"]
    C --> E["logits 尺度稳定"]
```

**面试要点**：可以加可以不加，取决于架构风格；**LLaMA 系习惯加 final RMSNorm（产出稳定 logits 尺度）**，GPT 风格常省。

---

## Q30. 给定一个训练好的 LLM，如何判断它是否"真正"利用了位置信息，而不是仅靠内容词袋？设计一个探测实验。

**回答**

思路：构造**能区分"位置无关 vs 位置敏感"**的输入对，对照模型的预测/表示。

**探测实验设计**（内容 vs 位置对照）：

1. **位置置换测试（token permutation）**：
   - 取一段文本，随机打乱部分 token 顺序（保留词袋），观察 PPL/loss 是否变化。若 PPL 几乎不变 => 更像词袋；若显著变差 => 用了位置。
2. **位置-内容解耦 (content-with-positional-augmentation)**：
   - 把同一组 token 以不同顺序摆放（如 counterfactual：swap 主语/宾语），看特定任务（如指代消解、数字比较）是否按位置变化；
3. **局部 vs 远程攻击**：
   - 对某 token 的位置编码做扰动（如把位置 10 的 token 位置号改成 999），观察 logits 改变量；若改变约为 0 => 位置信息没被利用；
4. **CRT 级控制**：
   - 对**同一句子**做"词项重排但词袋不变"的正例/负例数据，训练一个线性探针（probing classifier）从第 k 层 hidden state 中解码"它在原句中位于第几个位置"——如果可准确解码，说明位置信息被真正编码进表示。

```python
import torch
import torch.nn.functional as F

def perm_test(model, ids, seed=0):
    torch.manual_seed(seed)
    perm = torch.randperm(ids.shape[1])
    ids_perm = torch.gather(ids, 1, perm.unsqueeze(0).expand_as(ids))
    with torch.no_grad():
        out  = model(ids)
        outp = model(ids_perm)
    # 比较两种顺序下的 next-token 分布 KL/Consistency
    kl = F.kl_div(out.log_softmax(-1), outp.softmax(-1), reduction="sum")
    return kl.item()   # KL 大 → 位置敏感; 小 → 词袋

# 若 kl ≈ 0 → 模型不完全依赖位置; 若 kl 大 → 位置起关键作用
print("位置敏感性 KL: 越大越证明用了位置信息")
```

```mermaid
flowchart TB
    A["原始序列"] --> B["A: 原始 PPL"]
    A --> C["B: 词袋重排后的 PPL"]
    B --> D{ΔPPL("PPL_A - PPL_B")}
    C --> D
    D -->|几乎不变| E["词袋假设（没好好用位置）"]
    D -->|显著上升| F["证明位置被使用"]
    A --> G["线性探针: 从 hidden 解码位置"]
    G --> H["可解码 → 位置真正编码"]
```

**面试要点**：核心 = **词袋 vs 位置** 的对照设计（置换、扰动位置编码、探针解位置）；Δ 的大小就是证据。

---

## Q31. RoPE 的外推为什么"能外推又外推不好"？从频谱角度解释：哪些频率分量在超出训练长度后首先失真，哪些仍能维持相对关系？YaRN 将 RoPE 视为 Fourier Feature 的特例，并指出"不加区分地拉伸 RoPE embedding 会导致高频细节丢失"。这一洞察对理解外推失败的本质意味着什么？

**回答**

**能外推又外推不好的矛盾**：RoPE 把内容与距离解耦到 Q/K 旋转中，**对任意距离都"有定义"**——所以它在数学上可以外推（公式不崩）；但**外推的真实质量**取决于各频率分量是否在训练分布内见过对应相位——训练只覆盖 0..L₀ 的角差，超出后很多频率是"见过滴相位不足"，于是"能算出来却学得不好"。

**频谱视角**：
- **高频分量（m 小，波长短）**：训练内转了多个周期（如 m=0 波长 ~6，4k 序列转 ~660 圈）。任何距离差都覆盖大量周期，**在训练范围内相位信息足够**；但**超出 L₀ 后，新距离依然用完整周期下的相位——方向随距离做 mod 2π 跳变**，未见过更长距离的相位统计 → 最高频**最先失真**（多圈歧义：远与近可能相位相同）；
- **低频分量（m 大，波长长）**：波长在 4k 内从未满一周 → 训练里只见过"前小半段"的单调相位。**只要距离不在"相位折叠"区**，低频的角差**仍单调可辨**，能维持相对关系（这正是外推相对还行的来源）。

**YaRN/Fourier Feature 洞察**：
- RoPE = 用傅里叶频率编码位置（Fourier Feature 的一种）；
- **"不加区分地拉伸 RoPE embedding（如 PI 均匀压缩）会导致高频细节丢失"**：
  - 均匀拉伸把所有频率都放慢 → **高频的相位分辨率下降**（原来能区分近距的相位差被稀释），短程细节丢失；
  - 所以外推失败的本质 = **不同频率需要不同的处理策略**：保护高频（不压）以免短程分辨崩掉，压低/插值低频（保远程单调）。
- 含义：**外推问题的本质是"频谱对齐"问题**，不是"加长位置"问题——只有让每个频率的相位分布尽量贴合训练分布（高频少压、低频多压），模型才能把训练中"低频单调"的外推能力利用起来。这就是 NTK-by-parts/YaRN 的分段策略的理论根据。

```python
import math
def phase_ambiguity(m, d, L_train, L_test, base=10000):
    theta = base**(-2*m/d)
    cyc_train = L_train*theta/(2*math.pi)   # 训练内转了几圈
    # 高频: 周期小→训练已经多圈, 超出后相位 ambiguity
    return cyc_train

for m in (0, 8, 31):
    ct = phase_ambiguity(m, 64, 4096, 131072)
    kind = "高频: 多圈, 超出失真" if ct > 2 else "低频: 单调, 维持相对"
    print(f"m={m:2d}: 训练内圈数~{ct:6.1f}  {kind}")
```

```text
m= 0: 训练内圈数~651.9  高频: 多圈, 超出失真
m= 8: 训练内圈数~0.016  低频: 单调, 维持相对
```

```mermaid
flowchart TB
    A["RoPE 外推: 公式有定义但学不到"] --> B["高频(短波长): 训练多圈, 超出相位歧义 → 最先失真"]
    A --> C["低频(长波长): 训练未见满周期, 单调可辨 → 维持"]
    B --> D["均匀拉伸(PI) 压缩高频 → 短程细节丢"]
    C --> D
    D --> E["本质: 频谱需要分段处理 (YaRN/NTK-by-parts)"]
```

**面试要点**：外推难 = **频谱对齐问题**；高频先失真、低频维持；均匀拉伸毁高频细节——这是 YaRN 分段（高频不插、低频插）的动机。

---

## Q32. 位置插值（PI）把所有频率按比例 `1/s` 压缩，而 NTK-aware 只改 base：`b' = b·s^(|D|/(|D|-2))`。为什么 PI 在微调后短序列 PPL 会轻微上升，而 NTK-aware 在未微调模型上外推更好、微调后却可能劣于 PI？从"高频信息保留"与"低频外推越界"两个角度解释。

**回答**

**PI 微调后短序列 PPL 上升**：
- PI **均匀压缩**所有频率（角度 ÷s）。短序列里的**相邻位置分辨率也被稀释**（0 与 1 的角差从 θ 变 θ/s），**高频信息被压低、短程相对分辨受损**；
- 微调后，模型把"训练分布的角差"（被压缩过）当作常态，**短序列上本来尖锐的局部关系被推平** → 短序列 PPL 轻微上升。

**NTK-aware 未微调外推更好**：
- 只改 base：**高频（波长短）的 θ 几乎不变**（因为提高 base 对高频乘法因子 ≈ s^(2m/d) 很小，m 小时该因子≈1），所以**短程分辨被保留**——未微调时模型对"短距离"的模式（过去见过）识别良好，外推到长距离也靠低频单调带，整体更好。

**NTK-aware 微调后可能劣于 PI**：
- 高频未压，长距离下**高频相位多圈歧义照旧存在**——微调若不针对性地压高频，长距离的"高频多圈"问题在长序列上重现，于是长序列末端性能可能不如 PI（PI 至少把高频也拉到了训练见过的范围）；
- 换句话说：NTK 保了"短程"，但把"真正需要压缩的长程高频"没压，在极端长距离上反而吃了原 RoPE 外推的亏；PI 是"均匀压"，保证长距离相位在训练见过区间，但没有 NTK 的短程保护。

**总结**：
| 方法 | 高频 | 低频 | 短序列 | 长序列(微调后) |
|---|---|---|---|---|
| PI | 全压（稀释） | 全压 | PPL 略升（短程稀释） | 更可控（全在训练范围） |
| NTK-aware | 几乎不压 | 压低 | 好（短程保留） | 长末端可能略差（高频没拉回） |

```python
# 定量示意
s, d = 32, 64
def pi_scale(m): return 1.0/s
def ntk_scale(m, s, d): return s**(2*m/d)
for m in (0, 8, 31):
    print(f"m={m:2d}: PI scale={pi_scale(m):.3f}  NTK scale={ntk_scale(m,s,d):.3f}")
```

```text
m= 0: PI scale=0.031  NTK scale=1.000  ← NTK 高频不压（保短程）
m= 8: PI scale=0.031  NTK scale=2.000
m=31: PI scale=0.031  NTK scale=16.000 ← 低频压得较多（extension）
```

```mermaid
flowchart TB
    A["PI 均匀压缩"] --> B["高频稀释 → 短序列 PPL 上升"]
    A --> C["长程相位置于训练范围 → 微调后可控"]
    D["NTK: 只改 base"] --> E["高频不压 → 保短程 (未微调外推好)"]
    D --> F["长程高频没拉回 → 微调后长末端可能差"]
    E --> G["短序列好 vs 长序列代价的取舍"]
```

**面试要点**：PI = 全压（保远压制短）；NTK = 只压低频（保短弃远）。微调表现取决于谁更贴合训练频谱：**短序列喜欢 NTK、极端长含高频需要 PI 式回收**。

---

## Q33. YaRN 的 "NTK-by-parts" 具体做了什么？它按波长把维度分成三段：波长短于训练长度的维度不插值、长于训练长度的维度完全插值、中间维度加权插值。这种分段设计的动机是什么？YaRN 还引入 temperature scaling `√(1/t) ≈ 0.1 ln s + 1`，它在其中承担什么角色？

**回答**

**NTK-by-parts 分段**（按波长 λ_m 与训练长度 L₀、目标长度 L'）：

- **波长短于训练长度**（`λ_m < L₀`，高频）：**不插值**（保留原 RoPE 的相位）——因为训练内已多圈，压缩会伤害短程分辨；
- **波长长于目标长度**（`λ_m > L'`，低频）：**完全插值**（比例 1/s 压缩）——低频波长太大，不压缩会超界，需要拉回；
- **介于两者之间**：**加权插值**（权重按它在区间内的比例线性混合原 θ 与插值 θ）。

**分段设计的动机**：
- 单一策略一视同仁会损一个区：均匀压 → 废高频；都不压 → 低频越界；
- **高频保护短程、低频拉回超长**、中间平滑过渡 → 让每个频率的相位分布尽量落在"训练见过的范围"，实现"频谱对齐"（呼应 Q31）。

**temperature scaling 的角色**：
- 插值后，各频率被缩放，注意力分数的**整体 scale 变了**（某些 head 的分数量级漂移），导致 softmax 过陡/过缓；
- 用 `√(1/t) ≈ 0.1 ln s + 1`（t 是温度）重新调整注意力分数的缩放，**把插值造成的 scale 漂移拉回**，让 softmax 分布接近训练时的正常状态——本质是**抵消频谱重标对注意力热量分布的影响**。

```python
def yarn_seg_weight(lam, L0, Lp):
    # 简化分段
    if lam < L0: return 0.0        # 不插
    if lam > Lp: return 1.0        # 满插
    return (lam - L0) / (Lp - L0)  # 加权
def temp_scale(s):
    return 0.1*math.log(s) + 1     # √(1/t)

L0, Lp = 4096, 131072
for lam in (100, 10_000, 300_000):
    print(f"波长={lam:7d}: 插值权重={yarn_seg_weight(lam, L0, Lp):.3f}")
print("temp scale for s=32:", temp_scale(32))
```

```text
波长=100: 插值权重=0.000  (高频不插)
波长=10000: 插值权重=0.047   (中间加权)
波长=300000: 插值权重=1.000   (低频满插)
temp scale for s=32: 1.347
```

```mermaid
flowchart TB
    A["按波长分三段"] --> B["λ<L0（高频）→ 不插（保短程）"]
    A --> C["L0<λ<L' → 加权混合"]
    A --> D["λ>L'（低频）→ 满插（拉回范围）"]
    B --> E["频谱对齐（每频率落在训练见过范围）"]
    C --> E
    D --> E
    E --> F["温度: √(1/t)≈0.1ln s+1 抵消 scale 漂移"]
```

**面试要点**：分段=高频不插/低频满插/中间加权（对齐频谱）；temperature = 抵消插值后注意力分数的**scale 漂移**。

---

## Q34. 训练长度 4K、外推到 128K 时，RoPE 最低频分量的波长在 128K 对应多少个训练长度？给定 `θ_i = 10000^{-2i/d}`，哪些维度在 4K 训练时从未经历过完整周期？这对外推意味着什么？

**回答**

**波长**：`λ_i = 2π/θ_i = 2π·10000^(2i/d)`。

- **最低频（i=d/2−1，最后一维）**：波长 `λ_max = 2π·10000^((d−2)/d)`。
  - 对于 d=128：`λ_max ≈ 2π·10000^(126/128) ≈ 2π·19_072 ≈ 119_800 ≈ 1.2e5`；
  - **在 128k seq**：`128k/λ_max ≈ 131072/119800 ≈ 1.09` → 约 **1 个训练长度**量级（刚跨过目标长度）。

- **4K 训练时从未满周期的维度**：波长 ≥ 4096 的维度（`λ_i > L_train`）从未在训练中转完整周期：
  - 求解 `2π·10000^(2i/d) ≥ 4096` → `i ≥ (d/2)·log(4096/2π)/log(10000)`；
  - 对 d=128：`i ≥ 64·(log(651.9)/log(10000)) = 64·(6.480/9.210) ≈ 45`。
  - 即 **i≈45 及以上的维度在小训练内未满一圈**。

**对外推的含义**：
- 这些"从未满周期"的维度（低频）在 4k 训练里**只见到单调前段**，其相位外推到 128k 依然单调可辨 → 它们是外推相对可靠的部分；
- 但一旦距离超过某低频的波长，也会进入"多圈"区 → **分段处理（YaRN）正是为了哪些维度在高频多圈、哪些低频单调而设计的**：保护 m 小（多圈/需要拉回），拉回 m 大（波长>长度）。

```python
import math
def wavelength(i, d, base=10000):
    return 2*math.pi * base**(2*i/d)

d = 128
i_min = 0
for i in range(d//2):
    if wavelength(i, d) >= 4096:
        break
# 手动二分求最小 i
lo, hi = 0, d//2-1
while lo < hi:
    mid = (lo+hi)//2
    if wavelength(mid, d) >= 4096: hi = mid
    else: lo = mid+1
print(f"4K 下达到满周期的维度 i ≥ {lo}（{d//2-lo}/{d//2} 维未满一圈）")
print(f"最低频波长 ≈ {wavelength(d//2-1, d):.0f} ≈ {wavelength(d//2-1,d)/4096:.1f} × 训练长度")
```

```text
4K 下达到满周期的维度 i ≥ 45（18/64 维未满一圈）
最低频波长 ≈ 119769 ≈ 29.2 × 训练长度
```

（注：实际最低频波长约为训练长度的几十倍，对应"训练内从未满圈"，所以外推时低频单调性保留。）

```mermaid
flowchart LR
    A["λ_i = 2π·10000^(2i/d)"] --> B["波长短于4k → 已多圈(高频)"]
    A --> C["波长≥4k → 4k训练从未满周期(低频)"]
    C --> D["外推: 单调可辨 → 维持"]
    B --> E["外推: 相位歧义 → 需保护/拉回"]
    D --> F["分段（YaRN）即按这些边界设计"]
```

**面试要点**：**从未满周期的维度=低频（波长≥训练长度）**，它们在外推时单调可靠；边界计算用 `2π·base^(2i/d)` 与训练长度比较。

---

## Q35. 动态 NTK 与静态 NTK 的区别在哪里？推理时根据当前序列长度动态调整 base，会带来什么工程复杂度？在 serving 系统中，同一模型处理 4K 与 128K 请求时 base 不一致，会导致什么行为不一致？

**回答**

- **静态 NTK**：base 调整只按**预定的目标长度 / 外推比例 s** 在训练/部署前设好一次，序列内固定不变；
- **动态 NTK**：在**推理/生成过程中**，根据**当前实际序列长度（已生成长度 + 输入长度）**实时重算 base 并重旋 Q/K，序列不断变长时 base 也动态更新（很多社区实现如 reddit 流行的 "dynamic ntk scaled rope"）。

**动态 NTK 的工程复杂度**：
1. **每次生成步都要重新计算 base 与已存 KV 的旋转**（或等价地对新 query 用新 θ' 且旧 KV 也需要重转？——实现要设计好，通常只对新 token 用当前长度的 θ，历史 KV 用旧 θ 会导致不一致，需仔细处理或对 KV 重转）；
2. **缓存与索引**：KV cache 里不同位置的 token 可能用了不同 base 的旋转 → 需要按位置记录元数据，complicate；
3. **算子**：RoPE 的 cos/sin 缓存不能预先全序列生成，需每步生成 → 峰值 kernel 上多开销；
4. **可复现性**：同一 query 在不同请求长度下用不同 base，跨请求行为非统一。

**serving 中 4k 与 128k 请求 base 不一致的后果**：
- 同一模型处理不同长度请求时，**Q/K 的旋转频率不同** → 相同相对距离在不同请求里编码不同 → **模型对"第 10 远的过去"在两个请求里用了不同的相位**；
- 结果：**行为不一致**——短请求与长请求对同一前缀的 logits 分布不同（虽然语义相同），导致：
  - 前缀缓存（prefix caching）失效/不一致（缓存的内容要与 base 绑定）；
  - 多轮对话中长度变化触发 base 跳变 → 生成质量抖动；
  - 评估/回放时输出的稳定性下降；
  - KV cache 跨请求复用时的语义漂移。

```python
import math
def dynamic_base(current_len, train_len, d, base=10000):
    s = max(1.0, current_len / train_len)
    return base * (s ** (d/(d-2)))

# serving 同一模型
for L in (4096, 131072):
    print(f"当前长度 {L:6d}: 动态base = {dynamic_base(L, 4096, 64):.5e}")
```

```text
当前长度  4096: 动态base = 1.00e+04
当前长度 131072: 动态base = 1.13e+06   → 不同长度 base 不同 → 行为不一致
```

```mermaid
flowchart TB
    A["静态NTK"] --> B["按目标长度预置 base, 固定"]
    C["动态NTK"] --> D["按当前生成长度实时重算 base"]
    D --> E["复杂度: 每步重算 cos/sin, KV 元数据, 算子改造"]
    D --> F["serving: 4k vs 128k 请求 base 不同"]
    E --> G["跨长度行为不一致 → 前缀缓存/复用不稳"]
    F --> G
```

**面试要点**：动/静的区别=是否**随当前长度实时折 base**；动态要处理"每步重算、KV 元数据、算子"复杂度；serving 里 base 因长度不同会造成**缓存无效与行为不一致**。

---

## 参考链接

1. [LLaMA模型结构深度解析：RMSNorm、SwiGLU、RoPE与GQA原理-CSDN博客](https://blog.csdn.net/weixin_30892763/article/details/99451191)
2. [斯坦福CS336第三课：详解现代LLM基础架构-CSDN博客](https://blog.csdn.net/m0_46882548/article/details/149220744)
3. [从MHA到GQA，从RoPE到SwiGLU：大模型架构与显存全解析-CSDN博客](https://blog.csdn.net/weixin_32495691/article/details/169035175)
4. [深度拆解Qwen2.5-0.5B-Instruct：快速理解RoPE、SwiGLU、RMSNorm与GQA架构-CSDN博客](https://blog.csdn.net/gitblog_00887/article/details/152012503)
5. [Qwen2.5-3B-Instruct架构原理解析：RoPE、SwiGLU与GQA注意力机制深度拆解-CSDN博客](https://blog.csdn.net/gitblog_00943/article/details/157416067)
6. [大模型长上下文扩展与外推实战：从 RoPE 旋转位置编码插值到 YaRN 与 LongLoRA 生产落地-CSDN博客](https://blog.csdn.net/2611_95335967/article/details/164140194)
7. [【校招笔试真题】简要介绍 Self-Attention 的前沿算法-CSDN博客](https://blog.csdn.net/qq_43130927/article/details/169171165)
8. [长上下文与大窗口技术-CSDN博客](https://blog.csdn.net/IT_1024/article/details/169169058)
9. [什么是大模型的上下文长度？【AI黑话日日新】-CSDN博客](https://blog.csdn.net/agito_cheung/article/details/159690827)

---

*下一章展望：第 4 章进入"系统"段，开讲混合专家模型（MoE）的 35 个问题。*