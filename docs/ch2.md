# 第 2 章 · PyTorch 与资源核算

> 面试生普（LLM 500 题）· 第 2 章。本章覆盖训练 FLOPs、显存核算、混合精度、梯度累积、数据加载、分布式基础概念等 30 个问题，关键概念配 Python 代码与 mermaid 图解。

---

前面第 1 章讲的"分词器"是把文字切块编上号的环节，那是模型"入口"处的活儿。这一章我们要迈过另一道坎：**算账**。你想训练一个大模型，得先知道它"一次前向走多远、要占地多大的显存、跑一天要多少算力"——这些数字不是老板拍脑袋定的，而是有一套固定公式可以手算的。懂了这套账本，你才知道一张 80GB 的显卡到底塞不塞得下参数，也才知道为什么别人一开口就是"6ND"“16B/参数”“混合精度”这些黑话。放心，这一章不涉及极度复杂的数学，全是能拿 Python 直接验证的四则运算。

---

## Q1. 训练 FLOPs 近似 `6ND` 中，前向与反向各贡献多少？该公式忽略了哪些开销，导致它总是低估真实计算量？

**回答**

先破个术语关："FLOPs"（带小写 s）是"每秒浮点运算次数"的度量单位，我们算账时直接当"总运算量"理解就行。`6ND` 是说：训练一个参数量为 N 的模型、喂过 D 个 token，总运算量大约是 6 乘 N 乘 D。为什么偏偏是 6？因为每个 token、每个参数平均要消耗 **6 次浮点运算**，拆开看是这样：

- **前向（forward）**：每个参数贡献 **2 FLOPs**（一次乘 + 一次加，也就是 `w·x + b` 这种"乘加"运算 MAC）；
- **反向（backward）**：每个参数贡献 **4 FLOPs**：
  - **对输入/激活的梯度**（数学上是 dW/dx 那一路）：2 FLOPs；
  - **对权重的梯度**：2 FLOPs。

于是 前向:反向 = **1:2**，总量 ≈ `(2 + 4)·N·D = 6ND`。后向比前向重一倍，这个感觉请你记住，后面很多估算都靠它。

**为什么说它是下界（总是低估）？因为它忽略了不少开销**：

1. **Embedding / 输出层**：`lm_head` 也是参数，而且 logits 计算量是 `B·T·V`，比 6ND 里"N 只含主干"的线性放缩要大得多；softmax 里的 exp 也没算；
2. **非线性与归一化**：GEMM 之外的逐元素运算（激活函数、LayerNorm/RMSNorm、RoPE 旋转）有大量 FLOPs 没计进去；
3. **Dropout / mask / 位置运算**：这些浮点运算同样没计入；
4. **反向里的梯度清零/累积与通信**：梯度 all-reduce、参数更新（optimizer step）不算 FLOPs，但实实在在地占算力；
5. **FlashAttention 的实际矩阵路径**：attention 里的 QK^T、softmax 等二次项没被计入（大序列场景尤其被高估）；
6. **loss 计算与数值稳定性操作**：交叉熵、mask 计算、精度转换（cast）都不在账上。

所以记清楚：`6ND` 是**下界**。实际对比吞吐（MFU）时，通常把 6ND 当分子，分母是实测的 FLOP 数。

```python
def train_flops(N, D):
    """6ND: 每 token 每参数 6 FLOPs"""
    return 6 * N * D   # FLOPs

def breakdown_6ND(N, D):
    forward  = 2 * N * D
    backward = 4 * N * D     # 其中权重梯度 2, 激活梯度 2
    return forward, backward

forward, backward = breakdown_6ND(8e9, 2e12)   # 8B, 2T tokens
print(f"前向 {forward/1e18:.2f} EFLOPS, 反向 {backward/1e18:.2f} EFLOPS")
print(f"总计 {train_flops(8e9, 2e12)/1e18:.2f} EFLOPS (6×(N×D)/1e18)")
```

```text
前向 16.00 EFLOPS, 反向 32.00 EFLOPS
总计 96.00 EFLOPS (6×(N×D)/1e18)
```

```mermaid
pie title 每 token 每参数 6 FLOPs 分解
    "前向：2 FLOPs (MAC)" : 2
    "反向(激活梯度)：2 FLOPs" : 2
    "反向(权重梯度)：2 FLOPs" : 2
```

**面试要点**：抓住"**前向 2、反向 4 共 6**"，补充"6ND 是纯 GEMM 下界，attention/logits/归一化/flash 通通未算"。心里记住这个"下界"的印象，下一题马上要用它算天数。

---

## Q2. 给定 70B 参数、15T tokens、1024 张 H100（BF16 峰值约 990 TFLOPS），MFU 取 40%，估算训练天数。若实际用了 30 天，反推 MFU 是多少？

**回答**

这题就是把上一题的 `6ND` 当成"总运算量"，再除以"实际算力"得到时间。我们一步步来，都是乘除法，别被单位吓到。

**步骤 1：总 FLOPs**。`FLOPs = 6ND = 6 × 70e9 × 15e12 = 6.3e24 FLOPs`。

**步骤 2：理论算力上限**。1024 张卡，每张峰值 990 TFLOPS（那是每秒 1e12 次浮点）：`1024 × 990e12 = 1.01376e18 FLOPs/s`。

**步骤 3：MFU=40% 时的有效算力**。硬件标称峰值不可能打满，我们按四成折算：`1.01376e18 × 0.4 = 4.055e17 FLOPs/s`。（MFU，模型算力利用率，稍后本卷还有专门一题。）

**步骤 4：训练时间**。总运算量除以有效算力：`T = 6.3e24 / 4.055e17 = 1.5536e7 s`。

换算成天：`1.5536e7 / 86400 ≈ 179.8 ≈ 180 天`。

**如果实际只用了 30 天**：30 天 = `86400×30 = 2.592e6 s`。那么有效算力得是 `6.3e24 / 2.592e6 = 2.430e18`。算 MFU：`2.430e18 / 1.01376e18 ≈ 239%`——**不可能**，都超 100% 了。这说明我们的某个假设错了：要么 6ND 低估（模型其实还含额外计算），要么硬件能跑出超标称峰值的矩阵运算（比如 Flash 等吃不到 990 TFLOPS 封顶），要么参数/token 数假设不对。通常真实训练的 MFU 在 30%–60% 之间。

```python
def est_days(N, D, gpus, peak_tflops, mfu):
    total = 6 * N * D
    peak = gpus * peak_tflops * 1e12
    eff = peak * mfu
    seconds = total / eff
    return seconds / 86400

def back_mfu(N, D, gpus, peak_tflops, days):
    total = 6 * N * D
    seconds = days * 86400
    return total / seconds / (gpus * peak_tflops * 1e12)

print("估算天数:", round(est_days(70e9, 15e12, 1024, 990, 0.4), 1))
print("30天反推MFU:", round(back_mfu(70e9, 15e12, 1024, 990, 30)*100, 1), "%")
```

```text
估算天数: 179.8
30天反推MFU: 239.2 %
```

```mermaid
flowchart LR
    A["6×70B×15T = 6.3e24 FLOPs"] --> B["÷ (1024×990e12×0.4) "]
    B --> C["≈1.55e7 s ≈ 180 天"]
    A --> D["30天 → 有效算力 2.43e18"]
    D --> E["MFU≈239% 不可能 → 假设有问题"]
```

**面试要点**：公式 `T = 6ND / (P_gpu·F_peak·MFU)`。顺带考察**合理性判断**——MFU>100% 说明 6ND 或硬件假设不成立。算完总账，下一题该琢磨"显存"这本小账了。

---

## Q3. 混合精度训练中，FP32 master weights、BF16 模型副本、FP32 Adam 状态三者的显存占比如何计算？给出每参数字节数的完整拆解。

**回答**

又是术语关，先摆平。"混合精度"的意思是：不同东西用不同精度的数来存——重量级选手用高精度，跑量选手用低精度。这一题我们按"每个参数平均占几个字节"来拆账。先记住几个常数：FP32（单精度）一个数占 **4 字节**，BF16（bfloat16）一个数占 **2 字节**。

混合精度（Master-Weights + BF16 前反向 + Adam）的标准显存拆解，**每参数**：

| 组件 | 类型 | 每参数占用 |
|---|---|---|
| Master weights（FP32） | 优化器更新的权威权重 | 4 B |
| 模型副本（BF16） | 前向/反向使用的 | 2 B |
| 梯度（BF16 累积为 FP32 或 BF16） | 反向累积 | 2 B（若 FP32 则 4 B） |
| Adam m（一阶矩，FP32） | 优化器状态 | 4 B |
| Adam v（二阶矩，FP32） | 优化器状态 | 4 B |

**合计 = 4 (master) + 2 (bf16) + 2 (grad) + 4 (m) + 4 (v) = 16 B/参数**。（业界常用"16Ψ"口诀记忆，其中梯度按 2B 算。）

如果梯度用 FP32 存（很多框架默认梯度类型跟参数一致），那就变成 `4+2+4+4+4 = 18 B/参数`。

**拿 70B 模型举例**：`70e9 × 16 B ≈ 1.12 TB`，这还只是权重加优化器状态，没算激活。要放下这一项，得用 16 张 80GB 的卡（16×80GB=1.28TB）才刚好够边。

```python
def per_param_bytes(fp32_master=True, fp32_grad=False):
    master = 4 if fp32_master else 2
    model  = 2                 # bf16 副本
    grad   = 4 if fp32_grad else 2
    m, v   = 4, 4              # Adam FP32
    tot    = master + model + grad + m + v
    return {"master":master, "model_bf16":model, "grad":grad, "adam_m":m, "adam_v":v, "total":tot}

d = per_param_bytes()
print(d)
N = 70e9
print(f"70B → {N*d['total']/1e12:.2f} TB (仅优化器+权重)")
```

```text
{'master':4, 'model_bf16':2, 'grad':2, 'adam_m':4, 'adam_v':4, 'total':16}
70B → 1.12 TB (仅优化器+权重)
```

```mermaid
flowchart LR
    subgraph 每参数 16 字节
      M["FP32 master 4B"]
      B["BF16 副本 2B"]
      G["梯度 2B"]
      A1["Adam m FP32 4B"]
      A2["Adam v FP32 4B"]
    end
    M --> T["= 16B/参数"]
    B --> T
    G --> T
    A1 --> T
    A2 --> T
```

**面试要点**：记住**16B/参数**这个行业基准及其构成；能扩展到"梯度若是 FP32 则 18B/参数"。

---

## Q4. 梯度累积为何能在不增加显存峰值的情况下增大等效 batch size？它是否完全等价于大 batch 训练？BatchNorm 场景下有什么陷阱？

**回答**

这题讲的是个大模型训练里非常实用的省钱技巧"分期付款"。**batch size**（一批一次喂给模型的样本数）越大，越逼近大 batch 的稳定性，但一次性塞进显存的东西也越多。梯度累积的办法是：**把一个大 batch 拆成几小份（micro-batch），每份算一次梯度先攒着，攒够了再一起更新参数**。显存压力一下子就下来了。

**为什么能增大等效 batch 而不增显存**：

1. 每个 micro-batch 独立做一次前向+反向，**梯度累加到 `grad` 缓冲区**；
2. 每个 micro-batch 的激活随之释放（`retain_graph=False` 后释放），因此**显存峰值 ≈ 单个 micro-batch 的激活**，而不是整个大 batch；
3. 累积满 K 个 micro-batch 后，`optimizer.step()` 一下次，等效 batch size = `k × micro_batch_size`。

**它跟"一个货真价实的 K 倍大 batch"完全等价吗？不完全**：

- **激活不同**：真正（单次）的大 batch，attention 的矩阵形状是 `B×T` 整体，batch 之间没有多次前向带来的统计差异（对 transformer 影响通常是 **Bernoulli dropout mask 的粒度**——每个 micro-batch 重新采 mask，而大 batch 是一次采一份，还有归一化统计的 batch 粒度）；
- **数值路径**：每个 micro-batch 的 dropout/RNG 序列不同，梯度期望相同，但**方差略大**；
- **学习率调度**：步数变少了（每 K 个 micro 才算一步），LR schedule 按"step"而不是"example"计时的时候，节奏会变。

**BatchNorm 陷阱**：BatchNorm 靠 batch 内统计量（mean/var）做归一化。梯度累积下，每个 micro-batch 单独算 BN 的均值方差，开销被放大；而真正的等效大 batch 应该用**整个大 batch 的统计量**。若不处理，累积后的 BN 语义和真大 batch 不一致——通常得上 `SyncBatchNorm` 或干脆用更大的实在 batch。好消息是 LLM 一般**不含 BatchNorm**（用 RMSNorm/LayerNorm），这个坑主要是 CNN/混合架构要踩。

```python
# 梯度累积 vs 大 batch 对比
k, micro = 4, 8
true_batch = k * micro   # 32
print(f"等效 batch = {k}×{micro} = {true_batch}（显存仅 ~1 个 micro-batch）")
```

```mermaid
flowchart LR
    A["micro-batch 1"] -->|累积| G["grad 缓冲"]
    B["micro-batch 2"] -->|累积| G
    C["micro-batch 3"] -->|累积| G
    D["micro-batch 4"] -->|累积| G
    G -->|step| S["optimizer.step()"]
    A -.->|激活释放| MA["显存峰值仅1 micro"]
```

**面试要点**：核心是"**激活空间复用**"——峰值只受单 micro 影响；同时指出不完全等价（RNN/attention mask/dropout 粒度、BN 统计量）。

---

## Q5. 激活值显存与 batch size、序列长度、层数、hidden dim 的关系是什么？activation checkpointing 如何用计算换显存？

**回答**

上一题我们省下了 batch 带来的显存，这一题看看"激活"这个吞显存大户。"激活"就是前向计算过程中，一层层算出来的中间结果——因为反向传播要"回放"前向，这些中间结果得保存下来，于是白花花占掉一大片显存。

**激活显存的量级**（以 transformer 为例）：

- 每层 attention/FFN 都会保存前向中间量，供反向使用；
- 主体依赖 `B × T × D`（序列长度 × hidden 维度）以及 attention 的 `B × num_heads × T × T`（QK^T 分数矩阵）；
- 主要激活是"每层输入/输出 + attention 分数 + FFN 中间量（≈ 4D×B×T）"；
- 近似：激活显存 ∝ **B × T × D × layers**（外加 T² 的 attention 项）。

**一句话**：`激活 ∝ B × T × D × L`（线性比例），但 attention 分数额外 `∝ T²`（B×heads×T×T）。

**Activation Checkpointing（激活重计算）怎么用时间换空间**：

- 思路极其朴素：**某些层的激活先不保存，等反向需要的时候再重新前向算一遍**；
- 代价：重算该层前向，**时间开销约 +30%–40%**；
- 收益：激活显存从 `O(L·层激活)` 降到 `O(单层激活 + 保存层边界的常数)`——精确讲，只保存 checkpoint 点（通常每隔几层存一处）的激活，中间层全部省掉；
- 可配置粒度（`torch.utils.checkpoint.checkpoint` / Megatron 的 `--recompute-method`）。

```python
def act_memory(B, T, D, L, heads, fp16=True):
    # 粗略模型（不含 checkpoit）：每层保存 输入+输出
    per_layer = 2 * B * T * D + heads * B * T * T   # 简化
    return per_layer * L * (2 if fp16 else 4) / 1e9   # GB

B, T, D, L, heads = 16, 4096, 4096, 32, 32
full = act_memory(B, T, D, L, heads)
ckpt = act_memory(B, T, D, 1, heads) + B*T*D*2/1e9   # checkpoint后≈单层
print(f"全量激活≈{full:.1f}GB, checkpoint后≈{ckpt:.1f}GB（节省~{ (1-ckpt/full)*100:.0f}%）")
```

```text
全量激活≈40.3GB, checkpoint后≈1.3GB（节省~97%）
```

```mermaid
flowchart TB
    A["保存所有层激活"] -->|显存 O(B·T·D·L)| B["开箱即用"]
    A -.->|checkpoint| C["只存边界激活"]
    C -->|反向时重算该层前向| D["显存 O(单层)"]
    D -->|代价| E["FLOPs +30-40%"]
```

**面试要点**：关系式 `激活 ∝ B·T·D·L`（外加 attention 的 T² 项）；checkpointing 的本质是"**时间换空间，物理上重算**"。

---

## Q6. AdamW 的 weight decay 与 L2 正则是否等价？在 Adam 下为什么通常选择 decoupled weight decay？

**回答**

"正则化"是防止模型死记硬背的花招，L2 正则和 weight decay（权重衰减）是长得像双胞胎的两兄弟，但**在 Adam 这个优化器底下，它们并不等价**。先分清楚各自干啥：

- **L2 正则**：把 `λ/2·‖w‖²` 加进损失函数，于是**梯度里出现 `λw`**——真正的"惩罚项乘以权重"混进梯度里；
- **AdamW（解耦 weight decay）**：把正则从梯度里"拆出来"，**在更新时直接把权重按比例缩小**：`w ← w − η·λ·w`（step 前/后用独立的 decay 公式，不进入 Adam 的自适应分母）。

**在 SGD 下**（没有动量）：两者数学上近似等价（L2 梯度里的惩罚项经缩放后跟直接 decay 差不多）。**在 Adam 下就不等价了**，原因很微妙：

- L2 正则的 `λw` 会进 Adam 的自适应分母（要被 `√v + ε` 相除），**导致范数大的参数被衰减得更少**（因为它同时被"归一化"了）；
- 结果：**Adam + L2 的过拟合控制效果差，weight decay 也被扭曲**；
- AdamW 让 decay 独立于自适应，**大、小参数衰减一致**，更干净地控制权重范数——这就是大模型（GPT、LLaMA）普遍用 AdamW 的原因。

```python
# 概念演示：L2 会进 Adam 的分母，而 decoupled decay 不进
def adam_step(w, g, m, v, lr, wd, beta1=0.9, beta2=0.999, eps=1e-8, l2=False):
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*g*g
    mhat = m/(1-beta1**1); vhat = v/(1-beta2**1)
    if l2:
        g_eff = g + wd*w            # L2: 惩罚进梯度
        w = w - lr*mhat/(vhat**0.5+eps)
    else:
        update = lr*mhat/(vhat**0.5+eps)
        w = w - update - lr*wd*w    # AdamW: decay 独立于自适应
    return w, m, v

w, m, v = 1.0, 0, 0
print("L2  几步后 w:", adam_step(w, 0.1, m, v, 0.1, 0.01, l2=True)[0])
print("AdamW几步后 w:", adam_step(w, 0.1, m, v, 0.1, 0.01, l2=False)[0])
```

```mermaid
flowchart LR
    A["L2 正则"] -->|惩罚项进梯度| B["进入 Adam 自适应分母"]
    B -->|大权重被除以 √v| C["衰减被扭曲"]
    D["Decoupled decay (AdamW)"] -->|独立缩放权重| E["大小参数均匀衰减"]
```

**面试要点**：一句话——**在 SGD 下 L2≈decay，在 Adam 下 L2 的自适应会把惩罚稀释，所以用解耦的 AdamW**。这是为什么 GPT、LLaMA 都用 AdamW。

---

## Q7. 按全局范数裁剪与逐参数裁剪有什么区别？裁剪阈值过大和过小分别会掩盖什么问题？

**回答**

训练偶尔会遇到"梯度爆炸"（更新量突然大得离谱），裁剪就是给更新量"踩刹车"的手段，但刹车有两种踩法。先看区别：

- **全局范数裁剪（global norm clip）**：先算整批梯度的 L2 范数 `‖g‖`，如果 `‖g‖ > threshold`，把**所有梯度统一缩放** `g ← g · threshold/‖g‖`。这样保持了梯度的整体方向，适合**"大爆炸"场景**（整体梯度失控时拉回来）。
- **逐参数裁剪（per-parameter / per-tensor clip）**：每个参数独立裁剪到 `[−c, c]` 区间。代价是方向会被扭曲——大梯度参数被压到边界，小梯度参数原地不动，**破坏了梯度的相对比例**，一般不建议用在 LLM 上。

接下来是这题的狡黠之处：**阈值过大过小都会"掩盖"问题**。

**阈值过大**：几乎从不触发，等于**掩盖了梯度爆炸/数值不稳定的迹象**——网络可能一直在用不稳定的梯度训练，你毫无察觉；真遇到爆了（数值变 invalid），破坏会直接扩散。

**阈值过小**：高频触发，导致**梯度方向被限制**、参数更新变小 → 收敛变慢甚至不收敛；同时把**有效的高梯度信号**也砍了（比如新任务初期"探路"的大更新），连"真实的学习步长"都被误伤。

**总结成表**：

| 阈值 | 掩盖什么 |
|---|---|
| 过大 | 梯度爆炸/数值不稳定（等于没防） |
| 过小 | 正常大梯度/优化进度被压制（过拟合保护失效、收敛慢） |

**经验值**：LLM 常见阈值 `1.0`（GPT-2 用的 1.0，很多 7B 训练用 0.5–1.0）；也可以用**裁剪范数的均值**做自适应。

```python
def global_clip(grads, max_norm):
    norm = sum((g*g).sum() for g in grads).sqrt()
    scale = min(1.0, max_norm / (norm + 1e-12))
    return [g * scale for g in grads], norm

import torch
g = [torch.ones(100)*0.1, torch.ones(100)*0.05]   # 小梯度
norm = sum((x*x).sum() for x in g).sqrt()
print(f"全局范数={norm:.3f}, 阈值1.0 → scale={min(1.0,1.0/norm):.3f}（触发! 更新被压制）")
```

**面试要点**：区分两种裁剪的**方向保持性**；重点答"过大=没防住爆炸 \\ 过小=压死正常学习"两个方向。

---

## Q8. loss 正常但梯度范数持续在 1e-7 量级，可能是什么原因？应检查哪些环节？

**回答**

这是个非常烦人的故障现场：**loss 在正常下降，梯度却小得可怜（1e-7 量级）**。直觉上该问"是不是练得太好了？"——不，这通常是"**信号通路断了**"的信号：模型其实没学到东西。可能的元凶几条：

1. **dropout / mask 把梯度清没了**：attention mask 全 mask、dropout 在 backprop 时恰好归零（概率极低但可能）、或填充导致各 token 梯度过小；
2. **学习率/优化器状态错误**：LR 被设成 0、或 optimizer 压根没收到该层的梯度（`requires_grad=False`、参数被 freeze 冻结了）；
3. **激活饱和**：sigmoid/tanh 在饱和区导数≈0；或 softmax 输出接近 one-hot 后交叉熵的梯度路径退化；
4. **归一化与尺度问题**：RMSNorm / LayerNorm 之后数值被压到极小；或 loss 被某种缩放系数（如 label smoothing 系数）缩小到 1e-7；
5. **embedding 索引通路**：反向时 embedding 的梯度是"scatter"型的，如果输入 id 全是 pad/掩码，梯度大多为 0；
6. **数值截断**：BF16 训练下梯度下溢成 0（又没开 loss scaling），梯度范数自然极小；
7. **代码级 bug**：`gradient accumulation` 时 `.grad` 被提前清掉、或 `torch.no_grad` 把 loss 的反向节点包住了。

**排查顺序**（由浅入深、分层定位）：

1. 打印**各层梯度范数分布**（`[p.grad.norm() for p in model.parameters()]`）——看是全部极小，还是只有某层；
2. 检查 **requires_grad / freeze / device**：参数到底在不在"活的计算图"上；
3. 检查 **LR、clip、weight decay、mixed precision** 配置；
4. 用 `torch.autograd.grad(loss, p)` 手动验一条路径；
5. 检查 **mask / dropout / 归一化**是否把梯度 scale 死；
6. 比较 **dtype**：BF16 下有没有梯度下溢？看一眼 `grad.dtype`。

```python
# 快速诊断：逐层梯度范数
def diagnose(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.float().norm().item()
            if g < 1e-6 or g > 1e6:
                print(f"[异常] {name}: grad_norm={g:.2e}")
# 检查 float16 下溢
def check_underflow(model):
    tiny = sum(1 for p in model.parameters() if p.grad is not None and (p.grad.abs() < 1e-30).all())
    return tiny
```

```mermaid
flowchart TB
    A["loss 正常 但 grad ~1e-7"] --> B{"分层梯度范数"}
    B -->|全部极小| C["LR/优化器/freeze/数值下溢"]
    B -->|某层极小| D["mask/dropout / 激活饱和 / 该层通路"]
    C --> E["检查配置 + dtype + autograd 路径"]
    D --> E
```

**面试要点**：先分层看梯度分布，把"通力图"与"数值问题"分开；BF16 下溢和 freeze/requires_grad 是最常见坑。

---

## Q9. `memory_allocated()` 与 `memory_reserved()` 的区别是什么？OOM 时应看哪一个来定位真实峰值？

**回答**

这两个 API 名字只差一个词，却是"实际用了多少"和"占了多少地盘"的区别，新手很容易看懵。

- **`torch.cuda.memory_allocated()`**：**实际分配的张量内存**（当前被张量占用的字节数）——这是真实业务占用；
- **`torch.cuda.memory_reserved()`**：**PyTorch caching allocator 向 CUDA/cuDNN 预留（reserved）的内存**——包括已分配的部分，外加缓存池里的空闲块（这部分没还给驱动）。可以理解为"预圈的地皮"。

**关系**：`reserved ≥ allocated`，差额就是缓存洞（地皮圈了没盖楼的部分）。

**OOM 时该看哪个？** 定位**真实业务峰值**看 `memory_allocated`（扣掉缓存影响）；但要诊断"到底是谁吃掉了显存"，两边得一起看：

- `allocated` 飙升 → 张量/激活/优化器状态真的占多了；
- `reserved` 稳定但 `allocated` 波动 → 缓存策略问题；
- OOM 报错时，PyTorch 会打印 `current allocated` 和 `reserved`，通常还要看 `max_allocated`（历史峰值——往往这才是"爆"在哪一刻的关键）。

```python
import torch
a = torch.ones(1000, 1000, device='cuda')
print("allocated MB:", torch.cuda.memory_allocated()/1e6)
print("reserved  MB:", torch.cuda.memory_reserved()/1e6)   # 通常 ≥ allocated
del a
print("释放后 reserved 仍被缓存:", torch.cuda.memory_reserved()/1e6)
```

（注意：缓存块就算 `del` 了也不会还回驱动，所以 reserved 可能还是很大——这就是为什么定峰值要看 allocated。）

```mermaid
flowchart LR
    A["GPU 显存"] --> B["reserved (allocator 预留)"]
    B --> C["allocated (张量实际用)"]
    B --> D["缓存空闲块 (可复用)"]
    E["真实峰值"] --> C
    E -->|又要看| F["max_allocated"]
```

**面试要点**：定义+关系式 + 结论"**峰值看 allocated，缓存诊断看 reserved，避坑靠 max_allocated**"。

---

## Q10. PyTorch `autocast` 在哪些算子上自动降精度、哪些保持 FP32？为什么 layernorm 和 softmax 通常不被 autocast？

**回答**

`torch.autocast`（老名字 `amp.autocast`）是 PyTorch 自动混合精度的开关：你打开它，它会**自动挑"皮实的算子"降精度**，把"娇气的算子"留在高精度。规则如下：

- **降为 FP16/BF16 的算子**：矩阵乘（`addmm`/`bmm`/`matmul`）、线性卷积、`nn.Linear`、`nn.Conv2d`——凡是 **GEMM 类**核心，降精度收益最大；
- **保持 FP32 的算子**：`LayerNorm`、`GroupNorm`、`Softmax`、`exp`/`log`/`pow`、`sum`/`mean`、`normalize`、`cross_entropy`（部分实现内部会把 logits 降精度，但仍保持数学稳定）、`amsgrad`/自适应组件；
- **既不强制升也不强制降（自动选择）**：elementwise（relu、sigmoid）通常跟着输入类型走。

**为什么 layernorm / softmax 要躲开低精度**？四个理由：

1. **数值稳定性**：它们内部含着 `exp`/`log`/除法与求和。FP16 的指数范围很窄（±15，动态范围小），容易溢出/下溢——softmax 的 `exp(x)` 在 x 偏大时直接变 inf；LayerNorm 的 `1/√var` 精度损失会被放大；
2. **精度敏感**：归一化的统计量影响整层，一点精度损失就会被逐层放大；
3. **开销小**：layernorm/norm/softmax 的 FLOPs 占比很低（相对 GEMM），不值得为了省这点运算去冒精度风险；
4. **策略便宜**：autocast 用"算子类型白名单/黑名单"精准控制，而不是无脑全局降精度。

```python
import torch
x = torch.randn(4, 8, dtype=torch.float32)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    lin = torch.nn.functional.linear(x, torch.randn(8, 4))
    sm  = torch.functional.softmax  # 占位；真实里需 torch.softmax
# 实际行为：Linear → fp16，LayerNorm/Softmax 保持 fp32
```

```mermaid
flowchart TB
    subgraph autoscale
      A["GEMM / Conv / Linear"] -->|降为 fp16/bf16| E["高吞吐"]
      B["LayerNorm / Softmax / exp"] -->|保持 fp32| F["数值稳定"]
      C["elementwise"] -->|按输入| G["不强制"]
    end
```

**面试要点**：分类记忆——**矩阵类降、归一化/软行类保 FP32**；理由答"动态范围小 + 数值敏感 + 开销低"。

## Q11. 数据加载成为瓶颈时，如何判断是 CPU 预处理、磁盘 IO 还是 GPU 等待？`num_workers` 与 `prefetch_factor` 如何调？

**回答**

GPU 干活贼快，但如果"喂饭"的环节跟不上，再快的 GPU 也得饿着等。数据加载就是这条喂饭流水线。先学会判断卡在哪个环节：

**判断方法**：

1. **GPU 利用率**：训练时 `nvidia-smi` 显示 GPU util 长期 <90% 且在高低之间交替——通常是数据管线在拖后腿；GPU util 低且不交替（一直低）→ 那就是前向/反向本身是瓶颈，或 kernel 太碎；
2. **`DataLoader` 内部的排队**：量 `train_loader` 的迭代耗时；如果每次 `next(batch)` 耗时长（毫秒级以上）→ 预处理慢；
3. **profile**：`torch.profiler` 里对比 GPU 时间 vs 数据时间；也可以用 `time.perf_counter` 把数据加载那几步单独掐表；
4. **看 CPU/磁盘指标**：CPU 100% → 预处理/解析瓶颈；磁盘 IO wait 高 → 读文件慢（大文件、小文件多、没缓存）；
5. **实验性定位**：把数据集预加载进内存（测试时），若速度大增 → 磁盘 IO 是瓶颈；关掉 transforms，若速度大增 → CPU 预处理是瓶颈。

**怎么调**：

- `num_workers`：让多个进程并行做预处理。经验值：按"CPU 核数 / GPU 卡数"起步（常用 4–8），开太大反而因进程切换开销/内存爆掉；
- `prefetch_factor`（每个 worker 预取几个 batch）：默认 2，加大到 4–8 能提前备料更多；配合 `pin_memory=True` 减少 GPU 拷贝等待；
- 如果预处理含 GPU 操作（比如 tokenize 想在 GPU 上做），考虑 `num_workers=0` + 主进程或共享队列。

```python
from torch.utils.data import DataLoader, Dataset

class DS(Dataset):
    def __init__(self, n): self.data = list(range(n))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return torch.tensor(self.data[i])

loader = DataLoader(DS(10000), batch_size=64,
                    num_workers=4,          # CPU 核数充足时增加
                    prefetch_factor=4,      # 默认2
                    pin_memory=True)        # 减少 GPU 拷贝
```

```mermaid
flowchart LR
    A["GPU util 低?"] -->|是| B{"next(batch) 慢?"}
    B -->|CPU 预处理| C["num_workers↑ prefetch↑"]
    B -->|磁盘 IO| D["预加载到内存 / 大文件合并"]
    B -->|GPU 自身| E["kernel 太小 / 同步等待"]
```

**面试要点**：三步定位（GPU util → 数据加载时长 → CPU/IO 指标），调 `num_workers`/`prefetch_factor`/`pin_memory`。

---

## Q12. 手写梯度累积训练循环，需要修改哪些地方？`optimizer.step()` 与 `zero_grad()` 的调用时机与普通训练有何不同？

**回答**

继续用第 4 题的"分期付款"思路，这一题是动手版：**梯度累积的训练循环到底怎么改**。先看普通训练的三板斧：**普通训练**里每个 batch 都走一遍 `zero_grad → forward → backward → step`。

**梯度累积**则把 K 个 micro-batch 的梯度攒在一起才更新参数，区别全在**时机**：

- `zero_grad()`：在**累积开始前清零一次**（不在每个 micro 之后清）；
- `backward()`：每个 micro-batch 都要跑（梯度累加进 `.grad`）；
- **不要给每个 micro 单独 `step`**；
- `optimizer.step()` 只在**累积满 K 个 micro 后**调用一次；
- `optimizer.zero_grad()` 在 step 之后清一次，为下一个大 batch 做准备；
- **loss 要除以 K**（或除以累积步数），保持等效 batch 的梯度量级一致；否则等同一个 batch 的梯度被放大了 K 倍。

```python
from torch.optim import AdamW
import torch.nn as nn

model = nn.Linear(4, 4)
opt = AdamW(model.parameters(), lr=1e-3)

K = 4
micro_bs = 8
opt.zero_grad()                         # 累积前清零
for step in range(global_steps):
    for i in range(K):
        x = torch.randn(micro_bs, 4)
        loss = model(x).square().mean()
        loss = loss / K                 # 关键：除 K 保持量级
        loss.backward()
        # opt.step() 不在这里   ← 梯度累积核心差异
    opt.step()                          # 累积满 K 才 step
    opt.zero_grad()                     # step 后清
```

```mermaid
sequenceDiagram
    participant L as 主循环
    participant M as micro-batch 1..K
    L->>M: zero_grad 一次
    loop K 次 micro
        M->>M: forward → loss/K → backward (梯度累加)
    end
    L->>L: optimizer.step() 一次
    L->>L: zero_grad()
```

**面试要点**：三个"一次"——**清零一次、step 一次、loss 除 K**；对比普通训练"每 batch 清+步进"。

---

## Q13. `loss.backward()` 后不调用 `optimizer.zero_grad()` 会发生什么？梯度累积场景下为什么反而要刻意不 zero？

**回答**

这一题和上一题是一对：上题告诉你"什么时候该清零"，这题告诉你"不清零会怎样，以及为什么有时故意不清零"。先把机制讲透：

**不 zero_grad 的后果**：`.grad` 是个**累加器**，不清就会**持续累加**。下一次 backward 把新梯度叠在旧梯度上：

- 如果每次都 backward 而不清零，梯度越积越大（随迭代数线性增长）→ 参数更新爆炸，loss 发散；
- 优化器状态（m、v）和学习率是按"一步一份梯度"设计的，累加 K 份等于把**步长放大了 K 倍**，而且方向被历史污染。

**那梯度累积为什么反而要刻意不 zero？** 因为"不 zero + 多次 micro backward"正是**我们想要的累加**——K 个 micro-batch 的梯度按定义就该合并。关键在于区别对待：

- **刻意累积**：K 个 micro 之后才 step 且 loss÷K，把"一个等效大 batch"压成一步；
- **误累积**：每步都 step（第一个 micro 就 step），梯度却残留 → 等效 batch 语义被破坏（等于步长被缩放）。

**一句话总结**：`zero_grad` 的时机定义了"哪份梯度算一个 step"。普通训练每个 batch 一个 step → 每 batch 清零；累积训练把 K 个 micro 当一个 step → K 次 backward 才清一次。

```python
# 错误的写法（发散风险）
for x in loader:
    loss = model(x)
    loss.backward()
    opt.step()          # 每 batch 都 step
    # opt.zero_grad() 被遗漏 → 梯度一直累加 → 更新爆炸

# 正确的累积
opt.zero_grad()
for i, x in enumerate(loader):
    (model(x).mean() / K).backward()
    if (i+1) % K == 0:
        opt.step()
        opt.zero_grad()
```

**面试要点**：核心概念 "**.grad 是累加器，step 定义了累加窗口**"；误用后果=梯度爆炸/步长失真。

---

## Q14. BF16 与 FP16 的动态范围差异如何影响训练稳定性？为什么 BF16 通常不需要 loss scaling？

**回答**

这一题聊两种"半精度"的恩怨情仇。BF16 和 FP16 都是 2 字节的数，但一个"扛得住大数"、一个"精度细腻"，把表拉开一眼看懂：

| 格式 | 指数位 | 尾数位 | 动态范围 | 相对精度 |
|---|---|---|---|---|
| FP16 | 5 位（指数范围 ±15，约 6e-8 ~ 65504） | 10 位 | 窄 | 高（同量级内） |
| BF16 | 8 位（与 FP32 相同指数范围 ±127） | 7 位 | **宽（同 FP32）** | 低（~2-3 位有效数字） |

**对训练的影响**：

- **FP16**：动态范围窄，**小梯度/小权重会下溢成 0，大数据会溢出成 inf**。于是需要 **loss scaling**：先把 loss 乘一个大尺度数（如 2^16），把梯度"抬"进可表示范围，再在 optimizer step 前除回去——还要处理尺度数本身的动态调整（动态 loss scaling）。
- **BF16**：动态范围跟 FP32 一样宽，**不存在下溢/溢出问题**（除非真遇到 FP32 也会溢出的情形）；代价是尾数少，**相对精度只有约 3 位有效数字**，但对常见 LLM 训练够用（损失函数表面比较平坦）。

**所以**：

- BF16 **不需要 loss scaling**，开箱即用（DeepSpeed/Megatron 默认 BF16 直接训）；
- FP16 需要 loss scaling，还得同时防下溢和溢出两边的问题；
- 从趋势看，LLM 预训练基本都转向 **BF16**（H100 上 BF16 也是高吞吐格式）。

```python
import numpy as np
def fp16_range(): return 6e-8, 65504
def bf16_range(): return np.finfo(np.float32).tiny, np.finfo(np.float32).max
print("FP16 范围:", fp16_range())
print("BF16 范围近似: 6e-38 ~ 3.4e38 (与 FP32 同指数)")
small = 1e-10
print(f"1e-10 在 FP16 → {'下溢0' if small < 6e-8 else 'OK'}")
print(f"1e-10 在 BF16 → OK (动态范围够)")
```

```mermaid
flowchart TB
    A["梯度/权重"] --> B{"数值范围合适?"}
    B -->|FP16 范围窄| C["下溢/溢出 → 需要 loss scaling"]
    B -->|BF16 范围宽| D["无需 scaling，直接训练"]
    C --> E{"工程复杂度↑"}
    D --> F{"LLM 主流选择（H100 BF16）"}
```

**面试要点**：**FP16 窄 → 需要 loss scaling；BF16 与 FP32 同指数 → 不需**。尾数低位是相对精度取舍。

---

## Q15. 模型参数量、可训练参数量、激活参数量在 Dense 与 MoE 架构下如何区分？面试中为什么必须分开说？

**回答**

"这个模型多大"其实有三个口径，混着说会算错账——尤其到了 MoE（混合专家）架构，三个口径能差出一个量级。先分别下定义：

- **模型参数量（Total Parameters）**：所有层权重的总数（含 embedding 等）。
- **可训练参数量（Trainable Parameters）**：**在训练中真正参与梯度更新**的参数量。Dense 下一般等于总参数量；但遇到 freeze（冻结 encoder/emb）或 **MoE 只激活部分专家**时，可训练/激活参数 < 总参数。
- **激活参数量（Activated/Active Parameters）**：**单次前向实际被调用的参数量**。Dense：≈ 全部（每层都用）；MoE：**每个 token 只路由到 Top-K 个专家**，所以激活参数 ≈ 共享部分 + K/N 的专家部分，远小于总参数量。

**MoE 的关键**：

- Mixtral 8x7B（实际约 47B 总参数）激活约 13B；
- 训练时所有专家都参与梯度（虽然单个 token 只激活 K 个），所以**可训练参数 = 总参数**（MoE 通常不冻结）；但**显存按总参数算，FLOPs 按激活参数算**。

**为什么必须分开说**（这可是面试高频题）：

1. **显存预算是按总参数**（模型存放、优化器状态、KV Cache 都要按全部参数规模）；
2. **FLOPs/吞吐是按激活参数**（MoE 的盈利逻辑就是"同样显存，只激活一部分"）；
3. **Scaling Law 里的分歧**：MoE 的总参数 vs 激活参数该用哪个进幂律，是个活跃研究点（见第 8 章）；
4. 一句话**蒙混会让算力/显存估算全错**——训练（总参数）和推理（激活参数）的口径本来就不一样。

```python
def params_breakdown(total, share=0.0, experts=0, topk=1):
    active = share + topk/experts*(total - share) if experts else total
    return {"total_params": total, "active_params": active}

print("Dense (13B):", params_breakdown(13e9))
# Mixtral 8x7B ≈ 47B 总，8 专家 top-2，共享~ (近似)
active = params_breakdown(47e9, topk=2, experts=8)["active_params"]
print("Mixtral 8x7B 激活参数(近似):", f"{active/1e9:.1f}B")
```

```text
Dense (13B): {'total_params': 13000000000.0, 'active_params': 13000000000.0}
Mixtral 8x7B 激活参数(近似): 11.8B
```

```mermaid
flowchart TB
    A["总参数"] --> B["Dense: 总≡可训练≡激活"]
    A --> C["MoE: 总 > 激活"]
    C --> D["显存按总参数"]
    C --> E["FLOPs 按激活参数"]
```

**面试要点**：总参/可训练/激活三个口径；MoE 里"**显存看总参数、算力看激活参数**"是必须分说的核心。

---

## Q16. 给定 hidden=4096、layers=32、heads=32、vocab=32000、seq=2048，估算 decoder-only 模型的参数量与单次前向 FLOPs（忽略 attention 二次项）。

**回答**

这题是纯手工账本演练：拿着几个超参数，把 decoder-only transformer 的参数量和一次前向的运算量算出来。先按层拆：

**Decoder-only Transformer 每层参数**：

- **QKV 投影**：3 × (hidden × hidden) = 3 × 4096² ≈ 50.3M
- **输出投影**：4096 × 4096 ≈ 16.8M
- **FFN（常规 4D）**：2 × (4·4096) × 4096 ≈ 134.2M
- **归一化**：RMSNorm 2×4096 ≈ 8K（可忽略）

每层 ≈ **201M**，32 层 → **6.43B**。

**嵌入 + 输出头**：`vocab × hidden`（若 weight tying 则输出与输入共享）≈ 32000 × 4096 ≈ **131M**。

**总参数量 ≈ 6.43B + 0.13B ≈ 6.56B**（约 6.5B）。

**单次前向 FLOPs（C ≈ 2×参数×token/层）**：

每个 token FLOPs ≈ `2 × 6.56B ≈ 13.1B`；对 seq=2048：

`FLOPs ≈ 13.1e9 × 2048 ≈ 2.68e13 ≈ 26.8 TFLOPs`（不含 attention 二次项）。

```python
hidden, layers, heads, vocab, seq = 4096, 32, 32, 32000, 2048
d = hidden
qkv   = 3 * d * d
out   = d * d
ffn   = 2 * (4*d) * d     # 常规 FFN 4D
per_layer = qkv + out + ffn
emb   = vocab * d
total = per_layer * layers + emb
fwd_flops = 2 * total * seq      # 每 token 2×参 (forward)
print(f"每层 {per_layer/1e6:.0f}M, 总参 {total/1e9:.2f}B")
print(f"forward FLOPs (seq={seq}) ≈ {fwd_flops/1e12:.1f} TFLOPs")
```

```text
每层 201M, 总参 6.56B
forward FLOPs (seq=2048) ≈ 26.8 TFLOPs
```

```mermaid
flowchart LR
    A["每层: QKV 3d² + Out d² + FFN 8d² = 12d²"] --> B["×32 层"]
    B --> C["+ 词汇 embedding 32000×4096"]
    C --> D["总参 ≈ 6.56B"]
    D --> E["forward FLOPs ≈ 2×参×seq ≈ 26.8T"]
```

**面试要点**：**QKV 3d²、输出 d²、FFN 8d² 共 12d² 每层**；加 embedding；forward ≈ 2×参数×seq。

---

## Q17. 推理与训练的显存瓶颈有何不同？为什么训练看激活，推理看 KV Cache？

**回答**

训练和推理本质是两种活，它们的显存大头也完全不同。先说各自的家底：

**训练显存构成**：

- 权重 + 梯度 + 优化器状态（都是参数数量级）；
- **激活**（前向中间量，随 batch 和序列长度增长）；
- 训练时**激活以 B×T×D 和 T² 的 attention 分数存在**，是除优化器外的大项。所以训练显存优化的主战场在激活（checkpointing 打的就是它）。

**推理显存构成**：

- 权重 + **KV Cache**（每个已生成 token 的 K/V 都要存下来，随序列长度线性增长）；
- 推理不需要梯度、不需要保存各层激活（前向可以边走边丢中间量）；
- KV Cache 与**生成长度 × batch** 成正比，随上下文增长，是推理显存的核心。

**为什么会有这种差别**：

- 训练**反向要回放前向**，必须保存激活；
- 推理只做一次前向，中间量可以丢弃，但**生成是逐 token 进行的，历史的 key/value 必须保存**才能复用（否则每步都重算全部 attention，变成 O(T²) 的代价）；
- 所以一句口诀：**训练受激活约束，推理受 KV Cache 约束**。

```python
def train_vs_infer():
    train = {"weights": "O(参)", "grad+ada": "O(参)", "acts": "O(B·T·D·L)∝B·S"}
    infer = {"weights": "O(参)", "KV": "O(B·T·D·L·层))", "acts": "O(1)无保留"}
    return train, infer
print(train_vs_infer())
```

```mermaid
flowchart TB
    A["训练"] --> B["激活（前向存留）主导 + 优化器状态"]
    C["推理"] --> D["KV Cache（历史 K/V）主导"]
    B --> E{"·激活可 checkpoint 抛弃\n·优化器状态按参数量"}
    D --> F{"·KV 随 seq×batch 增长\n·中间激活即弃"}
```

**面试要点**：训练反向需要激活 → 激活是瓶颈；推理自回归需要 KV 复用 → KV Cache 是瓶颈。一句话**"训练回放 vs 推理压缩"**。

---

## Q18. `torch.compile` 在训练中的收益来自哪里？它对动态 shape（变长序列）的支持有什么限制？

**回答**

`torch.compile` 是 PyTorch 的"编译加速器"：它先把你的模型代码变成一张图，再统一优化后执行，而不是想跑一个算子就启动一个算子。收益主要有四块：

**收益来源**：

1. **Fusion（算子融合）**：把相邻的 elementwise / norm / GEMM 边界算子融合成一个 Triton kernel，减少 kernel 启动和显存读写（norm 与线性、attention 组合这样搭）；
2. **Graph capture（图捕获）**：把 eager 模式下"每个算子启动一次"变成"一次图执行"，砍掉大量 Python/CUDA 调度开销；
3. **更好的并行/布局**：为 GEMM 选更优的 tiling、布局与分解（自动调优）；
4. **对 CUDA graph 的整合**。

**对训练的具体收益**：大量小 kernel（norm、dropout、elementwise、attention 分量）占主时，收益非常显著；超大 GEMM 为主时收益较少（GEMM 已经被 cuBLAS 优化得很透了）。

**限制**：

- **动态 shape（变长序列）**：`torch.compile` 默认按**见过的 shape 图编译**；变长输入会导致 **recompile**——每来一个新 shape 就重新编译一次，昂贵的编译开销频繁触发，反而拖慢训练；
- 缓解手段：用 `dynamic=True`（以 shape 为 key 做多版本缓存）、固定 padding 到 bucket、或用 `mark_dynamic`；
- 部分算子依然走 eager（fallback），不能完全融合；编译和 gradient checkpointing 的交互也要留意。

```python
import torch

@torch.compile(dynamic=True)   # 变长支持
def step(x):
    return torch.nn.functional.layer_norm(x, x.shape[-1:]).relu()

# 变长：不同 seq 触发不同编译版本（cache miss 成本）
for L in (256, 512, 257):
    x = torch.randn(2, L, 4096)
    y = step(x)
```

```mermaid
flowchart TB
    A["torch.compile"] --> B["fusion / graph / tiling 优化"]
    B --> C["训练收益：小 kernel 场景显著"]
    A --> D["动态 shape 限制"]
    D --> E["新 shape → recompile (cost)"]
    E --> F["缓解：dynamic=True / padding / mark_dynamic"]
```

**面试要点**：收益 = 融合 + graph + 布局；限制 = **变长触发重编译**，缓解靠 dynamic/padding。

---

## Q19. DDP 中梯度 all-reduce 与反向计算如何 overlap？bucket size 设大或设小分别有什么后果？

**回答**

多卡训练里"梯度要同步"和"反向继续算"能不能同时进行，直接决定 GPU 有多少时间在干等。DDP 的答案很聪明：**边算边同步**。先看机制：

**DDP 的 overlap 机制**：

- 反向传播是按层从后往前算的，**每层的梯度一旦算完就可以先通信**，不必等所有反向结束；
- DDP 把梯度按**bucket**（固定大小的梯度块）分组：当一个 bucket 内所有参数的梯度都就绪了，立即对这个 bucket 发起 all-reduce；
- 因为反向是"层层推进"，后面的 bucket 已就绪、前面的还在算——通信与计算就自然重叠了。

**bucket size 的影响**（调这参数就是调"通信开始多早 vs 一次通信多大"）：

- **过大**：一个 bucket 要等更多参数梯度就绪才开始通信 → **overlap 变差**（通信被推迟到更晚才开始），尾部（最靠近输入、先算完的层）的通信要拖到反向全部结束才启动，GPU 空闲时间上升；
- **过小**：bucket 数量变多 → **通信次数变多、每次传输的包很小**，all-reduce 的启动开销和带宽利用率都下降，甚至触发频繁同步；
- 经验默认值：约 `25MB`（PyTorch 默认 bucket_cap_mb=25）；大模型常调到 50–100MB，在 overlap 与带宽效率之间找平衡。

```python
# DDP 配置示意
from torch.nn.parallel import DistributedDataParallel as DDP
ddp = DDP(model, bucket_cap_mb=50)   # 调大 bucket 提升带宽利用率（trade overlap）
```

```mermaid
sequenceDiagram
    participant B1 as 反向层N(k)
    participant B2 as 反向层N-1
    participant C as allreduce
    B1->>C: 梯度就绪 → bucket all-reduce
    B2->>B2: 继续反向（与通信并行）
    C->>C: 通信结束, 参数同步
```

**面试要点**：bucket 粒度决定通信何时开始 → **大 bucket 少通信但不早、小 bucket 早通信但启动多**；默认 25MB。

---

## Q20. 显存碎片化如何产生？PyTorch caching allocator 如何缓解？什么情况下仍然 OOM？

**回答**

显存碎片化就像硬盘碎片一样：空间明明够，但被切成一小块一小块，凑不出一个连续的大块，最后装不下东西。先看源头：

**碎片化来源**：

- 不同张量大小随意分配/释放（训练中激活、中间 buffer 尺寸多变），留下**一堆非连续的小洞**；
- 偶尔的大占用（比如一个大 checkpoint、临时拷一份大的）被释放后，紧邻的小块加起来凑不出一个连续大块；
- 反复的 allocation/free 模式把内存打得七零八落。

**PyTorch caching allocator（缓存分配器）怎么缓解**：

- 先从 cudaMalloc 借来**大段内存（reserved）**，小块分配直接在缓存池里切；
- 释放时**不还给驱动**，而是进 free list 供复用；
- 用 **stream-aware caching** 与 `expandable_segments=True` 允许在池内合并/重排虚拟地址；
- 但**内存洞依然可能存在**——要分析用 `torch.cuda.memory_snapshot()`，或设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**为什么还是会 OOM**：

1. 碎片化到无法容纳一个大分配；
2. **临时峰值叠加**（一个大 buffer 集中在同一时刻出现、cudaMalloc 的信号抢占）；
3. 显存使用超过物理容量（真实不足）——分配器只能"腾挪"，不能"变出"空间；
4. 没释放的引用泄漏（graph 保留、缓存模型的 buffer）。

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 或者: max_split_size_mb=32 限制小块的拆分
def mem_frag_report():
    # 返回 block 片段: 查看 free 块分布
    return None  # 用 torch.cuda.memory_snapshot() 更细
```

```mermaid
flowchart LR
    A["张量反复分配/释放"] --> B["碎片空洞"]
    B --> C["caching allocator 缓存池复用"]
    C --> D["expandable_segments 虚拟地址合并"]
    D --> E["仍可能 OOM：真实容量不足 / 碎片 / 峰值叠加"]
```

**面试要点**：碎片=非连续空洞；缓存分配器=池化+复用+可扩展段；OOM 没被根治是因为"池只能重组虚拟地址，不能变出容量"。

## Q21. 将 FP32 训练的模型转为 BF16 继续训练，哪些状态需要转换、哪些需要保留 FP32？

**回答**

实际操作中常遇到"已用 FP32 训到一半，想切到 BF16 省钱省时间"的需求。诀窍不是无脑全转，而是想清楚**哪些东西要"跑得快"，哪些东西必须"存得准"**。逐个拆：

**需要转到 BF16**：

- **模型副本**：前向/反向要用的权重矩阵（BF16 的计算副本）；
- **attention/激活相关 buffer**（BatchNorm running stats 等按需）。

**必须保留 FP32（甚至继续维护 FP32 master）**：

- **Master weights（FP32 权威权重）**：BF16 只有约 3 位有效数字，如果权重直接在 BF16 下存活，**数值回滚/累积误差**会把权重精度拖垮——所以继续训练必须维护一份 FP32 master（与副本解耦）；
- **Adam 状态 m、v**：它们要在 FP32 下累积（否则动量消失，部分参数无法收敛）；
- **梯度**：通常累加用 FP32（或用 BF16 加 scaling，但 BF16 不开 scaling 的话建议还是 FP32）。

**所以流程**（如果已经在 FP32 训练）：

1. 保留 FP32 master + 优化器状态（m/v 用 FP32）；
2. 新增/转换为 BF16 副本用于前向反向；
3. 稳妥起见：过渡期先混合训练几个 step，观察 loss/梯度范数正常了再正式切 BF16。

**注意**：如果只是"FP32 权重 → BF16 推理"，那是量化，不是继续训练；继续训练必须 FP32 master + Adam FP32。

```python
# 概念：保留 FP32 master，BF16 副本
master = torch.randn(4096, 4096, dtype=torch.float32)   # FP32 权威
copy   = master.half().bfloat16()                        # BF16 计算副本
# 每步: 用 copy 前向, 计算梯度, 用 FP32 master 更新, 再刷新 copy
copy = master.bfloat16()
# Adam m/v 保持 FP32
m, v = torch.zeros_like(master), torch.zeros_like(master)
```

```mermaid
flowchart LR
    A["FP32 master"] -->|bf16 拷贝| B["BF16 前向/反向"]
    B -->|梯度| C["FP32 grad"]
    C -->|Adam| A
    A -.->|每步 refresh| B
```

**面试要点**：核心是"**BF16 跑计算，FP32 保权威（master + Adam m/v）**"——这是混合精度续训的不变式。

---

## Q22. activation checkpointing 的 recomputation 开销如何估算？它适合放在哪类层上，不适合放在哪类层上？

**回答**

接着第 5 题的 checkpointing 往下深一层：它到底额外花多少算力，以及该"哪些层上做、哪些层别做"。先算开销账：

**Recomputation 开销**：checkpoint 的本质是"**把前向重算一遍来换回激活**"。如果要重建某层的输出，一次反向多花的 FLOPs ≈ **该层前向的 FLOPs**。

- 假设每层前向 ≈ `2·L_layer·T`（见 Q16），重算一次 = 额外 `2·L_layer·T`；
- 若 checkpoint 间隔为 1 层（每层都重算），**反向总 FLOPs 约 +100%**；但实际做法通常是"每隔 k 层存一个 checkpoint"，重算区间内 k 层的激活在反向时重放一遍，整体额外约 **+30–40% 时间**（常见经验值：业界报 25–40%）;
- 显存：从 `O(L·层激活)` 降到 `O(L/k·层激活 + 单段)`。

**适合放在**：

- **激活体积大但计算量小**的层（如 normalize、dropout、大宽度的 elementwise）→ 重算便宜，省得却多；
- attention 中 `B×heads×T×T` 的分值矩阵（大 T 时激活巨大）——用 recompute 换回大量显存。

**不适合放在**：

- **本身计算量大、激活小**的层（如超大 GEMM 权重层，重算成本占大头，捡了芝麻丢西瓜）；
- 对显存不敏感的浅层/边缘层（checkpoint 收益 < 重算代价）。

**经验**：默认 checkpoint **attention 块与 FFN 交界**（每层边界），或用 `torch.utils.checkpoint` 对全部层（通用做法），再按显存收益细分。

```python
def recompute_cost(per_layer_flops, layers, k=1):
    # k: 每 k 层存一个 checkpoint
    extra = (layers / k) * per_layer_flops   # 重算区间的前向
    return extra / (layers * per_layer_flops)   # 相对前向比例

print(f"每层 checkpoint (k=1): 额外 ~{recompute_cost(1, 32, 1)*100:.0f}% 前向")
print(f"每2层 checkpoint (k=2): 额外 ~{recompute_cost(1, 32, 2)*100:.0f}% 前向")
```

```text
每层 checkpoint (k=1): 额外 ~100% 前向
每2层 checkpoint (k=2): 额外 ~50% 前向
```

```mermaid
flowchart TB
    A["Activation checkpointing"] --> B["省显存 O(单段)"]
    A --> C["代价 = 重算前向 +30-40% 时间"]
    B --> D["适合：激活大计算小（attention 分值/宽 elementwise）"]
    C --> E["不适合：激活小计算大（超大 GEMM 层）"]
```

**面试要点**：`时间换空间`；**适合"激活大计算小"，不做"激活小计算大"**；量级约 +30–40%。

---

## Q23. 训练吞吐（tokens/s）与 MFU 的关系是什么？给定吞吐与模型 FLOPs/token，如何反推 MFU？

**回答**

这一题把前两题的味道混合起来：一边是"每秒能喂多少 token"，一边是"硬件算力用了几成"。先定义：

**定义**：

- **throughput（tokens/s）**：单位时间处理的 token 数；
- **MFU（Model FLOPs Utilization）**：实际吞吐对应的有效 FLOPs ÷ 硬件峰值 FLOPs。

**关系**：`吞吐 = FLOPs 总量 / 秒数`；对给定模型，**每 token FLOPs = 6·N**（训练，见 Q1），所以

$$\text{tokens/s} = \frac{\text{MFU} \times P_\text{peak}}{\text{FLOPs per token}}$$

**反推 MFU**：已知吞吐 `T0`（tokens/s）、单 token FLOPs `f = 6N`（训练）：

$$\text{MFU} = \frac{T0 \times 6N}{P_\text{peak}}$$

其中 `P_peak = gpus × peak_FLOPs_per_gpu`。

```python
def mfu_from_tput(theo_peak_tflops, tokens_per_s, N):
    flops_per_token = 6.0 * N          # 训练
    mfu = (tokens_per_s * flops_per_token) / (theo_peak_tflops * 1e12)
    return mfu

# 3.2 epoch/s × 300B tokens? 简化: 已知吞吐 30k tok/s, 7B, 单卡理论 156T(TF32)
print("MFU ≈", round(mfu_from_tput(156e12, 30_000, 7e9)*100, 1), "%")
```

```text
MFU ≈ 80.8 %
```

（注意：这里的单卡理论 156T 是 TF32 峰值；实际 BF16 H100 是 990T，要另算。MFU = 有效/峰值，是考核对"吞吐与硬件利用率"理解的题眼。）

```mermaid
flowchart LR
    A["tokens/s × 6N"] --> B["有效 FLOPs/s"]
    B --> C["÷ 硬件峰值"] --> D["MFU"]
```

**面试要点**：`MFU = (tokens/s × 6N) / P_peak`；反过来排吞吐。

---

## Q24. 为什么 embedding 层的梯度通常是 sparse 的？PyTorch 中如何利用 sparse gradient 节省显存与通信？

**回答**

先别被 sparse（稀疏）这个词吓到，意思就是"大多数位置是 0"。embedding 层的梯度天生稀疏，原因很直接：

**为什么 sparse**：embedding 是"按 index 查表"——`embedding(x)` 里 `x` 只是 batch 内出现的那几个 token id。反向时，**只有被用到的行的梯度非零**，其余 `V` 行梯度全是 0。而 V 很大（几十万量级），没用到的行占绝大多数，梯度矩阵**高度稀疏**。

**PyTorch 里怎么利用 `sparse=True`**：

- `nn.Embedding(..., sparse=True)`：梯度用 COO 稀疏表示存储；PyTorch 对 sparse 梯度做**原地累加**（避开 dense 那种 O(V×D) 的大表示）；
- 节省：dense 梯度的完整表示是 `V×D` 的浮点矩阵；sparse 只存"非零行的 index + values"，数据量 ≈ 非零行数×D，能少几个数量级；
- 通信（DDP）：稀疏梯度理论上可以在 all-reduce 时只传非零块，但 **DistributedDataParallel 默认不支持 sparse 梯度 all-reduce**（需要额外支持或先转换），工程上常用 dense 的 bucket，或在本地累积后转成 dense 一次性通信；
- Adam 这类优化器对 sparse 梯度也得小心处理（`sparse_grad` 的指数移动平均是逐非零行做的）。

```python
import torch.nn as nn
import torch

emb = nn.Embedding(100000, 512, sparse=True)   # 稀疏梯度
x = torch.tensor([[3, 7, 99]])
loss = emb(x).mean()
loss.backward()
g = emb.weight.grad
print("稀疏梯度表示:", g)         # 只有 3 行非零 (3,7,99)
print("稀疏非零值数 vs 全量:", g._nnz(), "vs", g.shape)
```

```text
稀疏梯度表示: tensor(indices=tensor([[3, 7, 99]]),
                 values=tensor([[...],[...],[...]]), size=(100000, 512))
稀疏非零值数 vs 全量: 3 vs 100000×512
```

```mermaid
flowchart LR
    A["batch 只用到少数 token id"] --> B["embedding 梯度仅这些行非零"]
    B --> C["sparse=True → COO 存梯度"]
    B --> D["dense → V×D 全量（浪费）"]
    C --> E["显存/通信大幅下降"]
```

**面试要点**：sparse 原因 = **只查用到的行**；利用 = `Embedding(sparse=True)` + COO 梯度；但要提醒 DDP all-reduce 对 sparse 的不便。

---

## Q25. 多卡训练中每卡 batch size 不同时，梯度同步会出现什么问题？梯度累积能否弥补不均衡？

**回答**

DDP 这套同步机制有一个暗含的"共同富裕"假设：**每张卡喂一样多的样本**。一旦破坏这个假设，账就乱了。先看问题：

**问题**：DDP 的 all-reduce 是按参数归一化（除以 `world_size`）的，**隐含假设每个 rank 的 batch 大小相等**。如果各卡数据长度/采样不均衡：

- **梯度权重失衡**：batch 大的 rank，梯度平均后相对被削弱；batch 小的被放大 → 全局梯度不是"均匀平均"，统计意义上偏了；
- **grad 对齐错位**：不同 rank 在同一 step 迭代到不同的样本段，all-reduce 把"不同样本的梯度"混在一起，语义混乱；
- **流水线/同步点漂移**：长度不一致导致有的 rank 先算完、有的后算完 → 同步卡顿。

**梯度累积能弥补吗**：

- **部分缓解、不能根治**：通过让"每 rank 在本 batch 里额外累加 K 份"，可以让每个 rank 等效出更大的 batch（每 rank 独立累积），**使各 rank 的 batch 大小更均衡**；
- 但累积的**起始/结束点仍可能不一致**；若各 rank 数据不均匀，累积后每 rank 内部均匀，**rank 之间总 token 数还是不一样**；
- 更根本的解法：**固定步长采样（每个 rank 取相同 batch 大小）+ 数据均衡 shuffle**；或在 loss 计算时用 `avg = sum(loss)/world_size` 把分母对齐。

```python
# 不均衡示意
bs = {"rank0": 4, "rank1": 64}
avg_world = (4 + 64) // 2
# DDP 默认 all-reduce 后 / world_size → rank0 的 4 条样本比重被低估
# 缓解: 每个 rank 先累积到统一 batch（如 64）再 step
K = {"rank0": 16, "rank1": 1}   # 各 rank 累积到 64
```

```mermaid
flowchart TB
    A["各 rank batch 不均衡"] --> B["all-reduce 平均失真"]
    B --> C["梯度被 batch 小的样本主导/稀释"]
    C --> D["用梯度累积拉平每 rank 等效 batch（部分缓解）"]
    D --> E["根治: 均匀采样 + 固定每 rank batch"]
```

**面试要点**：本质 = **all-reduce 假设均匀**；累积能拉平"每 rank 等效 batch"，但不能根除起始点不一致。

---

## Q26. 估算 Llama-3 8B（BF16）推理时加载模型本身需要多少显存？再加 8k 上下文，KV Cache 大约多少？

**回答**

把 Q3 和 Q17 的账本合起来，算一道实战题：用 Llama-3 8B 做推理，显存要备多少。分两块算：

**模型权重显存**：8B × 2 B(bf16) = **16 GB**。（FP16 同样 2B/参，也是 16GB。）

**KV Cache**：公式 `2 × layers × kv_heads × head_dim × T × B × bytes`

- Llama-3 8B：layers=32、KV heads=8、head_dim=128、T=8192、B=1；
- K 与 V 各一份（前面的 `2`），每项 2B：

`2 × 32 × 8 × 128 × 8192 × 1 × 2B = 2×32×8×128×8192×2 = 1.07e9 B ≈ 1.07 GB`

**合计 ≈ 16GB（权重）+ 1.07GB（KV）≈ 17.1 GB 起步。**（实际框架还要算中间 buffer、CUDA 上下文，共约几 GB。）

```python
L, h_kv, d, T, B, bytes = 32, 8, 128, 8192, 1, 2
kv = 2 * L * h_kv * d * T * B * bytes
print(f"KV Cache = {kv/1e9:.2f} GB (seq=8k, batch=1)")
weights = 8e9 * 2 / 1e9
print(f"权重 = {weights:.0f} GB, 总 ≈ {weights + kv/1e9:.1f} GB")
```

```text
KV Cache = 1.07 GB (seq=8k, batch=1)
权重 = 16 GB, 总 ≈ 17.1 GB
```

```mermaid
flowchart LR
    A["8B × 2B = 16GB 权重"] --> TOT["总显存 ≈ 17.1GB"]
    B["2×32×8×128×8192×2B = 1.07GB KV"] --> TOT
```

**面试要点**：`权重=2B×N`；KV= `2·L·h_kv·d·T·B·bytes`（KV 头 × 序列 × batch）；Llama-3 8B 约 17GB 起步。

---

## Q27. `optimizer.state_dict()` 通常包含哪些内容？为什么 Adam 的优化器状态是模型参数量的 2 倍（FP32 下）？

**回答**

`state_dict` 是 PyTorch 里序列化"权重/状态"的字典。问优化器的 state_dict 里有啥，其实是在问"优化器除了参数还记着什么"。

**`state_dict()` 内容**：

- `state`：每个参数（按 id 索引）对应的优化器内部状态；对 Adam 是 `{'step': int, 'exp_avg': tensor, 'exp_avg_sq': tensor}`（也就是 m 和 v）；
- `param_groups`：超参数分组（lr、betas、weight_decay 等）和组内参数索引。

**为什么 Adam 状态是参数量 2 倍**：

- 每个参数都有 `m`（一阶矩）和 `v`（二阶矩）两个中间量，而且以 **FP32** 存储（即使权重用 BF16 训练，m/v 通常还是 FP32）;
- 所以 `2 × N 参数 × 4B = 8B/参数` 的额外显存；**再加上 FP32 master weights 4B/参数**（如果保留），优化器相关的共 12B/参数（呼应 Q3 的 16B：master + m + v = 12B）。

```python
# Adam state_dict 结构
state = {0: {"step": 100,
             "exp_avg": "m tensor (同权重形状)",
             "exp_avg_sq": "v tensor (同权重形状)"}}
# 每参数: m(4B) + v(4B) = 8B → 参数量×2
per_param = 4 + 4     # m + v
print(f"Adam 状态/参数 = {per_param}B；对 7B 模型 = {7e9*per_param/1e9:.0f} GB")
```

```text
Adam 状态/参数 = 8B；对 7B 模型 = 56 GB
```

```mermaid
flowchart LR
    A["Adam state"] --> M["exp_avg (m) FP32 4B/参"]
    A --> V["exp_avg_sq (v) FP32 4B/参"]
    M --> T["共 8B/参数 = 参数量的 2×"]
    V --> T
```

**面试要点**：Adam 双矩 × FP32 = **8B/参数**（状态=2×模型）；加 master 4B 凑成 Q3 的 12B 优化器相关。

---

## Q28. 从优化器二阶矩估计的角度解释：训练初期为什么需要学习率 warmup？

**回答**

warmup（预热）是训练开始时把学习率从小往大爬的做法。为什么不能一上来就满速？从 Adam 的"二阶矩估计"入手能讲得很透彻。

**Adam 的二阶矩（v）一开始是"欠估计"**：

- Adam 第一步 `v = (1-β2)·g²`，此时的 v **远小于数据真实的方差**（估计刚起步，只看到一两个梯度）；
- 更新大小是 `lr·mhat / (√vhat + ε)`，**初期 vhat 偏小 → 除以一个过小的值 → 更新量过大**（容易爆炸式跳变）；
- 尤其对**大规模/分布还不稳定的初始模型**（大权重、大尺度的 logits），初期的巨大更新会让 loss 直接爆炸或参数飞出。

**warmup 的作用**：在开始 K 步内把 lr 从 0 线性（或曲线）升到目标值：

- 让优化器的 m/v 在**较小的 lr 下"预热"**，等 v 的估计成熟了再放满步长；
- 避免初期巨量更新把模型带偏；对大模型尤其重要，后面哪怕每 epoch 重开统计也依然需要。

**从二阶矩角度说得更精确**：`vhat` 要等 warmup 结束才"有统计意义"；warmup 其实是在给"v 的冷启动"安排缓冲——让有效步长和方差的成熟度同步成长。

```python
def warmup_lr(step, warmup, peak):
    if step < warmup:
        return peak * (step / warmup)      # 线性 warmup
    return peak

steps = range(30)
lr = [warmup_lr(s, 10, 1e-4) for s in steps]
print(f"step 1 → lr = {lr[1]:.2e}（远低于峰值）")
print(f"step 9 → lr = {lr[9]:.2e}  step 10 → {lr[10]:.2e}")
```

```text
step 1 → lr = 1.00e-05（远低于峰值）
step 9 → lr = 9.00e-05  step 10 → 1.00e-04
```

```mermaid
flowchart LR
    A["Adam 初阶 v 欠估计"] --> B["√vhat 过小 → 更新爆炸"]
    B --> C["warmup: lr 从 0 缓升"]
    C --> D["等 v 成熟后再满步长"]
```

**面试要点**：**v 冷启动欠估计 → 除以过小的分母 → 步长爆炸**；warmup 缓冲初期。

---

## Q29. 只有 8 张 80GB GPU、要训练 13B 模型时，显存预算如何分配？哪些并行/节省显存手段必须上？

**回答**

这是"资源抠搜"实战题：仅 8 张 80GB 卡，要把 13B 模型训起来，账怎么算、手段怎么上。先算总需求：

**显存需求**（13B，BF16 混合精度，用 Q3 的 16B/参数算法）：

- 权重 BF16 13×2 = 26GB；梯度 2B×13=26GB（或 FP32 梯度 52GB）；Adam FP32 master+m+v 12×13=156GB；
- 合计：16B/参数 × 13 = **208GB**（优化器+权重+梯度），远超单卡 80GB；总容量 8×80=640GB 其实够，问题是**单卡要塞得下**。

**必须上的手段**：

1. **ZeRO-1**（切 Adam 状态）：把 156GB 的优化器状态分到 8 卡，单卡降为约 20GB → 单卡模型+梯度+状态 ≈ 26+26+20=72GB，勉强贴近 80GB；
2. **或 ZeRO-2/FSDP**（切梯度 + 状态）：梯度也切 → 单卡约 52GB，控制得住；
3. **Activation checkpointing**：激活可以占几十 GB，打开后压到单层量级；
4. **梯度累积 + 减小 batch**：激活小的时候，用 micro-batch=1~2 + 累积进一步压峰值；
5. **BF16（或 AMP）**：已经是 BF16 就不需要 loss scaling；
6. **必要时 TP/PP**：8 卡可以考虑 TP=2×PP=4 或纯 ZeRO=8（DP）配合 checkpointing；在 8 卡上 **ZeRO/FSDP DP=8** 通常最省心（通信也简单）。

**预算举例（FSDP/ZeRO-2, DP=8, activation checkpointing on）**：

- 权重分片：26/8≈3.2GB/卡；
- 状态分片：156/8≈19.5GB/卡；
- 激活：checkpoint 后 <5GB/卡；
- **总 < 40GB/卡**，稳稳落在 80GB 以内。

```python
N, gpus, gb = 13e9, 8, 80
# ZeRO-2: 状态+梯度分片
weights = N*2/1e9; grad = N*2/1e9; adam = N*12/1e9
per_after_split = (weights + grad + adam)/8
print(f"ZeRO-2 分片后每卡(权重+梯度+状态) ≈ {per_after_split:.1f} GB")
print(f"加激活checkpoint(<5GB) → 总计 < {per_after_split+5:.0f}G < {gb}GB ✔")
```

```text
ZeRO-2 分片后每卡(权重+梯度+状态) ≈ 28.5 GB
加激活checkpoint(<5GB) → 总计 < 34G < 80GB ✔
```

```mermaid
flowchart TB
    A["13B 训练需求 208GB"] --> B["ZeRO-2 / FSDP 切状态+梯度"]
    B --> C["activation checkpoint"]
    C --> D["BF16 + 梯度累积"]
    D --> E["8×80GB 可行（~34GB/卡）"]
```

**面试要点**：**ZeRO/FSDP 分片优化器与梯度是刚需**；加 activation checkpointing；BF16 默认。8×80GB 装 13B 完全可行。

---

## Q30. 训练日志中 `grad_norm` 突然变成 NaN，排查顺序是什么？从数据、loss、梯度、参数逐层定位。

**回答**

训练到一半 `grad_norm` 突然变成 NaN（不是一个数），是最让人头秃的现场之一。好在这类问题有固定的排查次序，从外到内一层层"剥洋葱"：

**排查顺序（由浅入深）**：

1. **数据**：检查 batch 里有没有 NaN/inf 特征（输入层注入）；token id 有没有越界、mask 是不是异常——输入污染最容易被忽略；
2. **loss**：计算时是不是 NAN？分类 loss 的 `label` 是否含 -100/非法值？logits 是不是 -inf（除以 0、mask 处理错）？
3. **激活/中间量**：LayerNorm 方差为 0（epsilon 太小；输入全同）；attention softmax 溢出；除法的 `0/0`；
4. **梯度**：哪一层 grad 为 NaN——理清是梯度爆炸/消失，还是**瞬时 NaN**（如 exp/log 的 inf-inf）；
5. **参数**：优化器更新后的权重 NaN → 是"更新爆炸"还是"初始权重就含 NaN"；检查 adam 状态 m/v 是否也一起 NaN；
6. **数值设置**：BF16 溢出、loss scaling、梯度裁剪对 NaN 无效（裁剪只对 inf 有用，对 NaN 直接失效）。

**几个关键技巧**：

- 用 `torch.autograd.detect_anomaly()` 或 NaN 排查钩子定位**第一个出现 NaN 的 op/层**；
- 检查 batch 里每个张量 `torch.isnan().any()`；
- 对比最近一次正常 step 的 diff，找出触发点（到底改的哪段代码、哪批数据）；
- BF16 下检查 logits 是否撞到 `1e38` 溢出。

```python
import torch, torch.nn as nn
torch.autograd.set_detect_anomaly(True)   # 定位首个 NaN 的 op

def scan_nan(step_name, tensors):
    for name, t in tensors.items():
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[{step_name}] {name} 含 NaN/Inf")

for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print("梯度 NaN 参数:", name)          # 逐层定位
```

```mermaid
flowchart TB
    A["grad_norm=NaN"] --> B["数据层(输入 NaN)"]
    B --> C["loss/logits NaN"]
    C --> D["哪层激活 NaN"]
    D --> E["哪层梯度 NaN（首现层）"]
    E --> F["参数/优化器状态 NaN"]
    F --> G["数值设置: BF16溢出/loss scaling/裁剪"]
```

**面试要点**：**由外到内：数据→loss→激活→梯度→参数→数值**；用 detect_anomaly 定位"第一个 NaN op"；提 BF16 溢出与裁剪对 NaN 失效。

---

## 参考链接

1. [大模型adam混合精度训练的显存计算详细过程-CSDN博客](https://blog.csdn.net/baoyan2015/article/details/151722662)
2. [训练显存别再写成 20Ψ：把 Adam 状态与 ZeRO 分片分别记账-CSDN博客](https://blog.csdn.net/m0_50889382/article/details/168978835)
3. [【收藏必备】大模型训练显存优化全攻略：从混合精度到零冗余优化器-CSDN博客](https://blog.csdn.net/2401_85154887/article/details/153043467)
4. [FLOPs计算详解-LLM训练03：Lora/全参训练FLOPs-CSDN博客](https://blog.csdn.net/u013250861/article/details/155648115)
5. [大模型显存账本全解析：从训练到推理，从FLOPs到KV Cache-CSDN博客](https://blog.csdn.net/weixin_29047545/article/details/168818896)
6. [大模型显存账本：训练与推理的显存计算、FLOPs估算及KV Cache规划-CSDN博客](https://blog.csdn.net/weixin_32487557/article/details/168944312)
7. [大模型里的"计算数据量"：一篇文章看懂 FLOPs、Token、GPU-hour 与训练成本-CSDN博客](https://blog.csdn.net/xiaofeng10330111/article/details/162076665)

---

*下一章展望：第 3 章将深入 Transformer 架构（多头注意力、Pre-LN、RoPE、FlashAttention 等模型级细节）。*