# GPU Profiling 与性能优化实战 —— rocprof、kernel trace 与 Roofline 图

> 面试生普（LLM 500 题）· GPU Profiling 与性能优化实战篇。本章覆盖 ROCm 性能分析工具链的实战使用：rocprofv3 热点分析（--stats --kernel-trace）、各类 tracing 选项（--kernel-trace/--hip-trace/--sys-trace/--marker-trace）、pftrace 与 Perfetto 可视化、公平对照实验设计、kernel trace 时间轴解读（kernel 执行、HIP API、内存拷贝、同步间隙）、hipDeviceSynchronize 的影响、Roofline 图原理（算术强度、峰值算力、峰值带宽、拐点/机器平衡点）、计算密集型 vs 存储密集型判断、硬件计数器收集（--pmc、VALUUtilization/VALUBusy）、occupancy 查看与优化、Profiling 优化闭环，共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Bash/Python 代码和 ASCII 图。

---

## Q1. 什么是 Profiling？为什么做 GPU 性能分析？

**回答**

Profiling（性能分析）就是“给程序做体检”，找出程序运行时哪里慢、哪里快、瓶颈在哪里。

对于 GPU 程序来说，光知道“跑了多久”还不够，我们需要知道：
- 是哪个 kernel 最耗时？（热点分析）
- kernel 里花了多少时间在算、多少时间在等内存？（算力 vs 带宽）
- GPU 核心忙不忙？（占用率、VALUBusy）
- 有没有不必要的等待？（hipDeviceSynchronize 间隙、内存拷贝阻塞）

没有 profiling，优化就是“瞎猜”；有了 profiling，优化就变成“有的放矢”。

大白话比喻：Profiling 像给汽车装行车记录仪 + 发动机诊断仪，告诉你到底是引擎没力，还是油管堵了，还是刹车一直踩着。

> 面试要点：Profiling 是找程序性能瓶颈的过程，GPU profiling 需要关注哪个 kernel 最耗时、算力与带宽的瓶颈、占用率、同步等待。没有 profiling，优化就是瞎猜。

---

## Q2. ROCm 自带哪些 Profiling 工具？它们分别干什么？

**回答**

ROCm 提供三类 Profiling 工具，分工不同：

| 工具 | 曾用名 | 主要用途 |
|------|--------|---------|
| **rocprofv3** | rocprof / rocprofv2 | 轻量级命令行 Profiling，做 kernel 热点分析、API tracing、硬件计数器收集，输出 CSV/JSON/pftrace |
| **rocprof-compute** | Omniperf | 专门做 kernel 级性能分析，生成 Roofline 图、收集硬件计数器、自动多次运行收集 counter |
| **rocprof-sys** | Omnitrace | 做 CPU+GPU 综合跟踪，适合分布式/MPI 场景，输出 perfetto 格式可视化 |

大白话：rocprofv3 像“快速体检仪”，rocprof-compute 像“专项体检仪”，rocprof-sys 像“全身CT”。

> 面试要点：ROCm 自带 rocprofv3（轻量命令行 Profiling）、rocprof-compute（kernel 性能分析 + Roofline）、rocprof-sys（CPU+GPU 综合跟踪）。初学者优先从 rocprofv3 开始。

---

## Q3. rocprofv3 和 rocprof-compute 有什么区别？怎么选？

**回答**

| 维度 | rocprofv3 | rocprof-compute |
|------|----------|----------------|
| 定位 | 轻量命令行 Profiling 工具 | 专门的 kernel 性能分析工具（前身 Omniperf） |
| 核心能力 | 统计热点、采集 trace、硬件计数器 | Roofline 图、kernel 对比、自动多次运行收集 counter |
| 输出 | CSV / JSON / pftrace（Perfetto 可视化） | Roofline HTML、kernel stats、counter 数据 |
| 适合场景 | 快速看哪个 kernel 最耗时、看时间轴 | 深挖单个 kernel 为什么慢、对比优化前后 |
| 运行次数 | 通常运行一次 | 自动运行多次（采集不同 counter 需要多次 pass） |

**怎么选？**
- 先用 rocprofv3 找“哪个 kernel 最慢”（热点定位）。
- 再用 rocprof-compute 深挖“这个 kernel 为什么慢”、画 Roofline 图看上限。

ASCII 图示（工具选择流程）：

```
怀疑程序慢
      │
      ▼
┌─────────────┐
│ rocprofv3   │──▶ 找热点 kernel（哪个最耗时）
│ (快速体检)  │
└──────┬──────┘
       │
       ▼ (锁定目标 kernel)
┌─────────────┐
│ rocprof-compute│──▶ 画 Roofline、看计算/存储瓶颈
│ (专项体检)   │    对比优化前后
└─────────────┘
```

> 面试要点：rocprofv3 负责“快速定位热点”，rocprof-compute 负责“深挖单个 kernel 并画 Roofline”。先用 rocprofv3 找最耗时的 kernel，再用 rocprof-compute 深挖原因。

---

## Q4. 怎么用 rocprofv3 做热点分析？（--stats --kernel-trace）

**回答**

rocprofv3 做热点分析最常用的命令是：

```bash
rocprofv3 --stats --kernel-trace --summary -- ./你的程序
```

各参数含义：
- `--stats`：收集统计信息（调用次数、总耗时、平均耗时等）。
- `--kernel-trace`：记录 kernel 调用的时间轴。
- `--summary`：在控制台打印汇总摘要。
- `-- ./你的程序`：`--` 后面跟你要分析的程序及其参数。

运行后，控制台会输出类似下面的热点表：

```
ROCPROFV3 SUMMARY:
NAME                                      CALLS  DURATION(ns)  AVERAGE(ns)  PERCENT(INC)
JacobIterationKernel                      1000   527366199       527366       48.71
NormKernel1                               1001   418842848       418842       31.72
LocalLaplacianKernel                      1000   329779388       329779       25.47
HaloLaplacianKernel                       1000   15328759        15328        1.18
```

各列含义：
- **NAME**：kernel 名字。
- **CALLS**：调用次数。
- **DURATION (ns)**：总耗时（纳秒）。
- **AVERAGE (ns)**：平均每次耗时。
- **PERCENT (INC)**：占总时间的百分比。

看这个表，你就能一眼看出哪个 kernel 最耗时（上面的例子里，JacobIterationKernel 占了 48.71%，是第一优化目标）。

> 面试要点：`rocprofv3 --stats --kernel-trace --summary -- ./程序` 是 rocprofv3 最常用的热点分析命令。输出表里按 PERCENT(INC) 排序，排在最前面的 kernel 是第一优化目标。

---

## Q5. rocprofv3 的常用 tracing 选项有哪些？

**回答**

rocprofv3 的 tracing 选项决定“记录什么事件”。常用的有：

| 选项 | 记录什么 |
|------|---------|
| `--kernel-trace` | GPU kernel 调用（启动、开始、结束时间） |
| `--hip-trace` | HIP API 调用（hipMalloc、hipMemcpy、kernel launch 等） |
| `--sys-trace` | HSA + HIP + Kernel + Memory Copy + Marker（综合） |
| `--marker-trace` | 用户代码里插入的自定义标记（ROCTx） |
| `--memory-copy-trace` | 主机 ↔ 设备的内存拷贝 |

**常用组合：**
- 只想看 kernel 花了多久：`--kernel-trace`
- 想看 HIP API 和 kernel 都发生了什么：`--sys-trace`（最全面）
- 想看内存拷贝开销：`--hip-trace` 或 `--memory-copy-trace`

输出格式建议用 `--output-format pftrace`，这样可以用 Perfetto（ui.perfetto.dev）做可视化时间轴。

示例：

```bash
rocprofv3 --sys-trace --output-format pftrace -d ./outdir -o shallow -- ./你的程序
```

> 面试要点：常用 tracing 选项＝`--kernel-trace`（看 kernel）/`--hip-trace`（看 HIP API）/`--sys-trace`（看全部）/`--marker-trace`（看自定义标记）。输出格式选 `pftrace` 方便用 Perfetto 看时间轴。

---

## Q6. 什么是 pftrace 格式？怎么用 Perfetto 看 trace？

**回答**

pftrace 是 Perfetto（Google 开源的性能追踪可视化工具）的输入格式。rocprofv3 生成的 pftrace 文件可以直接拖进 Perfetto 的网页界面（ui.perfetto.dev），就能看到 GPU kernel、HIP API、内存拷贝在时间轴上的分布。

**步骤：**
1. 用 rocprofv3 生成 pftrace：
   ```bash
   rocprofv3 --sys-trace --output-format pftrace -d ./outdir -o shallow -- ./程序
   ```
2. 打开浏览器访问 `ui.perfetto.dev`。
3. 点击“Open trace file”，选中生成的 `.pftrace` 文件。
4. 在 Perfetto 里就能看到时间轴上的各种事件条。

Perfetto 时间轴里你能看到：
- 每个 kernel 启动和结束的时间点（条形长度 = 耗时）。
- HIP API 调用（如 hipMemcpy）什么时候发生。
- 两个 kernel 之间有没有空隙（gap），那个空隙可能是 `hipDeviceSynchronize()` 或 CPU 侧准备下一个 kernel 的时间。

> 面试要点：pftrace 是 Perfetto 可视化格式。rocprofv3 生成后，拖进 ui.perfetto.dev 就能看到 kernel 和 API 的时间轴，直观发现“哪里有空隙、哪里有重叠”。

---

## Q6(续). 怎么设计公平的对照实验？（相同输入、相同 block、相同 warmup/repeat、新进程）

**回答**

想证明“优化后的版本比原版快”，必须做公平的 A/B 对照。公平的关键是**只改一个变量，其余全部保持一致**：

1. **相同输入规模**：同一个数组长度、同一个 dtype。
2. **相同 block 大小**：两边都用 `block=256`，不要一边 256 一边 512。
3. **相同 warmup/repeat**：两边都 warmup 20 次、repeat 100 次。
4. **同一机器、同一状态**：不要一边跑在空闲机，另一边跑在有人在跑别的任务的机器上。
5. **新进程跑基线**：为了避免进程内状态（JIT 缓存、时钟频率）互相干扰，最好在**新进程**里分别跑 A 和 B，再对比。

ASCII 图示（公平对照）：

```
版本 A（原版）                    版本 B（优化后）
┌─────────────┐                 ┌─────────────┐
│ size=16M    │                 │ size=16M    │（相同输入）
│ block=256   │                 │ block=256   │（相同块大小）
│ warmup=20   │                 │ warmup=20   │（相同预热）
│ repeat=100  │                 │ repeat=100  │（相同重复）
│ 新进程A     │                 │ 新进程B     │（分开进程）
└─────────────┘                 └─────────────┘
       │                            │
       ▼                            ▼
   取 median 时间               取 median 时间
       │                            │
       └───────── 比较 ─────────────┘
             谁更小谁更快
```

> 面试要点：公平 A/B 对照＝相同输入、相同 block 大小、相同 warmup/repeat、同一机器状态、最好分新进程跑。只改一个优化变量，其余全部锁死，才能得出可信结论。

---

## Q7. kernel trace 时间轴怎么看？（kernel 执行、HIP API、内存拷贝、同步间隙）

**回答**

用 Perfetto 打开 pftrace 后，时间轴上会出现不同颜色的条形，每种颜色代表一种事件：

| 事件类型 | 在时间轴上是什么意思 |
|---------|---------------------|
| GPU Kernel | 一个条形，长度 = kernel 真正执行的时间 |
| HIP API（如 hipMemcpy） | CPU 发起拷贝请求的时间 |
| Memory Copy（H2D/D2H） | 数据在主机和设备之间传输的时间 |
| hipDeviceSynchronize | CPU 等 GPU 完成的阻塞时间（表现为 gap） |

看 kernel trace 时，重点关注三件事：
1. **哪个 kernel 的条形最长？** → 那是耗时大头，优化优先级最高。
2. **两个 kernel 之间有没有空隙（gap）？** → 空隙可能是 `hipDeviceSynchronize()` 或 CPU 准备下一个 kernel 的时间，可以优化掉。
3. **内存拷贝（hipMemcpy）占用多少？** → 如果拷贝时间占比大，说明数据搬运成了瓶颈，可能需要减少拷贝次数或用 pinned 内存。

ASCII 图示（kernel trace 时间轴）：

```
时间 ─────────────────────────────────────────────────────────▶

CPU 侧（HIP API）：
┌─ hipMalloc ─┐   ┌─ hipMemcpy H2D ─┐   ┌─ kernel launch ─┐   ┌─ hipDeviceSynchronize ─┐
              │           │                     │                     │
              ▼           ▼                     ▼                     ▼
GPU 侧（Kernel 执行）：                 ┌─── kernel A ───┐   ┌─── kernel B ───┐
                                      │              │   │              │
                                      └──────────────┘   └──────────────┘
                                                      ▲
                                                      │
                                                这里的 gap 可能是
                                                hipDeviceSynchronize()
                                                或者 CPU 准备下一个 kernel
```

> 面试要点：kernel trace 时间轴上，条形长度 = 耗时，两个条形之间的 gap 可能是同步等待。重点看“哪个 kernel 最长”“两个 kernel 之间有没有 gap”“内存拷贝占不占大头”。

---

## Q8. hipDeviceSynchronize 在 trace 里表现为什么？（两个 kernel 之间的空隙）

**回答**

`hipDeviceSynchronize()` 是让 CPU 线程挂起，直到 GPU 上所有之前提交的任务全部完成。在 Perfetto 的时间轴上，它表现为 **两个 GPU kernel 条形之间的空白间隙（gap）**。

在这个 gap 里：
- GPU 可能已经算完上一个 kernel，正在等 CPU 发下一个指令；
- 或者 CPU 正在准备下一个 kernel 的参数、分配内存、做逻辑判断。

如果这个 gap 很长，说明 CPU 成了瓶颈，GPU 白白闲着。优化方法是：减少不必要的 `hipDeviceSynchronize()`，让多个 kernel 能尽可能流水线地提交（用 CUDA Graph 或多个 stream 实现重叠）。

ASCII 图示（去掉同步前后的对比）：

```
有 hipDeviceSynchronize（有 gap）：
Kernel A ───────┐       ┌─────── Kernel B
                │ gap   │
                │ (CPU  │
                │ 等 GPU│
                │ 完成) │
                ▼       ▼

去掉 hipDeviceSynchronize（无 gap，流水线）：
Kernel A ───────┐
                │
                ▼
Kernel B ───────┐
                │
                ▼
（B 可以在 A 还没完全结束时就开始准备，或紧随其后）
```

> 面试要点：`hipDeviceSynchronize()` 在 trace 上表现为两个 kernel 之间的空白间隙。如果 gap 很大，说明 GPU 在等 CPU，可以尝试去掉不必要的同步，让 kernel 流水线化提交。

---

## Q9. 去掉 hipDeviceSynchronize 后性能怎么变？（流水线重叠，消除等待间隙）

**回答**

去掉不必要的 `hipDeviceSynchronize()` 后，GPU 不会在两个 kernel 之间白白等待，CPU 可以尽快把下一个 kernel 提交给 GPU。两个 kernel 之间的 gap 缩小或消失，总体耗时下降。

ROCm 官方的 Profiling 入门文档里就有一个真实例子：去掉每个 kernel 后的 `hipDeviceSynchronize()` 后，总耗时从 0.965 s 降到 0.887 s，吞吐量从 8696 MCUPS 提升到 9452 MCUPS（约 1.09 倍加速），而正确性完全不变。

ASCII 图示（去掉同步后的效果）：

```
去掉同步前：
Kernel A ──[gap: hipDeviceSynchronize]── Kernel B ──[gap]── Kernel C
总耗时 = A + gap1 + B + gap2 + C

去掉同步后：
Kernel A ──▶ Kernel B ──▶ Kernel C
总耗时 ≈ A + B + C（gap 几乎为零）
加速比 ≈ (A+gap1+B+gap2+C) / (A+B+C)
```

> 面试要点：去掉不必要的 `hipDeviceSynchronize()` 可以消除 kernel 之间的等待间隙，让 GPU 流水线化工作，总耗时下降。ROCm 官方示例显示去掉后总耗时从 0.965 s 降到 0.887 s（约 1.09 倍加速），正确性不变。

---

## Q9(续). 什么是 Roofline 图？

**回答**

Roofline 图（屋顶图）是一种性能分析可视化模型，用来判断一个程序是**受计算能力限制**还是**受内存带宽限制**，以及还有多少优化空间。

它的横轴是 **算术强度（Arithmetic Intensity, AI）**，单位是 FLOP/Byte（每搬运一个字节数据做多少次浮点运算）；纵轴是 **可达性能（Attainable Performance）**，单位是 GFLOP/s。

图上有两条“屋顶”：
1. **斜线（内存带宽上限）**：性能 = 算术强度 × 峰值内存带宽。算术强度低的时候，性能沿这条斜线上升，说明受内存带宽限制。
2. **水平线（峰值算力上限）**：性能 = 峰值算力。算术强度高的时候，性能撑不住这条水平线，说明受计算能力限制。

两条线的交点叫 **拐点（Knee / Machine Balance）**，是从“内存受限”转为“计算受限”的临界算术强度。

ASCII 图示（Roofline 图）：

```
性能 (GFLOP/s)
   ▲
   │  ┌─────────────────┐  ← 峰值算力（水平屋顶）
   │  │                 │
   │  │                 │
   │  │                 │
   │  │                 │
   │  │                 │
   │  │                /│
   │  │               / │
   │  │              /  │
   │  │             /   │ ← 斜线：峰值带宽 × 算术强度
   │  │            /    │
   │  │           /     │
   │  │          /      │
   │  │         /       │
   │  │        /        │
   │  │       /         │
   │  │      /          │
   │  │     /           │
   │  │    /            │
   │  │   /             │
   │  │  /              │
   │  │ /               │
   │  │/                │
   └──┬─────────────────┴──▶ 算术强度 (FLOP/Byte)
      │
      拐点（Machine Balance）
      左边：存储密集型（受带宽限制）
      右边：计算密集型（受算力限制）
```

> 面试要点：Roofline 图用算术强度（横轴）和可达性能（纵轴）作图，由“峰值带宽斜线”和“峰值算力水平线”组成。交点是拐点，左边是内存受限，右边是计算受限。

---

## Q10. 什么是算术强度（Arithmetic Intensity）？怎么算？

**回答**

算术强度（Arithmetic Intensity, AI）是指 **每搬运一个字节数据，做多少次浮点运算**，单位是 FLOP/Byte。

公式：
```
算术强度 = 浮点运算次数 (FLOPs) / 数据搬运量 (Bytes)
```

算术强度越高，意味着每搬运一次数据能做更多计算，对内存带宽的依赖就越低，更容易受计算能力限制；算术强度越低，越容易受内存带宽限制。

举例：
- **向量加法 `c[i] = a[i] + b[i]`**：每个元素做 1 次加法，读 2 个 float（8 字节），写 1 个 float（4 字节），共搬运 12 字节。AI = 1 / 12 ≈ 0.083 FLOP/Byte（很低，典型的存储密集型）。
- **矩阵乘法 `C = A × B`（大矩阵）**：每个元素需要多次乘加，且中间结果可以复用，AI 很高（大矩阵可达几十甚至上百 FLOP/Byte），典型的计算密集型。

> 面试要点：算术强度 = 浮点运算次数 ÷ 数据搬运量（FLOP/Byte）。向量加法 AI 很低（≈0.083），是存储密集型；大矩阵乘法 AI 很高，是计算密集型。

---

## Q11. 什么是峰值算力（Peak FLOPS）和峰值带宽（Peak Bandwidth）？

**回答**

- **峰值算力（Peak FLOPS）**：GPU 在理想情况下，每秒能做的最大浮点运算次数。由 GPU 架构、核心数、频率决定，是“算力天花板”。
- **峰值带宽（Peak Bandwidth）**：GPU 显存每秒能传输的最大数据量（如 1.5 TB/s）。由显存类型（GDDR6/HBM）和位宽决定，是“搬运天花板”。

这两个值决定了 Roofline 图的水平线高度和斜线斜率。它们通常可以在 GPU 的官方规格书里找到，或者通过经验测量工具（如 Empirical Roofline Toolkit ERT）测得。

> 面试要点：峰值算力是 GPU 每秒最大浮点运算数（水平屋顶），峰值带宽是显存每秒最大传输量（决定斜线斜率）。两者都来自硬件规格或实测，是 Roofline 图的两条基准线。

---

## Q12. Roofline 图的“拐点”是什么意思？（机器平衡点 Machine Balance）

**回答**

拐点（Knee，也叫 Machine Balance 机器平衡点）是 Roofline 图上**斜线（带宽上限）和水平线（算力上限）的交点**。

拐点对应的算术强度：
```
机器平衡点 = 峰值算力 / 峰值带宽  （单位：FLOP/Byte）
```

拐点的意义：
- **算术强度 < 拐点**：程序在斜线上，性能受内存带宽限制（存储密集型）。优化应集中在减少数据搬运、提高缓存复用、合并访存。
- **算术强度 > 拐点**：程序在水平线上，性能受计算能力限制（计算密集型）。优化应集中在提高指令效率、向量化、减少分支。

> 面试要点：拐点（Machine Balance）= 峰值算力 ÷ 峰值带宽，单位 FLOP/Byte。它是 Roofline 图上“斜线”和“水平线”的交点，左边是内存受限，右边是计算受限。

---

## Q13. 计算密集型和存储密集型怎么区分？（在拐点左边还是右边？）

**回答**

- **存储密集型（Memory-bound）**：算术强度低，位于拐点**左边**。程序大部分时间在等数据从显存搬运，算力闲着。典型例子：向量加法、归约、Softmax。
- **计算密集型（Compute-bound）**：算术强度高，位于拐点**右边**。程序大部分时间在算，内存带宽闲着。典型例子：大矩阵乘法（GEMM）、卷积。

判断口径：把你的 kernel 的算术强度算出来，和拐点比一下，小于拐点就是存储密集型，大于拐点就是计算密集型。

ASCII 图示：

```
性能
   ▲
   │  ┌─────────────┐  ← 计算密集型（右边，受算力限制）
   │  │      ● GEMM   │
   │  │        ●       │
   │  │      ●          │
   │  │    ●            │
   │  │   ●             │
   │  │  ●              │
   │  │ ●               │
   │  │●                │
   │  ●                 │
   │ ●   ●  向量加法    │  ← 存储密集型（左边，受带宽限制）
   │●     ●             │
   └──┬──┬─────────────┴──▶ 算术强度
      │  │
      │  拐点（Machine Balance）
      │
   存储密集型 ←        → 计算密集型
```

> 面试要点：算术强度小于拐点 = 存储密集型（左边，受带宽限制）；算术强度大于拐点 = 计算密集型（右边，受算力限制）。向量加法是典型存储密集型，大矩阵乘法是典型计算密集型。

---

## Q14. 怎么判断一个 kernel 是受计算限制还是受内存限制？（看点在水平线下方还是斜线下方）

**回答**

把你的 kernel 在 Roofline 图上画一个点（横坐标 = 它的算术强度，纵坐标 = 它的实际测得性能）：

- 如果点在 **斜线下方** 或 **贴近斜线**：说明它受**内存带宽限制**，优化方向是减少数据搬运、提高缓存命中、合并访存。
- 如果点在 **水平线下方** 或 **贴近水平线**：说明它受**计算能力限制**，优化方向是提高指令效率、向量化、减少分支、利用好 tensor core。
- 如果点远低于两条线：说明除了带宽和算力，还有其他瓶颈（如占用率低、分支偏差严重、同步开销大）。

> 面试要点：在 Roofline 图上把 kernel 打个点：在斜线附近 = 受内存带宽限制；在水平线附近 = 受计算能力限制；远低于两条线 = 还有其他瓶颈（占用率、同步、分支）。

---

## Q15. Roofline 图怎么指导优化？

**回答**

Roofline 图不仅能“诊断”，还能“开处方”：

| 点的位置 | 诊断 | 优化方向 |
|---------|------|---------|
| 斜线下方（左侧） | 受内存带宽限制 | 减少全局内存访问、用 LDS 复用数据、合并访存（coalesced access）、增加数据复用率 |
| 水平线下方（右侧） | 受计算能力限制 | 提高指令级并行、向量化（FP16/BF16）、利用 tensor core、减少分支偏差 |
| 远低于两条线 | 既不是纯带宽也不是纯算力瓶颈 | 先提高占用率、减少同步等待、优化线程块大小、减少寄存器压力 |

优化的基本思路：尽量把你的点往**右上角**推——要么提高算术强度（右移，做更多复用），要么提高实际性能（上移，贴近屋顶）。

> 面试要点：Roofline 图指导优化：斜线下方优化访存（复用、合并、减少搬运），水平线下方优化计算（向量化、tensor core、减少分支），远低于两条线则先解决占用率和同步问题。总体目标是把点往右上角推。

---

## Q16. 向量加法的算术强度是多少？（两读一写，AI = 2 FLOP / 12 Byte ≈ 0.167）

**回答**

向量加法 `c[i] = a[i] + b[i]`，每个元素：
- 浮点运算：1 次加法 = 1 FLOP
- 数据搬运：读 `a[i]`（4 字节）+ 读 `b[i]`（4 字节）+ 写 `c[i]`（4 字节）= 12 字节

算术强度：
```
AI = 1 FLOP / 12 Byte ≈ 0.083 FLOP/Byte
```

（如果按两次读一次写严格算，是 1/12 ≈ 0.083；有些简化写法会说 2/8=0.25，但严格的两读一写是 1/12。）

0.083 远小于大多数 GPU 的拐点（通常几十到上百 FLOP/Byte），所以向量加法是**典型的存储密集型**算子，性能基本上受显存带宽限制，跑得再快也撑不起峰值算力。

> 面试要点：向量加法每个元素 1 次加法（1 FLOP）、搬运 12 字节（读 a、读 b、写 c），AI = 1/12 ≈ 0.083 FLOP/Byte。远小于典型 GPU 的拐点，属于典型存储密集型算子。

---

## Q17. GEMM 的算术强度为什么高？（数据复用，分块）

**回答**

GEMM（矩阵乘法 `C = A × B`）是计算密集型的代表，算术强度很高。原因：

1. **数据复用**：矩阵乘法中，A 和 B 的同一行/同一列可以被多次复用。比如一个 `A` 的元素可能参与计算 `C` 里的一整行，只需从内存读一次，就能用多次。
2. **分块（Tiling）**：把大矩阵切成小块装进 LDS（片上共享内存）或缓存，反复使用，大幅减少从显存搬运数据的次数。

理想情况下，大方阵乘法的算术强度可达 `2n / (3s)`（n 是矩阵阶数，s 是每个元素字节数）。n 越大，AI 越高，越容易逼近峰值算力。

例如，FP32 的 `n=128` 时，AI ≈ 21.3 FLOP/Byte；`n=2048` 时，AI ≈ 341.3 FLOP/Byte。这说明矩阵越大，越容易成为计算密集型。

> 面试要点：GEMM 算术强度高，因为矩阵元素可以被多次复用（同一行/列参与多次乘加），再配合分块（tiling）把数据留在片上缓存/LDS 里反复用，大幅减少显存搬运。矩阵越大，AI 越高，越容易成为计算密集型。

---

## Q18. 什么是 operational intensity？（和算术强度一样）

**回答**

Operational intensity（运算强度）和 Arithmetic intensity（算术强度）是同一个概念的不同翻译，都指 **每搬运一个字节数据，做多少次浮点运算**，单位 FLOP/Byte。

在 Roofline 文献里，这两个词混用，不需要区分。公式一样：

```
Operational intensity = 浮点运算次数 (FLOPs) / 数据搬运量 (Bytes)
```

> 面试要点：Operational intensity 和 Arithmetic intensity 是同一个概念，都是“每搬运一个字节做多少次浮点运算”，单位 FLOP/Byte，Roofline 图的横轴就是这个指标。

---

## Q19. 怎么用 rocprof-compute 收集硬件计数器？（--pmc、VALUUtilization、VALUBusy）

**回答**

rocprof-compute（前身 Omniperf）可以收集 GPU 硬件性能计数器（PMC），帮你判断算单元忙不忙、存储单元忙不忙。

常用硬件计数器：
- **VALUUtilization**：向量算术逻辑单元利用率。
- **VALUBusy**：向量 ALU 处理指令的时间占比（越接近 100% 越好）。
- **FetchSize / WriteSize**：读取/写入的数据大小。
- **MemUnitStalled**：存储单元等待的周期数。

用法（指定 counter 文件）：

```bash
# 先写一个计数器文件 rocprof_counters.txt
cat > rocprof_counters.txt << 'EOF'
pmc: VALUUtilization VALUBusy FetchSize WriteSize MemUnitStalled
pmc: GPU_UTIL CU_OCCUPANCY MeanOccupancyPerCU MeanOccupancyPerActiveCU
EOF

# 运行 rocprof-compute 收集
rocprof-compute profile -n v1 -- ./你的程序
```

`rocprof-compute` 会自动运行程序多次（因为不同计数器组不能同时采集），最后输出汇总结果，帮你判断瓶颈在算力还是带宽。

> 面试要点：rocprof-compute 用 `--pmc` 收集硬件计数器（如 VALUUtilization、VALUBusy、MemUnitStalled），需要指定计数器文件并运行多次。VALUBusy 越接近 100%，说明向量算单元越忙，计算利用率越高。

---

## Q20. 什么是 VALUBusy / VALUUtilization？怎么看算子忙不忙？

**回答**

- **VALUBusy**：向量算术逻辑单元（VALU）处理指令的周期占比。越接近 100%，说明向量算单元一直在干活，计算利用率高。
- **VALUUtilization**：更宽泛的利用率指标，反映 VALU 整体忙碌程度。

如果 VALUBusy 很低（比如只有 50%），说明算单元有一半时间在闲等，可能是因为在等内存数据（存储密集型瓶颈）或者有分支偏差导致部分线程闲置。

ROCm 官方 Profiling 入门文档里有个真实例子：通过把 tile 从 16×16 调整到 32×32，VALUBusy 从 50% 提升到 71%，性能相应提升了约 1.12 倍。

> 面试要点：VALUBusy 是向量算单元忙碌的时间占比，越接近 100% 越好。如果很低，说明算单元在等数据或有分支偏差，需要优化访存或线程组织。

---

## Q21. 什么是 occupancy？rocprofv3 怎么看占用率？（--occupancy）

**回答**

Occupancy（占用率）是指：某一时刻，CU 上实际驻留的 wavefront 数量，除以硬件理论上最多能驻留的 wavefront 数量。

它受三个因素限制（取最小值）：
1. **寄存器限制**：每个 wavefront 用的 VGPR/SGPR 太多，放不下多个 wavefront。
2. **LDS 限制**：每个 work-group 用的 LDS 太多，放不下多个 work-group。
3. **wave slot 限制**：每个 CU 的硬件波前槽位数固定。

用 rocprofv3 查看占用率：

```bash
rocprofv3 --occupancy ./你的程序
```

低 occupancy 意味着 GPU 核心没有被充分利用，在等内存时没法切换去执行别的 wavefront，延迟隐藏能力差。

> 面试要点：Occupancy = 实际驻留 wavefront 数 / 硬件理论最大驻留 wavefront 数，受寄存器、LDS、wave slot 三者限制。用 `rocprofv3 --occupancy` 查看。低 occupancy 意味着 GPU 核心利用不充分，延迟隐藏能力差。

---

## Q22. 占用率低怎么优化？（减少寄存器、减少 LDS 使用、调整块大小）

**回答**

提高占用率的核心思路是：**让每个 wavefront 用更少的资源，这样一个 CU 能同时驻留更多的 wavefront，在等内存时切换去执行别的。**

具体做法：

1. **减少寄存器使用**：把不需要每个线程私有的值改成 SGPR（共用），简化算术表达式，降低 VGPR 数量。
2. **减少 LDS 使用**：每个 work-group 用的 LDS 越少，能放下的 work-group 越多。
3. **调整 block 大小**：block 太小（比如只有 32 个线程）会浪费 wavefront 槽位；block 太大会占用太多资源。通常选波前大小的倍数（如 256、512），对 AMD Instinct 建议 64/128/256。

ASCII 图示（减少寄存器提高占用率）：

```
每个 wavefront 用 256 个 VGPR ──▶ CU 只能放 1 个 wavefront ──▶ 占用率低
      │
      ▼（优化：简化代码，减少 VGPR）
每个 wavefront 用 64 个 VGPR ──▶ CU 能放 4 个 wavefront ──▶ 占用率高
```

> 面试要点：提高占用率＝减少每个 wavefront 的寄存器和 LDS 使用，让一个 CU 能同时驻留更多 wavefront；调整 block 大小为波前大小倍数（如 256）。在等内存时切换执行其他 wavefront，隐藏延迟。

---

## Q23. 什么是成组访问（coalesced access）？为什么重要？

**回答**

成组访问（内存合并访问）是指 wavefront 里的多个线程访问全局内存时，如果访问的地址是连续的，硬件可以把这些访问合并成一次或少数几次内存请求，而不是每个线程单独发一次请求。

如果访问地址是散乱的（比如线程 0 访问地址 0，线程 1 访问地址 1000），硬件就无法合并，每个线程都要单独去显存取数据，这叫“非合并访问”，会严重拖慢速度。

大白话比喻：成组访问就像“集体买团购券”，一群人要买相邻座位的电影票，票务员可以一次性打包处理；非合并访问就像“每人单独去柜台”，一个一个来，效率低得多。

ASCII 图示：

```
成组访问（好）：线程访问连续地址，合并成一次内存请求
线程0 ──▶ 地址 0   │
线程1 ──▶ 地址 1   │──▶ 一次内存请求，取 4 个连续 int
线程2 ──▶ 地址 2   │
线程3 ──▶ 地址 3   │

非合并访问（差）：线程访问散乱地址，每个都要单独请求
线程0 ──▶ 地址 0   ──▶ 请求1
线程1 ──▶ 地址 1000 ──▶ 请求2
线程2 ──▶ 地址 2000 ──▶ 请求3
线程3 ──▶ 地址 3000 ──▶ 请求4
```

> 面试要点：成组访问＝wavefront 里多个线程访问连续全局内存地址，硬件合并为一次请求，能极大提升带宽利用率。散乱访问会导致每个线程单独请求，速度慢。

---

## Q24. 全局内存访问怎么合并？Add-TID 寻址是什么？

**回答**

要让全局内存访问合并，最基本的原则是：**让相邻的线程访问相邻的内存地址**。

在 HIP/CUDA 编程里，最常见的写法是“Add-TID”寻址：每个线程用自己的线程 ID（tid）加上一个基础偏移，来确定自己要访问的数据。如果 tid 是连续的，访问的地址也就连续，自然能合并。

Python 示例（模拟合并 vs 不合并访问）：

```python
# 模拟全局内存访问（概念版，CPU 上演示地址模式）
NUM_THREADS = 8

# 情况1：合并访问（相邻线程访问相邻地址）
print("=== 合并访问（Add-TID，tid 连续）===")
for tid in range(NUM_THREADS):
    addr = 1000 + tid  # 基址 1000 + 线程 ID，地址连续
    print(f"线程 {tid:2d} 访问地址 {addr}")

# 情况2：非合并访问（相邻线程访问散乱地址）
print("\n=== 非合并访问（stride 大，地址散乱）===")
stride = 100
for tid in range(NUM_THREADS):
    addr = 1000 + tid * stride  # 地址跳跃很大
    print(f"线程 {tid:2d} 访问地址 {addr}")
```

输出示例：
```
=== 合并访问（Add-TID，tid 连续）===
线程  0 访问地址 1000
线程  1 访问地址 1001
线程  2 访问地址 1002
线程  3 访问地址 1003
...

=== 非合并访问（stride 大，地址散乱）===
线程  0 访问地址 1000
线程  1 访问地址 1100
线程  2 访问地址 1200
线程  3 访问地址 1300
...
```

> 面试要点：全局内存访问合并的核心原则＝相邻线程访问相邻地址。Add-TID 寻址（基址 + 线程 ID）是最常见的合并写法，能让 wavefront 的访问被硬件合并为少数几次内存请求。

---

## Q25. 什么是 LDS bank conflict？怎么避免？

**回答**

LDS 虽然快，但它内部被分成很多个“银行”（bank）。同一时刻，如果多个线程试图访问**同一个银行**里的不同数据，这些访问就得排队串行执行，这就叫 **bank conflict（银行冲突）**，会拖慢 LDS 访问速度。

大白话比喻：LDS 像一个有多个柜员窗口的银行。如果每个客户都去不同的窗口，大家可以同时办理业务；如果多个客户都挤在同一个窗口前，后面的人就得排队。

**避免方法：**
1. **让相邻线程访问相邻的 LDS 地址**：这样它们会落在不同的银行，避免冲突。
2. **避免 stride 为银行数倍数的访问**：比如如果 LDS 有 32 个银行，线程 i 访问地址 `i * 32`，那所有线程都会落在同一个银行，造成最严重的冲突。
3. **填充（padding）**：在数组每行末尾多留几个空位，打破规律性的冲突。

ASCII 图示（bank conflict）：

```
LDS 有 4 个银行（Bank 0~3），地址按 4 循环分配：

地址：  0   1   2   3   4   5   6   7
银行： [0] [1] [2] [3] [0] [1] [2] [3]

无冲突：线程0→地址0(Bank0)，线程1→地址1(Bank1)，线程2→地址2(Bank2)，线程3→地址3(Bank3)
        4个线程同时访问，4个银行并行，一次完成

有冲突：线程0→地址0(Bank0)，线程1→地址4(Bank0)，线程2→地址8(Bank0)，线程3→地址12(Bank0)
        4个线程都访问 Bank0，必须排队串行，4次才完成
```

> 面试要点：LDS bank conflict 是指多个线程同时访问同一个 LDS 银行，导致串行化。避免方法：相邻线程访问相邻地址、避免 stride 等于银行数倍数的访问、必要时做 padding 填充。

---

## Q26. 标量内存和向量内存有什么区别？

**回答**

GPU 里访存分为标量访存（scalar memory）和向量访存（vector memory），对应两类指令和缓存：

| 维度 | 标量内存访存 | 向量内存访存 |
|------|-------------|-------------|
| 使用寄存器 | SGPR（标量寄存器） | VGPR（向量寄存器） |
| 访存粒度 | 每次访存一个标量值（整个 wavefront 共用同一个值） | 每个 lane 访存一个值，wavefront 里每个线程不同 |
| 典型用途 | 循环计数、基础指针、所有线程共用的常量 | 数组元素、每个线程私有的数据 |
| 缓存 | 标量 L0/L1 缓存 | 向量 L0/L1/L2 缓存 |

大白话：标量内存就像“老师黑板上写一个数字，全班同学都看同一个”；向量内存就像“每个学生桌上有自己的本子，上面记的数字不一样”。

> 面试要点：标量内存访存用 SGPR，整个 wavefront 共用一个值；向量内存访存用 VGPR，每个 lane 有自己的值。标量用于循环计数、基础指针；向量用于数组元素等私有数据。

---

## Q27. 缓存（L0/L1/L2）在 GPU 访存里起什么作用？

**回答**

GPU 的缓存层级（L0/L1/L2）是为了弥补“算力快、显存慢”的矛盾。算术逻辑单元（ALU）算得飞快，但每次去显存取数据要等几百个周期，ALU 就会空等。缓存把常用数据留在靠近 ALU 的地方，减少去显存的次数。

- **L0 缓存**：每个 CU 私有，最快，容量最小。
- **L1 缓存**：WGP 内多个 CU 共享，速度次之。
- **L2 缓存**：整个 GPU 共享，最大，速度慢于 L1 但快于显存。

访存时，先查 L0，没命中查 L1，再查 L2，最后才去显存。命中率越高，程序越快。

> 面试要点：L0/L1/L2 缓存按距离 ALU 从近到远、从快到慢分层，用于缓存常用数据，减少去慢速显存的次数，提高访存效率。L0 每 CU 私有，L1 WGP 共享，L2 全 GPU 共享。

---

## Q28. 内存一致性模型在 GPU 里怎么理解？（s_wait_loadcnt 等）

**回答**

GPU 是并行执行的，多个 wavefront 可能同时读写同一块内存。如果不加控制，一个 wavefront 写完数据，另一个 wavefront 可能还没看到，这就是内存一致性问题。

为了解决这个问题，GPU 提供了屏障/等待指令，常见的有：

- `s_wait_loadcnt 0`：等待所有标量读取完成。
- `s_wait_storecnt 0`：等待所有标量写入完成。
- `s_wait_dscnt 0`：等待所有 LDS 操作完成。

这些指令告诉 GPU：“在继续执行下一条指令之前，先确保前面的内存操作都做完了。”

大白话：就像工厂流水线，工人 A 把零件放在传送带上，工人 B 要拿这个零件继续加工。如果 B 没等 A 放好就拿，就会拿空手。`s_wait_*` 指令就是那个“等 A 放好”的信号。

> 面试要点：GPU 内存一致性靠 `s_wait_loadcnt`/`s_wait_storecnt`/`s_wait_dscnt` 等屏障指令保证，确保前面的读写操作完成后再继续，避免多个 wavefront 并发访问同一内存时出现脏数据。

---

## Q29. 为什么要关心内存访问模式？（带宽利用率、70-90% vs 5-15%）

**回答**

内存访问模式决定了显存带宽能用多少。这是 GPU 性能优化里最重要的因素之一。

实测经验法则：
- **合并访问（连续地址）**：能达到峰值带宽的 70%～90%。
- **随机/散乱访问**：可能只有峰值带宽的 5%～15%。

差距巨大，因为散乱访问导致硬件无法合并请求，每次只能取很少的数据，大部分带宽被浪费在传输无关数据上。

> 面试要点：内存访问模式决定带宽利用率。合并访问可达峰值带宽 70%～90%，随机散乱访问可能只有 5%～15%。优化访存模式是提升 GPU 程序性能的最重要手段之一。

---

## Q30. 本章小结：Profiling 优化闭环怎么做？

**回答**

GPU 性能优化不是“随便改一段代码看看”，而是一个闭环：

```
Profiling 优化闭环
┌─────────────┐
│ 1. Profile  │ rocprofv3 --stats --kernel-trace 找热点 kernel
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 2. 分析瓶颈   │ Roofline 图：受带宽限制还是受算力限制？
│             │ VALUBusy：算单元忙不忙？
│             │ Occupancy：核心利用够不够？
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 3. 定向优化   │ 受带宽限制→优化访存（合并、复用、减少搬运）
│             │ 受算力限制→优化计算（向量化、tensor core、减少分支）
│             │ 占用率低→减少寄存器/LDS，调整 block 大小
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 4. Re-profile │ 再跑 rocprofv3/rocprof-compute，确认
│             │ 提升没有退步（其他指标没掉下来）
└─────────────┘
       │
       ▼ (如未达标，回到步骤 2)
```

一句话总结：**Profiling 先找热点，Roofline 判瓶颈（带宽还是算力），定向优化后 re-profile 验证，不要瞎猜，要让数据说话。**

> 面试要点：本章核心＝rocprofv3 找热点（--stats --kernel-trace）→Roofline 图判瓶颈（算术强度、拐点、计算密集型/存储密集型）→定向优化（访存合并、复用、减少寄存器/LDS 提升占用率）→re-profile 验证。闭环反复，不要瞎猜。
