# HIP 编程基础与性能计时 —— 第一个 HIP 程序、GPU 计时与可信 benchmark

> 面试生普（LLM 500 题）· HIP 编程基础与性能计时篇。本章覆盖从零写出第一个 HIP 程序并给它做可信性能测量的全流程：HIP 与 CUDA 的关系、__global__/<<<>>>/blockIdx/threadIdx、HIP 程序六步骨架（主机准备数据、设备分配显存、主机↔设备拷贝、启动 kernel、拷回结果、正确性检查）、HIP 错误处理（HIP_CHECK）、为什么不能用 time.time() 给 GPU 计时、GPU event 计时（record → synchronize → elapsed_time）、warmup 与 repeat、有效带宽计算（两读一写）、median/min 口径、第一次运行慢的原因、wall-clock 与 GPU 时间的区别、JIT 编译与时钟频率波动、公平 A/B 对照检查清单，共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Python/C++ 代码和 ASCII 图。

---

## Q1. 什么是 HIP？它和 CUDA 有什么关系？

**回答**

HIP（Heterogeneous-compute Interface for Portability）是 AMD 为 ROCm 平台提供的 C++ 异构编程语言。它的设计目标是：**让写 CUDA 代码的开发者，能以最小改动把代码跑在 AMD GPU 上**。

HIP 和 CUDA 的关系可以用一句话说概括：**HIP 语法和 CUDA 极像，把 `cuda` 前缀改成 `hip`，把 `nvcc` 改成 `hipcc`，大部分代码就能直接跑。**

对照表：

| 概念 | CUDA（NVIDIA） | HIP（AMD ROCm） |
|------|---------------|----------------|
| 编译器 | nvcc | hipcc |
| 内核限定词 | `__global__` | `__global__`（一样） |
| 启动语法 | `kernel<<<grid, block>>>` | `kernel<<<grid, block>>>`（一样） |
| 线程索引 | `threadIdx.x` | `hipThreadIdx_x`（或 `threadIdx.x`） |
| 块索引 | `blockIdx.x` | `hipBlockIdx_x`（或 `blockIdx.x`） |
| 内存分配 | `cudaMalloc` | `hipMalloc` |
| 内存拷贝 | `cudaMemcpy` | `hipMemcpy` |
| 设备同步 | `cudaDeviceSynchronize` | `hipDeviceSynchronize` |

> 面试要点：HIP 是 AMD 版的 CUDA，语法极像，大部分情况下把 `cuda` 换成 `hip`、`nvcc` 换成 `hipcc` 即可迁移。HIP 程序跑在 ROCm 上，需要 AMD GPU 和 ROCm 运行时。

---

## Q2. 怎么写第一个 HIP 向量加法程序？（__global__、<<<>>>、blockIdx/threadIdx）

**回答**

第一个 HIP 程序通常就是“向量加法”：给定两个等长数组 `a`、`b`，让 GPU 计算 `c[i] = a[i] + b[i]`。这个例子简单，但包含了 GPU 编程的所有核心概念。

下面是一个最小的 HIP 向量加法 kernel：

```cpp
// vector_add.hip
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // 计算当前线程负责的全局下标
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查：防止最后一个 block 越界
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

关键语法速查：

- `__global__`：告诉编译器“这是一个 GPU 函数，由 CPU 调用、在 GPU 上运行”。
- `<<<blocks, threads>>>`：HIP/CUDA 特有的 kernel 启动语法，`blocks` 是启动多少组，`threads` 是每组多少个线程。
- `blockIdx.x`：当前 block 在 grid 里的编号（工号）。
- `threadIdx.x`：当前线程在 block 里的编号（小工号）。
- `blockIdx.x * blockDim.x + threadIdx.x`：算出当前线程负责数组里的哪个位置。

大白话比喻：`vector_add<<<blocks, threads>>>(...)` 就是在 GPU 上同时叫起 `blocks × threads` 个工人，每个工人执行一次 `vector_add` 函数，每个工人负责数组里的一个元素。

ASCII 图示（线程编号到全局下标的映射）：

```
grid（所有 block）
│
├─ block 0
│   ├─ thread 0 ──▶ 全局下标 0
│   ├─ thread 1 ──▶ 全局下标 1
│   └─ ...
│
├─ block 1
│   ├─ thread 0 ──▶ 全局下标 blockDim.x
│   ├─ thread 1 ──▶ 全局下标 blockDim.x + 1
│   └─ ...
│
└─ ...

公式：全局下标 = blockIdx.x × blockDim.x + threadIdx.x
```

> 面试要点：第一个 HIP 程序＝写 `__global__` kernel、用 `<<<blocks, threads>>>` 启动、用 `blockIdx.x * blockDim.x + threadIdx.x` 算全局下标、用 `if (idx < n)` 做边界检查。

---

## Q3. HIP 程序的六步骨架是什么？

**回答**

几乎所有 HIP 程序都遵循同一个六步骨架，就像做菜的“买菜→洗菜→切菜→炒菜→装盘→尝味道”：

| 步骤 | 做什么 | 对应 HIP API |
|------|--------|-------------|
| 1. 主机准备数据 | 在 CPU 内存里准备输入数组 | `std::vector` / `new` / `malloc` |
| 2. 设备分配显存 | 在 GPU 显存里申请空间 | `hipMalloc` |
| 3. 主机→设备拷贝 | 把输入数据从 CPU 拷到 GPU | `hipMemcpy(..., hipMemcpyHostToDevice)` |
| 4. 启动 kernel | 在 GPU 上启动并行计算 | `kernel<<<grid, block>>>(...)` |
| 5. 设备→主机拷回 | 把计算结果从 GPU 拷回 CPU | `hipMemcpy(..., hipMemcpyDeviceToHost)` |
| 6. 检查结果 | 在 CPU 上验证输出是否正确 | 循环比对 `max_error` |

ASCII 图示（六步骨架数据流向）：

```
主机内存 (CPU)                      设备显存 (GPU)
┌─────────────┐                     ┌─────────────┐
│ h_a, h_b    │──── hipMemcpy �────▶│ d_a, d_b    │
│ h_c         │◀──── hipMemcpy �────│ d_c         │
└─────────────┘                     └─────────────┘
      │                                   │
      │  1. 准备数据                        │  2. 分配显存
      │                                   │
      ▼                                   │
┌─────────────┐                     ┌─────────────┐
│ 检查 h_c    │◀────────────────────┤ 4. 启动 kernel   │
│  结果正确？ │                     │ (GPU 并行计算)    │
└─────────────┘                     └─────────────┘
          ▲                                   │
          └─────────── 6. 检查结果 ───────────┘
              5. 拷回结果
```

> 面试要点：HIP 程序六步骨架＝主机准备数据→设备分配显存→主机到设备拷贝→启动 kernel→设备到主机拷回→检查结果。后面无论多复杂的 kernel（归约、Softmax、GEMM），骨架都是这六步。

---

## Q4. 主机（Host）和设备（Device）有什么区别？

**回答**

在 HIP/ROCm 术语里：

- **主机（Host）**：指 CPU 所在的那一侧，运行普通的 C++ 程序，负责“指挥”——分配内存、发启动指令、检查结果。
- **设备（Device）**：指 GPU 所在的那一侧，负责“干活”——执行 `__global__` kernel，做大规模并行计算。

两者的内存是分开的：主机内存（RAM）和设备显存（VRAM）不互通。如果 GPU 要处理 CPU 里的数据，必须通过 `hipMemcpy` 显式拷过去；计算完结果也要拷回来，CPU 才能看到。

大白话比喻：主机像“工厂总监”，设备像“车间工人”。总监把原料（数据）送到车间，工人在车间里加工，加工完总监再把成品（结果）拿回来检查。

ASCII 图示：

```
主机 (Host, CPU)                    设备 (Device, GPU)
┌─────────────┐                     ┌─────────────┐
│ 决定干什么   │                     │ 真正干活     │
│ 分配内存     │                     │ 并行计算     │
│ 发启动指令   │                     │ (kernel)     │
│ 检查结果     │                     │             │
└──────┬──────┘                     └──────┬──────┘
       │ hipMemcpy                       │
       ▼ (H2D)                           │
┌─────────────┐                     ┌─────────────┐
│ 原数据 (RAM)│───────────────────▶│ 数据 (VRAM)  │
└─────────────┘    拷贝过去       └─────────────┘
                                              │
                                              ▼
┌─────────────┐                     ┌─────────────┐
│ 结果 (RAM)  │◀───────────────────│ 结果 (VRAM)  │
└─────────────┘     拷贝回来         └─────────────┘
       ▲                                        │
       └──────── 6. 检查结果 ───────────────────┘
```

> 面试要点：主机（CPU）负责指挥和内存管理，设备（GPU）负责并行计算。两者内存分离，数据必须通过 `hipMemcpy` 显式拷贝，方向分为 H2D（主机→设备）和 D2H（设备→主机）。

---

## Q5. hipMalloc 和 hipMemcpy 分别做什么？

**回答**

- **hipMalloc**：在 GPU 显存里申请一块空间，返回一个设备指针。就像在 GPU 上“开辟一块地”，专门给 GPU 程序读写。
- **hipMemcpy**：在主机内存和设备显存之间拷贝数据。有三种方向：
  - `hipMemcpyHostToDevice`：CPU → GPU（常用于把输入数据送进去）
  - `hipMemcpyDeviceToHost`：GPU → CPU（常用于把结果拷回来）
  - `hipMemcpyDeviceToDevice`：GPU → GPU（两块显存之间拷贝）

C++ 示例（六步骨架里的核心部分）：

```cpp
// 2. 设备分配显存
float *d_a, *d_b, *d_c;
size_t bytes = n * sizeof(float);
HIP_CHECK(hipMalloc(&d_a, bytes));
HIP_CHECK(hipMalloc(&d_b, bytes));
HIP_CHECK(hipMalloc(&d_c, bytes));

// 3. 主机 → 设备拷贝（把输入送进 GPU）
HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

// 4. 启动 kernel（GPU 并行计算）
int block = 256;
int grid = (n + block - 1) / block;
vector_add<<<grid, block>>>(d_a, d_b, d_c, n);

// 5. 设备 → 主机拷回（把结果取回来）
HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
```

> 面试要点：hipMalloc 在 GPU 显存申请空间，hipMemcpy 在主机和设备之间拷数据。方向用 `hipMemcpyHostToDevice`（送进 GPU）和 `hipMemcpyDeviceToHost`（取回结果）两种最常见。

---

## Q6. 怎么验证 HIP 程序的结果正确？（max_error、status PASS）

**回答**

GPU 程序跑完后，最直接的验证方法是：把结果拷回主机后，用 CPU 逐元素比对，计算最大误差 `max_error`，如果误差在可接受范围内（比如小于 `1e-5`），就判定通过。

C++ 示例：

```cpp
// 6. 检查结果
float max_error = 0.0f;
for (int i = 0; i < n; i++) {
    float expected = h_a[i] + h_b[i];
    float diff = fabsf(h_c[i] - expected);
    if (diff > max_error) max_error = diff;
}

bool passed = (max_error < 1e-5f);
printf("max_error: %e\n", max_error);
printf("status: %s\n", passed ? "PASS" : "FAIL");
```

运行输出示例（RX 9070 XT）：

```
device_name: AMD Radeon RX 9070 XT
vector_size: 1048576
blocks: 4096
threads_per_block: 256
max_error: 0
status: PASS
```

看到 `status: PASS`，就说明第一个 HIP kernel 跑对了。

> 面试要点：验证 HIP 程序正确性＝把结果拷回主机→CPU 逐元素比对→计算 `max_error`→判断是否小于阈值（如 `1e-5`）→输出 `PASS`/`FAIL`。这是所有 GPU 程序的基本验收仪式。

---

## Q7. 为什么不能用 time.time() 来给 GPU 程序计时？

**回答**

这是 GPU 性能测量里最经典的坑。GPU 任务通常是**异步提交**的：当你在 CPU 端调用 `kernel<<<...>>>` 或 `torch.softmax(x)` 时，CPU 只是把命令“塞进队列”就立刻返回了，GPU 真正执行完可能还要过一会儿。

如果你用 `time.time()`（壁钟时间）在调用前后取差，量到的大概率是 **“CPU 把命令塞进队列花了多久”**，而不是 **“GPU 算了多久”**。这就会导致“GPU 时间很短，说明程序很快”的错觉，其实 GPU 还在后台辛苦算着呢。

ASCII 图示（time.time() 的陷阱）：

```
CPU 时间线                         GPU 时间线
┌─────────────┐                   ┌─────────────┐
│ time.time() │                   │             │
│   start     │──命令塞进队列────▶│  kernel 开始  │
│             │  返回（很快）     │   执行中...   │
│   end       │                   │   执行中...   │
│   差值≈0ms  │                   │   执行中...   │
└─────────────┘                   │  kernel 结束  │
                                  └─────────────┘
                            time.time() 只量了“塞命令”，没量“算多久”
```

正确的做法：用 **GPU 自己的计时机制**，在设备时间线上打两个事件（event），再算它们之间的间隔。下一题会讲具体怎么做。

> 面试要点：不能用 `time.time()` 给 GPU 程序计时，因为 GPU 任务是异步的，`time.time()` 量的是 CPU 塞命令的时间，不是 GPU 算多久的时间。必须用 GPU event 计时。

---

## Q8. 什么是 GPU event？怎么用 torch.cuda.Event 计时？

**回答**

GPU event 是 GPU 提供的计时机制，可以在设备时间线上打两个“时间戳事件”（start 和 end），再计算它们之间的时间间隔。这个时间是 **GPU 真正执行的时间**，不受 CPU 排队和调度干扰。

在 PyTorch（ROCm 版）里，用法是 `torch.cuda.Event(enable_timing=True)`。

Python 骨架（PyTorch 端到端 op 计时）：

```python
import torch

def bench(shape, dtype, repeats=200, warmup=20):
    x = torch.randn(*shape, dtype=dtype, device="cuda")

    # warmup：让 JIT、缓存、时钟进入稳定状态
    for _ in range(warmup):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # 用 GPU event 计时（不要用 time.time()）
    start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end   = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start[i].record()
        torch.softmax(x, dim=-1)
        end[i].record()

    # 所有 event 都记完后，再同步读取
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start, end)]
    return times_ms
```

关键点：
1. `enable_timing=True` 是必须的，否则 `elapsed_time` 不可用。
2. 必须在所有 event 记录完后调用 `torch.cuda.synchronize()`，否则 host 可能读到 stale（过时）的值。
3. `elapsed_time` 返回的是毫秒（ms）。

> 面试要点：GPU event 是在设备时间线上打时间戳的计时机制。PyTorch 里用 `torch.cuda.Event(enable_timing=True)`，调用 `.record()` 打事件，`.elapsed_time()` 算间隔（ms）。用完后必须 `torch.cuda.synchronize()` 再读，否则拿到的是旧值。

---

## Q9. GPU event 计时的最小流程是什么？（record → synchronize → elapsed_time）

**回答**

GPU event 计时的最小流程只有三步，按顺序不能错：

```
第1步：record —— 在 GPU 命令队列里打一个“开始事件”
第2步：执行目标代码 —— GPU 开始干活
第3步：record —— 在 GPU 命令队列里打一个“结束事件”
第4步：synchronize —— 等 GPU 把上面的事全做完
第5步：elapsed_time —— 算两个 event 之间的时间差（ms）
```

Python 代码骨架（最小可运行版）：

```python
import torch

# 准备数据
x = torch.randn(1024, 1024, device="cuda")

# 打开始事件
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
y = x @ x          # GPU 开始做矩阵乘
end.record()

# 等 GPU 把上面的事都做完
torch.cuda.synchronize()

# 算时间差（ms）
elapsed_ms = start.elapsed_time(end)
print(f"GPU 矩阵乘耗时：{elapsed_ms:.3f} ms")
```

ASCII 图示（event 计时时序）：

```
host 端（CPU）                      device 端（GPU）
┌─────────────┐                   ┌─────────────┐
│ start.record() │────命令───────▶│ [事件: start] │
│              │                   │             │
│   y = x @ x  │────命令───────▶│  执行矩阵乘...  │
│              │                   │   执行中...   │
│ end.record() │────命令───────▶│ [事件: end]   │
│              │                   │             │
│ synchronize()│                   │（做完再通知） │
│              │                   └─────────────┘
│ elapsed_time │                   │
│   (ms)       │                   │
└─────────────┘
       ▲
       │
   只能在 synchronize() 之后读，否则是旧值
```

> 面试要点：GPU event 计时最小流程＝`start.record()`→目标代码→`end.record()`→`torch.cuda.synchronize()`→`start.elapsed_time(end)`。`synchronize()` 是关键，不加它读到的可能是旧值。

---

## Q10. 什么是 warmup？为什么正式计时前要跑几次？

**回答**

warmup（预热）是指在正式计时前，先跑若干次相同的操作，但**不把这些次数算进统计结果**。

为什么要 warmup？因为第一次运行一个 GPU kernel 通常会比后面慢，原因包括：
- **JIT 编译**：HIP/PyTorch 可能首次遇到某个 kernel 时才编译生成目标代码（hsaco）。
- **缓存准备**：显存分配、缓存预填、时钟频率升频等需要时间。
- **运行时初始化**：HIP 上下文、CUDA 图等首次建立有开销。

如果不 warmup，第一次运行的时间会被混进统计，导致结果偏大、不可比。

ASCII 图示（warmup 与正式计时）：

```
时间轴
┌──────────┬──────────┬──────────┬──────────┐
│ warmup   │ warmup   │ 正式     │ 正式     │
│ 第1次    │ 第2次    │ 第1次    │ 第2次    │
│（不计入）│（不计入）│（计入）  │（计入）  │
└──────────┴──────────┴──────────┴──────────┘
   ↓           ↓           ↓           ↓
   慢（初始化）  较快         稳定        稳定
   不统计       不统计       统计        统计
```

> 面试要点：warmup ＝正式计时前先跑几次不计入统计的预热运行，目的是让 JIT 编译、缓存、时钟频率进入稳定状态，避免首次运行的开销污染结果。

---

## Q11. 为什么 benchmark 要重复运行多次？（repeat、median/min）

**回答**

GPU 计时有一定的随机波动（通常 ±5%～15%），来源包括操作系统调度、时钟频率波动、后台进程、热节流等。如果只跑一次，可能碰巧测到一个异常快或异常慢的值，结论不可靠。

所以 benchmark 需要**重复运行多次**（比如 20～200 次），然后取统计量：
- **mean（平均值）**：代表典型表现，但受极端值影响。
- **median（中位数）**：排序后中间的值，对极端值不敏感，比 mean 更稳健。
- **min（最小值）**：最快的那次，最接近“没有外部干扰时的真实性能”，常用于估算带宽上限。

Python 示例（统计汇总）：

```python
import statistics

times_ms = [0.345, 0.348, 0.342, 0.351, 0.347]  # 5 次重复测量

print(f"mean : {statistics.mean(times_ms):.4f} ms")
print(f"median: {statistics.median(times_ms):.4f} ms")
print(f"min  : {min(times_ms):.4f} ms")
```

> 面试要点：benchmark 要重复运行多次，因为 GPU 计时有波动。统计量选用 median（稳健）或 min（最接近真实性能），不要只信单次运行结果。

---

## Q12. 怎么计算有效带宽？（bytes / time）

**回答**

有效带宽（effective bandwidth）衡量的是“实际利用了显存带宽的多少”。公式非常简单：

```
有效带宽 = 访问的字节数 / 耗时
```

对于向量加法 `c[i] = a[i] + b[i]`，每个元素涉及：
- 读 `a[i]`：4 字节（float32）
- 读 `b[i]`：4 字节
- 写 `c[i]`：4 字节
- 合计：每个元素 12 字节（两读一写）

所以总访问字节数 = `元素个数 × 3 × 4`。

Python 示例：

```python
size = 1 << 24          # 16,777,216 个元素
min_ms = 0.335          # 最快一次耗时（秒）
bytes_moved = size * 3 * 4  # 两读一写，每个元素 12 字节
bandwidth_gb_s = bytes_moved / (min_ms / 1000) / 1e9

print(f"有效带宽：{bandwidth_gb_s:.1f} GB/s")
```

> 面试要点：有效带宽＝访问字节数 ÷ 耗时。向量加法每个元素访问 12 字节（两读一写）。通常用 min 时间来估算带宽，因为 min 最接近“无外部干扰时的真实性能”。

---

## Q13. 有效带宽计算的公式和口径是什么？（两读一写）

**回答**

有效带宽的计算口径需要明确“算多少字节”，不同算子口径不同：

| 算子 | 访问模式 | 每个元素字节数（float32） | 公式 |
|------|---------|------------------------|------|
| 向量加法 `c=a+b` | 读 a + 读 b + 写 c | 12 | `size × 3 × 4` |
| 向量复制 `y=x` | 读 x + 写 y | 8 | `size × 2 × 4` |
| 点积 `dot` | 读 a + 读 b + 写 标量 | 8（标量忽略不计） | `size × 2 × 4` |

**关键原则**：口径要固定，同一种算子前后对比时不能换口径，否则带宽数字不可比。

ASCII 图示（向量加法的字节流向）：

```
内存                          GPU
┌─────────┐                  ┌─────────┐
│ a[i] 4B │──── read ───────▶│  算单元   │
│ b[i] 4B │──── read ───────▶│  c=a+b    │
│         │                  │           │
│ c[i] 4B │◀──── write ──────│  结果     │
└─────────┘                  └─────────┘
  12 字节/元素（两读一写）
```

> 面试要点：有效带宽口径要固定。向量加法按“两读一写”算，每个 float32 元素 12 字节；向量复制按“一读一写”算，8 字节。同一种算子前后对比时，公式不能换。

---

## Q14. 测量结果里 mean、median、min 分别代表什么？该看哪个？

**回答**

在多次重复测量的时间列表里：

- **mean（平均值）**：所有时间相加除以次数。代表“平均表现”，但被特别慢或特别快的一次拉得比较厉害。
- **median（中位数）**：把所有时间排个序，中间那个值。对极端值不敏感，比 mean 更稳健，推荐作为“典型性能”的报告值。
- **min（最小值）**：最快的那一次。最接近“没有外部干扰时的理想性能”，常用于估算带宽上限。

**该看哪个？**
- 要报告“典型延迟” → 看 **median**。
- 要报告“带宽上限/最佳性能” → 看 **min**。
- 要判断“是否有异常波动” → 对比 mean 和 median 的差距，差距大说明有抖动。

> 面试要点：mean 代表平均，median 更稳健（推荐报告典型延迟），min 最接近理想性能（用于带宽上限）。不要只信单次运行，要看多次重复的统计分布。

---

## Q15. 第一次运行为什么通常很慢？（初始化、编译、缓存）

**回答**

GPU 程序第一次运行慢，是正常现象，主要原因有三：

1. **JIT 编译**：HIP/PyTorch 可能第一次遇到某个 kernel 时，才把源码或中间表示编译成目标设备代码（hsaco），这个编译过程耗时。
2. **运行时初始化**：HIP 上下文、显存分配器、CUDA 图等首次建立有开销。
3. **缓存冷启动**：显存、L1/L2 缓存都是空的，第一次访问数据要走慢路径；时钟频率也可能在低功耗状态，需要升频。

**应对方法**：正式计时前一定要 warmup（预热），把这些一次性开销跑掉，再开始记录时间。

ASCII 图示（首次 vs 后续）：

```
首次运行：                      后续运行：
┌─────────────┐                 ┌─────────────┐
│ JIT 编译    │（慢）            │ 直接执行    │（快）
│ 上下文初始化 │（慢）            │ 缓存命中    │（快）
│ 时钟升频    │（慢）            │ 已在高频    │（快）
└─────────────┘                 └─────────────┘
   ↓                            ↓
  耗时长                        耗时短
   不计入统计                   计入统计
```

> 面试要点：第一次运行慢是因为 JIT 编译、运行时初始化、缓存冷启动。这是正常现象，正式计时前必须 warmup，把首次开销跑掉，不计入统计。

---

## Q16. wall-clock 时间和 GPU event 时间有什么区别？

**回答**

- **wall-clock 时间（wall time）**：用 `time.time()` 或 `time.perf_counter()` 测量的“钟表时间”。它从 CPU 调用开始算起，到 CPU 收到返回才结束。中间包含 CPU 把命令塞进队列的时间、操作系统调度、以及GPU 执行时间。因为 GPU 是异步的，wall-clock 往往只量到了“命令提交”，没量到“GPU 真正干完”。
- **GPU event 时间**：用 GPU 自己的 event 机制测量，在设备时间线上打两个时间戳，只量 **GPU 真正执行 kernel 的时间**，不包含 CPU 提交命令的开销。

对比表：

| 指标 | 测什么 | 适合场景 |
|------|--------|---------|
| wall-clock | CPU 视角的端到端耗时 | 整个程序的总体耗时（含数据拷贝、CPU 预处理） |
| GPU event | GPU 视角的 kernel 耗时 | 单个 kernel 或算子的精确性能 |

> 面试要点：wall-clock 是 CPU 视角的“钟表时间”，包含异步提交开销；GPU event 是设备视角的纯计算时间。给 GPU kernel 计时，必须用 GPU event，不能用 wall-clock。

---

## Q17. 什么是 JIT 编译？它对第一次运行有什么影响？

**回答**

JIT（Just-In-Time，即时）编译是指程序在运行时，才把中间代码编译成目标设备能执行的机器码。

在 ROCm/HIP 里，kernel 首次被调用时，可能需要把源码或中间表示（如 HSACO、MLIR）编译成针对当前 GPU 架构（如 gfx1201）的指令。这个编译过程发生在第一次运行时，会显著拖慢首次耗时。

**影响**：首次运行可能比后面慢几倍甚至几十倍。如果不 warmup 直接计时，会把 JIT 编译的时间也算进去，导致结果严重偏高，不具备可比性。

> 面试要点：JIT 编译是“运行时再编译成机器码”，首次运行 kernel 时会发生，耗时较长。正式 benchmark 前要 warmup，避免把 JIT 开销算进性能统计。

---

## Q18. GPU 时钟频率波动会影响测量吗？怎么应对？（DPM、热节流）

**回答**

会。GPU 的时钟频率不是固定不变的，它会根据负载、温度、功耗动态调整（这就是 DPM，Dynamic Power Management）。比如：
- 刚开始跑时，GPU 可能在低频待机，第一次升频需要时间；
- 跑久了芯片变热，可能触发热节流（thermal throttling），频率被迫降低，导致后面运行变慢。

这会导致同一段代码在不同时间点测出来的时间不一致。

**应对方法：**
1. **warmup 时让 GPU 升频**：预热时跑几次，让时钟频率爬到高性能状态。
2. **多次重复取 median/min**：抵消单次波动。
3. **固定时钟频率（高级）**：如果需要极严苛的对比，可以锁定 GPU 频率（如通过 `rocm-smi` 或环境变量），但这通常只在实验室环境做。

> 面试要点：GPU 时钟频率会因负载和温度波动（DPM/热节流），影响测量结果。应对方法：warmup 升频、多次重复取 median/min、严苛场景下锁定频率。

---

## Q19. 怎么设计一个公平的 A/B 对照实验？（相同输入、相同 block 大小、相同 warmup/repeat、新进程）

**回答**

要证明“优化后的版本比原版快”，必须做公平的 A/B 对照。公平的关键是**只改一个变量，其余全部保持一致**：

1. **相同输入规模**：同一个数组长度、同一个 dtype。
2. **相同 block 大小**：`block=256` 两边都用，不要一边 256 一边 512。
3. **相同 warmup/repeat**：两边都 warmup 20 次、repeat 100 次。
4. **同一机器、同一状态**：不要一边跑在空闲机上，另一边跑在有人在跑别的任务的机器上。
5. **新进程跑基线**：为了避免进程内状态（如 JIT 缓存、时钟频率）互相干扰，最好在**新进程**里分别跑 A 和 B，再对比。

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

## Q20. 可信 benchmark 的检查清单是什么？（固定规模、warmup、repeat、同步、min/median、区分初始化和正式计时）

**回答**

做一个可信的 GPU benchmark，按这个清单自查：

```
□ 输入规模固定：两边用同一个数组长度、同一个 dtype
□ block 大小一致：两边都用 block=256（或其他同一值）
□ warmup 足够：正式计时前跑 10～20 次不计入统计
□ repeat 足够：正式跑 30～200 次，取 median/min
□ GPU event 计时：不用 time.time()，用 torch.cuda.Event / hipEvent
□ synchronize 后再读：所有 event 记完后调用 torch.cuda.synchronize()
□ 区分初始化和正式计时：首次运行的 JIT/分配开销不计入
□ 新进程对照：A/B 对照时最好分进程，避免状态污染
□ 固定口径：带宽计算公式两边一致（如都按“两读一写”）
```

> 面试要点：可信 benchmark 检查清单＝固定输入规模→一致 block 大小→足够 warmup→足够 repeat→GPU event 计时→synchronize 后读→区分初始化和正式计时→新进程对照→固定带宽口径。每一条缺了，结果都可能不可信。

---

## Q21. HIP 里的 __global__、__device__、__host__ 有什么区别？

**回答**

HIP 里有三个常见的函数限定词，决定了函数在哪里编译、在哪里运行、谁能调用：

| 限定词 | 编译在 | 运行在 | 调用者 | 大白话 |
|--------|--------|--------|--------|--------|
| `__global__` | GPU 设备码 | GPU | CPU 发起，GPU 执行 | “核函数”，GPU 上干活的主力 |
| `__device__` | GPU 设备码 | GPU | 只能被 GPU 里的其他函数调用 | GPU 端的“私有助手函数” |
| `__host__` | CPU 主机码 | CPU | CPU 调用 | 普通的 CPU 函数，默认就是这个 |

可以组合使用，比如 `__host__ __device__` 表示既能在 CPU 编译运行，也能在 GPU 编译运行。

> 面试要点：`__global__` 是 GPU 核函数（CPU 发起、GPU 执行）；`__device__` 是 GPU 端私有助手函数（只能被 GPU 函数调用）；`__host__` 是普通 CPU 函数。可组合为 `__host__ __device__` 表示两端都能用。

---

## Q22. blockIdx/threadIdx/blockDim/gridDim 怎么用来算全局下标？

**回答**

在 HIP kernel 里，每个线程需要知道自己负责处理数组的哪个位置。这靠四个内置变量：

- `blockIdx.x`：当前 block 在 grid 里的索引（第几组）。
- `blockDim.x`：每个 block 里有多少个线程。
- `threadIdx.x`：当前线程在 block 里的索引（第几个）。
- `gridDim.x`：grid 里总共有多少个 block。

**一维全局下标公式：**
```
全局下标 = blockIdx.x × blockDim.x + threadIdx.x
```

二维和三维同理，把 x/y/z 分别乘加即可。

C++ 示例：

```cpp
__global__ void my_kernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * 2.0f;
    }
}
```

ASCII 图示（二维 grid 的下标计算）：

```
gridDim.x = 2, blockDim.x = 3

blockIdx.x=0: threadIdx.x=0→0, 1→1, 2→2
blockIdx.x=1: threadIdx.x=0→3, 1→4, 2→5

公式：idx = blockIdx.x × 3 + threadIdx.x
```

> 面试要点：全局下标＝blockIdx.x × blockDim.x + threadIdx.x。四个变量分别是块索引、块内线程数、线程内索引、总块数。二维/三维按各分量分别计算。

---

## Q23. 怎么处理数组长度不是 block 大小整数倍的情况？（if (idx < n) 边界检查）

**回答**

启动 kernel 时，grid 大小通常是 `ceil(n / block)`，向上取整。这会导致最后一个 block 里可能有一些线程的全局下标超过了数组长度 `n`。如果不处理，这些线程会访问非法内存，导致段错误或错误结果。

解决办法：在 kernel 里加边界检查 `if (idx < n)`，让超出范围的线程什么都不做（相当于空转）。

C++ 示例：

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {           // 边界检查：只处理有效下标
        c[idx] = a[idx] + b[idx];
    }
}
```

假设 `n=10`，`block=4`，`grid=3`（共12个线程）：

```
线程 idx：0  1  2  3  4  5  6  7  8  9  10 11
有效?    : ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✗  ✗
                     │                    │
                     │                    └─ 超出 n=10，不执行
                     └─ 最后一个有效下标
```

> 面试要点：数组长度往往不是 block 大小的整数倍，最后一个 block 会有超出范围的线程。必须在 kernel 里加 `if (idx < n)` 边界检查，避免越界访问。

---

## Q24. GPU 程序和 CPU 程序的错误处理有什么区别？（HIP_CHECK、hipError_t）

**回答**

CPU 程序出错通常会抛异常或返回错误码，程序员可以用 `try/catch` 或 `if (err)` 处理。GPU 程序不同：kernel 在 GPU 上异步执行，如果 kernel 里出错（比如越界访问），GPU 可能直接挂掉，而 CPU 端可能不知道。

HIP 的做法是：**大部分 HIP API 返回 `hipError_t` 错误码**，需要程序员显式检查。常见做法是封装一个 `HIP_CHECK` 宏：

```cpp
#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                             \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, hipGetErrorString(err));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// 使用示例
HIP_CHECK(hipMalloc(&d_a, bytes));
HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipGetLastError());  // 启动 kernel 后，检查 kernel 启动本身是否成功
```

注意：`kernel<<<...>>>` 启动本身是异步的，启动成功只表示“命令发进队列了”，不代表 kernel 执行正确。kernel 内部的运行时错误（如越界）需要在后续 `hipDeviceSynchronize()` 或下一次 HIP API 调用时通过 `hipGetLastError()` 捕获。

> 面试要点：HIP API 返回 `hipError_t`，用 `HIP_CHECK` 宏显式检查。kernel 启动是异步的，启动成功不代表执行正确，后续需配合 `hipDeviceSynchronize()` + `hipGetLastError()` 捕获运行时错误。

---

## Q25. 什么是 hipDeviceSynchronize？什么时候需要调用？

**回答**

`hipDeviceSynchronize()` 是把 CPU 线程挂起，直到 GPU 上所有之前提交的任务全部完成。它是 CPU 和 GPU 之间的“同步点”。

需要调用的场景：
1. **kernel 启动后取结果**：启动 kernel 是异步的，拷回结果前必须 `hipDeviceSynchronize()`，确保 GPU 算完了。
2. **GPU event 计时**：`end.record()` 后，必须 `hipDeviceSynchronize()`（或 PyTorch 里的 `torch.cuda.synchronize()`），再读 `elapsed_time()`，否则拿到的是旧值。
3. **错误检查**：`hipGetLastError()` 捕获 kernel 错误前，通常要先同步，确保 kernel 确实跑完了。

C++ 示例：

```cpp
vector_add<<<grid, block>>>(d_a, d_b, d_c, n);  // 异步启动，立即返回
hipDeviceSynchronize();                         // 等 GPU 算完
HIP_CHECK(hipGetLastError());                   // 再检查是否有错误
```

> 面试要点：`hipDeviceSynchronize()` 让 CPU 等 GPU 全部任务完成。在 kernel 启动后取结果、GPU event 计时读数、错误检查前，都需要先同步。

---

## Q26. 本章小结：性能计时的核心原则清单

**回答**

本章讲了怎么写第一个 HIP 程序，以及怎么可信地测它的性能。核心原则清单如下：

```
性能计时核心清单
┌─────────────────────────────────────────────────────────┐
│ 1. GPU 任务是异步的，绝不用 time.time() 给 kernel 计时  │
│ 2. 必须用 GPU event（record → synchronize → elapsed）   │
│ 3. 正式计时前必须 warmup，跑掉 JIT/初始化开销           │
│ 4. 重复运行多次（repeat），取 median/min 而非单次       │
│ 5. 带宽＝字节数÷耗时，口径要固定（如两读一写）          │
│ 6. A/B 对照：只改一个变量，其余全部锁死（新进程最好）   │
│ 7. 区分 wall-clock（CPU视角）和 GPU time（设备视角）    │
│ 8. kernel 启动后取结果或检错误，先 hipDeviceSynchronize │
└─────────────────────────────────────────────────────────┘
```

一句话总结：**用 GPU event 计时、warmup 后再统计、多次重复取 median/min、带宽口径固定、A/B 对照只改一个变量——这是可信 GPU 性能测量的四铁律。**

> 面试要点：本章核心＝GPU event 计时（不用 time.time）→warmup→repeat 取 median/min→带宽公式固定→A/B 对照只改一个变量→wall-clock 与 GPU time 区分→kernel 启动后同步。这是做可信 GPU 性能测量的基础。

---

## Q27. 下一章（Profiling 与 rocprof）需要带什么基础过去？

**回答**

本章完成后，你已经会写 HIP 程序，也能用 GPU event 可信地测一个 kernel 的耗时和带宽。下一章《GPU Profiling 与性能优化实战（rocprof、kernel trace、Roofline 图）》会把视角从“快不快”转到“慢在哪里”，需要你带着以下基础过去：

1. **会写并编译 HIP kernel**：下一章会拿 `vector_add` 做对照实验，比较不同写法的快慢。
2. **懂 GPU event 计时和 warmup/repeat**：下一章的公平对照实验会直接复用这套计时骨架。
3. **知道有效带宽怎么算**：下一章用 Roofline 图时，需要把测得的点画在“带宽利用率”坐标轴上。
4. **知道 block/grid 和线程编号**：下一章分析 `linecross` 等对照实验时，需要理解线程是怎么访问内存的。

不需要你已经懂 rocprof、Roofline、kernel trace 这些工具，那些是下一章要讲的。只要你能写出一个能跑通、能算带宽的 HIP 向量加法，就有足够的基础进入下一章。

> 面试要点：下一章需要带的基础＝会写 HIP kernel 并编译、会用 GPU event 计时和 warmup/repeat、知道有效带宽计算、明白 block/grid 和线程编号。rocprof、Roofline 等工具是下一章内容，不需要预先掌握。

---

## Q28. 下一章（Profiling 与 rocprof）需要带什么基础过去？

**回答**

（本题与 Q27 内容重复，作为章末兜底确认。）下一章《GPU Profiling 与性能优化实战》需要你已经掌握：HIP 程序能跑通、GPU event 计时能用、有效带宽能算、block/grid 线程编号明白。这些是下一章用 rocprof 做公平对照、读 kernel trace、画 Roofline 图的前提。

> 面试要点：与 Q27 相同，确认下一章 rocprof/Roofline 的前置基础已具备。

---

## Q29. HIP 错误处理 HIP_CHECK 怎么用？

**回答**

（本题对 Q24 的补充强调，作为章末兜底。）HIP 的大部分 API 返回 `hipError_t`，最稳妥的做法是封装 `HIP_CHECK` 宏并在每个 API 调用后使用：

```cpp
#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                             \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, hipGetErrorString(err));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

HIP_CHECK(hipMalloc(&d_a, bytes));
HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
hipDeviceSynchronize();
HIP_CHECK(hipGetLastError());  // 捕获 kernel 运行时错误
```

> 面试要点：用 `HIP_CHECK` 宏包裹每个 HIP API 调用；kernel 启动后必须 `hipDeviceSynchronize()` 再用 `hipGetLastError()` 捕获运行时错误。

---

## Q30. GPU 计时的精度是多少级别？（微秒 μs）

**回答**

GPU event 的精度通常在 **微秒（μs）级别**，足以测量大部分 GPU kernel（通常耗时从几十微秒到几毫秒不等）。它远比 `time.time()` 适合测 GPU，因为 `time.time()` 受操作系统调度和 CPU 线程挂起影响，粒度可能只有毫秒甚至更粗，且量的是 CPU 视角的时间。

如果需要更细粒度的纳秒级分析（比如想知道某条指令花了多少个周期），需要用硬件性能计数器（rocprof 的 `--pmc` 或 kernel trace），那已经是下一章 Profiling 的范畴了，不属于本章的 event 计时。

> 面试要点：GPU event 计时精度在微秒（μs）级，足够测量绝大多数 GPU kernel。纳秒级或指令级分析需要硬件性能计数器，属于下一章 Profiling 内容。
