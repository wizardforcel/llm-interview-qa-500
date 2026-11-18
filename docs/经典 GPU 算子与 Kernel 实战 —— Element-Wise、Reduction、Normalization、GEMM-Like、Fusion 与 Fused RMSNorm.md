# 经典 GPU 算子与 Kernel 实战 —— Element-Wise、Reduction、Normalization、GEMM-Like、Fusion 与 Fused RMSNorm

> 面试生普（LLM 500 题）· 经典 GPU 算子与 Kernel 实战篇。本章覆盖六类经典算子的实现原理与优化思路：Element-Wise（逐元素算子、依赖关系、float4 向量化、边界检查、HIP 与 Triton 对照）、Reduction（归约算子、求和树、数据竞争、atomic baseline、LDS 局部和、两阶段归约）、Normalization（归一化算子、LayerNorm/RMSNorm、方差与 epsilon、在线算法、融合）、GEMM-Like（矩阵乘类算子、点积、分块 tiling、共享内存复用、MFMA/张量核心、内存带宽 vs 算力）、Fusion（融合算子、LayerNorm+残差+激活融合、寄存器压力、融合边界、FlashAttention 思路）、Fused RMSNorm（综合实战、方差+归一化+缩放融合、二阶段归约、代码、性能对照），共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整代码和 ASCII 图。

---

## Q1. 什么是 Element-Wise（逐元素）算子？它的核心特征是什么？

**回答**

Element-Wise（逐元素）算子是指输出的每个位置 `i` 只依赖输入的同一位置 `i`，不依赖其他位置。不同输出位置之间没有数据依赖，可以彼此独立计算。

最典型的例子是向量加法 `C[i] = A[i] + B[i]`：想算 `C[3]`，只需要 `A[3]` 和 `B[3]`，跟 `A[0]`、`A[1]` 等其他位置无关。

因为没有跨位置依赖，逐元素算子天然适合并行：每个线程/每个 program 处理一块输出位置，互不干扰。

同一模式的算子还有：
- ReLU：`Y[i] = max(X[i], 0)`
- Scale & Bias：`Y[i] = alpha * X[i] + beta`
- Clamp：`Y[i] = min(max(X[i], low), high)`

ASCII 图示（逐元素依赖）：

```
输入 A：  A[0]  A[1]  A[2]  A[3]  A[4]  ...
输入 B：  B[0]  B[1]  B[2]  B[3]  B[4]  ...
输出 C：  C[0]  C[1]  C[2]  C[3]  C[4]  ...
            ↑       ↑       ↑       ↑       ↑
            │       │       │       │       │
            └───────┴───────┴───────┴───────┘
              每个输出只依赖同一位置的两个输入
```

> 面试要点：Element-Wise 算子的核心特征是“输出位置 i 只依赖输入位置 i”，输出之间没有数据依赖，天然适合按位置并行计算。

---

## Q2. 用 HIP 写向量加法 kernel，代码长什么样？

**回答**

下面是一个基于标量线程的 HIP 向量加法 kernel，每个线程处理一个元素：

```cpp
// vector_add_scalar.hip
#include <hip/hip_runtime.h>

__global__ void vector_add_scalar(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {                    // 边界检查：防止最后一个 block 越界
        c[idx] = a[idx] + b[idx];
    }
}
```

关键点：
- `blockIdx.x * blockDim.x + threadIdx.x`：算出当前线程负责的全局下标。
- `if (idx < n)`：边界检查。因为 grid 大小通常向上取整，最后一个 block 可能会有越界线程，必须防止它们访问非法内存。

启动示例：

```cpp
int n = 1 << 24;                      // 16,777,216 个元素
int block = 256;                      // 每个 block 256 个线程
int grid = (n + block - 1) / block;   // 向上取整，计算需要多少个 block
vector_add_scalar<<<grid, block>>>(d_a, d_b, d_c, n);
```

> 面试要点：HIP 向量加法 kernel 的核心是 `idx = blockIdx.x * blockDim.x + threadIdx.x` 算全局下标，并用 `if (idx < n)` 做边界检查。

---

## Q3. 什么是 float4 向量化？它对 Element-Wise 算子有什么好处？

**回答**

float4 是一种“短向量类型”，把 4 个 float 打包成一个 128 位（16 字节）的载体，一次 load/store 指令就能搬运 4 个连续 float，而不是 4 次分别搬运 1 个 float。

好处：
1. **访存指令数减为 1/4**：每条 load/store 指令搬运 4 个元素，指令发射压力降为 1/4。
2. **每线程 ILP 提升**：每个线程一次拿到 4 个独立输入，指令级并行（ILP）翻两番。
3. **地址依然连续**：vec4 线程 t 处理 `[4t, 4t+3]`，warp 内 32 个线程覆盖连续的 128 字节（4 个 sector），沿着带宽利用率仍能保持在 100%。

向量化与 coalescing（访存合并）是正交的：前者减指令，后者减无效字节；最好的情况是两者兼得——一条 128-bit load 恰好铺满 4 个整 sector。

C++ 示例（float4 向量加法）：

```cpp
__global__ void vector_add_vec4(const float4* a, const float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {                      // n 这里是 float4 个数，必须是 4 的倍数
        c[idx] = a[idx] + b[idx];       // float4 支持直接 + 运算
    }
}
```

注意：`n` 这里表示 float4 个数，必须是原数组长度的 4 倍数；若原数组长度不是 4 的倍数，需要额外处理尾部（见下一题）。

> 面试要点：float4 向量化把 4 个 float 打包成一次 128-bit load/store，访存指令数降为 1/4，每线程 ILP 提升，地址依然连续。与 coalescing 正交：前者减指令，后者减无效字节。

---

## Q4. 向量化时 N 不是 4 的倍数，怎么处理尾部？

**回答**

当原数组长度 `N` 不是 4 的倍数时，直接用 float4 处理会在最后一组越界读取（比如只剩 2 个元素，却读了 4 个）。必须做尾部处理。

安全范式（推荐）：采用“向量主路径 + 标量尾部”的两段式：

```cpp
__global__ void vector_add_vec4_safe(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx;                      // float4 索引
    int vec_n = n / 4;                      // 完整的 float4 组数

    if (vec_idx < vec_n) {
        // 主路径：完整的 float4，安全向量加载
        float4 va = reinterpret_cast<const float4*>(a)[vec_idx];
        float4 vb = reinterpret_cast<const float4*>(b)[vec_idx];
        reinterpret_cast<float4*>(c)[vec_idx] = va + vb;
    } else if (idx < n) {
        // 尾部：标量处理剩余 1～3 个元素
        c[idx] = a[idx] + b[idx];
    }
}
```

关键：
- `vec_idx < vec_n` 走 float4 主路径，保证 16 字节对齐且不越界。
- 剩下的 1～3 个元素用标量路径处理，避免越界读。

> 面试要点：float4 向量化时 N 不是 4 的倍数，必须处理尾部。安全范式是“`vec_idx < vec_n` 走 float4 主路径，剩余 1～3 个元素用标量处理”，避免越界读。

---

## Q5. 用 Triton 写向量加法，代码长什么样？与 HIP 有什么区别？

**回答**

Triton 是一种 Python 风格的 GPU 编程语言，抽象层级比 HIP 更高。它不让程序员管“当前 thread 得到哪个标量下标”，而是让程序员思考“当前 program 生成哪一块下标”。

Triton 向量加法：

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    input_a_ptr,
    input_b_ptr,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)                       # 当前 program 编号
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 该 block 的下标
    valid = offsets < size                              # 边界掩码

    input_a = tl.load(input_a_ptr + offsets, mask=valid, other=0.0)
    input_b = tl.load(input_b_ptr + offsets, mask=valid, other=0.0)
    output = input_a + input_b

    tl.store(output_ptr + offsets, output, mask=valid)
```

Host 端启动：

```python
import triton
import torch

def launch(input_a, input_b, output, size, BLOCK_SIZE=256):
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    vector_add_kernel[grid](
        input_a, input_b, output, size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
```

与 HIP 的对照：

| 问题 | HIP 写法 | Triton 写法 |
|------|---------|------------|
| 保护尾部 | `if (idx < n)` | `mask=valid, other=0.0` |
| 算全局下标 | `blockIdx.x * blockDim.x + threadIdx.x` | `program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` |
| 思维模型 | 当前 thread 得到哪个标量下标 | 当前 program 生成哪一块下标 |

两者最后都要回答同一件事：**读了哪些地址，算了什么，写回哪里。**

> 面试要点：Triton 向量加法用 `program_id` 算 block 起点，用 `tl.arange` 生成块内局部下标，用 `mask` 保护尾部。与 HIP 的区别是思维模型：HIP 从“当前 thread 得到哪个标量下标”出发，Triton 从“当前 program 生成哪一块下标”出发。

---

## Q6. 什么是 Reduction（归约）算子？它和逐元素算子有什么根本区别？

**回答**

归约（Reduction）算子把多个输入合并成一个（或少量）输出，比如把一个数组加成一个标量求和 `y = sum(x_i)`。与逐元素算子不同，归约的输出依赖**所有**输入，线程之间需要协作和同步。

根本区别：
- **逐元素**：输出 `C[i]` 只依赖 `A[i]`、`B[i]`，位置之间无依赖，可完全独立并行。
- **归约**：输出 `y` 依赖所有 `x[0]...x[N-1]`，线程间需要交换中间结果，存在跨线程数据依赖。

ASCII 图示（逐元素 vs 归约）：

```
逐元素（无跨位置依赖）：           归约（有跨位置依赖）：
C[0]=A[0]+B[0]                   y = x[0]+x[1]+...+x[N-1]
C[1]=A[1]+B[1]                       ↑
C[2]=A[2]+B[2]                   ┌───┴───┬───┬───┬───┐
C[3]=A[3]+B[3]                   │  x[0] │x[1]│x[2]│...│x[N-1]│
   ↑   ↑   ↑   ↑                   └───┴───┴───┴───┴───┘
   │   │   │   │                           │
   └───┴───┴───┘                   所有输入汇聚到一个输出
   每个输出独立                   需要线程间协作（求和树、原子加等）
```

> 面试要点：归约算子的输出依赖所有输入，线程间有数据依赖，需要协作（求和树、原子加等）；逐元素算子的输出只依赖同一位置输入，线程间无依赖。

---

## Q7. 什么是求和树？它怎么把串行依赖变成并行？

**回答**

如果把 N 个数依次相加 `((((x0+x1)+x2)+x3)+...)`，需要 N-1 次加法，且每次都依赖上一次的结果，串行依赖很深，无法并行。

求和树把同样的 N-1 次加法重新排列成“树”形，让互不依赖的加法放在同一轮做，深度从 O(N) 降到 O(log N)。

以 8 个数 `[3, 1, 7, 0, 4, 1, 6, 2]` 为例：

```
输入：      [3, 1, 7, 0, 4, 1, 6, 2]
第1轮：      [4,  7,  5,  8]       (3+1, 7+0, 4+1, 6+2)
第2轮：      [11, 13]              (4+7, 5+8)
第3轮：      [24]                   (11+13)
```

每一轮的加法互不依赖，可以并行执行；只需 log2(N) 轮就能得到最终结果。

> 面试要点：求和树把串行的依赖链重新排列成树形，让同一轮的加法互不依赖并行执行，深度从 O(N) 降到 O(log N)。

---

## Q8. 归约算子中的数据竞争是什么？怎么避免？

**回答**

数据竞争（Data Race）是指多个线程未经协调地访问同一内存位置，且至少有一次写入，导致结果取决于执行顺序。

在归约中，最直接的想法是“每个线程把自己的值加到同一个全局变量 `output`”上：

```cpp
// 错误示范：没有原子保护的并行累加
if (idx < size) {
    output[0] += input[idx];   // 多个线程同时读旧值、加、写回，会丢失更新
}
```

两个线程可能都读到旧值 `0`，分别算出 `3` 和 `1`，最后只写回其中一个，正确答案 `4` 丢失。

**避免方法：**
1. **原子加法 `atomicAdd`**：把“读旧值、相加、写回”封装成不可拆开的原子操作，避免丢失更新，但所有线程竞争同一地址，有热点。
2. **局部归约 + 原子合并**：先在每个 block 内用 LDS/共享内存做树归约，只让每个 block 的 thread 0 把块局部和原子加到全局，把全局竞争从 N 次降到 block 数次。

> 面试要点：归约中的数据竞争是指多个线程未经协调访问同一输出位置，导致丢失更新。避免方法：用 `atomicAdd` 保证原子性，或更推荐地“先在 block 内局部归约，再让每个 block 的 thread 0 原子加到全局”，把全局竞争从 N 次降到 block 数次。

---

## Q9. HIP 怎么用 LDS（共享内存）做块内归约？

**回答**

LDS（Local Data Share，HIP 里用 `__shared__` 声明）是片上高速 SRAM，供同一个 block 内的线程共享。块内归约的思路是：

1. 每个线程把自己的值写入 LDS。
2. `__syncthreads()` 屏障，确保所有线程写完。
3. 按 stride 半减（stride /= 2）依次相加：stride=blockDim/2 时，线程 0 加 thread 0+stride，线程 1 加 thread 1+stride，以此类推；每轮后 `__syncthreads()`。
4. 最后 thread 0 把 `shared[0]`（块局部和）原子加到全局输出。

C++ 示例：

```cpp
__global__ void block_reduce_kernel(const float* input, float* output, int n) {
    extern __shared__ float shared[];   // 动态共享内存，大小由启动时指定
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (idx < n) ? input[idx] : 0.0f;  // 越界位置填加法单位元 0
    __syncthreads();

    // 树归约：stride 半减
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);  // 把块局部和加到全局
    }
}
```

ASCII 图示（块内树归约）：

```
共享内存（初始）：[3, 1, 7, 0, 4, 1, 6, 2]
 stride=4：      [4,  7,  5,  8,  _,  _,  _,  _]   (0+4, 1+1, 7+0... 不对，是 3+4, 1+1... 实际上相邻配对)
 stride=2：      [11, 13,  _,  _,  _,  _,  _,  _]   (4+7, 5+8)
 stride=1：      [24,  _,  _,  _,  _,  _,  _,  _]   (11+13)
thread 0 把 shared[0]=24 原子加到全局输出
```

更准确的相邻配对图示：

```
第1轮 (stride=4，相邻距离4)：3+4=7, 1+1=2, 7+0=7...  ← 实际 HIP 代码是相邻配对 shared[tid]+shared[tid+stride]
```

> 面试要点：HIP 块内归约用 `__shared__` LDS：每个线程写入共享数组，`__syncthreads()` 屏障，按 stride 半减做树归约，最后 thread 0 把块局部和 `atomicAdd` 到全局。越界位置填加法单位元 0，不影响结果。

---

## Q10. Triton 怎么做归约？`tl.sum` 是什么？

**回答**

Triton 做归约通常分两阶段：
1. **程序局部归约**：每个 program（对应一个 tile）把自己负责的一块数据在线程内/ warp 内归约成一个部分和（partial）。
2. **第二阶段归约**：把所有 program 的部分和再归约成最终结果。

Triton 提供 `tl.sum(x, axis)` 对指定轴做归约，返回降维后的标量或向量。

Triton 归约示例（单 program，适合中小张量）：

```python
import triton
import triton.language as tl

@triton.jit
def reduce_then_epilogue(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    acc = tl.zeros((), dtype=tl.float32)   # 寄存器里的累加器
    offs = tl.arange(0, BLOCK)
    for start in range(0, n, BLOCK):
        idx = start + offs
        mask = idx < n
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)            # 该 tile 的局部和
    tl.store(out_ptr, acc / n)              # 最后除以 n 得到均值（示例）
```

对于大张量，需要多个 program 各算一个 partial，再启动第二个 kernel 把 partial 汇总成最终结果。

> 面试要点：Triton 归约用 `tl.sum(x, axis)` 对轴做归约。大张量通常分两阶段：每个 program 先算 tile 局部和（partial），再用第二个 program 把所有 partial 汇总成最终结果。

---

## Q11. 归约中的浮点数不满足结合律，怎么处理正确性？

**回答**

实数中 `(a+b)+c = a+(b+c)`，但有限精度浮点数（FP32）中不一定相等。不同 block 数、不同树形顺序、或者用 Warp Shuffle 还是 LDS，都可能改变加法顺序，导致舍入误差不同，并行归约结果通常不能用 `bitwise equality` 与串行 FP32 结果比较。

本章采用的裁判规则：
- 输入类型：FP32。
- 教学输入：交替的 +1 与 -1（在默认 shape 下保持 FP32 加法精确可检查）。
- CPU reference：Host 侧用 FP64 累加同一输入作为基准。
- 容差：绝对误差不超过 1e-3。
- 正式计时前后各跑一次正确性检查（precheck / postcheck）。

边界长度（如 1, 31, 32, 33, 255, 256, 257, 1027）会特意覆盖不是 2 的幂的情况，测试越界位置是否被当成加法单位元 0 正确处理。

> 面试要点：浮点加法不满足结合律，并行归约的结果通常不能与串行 FP32 结果 bitwise 相等。正确性判断应用 FP64 做参考基准，设置绝对误差容差（如 1e-3），并测试非 2 的幂长度（越界位置填 0）。

---

## Q12. 什么是 Normalization（归一化）算子？LayerNorm 和 RMSNorm 有什么区别？

**回答**

归一化算子把输入按一定规则缩放，使其均值和方差（或仅方差）符合某种目标，常用于深度学习中的层归一化。

- **LayerNorm**：`y = (x - mean) / sqrt(var + eps) * weight + bias`。既减均值又除标准差，还有可学习的 `weight`（缩放）和 `bias`（偏移）。
- **RMSNorm**：`y = x / sqrt(mean(x^2) + eps) * weight`。只除根方均方（RMS），不减均值，也没有 bias，计算更简单，在大模型（如 LLaMA、PaLM）中常用。

两者都需要先算出整个归一化维度上的统计量（均值、方差），再对每个元素做逐元素缩放，因此归约是归一化算子的核心子环节。

ASCII 图示（LayerNorm 数据流）：

```
输入向量 x：   [x0, x1, x2, x3, ...]
      │
      ▼
  ┌─────────────┐
  │ 归约求和    │──▶ mean = sum(x) / N
  │ 归约求平方和│──▶ var  = mean(x^2)
  └─────────────┘
      │
      ▼
  ┌─────────────────────────────┐
  │ 逐元素：(xi - mean) / sqrt(var+eps) * weight + bias │
  └─────────────────────────────┘
      │
      ▼
输出向量 y：[y0, y1, y2, y3, ...]
```

> 面试要点：LayerNorm 减均值除标准差并加偏置；RMSNorm 只除根方均方且无 bias，计算更简单。两者都需要先对归一化维度做归约（求和/求平方和），再做逐元素缩放。

---

## Q13. LayerNorm 的在线算法是什么？为什么要用它？

**回答**

传统做法需要两次遍历数据：第一次算均值，第二次算方差（需要知道均值后才能算 `(xi-mean)^2`）。这会把数据从显存读两次，访存开销大。

**在线算法（Welford 算法思路）**在一次遍历中同时累加 `sum` 和 `sum_of_squares`，最后用公式推导方差：

```
var = E[x^2] - (E[x])^2
```

即：
```
mean = sum / N
var = (sum_of_squares / N) - mean * mean
```

这样只需遍历数据一次，把两次 Pass 合并为一次，减少一半的全局内存读取，对带宽受限的归一化算子非常有意义。

ASCII 图示（两次遍历 vs 一次遍历）：

```
两次遍历（传统）：           一次遍历（在线算法）：
读 x[0..N-1] ──▶ 算 mean   读 x[0..N-1] ──▶ 同时累加 sum 和 sum_of_squares
读 x[0..N-1] ──▶ 算 var    │
                            │
                            ▼
                      算 mean = sum/N
                      算 var = sum_of_squares/N - mean^2
```

> 面试要点：LayerNorm 的在线算法在一次遍历中同时累加 `sum` 和 `sum_of_squares`，再用 `var = E[x^2] - (E[x])^2` 推导方差，把两次 Pass 合并为一次，减少一半全局内存读取。

---

## Q14. 什么是 GEMM（通用矩阵乘法）？它和 Element-Wise/Reduction 有什么区别？

**回答**

GEMM（General Matrix Multiply，通用矩阵乘法）计算 `C = A * B`，其中 `A` 是 M×K，`B` 是 K×N，`C` 是 M×N。输出 `C[i][j]` 依赖 `A` 的第 i 行和 `B` 的第 j 列，涉及 K 个元素的点积。

与 Element-Wise 和 Reduction 的对比：

| 算子类型 | 输出依赖 | 数据复用特征 | 典型算术强度 |
|---------|---------|-----------|-----------|
| Element-Wise | 仅同一位置 | 无复用，每个数据读一次写一次 | 很低（≈0.083 FLOP/B） |
| Reduction | 全部输入 | 部分复用（局部和） | 低～中 |
| GEMM | 一行 × 一列 | 高：A 的行和 B 的列可反复复用 | 高（大矩阵可达几十～上百 FLOP/B） |

大矩阵乘法中，`A` 的同一行可用于计算 `C` 的整一行，`B` 的同一列可用于计算 `C` 的整一列，通过分块（Tiling）把数据留在共享内存/缓存里反复使用，算术强度很高，是典型的计算密集型算子。

> 面试要点：GEMM 是矩阵乘法，输出 `C[i][j]` 依赖 `A` 的第 i 行和 `B` 的第 j 列。与逐元素（无依赖）和归约（全部输入汇聚）不同，GEMM 有高数据复用，大矩阵时算术强度很高，是计算密集型算子。

---

## Q15. GEMM 里什么是分块（Tiling）？为什么要用共享内存？

**回答**

大矩阵无法一次性全部放入片上缓存/共享内存。分块（Tiling）是指把大矩阵切成小方块（Tile），每次只把一个 `A` tile 和一个 `B` tile 加载到共享内存（LDS）里，在 block 内的线程协作下完成这两个 tile 的部分乘积，累加到寄存器里的 `C` tile，然后再加载下一对 tile。

这样：
- `A` 的一个 tile 可被 `C` 的一行内多个元素复用；
- `B` 的一个 tile 可被 `C` 的一列内多个元素复用；
- 大幅减少从显存反复读取同一数据的次数，提高算术强度。

ASCII 图示（GEMM 分块）：

```
矩阵 A (M×K)              矩阵 B (K×N)              矩阵 C (M×N)
┌─────────┐               ┌─────────┐               ┌─────────┐
│ A_tile  │               │ B_tile  │               │ C_tile  │
│ [M_tile │               │ [K_tile │               │ [M_tile │
│  ×K_tile]│               │  ×N_tile]│               │  ×N_tile]│
└─────────┘               └─────────┘               └─────────┘
      │                         │                         │
      │  加载到 LDS           │  加载到 LDS           │
      ▼                         ▼                         ▼
   [共享内存]               [共享内存]              [寄存器累加]
      │                         │                         │
      └───────────┬───────────┴───────────┬─────────────┘
                  │                       │
                  └──────── C_tile += A_tile × B_tile ────┘
```

> 面试要点：GEMM 分块是把大矩阵切成小 tile，每次把 A tile 和 B tile 加载到共享内存，在 block 内协作完成部分乘积并累加到寄存器。这样 A 的行和 B 的列可在片上反复复用，大幅减少显存搬运，提高算术强度。

---

## Q16. 什么是 MFMA（矩阵乘累加）？它和普通向量运算有什么不同？

**回答**

MFMA（Matrix Fused Multiply-Add，矩阵乘累加）是 GPU 硬件上的矩阵乘累加指令（如 AMD 的 `wmma` 系列），一次指令完成一小块矩阵的乘加：`D = A × B + C`，其中 A、B、C 都是小矩阵（tile），不是单个标量。

与普通向量运算的区别：
- 普通向量运算（如 `c[i] = a[i] + b[i]`）：一条指令对应一个标量运算，数据在标量寄存器（VGPR）里。
- MFMA：一条指令对应一小块矩阵乘累加，数据在专门的矩阵乘累加寄存器（AccVGPR/AGPR）里，吞吐量远高于标量逐元素运算。

大模型（Transformer、LLM）中的注意力和前馈网络大量使用 GEMM，MFMA 是这些算子在 GPU 上跑得快的硬件基础。

> 面试要点：MFMA 是矩阵乘累加指令，一次指令完成一小块矩阵的 `D = A×B + C`，数据在矩阵专用寄存器里。与普通标量向量运算相比，MFMA 吞吐量远高，是大模型 GEMM 算子在 GPU 上高效的硬件基础。

---

## Q17. 什么是 Fusion（算子融合）？它为什么能提升性能？

**回答**

算子融合（Operator Fusion）是指把多个原本串行的小算子合并成一个 kernel，避免中间结果反复写回显存、再读回。

经典例子：`y = relu(layer_norm(x) + residual)`。不融合时：
1. LayerNorm 写中间结果到显存；
2. 残差相加又读中间结果、再写；
3. ReLU 又读又写。
中间结果在显存里来回搬运，浪费带宽。

融合后：所有计算在一个 kernel 内完成，中间结果全程留在寄存器或共享内存里，只在最后写回最终结果一次。

收益：
- 减少显存读写次数（带宽受限算子获益最大）。
- 减少 kernel 启动开销。
- 提高缓存/寄存器复用。

但融合也有边界：融合后寄存器/共享内存占用可能增加，导致占用率下降；中间结果若需复用多次，重算可能比读写便宜；融合后代码膨胀，编译器优化空间可能变小。

> 面试要点：算子融合把多个串行小算子合成一个 kernel，中间结果留在寄存器/共享内存，只写回最终结果，减少显存读写和启动开销。边界是寄存器/LDS 占用增加可能降低占用率，多次复用的中间结果重算可能更划算。

---

## Q18. FlashAttention 的核心思想是什么？它和融合有什么关系？

**回答**

FlashAttention 是 Transformer 注意力机制的一个经典融合优化。传统注意力会显式构造完整的 `N×N` 注意力矩阵 `S = Q × K^T`，再做 softmax，再乘 `V`，中间 `S` 很大，显存和带宽开销大。

FlashAttention 的核心思想：
1. **分块（Tiling）**：把 Q、K、V 切成小块，每次只加载一块到 SRAM（片上快存）。
2. **在线软最大（Online Softmax）**：不需要等整个 `S` 行都算完才做 softmax，而是维护一个在线的最大值和归一化常数，边算边累加，最终得到正确的 softmax 结果。
3. **重计算（Recompute）**：反向传播时不存储中间注意力矩阵，而是重新算一遍前向所需的中间值，节省显存。

注意力矩阵和中间结果全程留在片上（SRAM/寄存器），只在最后把输出写回显存，大幅减少 HBM 读写，是“算子融合”在注意力上的典型应用。

> 面试要点：FlashAttention 是注意力机制的算子融合典范：分块加载 Q/K/V 到片上，在线软最大边算边累加，反向重计算中间值，避免显式构造大的 `N×N` 注意力矩阵，中间结果全程留在片上，显著减少 HBM 读写。

---

## Q19. 用 HIP/Triton 实现 Fused RMSNorm，核心代码长什么样？

**回答**

Fused RMSNorm 的核心是：
1. 对输入做归约：算出 `sum(x^2)` 和 `mean(x^2)`；
2. 算出 `rms = sqrt(mean(x^2) + eps)`；
3. 逐元素输出 `y = x / rms * weight`。

下面是一个 Triton 单 program 版本的简化实现（适合中小张量，大张量需多阶段归约）：

```python
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    n,              # 归一化维度长度
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    # 第一阶段：每个 program 先算自己 tile 的 sum 和 sum_of_squares
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = x * x
    local_sum = tl.sum(x, axis=0)
    local_sum2 = tl.sum(x2, axis=0)

    # （此处省略跨 program 的部分和归约；实际大张量需第二阶段把各 program 的 partial 汇总）
    # 假设已经得到全局 mean 和 rms：
    mean = ...  
    rms = tl.sqrt(mean + eps)

    # 第二阶段：逐元素缩放，留在寄存器里，不写中间结果
    y = x / rms * tl.load(w_ptr + offs, mask=mask, other=1.0)
    tl.store(out_ptr + offs, y, mask=mask)
```

关键点：
- `x` 和 `x2` 在寄存器里反复使用，不用写回显存再读。
- 归一化系数 `1/rms` 广播到整个 tile，逐元素乘法在寄存器里完成。
- 只在最后把 `y` 写回显存一次。

> 面试要点：Fused RMSNorm 的核心是“归约统计量 + 逐元素缩放”融合在一个 kernel：`sum`/`sum_of_squares` 在寄存器里累加，`rms` 算出来后对整个 tile 广播缩放，中间结果不写显存，只在最后写回输出。

---

## Q20. 本章（经典算子与 Kernel 实战）小结：怎么优化一个 GPU 算子？

**回答**

优化一个 GPU 算子，建议按以下顺序，不要瞎猜：

1. **先算 Roofline 上限**：算出算术强度 `AI = FLOPs / Bytes`，对比硬件峰值算力和带宽，看它是计算受限还是带宽受限，优化空间有多大。
2. **先优化访存**：成组访问（coalesced access）、向量化（float4）、减少显存往返、用 LDS/共享内存复用数据、分块（Tiling）。
3. **再优化计算**：减少指令数、用向量类型、利用 MFMA/张量核心、降低寄存器压力提高占用率。
4. **考虑融合**：把相邻的带宽受限小算子融合，中间结果留在寄存器/共享内存。
5. **用 Profiling 验证**：rocprofv3 看热点、Roofline 图看离上限还有多远、kernel trace 看有没有等待间隙。

ASCII 图示（优化优先级）：

```
优先级从高到低：
┌─────────────┐
│ 1. Roofline  │──▶ 先算上限，知道优化空间有多大
│ 2. 访存优化   │──▶ 合并访问、向量化、LDS 复用、分块（收益最大）
│ 3. 计算优化   │──▶ 向量化、MFMA、减少指令
│ 4. 算子融合   │──▶ 中间结果留在片上，减少显存读写
│ 5. 占用率调优 │──▶ 减少寄存器/LDS，调整 block 大小
└─────────────┘
```

一句话总结：**先算上限，再优化访存，再优化计算，再考虑融合，最后用 Profiling 验证；不要在离上限很远的代码上做微观优化。**

> 面试要点：本章小结＝先算 Roofline 上限→优化访存（合并、向量化、LDS 复用、分块）→优化计算（向量化、MFMA）→算子融合（中间结果留在片上）→调整占用率→Profiling 验证。优化要有数据支撑，不要瞎猜。

---

## Q21. 逐元素算子和归约算子的数据依赖有什么根本区别？

**回答**

逐元素算子的输出位置 `i` 只依赖输入位置 `i`，不同输出之间无数据依赖，可以完全独立并行。

归约算子的输出依赖全部输入，不同线程计算的中间结果需要汇总到同一个输出，存在跨线程数据依赖，必须通过同步（`__syncthreads()`）或原子操作（`atomicAdd`）来协作。

> 面试要点：逐元素算子输出之间无数据依赖，可完全独立并行；归约算子输出依赖所有输入，线程间有数据依赖，需要同步或原子操作协作。

---

## Q22. 归约中什么是“部分和（partial）”？两阶段归约怎么分工？

**回答**

“部分和（partial）”是指一个 block/thread 只处理输入的一部分，算出这部分数据的局部和，而不是直接算最终结果。

两阶段归约的分工：
- **第一阶段**：每个 block（或 program）对自己负责的 tile 做局部归约，写出一个 `partial`（该 tile 的和）。
- **第二阶段**：启动另一个 kernel（或同一个 kernel 的第二阶段），把所有 `partial` 值再归约成最终标量结果。

这样可以把大规模归约拆成许多独立的小归约，避免所有线程竞争同一个全局地址。

> 面试要点：部分和是一个 block/thread 对自己负责数据的一块算出的局部和。两阶段归约＝第一阶段各 block 算局部和（partial）→第二阶段把所有 partial 汇总成最终结果，避免全局热点竞争。

---

## Q23. 归一化算子为什么需要“归约”作为子环节？

**回答**

LayerNorm/RMSNorm 的输出公式里包含 `mean`（均值）和 `variance`（方差），而均值和方差本身是对整个归一化维度做归约（求和/求平方和）得到的：

```
mean = sum(x) / N
var = mean(x^2) - mean^2
```

只有先把整个维度的数据归约出 `mean` 和 `rms`，才能对每个元素做 `y = (x - mean) / rms * weight`。因此，归约是归一化算子的核心子环节，归约的效率直接决定归一化的效率。

> 面试要点：LayerNorm/RMSNorm 需要先算出均值和方差（或根方均方），而均值/方差是对整个维度做归约（求和/求平方和）得到的，所以归约是归一化算子的核心子环节。

---

## Q24. GEMM 分块中，共享内存（LDS）起什么作用？

**回答**

在 GEMM 分块中，共享内存（LDS）用来暂存当前 tile 的 `A` 块和 `B` 块。一个 `A` tile 可被 `C` 输出行内多个元素复用，一个 `B` tile 可被 `C` 输出列内多个元素复用。把它们放在片上共享内存里，后续复用时不需要再从显存读，显著减少显存带宽压力。

> 面试要点：GEMM 分块中，共享内存用来暂存当前 tile 的 A 块和 B 块，使同一 tile 的数据在 block 内被多个输出元素复用，减少从显存反复读取同一数据的次数。

---

## Q25. 算子融合什么时候“不该”做？

**回答**

算子融合并非总是有益，以下情况应谨慎：
1. **寄存器/共享内存占用增加导致占用率下降**：融合后每个线程/block 需要的寄存器更多，能同时驻留的 wavefront 变少，延迟隐藏能力下降，反而变慢。
2. **中间结果需要被多次复用**：如果融合后的中间结果会被后续多个不同算子使用，融合后可能需要重算多次，不如写一次显存多次读。
3. **代码膨胀与编译器优化空间减少**：融合后 kernel 变大，编译器的寄存器分配和指令调度空间可能变小。

工程判断：融合带宽受限的相邻小算子通常稳赚；融合计算密集型大算子或需要多次复用中间结果的场景，需谨慎评估。

> 面试要点：算子融合不该在“融合后寄存器/LDS 占用导致占用率下降”“中间结果需多次复用（重算比读写贵）”“代码膨胀影响编译器优化”的情况下盲目进行。

---

## Q26. 向量化（float4）和访存合并（coalescing）有什么区别？能一起用吗？

**回答**

- **向量化（float4）**：一条指令搬运 4 个连续 float（128 位），减少**指令发射次数**，每个线程拿到 4 个独立输入（ILP 提升）。
- **访存合并（coalescing）**：warp 内相邻线程访问相邻地址，硬件把多次访存合并成一次宽总线事务，减少**无效字节传输和总线事务次数**。

两者正交，可以一起用：最好的情况是一条 128-bit load 恰好铺满 4 个整 sector，既减指令又减无效字节。

> 面试要点：向量化（float4）减指令发射次数，访存合并（coalescing）减无效字节和总线事务，两者正交，可以一起用，最好情况是一条 128-bit load 恰好铺满 4 个整 sector。

---

## Q27. 归约中 atomic baseline 和 LDS 树归约有什么区别？各适合什么场景？

**回答**

- **Atomic baseline**：每个线程直接 `atomicAdd` 到同一个全局地址。实现最简单，但所有线程竞争同一地址，全局热点，适合数据量小或线程数少的场景作为 baseline。
- **LDS 树归约**：每个 block 先在片上共享内存（LDS）内做树归约，只让 thread 0 把块局部和 `atomicAdd` 到全局。全局竞争从 N 次降到 block 数次，适合大规模归约，性能通常好得多。

生产环境通常优先用 LDS 树归约（或两阶段归约），atomic 只用于极小规模或作为正确性 baseline。

> 面试要点：atomic baseline 每个线程竞争同一全局地址，实现简单但有全局热点；LDS 树归约先在 block 内片上归约，只让 thread 0 原子加全局，全局竞争大幅减少，生产环境通常优先用 LDS 树归约或两阶段归约。

---

## Q28. 什么是波形（wavefront）？在归约算子中为什么要关心它？

**回答**

波形（wavefront）是 GPU 里一组锁步执行的线程，共享同一个程序计数器，同一时刻执行同一条指令（AMD 对应 NVIDIA 的 warp）。RDNA 架构支持 wave32（32 线程）和 wave64（64 线程）。

在归约中，波形大小影响：
- 线程如何划分为波形，warp 内 shuffle（`__shfl_down_sync`）能在波形内极快地交换寄存器值，比写 LDS 再读快。
- block 大小最好是波形大小的倍数，避免波形不满（partial wavefront）造成执行槽位浪费。

> 面试要点：波形（wavefront）是 GPU 锁步执行的线程组（AMD 对应 NVIDIA warp）。归约中，波形内可用 warp shuffle 快速交换寄存器值，block 大小最好是波形大小倍数，避免波形不满浪费执行槽位。

---

## Q29. 归约中浮点数不满足结合律，并行结果和串行结果为什么会不一样？怎么判断“正确”？

**回答**

有限精度浮点数中 `(a+b)+c` 不一定等于 `a+(b+c)`，不同的加法顺序（不同 block 数、不同树形、不同 shuffle 路径）会产生不同的舍入误差，因此并行归约结果通常不能用 `bitwise equality` 与串行 FP32 结果比较。

判断“正确”的方法：
- 用 FP64 在 Host 侧对同一输入做“高精度参考值”。
- 并行结果与参考值的绝对误差在容差内（如 1e-3）即视为正确。
- 测试非 2 的幂长度（如 31, 33, 257），确保越界位置被当作加法单位元 0 正确处理，不丢数据。

> 面试要点：浮点加法不满足结合律，并行归约结果与串行 FP32 结果通常不 bitwise 相等。正确性判断：用 FP64 作为参考基准，绝对误差在容差内（如 1e-3）即正确，并测试非 2 的幂长度以验证越界处理。

---

## Q30. 本章小结：怎么优化一个典型 GPU 算子？

**回答**

以本章覆盖的 Element-Wise、Reduction、Normalization、GEMM-Like、Fusion 为例，优化一个 GPU 算子的通用路径：

1. **先算 Roofline 上限**：算出算术强度，看它是计算受限还是带宽受限，优化空间有多大。
2. **优化访存**：成组访问（coalesced access）、向量化（float4）、分块（Tiling）、用 LDS/共享内存复用数据。
3. **优化计算**：减少指令数、用 MFMA/张量核心、降低寄存器压力提高占用率。
4. **考虑融合**：把相邻的带宽受限小算子融合，中间结果留在寄存器/共享内存，只在最后写回。
5. **用 Profiling 验证**：rocprofv3 看热点、Roofline 图看离上限还有多远、kernel trace 看有没有等待间隙。

ASCII 图示（优化闭环）：

```
Roofline 算上限 ──▶ 访存优化 ──▶ 计算优化 ──▶ 算子融合 ──▶ 占用率调优 ──▶ Re-profile 验证
   │                                                       │
   └───────────────────────────────────────────────────────┘
   (如未达标，回到对应步骤再优化)
```

一句话总结：**先算上限，再优化访存，再优化计算，再考虑融合，最后用 Profiling 验证；不要在离上限很远的代码上做微观优化。**

> 面试要点：本章小结＝先算 Roofline 上限→优化访存（合并、向量化、LDS 复用、分块）→优化计算（MFMA、减少指令）→算子融合（中间结果留在片上）→调整占用率→Profiling 验证。闭环反复，让数据说话，不要瞎猜。
