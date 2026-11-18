# GPU 与 ROCm 环境准备 —— 9070XT、Ubuntu、ROCm 与 WSL2 边界

> 面试生普（LLM 500 题）· GPU 与 ROCm 环境准备篇。本章覆盖从零搭建 AMD GPU 计算环境的全流程：为什么需要 GPU、ROCm 与 CUDA 的关系、9070XT 与 gfx1201 架构、BIOS 前置设置（Above 4G Decoding / Resizable BAR）、ROCm 10.0 安装与验证、原生 Linux 与 WSL2 的边界（ROCDXG、/dev/dxg）、uv 环境管理与 ROCm wheel 源、以及最小 smoke test（rocminfo、PyTorch ROCm 矩阵乘、最小 HIP 程序），共 30 个问题。正文面向“会一点编程语言和深度学习基础，除此之外没有任何其他背景”的读者，每个部分都配大白话解释、完整 Python 代码和 ASCII 图。

---

## Q1. 为什么学大模型推理/训练一定要碰 GPU？

**回答**

这是很多初学者的第一个问题：“我会写 Python，为什么不能在 CPU 上跑大模型，非得搞 GPU？”

用大白话说：**CPU 像一个学识渊博但只能做一件事的教授，而 GPU 像一千个只会做加法的小学生。** CPU 擅长处理复杂的逻辑判断（if-else、循环、中断），但一次只能处理很少的数据；GPU 擅长“画饼”——把一大堆相同的简单计算（比如矩阵乘法里的无数个乘加）同时分给成千上万个核心一起算。

大模型的核心计算本质上就是矩阵乘法和向量运算，数据量极大、计算高度重复，这正好是 GPU 的强项。在 CPU 上跑一个 70B 参数的大模型推理，可能要等好几分钟才能生成一句话；在 GPU 上，同样的任务可能只要几秒钟。

ASCII 图示：

```
CPU（擅长逻辑，串行）            GPU（擅长并行，吞吐大）
┌─────────┐                     ┌─────────────────────────┐
│  任务A  │                     │  任务1  任务2  任务3 ...  │
│   │   │                     │  │     │     │           │
│   ▼   ▼                     │  ▼     ▼     ▼           │
│  任务B  │                     │  ▼     ▼     ▼           │
│   │   │                     │  │     │     │           │
│   ▼   ▼                     │  ▼     ▼     ▼           │
│  任务C  │                     │  千万个任务同时跑          │
└─────────┘                     └─────────────────────────┘
    一次做一件事                    一次做成千上万件同样的事
```

> 面试要点：GPU 适合大模型，是因为大模型核心是海量重复的矩阵/向量运算，GPU 的大量并行核心能同时处理这些计算，速度比 CPU 快几个数量级。

---

## Q2. ROCm 是什么？它和 CUDA 有什么关系？

**回答**

ROCm（Radeon Open Compute）是 AMD 推出的一套 GPU 计算平台，可以简单理解为 **AMD 版本的 CUDA**。

CUDA 是 NVIDIA 的专有生态：只有 NVIDIA 显卡能跑，工具链（编译器、库、框架适配）都由 NVIDIA 控制。ROCm 则是 AMD 的开源计算栈，让 AMD 显卡也能做深度学习训练和推理。

两者的对应关系：

| 概念 | NVIDIA 侧（CUDA） | AMD 侧（ROCm） |
|------|------------------|---------------|
| 驱动 | NVIDIA 驱动 | amdgpu 内核驱动 + ROCm 运行时 |
| 编程语言 | CUDA C++ | HIP（看起来和 CUDA 很像） |
| 编译器 | nvcc | hipcc |
| 深度学习框架 | PyTorch CUDA 版 | PyTorch ROCm 版 |
| 查看 GPU 工具 | nvidia-smi | rocm-smi / amd-smi / rocminfo |

**关键点**：HIP 是 ROCm 里用来写 GPU 程序的语言，它的语法和 CUDA 极像，很多时候把 CUDA 代码里的 `cuda` 换成 `hip` 就能跑。但这并不意味着完全兼容——有些 NVIDIA 专用库（如 cuDNN 的某些特定行为）在 ROCm 里有对应的替代品（如 MIOpen），但用法可能不完全一样。

> 面试要点：ROCm 是 AMD 的 GPU 计算平台，对应 NVIDIA 的 CUDA；HIP 对应 CUDA C++，hipcc 对应 nvcc。ROCm 是开源的，主要支持 AMD 自家显卡。

---

## Q3. 为什么选 AMD Radeon RX 9070 XT 做实验基线？

**回答**

选 9070 XT 主要基于三点考虑：

1. **架构新、代表性强**：9070 XT 基于 RDNA4 架构，ISA 名为 gfx1201（通用 ISA 名 gfx12-generic）。这是 AMD 当前面向消费者市场的较新架构，学了它，迁移到其他 RDNA3/RDNA4 显卡（如 7900 XTX、7900 XT）成本较低。
2. **ROCm 支持逐渐完善**：早期 ROCm 主要支持数据中心卡（Instinct MI 系列），消费者卡（Radeon RX 系列）支持较晚。9070 XT 作为较新的消费者卡，在 ROCm 10.0 上已经有较好的适配，适合做入门实验。
3. **显存容量适中**：9070 XT 通常有 16GB 显存，能跑起不小的模型进行推理实验，又不会像数据中心卡那样贵得离谱。

当然，9070 XT 不是唯一选择。如果你手头是 7900 XT 或其他支持的 AMD 显卡，也可以照着本章的验证顺序做，只是包版本、设备名、工具输出会有差异，自己对照一下即可。

> 面试要点：9070 XT 选因为架构新（RDNA4/gfx1201）、ROCm 支持较好、显存 16GB 适中、价格相对可接受。其他支持的 AMD 显卡也可沿用本章验证流程，只需对照版本差异。

---

## Q4. gfx1201 / RDNA4 是什么？ ISA 名 gfx12-generic 是什么意思？

**回答**

这些名字一开始看起来像天书，其实只是 AMD GPU 的“型号代号”和“指令集代号”：

- **RDNA4**：AMD 显卡架构的代名词（就像 Intel 有酷睿 i 系列，AMD 有 RDNA 系列）。RDNA4 是较新一代，9070 XT 就用这一代。
- **gfx1201**：这个 GPU 具体的“目标代号”（target），ROCm 内部用它来区分不同型号的 GPU。
- **gfx12-generic**：这一代（gfx12 系列）的通用指令集架构（ISA）。你可以把它理解为“这一代显卡共同讲的方言”。

验证时，`rocminfo` 输出里会看到类似 `amdgcn-amd-amdhsa--gfx1201` 和 `amdgcn-amd-amdhsa--gfx12-generic` 两条，前者是具体型号 target，后者是 gfx12 系列的通用 ISA。

> 面试要点：gfx1201 是 9070 XT 的具体 target，gfx12-generic 是 RDNA4 系列的通用指令集架构。rocminfo 里能看到这两条，说明驱动认出了你的卡。

---

## Q5. 安装 ROCm 前，主板 BIOS 里需要开什么？（Above 4G Decoding、Resizable BAR）

**回答**

这是很多新手容易踩的坑：ROCm 安装完了，`rocminfo` 却看不到 GPU，或者 PyTorch 报“No GPU detected”。除了驱动问题，另一个常见原因是主板 BIOS 里两个选项没开：

1. **Above 4G Decoding**：允许系统访问超过 4GB 的 PCIe 地址空间。现代显卡显存较大（比如 16GB），如果不开这个，系统可能无法正确映射显存。
2. **Resizable BAR（可调整大小的基址寄存器）**：让 CPU 能一次性访问完整的显存，而不是分小块访问。开启后，CPU 与 GPU 之间的数据传输效率更高。

这两个选项通常在主板 BIOS 的“高级”或“芯片组”设置里，名字可能略有不同（比如“Above 4G Decoding”、“Re-Size BAR Support”）。安装 ROCm 之前，先进 BIOS 把这两个都开成 Enabled，然后保存重启。

ASCII 图示（数据通路）：

```
CPU ─────────────┐
                 ▼
           PCIe 总线
                 ▼
           ┌─────────┐
           │ 显卡    │◀──── Above 4G Decoding: 允许访问 >4GB 地址空间
           │ (显存16GB)│      Resizable BAR: 一次性访问完整显存
           └─────────┘
```

> 面试要点：安装 ROCm 前，BIOS 里开启 Above 4G Decoding 和 Resizable BAR，否则系统可能无法正确映射大显存，导致 ROCm 看不到 GPU。

---

## Q6. ROCm 驱动和普通显卡驱动有什么区别？

**回答**

AMD 显卡在 Linux 上其实涉及两层东西，容易混淆：

1. **内核显卡驱动（amdgpu）**：负责把显卡当成一个普通设备让操作系统认出来，输出图像到显示器、处理基本的图形显示。这个驱动通常已经内嵌在 Linux 内核里了，插上显卡就能用。
2. **ROCm 运行时**：负责让显卡做“通用计算”（矩阵乘法、卷积、深度学习），提供 HIP 编译器、rocBLAS、MIOpen 等库。这个需要单独安装 ROCm 软件包。

换句话说：普通驱动让显卡“能显图”，ROCm 让显卡“能算大模型”。只有装了 ROCm，PyTorch 才能调用显卡做矩阵运算。

> 面试要点：普通 amdgpu 驱动负责图形显示，ROCm 运行时负责通用计算（深度学习）。两者分开安装，ROCm 是让 PyTorch/HIP 能用显卡算东西的关键。

---

## Q7. Ubuntu 24.04 上安装 ROCm 10.0 的步骤是什么？

**回答**

在 Ubuntu 24.04 上安装 ROCm 10.0，大致分以下几步（以官方 amdgpu-install 工具为例）：

**第一步：安装内核头文件和额外模块**
```bash
sudo apt update
sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
```
内核头文件是编译 amdgpu 内核模块时需要的，缺了它驱动装不上。

**第二步：下载并安装 amdgpu-install 安装器**
```bash
wget https://repo.radeon.com/amdgpu-install/10.0/ubuntu/noble/amdgpu-install_10.0_all.deb
sudo apt install ./amdgpu-install_10.0_all.deb
```
（版本号可能随时间变化，以 AMD 官网最新为准。）

**第三步：安装 ROCm（显卡驱动 + 计算运行时）**
```bash
sudo amdgpu-install -y --usecase=graphics,rocm
```
`--usecase=graphics,rocm` 表示既装图形驱动，也装计算运行时。

**第四步：把当前用户加入 render 和 video 组**
```bash
sudo usermod -a -G render,video $LOGNAME
```
这一步非常重要，否则普通用户没有权限访问 GPU 设备文件（如 `/dev/kfd`、`/dev/dri/render*`），跑程序会报“Permission denied”。

**第五步：重启**
```bash
sudo reboot
```
重启后，新的组权限和内核模块才会生效。

**第六步：验证**
```bash
/opt/rocm/bin/rocminfo | grep -E "Marketing Name|gfx"
```
如果能看到 `AMD Radeon RX 9070 XT` 和 `gfx1201`，说明安装成功。

> 面试要点：Ubuntu 安装 ROCm 10.0 的步骤＝装内核头文件→装 amdgpu-install→`amdgpu-install --usecase=graphics,rocm`→用户加 render/video 组→重启→`rocminfo` 验证。缺任何一步都可能导致 GPU 不可见。

---

## Q8. 安装完 ROCm 后，为什么要把用户加入 render 和 video 组？

**回答**

在 Linux 中，GPU 相关的设备文件（比如 `/dev/kfd`、 `/dev/dri/render*`）默认只允许特定组的用户访问：

- **render 组**：控制对 GPU 计算引擎的访问权限，跑 HIP 程序、PyTorch 计算需要这个权限。
- **video 组**：控制对显示输出相关设备的访问。

如果不把用户加入这两个组，普通用户运行程序时会碰到 `Permission denied`，比如无法打开 `/dev/kfd`，ROCm 运行时就无法与显卡通信，最终表现为“ROCm 看不到 GPU”或“HIP 错误：无法初始化设备”。

加入组后，需要**重新登录**（或重启）才会生效，因为组成员身份是在登录时确定的。

验证是否加入成功：
```bash
groups $LOGNAME
```
输出里应该包含 `render` 和 `video`。

> 面试要点：render 组控制 GPU 计算访问，video 组控制显示设备访问。安装完 ROCm 后必须用 `sudo usermod -a -G render,video $LOGNAME` 把用户加进去，并重新登录，否则程序会因权限不足无法访问 GPU。

---

## Q9. rocminfo 是什么？怎么用它验证 GPU 被 ROCm 看到？

**回答**

`rocminfo` 是 ROCm 自带的一个查询工具，专门用来显示 ROCm 能识别到的 GPU 信息，包括显卡型号、计算单元数量、ISA 名（gfx 代号）、内存大小等。它是验证“ROCm 能否看到 GPU”的最直接工具，而且**不依赖 amdgpu 内核状态监控路径**，所以在 WSL2 下也能运行。

常用验证命令：

```bash
# 查看显卡型号和 gfx 代号
/opt/rocm/bin/rocminfo | grep -E "Marketing Name|gfx"

# 更详细地看计算单元
/opt/rocm/bin/rocminfo | grep "Compute Unit"
```

预期输出示例（9070 XT）：
```
Marketing Name: AMD Radeon RX 9070 XT
...
gfx1201
```

ASCII 图示（rocminfo 验证流程）：

```
安装完 ROCm
      │
      ▼
┌─────────────┐
│ 运行 rocminfo │
│  (ROCm工具)  │
└──────┬──────┘
       │
   ┌───▼───┐
   │ 能看到  │──是──▶ 第一道门过了：底层驱动和运行时基本可用
   │ GPU？   │
   └───┬───┘
       │否
       ▼
   回到驱动安装和 BIOS 设置排查
```

> 面试要点：rocminfo 是 ROCm 自带的 GPU 查询工具，用 `rocminfo | grep -E "Marketing Name|gfx"` 验证 GPU 是否被看到。它是“第一道门”验证工具，WSL2 下也能用。

---

## Q10. rocm-smi 和 amd-smi 是干什么的？为什么 WSL2 下它们不可用？

**回答**

`rocm-smi` 和 `amd-smi` 都是用来查看 GPU 运行时状态的工具，类似 NVIDIA 的 `nvidia-smi`：

- **rocm-smi**：显示显存占用、温度、功耗、风扇转速等运行时状态，输出比较紧凑。
- **amd-smi**：rocm-smi 的后继者，提供更完整的硬件拓扑信息（PCIe 版本、带宽、功耗限制、固件版本等），分 `static`（静态信息）和 `metric`（运行时指标）两种模式。

**为什么 WSL2 下不可用？**

WSL2 不是原生 Linux 内核，它通过 Windows 的 GPU 驱动（`/dev/dxg`）把显卡暴露给 Linux 子系统，而不是通过 Linux 原生的 `amdgpu` 内核模块。而 `rocm-smi` 和 `amd-smi` 都依赖 `amdgpu` 内核模块来读取硬件状态，WSL2 里没有这个模块，所以它们会报 `Driver not initialized (amdgpu not found in modules)`。

但是，WSL2 下的计算路径（`rocminfo`、`hipcc`、`hipMalloc`、PyTorch ROCm、Triton）通常是可用的，因为这些走的是 ROCm 运行时，不一定非要 `amdgpu` 状态监控模块。

> 面试要点：rocm-smi/amd-smi 用于查看显存、温度、功耗等运行时状态。WSL2 下它们不可用，因为 WSL2 通过 `/dev/dxg` 暴露 GPU，没有 Linux 原生的 `amdgpu` 内核模块。但这不影响 rocminfo 和计算路径。

---

## Q11. 原生 Linux 和 WSL2 跑 ROCm 的主要区别是什么？

**回答**

| 维度 | 原生 Linux（Ubuntu 24.04） | WSL2（Windows 11） |
|------|--------------------------|-------------------|
| 驱动模式 | Linux 原生 `amdgpu` 内核模块 | Windows Adrenalin 驱动 + ROCDXG（`/dev/dxg`） |
| rocminfo | ✅ 可用 | ✅ 可用 |
| rocm-smi / amd-smi | ✅ 可用 | ❌ 不可用（无 amdgpu 内核模块） |
| 硬件性能计数器（rocprof --pmc） | ✅ 可用 | ❌ 不可用 |
| 计算路径（HIP、PyTorch ROCm） | ✅ 可用 | ✅ 通常可用 |
| 推荐程度 | 首选，坑少，工具链完整 | 备选，能跑通但部分状态工具缺失 |

**大白话比喻：**

原生 Linux 像“直接插电”，所有工具都能用；WSL2 像“通过变压器连电”，能算能跑，但看电表（温度、功耗）的工具用不了。

ASCII 图示：

```
原生 Linux：显卡 ──▶ amdgpu 内核驱动 ──▶ ROCm 运行时 ──▶ 所有工具可用
WSL2：     显卡 ──▶ Windows Adrenalin驱动 ──▶ /dev/dxg ──▶ ROCm 运行时 ──▶ 计算可用，状态工具不可用
```

> 面试要点：原生 Linux 工具链最完整，rocm-smi/amd-smi 和性能计数器都能用；WSL2 下 rocm-smi/amd-smi 不可用，但 rocminfo 和计算路径通常可用。有条件优先用原生 Linux。

---

## Q12. WSL2 下 ROCm 是怎么工作的？（ROCDXG、/dev/dxg）

**回答**

在原生 Linux 上，ROCm 通过 `amdgpu` 内核驱动直接和显卡对话。但在 WSL2 里，Linux 子系统无法直接控制 Windows 的显卡硬件，所以 AMD 提供了一套替代方案：

1. **Windows 侧**：安装 AMD Software: Adrenalin Edition 驱动（显卡驱动）。
2. **WSL2 侧**：通过 `/dev/dxg` 设备文件访问 Windows 的 GPU，这是 Windows 图形子系统暴露给 WSL2 的通道。
3. **ROCDXG（librocdxg）**：WSL2 下的 ROCm 驱动层，把 ROCm 的调用翻译成 `/dev/dxg` 可以理解的指令，而不是使用 Linux 原生的 `amdgpu` 模块。

验证 WSL2 下 ROCm 是否就绪，可以看：
```bash
ls -l /dev/dxg          # 应该存在，说明 WSL2 GPU bridge 可见
rocminfo                # 应该能看到 GPU 和 gfx1201
hipconfig --full        # HIP_PLATFORM: amd，HIP_RUNTIME: rocclr
```

如果 `rocminfo` 能看到 `Device Type: GPU` 和 `ISA: amdgcn-amd-amdhsa--gfx1201`，说明 WSL2 下的 ROCm 计算路径已经通了。

> 面试要点：WSL2 下 ROCm 通过 Windows 驱动 + `/dev/dxg` + ROCDXG 工作，不依赖 Linux 原生 amdgpu 模块。验证重点看 `/dev/dxg` 是否存在、`rocminfo` 是否看到 GPU、`hipconfig --full` 是否成功。

---

## Q13. uv 是什么？为什么环境管理用 uv 而不是 pip/conda？

**回答**

`uv` 是由 Astral 团队开发的 Python 包管理器，用 Rust 写成，速度比 pip 快几十倍。它集成了 Python 版本管理、虚拟环境创建、包安装、锁文件生成等功能，可以替代 `pip + venv + pip-tools + conda` 的组合。

在 ROCm 环境准备里，用 uv 有几个好处：

1. **版本锁定**：`uv.lock` 把所有依赖的精确版本锁死，避免“我这儿能跑，你那儿跑不了”的环境漂移。
2. **ROCm wheel 源切换方便**：PyTorch ROCm 版需要从 AMD 的专用 wheel 源安装，uv 的 `[[tool.uv.index]]` 可以清晰地配置多个索引源。
3. **统一管理 Python 版本**：`uv python install 3.12` 可以直接装 Python，不用单独装 pyenv。

对比：

```
pip + venv + conda          uv
┌─────────┐                 ┌─────────┐
│ pip     │                 │   uv    │
│ venv    │   ──▶  多个工具  │  一体化   │
│ conda   │   组合使用，易冲突 │  一个命令│
│ pip-tools│                │  搞定    │
└─────────┘                 └─────────┘
    慢，配置复杂               快，简单
```

> 面试要点：uv 是快速的 Python 包管理器，集成了版本管理、虚拟环境、包安装、锁文件。用 uv 避免 pip/conda 组合的环境漂移，并方便配置 ROCm 专用 wheel 源。

---

## Q14. pyproject.toml 和 uv.lock 有什么作用？

**回答**

这两个文件是现代 Python 项目管理的“身份证”和“购物清单”：

- **pyproject.toml**：项目的配置文件，里面写明项目需要哪些依赖包、支持哪些 Python 版本、使用哪些包索引源。它是人写的、可读的。
- **uv.lock**：锁定文件，由 `uv lock` 生成，记录每个依赖包的**精确版本号**和下载哈希。它是机器读的，确保每次 `uv sync` 安装出来的环境完全一致。

举个例子：

```toml
# pyproject.toml（人写的，声明意图）
[project]
dependencies = [
    "torch>=2.10.0",
]

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://stable.repo.amd.com/rocm/whl-next/"
explicit = true
```

```toml
# uv.lock（机器生成的，锁定精确版本）
[[package]]
name = "torch"
version = "2.13.0+rocm10.0.0"
```

有了这两个文件，新同学只需要运行 `uv sync`，就能拿到和你完全一样的环境，不会出现“ torch 2.1 vs 2.13 导致行为不一致”的问题。

> 面试要点：pyproject.toml 声明项目依赖和索引源，uv.lock 锁定精确版本号。两者配合 `uv sync`，能让不同机器上复现一模一样的 Python 环境，避免版本漂移。

---

## Q15. 怎么用 uv 创建和同步 Python 环境？

**回答**

在本章的代码目录（比如 `code/part0-intro/`）下，操作如下：

**第一步：安装 uv（如果还没装）**
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

**第二步：安装指定 Python 版本并创建虚拟环境**
```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate   # Linux/macOS
# 或者 .venv\Scripts\activate  # Windows
```

**第三步：同步依赖（根据 pyproject.toml 和 uv.lock 安装所有包）**
```bash
uv sync
```

**第四步：运行脚本**
```bash
uv run python check_torch_rocm.py
```

如果需要安装额外的 ROCm 相关包，可以用：
```bash
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.13.0+rocm10.0.0"
```

ASCII 图示（uv 工作流）：

```
pyproject.toml + uv.lock
            │
            ▼
      ┌─────────────┐
      │   uv sync   │◀──── 读取锁定版本，安装所有依赖
      └─────────────┘
            │
            ▼
      .venv/ 虚拟环境
            │
            ▼
      uv run python xxx.py   用该环境运行脚本
```

> 面试要点：用 uv 创建环境的步骤＝安装 uv→`uv python install 3.12`→`uv venv`→激活→`uv sync` 安装依赖→`uv run python xxx.py` 运行。`uv sync` 会根据 pyproject.toml 和 uv.lock 把环境装得和别人一模一样。

---

## Q16. ROCm 版 PyTorch 怎么安装？ wheel 源怎么选？

**回答**

ROCm 版 PyTorch 不能从默认的 PyPI 源安装（默认源是 CUDA 版），需要从 AMD 的专用 wheel 源安装。

使用 uv 时，在 `pyproject.toml` 里配置索引源：

```toml
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://stable.repo.amd.com/rocm/whl-next/"
explicit = true
```

然后安装带架构后缀的包：

```bash
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch[device-gfx1201]==2.13.0+rocm10.0.0" \
    "torchvision[device-gfx1201]==0.28.0+rocm10.0.0" \
    "torchaudio==2.11.0.2+rocm10.0.0"
```

**关于 `device-gfx1201`**：方括号里的 `device-gfx1201` 是“额外依赖标签”，告诉 PyTorch wheel 针对哪个 GPU 架构优化。不同显卡选不同标签：

| 显卡 | device 标签 |
|------|------------|
| RX 9070 XT / 9070 GRE / AI PRO R9700S | `device-gfx1201` |
| RX 9060 XT / 9060 / RX 9050 | `device-gfx1200` |
| Ryzen AI 9 HX 475 / 375 等 | `device-gfx1150` |
| 所有架构（兼容性最好，体积最大） | `device-all` |

版本号里的 `+rocm10.0.0` 表示这是针对 ROCm 10.0 编译的 PyTorch，必须和你安装的 ROCm 主版本一致，否则会报版本不匹配。

> 面试要点：ROCm 版 PyTorch 须从 AMD wheel 源（如 `https://stable.repo.amd.com/rocm/whl-next/`）安装，并带上 `device-gfx1201` 等架构标签，版本号中的 `+rocm10.0.0` 必须与已安装的 ROCm 主版本一致。

---

## Q17. 为什么 ROCm 版 PyTorch 里到处都是 "cuda"？

**回答**

这是新手最容易困惑的问题之一：明明装的是 ROCm 版 PyTorch，为什么代码里写 `device="cuda"`，`torch.cuda.is_available()` 返回 `True`，而不是 `hip` 或 `rocm`？

这是因为 PyTorch 的设备字符串有历史包袱：早期 PyTorch 只支持 NVIDIA CUDA，所以设备统一叫 `cuda`。后来支持了 ROCm，为了不改动海量已有代码，PyTorch 没有给 ROCm 开一条新的设备名称，而是继续沿用 `cuda` 字符串。在 ROCm 版 PyTorch 里，`cuda` 实际上被映射到了 ROCm 的 HIP 运行时上。

所以在 ROCm 环境下：
- `torch.cuda.is_available()` 返回 `True` 表示“ROCm 后端可用”，不代表你有 NVIDIA 卡。
- `device="cuda"` 在 ROCm 版 PyTorch 里等价于“用 GPU 计算”。
- `torch.version.hip` 才是 ROCm/HIP 的版本号（比如 `7.15.26333`）。

> 面试要点：ROCm 版 PyTorch 里 `cuda` 是历史遗留名称，实际指代 ROCm/HIP 后端。`torch.cuda.is_available()` 为 True 表示 ROCm 可用；ROCm 版本号要看 `torch.version.hip`，不要看 `torch.version.cuda`。

---

## Q18. 最小 smoke test 到底测什么？（三道门）

**回答**

把环境准备想象成“开工前体检表”，只回答一个问题：**当前环境，够不够支撑你继续往后学？**

本章把验证分成三道门，必须依次推开，一道都跳不过去：

| 门 | 验证什么 | 推开之后说明 |
|------|----------|-------------|
| 第一道门 | ROCm 能看到 GPU | 底层驱动和运行时基本可用 |
| 第二道门 | PyTorch ROCm | 框架层能把计算放到 AMD GPU 上 |
| 第三道门 | 最小 HIP 程序 | 后续手写 kernel 的路径基本打通 |

**为什么要按顺序？** 因为上层框架全都建在底层运行时之上。如果 `rocminfo` 连 GPU 都看不到，那就别急着去跑 PyTorch 或 HIP——底层不通，上层报的错只会更让你迷惑。先回到驱动安装和 BIOS 设置排查，确认 `rocminfo` 能看到 GPU 之后再继续。

ASCII 图示：

```
三道门验证流程
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ 第一道门    │ ──▶  │ 第二道门    │ ──▶  │ 第三道门    │
│ rocminfo    │      │ PyTorch ROCm│      │ 最小 HIP 程序 │
│ 看 GPU？    │      │ 矩阵乘跑通？│      │ hipcc 编译？  │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │ 是                  │ 是                │ 是
       ▼                     ▼                   ▼
   底层可用             框架层可用             手写 kernel 路径打通
```

> 面试要点：smoke test 三道门＝rocminfo 看 GPU→PyTorch ROCm 跑矩阵乘→最小 HIP 程序编译运行。按顺序验证，底层不通别急着上层。

---

## Q19. 验证 PyTorch ROCm 的 Python 代码长什么样？（1024x1024 矩阵乘）

**回答**

第二道门的验证脚本非常简单，就是导入 PyTorch、检查 GPU 后端、在 GPU 上跑一次矩阵乘。越简单越好，复杂的例子只会让排查时变量更多。

```python
# check_torch_rocm.py
import platform
import torch

print(f"python: {platform.python_version()}")
print(f"torch:  {torch.__version__}")
print(f"hip:    {torch.version.hip}")          # ROCm/HIP 版本号
print(f"cuda_available: {torch.cuda.is_available()}")  # 在 ROCm 版里表示 ROCm 后端可用
print(f"device_count: {torch.cuda.device_count()}")
print(f"device_name: {torch.cuda.get_device_name(0)}")

# 在 GPU 上跑一次 1024 x 1024 矩阵乘，最小 smoke test
x = torch.randn(1024, 1024, device="cuda")   # 在 ROCm 版 PyTorch 里 "cuda" 即指 GPU
y = x @ x

print(f"result_shape: {tuple(y.shape)}")
print(f"result_dtype: {y.dtype}")
print(f"result_device: {y.device}")
print(f"result_checksum: {y.sum().item():.4f}")
```

运行：
```bash
uv run python check_torch_rocm.py
```

预期输出示例：
```
python: 3.12.3
torch:  2.13.0+rocm10.0.0
hip:    7.15.26333
cuda_available: True
device_count: 1
device_name: AMD Radeon RX 9070 XT
result_shape: (1024, 1024)
result_dtype: torch.float32
result_device: cuda:0
result_checksum: 57777.9727
```

看到 `device_name: AMD Radeon RX 9070 XT` 和 `cuda_available: True`，第二道门就过了。

> 面试要点：PyTorch ROCm smoke test 脚本＝导入 torch→打印版本（torch/hip）→检查 `cuda.is_available()` 和设备名→在 GPU 上跑 1024x1024 矩阵乘。输出里 `device_name` 显示你的显卡型号，`cuda_available: True` 表示第二道门通过。

---

## Q20. 最小 HIP 程序长什么样？（向量加法，hipcc 编译）

**回答**

第三道门验证“后续手写 kernel 的路径”。最小 HIP 程序可以是一个向量加法：两个长度为 N 的数组逐元素相加。HIP 语法和 CUDA 极像，把 `cuda` 换成 `hip` 即可。

```cpp
// vec_add.cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1 << 20;  // 100 万个元素
    const size_t size = n * sizeof(float);

    // 申请主机内存
    float *ha = (float*)malloc(size);
    float *hb = (float*)malloc(size);
    float *hc = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        ha[i] = 1.0f;
        hb[i] = 2.0f;
    }

    // 申请设备内存
    float *da, *db, *dc;
    hipMalloc(&da, size);
    hipMalloc(&db, size);
    hipMalloc(&dc, size);

    // 拷贝主机→设备
    hipMemcpy(da, ha, size, hipMemcpyHostToDevice);
    hipMemcpy(db, hb, size, hipMemcpyHostToDevice);

    // 启动 kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(da, db, dc, n);

    // 拷贝设备→主机
    hipMemcpy(hc, dc, size, hipMemcpyDeviceToHost);

    // 验证前 5 个结果
    printf("前 5 个结果：");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", hc[i]);
    }
    printf("\n");

    // 释放内存
    hipFree(da); hipFree(db); hipFree(dc);
    free(ha); free(hb); free(hc);

    printf("HIP 向量加法 smoke test 通过！\n");
    return 0;
}
```

编译运行：
```bash
hipcc vec_add.cpp -o vec_add
./vec_add
```

预期输出：
```
前 5 个结果：3.0 3.0 3.0 3.0 3.0
HIP 向量加法 smoke test 通过！
```

`hipcc` 是 HIP 的编译器前端（对应 CUDA 的 nvcc），它会自动调用底层的 clang 和 HIP 运行时把 `.cpp` 编译成可在 AMD GPU 上运行的程序。能编译并运行通过，说明第三道门过了。

> 面试要点：最小 HIP 程序＝向量加法 kernel（`__global__` + `<<<blocks, threads>>>` 启动语法）→`hipMalloc`/`hipMemcpy` 管理设备内存→`hipcc` 编译运行。输出正确且程序不崩溃，表示第三道门通过，后续手写 kernel 的路径基本打通。

---

## Q21. HIP 和 CUDA 编程有什么区别？

**回答**

HIP 是 AMD 为 ROCm 提供的 C++ 异构编程语言，故意和 CUDA 长得很像，以便 NVIDIA 代码能快速迁移。但两者不是 100% 兼容的。

| 对比项 | CUDA (NVIDIA) | HIP (AMD ROCm) |
|--------|--------------|---------------|
| 编程语言 | CUDA C++ | HIP C++（语法极像 CUDA） |
| 编译器 | nvcc | hipcc |
| 启动语法 | `kernel<<<grid, block>>>` | `kernel<<<grid, block>>>`（几乎一样） |
| 内存拷贝 | `cudaMemcpy` | `hipMemcpy`（名几乎一样） |
| 设备查询 | `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| 官方库 | cuDNN, cuBLAS, cuFFT | MIOpen, rocBLAS, rocFFT（对应替代） |
| 支持硬件 | 仅 NVIDIA 显卡 | 主要是 AMD 显卡（也可通过 HIP-CUDA 回传运行在 NVIDIA 上） |

**迁移心法**：大部分情况下，把 CUDA 代码里的 `cuda` 前缀替换成 `hip`，把 `nvcc` 换成 `hipcc`，就能在 ROCm 上跑通。但如果用了 cuDNN 中某些极新或极特殊的功能，可能需要在 ROCm 里找对应的 MIOpen 接口，代码会有细微差别。

ASCII 图示（HIP-CUDA 对应关系）：

```
CUDA 代码                 HIP 代码（替换后）
┌─────────────┐           ┌─────────────┐
│ cudaMalloc  │  ──▶      │ hipMalloc   │
│ cudaMemcpy  │  ──▶      │ hipMemcpy   │
│ cudaFree    │  ──▶      │ hipFree     │
│ nvcc        │  ──▶      │ hipcc       │
│ cuDNN       │  ──▶      │ MIOpen      │
└─────────────┘           └─────────────┘
    只跑 NVIDIA               主要跑 AMD 显卡
```

> 面试要点：HIP 语法和 CUDA 极像，大部分情况下把 `cuda` 换成 `hip`、`nvcc` 换成 `hipcc` 即可迁移到 ROCm。官方库有对应替代（cuDNN→MIOpen 等），但极特殊功能可能需要细微调整。

---

## Q22. 环境搭建中最常见的坑有哪些？

**回答**

环境搭建是“九曲十八弯”，下面是最常见的几个坑和应对方法：

| 常见坑 | 现象 | 应对方法 |
|--------|------|---------|
| BIOS 没开 Above 4G Decoding / Resizable BAR | `rocminfo` 看不到 GPU，或 PyTorch 报设备错误 | 进 BIOS 开启这两个选项，重启 |
| 用户没加 render/video 组 | 跑程序报 `Permission denied`，无法打开 `/dev/kfd` | `sudo usermod -a -G render,video $LOGNAME`，重新登录 |
| ROCm 版本和 PyTorch 版本不匹配 | 导入 torch 报错，或 `device_map="auto"` 失效 | PyTorch 版本号里的 `+rocm10.0.0` 必须和 ROCm 主版本一致 |
| 缺内核头文件 | `amdgpu-install` 失败，驱动编译不过 | 先装 `linux-headers-$(uname -r)` 和 `linux-modules-extra-$(uname -r)` |
| 需要 HSA_OVERRIDE_GFX_VERSION | 某些新卡没被 ROCm 直接识别，报架构不支持 | 导出环境变量 `export HSA_OVERRIDE_GFX_VERSION=12.0.0`（具体值看显卡架构） |
| WSL2 下用 rocm-smi 看温度/功耗 | 工具报 `Driver not initialized` | 这是 WSL2 的已知限制，改用原生 Linux 或只看 rocminfo 和计算结果 |

> 面试要点：常见坑包括 BIOS 前置设置、用户组权限、ROCm/PyTorch 版本不匹配、缺内核头文件、HSA_OVERRIDE_GFX_VERSION 架构覆盖、WSL2 状态工具不可用。遇到问题先对照表排查，不要盲目重装。

---

## Q23. 怎么排查 "No GPU detected" 问题？

**回答**

当 PyTorch 报 `cuda_available: False` 或程序说“No GPU detected”时，不要直接猜，要按顺序排查：

**第一步：看底层驱动是否加载**
```bash
lsmod | grep amdgpu
```
如果没有输出，说明 `amdgpu` 内核模块没加载，可能是驱动没装好或 BIOS 没开 Above 4G Decoding。

**第二步：看 ROCm 能否看到 GPU**
```bash
/opt/rocm/bin/rocminfo | grep -E "Marketing Name|gfx"
```
如果这里空空如也，说明底层不通，PyTorch 肯定也用不了。先解决驱动问题。

**第三步：看用户权限**
```bash
groups $LOGNAME
ls -l /dev/kfd /dev/dri/render*
```
如果用户不在 render/video 组，或设备文件权限不对，加入组并重新登录。

**第四步：看 PyTorch 版本**
```bash
python -c "import torch; print(torch.version.hip)"
```
如果版本号里没有 `+rocm`，说明装的是 CUDA 版 PyTorch，需要重新从 AMD 源装 ROCm 版。

ASCII 排错流程：

```
"No GPU detected"
       │
       ▼
┌─────────────┐
│ lsmod       │──无 amdgpu ──▶ 装驱动、开 BIOS Above 4G/Resizable BAR
│ | grep amdgpu│
└──────┬──────┘
       │有
       ▼
┌─────────────┐
│ rocminfo    │──看不到 GPU ──▶ 检查驱动和硬件兼容性
│ 看 GPU？    │
└──────┬──────┘
       │能看到
       ▼
┌─────────────┐
│ torch.version.hip │──没 rocm ──▶ 重装 ROCm 版 PyTorch
└─────────────┘
```

> 面试要点：排查"No GPU detected"按顺序：`lsmod | grep amdgpu`（驱动是否加载）→`rocminfo`（ROCm 是否看到 GPU）→`groups`（用户权限）→`torch.version.hip`（PyTorch 是否为 ROCm 版）。底层不通别急着修上层代码。

---

## Q24. 显存（VRAM）怎么看？ rocm-smi 显示什么？

**回答**

显存是 GPU 上专门用来存放模型权重、中间激活值、KV Cache 等数据的内存。跑大模型前，先知道自己的显卡有多少显存，再决定能跑多大的模型。

在原生 Linux 上，可以用 `rocm-smi` 或 `amd-smi` 看：

```bash
# 简洁版：显存、温度、功耗
rocm-smi

# 详细静态信息：显存大小、CU 数、PCIe 版本等
amd-smi static

# 运行时指标：显存占用、GPU 活跃度
amd-smi metric
```

`rocm-smi` 输出示例（部分）：
```
======================= ROCm System Management Interface =======================
GPU[0]: AMD Radeon RX 9070 XT
      VRAM Total Memory : 16384 MB
      VRAM Used Memory  : 2048 MB
      GPU Temperature   : 55 °C
      GPU Power         : 120 W
      ...
```

在 WSL2 下 `rocm-smi` 不可用，可以用 Python + PyTorch 查看显存：
```python
import torch
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"显存已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"显存缓存:   {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

ASCII 图示（显存组成）：

```
显存总量 (16GB)
┌─────────────────────────────────┐
│ 模型权重 (权重参数)              │████████      │
│ KV Cache (推理时的键值缓存)     │████          │
│ 中间激活值 (前向传播中间结果)   │██            │
│ 系统预留 / 未分配               │              │
└─────────────────────────────────┘
```

> 面试要点：显存看用 `rocm-smi`（原生 Linux）或 PyTorch 的 `cuda.get_device_properties`/`memory_allocated`。9070 XT 通常有 16GB 显存，模型权重、KV Cache、中间激活值都要占显存，跑大模型前要先算好显存预算。

---

## Q25. 环境指纹（environment fingerprint）是什么？为什么要记录？

**回答**

环境指纹就是把你当前运行环境的关键信息“拍张照片”记录下来，包括：硬件型号、GPU 架构（gfx 代号）、ROCm 版本、HIP 版本、PyTorch 版本、Python 版本、操作系统、uv 版本等。

记录它的目的：

1. **复现问题**：同事说“我的环境跑不通”，你可以对比环境指纹，看是不是版本不一致导致的。
2. **回归对照**：你优化了算子，性能提升了，要确认是不是因为环境变了（比如不小心升级了 ROCm），而不是真的优化有效。
3. **换机器迁移**：换一台机器重新搭环境，对照指纹可以快速找出差异。

一个简单的环境指纹脚本：

```python
# env_fingerprint.py
import platform
import subprocess
import torch

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "未知"

fingerprint = {
    "os": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无",
    "gfx": run("/opt/rocm/bin/rocminfo | grep gfx12"),
    "uv": run("uv --version"),
}

print("=== 环境指纹 ===")
for k, v in fingerprint.items():
    print(f"{k}: {v}")
```

运行后把输出贴到 issue 或笔记里，排查问题时就有据可依。

> 面试要点：环境指纹＝记录硬件、GPU 架构、ROCm/HIP 版本、PyTorch 版本、Python 版本等关键环境信息。用于复现问题、性能回归对照、换机器迁移时快速定位差异。

---

## Q26. 换一张显卡（比如从 9070XT 换到 7900 XTX）需要调整什么？

**回答**

换显卡时，不需要重新学所有东西，但需要对照调整几个地方：

1. **device-gfx 标签**：安装 PyTorch wheel 时，`device-gfx1201`（9070 XT）可能需要换成 `device-gfx1100`（7900 XTX，RDNA3）或对应的 gfx 代号。
2. **HSA_OVERRIDE_GFX_VERSION**：如果新卡没被 ROCm 直接识别，可能需要设置这个环境变量，具体值看新卡架构。
3. **显存预算**：7900 XTX 有 24GB 显存，比 9070 XT 的 16GB 更大，能跑更大的模型，batch size 可以相应调大。
4. **性能基线**：不同卡的算力和带宽不同，以前的性能测试数字不能直接套用，需要重新跑 benchmark。

不需要换的是：验证流程（三道门）、uv 环境管理方式、代码结构、排错思路。

> 面试要点：换显卡主要调整 PyTorch 的 `device-gfx` 标签、可能的 `HSA_OVERRIDE_GFX_VERSION`、显存预算和性能预期。验证流程和排错思路不变，可以沿用。

---

## Q27. ROCm 版本和 HIP 版本号为什么不一样？（ROCm 10.0 vs HIP 7.15.26333）

**回答**

这是另一个常见困惑：安装了 ROCm 10.0，为什么 `torch.version.hip` 显示 `7.15.26333`，或者 `hipcc --version` 报 7.x，而不是 10.x？

原因是：**ROCm 是一个软件栈（包含驱动、运行时、编译器、库等多个组件），而 HIP 是其中的一个组件。** ROCm 整个发行版用一个大版本号（比如 10.0.0），但里面的 HIP 编译器、HIP 运行时、rocBLAS、MIOpen 等子组件各自有独立的版本号，不一定和 ROCm 大版本同步。

类比：ROCm 像一个“操作系统发行版”（如 Ubuntu 24.04），HIP 就像其中的一个软件包（如内核版本 6.8），它们的版本号自然不同。

查看各组件版本：
```bash
hipcc --version              # HIP 编译器版本
/opt/rocm/bin/rocminfo --version  # ROCm 运行时信息
cat /opt/rocm/.info/version   # ROCm 整个发行版版本
python -c "import torch; print(torch.version.hip)"  # PyTorch 编译时对应的 HIP 版本
```

> 面试要点：ROCm 是软件栈大版本号，HIP 是其中子组件，各自版本号不同步。ROCm 10.0 里 HIP 可能是 7.15.x，这很正常。查看版本时，ROCm 整体看 `/opt/rocm/.info/version`，HIP 编译器看 `hipcc --version`，PyTorch 对应的 HIP 看 `torch.version.hip`。

---

## Q28. smoke test 过不去，优先检查哪三道门？

**回答**

smoke test 过不去时，不要随便乱改，按三道门优先级排查：

1. **第一道门（优先级最高）：`rocminfo` 能否看到 GPU？**
   - 不能 → 问题在底层驱动/BIOS/硬件兼容性。检查 BIOS Above 4G Decoding/Resizable BAR、amdgpu 内核模块、用户组权限。
   - 能 → 进入第二道门。

2. **第二道门：PyTorch ROCm 能否跑通矩阵乘？**
   - 不能 → 问题在 PyTorch 版本或 ROCm 运行时绑定。检查 `torch.version.hip` 是否和 ROCm 主版本一致、是否从 AMD 源装了 ROCm 版 PyTorch。
   - 能 → 进入第三道门。

3. **第三道门：最小 HIP 程序能否编译运行？**
   - 不能 → 问题在 HIP 工具链。检查 `hipcc` 是否可用、HIP 路径是否在 PATH 里、内核头文件是否装齐。

如果三道门都过了，环境就算准备好了，后续章节的算子优化、Profiling 都可以正常进行。

ASCII 图示（优先级）：

```
优先级：第一道门 > 第二道门 > 第三道门
       rocminfo    PyTorch ROCm    最小 HIP 程序
          │             │               │
          ▼             ▼               ▼
    底层不通         底层通，框架层不通   全部通，可开始算子优化
    （先修驱动）     （先修 PyTorch/ROCm 版本）
```

> 面试要点：smoke test 过不去按三道门优先级排查：rocminfo 看 GPU（底层）→PyTorch ROCm 跑矩阵乘（框架层）→最小 HIP 程序编译运行（工具链）。先解决第一道门，再解决第二、第三道。

---

## Q29. 本章小结：环境准备的检查清单是什么？

**回答**

如果你只记住一件事，那就是：**环境准备不是“随便装个驱动就完事”，而是一套有顺序的体检流程。** 下面是本章的检查清单，按顺序对勾：

### 阶段一：硬件与 BIOS
- [ ] 主板 BIOS 开启 Above 4G Decoding
- [ ] 主板 BIOS 开启 Resizable BAR
- [ ] 确认显卡是 ROCm 支持列表里的型号（如 9070 XT）

### 阶段二：驱动与 ROCm 安装
- [ ] 安装内核头文件 `linux-headers-$(uname -r)`、`linux-modules-extra-$(uname -r)`
- [ ] 安装 `amdgpu-install` 并执行 `--usecase=graphics,rocm`
- [ ] 用户加入 `render` 和 `video` 组，并重新登录/重启

### 阶段三：基础验证（三道门）
- [ ] `rocminfo` 能看到 `Marketing Name` 和 `gfx1201`（第一道门）
- [ ] PyTorch ROCm 能导入，`cuda.is_available()` 为 `True`，设备名正确（第二道门）
- [ ] `hipcc` 能编译并运行最小 HIP 向量加法程序（第三道门）

### 阶段四：环境管理与记录
- [ ] 用 `uv` 创建虚拟环境，`uv sync` 安装依赖
- [ ] PyTorch 从 AMD wheel 源安装，带正确的 `device-gfx` 标签
- [ ] 记录环境指纹（硬件、ROCm 版本、PyTorch 版本、Python 版本等）

ASCII 图示（完整链路）：

```
BIOS设置 ──▶ 内核头文件 ──▶ amdgpu-install ──▶ 用户组 ──▶ 重启
   │             │             │             │          │
   ▼             ▼             ▼             ▼          ▼
 Above 4G/     为驱动编译     装驱动+运行时   权限访问GPU   生效
 Resizable BAR
                                              │
                                              ▼
                                      ┌─────────────┐
                                      │ 三道门验证   │
                                      │ rocminfo    │
                                      │ PyTorch ROCm│
                                      │ 最小 HIP    │
                                      └─────────────┘
                                              │
                                              ▼
                                      环境准备完成，进入下一章
```

> 面试要点：环境准备检查清单＝BIOS 前置设置→内核头文件→amdgpu-install→用户组→重启→三道门验证（rocminfo/PyTorch ROCm/最小 HIP）→uv 环境与 PyTorch ROCm wheel 源→记录环境指纹。全部打勾后，才能进入后续章节的算子优化和性能分析。

---

## Q30. 下一章（GPU 体系结构）需要带什么基础过去？

**回答**

本章完成后，你已经打通了“能看到 GPU、能在 GPU 上跑计算、能编译 HIP 程序”的基础链路。下一章《GPU 体系结构（上）：编程模型与 wavefront 执行》会把视角从“能跑”转到“为什么快”，需要你带着以下基础过去：

1. **知道 GPU 有很多核心，能并行跑成千上万个线程**：这是 HIP 向量加法里 `<<<blocks, threads>>>` 的直观体现。
2. **知道 GPU 有自己的内存（显存），和 CPU 内存是分开的**：这是 HIP 代码里 `hipMalloc`、`hipMemcpy` 拷贝数据的来由。
3. **知道 hipcc 能把 HIP 代码编译成在 GPU 上跑的程序**：下一章会深入到编译后的指令是怎么在硬件上执行的。
4. **知道 rocminfo 里的 gfx1201 是什么**：下一章会讲 gfx12 架构下，线程是怎么组织成 wavefront、怎么执行的。

不需要你已经懂 wavefront、LDS、寄存器分配这些细节，那些是下一章要讲的。只要你能跑通本章的 smoke test，就有足够的基础去理解“GPU 到底是怎么执行程序的”了。

> 面试要点：下一章需要带的基础＝GPU 并行线程概念、显存与主存分离、hipcc 编译 HIP 程序、知道 gfx1201 是 GPU 架构代号。不需要预先懂 wavefront 等底层细节，那些是下一章内容。
