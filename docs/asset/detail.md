

## 第一段 · 地基（新第 1—3 章）

### 新第 1 章 · 分词器（BPE / Unicode）【30 题】

**Q1.** BPE 的训练与使用是两个独立阶段。训练输出的 merge rules 在编码时以什么顺序应用？顺序为什么不能乱？

**Q2.** 字节级 BPE 与字符级 BPE 在处理 OOV 上的根本区别是什么？字节级 BPE 的初始词表大小由什么决定？

**Q3.** NFC、NFD、NFKC、NFKD 分别做什么？NFKC 在什么情况下会造成信息损失？举一个具体例子。

**Q4.** GPT-2 预分词正则把文本切成哪些类别？这种切分对后续 BPE 合并有什么影响？

**Q5.** 中文场景下 BPE token 效率通常低于英文，从 UTF-8 字节长度、常用字覆盖率、merge 优先级三个角度解释。

**Q6.** 词表调大如何影响 embedding 参数量、softmax 计算量、序列长度与低频 token 的训练充分性？

**Q7.** 给定“low lower newest widest”，手动执行 BPE 两轮合并，写出每轮最高频 pair 与更新后的词表。

**Q8.** BPE、WordPiece、Unigram 在选择 merge 时的准则分别是什么？各自偏向高频共现还是似然改善？

**Q9.** SentencePiece 与 HuggingFace Tokenizers 在架构上有何不同？为什么 SentencePiece 更适合多语言？

**Q10.** 特殊 token（`<pad>`、`<eos>`、`<bos>`、`<unk>`）的 embedding 应冻结还是学习？设计考量是什么？

**Q11.** 分词器训练语料与模型训练语料分布不一致会带来什么后果？如何检测这种不一致？

**Q12.** BPE 的 merge 是否可逆？解码时如何保证 `decode(encode(text)) == text`？哪些情况下会失败？

**Q13.** byte fallback 与 byte-level BPE 有什么区别？什么场景下需要显式启用 byte fallback？

**Q14.** 分词器的“压缩率”如何定义？压缩率越高越好吗？它与下游表现之间是什么关系？

**Q15.** 手写 BPE 训练代码时，如何高效统计相邻 pair 频率？朴素实现的时间复杂度是多少，如何优化？

**Q16.** 为什么现代 LLM 几乎不在分词器层面做 stemming 或 lemmatization？

**Q17.** 词表中存在大量低频 token 时训练会面临什么问题？剪枝词表的代价是什么？

**Q18.** 多语言分词器中，如何避免高资源语言“挤占”低资源语言的 merge 预算？

**Q19.** 分词器对数字的处理方式（逐位切分 vs 整体 token）如何影响算术能力？

**Q20.** 为一个代码模型设计分词器时，对缩进、标识符、运算符分别应做哪些特殊处理？

**Q21.** BPE vocab 从 32k 扩大到 128k，在固定模型参数量下，embedding 层与输出层的显存/计算量如何变化？

**Q22.** 预分词正则设计不当可能把有意义的子词切碎。举一个中文或代码场景的反例。

**Q23.** 如何评估一个分词器在特定领域（医学、法律）上的适配度？需要看哪些指标？

**Q24.** 训练分词器时是否应该对语料去重？不去重会如何扭曲 merge 优先级？

**Q25.** 分词器的 determinism 在分布式训练中为什么重要？不同 rank 使用不同版本会引发什么隐蔽 bug？

**Q26.** BPE 中“合并顺序即优先级”的含义是什么？两个 pair 频率相同时如何打破平局？不同策略差异多大？

**Q27.** 分词器能否在模型训练后单独替换？替换后需要做哪些重新适配（embedding 重初始化、继续预训练）？

**Q28.** 中文场景下 WordPiece 与 BPE 的实际 token 效率差异有多大？设计一个可控实验来测量。

**Q29.** `max_length` 截断策略（截头、截尾、中间截断）分别适合什么任务？截断位置对位置编码有何影响？

**Q30.** 从零实现一个可用的 BPE 分词器，你会把哪些部分做成可配置的？给出类/函数签名级别的接口设计。


### 新第 2 章 · PyTorch 与资源核算【30 题】

**Q1.** 训练 FLOPs 近似 \(6ND\) 中，前向与反向各贡献多少？该公式忽略了哪些开销，导致它总是低估真实计算量？

**Q2.** 给定 70B 参数、15T tokens、1024 张 H100（BF16 峰值约 990 TFLOPS），MFU 取 40%，估算训练天数。若实际用了 30 天，反推 MFU 是多少？

**Q3.** 混合精度训练中，FP32 master weights、BF16 模型副本、FP32 Adam 状态三者的显存占比如何计算？给出每参数字节数的完整拆解。

**Q4.** 梯度累积为何能在不增加显存峰值的情况下增大等效 batch size？它是否完全等价于大 batch 训练？BatchNorm 场景下有什么陷阱？

**Q5.** 激活值显存与 batch size、序列长度、层数、hidden dim 的关系是什么？activation checkpointing 如何用计算换显存？

**Q6.** AdamW 的 weight decay 与 L2 正则是否等价？在 Adam 下为什么通常选择 decoupled weight decay？

**Q7.** 按全局范数裁剪与逐参数裁剪有什么区别？裁剪阈值过大和过小分别会掩盖什么问题？

**Q8.** loss 正常但梯度范数持续在 1e-7 量级，可能是什么原因？应检查哪些环节？

**Q9.** `memory_allocated()` 与 `memory_reserved()` 的区别是什么？OOM 时应看哪一个来定位真实峰值？

**Q10.** PyTorch `autocast` 在哪些算子上自动降精度、哪些保持 FP32？为什么 layernorm 和 softmax 通常不被 autocast？

**Q11.** 数据加载成为瓶颈时，如何判断是 CPU 预处理、磁盘 IO 还是 GPU 等待？`num_workers` 与 `prefetch_factor` 如何调？

**Q12.** 手写梯度累积训练循环，需要修改哪些地方？`optimizer.step()` 与 `zero_grad()` 的调用时机与普通训练有何不同？

**Q13.** `loss.backward()` 后不调用 `optimizer.zero_grad()` 会发生什么？梯度累积场景下为什么反而要刻意不 zero？

**Q14.** BF16 与 FP16 的动态范围差异如何影响训练稳定性？为什么 BF16 通常不需要 loss scaling？

**Q15.** 模型参数量、可训练参数量、激活参数量在 Dense 与 MoE 架构下如何区分？面试中为什么必须分开说？

**Q16.** 给定 hidden=4096、layers=32、heads=32、vocab=32000、seq=2048，估算 decoder-only 模型的参数量与单次前向 FLOPs（忽略 attention 二次项）。

**Q17.** 推理与训练的显存瓶颈有何不同？为什么训练看激活，推理看 KV Cache？

**Q18.** `torch.compile` 在训练中的收益来自哪里？它对动态 shape（变长序列）的支持有什么限制？

**Q19.** DDP 中梯度 all-reduce 与反向计算如何 overlap？bucket size 设大或设小分别有什么后果？

**Q20.** 显存碎片化如何产生？PyTorch caching allocator 如何缓解？什么情况下仍然 OOM？

**Q21.** 将 FP32 训练的模型转为 BF16 继续训练，哪些状态需要转换、哪些需要保留 FP32？

**Q22.** activation checkpointing 的 recomputation 开销如何估算？它适合放在哪类层上，不适合放在哪类层上？

**Q23.** 训练吞吐（tokens/s）与 MFU 的关系是什么？给定吞吐与模型 FLOPs/token，如何反推 MFU？

**Q24.** 为什么 embedding 层的梯度通常是 sparse 的？PyTorch 中如何利用 sparse gradient 节省显存与通信？

**Q25.** 多卡训练中每卡 batch size 不同时，梯度同步会出现什么问题？梯度累积能否弥补不均衡？

**Q26.** 估算 Llama-3 8B（BF16）推理时加载模型本身需要多少显存？再加 8k 上下文，KV Cache 大约多少？

**Q27.** `optimizer.state_dict()` 通常包含哪些内容？为什么 Adam 的优化器状态是模型参数量的 2 倍（FP32 下）？

**Q28.** 从优化器二阶矩估计的角度解释：训练初期为什么需要学习率 warmup？

**Q29.** 只有 8 张 80GB GPU、要训练 13B 模型时，显存预算如何分配？哪些并行/节省显存手段必须上？

**Q30.** 训练日志中 `grad_norm` 突然变成 NaN，排查顺序是什么？从数据、loss、梯度、参数逐层定位。


### 新第 3 章 · 语言模型架构与训练细节【35 题 · 重点】

**Q1.** 手写多头自注意力：给定 Q、K、V 的 shape `(B, H, T, d)`，写出缩放点积注意力的完整计算过程，包括 mask 的位置与形状。

**Q2.** 缩放因子 \(1/\sqrt{d_k}\) 不加时，softmax 在 \(d_k\) 较大时会出现什么数值问题？对梯度有什么影响？

**Q3.** Pre-LN 与 Post-LN 的梯度流差异是什么？为什么 Pre-LN 可以去掉 warmup，而 Post-LN 通常需要？

**Q4.** RMSNorm 与 LayerNorm 的区别在哪里？去掉均值中心化为什么在大模型上仍然有效甚至更好？

**Q5.** SwiGLU 的前向公式是什么？相比 ReLU FFN 增加了多少参数量？为什么通常要把 FFN 中间维度调小来补偿？

**Q6.** RoPE 的旋转矩阵如何构造？为什么 RoPE 的内积只依赖相对位置 \(i-j\)？推导二维情况下的形式。

**Q7.** RoPE 的外推缺陷是什么？NTK-aware 插值与 YaRN 分别从什么角度缓解长度外推问题？

**Q8.** ALiBi 与 RoPE 的设计哲学有何根本不同？ALiBi 的线性偏置为什么在某些长上下文任务上仍有竞争力？

**Q9.** MHA、MQA、GQA 在 KV 头数上的区别是什么？GQA 如何在推理显存与模型质量之间取折中？

**Q10.** 因果 mask 在训练与推理时的处理方式有何不同？训练时一次性计算全序列、推理时逐 token 生成，mask 如何保证等价？

**Q11.** 残差连接在深层 Transformer 中承担什么角色？去掉残差后梯度会怎样传播？

**Q12.** decoder-only 架构在 LLM 时代胜过 encoder-decoder，从训练目标、上下文利用、推理效率三个角度分析。

**Q13.** Weight tying（输入 embedding 与输出投影共享）的利弊是什么？什么情况下应该解绑？

**Q14.** 位置编码加在 embedding 上（绝对位置）与加在 attention logits 上（相对位置 / ALiBi），在信息流动上有何本质差异？

**Q15.** 把 RoPE base 从 10000 调到 500000 会影响什么？这个调整与上下文长度扩展是什么关系？

**Q16.** FFN 中间维度通常是 hidden 的 4 倍，这个比例从何而来？SwiGLU 架构下为什么常用 \(8/3\) 倍？

**Q17.** Attention 的 \(O(T^2)\) 复杂度在长序列下如何成为瓶颈？除了 FlashAttention，还有哪些降低注意力复杂度的路线？

**Q18.** attention dropout 与 residual dropout 分别放在哪里？大模型预训练中 dropout 通常设得很低甚至为 0，为什么？

**Q19.** LayerNorm 的 \(\epsilon\) 设得过大或过小分别有什么后果？

**Q20.** 为什么现代 LLM 几乎都使用 Pre-Norm？Post-Norm 在什么场景下仍然有优势？

**Q21.** 手写 RMSNorm 的前向与反向（概念级别），说明它相比 LayerNorm 少算了什么。

**Q22.** 序列长度从 4k 扩展到 128k 时，RoPE 的哪些频率分量会首先“失效”？从波长角度解释。

**Q23.** Attention 的 softmax 能否用 sigmoid 或线性归一化替代？各自会损失什么性质？

**Q24.** 为什么 LLM 的输出层通常 `bias=False`，而 LayerNorm 通常保留 bias（或 RMSNorm 去掉 bias）？

**Q25.** teacher forcing 与 scheduled sampling 的取舍是什么？LLM 预训练为什么几乎不用 scheduled sampling？

**Q26.** 位置插值（PI）与 NTK-aware 缩放的核心区别是什么？哪个对短序列性能的破坏更小？

**Q27.** 解释“attention sink”现象。它对长上下文推理与 KV Cache 压缩有什么启示？

**Q28.** 设计一个比 SwiGLU 更高效的激活函数，你会从哪些性质入手？门控机制是否必要？

**Q29.** Decoder-only 模型的最后一层 hidden state 到 logits 的映射中，是否应该加 LayerNorm？不同模型的做法有什么差异？

**Q30.** 给定一个训练好的 LLM，如何判断它是否“真正”利用了位置信息，而不是仅靠内容词袋？设计一个探测实验。

**Q31.** RoPE 的外推为什么“能外推又外推不好”？从频谱角度解释：哪些频率分量在超出训练长度后首先失真，哪些仍能维持相对关系？YaRN 将 RoPE 视为 Fourier Feature 的特例，并指出“不加区分地拉伸 RoPE embedding 会导致高频细节丢失”。这一洞察对理解外推失败的本质意味着什么？

**Q32.** 位置插值（PI）把所有频率按比例 \(1/s\) 压缩，而 NTK-aware 只改 base：\( b' = b \cdot s^{|D|/(|D|-2)} \)。为什么 PI 在微调后短序列 PPL 会轻微上升，而 NTK-aware 在未微调模型上外推更好、微调后却可能劣于 PI？从“高频信息保留”与“低频外推越界”两个角度解释。

**Q33.** YaRN 的 “NTK-by-parts” 具体做了什么？它按波长把维度分成三段：波长短于训练长度的维度不插值、长于训练长度的维度完全插值、中间维度加权插值。这种分段设计的动机是什么？YaRN 还引入 temperature scaling \( \sqrt{1/t} \approx 0.1 \ln s + 1 \)，它在其中承担什么角色？

**Q34.** 训练长度 4K、外推到 128K 时，RoPE 最低频分量的波长在 128K 对应多少个训练长度？给定 \(\theta_i = 10000^{-2i/d}\)，哪些维度在 4K 训练时从未经历过完整周期？这对外推意味着什么？

**Q35.** 动态 NTK 与静态 NTK 的区别在哪里？推理时根据当前序列长度动态调整 base，会带来什么工程复杂度？在 serving 系统中，同一模型处理 4K 与 128K 请求时 base 不一致，会导致什么行为不一致？


## 第二段 · 系统（新第 4—7 章）

### 新第 4 章 · 混合专家模型（MoE）【35 题 · 重点】

**Q1.** MoE 的 Total Parameters 与 Activated Parameters 分别指什么？为什么面试中必须分开讲？各举一个模型说明。

**Q2.** Top-K 路由的数学表达式是什么？K=1 与 K=2 在训练稳定性、推理成本、专家特化上分别有什么差异？

**Q3.** Expert Collapse（路由坍塌）是什么？从梯度角度解释为什么某些专家会“饿死”。

**Q4.** 负载均衡辅助损失的形式是什么？它如何惩罚不均衡的路由分布？系数设得过大有什么副作用？

**Q5.** 容量因子设得太小会导致什么？设得太大又会浪费什么？实际系统中如何动态调整？

**Q6.** Router Z-Loss 解决什么问题？它与负载均衡损失是互补还是替代关系？

**Q7.** MoE 的 All-to-All 通信发生在哪些阶段？为什么 Expert Parallelism 对网络带宽的要求远高于数据并行？

**Q8.** DeepSeekMoE 的“细粒度专家 + 共享专家”设计解决了传统 MoE 的什么痛点？

**Q9.** Mixtral 8x7B 的“8x7B”命名具有误导性。它的总参数量与激活参数量各是多少？为什么不是简单的 8×7B？

**Q10.** MoE 推理时是否所有专家都必须加载到显存？为什么“激活参数少”不等于“推理显存低”？

**Q11.** 路由分布长尾非常严重时，训练会面临哪些具体问题？如何检测？

**Q12.** MoE 的专家应放在同一张卡上还是不同卡上？各自的通信模式与显存权衡是什么？

**Q13.** 辅助损失系数、路由温度、专家数量三者之间如何相互影响？是否存在“调一个就必须重调另一个”的耦合？

**Q14.** 为什么 MoE 在相同激活参数量下通常比 Dense 模型有更高的容量，但不一定在困惑度上等价获胜？

**Q15.** MoE 训练中的 token dropping 发生在哪个环节？被丢弃 token 的梯度如何处理？

**Q16.** 如何在一个已训练好的 Dense 模型上“升级”为 MoE？继续训练的策略与初始化方式有哪些？

**Q17.** MoE 的负载均衡能否完全交给路由器自己学，而不加辅助损失？无辅助损失的 bias-based 方案如何工作？

**Q18.** 推理时 batch 内不同 token 路由到不同专家，如何高效实现这种“稀疏 GEMM”？Grouped GEMM 的作用是什么？

**Q19.** MoE 的 EP 与 TP 如何搭配？EP 过大时，All-to-All 的延迟如何成为瓶颈？

**Q20.** 给定 64 个专家、Top-2 路由、每卡 8 个专家，一个 batch 中有 1024 个 token，估算平均每卡需要处理多少 token，以及最坏情况下可能多少。

**Q21.** 路由器的输入应该用 token 的 hidden state 还是其他表示？路由器本身的参数量通常多大？为什么不能设得太复杂？

**Q22.** MoE 在 SFT / RLHF 阶段容易遇到什么问题？专家会变得更均衡还是更不均衡？

**Q23.** 如何评估一个 MoE 模型的“专家利用率”？除了平均负载，还应该看哪些分布指标？

**Q24.** 某些专家在推理时被频繁调用而其他专家几乎不用时，能否做专家剪枝或合并？风险是什么？

**Q25.** MoE 中的“专家并行”与“张量并行”在切分维度上有何本质不同？

**Q26.** 辅助损失是在每个 micro-batch 上算还是在累积后的全局 batch 上算？不同粒度对均衡效果有何影响？

**Q27.** 为什么 MoE 在低 batch size 推理场景下效率可能不如 Dense 模型？batch size 到多大时 MoE 才开始显现优势？

**Q28.** 共享专家的设计动机是什么？它是否削弱了 MoE 的“条件计算”优势？

**Q29.** 设计一个支持动态专家数量的 MoE 框架，你会如何在路由、通信、显存管理上做抽象？

**Q30.** MoE 的 Scaling Law 与 Dense 模型有何不同？在固定激活 FLOPs 下，增加专家数量还是增加单专家容量更划算？

**Q31.** 负载均衡辅助损失的解析形式是什么？Switch Transformer 的形式为 \( \text{loss} = N_{\text{exp}} \cdot \sum_i f_i \cdot P_i \)，其中 \( f_i \) 是被路由到专家 \( i \) 的 token 比例，\( P_i \) 是路由器给专家 \( i \) 的平均概率。为什么 \( f_i \cdot P_i \) 的乘积能同时约束“实际分配”与“平均偏好”？如果只约束其中一个，会发生什么？

**Q32.** Router Z-Loss 的具体形式是 \( L_z = (\log \sum_i e^{r_i})^2 \)。它惩罚的是 router logits 的 log-sum-exp，而不是负载分布本身。为什么 ST-MoE 要把它与负载均衡损失同时使用？Z-Loss 过大时，router 会变得“过于平滑”还是“过于尖锐”？

**Q33.** 容量公式 \( C = \lceil \alpha \cdot T \cdot k / N \rceil \) 中，\( T \) 是总 token 数、\( k \) 是 Top-K、\( N \) 是专家数、\( \alpha \) 是容量因子。\( \alpha=1.0 \) 且路由完全均匀时，每个专家刚好满载；路由不均时会发生什么？\( \alpha \) 取 1.25 的代价是什么？为什么实际系统很少让 \( \alpha \) 远大于 1.25？

**Q34.** Token dropping 发生在专家容量溢出时。被丢弃 token 的梯度如何处理？如果大量 token 被丢弃，训练信号会偏向什么类型的样本？长序列中容量约束往往在后期才触发，这会带来什么额外偏差？

**Q35.** 无辅助损失的负载均衡（如 DeepSeek 的 bias-based 方法）如何工作？它给每个专家维护一个可学习的 bias 项并动态调整。与辅助损失相比，这种方法在训练稳定性上有什么不同？辅助损失通过梯度同时约束“平均偏好”与“实际分配”，无辅助路径则根据历史负载调整下一轮选择偏置。两条路径的反馈机制有何本质差异？


### 新第 5 章 · GPU 与相关优化【30 题】

**Q1.** GPU 的 SM 数量与 CPU 核心数量在架构哲学上有何根本不同？为什么 GPU 用大量线程来“掩盖”延迟？

**Q2.** 计算强度（arithmetic intensity）的定义是什么？一个算子是 memory-bound 还是 compute-bound，如何用 roofline model 判断？

**Q3.** FlashAttention 解决的核心瓶颈是显存带宽还是显存容量？它通过 tiling 与 online softmax 避免了什么？

**Q4.** Online softmax 的数值稳定性如何保证？它在分块计算时如何累积 running max 与 running sum？

**Q5.** FlashAttention 的 \(O(T^2)\) 计算量并没有减少，为什么实际速度却大幅提升？

**Q6.** PagedAttention 借用了操作系统的什么思想？它解决的是 KV Cache 的哪种碎片化问题？

**Q7.** Kernel Fusion 在什么条件下收益最高？什么情况下融合反而会因为寄存器压力导致 occupancy 下降？

**Q8.** BF16 与 TF32 在精度和吞吐上分别如何取舍？训练中哪些矩阵乘法适合 TF32，哪些必须 BF16/FP16？

**Q9.** H100 的 Tensor Core 与 CUDA Core 在执行矩阵乘法时有何不同？为什么 Tensor Core 能达到远高于 CUDA Core 的 FLOPS？

**Q10.** 显存带宽与大模型推理 token 生成速度之间是什么定量关系？给出 decode 阶段的理论上限估算公式。

**Q11.** 为什么 decode 阶段是 memory-bound，而 prefill 阶段更接近 compute-bound？这种差异如何影响 serving 系统的调度策略？

**Q12.** FlashAttention 在训练与推理中的收益是否相同？decode 阶段能否直接套用 FlashAttention 的 tiling 思路？

**Q13.** 共享内存在 GEMM kernel 中承担什么角色？为什么 tiling 能减少对 HBM 的访问次数？

**Q14.** L2 Cache 命中率对大模型 kernel 性能有多大影响？哪些操作的 L2 局部性天然较差？

**Q15.** 混合精度训练中哪些算子必须保持 FP32？如果强行全部 BF16，训练会在哪里先崩溃？

**Q16.** 解释 element-wise 融合与 reduction 融合的区别。softmax 属于哪一类，融合难度在哪？

**Q17.** attention 序列长度很短（如 128）时，FlashAttention 还能带来收益吗？短序列下瓶颈转移到了哪里？

**Q18.** KV Cache 的显存占用公式是什么？给定 layers=32、heads=32、head_dim=128、seq=8192、batch=16、FP16，估算 KV Cache 大小。

**Q19.** 量化推理（GPTQ/AWQ）在 GPU 上加速的底层原理是什么？为什么不是所有层都适合低比特量化？

**Q20.** 什么是“算子融合的边界”？哪些相邻算子可以融合，哪些因为数据依赖或布局变换而无法融合？

**Q21.** GPU 的 occupancy 是什么？occupancy 高是否一定意味着性能好？举一个反例。

**Q22.** attention 的 softmax 在长序列下为什么容易成为数值稳定性瓶颈？FlashAttention 如何在不实例化完整矩阵的情况下完成归一化？

**Q23.** serving 中 prefill 与 decode 分别用什么并行策略？Chunked prefill 解决了什么问题？

**Q24.** 为一个自定义算子写 Triton kernel 时，如何判断它是否已经达到了合理的性能（与 cuBLAS / 手写 CUDA 对比的容忍度）？

**Q25.** GPU 显存层级（寄存器、shared memory、L1/L2、HBM）的带宽与延迟各是什么量级？为什么寄存器溢出会严重拖慢性能？

**Q26.** 投机解码为什么能加速？加速比受什么因素限制？draft model 的质量如何影响收益？

**Q27.** Continuous Batching 与静态 batching 的核心区别是什么？它如何提高 GPU 在 decode 阶段的利用率？

**Q28.** 为什么大模型推理中算术强度会随 batch size 增大而提高？batch size 增大到多少时 decode 从 memory-bound 转为 compute-bound？

**Q29.** 训练中 activation 的显存访问模式与推理中 KV Cache 的访问模式有何相似之处？FlashAttention 的 tiling 思路能否迁移到 KV Cache 管理？

**Q30.** 在单卡上优化 7B 模型的推理延迟，你会按什么优先级排列：量化、KV Cache 压缩、FlashAttention、CUDA Graph、投机解码？给出理由。


### 新第 6 章 · GPU 高性能编程【30 题】

**Q1.** CUDA 的 grid、block、thread 三层结构分别对应硬件的什么？一个 block 内的线程如何共享 shared memory？

**Q2.** 什么场景下你会选择手写 CUDA C++，什么场景下 Triton 就够了？给出判断框架。

**Q3.** Triton 的 `tl.load` / `tl.store` 与 CUDA 的全局内存访问在抽象层级上有何不同？Triton 替你做了哪些优化决策？

**Q4.** 编写高效 GEMM kernel 时，shared memory 的 tiling 为什么能提升性能？Bank conflict 是什么，如何避免？

**Q5.** Warp 级原语（`__shfl_sync`、`__ballot_sync`）在 FlashAttention 类 kernel 中承担什么角色？没有它们哪些操作会变慢？

**Q6.** 寄存器溢出（register spilling）为什么对性能是灾难性的？如何检测和缓解？

**Q7.** CUDA 的异步拷贝（`cp.async` / TMA）解决了什么问题？它与 load-then-compute 模式相比有什么优势？

**Q8.** 手写 FlashAttention 为什么比调用 PyTorch 的 `scaled_dot_product_attention` 更难获得性能优势？什么情况下仍然值得手写？

**Q9.** Triton kernel 的 `BLOCK_SIZE` 应该如何选择？它和 shared memory 容量、寄存器压力、occupancy 之间是什么关系？

**Q10.** 解释 coalesced memory access。一个 warp 的 32 个线程访问连续的 128 字节时发生几次内存事务？访问步长为 32 时又会怎样？

**Q11.** Tensor Core 的 MMA 指令对矩阵 shape 与数据布局有什么要求？为什么需要特定的 shared memory 排布（如 swizzling）？

**Q12.** Triton 的 `tl.dot` 底层会映射到 Tensor Core 吗？什么条件下 Triton 能生成高效的 MMA 指令？

**Q13.** 手写 reduction kernel（sum 或 max）时，如何避免 bank conflict 与 warp divergence？

**Q14.** CUDA Graph 解决了什么开销？为什么在 decode 阶段收益明显，而在 prefill 阶段收益有限？

**Q15.** 什么是 persistent kernel？它在什么场景下比启动大量小 kernel 更有优势？

**Q16.** 一个 Triton kernel 的性能只有 cuBLAS 的 60% 时，你会从哪些维度排查：内存访问、计算密度、occupancy、指令选择？

**Q17.** attention 的 softmax 在 CUDA 实现中为什么通常要和 max/sum reduction 融合在一起？分开写会多哪些 overhead？

**Q18.** Shared memory 的 bank 数量是多少？对于 FP32 数据，怎样的访问模式会产生 32-way bank conflict？

**Q19.** Triton 中如何实现“分块 + 重计算”（类似 FlashAttention 的 recomputation）？Triton 的编程模型对这种模式友好吗？

**Q20.** 什么是 warp specialization？Hopper 的 TMA + warp specialization 如何改变 kernel 设计范式？

**Q21.** 手写一个简单的 layer normalization kernel 需要考虑哪些优化点？与 PyTorch 内置实现相比，差距通常在哪里？

**Q22.** CUDA 的 `__syncthreads()` 在什么位置必须调用？漏调或多调分别会导致什么后果？

**Q23.** Triton 的 autotuning 如何工作？它搜索哪些配置？实际项目中如何控制 autotune 的时间成本？

**Q24.** 训练 kernel 中反向传播的原子操作（`atomicAdd`）可能成为瓶颈，有哪些替代方案？

**Q25.** 用 Triton 实现 fused SwiGLU + element-wise multiply 的 kernel，你会如何设计 block 维度与数据布局？

**Q26.** Tensor Core 的稀疏性支持（2:4 sparsity）是什么？它能在不损失太多精度的情况下加速哪些层？

**Q27.** NVLink 与 PCIe 的带宽差异对 kernel 设计有什么影响？什么时候应该把通信与计算 overlap？

**Q28.** 什么是 kernel launch overhead？在什么规模的问题下，launch overhead 会超过实际计算时间？

**Q29.** 手写 CUDA softmax 时，如何在一个 block 内完成跨 warp reduction？warp shuffle 与 shared memory 各用在什么阶段？

**Q30.** 为一个新的注意力变体（如 sliding window + global token）写 kernel，你会如何复用 FlashAttention 的 tiling 框架？需要修改哪些部分？


### 新第 7 章 · 分布式训练【35 题 · 重点】

**Q1.** 数据并行、张量并行、流水线并行各自切分的是什么维度？它们分别解决什么扩展性问题，代价是什么？

**Q2.** ZeRO-1/2/3 分别切分优化器状态、梯度、参数的哪一部分？每个 stage 节省的显存与增加的通信量如何量化？

**Q3.** FSDP 与 ZeRO-3 在核心思想上是否等价？工程实现上（parameter sharding、all-gather 时机）有何异同？

**Q4.** Ring All-Reduce 的通信量为什么约为参数量的 2 倍？Reduce-Scatter + All-Gather 两个阶段如何实现？

**Q5.** 张量并行中，Megatron-LM 的列并行与行并行分别切分线性层的哪个维度？为什么 attention 的 QKV 投影适合列并行？

**Q6.** 流水线并行中，GPipe 与 1F1B 调度策略的显存-吞吐权衡是什么？bubble 比例如何估算？

**Q7.** 给定 70B 模型、64 张 GPU、每张 80GB，显存预算如何拆解？DP/TP/PP/ZeRO 如何组合才能装下？

**Q8.** 梯度累积在分布式训练中如何与数据并行配合？global batch size 的语义是否改变？

**Q9.** 跨机 InfiniBand 与机内 NVLink 的带宽差异如何影响并行策略？TP 通常限制在机内，为什么？

**Q10.** All-Reduce 与 All-Gather 的通信模式有何不同？Reduce-Scatter 在 ZeRO 中如何使用？

**Q11.** 训练中某张卡比其他卡慢很多（straggler），可能的原因有哪些？如何检测和缓解？

**Q12.** FSDP 的 `sharding_strategy` 中，FULL_SHARD、SHARD_GRAD_OP、NO_SHARD 分别对应什么？如何选择？

**Q13.** 混合并行中，TP 与 PP 的通信模式有何本质不同？为什么 TP 需要每层都通信，而 PP 只需在 stage 边界通信？

**Q14.** 序列并行（SP）解决什么问题？它与张量并行如何配合降低激活显存？

**Q15.** 分布式训练中 checkpoint 的保存策略有哪些？sharded checkpoint 与 consolidated checkpoint 的取舍是什么？

**Q16.** All-Reduce 的 bucket size 设得过大或过小分别会有什么问题？

**Q17.** 计算 Ring All-Reduce 在 8 卡、每卡梯度 10GB 下的通信量，并与 Tree All-Reduce 的延迟特性做对比。

**Q18.** 3D 并行中，DP、TP、PP 的并行度应如何随集群规模调整？存在“最优比例”吗？

**Q19.** 梯度压缩（FP16 通信、Top-K sparsification）在什么场景下值得使用？对收敛有什么风险？

**Q20.** 大模型训练为什么通常不用 `nn.DataParallel` 而用 DDP？DataParallel 的瓶颈在哪里？

**Q21.** 流水线并行中的 micro-batch 与梯度累积中的 micro-batch 是同一个概念吗？目的有何不同？

**Q22.** TP 通信成为瓶颈时，有哪些优化手段（通信 overlap、NVLink 拓扑感知、异步 TP）？

**Q23.** ZeRO-Offload 与 ZeRO-Infinity 分别把什么卸载到哪里？PCIe 带宽如何成为新的瓶颈？

**Q24.** 分布式训练中的随机种子如何管理？数据加载器的 shuffle 在不同 rank 之间如何保证不重叠且可复现？

**Q25.** 1000 卡扩展到 2000 卡时吞吐没有翻倍，你会从哪些维度排查扩展效率损失？

**Q26.** 专家并行（EP）与数据并行、张量并行在通信模式上有何本质不同？All-to-All 的延迟为什么难以完全隐藏？

**Q27.** FSDP 的 `auto_wrap_policy` 应如何设计？wrap 粒度太粗或太细分别有什么后果？

**Q28.** 梯度同步的 overlap 是如何实现的？在反向传播的哪个阶段开始 all-reduce 才能最大化 overlap？

**Q29.** 为一个 1T 参数模型设计训练集群网络拓扑，你会如何考虑 NVLink domain、InfiniBand rail-optimized 与存储带宽？

**Q30.** 分布式训练中 loss 在不同 rank 上出现微小差异是正常的吗？什么情况下会从“正常浮点差异”演变成“实质性不一致”？

**Q31.** ZeRO 的单卡显存如何精确拆解？FP16 训练、Adam、参数量 \( \Phi \) 下，baseline 单卡显存为 \( \Phi(2+2+12) = 16\Phi \) 字节（2 参数 + 2 梯度 + 12 优化器状态：FP32 master weights 4 + Adam m 4 + Adam v 4）。ZeRO-1/2/3 分别把它降到多少？写出每个阶段的显存表达式。

**Q32.** ZeRO-3 的通信量为什么比标准数据并行多出约 50%？参数被分片后，前向需要 all-gather 参数，反向需要再次 all-gather。标准 DDP 的 all-reduce 每卡通信量约为 \( 2\Psi \)，ZeRO-3 的约 \( 1.5 \times 2\Psi = 3\Psi \)，额外开销从何而来？

**Q33.** Ring All-Reduce 的单卡通信量为什么约为 \( 2\Phi \)？它由 Reduce-Scatter 与 All-Gather 两个阶段各 \( \Phi \) 组成；每个阶段在 \( N \) 卡环上每卡发送 \( (N-1)/N \cdot \Phi \) 的数据，当 \( N \) 较大时近似为 \( \Phi \)。推导这个结果，并说明为什么它与卡数近似无关。

**Q34.** Ring All-Reduce 与朴素 All-Reduce 的系统总通信量相近，为什么 Ring 更适合大规模集群？朴素方案的单点瓶颈具体出现在哪里？

**Q35.** 流水线并行的 bubble ratio 近似为 \( (P-1)/M \)，其中 \( P \) 是 stage 数、\( M \) 是 micro-batch 数。如果 PP=8、micro-batch=32，bubble 占比是多少？要把 bubble 压到 10% 以下，micro-batch 至少需要多少？


## 第三段 · 规模与数据（新第 8—11 章）

### 新第 8 章 · Scaling Laws【30 题】

**Q1.** Kaplan 2020 与 Chinchilla 2022 的 Scaling Law 结论有何关键差异？Chinchilla 推翻了什么实践？

**Q2.** Chinchilla 的 compute-optimal 结论用一句话概括是什么？固定算力 \( C \) 下，参数量 \( N \) 与 token 数 \( D \) 大致按什么比例分配？

**Q3.** “20 tokens per parameter”这个经验法则从何而来？什么条件下成立，什么条件下会偏离？

**Q4.** 算力预算翻 10 倍，按 Chinchilla 最优，参数量与 token 数各应增加多少倍？\( N^* \propto C^{0.5} \)、\( D^* \propto C^{0.5} \) 的直观含义是什么？

**Q5.** 拟合 Scaling Law 通常用哪几个变量？\( L(N,D) = E + A/N^\alpha + B/D^\beta \) 中每一项的物理含义是什么？

**Q6.** 如何用少量小规模实验（1M、10M、100M 参数）外推 10B 模型的 loss？外推失效的常见原因有哪些？

**Q7.** Chinchilla 最优是“训练最优”，但推理成本呢？如果考虑部署成本，最优的 \( N \) 与 \( D \) 会如何偏移？

**Q8.** Llama 系列训练了远超 Chinchilla 最优的 token 数（8B 模型训 15T tokens），是否违反 Scaling Law？如何解释？

**Q9.** Scaling Law 对数据质量是否敏感？数据质量提升时拟合曲线会怎样变化？

**Q10.** 什么是 emergence 现象？Scaling Law 的平滑幂律与能力涌现之间是否存在矛盾？

**Q11.** 给定 \( C = 6ND \)，如果 \( N=10B \)，Chinchilla 最优下 \( D \) 约为多少？训练总 FLOPs 约为多少？

**Q12.** 多模态模型、MoE 模型的 Scaling Law 与 Dense 语言模型有何不同？激活参数量还是总参数量进入幂律？

**Q13.** 学习率与 batch size 是否也需要随模型规模按 Scaling Law 调整？有哪些经验规则？

**Q14.** 小规模实验显示 loss 已进入“不可约损失”平台时，外推到大模型这个平台会平移吗？

**Q15.** 如何设计 Scaling 实验的网格？模型规模、数据规模、学习率各取几个点才能可靠拟合？

**Q16.** Scaling Law 的拟合为什么通常用“最后一步 loss”而不是“最优 loss”？两者在高学习率下有何差异？

**Q17.** 数据受限场景下（高质量 token 不够），Scaling Law 应如何修正？重复训练数据如何影响幂律指数？

**Q18.** “推理最优”与“训练最优”如何取舍？固定推理预算时，应选更大模型配更少 token，还是更小模型配更多 token？

**Q19.** 如何判断一个模型已经“过参数化”或“欠训练”？从 loss、下游指标、训练 token 数三者关系分析。

**Q20.** Scaling Law 能否预测下游任务表现（MMLU、HumanEval）？如果不能，缺了什么？

**Q21.** Chinchilla 实验中 70B 模型训 1.4T tokens 击败 280B 训 300B tokens，这个结论对工业界的直接影响是什么？

**Q22.** 把模型宽度与深度分开看，哪一个对 loss 幂律指数影响更大？有统一结论吗？

**Q23.** MoE 的 Scaling Law 中，“总参数”与“激活参数”哪个是更好的 scaling 变量？

**Q24.** Scaling Law 的指数为什么在不同数据分布、不同分词器下会变化？跨数据集外推可靠吗？

**Q25.** 训练 loss 突然偏离 Scaling 预测曲线，通常意味着什么？数据污染、架构改动、数值问题如何区分？

**Q26.** 如何用 Scaling Law 指导“是否继续训练一个已有模型”的决策？继续训练的边际收益如何估算？

**Q27.** 合成数据是否改变了 Scaling Law 的适用边界？什么条件下合成数据可以“替代”真实 token？

**Q28.** 两个模型 A（7B，2T tokens）与 B（13B，1T tokens），仅凭 Scaling Law 能判断谁更强吗？还需要什么信息？

**Q29.** 多语言场景下，Scaling Law 是按语言分别拟合还是统一拟合？低资源语言的幂律指数通常如何表现？

**Q30.** 向面试官解释“Chinchilla 之后我们为什么还在训更大的模型”，你会用哪三个论点组织回答？


### 新第 9 章 · 推理【35 题 · 重点】

**Q1.** Prefill 与 Decode 两个阶段的计算特征有何本质区别？为什么 Decode 是 memory-bound 而 Prefill 更接近 compute-bound？

**Q2.** KV Cache 为什么能加速自回归生成？它避免了哪些重复计算？

**Q3.** 给定 layers=80、KV heads=8、head_dim=128、seq=32k、batch=8、FP16，估算 KV Cache 显存占用。

**Q4.** PagedAttention 的“分页”思想具体解决了 KV Cache 的哪种碎片化问题？逻辑块与物理块如何映射？

**Q5.** PagedAttention 的“块共享”在什么场景下节省显存？多轮对话与并行采样如何利用它？

**Q6.** Continuous Batching 解决了静态 batching 的什么浪费？它和 PagedAttention 之间是什么关系？

**Q7.** 投机解码为什么能加速？draft model 的质量与接受率如何决定加速比？

**Q8.** 投机解码的验证阶段如何保证输出分布与目标模型完全一致？拒绝采样如何工作？

**Q9.** GPTQ 与 AWQ 在量化思路上有何不同？各自适合什么场景？

**Q10.** KV Cache 量化（FP8 KV）的收益与风险是什么？哪些层/头的 KV 对量化更敏感？

**Q11.** FlashAttention 在 prefill 阶段能加速，但 decode 阶段为什么不直接适用？FlashDecoding 做了什么调整？

**Q12.** decode 阶段 batch size 增大时，算术强度如何变化？batch size 到多大时可能转为 compute-bound？

**Q13.** 什么是 TTFT 与 TPOT？它们分别受哪些因素支配？

**Q14.** 在线 serving 中如何权衡 TTFT 与 TPOT？Chunked prefill 如何影响两者？

**Q15.** KV Cache 的显存占用如何随序列长度、层数、头数增长？线性增长的后果是什么？

**Q16.** 除了 PagedAttention，还有哪些 KV Cache 压缩手段（滑动窗口、H2O、attention sink）？各自牺牲了什么？

**Q17.** 推理框架为什么需要 prefix caching？多轮对话中如何最大化复用？

**Q18.** 量化推理中 W8A8、W4A16、W4A8 分别指什么？为什么激活量化通常比权重量化更困难？

**Q19.** 目标模型与 draft model 的词表不同时，投机解码还能用吗？需要什么额外处理？

**Q20.** 解释“推理中的内存墙”。HBM 带宽如何成为 decode 速度的硬上限？给出理论 tok/s 估算公式。

**Q21.** GQA 在推理时如何减少 KV Cache？MQA 与 GQA 的显存节省与模型质量之间如何权衡？

**Q22.** vLLM 的调度器如何在延迟与吞吐之间做 trade-off？preemption 与 swapping 分别解决什么问题？

**Q23.** 请求的输入长度差异很大时，静态 batching 会造成什么浪费？Continuous Batching 如何缓解？

**Q24.** 投机解码在多轮对话中是否总是有效？什么情况下加速比接近 1 甚至变慢？

**Q25.** 推理引擎的 CUDA Graph 优化在什么阶段收益最大？动态 shape 如何限制其应用？

**Q26.** 给定一个 7B 模型在 A100 上的理论显存带宽约 2TB/s，估算 FP16 下 decode 的 tok/s 上限（忽略 KV Cache 读取）。

**Q27.** 量化校准为什么必要？校准数据分布与真实推理分布不一致会怎样？

**Q28.** 部署 128k 上下文的模型时，KV Cache 会占多少显存？有哪些工程手段让它在单卡上可行？

**Q29.** 推理 serving 中的“公平性”如何定义？一个长请求与多个短请求并发时，调度器应如何分配资源？

**Q30.** 从“训练—推理一致性”角度看，量化、KV Cache 压缩、投机解码各自可能引入什么数值偏差？如何检测和缓解？

**Q31.** PagedAttention 常被描述为“memory manager, not a faster kernel”。它的吞吐提升来自“在相同显存下容纳更多请求”，而不是单次 kernel 变快。如果只优化 kernel 而不做分页，瓶颈会落在哪里？传统连续分配下 KV Cache 的显存利用率可能低至 20%–40%，分页后接近满利用率，被消除的浪费具体分为内部碎片、外部碎片与预留浪费三类。

**Q32.** KV Cache 的精确估算公式为 \( 2 \times L \times H_{kv} \times d \times \text{bytes} \times T \times B \)，其中 \( 2 \) 来自 K 和 V，\( L \) 是层数，\( H_{kv} \) 是 KV 头数，\( d \) 是 head_dim，\( T \) 是序列长度，\( B \) 是 batch。对 LLaMA-3.1 8B（32 层、8 KV 头、head_dim=128、FP16），计算 seq=8192、单请求的 KV Cache；若模型本身占 16GB、总显存 80GB，留给 KV Cache 约 60GB，理论最大并发请求数是多少？

**Q33.** Continuous Batching 与 PagedAttention 为什么“需要彼此”？没有 Continuous Batching，分页的细粒度资源管理能否单独发挥吞吐优势？没有分页，Continuous Batching 的动态入批会受什么限制？

**Q34.** Chunked Prefill 解决的是什么问题？它如何保护 inter-token latency 不被长 prompt 的 prefill 阶段拖垮？它与 Continuous Batching 是同一层的东西吗？

**Q35.** 投机解码的加速比近似为 \( \text{speedup} \approx \frac{1-\alpha^{k+1}}{(1-\alpha)(1+ck)} \)，其中 \( \alpha \) 是接受率、\( k \) 是 draft 长度、\( c \) 是 draft 相对 target 的成本比。当 \( \alpha \) 很低时，加速比为什么趋近于 1 甚至小于 1？什么条件下增加 \( k \) 反而有害？


### 新第 10 章 · 数据工程【30 题】

**Q1.** Common Crawl 到预训练数据集之间通常需要经过哪几类处理步骤？每一步的核心目标是什么？

**Q2.** 为什么去重是性价比最高的数据质量手段之一？重复数据对模型训练有哪些具体危害？

**Q3.** MinHash + LSH 的去重原理是什么？Jaccard 相似度如何被 MinHash 估计？

**Q4.** 精确去重（MD5）与模糊去重（MinHash/SimHash）分别适合什么场景？为什么大规模预训练以模糊去重为主？

**Q5.** 质量过滤有哪些常见信号（分类器分数、困惑度、启发式规则）？各自有什么偏差？

**Q6.** 数据配比（中英比例、代码比例、数学比例）如何影响模型能力？配比实验通常怎么设计？

**Q7.** 目标模型以中文为主时，数据配比策略需要做哪些调整？英文比例下降会牺牲什么？

**Q8.** PII 脱敏在预训练数据中的重要性是什么？过度脱敏与脱敏不足分别有什么风险？

**Q9.** Data Curriculum 是什么？从简单到难的数据排序是否真的有效？

**Q10.** 如何检测预训练数据中的基准测试污染？有哪些实用的去污染策略？

**Q11.** 给定 5TB 文本、重复率 28%，去重后降到 3%，估算节省了多少训练 token。去重对训练时间的影响是否线性？

**Q12.** MinHash 的 band 数与 row 数如何影响召回率和精度？设计一个参数组合并说明理由。

**Q13.** 为什么“数据质量”很难定义？分类器打分会引入什么新的偏差？

**Q14.** 数据配比在预训练、继续预训练、SFT 三个阶段是否需要变化？

**Q15.** 代码数据配比增加如何影响自然语言能力？是否存在“代码数据提升推理”的因果证据？

**Q16.** 合成数据在预训练中的使用边界在哪里？什么情况下合成数据会“毒化”模型？

**Q17.** 多语言数据配比中如何避免低资源语言被高资源语言淹没？温度采样如何工作？

**Q18.** 去重应在文档级、段落级还是句子级进行？不同粒度分别会漏掉或误删什么？

**Q19.** 解释 data lineage 的概念。一个可审计的数据流水线需要记录哪些元数据？

**Q20.** 训练 loss 在某个数据域上异常低可能说明什么？如何用数据配比与去重来解释？

**Q21.** 质量分类器的训练数据从何而来？如何避免分类器把“风格差异”误判为“质量差异”？

**Q22.** 数据清洗中的毒性过滤是否会过度删除某些群体的语言？如何评估公平性影响？

**Q23.** 为什么在预训练后期调整数据配比（annealing）可能比从头到尾固定配比更有效？

**Q24.** 数据配比的消融实验在小模型上做，迁移到大模型时需要注意什么？配比是否随模型规模变化？

**Q25.** 去重时如何平衡“删除重复”与“保留自然重复”（常见短语、成语）？

**Q26.** 数据流水线的吞吐瓶颈通常在哪里？如何用分布式 MinHash 避免数据倾斜与内存溢出？

**Q27.** 只能做一种数据质量干预（去重、过滤、配比）时，你会选哪个？依据是什么？

**Q28.** 训练数据的时效性如何管理？知识密集型任务中，旧数据应降采样还是完全剔除？

**Q29.** 如何评估数据工程对最终模型能力的贡献？设计一个可操作的消融框架。

**Q30.** 从数据工程角度解释：为什么“更多数据”不总是等于“更好模型”？举出至少三个机制。


### 新第 11 章 · 评估与基准测试【30 题】

**Q1.** Perplexity 能反映什么、不能反映什么？为什么低 PPL 不等于强推理能力？

**Q2.** PPL 按 token 平均与按句子平均对长文本评估有何不同影响？

**Q3.** MMLU 的 few-shot 设置如何影响分数？5-shot 与 0-shot 的差距说明了什么？

**Q4.** HumanEval 的 pass@k 如何计算？朴素 pass@k 估计为什么有偏？无偏估计量的形式是什么？

**Q5.** pass@k 与 pass^k 分别适合什么场景？k 增大时两者如何分道扬镳？

**Q6.** lm-evaluation-harness 中 prompt 模板与答案抽取规则如何影响分数？如何避免“评测过拟合”？

**Q7.** HELM 与 MMLU 的评估哲学有何不同？HELM 的多维度评估包含哪些方面？

**Q8.** Arena 的 Elo 排名与静态 benchmark 分数分别捕捉了什么、遗漏了什么？

**Q9.** 静态 benchmark 为什么会“饱和”？饱和后应用什么评估范式替代？

**Q10.** C-Eval 与 MMLU 在中文场景下的差异是什么？直接用 MMLU 评估中文模型会有什么问题？

**Q11.** AlpacaEval 的胜率评估与人类评估的相关性如何？它的长度偏差是什么？

**Q12.** 代码评估中功能正确性（unit test）与 pass@k 是什么关系？pass@k 高是否一定意味着代码质量好？

**Q13.** 数学评估中精确匹配会漏掉哪些正确但格式不同的答案？如何设计更鲁棒的答案抽取？

**Q14.** 模型在 MMLU 上 90 分但在真实对话中表现很差，可能的原因有哪些？

**Q15.** 如何检测 benchmark 污染？成员推断在污染检测中能起什么作用？

**Q16.** 多轮对话如何评估？单轮 benchmark 分数能否预测多轮表现？

**Q17.** 安全评估应包含哪些维度？红队测试与自动化安全 benchmark 各自的局限是什么？

**Q18.** 模型只在 A100 上评测，换到 H100 后分数变化说明什么？数值差异与真实能力差异如何区分？

**Q19.** 两个模型在 MMLU 上差 1 分，需要多少样本才能判断这个差异是统计显著的？

**Q20.** 为什么“评估集设计”本身是一项研究能力？设计一个能区分“记忆”与“泛化”的评估集。

**Q21.** PPL 在领域偏移下如何失效？领域内 PPL 低是否意味着领域外也好？

**Q22.** 用 LLM-as-a-Judge 做评估的偏差来源有哪些（位置偏差、长度偏差、自我偏好）？如何缓解？

**Q23.** 一个模型 pass@1=80%，另一个 pass@10=90%，哪个更强？取决于什么使用场景？

**Q24.** 基准测试的天花板效应与地板效应分别是什么？如何设计一个避免两者的评估集？

**Q25.** 如何评估模型的校准？ECE 在 LLM 上如何计算？

**Q26.** 多语言评估中，翻译 benchmark 与原生 benchmark 的分数差距说明了什么？

**Q27.** RAG 任务的评估应覆盖哪些维度？检索质量、答案忠实度、引用准确性如何分别度量？

**Q28.** needle-in-a-haystack 测试的局限性是什么？它没有测到什么？

**Q29.** 为垂直领域（医疗、法律）构建可信评估集时，专家标注的成本与质量如何平衡？

**Q30.** 从“评估驱动开发”角度，你会在预训练、SFT、RL 各阶段引入哪些核心评估？目标有何不同？


## 第四段 · 对齐与前沿（新第 12—15 章）

### 新第 12 章 · 大模型的基本训练流程【35 题 · 重点】

**Q1.** 预训练、SFT、RLHF/RL 三个阶段的目标函数分别是什么？它们各自解决什么问题？

**Q2.** SFT 的模仿学习与 RL 的探索优化在对齐中各自擅长什么、各自留下什么隐患？

**Q3.** 只做 SFT 不做 RLHF，模型会出现哪些典型行为问题？为什么 SFT 难以解决这些问题？

**Q4.** RLHF 的三步流程（SFT → Reward Modeling → PPO）中，每一步的失败如何传导到下一步？

**Q5.** DPO 与 PPO 的核心区别是什么？DPO 跳过了什么、假设了什么？

**Q6.** DPO 的损失函数如何从 Bradley-Terry 偏好模型推导出来？它隐式优化的策略是什么？

**Q7.** 奖励模型（RM）的训练数据从何而来？人类标注的一致性如何影响 RM 质量？

**Q8.** Alignment Tax 是什么？RLHF 之后模型在某些能力上下降的现象如何解释和缓解？

**Q9.** SFT 的数据配比与格式对最终模型行为有多大影响？为什么 SFT 数据“质量 > 数量”？

**Q10.** PPO 中的 KL 惩罚项约束的是什么？KL 系数设得过大和过小分别会怎样？

**Q11.** 在线 PPO 与离线 DPO 在工程复杂度、稳定性、最终上限上如何对比？

**Q12.** SFT 应训到“过拟合”还是“欠拟合”？如何判断 SFT 停止的时机？

**Q13.** RLHF 中的 reward hacking 是什么？模型可能如何“欺骗”奖励模型？

**Q14.** SFT 数据中混入低质量样本时，模型会学到什么？如何检测和清洗？

**Q15.** 继续预训练与 SFT 的边界在哪里？领域适配应该用哪一种？

**Q16.** RLHF 为什么通常需要独立的 reward model，而不是直接用人类实时反馈？实时反馈的代价是什么？

**Q17.** SFT 的 loss 只在 assistant 回复的 token 上计算还是包括 user prompt？不同选择对模型行为有何影响？

**Q18.** DPO 中的 \( \beta \) 参数控制什么？\( \beta \) 增大和减小分别使模型更保守还是更激进地偏离参考策略？

**Q19.** 多轮对话的 RLHF 如何做 credit assignment？每一轮都奖励还是只在最后一轮奖励？

**Q20.** reward model 与 policy model 共享底座会带来什么好处和风险？

**Q21.** RLHF 训练不稳定时，应优先调哪些超参数？reward 归一化、advantage 归一化、KL 系数、学习率分别起什么作用？

**Q22.** SFT 阶段是否需要做数据去重与去污染？为什么？

**Q23.** 拒绝采样在 SFT/RL 流程中如何使用？它与 best-of-n 的关系是什么？

**Q24.** 模型经过 RLHF 后“变得更啰嗦”，从奖励建模与优化目标角度解释原因。

**Q25.** 如何评估 SFT 与 RLHF 各自对最终模型的贡献？设计一个消融实验。

**Q26.** 为代码模型设计后训练流程时，SFT 数据与 RL 奖励应如何定制？

**Q27.** RLHF 中的偏好数据与示范数据在信息量上有何不同？为什么偏好数据更便宜？

**Q28.** DPO 是否完全不需要 reward model？它的隐式奖励与显式 RM 在行为上有何差异？

**Q29.** 训练流程中，分词器、模型架构、数据配比三者哪一个的改动代价最大？为什么？

**Q30.** 从“预训练—SFT—RLHF”全流程看，哪些能力是预训练决定的、哪些是后训练“唤醒”的？举一个具体例子。

**Q31.** DPO 损失的完整推导分几步？从 KL 约束的 reward maximization 出发，最优策略满足 \( \pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta) \)。反解出隐式奖励 \( r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x) \)。代入 Bradley-Terry \( P(y_w \succ y_l) = \sigma(r_w - r_l) \) 后，\( \beta \log Z(x) \) 为什么在 pairwise 差中消掉？这个消去对 DPO 的“无 reward model”性质意味着什么？

**Q32.** DPO 的最终损失为 \( -\log\sigma(\beta \log\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}) \)。这里的 \( \beta \) 与 RLHF 中 KL 惩罚系数是什么关系？\( \beta \) 增大时，模型是更接近还是更远离参考策略？如果 \( \beta \to \infty \)，DPO 退化成什么？

**Q33.** PPO 的 clipped surrogate objective 中，\( r_t = \pi_\theta / \pi_{\theta_{\text{old}}} \)，clip 范围是 \( [1-\epsilon, 1+\epsilon] \)。为什么要 clip？如果不 clip，重要性采样比过大会导致什么？clip 后的梯度在什么条件下为零？

**Q34.** GAE 在 PPO 中平衡偏差与方差。\( \lambda=0 \) 与 \( \lambda=1 \) 分别对应什么极端？LLM 的 RLHF 中通常如何设置 \( \lambda \)？为什么长序列生成中 GAE 的方差控制尤其重要？

**Q35.** 在线 PPO 与离线 DPO 的样本效率如何比较？PPO 需要在线 rollout 但能探索；DPO 是离线监督但不能探索。什么任务上 DPO 可能反超 PPO，什么任务上 PPO 的优势不可替代？


### 新第 13 章 · 可验证奖励的强化学习（RLVR）【35 题 · 重点】

**Q1.** RLVR 与经典 RLHF 的核心区别是什么？“奖励可验证”带来了哪些工程优势与新的挑战？

**Q2.** GRPO 相对于 PPO 在优化目标或采样方式上做了哪些简化？它为什么不需要 value model？

**Q3.** GRPO 的组内相对优势如何计算？组大小对优势估计的方差有什么影响？

**Q4.** 可验证奖励的验证器如果存在漏洞（只看答案格式），模型会如何利用？如何缓解 reward hacking？

**Q5.** Outcome Reward 与 Process Reward 分别适合什么任务？各自的标注成本与质量瓶颈在哪里？

**Q6.** RLVR 在数学推理上效果显著，但在开放域对话上为什么难以直接套用？

**Q7.** 验证器只能判断最终答案对错时，模型是否可能“猜对但过程错误”？如何检测？

**Q8.** GRPO 的组内归一化在奖励方差很小或很大时分别会怎样？需要什么修正？

**Q9.** RLVR 训练中模型会不会退化到“只输出验证器喜欢的格式”？如何设计奖励来避免？

**Q10.** DeepSeek-R1 的训练方案中，冷启动 SFT 与 RLVR 的先后顺序如何安排？为什么需要冷启动？

**Q11.** Rule-based verifier 的覆盖范围有限。无法规则验证的任务，RLVR 的替代路线是什么？

**Q12.** PPO 中的 value model 在 RLVR 中能否被完全省略？省略后 credit assignment 的精度损失有多大？

**Q13.** RLVR 的样本效率如何？相比 SFT，达到相同数学准确率需要多少倍 rollout？

**Q14.** 如何设计一个多答案验证器（数学题有多种正确形式）？验证器的召回率与精度如何权衡？

**Q15.** RLVR 训练中 KL 约束是否仍然必要？不加 KL 时模型会怎样漂移？

**Q16.** GRPO 的组内基线与 PPO 的 GAE 有何本质不同？哪种方差更低？

**Q17.** 验证器本身是一个小模型而不是规则系统时，RLVR 的性质会发生什么变化？它还“可验证”吗？

**Q18.** 代码任务的 RLVR 中，单元测试作为验证器的局限是什么？测试覆盖率如何影响奖励信号？

**Q19.** RLVR 能否提升通用推理能力，还是只提升特定任务的模式匹配？设计实验区分两者。

**Q20.** 训练中大部分 rollout 都失败（奖励全 0）时，GRPO 的梯度会怎样？如何缓解稀疏奖励？

**Q21.** RLVR 与拒绝采样微调（RFT）之间是什么关系？RFT 可以看作 RLVR 的什么特例？

**Q22.** 验证器的“严格程度”如何影响训练？太严格导致奖励稀疏，太宽松导致 reward hacking，如何找平衡？

**Q23.** RLVR 在多语言数学题上的表现如何？验证器需要为每种语言单独设计吗？

**Q24.** 从信息论角度看，可验证奖励比人类偏好奖励“信息密度”更高还是更低？为什么？

**Q25.** RLVR 训练后模型在验证集上提升但在人类评估中下降，可能的原因是什么？

**Q26.** GRPO 的组大小在训练不同阶段是否需要调整？早期与后期分别适合多大的组？

**Q27.** RLVR 训练中是否应混合一部分通用 SFT 数据来防止能力遗忘？混合比例如何确定？

**Q28.** 如何评估一个 RLVR 系统的样本效率？rollout 成本、训练步数、最终准确率三者如何综合衡量？

**Q29.** 可验证奖励的边界在哪里？哪些任务原则上不可能有可靠的自动验证器？

**Q30.** 从“RLVR → 通用 RL”的路径看，当前方法缺失的关键组件是什么？你如何看待 process reward 与 outcome reward 的融合方向？

**Q31.** GRPO 的组内归一化 advantage 为 \( \hat{A}_i = (r_i - \text{mean}(\mathbf{r}))/\text{std}(\mathbf{r}) \)，其中 \( \mathbf{r} = (r_1,\dots,r_G) \) 是同一 prompt 下 \( G \) 条 rollout 的奖励。如果组内所有奖励完全相同，\( \hat{A}_i \) 是什么？梯度会变成什么？这种“零方差组”在训练中占比很高时，样本效率如何下降？

**Q32.** GRPO 组内归一化省掉了 value model。对一个 7B policy，value model 通常也是 7B 量级，省掉它相当于把训练显存降低多少比例？省掉的代价是 advantage 估计从“逐 token GAE”退化为“整条回复广播”。这种广播在数学推理中是否等价于假设“回复内每个 token 贡献相同”？这个假设合理吗？

**Q33.** RLVR 的验证器如果是规则系统，它的精度（把错误判对）与召回率（把正确判错）如何影响训练？精度低导致 reward hacking，召回率低导致奖励稀疏。如何设计一个可调的验证器来平衡两者？验证器噪声下的 GRPO 修正方向是什么？

**Q34.** 代码任务的 RLVR 中，单元测试的覆盖率通常很低。模型可能学会“通过测试”而不是“写对代码”。这与 reward hacking 是什么关系？隐藏测试与性质测试（property-based testing）分别从哪个角度缓解？

**Q35.** DeepSeek-R1 的训练流程中，冷启动 SFT 与 RLVR 的先后顺序是什么？冷启动解决了 RL 初期的什么问题？如果没有冷启动、直接从 base model 做 RLVR，会面临什么困难？RLVR 在 GRPO 中只使用二元信号（对/错）就足以区分组内 winning completion 与 losing completion，这个说法在什么条件下成立？


### 新第 14 章 · 多模态模型（CLIP → LLaVA → Omni）【30 题】

**Q1.** CLIP 的对比学习目标为什么能让图文表征对齐？InfoNCE 损失如何工作？

**Q2.** CLIP 的零样本分类能力从何而来？边界在哪里？

**Q3.** LLaVA 类模型将视觉编码器输出接入 LLM 时，projector（MLP）的关键工程权衡是什么？

**Q4.** 直接取 CLIP 的 pooled 输出（单个向量）而不是 patch tokens，会丢失什么信息？

**Q5.** LLaVA 的两阶段训练分别冻结/解冻哪些部分？第二阶段为什么通常要解冻 LLM？

**Q6.** 视觉编码器用 CLIP ViT 与 SigLIP 有何区别？SigLIP 的 sigmoid 损失相比 softmax 对比学习有何优势？

**Q7.** 一张图片在 LLaVA 中通常被编码为多少个 visual token？token 数量过多和过少分别有什么后果？

**Q8.** 视觉 token 与文本 token 在序列中如何排列？放在文本前面、后面、交错放置分别有什么影响？

**Q9.** Qwen-VL 的视觉位置编码与 LLaVA 有何不同？它如何支持不同分辨率？

**Q10.** Chameleon / Omni Model 与“视觉编码器 + projector + LLM”拼接路线在架构哲学上有何根本不同？

**Q11.** 多模态训练中，视觉编码器与 LLM 的学习率通常如何设置？为什么不能相同？

**Q12.** 视觉指令微调数据从何而来？合成多模态数据的质量如何保证？

**Q13.** 多模态模型中的“幻觉”（描述图片中不存在的内容）如何检测和缓解？

**Q14.** 图片分辨率很低时视觉编码器应如何处理？直接 resize 会丢失什么？

**Q15.** 视频理解中帧采样策略如何影响模型能力？时间位置编码需要特殊设计吗？

**Q16.** CLIP 的对比学习为什么需要大 batch？batch 内负样本的质量如何影响表征？

**Q17.** 多模态评估中，VQA accuracy 与生成式评估（CIDEr）分别适合什么场景？

**Q18.** 视觉编码器输出维度与 LLM hidden dim 不匹配时，projector 除了维度对齐还承担什么角色？

**Q19.** 只解冻 projector 而冻结视觉编码器与 LLM 时，模型能学到什么、学不到什么？

**Q20.** 多模态模型如何处理图中文字（OCR）？视觉编码器本身能识别文字吗，还是需要额外模块？

**Q21.** 交错图文数据的训练目标与纯图文对训练有何不同？

**Q22.** 音频、视频、图像统一到同一个 LLM 时，各自的 tokenizer 设计有什么共性挑战？

**Q23.** 多模态模型中的“模态遗忘”是什么？加入视觉训练后语言能力下降如何缓解？

**Q24.** CLIP 的 zero-shot 与 linear probe 分别评估了什么？两者差距说明了什么？

**Q25.** 视觉编码器的感受野与图片分辨率之间是什么关系？ViT 的 patch size 如何影响细粒度理解？

**Q26.** 为文档理解任务设计多模态架构，你会选高分辨率 ViT + 更多 visual token，还是先 OCR 再走纯文本？

**Q27.** 多模态 RLHF 的奖励如何设计？视觉忠实度与文本有用性如何联合优化？

**Q28.** Chameleon 类“原生多模态”模型在训练稳定性上面临什么额外挑战？

**Q29.** 多模态模型的推理延迟主要花在哪里？视觉编码、projector、LLM prefill 各占多少？

**Q30.** 从 CLIP 到 LLaVA 到 Omni，多模态架构的演进主线是什么？每个阶段解决了前一个阶段的什么根本局限？


### 新第 15 章 · 扩展内容【30 题】

**Q1.** 从系统角度重新审视推理：TTFT 与 TPOT 分别受哪些因素支配？在线上服务中如何联合优化？

**Q2.** 用户对首字延迟极敏感时，应优先调整 serving 系统的哪些参数？对吞吐更敏感时又如何调整？

**Q3.** LeCun 对 LLM 路线的核心批评是什么？他认为“下一 token 预测”范式的根本局限在哪里？

**Q4.** 你如何看待“世界模型”与 LLM 的关系？LLM 是否已经隐式学到了世界模型？

**Q5.** 推理时间扩展（test-time compute，长思维链）是否改变了 Scaling Law 的形态？

**Q6.** 推理阶段可以无限延长思维链时，模型能力的上限由什么决定？

**Q7.** Agent 范式与纯 LLM 推理的核心区别是什么？工具使用、记忆、规划分别给模型增加了什么？

**Q8.** RAG 与长上下文是否是替代关系？各自适合什么知识更新频率与任务类型？

**Q9.** 多模态、Agent、推理时间扩展三条路线，哪一条最有可能突破当前 LLM 的瓶颈？为什么？

**Q10.** 从“训练最优”到“推理最优”的范式转移，对模型架构设计会带来什么反向影响？

**Q11.** 预测未来 3 年 LLM 架构最可能的一个变化，你会选什么？依据是什么？

**Q12.** 稀疏化（MoE、结构化剪枝、动态计算）与稠密化两条路线，你更看好哪一条？为什么？

**Q13.** LLM 的“记忆”应如何实现？参数化记忆、外部检索、KV 持久化各自的优劣是什么？

**Q14.** 推理中的“自我验证”为什么困难？模型能否可靠地判断自己答案的对错？

**Q15.** o1/R1 类长推理模型对 serving 基础设施提出了什么新要求？

**Q16.** 推理成本持续下降后，哪些当前“不经济”的应用会变得可行？举一个具体例子。

**Q17.** LeCun 提出的 JEPA 路线与 LLM 路线的本质分歧是什么？JEPA 缺少什么才没有成为主流？

**Q18.** Agent 的“规划”能力能否完全由 LLM 本身涌现，还是必须依赖外部搜索/规划器？

**Q19.** 多轮 Agent 任务中错误累积如何缓解？自我纠错在什么条件下有效？

**Q20.** 从“下一个 token 预测”到“下一个概念预测”，是否是一个可行方向？技术障碍在哪里？

**Q21.** 推理时间扩展的边际收益何时递减？如何判断一个任务是否值得用长思维链？

**Q22.** 模型参数不再增长时，仅靠推理时间扩展与数据质量提升，能否继续获得显著能力进步？

**Q23.** LLM 的“可解释性”在推理模型上变得更难还是更容易？思维链是否忠实反映了真实计算？

**Q24.** 开源模型与闭源模型的差距在扩大还是缩小？从训练流程、数据、算力三个维度分析。

**Q25.** 端侧 LLM 的瓶颈是什么？量化、蒸馏、MoE 哪条路线最有可能让 7B 模型在手机上流畅运行？

**Q26.** 设计一段“LLM 全栈能力”的 2 分钟面试自述，你会用《diy-llm》重排后的 15 章地图如何组织？

**Q27.** 同一个模型在不同 serving 框架下吞吐可以差数倍，框架的哪些设计决策影响最大？

**Q28.** 从“模型中心”到“数据中心”再到“推理中心”，LLM 工程的范式正在如何迁移？

**Q29.** 你认为当前 LLM 最被高估的能力与最被低估的能力分别是什么？给出判断依据。

**Q30.** 面试官问“你如何看待 LLM 的未来”，你会如何组织一个既有技术深度又有个人判断的回答？

