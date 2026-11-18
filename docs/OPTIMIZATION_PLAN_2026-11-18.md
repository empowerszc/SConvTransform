# 单线程追平 PyTorch：现状分析、方向与实施计划

- 日期：**2026-11-18**
- 目标（本阶段）：**优先级 (a)** —— 让 MLIR/ConvBench 交付路径的单线程性能达到或超过
  PyTorch `conv2d`；多核只在一个 NUMA 节点内推进，**线程数不超过 38**，暂不涉及跨 NUMA。
- 前置状态：M1 快打包、M2 微内核放大、A 形状自适应 mK 均已交付（`a4c80b5`）。
- 相关文档：`OPTIMIZATION_DESIGN_AND_RESULTS.md`、`FEASIBILITY_VECTORIZED_PACK_MULTICORE.md`、
  `PERFORMANCE_VS_PYTORCH.md`。

> **频率口径（2026-11-18 晚补记）**
> 白天这台机器实际运行在 **1.55 GHz**（`scaling_cur_freq` 实测），因此上午写下的
> 绝对 ms/GFLOPS 互相不可比：其中 MLIR 侧数字是 1.55 GHz 测的，而 torch 列沿用的是
> 更早的 2.0 GHz 数据，**两者不同频**，附录 C 的比值被系统性压低了约 1.2×。
> 20:30 起 CPU 固定回 **2.0 GHz**（userspace governor，1999998 kHz）。
> 本文 §8、§9 的全部数字都是 **2.0 GHz 重测**；1.55 GHz 时代的数字只保留在 git 历史里，
> 不再作为结论使用。

---

## 1. 结论摘要（2.0 GHz 重测）

1. **单线程已经全面达到/超过 PyTorch t1**：MLIR/ConvBench 交付路径 11 个 custom 形状
   **geomean 1.57×，10/11 ≥ 1.0×**（唯一落后的 custom_7 为 0.83×）。
   文档正文里 “1.5× 全面领先” 说的是 `native_impl`（C++ oracle），不是 MLIR 路径；
   MLIR 路径从 0.66× 追到 1.57× 靠的是 A（快速边界拷贝）+ C（宽 tile 下 `Nc`
   自适应）+ G（窗口 tile 160/80），见 §8。
2. **最大的剩余开销不是 index 计算**：分解显示 custom_0（`[64,32]`、t1、改动前 85 ms）
   中 **≈24 ms 是函数边界的 tensor→memref 拷贝**、4.6 ms 是 benchmark 输出清零，
   真正的循环/index 只有 ≈1 ms。旧文档 “index ≈33 ms” 的归因已更正（§2.1）。
3. **多核（M3）已从原型做成正式实现**：transform 给窗口 tile 循环打 `sconv.parallel`，
   `sconv-opt -sconv-parallelize -sconv-lower-parallel` 一次产出可执行的 OpenMP 版本，
   **11/11 正确（t8、t32 各验一遍）**，2.0 GHz 单节点实测 **geomean 2.36×**，
   custom_1/custom_4 达 3.1×、最高 338 GFLOPS（§9）。
   **t16 与 t32 基本持平**，说明当前瓶颈已经从核数变成「并行宽度只有 5–20」。
4. **M4 的方向是“加宽”而不是“加核”**：窗口 tile 循环的 trip count 只有 5–20，
   t>16 时任务不够分。下一步把 filter 打包提出 filter-tile 循环、让窗口组循环也能入链，
   或按线程数下调 `Nwin` 换取任务数（§9.6）。
5. **未做的两刀仍然有效但收益变小**：E（filter 打包去冗余、输入打包优化）预期 -2~4 ms；
   B（边界拷贝归零）在快 copy 之后只剩 ~2 ms，主要价值是把串行项从多核里拿掉。

---

## 2. 测量方法与新分解（custom_0，t1，`[64,32]`，改动前，1.55 GHz 口径）

方法：在已变换/已 lower 的 IR 上做差分（不靠反推）：

- 删除 `sgemm_blas_kernel` / `pack_input_tile` 调用，称剩余时间；
- 删除 filter packing generic；
- 把边界 `memref.copy` 复制 3 份，斜率即单次拷贝成本；
- 删除 benchmark 计时循环里的 `linalg.fill` 做对照。

| 组成 | ms/iter | 占比 | 依据 |
|---|--:|--:|---|
| GEMM（kblas，m64/n32/k54 × 19,200 次） | ≈43 | 68% | 删调用差分 |
| 函数边界输入/输出拷贝 | ≈24（通用 copy） | 28% | 复制 3 份，单次 ≈12 ms，≈0.45 GB/s |
| 输入打包 | ≈7.7 | 12% | 删 `pack_input_tile` |
| filter 打包（仍是标量 gather） | ≈3.7–4.7 | 6% | 删 generic 差分 |
| 循环/index/分配 | ≈1 | ~2% | 上述全部删除后的残差 |
| harness 输出清零 | **≈0.3**（旧记 4.6 为误测） | ~1% | 删 `linalg.fill`；1.2 MB/iter 只应 ≈0.1–0.2 ms |

> 注：各行相加略大于总时，是不同差分批次与 ±4–7% 噪声所致；比例关系稳定。
> 旧结论 “每迭代 index ≈33 ms” 实际是 **copy + fill + 少量循环**，不是地址算术。

每秒拷贝量只有 ≈0.45 GB/s，是因为函数边界 memref 带**动态 stride**，
`memrefCopy` 退化为逐元素循环——与 M1 之前的标量打包同一类问题。

### 2.1 差分法的两个陷阱（2026-11-18 补记）

用「删调用再相减」必须确认删掉的 op 的产品**没有消费者**，否则结论会自相矛盾：

1. **DCE**：删掉 `sgemm_blas_kernel` 后，只被它使用的 filter packing generic 会被
   死代码消除，于是「删 GEMM」实际删的是「GEMM + filter 打包」。
2. **未初始化内存**：删掉 `pack_input_tile` 或边界 copy 后，GEMM 读到 malloc 出来的
   随机位，里面常有**非规格化数**；在这台机器上会让 SGEMM 慢 5–6×（实测
   custom_0「删 copy」变体 39 → 257 ms/iter），完全不可用。

结论：**能删的只有输出端**（GEMM 的 C、fill），输入端要用「复制一份取斜率」或
「换成写零的 stub」来测。本文后续数字都遵守这条。

---

## 3. 方案比较（本阶段）

### 方案 A（采用）：运行时快速 `memrefCopy`

- 做法：在 `scripts/convbench/runtime/` 提供行分段实现的 `memrefCopy`，优先于
  `libmlir_runner_utils.so` 加载；内层连续维用 `memmove`，外层用里程计推进。
- 实测：custom_0 **50.9 → 67.6 GFLOPS**；与标准实现输出**逐元素完全一致**
  （custom_0/custom_7 `maxdiff = 0`）。
- 优点：改动小、语义不变、对所有 `memref.copy`（含 self-copy）生效；
  对真实集成同样有效（动态 stride 是常见 ABI）。
- 缺点：只是“把不可避免的搬运做快”，没有消除搬运；仍是 28%→~3% 的优化而非归零。

### 方案 B（后续）：消除边界拷贝

- 做法：让 tensor→memref 边界保留**连续性**（identity layout）或改用
  destination-passing / memref ABI，使 `collapse_shape`/累积写可以原地进行。
- 预期：再省 ~1.9 ms（快 copy 后剩余），并把“算法时间”和“ABI 时间”分离；
  对 M3 很关键（去掉串行项）。
- 成本/风险：中高。`function_boundary_type_conversion=IdentityLayoutMap` 在本版本
  会触发 bufferization 崩溃（已复现），需要换实现路径（自定义 pass 或 ABI 约定）。

### 方案 C（下一步验证）：Nc / GEMM 形状调优

- 现状：`[64,32]` 下 L1 约束把 `Nc` 压到 6 → `k=54`、19,200 次调用；k 太短，
  每次调用摊销的固定开销高。
- 假设：放宽到 `Nc=12/24`（A/B tile 进 L2 而非 L1），k=108/216，调用数减半/减 4 倍，
  GEMM 有望从 43 ms 降到 30 ms 量级。
- 成本：中。需要把 `Nc` 变成可覆盖/可自动选择的参数（类似 `mK_info` 的第三槽位），
  并用 11 个形状验证不回退。

### 方案 D（后续）：缓存分块 / 循环序

- 目标：让 A/C 留在 L2，GEMM 43 → ~28 ms。成本高、风险高，放在 C 之后。

### 方案 E（小修，随时可做）

- filter 打包快路径：-3.7 ms；
- 输入打包继续优化（Nwin=64 的 256 B 段）：7.7 → 3–4 ms；
- 去掉/内化 harness fill：-4.6 ms（需要明确“算法时间 vs benchmark 时间”口径）。

### 方案 F（阶段 2）：M3 单 NUMA 多核

- 约束：**≤38 线程**，只在一个 NUMA 节点内；不做跨 NUMA。
- 前提：方案 A/B 先把串行项降到 <2 ms，否则 t32 上限只有 ~150 GF。
- 预期：并行部分 56 ms，8–16× 现实加速 → custom_0 4–7 ms / 0.6–1.1 TF；
  验收线仍按 800 GF、≥0.5× `native_impl` t32、11/11 正确。

---

## 4. 选择与理由

1. **先做 A**：它是已验证的 +33%，语义零风险，且能把“copy 到底值多少”从假设变成
   已落地收益；同时保留 B 作为真正的归零方案。
2. **再做 C**：快 copy 后 GEMM 占 68%，且 `Nc=6` 的结构性问题明确、可用小实验快速证伪；
   若成立，收益（~10 ms）大于 E 的所有小修之和。
3. **E 穿插做**：filter/输入打包是低风险、可独立验收的 -5~8 ms。
4. **B/D 视 C 的结果**：如果 C 把 GEMM 拉到 30 ms 以内，单线程已接近 PyTorch，
   再决定是否投入 B（归零）或 D（缓存分块）。
5. **F 只在 A/B 完成后启动**，避免把串行开销带进多核。

---

## 5. 里程碑与验收

| 阶段 | 交付 | 验收（t1，`numactl` node0） | 状态 |
|---|---|---|---|
| **A** | 快速 `memrefCopy` 运行时 + 接入 ConvBench | custom_0 **≥65 GF**；11/11 正确；`test_fastpack.py --stage all`、`test_mk_auto.py` PASS | ✅ 67.6 GF；逐元素一致；门禁 PASS |
| **C** | `Nc` 可覆盖/自动选择 | custom_0 **≤55 ms**；11/11 正确；任一形状回退 >5% 即回滚 | ✅ 40 ms；11/11；无回退 |
| **G（新增）** | 窗口 tile 候选扩到 160/128/80 + `Nc` 规则 v2 | 11/11；geomean ≥1.2×；单形状回退 ≤5% | ✅ geomean **1.57×**（2 GHz 重测）；10/11 ≥1.0× |
| **E（下一步）** | filter 打包去冗余 / 输入打包优化 | custom_0 再 **-2~4 ms**；11/11 | ⏳ |
| **B（按需）** | 消除边界拷贝 | 串行 copy <2 ms；算法口径与 ABI 口径分离 | 未启动（identity layout 会崩） |
| **D（可选）** | 缓存分块 / 余数 peeling（c7） | c7 ≥1.0×；无回退 | ⏳ |
| **F/M3** | 单 NUMA 多核，≤38 线程 | t16/t32 ≥800 GF、11/11 | 🟡 正式实现已落地：geomean **2.36×**、最高 338 GF、11/11（t8/t32）；未达 800 GF，瓶颈是并行宽度（§9） |
| **M4（下一步）** | 加宽并行（归约环内移 / 窗口组入链 / 多核专用 `Nwin`） | t16 ≥3.5×、≥4 个形状 ≥600 GF | ⏳ 见 §9.6 |

**硬性约束**：所有多核实验线程数 ≤38，且固定在一个 NUMA 节点；t≥35 的慢模式只允许在
本节点内做对照，不作为交付数字（该机器 t36+ 的异常见设计文档 §6.1）。

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 覆盖 `memrefCopy` 依赖加载顺序 | 换 harness 后失效 | 在 `get_libs` 中显式置顶；测试里断言 fast copy 生效 |
| 快速 copy 的正确性（stride/offset/别名） | 数值错误 | 逐元素对比 + 11/11；非单位内层 stride 走回退路径 |
| `Nc` 放宽后 L2/L1 抖动 | 反而变慢 | 按形状枚举 6/12/24 + min-of-3，形状回退即回滚 |
| 去掉 fill 改变语义 | 正确性/口径争议 | 只作为“算法时间”报告，不替换默认计时；正确性仍按原语义 |
| 多核任务粒度/超订 | t32 不升反降 | 外层并行 + kblas 单线程；任务数 ≥ 4×线程数；≤38 线程 |

---

## 7. 明确不做

- 不做跨 NUMA、不使用 >38 线程的多核方案。
- 不继续在 `native_impl` 上做打包/多核微调（已到 roofline，且它只是 oracle）。
- 不在本轮扩大 SME 覆盖（已有结论：c0/c2/c5/c7 不划算）。
- 不为了“单线程数字”改变数值语义或跳过必要初始化；口径差异只做并列报告。

---

## 8. 执行结果（2026-11-18 实测）

### 8.1 三个改动与效果

> 下表的绝对 ms 是 **1.55 GHz** 时代测的，只能用来看**相对变化**；
> 2.0 GHz 下的最终值是 custom_0 **33.3 ms / 127.7 GF**（§8.2）。

| 改动 | 内容 | custom_0 | 说明 |
|---|---|--:|---|
| **A** 快速 `memrefCopy` | `scripts/convbench/runtime/fast_memref_copy.c`，`get_libs` 置顶加载 | 63.4 → 63.1 ms（GF 50.9 → 67.6） | 边界拷贝 24 ms → ~2 ms，结果逐元素一致 |
| **C** `Nc` 自动 | `mK_info` 第三槽 + `computeTileC` 广 tile 规则 | 63 → 40 ms | `Nc=6`（k=54）→ `Nc=32`（k=288） |
| **G** 窗口 tile 放大 | `selectAutoNwin` 候选 `{160,128,80,64,32,16}`；`Nc` 规则 v2 | 40 → 41 ms（持平） | 弱形状大改：c6 525→496、c5 79→56、c3 284→265 |

**设计哲学（为什么是这两刀）**：

1. **先把「ABI 时间」从「算法时间」里拆出来**。tensor→memref 边界的动态 stride 拷贝
   本质是 ABI 的代价，不是卷积算法的一部分。它的 roofline 是 memcpy 带宽（~10 GB/s），
   而通用 `memrefCopy` 只跑到 0.45 GB/s——差 20×，所以先修它，收益确定且零语义风险。
2. **再让每个 GEMM 调用「干满活」**。tile 是 `[Nwin, Nf]`，`k = Nc·Fh·Fw`。宽 tile 把
   `Nwin` 拉大后，L1 规则算出的 `Nc=6` 让 `k` 只有 54，每次调用摊销的固定开销过高。
   把 `Nc` 的约束从 L1 换到「A/B tile 仍能留在 L2」，k 变成 144–288，调用数降一个量级。
3. **窗口 tile 再大一档**是同一逻辑的延伸：`Nwin` 从 64 到 160，输入打包次数和
   GEMM 调用数同比减少，而打包写放大（im2col 的 9×）不变。
   实测 160 在 1600/6400/25600 三种空间尺寸上都最好（400 落到 80）。

### 8.2 11 个形状：MLIR 单线程 vs PyTorch t1（2.0 GHz）

`numactl --cpunodebind=0 --membind=0`，单线程（已验证 `OMP_NUM_THREADS=1` 与默认同速，
kblas 对这些尺寸本来就不并行），payload runs=5/warmup=3，best-of-2。

| 形状 | GFLOP | MLIR ms | MLIR GF | torch GF | 比值 |
|---|--:|--:|--:|--:|--:|
| custom_0 | 4.25 | 33.3 | 127.7 | 87.7 | **1.46×** |
| custom_1 | 67.95 | 660.9 | 102.8 | 46.5 | **2.21×** |
| custom_2 | 4.25 | 37.5 | 113.2 | 75.9 | **1.49×** |
| custom_3 | 16.99 | 218.8 | 77.6 | 38.5 | **2.02×** |
| custom_4 | 67.95 | 653.0 | 104.1 | 73.3 | **1.42×** |
| custom_5 | 4.25 | 47.4 | 89.5 | 62.3 | **1.44×** |
| custom_6 | 33.97 | 412.2 | 82.4 | 58.0 | **1.42×** |
| custom_7 | 1.06 | 15.1 | 70.5 | 84.4 | 0.83× |
| custom_8 | 16.99 | 168.4 | 100.9 | 65.9 | **1.53×** |
| custom_9 | 16.99 | 134.0 | 126.8 | 72.9 | **1.74×** |
| custom_10 | 8.49 | 76.0 | 111.8 | 50.0 | **2.24×** |

- **geomean 1.57×，10/11 ≥1.0×**（改动前 0.66×、0/11）。
  1.55 GHz 时代这张表写的是 1.35×，差异主要来自**同频重测**：MLIR 侧是 1.55 GHz 数据、
  torch 侧是更早的 2.0 GHz 数据，不同频把比值压低了；改动本身没有变化。
- 机器单核 SVE 峰值 ≈128 GF（2 GHz × 2×512-bit FMA），custom_0/custom_9 已到 99–100%、
  custom_10 87%。**超过 SVE 口径的部分只能来自 KBLAS 内部对矩阵引擎（SME）的使用**，
  所以“单线程还有多少 SVE 余量”这个问题已经没有意义——剩下的空间在固定开销和
  多核（§9）。
- **正确性**：11/11 PASS（与参考实现 maxdiff 为整数级舍入差，与改动前同量级）。

### 8.3 仍然落后的 custom_7（0.83×）——结构性原因

- `Ohw=400`，`fn=192`，总计算量只有 1.06 GFLOP。tile 必须整除 400，候选只剩
  `{80,100,200,400}`；`Nwin=80` 时每次 GEMM 是 `m80/n32/k288`。
- 720 次调用，每次 ≈21 µs（2.0 GHz 下 15.1 ms / 720），有效 70 GF；而 torch 在这里是
  **一个**大 GEMM（`m≈1600, n=192, k=1728`），效率 84 GF。
- 已试过的方向：`Nf` 64/96、`Nc` 48/64/96、`Nwin` 100/200/400 —— 最好仍是
  `Nwin=80, Nc=48`（1.55 GHz 下 16.3 ms，2.0 GHz 下 ≈15.1 ms），没有质变。
  `Nwin=400` 会因 A/B/C 合计 ≈730 KB 撞上 768 KB 的 L2 而崩到 1.3 s。
- 真正要赢它需要**余数 peeling**（允许 `Nwin=128`，最后一块 16），而不是要求整除；
  这要改 tiling/packing 的余数处理，属于 D 类改动，收益 ≈3 ms / 0.03 geomean，
  优先级低于多核。

### 8.4 下一步（按收益/成本排序）

1. **M4 加宽并行**：t16→t32 已经不再变快（§9.3），瓶颈是窗口 tile 循环只有 5–20 个
   迭代。把 filter 打包提到 filter-tile 循环外、让窗口组循环也能入链，可把宽度提到
   40–640；这是目前最大的一块，做法与风险见 §9.6。
2. **E1 filter 打包去冗余**：现在 filter pack 在 batch 循环内，4 个 batch 重复打包
   同一份 filter（custom_0：24 次/iter × 221 KB ≈ 5.3 MB）。提到 batch 外可省 3/4，
   对小形状（c7）≈0.5 ms，对大形状 2–4 ms；顺带也是 M4 的前置条件。
3. **E2 输入打包**：`pack_input_tile` 仍是 9× 写放大，但在 10 GB/s 量级已接近带宽
   roofline；可试「按行分段 + 复用行缓冲」减少重复读，预期 -20~30%。
4. **B 边界拷贝归零**：快 copy 后只剩 ~2 ms，单线程收益小；但它是多核里的串行项，
   M4 之后权重会上升。
5. **D 余数 peeling（c7）**：0.83 → 1.0× 需要，收益 ≈0.03 geomean，最低优先级。

### 8.5 顺带测到的两个小结论（1.55 GHz 口径，只看相对量）

- **GEMM 后的 `memref.copy %x, %x` 是自拷贝**（源和目标是同一个 SSA 值），
  每个微内核一次。删掉它 custom_0 只从 39.0 → 39.1 ms —— 在噪声内，
  不值得为它加 pass；记录在此避免以后重复怀疑。
- **分配器抖动 ≈5%**：`mallopt(M_MMAP_THRESHOLD=512MB, M_TRIM_THRESHOLD=-1)`
  让 custom_0 从 106.7 → 112.3 GFLOPS（缺页 72k → 20k）。这是每次调用
  重新 mmap 5 MB 级 buffer 的代价，PyTorch 用 caching allocator 天然规避。
  **未默认启用**（属于 harness/分配器口径，不进交付数字）；如果要报“生产口径”，
  应连同 buffer 复用一起做，而不是只调 allocator。

---

## 9. M3 多核正式实现（2026-11-18，单 NUMA，≤32 线程实测）

### 9.1 设计理念：把「并行」留在 IR 里，而不是藏进运行时

原型阶段验证过三种做法（`FEASIBILITY_VECTORIZED_PACK_MULTICORE.md` §6.1）：
在 transform 里直接生成 `omp.parallel`、用 `tile_to_parallel`、或把整段循环塞进一个
C 线程池函数。**最终选第一条路线**：transform 只给循环打属性，独立的 pass 负责
`scf.for → scf.parallel` 与 scratch 私化，再由 `sconv-opt` 内置的 lowering 驱动
`--convert-scf-to-openmp`。

理由：

1. **并行维度是 schedule 的属性**，只有 transform 知道哪个循环是归约维、哪个写不相交的
   输出。放到运行时黑盒里，等于把这个知识丢掉（§9.2 的错误正是这样产生的）。
2. **可检查、可回退**：串行路径完全不变（不加 `-sconv-parallelize` 就是原 IR），
   出问题能二分到「属性 → 折叠 → 私化 → OpenMP lowering」中的一层。
3. **不引入第二种线程池**：libomp 本来就被 KBLAS 使用，复用它的线程池比自建池更省，
   也让 `OMP_NUM_THREADS` / `numactl` 这些标准旋钮继续有效。
4. **代价**是必须处理 MLIR 的低层细节（§9.5 的三个坑），并且并行度上限被循环结构
   硬约束（§9.4）。

流水线（全部已在仓库里）：

```bash
# 一步产出可被 mlir-cpu-runner 执行的 LLVM IR（内部含 convert-scf-to-openmp）
build/bin/sconv-opt -transform=test/lowering/sconv-blas-fastp.mlir \
    -sconv-parallelize -sconv-max-threads=64 -sconv-lower-parallel \
    conv.mlir -o conv.llvm.mlir

OMP_NUM_THREADS=16 numactl --cpunodebind=0 --membind=0 \
    mlir-cpu-runner conv.llvm.mlir -e main -entry-point-result=void \
    -shared-libs=...,libomp.so
```

- `lib/SConv.cpp::applyTileTo` 给**最内层窗口 tile 循环**打 `sconv.parallel`；
- `lib/LowerToPack.cpp` 给输入打包的 `memref.alloc` 打 `sconv.parallel_scratch`；
- `lib/Parallelize.cpp`：
  - `sconv-parallelize`：把「直接嵌套的已标记 `scf.for` 链」折成多维 `scf.parallel`
    （MLIR 会把它折成一条 `__kmpc_for_static_init` 的一维循环），再把 scratch 扩成
    `[maxThreads][...]`、在并行区入口用 `omp_get_thread_num()` 取自己的槽；
  - `sconv-erase-alloca-scope`：删掉 `convert-scf-to-openmp` 无条件包上的空
    `memref.alloca_scope`（必须在 `--convert-scf-to-cf` 之前）；
  - `registerSConvPasses()`：让 `sconv-opt -sconv-lower-parallel` 直接串完整 lowering。

### 9.2 正确性的根因：**归约维不能并行**

原型为了拿宽度，把 `loopOps[0]`（batch）和 `loopOps[1]`（输入通道块 `Nc`）也标了。
结果是 **8/11 形状错**，max_diff 1e6–1e8：

- `Nc` 是微内核 GEMM 的**归约维**：同一个输出 tile 被 6 个 `Nc` 迭代**累加**。
  并行化它等于让多个线程不加锁地 `C += ...`，必然错；
- 而且 C 的地址还带 `Nc` 偏移，错的方式是「随机丢一部分累加」，用差分很难定位。

循环嵌套的真实语义（11 个形状结构一致，下表以 custom_0 为例）：

| 层 | 语义 | 迭代数（c0） | 能并行吗 | 原因 |
|---|---|--:|---|---|
| [0] | batch | 4 | 否（在当前结构下） | 上面还有 `Nc` 归约环，无法与窗口 tile 直接成链 |
| [1] | 输入通道块 `Nc` | 6 | **否** | GEMM 的归约维，迭代累加同一个 C |
| [2] | 窗口组 | 1 | 否 | 体内含 filter 打包（写共享 scratch） |
| [3] | filter tile | 1 | 否 | 同上；且 body 不只有内层循环 |
| [4] | 窗口 tile | 10 | **是** | 写不相交的 C 列，body 是一个完整工作项 |
| [5] | filter 子块 | 6 | 否 | 复用上一层刚打包好的 A，拆开会让打包次数乘 6 |

所以正式实现**只标 [4]**。用 `-sconv-parallelize` 后的 IR 可以看到并行区的位置：

```mlir
scf.for %n   = 0 to 4   step 1        // batch
  scf.for %nc = 0 to 192 step 32      // Nc（归约，串行）
    scf.for %g  = 0 to 1600 step 1600 // 窗口组
      scf.for %f  = 0 to 192 step 192 // filter tile（filter 打包在这里）
        scf.parallel (%w) = 0 to 1600 step 160 {   // ← 唯一并行维
          %tid = func.call @omp_get_thread_num()
          ... 输入打包（写 tid 自己的槽） + 微内核 ...
        }
```

注意并行区被**进入 4×6=24 次**（batch × Nc），每次只有 10 个迭代。这解释了两件事：
t1 的固定开销（24 次 fork/join），以及为什么加线程数到 16 以上没有收益（§9.4）。

**验证**：t8 与 t32 各跑一遍 11/11，`max_diff` 与串行同量级
（0–4e4，属浮点结合序噪声；t32：c0 5e2、c1 4e4、c4 1e4、c5 0、c10 2e4）。

### 9.3 实测（2.0 GHz，node0，ms/iter，best-of-2）

| 形状 | 串行 | t1 | t4 | t8 | t16 | t32 | 最佳 | 最佳 GF | 并行宽度 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| custom_0 | 34.4 | 34.6 | 20.8 | 18.2 | **16.2** | 15.8 | 2.18× | 269 | 10 |
| custom_1 | 620.9 | 665.3 | 325.9 | 259.4 | **200.8** | 201.3 | 3.09× | 338 | 10 |
| custom_2 | 37.1 | 41.0 | 24.8 | 22.5 | 19.2 | **18.4** | 2.02× | 231 | 10 |
| custom_3 | 232.5 | 226.3 | 162.5 | **138.5** | 139.5 | 138.6 | 1.68× | 123 | 5 |
| custom_4 | 649.6 | 676.4 | 333.4 | 272.2 | **210.7** | 214.2 | 3.08× | 323 | 10 |
| custom_5 | 46.6 | 46.7 | 24.8 | 21.1 | 18.6 | **17.0** | 2.74× | 250 | 20 |
| custom_6 | 407.1 | 422.8 | 229.2 | 180.6 | 148.3 | **147.5** | 2.76× | 230 | 10 |
| custom_7 | 14.4 | 15.9 | 12.7 | 11.0 | **10.8** | 13.8 | 1.33× | 98 | 5 |
| custom_8 | 168.0 | 172.9 | 85.8 | 72.4 | **57.6** | 61.7 | 2.92× | 295 | 10 |
| custom_9 | 130.9 | 141.1 | 80.3 | 66.7 | 56.4 | **51.2** | 2.56× | 332 | 10 |
| custom_10 | 75.0 | 80.3 | 44.5 | 37.9 | **31.3** | 33.1 | 2.40× | 271 | 10 |

- **geomean 2.36×**（11 个形状，最佳档）。串行总时间 2.42 s → t16 0.91 s。
- **t16 与 t32 基本持平**：10 个形状在 ±10% 内，c9 t32 快 10%，而 c7（宽度只有 5）
  在 t32 反而慢 22%——线程远多于任务时的调度抖动。这不是 OpenMP 的问题，
  而是宽度只有 5–20：t≥16 时已经「一个迭代一个线程」，再加线程只是空转。
- **t1 比串行慢 ≈5%（0–10%）**：24 个并行区的 fork/join + 每线程槽的地址计算。
  这是多核模式的固定代价，单线程场景继续走串行 pipeline 即可。
- 最高 338 GF，**没有达到 800 GF 的验收线**。原因不是实现错误，而是并行宽度（§9.4）；
  800 GF 需要 M4 把宽度做到 ≥40（§9.6）。

### 9.4 并行宽度是当前唯一的硬约束

每个并行区的宽度 = 窗口 tile 循环的 trip count：

| 形状 | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 宽度 | 10 | 10 | 10 | 5 | 10 | 20 | 10 | 5 | 10 | 10 | 10 |

这个数由 `Ohw / Nwin` 决定（`Ohw` = 输出空间元素数，`Nwin` 是窗口 tile）。
单线程最优要求 `Nwin` 尽量大（160，为了 GEMM 的 `m` 和打包摊销），
而多核要求 `Nwin` 尽量小（为了任务数）——**这两个目标直接冲突**，是 M3 的核心矛盾。

用 Amdahl 估一下上限：串行项（filter 打包 ≈3.7 ms、边界拷贝 ≈2 ms、fill ≈0.3 ms、
循环框架）约占 10–15%；宽度 10 时理论上限 ≈3.5–4×，实测 2.2–3.1×。
差额来自打包带宽在多核上共享（`pack_input_tile` 是 9× 写放大、带宽受限），
以及每个并行区都要重新走一次 fork/join。

### 9.5 三个实现坑（都已解决，写在代码注释里）

1. **`memref.alloca_scope` 会锁死 lowering**。MLIR 19.1.7 的 `convert-scf-to-openmp`
   无条件把循环体包进 `memref.alloca_scope`；随后 `--convert-scf-to-cf` 把体展开成
   多块，verifier 报 “expects region #0 to have 0 or 1 blocks”。
   解法：新增 `sconv-erase-alloca-scope`，在 `--convert-scf-to-cf` **之前**删掉空 scope。
2. **`memref.reinterpret_cast` 的 offset 是相对 base buffer 的**。私化后，槽的起点在
   `memref.subview` 的 offset 上，而下游的 `reinterpret_cast` 会**丢掉**源 memref 的
   offset、从 base 重新算。更隐蔽的是：如果结果类型是静态 offset 0 的 identity layout，
   `extract_strided_metadata` 会把运行时 offset 直接折成 0。
   解法：pass 给每个下游 cast 的（线性化）offset 加上 `tid * slotElements`，
   同时把结果类型改成 `strided<[...], offset: ?>`，让元数据路径也拿到动态 offset。
3. **不能每迭代 `memref.alloc`**。原型用「每次迭代新分配」消除共享，结果 malloc/缺页
   让 t1 从 499 ms 涨到 820 ms。正式实现改成 `[64][Nc·Fh·Fw·Nwin]` 的 per-thread 槽
   （custom_0 一个槽 184 KB，64 槽合计 11.8 MB），整轮复用，`omp_get_thread_num()`
   索引一次。槽在整个并行区稳定，所以「打包 → 微内核」这一对总是看到同一块内存。

### 9.6 可行性评估：优点、缺点与 M4

**优点**

- 完全在编译器侧，交付物仍是 `sconv-opt` 一个命令；没有第二套线程池、没有运行时约定。
- 串行路径零改动、可随时回退；并行只在显式加 `-sconv-parallelize` 时生效。
- 与 `numactl` / `OMP_NUM_THREADS` 等标准工具兼容，单 RUMA 节点内的调度可控。
- 正确性边界清楚：只有「写不相交 + body 是完整工作项」的循环才能入链，
  pass 对不满足条件的链直接放弃并留 remark，不会生成半并行 IR。

**缺点/代价**

- **单线程回退 0–10%**：多核模式不适合当默认；需要一个「按线程数选模式」的入口。
- **内存放大 64×**（scratch 槽）：custom_0 11.8 MB/次。`-sconv-max-threads` 可以调小，
  但 Slot 数必须 ≥ 实际线程数，否则两个线程会共用一块槽（正确性直接坏掉）。
- **并行宽度被 schedule 锁死**（§9.4），当前 5–20，t>16 全部浪费。
- 依赖 OpenMP 运行时与 `omp_get_thread_num` 符号；非 OpenMP 环境需要走串行。
- 打包仍是带宽瓶颈：多核时 8 个线程同时做 9× 写放大的 im2col，带宽会先饱和。

**M4：把宽度做上去（按性价比排序）**

| 路线 | 做法 | 预期宽度 | 改动量 | 风险 |
|---|---|--:|---|---|
| **M4-a 归约环内移** | 交换 `Nc` 与窗口 tile 的循环序，让 batch 与窗口 tile 直接成链；`Nc` 在并行体内串行累加 | 40（=4×10，c0） | 中：改 `tileInterchange` + 校验数值一致 | 中：C 的访问序变化，需重测 cache 行为 |
| **M4-b 窗口组入链** | 把 filter 打包提到 filter-tile 循环外（E1 的副产品），让 [2][3][4] 成链 | 40–160（c2 4×10、c6 16×10） | 中：改 packing 插入点 + 允许 [2] body 只有内层循环 | 低-中：filter 打包会重复，需权衡 |
| **M4-c 多核专用 `Nwin`** | 用 `mK_info` 把 `Nwin` 降到 80/40（单线程会变慢） | 20–40 | 小：只改调参 | 低：GEMM 效率下降 10–20% |
| M4-d 每线程 C + 归约 | 并行 `Nc`，每线程私有 C，最后归约 | 60–120 | 大 | 高：多一次 C 读写，内存放大 |
| M4-e 运行时线程池 | 整段循环交给 C 池 | 任务数不限 | 大 | 高：等于放弃编译器并行 |

建议顺序：**M4-c（一周内可验证）→ M4-a → M4-b**。M4-c 能立刻告诉我们在宽度 20–40
时扩展性能到哪，从而判断 a/b 是否值得做；如果宽度 40 仍然到不了 600 GF，
问题就在串行项（B/E）而不是宽度，届时应先做 B。

验收（保持现约束：单 NUMA、≤32 线程报数）：

- 11/11 正确（t8、t32 各一遍）；
- t16 或 t32 geomean ≥ 2.5×（当前 2.36×；M4 目标 ≥3.5×）；
- 至少 4 个形状 ≥ 600 GF；
- 串行路径与 t1 不回退（t1 允许 ≤5% 于串行）。
