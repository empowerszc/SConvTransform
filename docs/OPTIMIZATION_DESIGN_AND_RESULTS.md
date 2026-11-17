# SConvTransform 优化：设计理念、实现地图与效果汇总

> 适用范围：本仓库在 aarch64（Kunpeng，SVE2 + SME）上的工程化优化工作。
> 性能数字的**权威出处**是 `docs/PERFORMANCE_VS_PYTORCH.md`（含完整 18 用例 × 6 线程档），
> 本文负责**把设计思路、代码位置、已生效/已否决的优化及其证据串成一条线**，不重复抄表。
>
> 最近更新：2026-11-17（新增 `Nc` 精调 + 调优计时稳健化；修正 ConvBench FLOPs 口径）

---

## 0. 一分钟版本

| 问题 | 答案 |
|---|---|
| 现在领先 PyTorch 吗？ | 是。18 用例、t1–t32 geometric-mean **1.51×–1.62×**，每个线程档 16–18/18 取胜。 |
| 提升来自哪里？ | 四处工程改动：`Bc` 并行、`(Nwin,Nf)` 自适配、SME shim 去固定开销、`Nc` 精调。 |
| 还有大空间吗？ | `native_impl` 已贴着自己的 roofline（打包 13 GB/s/core 是带宽上限，非代码上限）；**真正的大空间在 MLIR/ConvBench 管线**（单线程 + 标量打包）。 |
| 下一步建议 | 见 `docs/FEASIBILITY_VECTORIZED_PACK_MULTICORE.md`：先在 MLIR 侧做「打包向量化 + 微内核 tile 放大」，再做多核。 |

---

## 1. 两条实现路径：为什么同时存在

仓库里有**两条**把卷积变成「打包 + GEMM」的路径，职责完全不同，优化空间也不同。
不区分这两条，讨论「还有没有优化空间」必然得出矛盾结论。

### 1.1 编译器路径（MLIR，仓库本体）

```
lib/LowerToBLAS.cpp + lib/SConv.cpp
  linalg.conv_2d_nchw_fchw
    → transform.structured.sconv              ← 本仓库的贡献
        CSA 分析（Nc / K2 / K3 / IS·WS 调度）
        两级 tiling（scf::tileUsingSCF，串行 scf.for）
        packing（linalg.generic 逐元素搬运）
    → bufferize
    → transform.structured.lower.to_blas      ← 微内核替换为 func.call @sgemm_blas_kernel
    → mlir-opt lowering → mlir-cpu-runner
scripts/convbench/runtime/sgemm_blas_kernel.c ← 单线程 cblas_sgemm 包装
```

- 这是**论文/工程交付物**：验证 SConv 变换本身能否生成正确的优化循环。
- 现状（custom_0）：**M1+M2+形状自适应 mK 后 ≈87 ms/iter，≈49 GFLOPS**（单线程；
  M1 前 156.6 ms/27.1 GF）。
  注意：ConvBench 表格里打印的 6.78 GFLOPS 是旧口径（FLOPs 漏乘 batch，见 §6.2），已修正为 4×。
- 结构性问题：**微内核极窄**（`C[16,8] += A[216,16]^T B[216,8]`，每次 0.055 MFLOP，每轮卷积 **76,800 次调用**），
  打包原为逐元素标量搬运。stub 内核实测的分解是 **GEMM ≈41% / 标量打包 ≈33% / 循环+分配 ≈26%**
  （见可行性论证 §2.3），M1 已把打包换成运行时行分段拷贝并复用打包目标 buffer（见 §3.6）。

### 1.2 参考实现路径（`native_impl/`，性能探针）

- 同一套算法（CSA → 两级 tiling → pack → kblas sgemm → unpack），但用 C++ 手写 + OpenMP，
  用来回答「这套算法在目标机上**最多**能多快」。
- 现状（custom_0）：t1 **99.5 GFLOPS**，t32 **1.7–2.1 TFLOPS**（视测量批次）。
- 它的作用是**给出上界与瓶颈诊断**，不是交付物；因此可以放手做 OpenMP、线程池这种 MLIR 侧暂时不好做的事。

> 结论：`native_impl` 的优化已经接近极限（见 §4 的证伪清单），继续在它身上抠 3–10% 不划算；
> 想拿数量级收益，必须回到 §1.1 的管线。这也是可行性论证的出发点。

---

## 2. 设计理念（六条，都能对应到代码）

### 2.1 把「搬运」和「计算」显式分开，各自贴各自的 roofline

卷积被拆成三个物理阶段：`pack`（im2col 物化）→ `GEMM`（kblas 微内核）→ `unpack`（还原布局）。
好处是每段都有独立、可测量的 roofline：

| 阶段 | roofline | 目标机实测 | 结论 |
|---|---|---|---|
| pack | 单核存储带宽 | **~13 GB/s/core**（三种写法：标量 13.17 / `memcpy` 13.33 / 合并写 12.39，参考实现） | 已贴顶 |
| GEMM | 微内核峰值 | 0.7–2.3 TFLOPS（t32，18 用例；极小用例 `r18_7` 只有 0.13） | 取决于 K/Nf |
| unpack | 写带宽 | 5–13% 总时间（t1 实测） | 非瓶颈 |

这条理念直接解释了两个「看起来该优化、实际无用」的方向：
打包的 SIMD/`memcpy` 化、以及 `Nf` 继续放大（两者都在 §4 被证伪）。

### 2.2 复用优先于重算，且复用要**一次性打包、多轮消费**

- 滤波器 `Bc` 预打包为 `[ft][ic0][k][nf]`（`ldb = nf_str`），使每次 `sgemm` 的 B 操作数是
  连续切片，直接喂 `CblasTrans`；`Nf` 改变时才重建。
- 输入 tile 的打包**提升到 `oc` 循环之外**：一个 `[Nc, Fh, Fw, Nwin]` 打包块被 `Oc/Nf` 次 GEMM 复用。

### 2.3 并行任务粒度必须**跟着形状走**，不能写死

并行维度是 `batch × 空间窗tile`，任务数 = `N · ceil(Ohw/Nwin)`。
batch=1 的低分辨率用例（`r18_7`：`N=1, Ohw=49`）用 `Nwin=32` 时只有 **2 个任务**，32 核只能干瞪眼；
而单线程反而偏好大 `Nwin`（`r18_7` t1：nw32 31.4 GF > nw8 23.8 GF）。
所以 `(Nwin,Nf)` 只能**运行时实测决定**（§3.2）。

同一条理念在 2026-11 延伸到第三个旋钮 `Nc`：CSA 的纯 cache 代价模型给出的通道块，
在**并行饥饿**（任务数 < 线程数）形状上会偏离最优 5%–23%，因此对它做受限搜索（§3.5）。

### 2.4 修固定开销，先看「端到端口径」

`custom_9/10` 早期报的是 **kernel-only 上界**（输入已 pad+pack、输出未 unpad）。
换成端到端后，暴露了 shim 的三块固定开销（单线程 pad、每次 `calloc`、每次 `pthread_create/join`），
c10 t32 一度只有 0.39×。这类问题**只有端到端口径能发现**（§3.4）。

### 2.5 用「同二进制 A/B + 交错重复」判定收益，不做跨进程比较

本机是共享节点，t32 噪声约 ±4–7%；自动调优本身也会因为计时噪声选到不同配置。
判定标准因此固定为：

- 同一份代码，只切换目标机制（`#if` / 环境变量 / 两个二进制）；
- 单/多线程各 3–6 轮**交错**执行，取**中位数**；
- **未改动的对照用例**给出噪声底线（实测 −4%…−3% 属正常漂移）；
- 收益 < 噪声底线的改动一律视为「无收益」（§4 的 `std::copy` 就是这么被否决的）。

### 2.6 特殊硬件只喂给它擅长的形状（SME 不扩面）

`custom_9/10` 走手写 SME 直接卷积（`sme/`，`LowerToSME.cpp` 做静态代价筛选）。
扩到 `custom_0/2/5/7` 的实测是**全面更慢**（§4），因为 SME tile 利用率取决于
`OcP`/通道数能否铺满 16×16 tile，且固定开销摊不薄。**筛选器保持窄**是设计决策，不是遗漏。

---

## 3. 已实施优化与效果

> 所有 A/B 均为 clang 19.1.7、`numactl --cpunodebind=0 --membind=0`；
> 完整数据见 `docs/PERFORMANCE_VS_PYTORCH.md` §2 与附录 A。

### 3.1 `Bc` 滤波器预打包并行化（`native_impl/src/sconv.cpp:build_Bc`）

原为单线程，且每次调用重算（`r18_7` 有 2.36M 元素，约占单次耗时 44%）。
改为对互不相交的 `(ft, it)` 切片 `#pragma omp parallel for collapse(2)`。

A/B（GFLOPS，before → +并行）：

| 用例 | t1 | t8 | t32 |
|---|:--:|:--:|:--:|
| common_r18_7 | 0.94× | **1.80×** | **1.89×** |
| common_r18_14 | 0.98× | **1.96×** | **1.68×** |
| common_r18_28 | 0.97× | 1.23× | **1.68×** |
| custom_0/2/5 | 0.96–1.02× | 1.01–1.02× | 1.02–1.14× |

> t1 不变是**符合预期**的（`Bc` 仍串行），这本身是 A/B 有效性的一个对照。

### 3.2 `(Nwin, Nf)` 运行时自适应（`native_impl/src/sconv.cpp` 自动调优块）

以 `(N, Ic, Ohw, Oc, stride, Fh, Fw, threads)` 为键缓存；未命中时对
`Nwin ∈ {8,16,32,64} × Nf ∈ {16,32}` 做「预热 + 计时取最快」，再据此重建 `Bc`。
`--nwin 0 --nf 0` 走自适应；`SCONV_NO_AUTOTUNE=1` 回退 32/16。

A/B（+并行 vs +并行+自适应）：`r18_7` t32 **1.45×**、`r18_14` t32 **1.74×**、
`r50_ds28` t32 **1.59×**、`custom_0` t32 1.01×。

### 3.3 修复：自适应路径的冗余 `memset`

调优需要清零输出，但 `memset` 原先**每次调用**都执行（输出可达 20 MB ≈ 0.36 ms）。
改为仅缓存未命中时执行：`custom_5` t32 **1506 → 1766 GFLOPS**（对齐显式最优 1701）。

### 3.4 SME shim 端到端优化（`sme/runtime/sconv_sme_shim.c`）

三处固定开销 + 三项对策：

1. **持久化线程池**（`_Thread_local` 常驻 worker，mutex/cond 两阶段广播）——省每次 ~2 ms 的 create/join；
2. **缓冲复用**（`wpack`/`pin`/`pout` 按容量 `realloc`，仅增大时）——消除 `calloc` + 缺页；
3. **并行 pad/pack**（并入线程池阶段 A，切片互不相交）——c10 的 10 ms 单线程 pad → ~1 ms。

端到端 GFLOPS：`custom_9` t32 **1875 → 2550**、`custom_10` t32 **491 → 1257**、`common_r18_7` t32 **27 → 126**。
对 torch：`c9` t32 1.30×、`c10` t32 1.06×（此前 `c9` 0.95×、`c10` 0.39–0.41×）。

### 3.5 `Nc` 精调 + 调优计时稳健化（`native_impl/src/sconv.cpp`，2026-11）

CSA 的 `Nc` 来自 `halfHeuristic`：把可用 cache 切成两半后按 K 维容纳量取块
（`arch_info` 已按目标机真实值填：L1 32 KB、L2 768 KB、L3=0、line 128）。
它决定 GEMM 的 `K = Nc·Fh·Fw`，是纯 cache 视角的启发式，没有把「并行任务数」纳入考虑。
实测该启发式对**并行饥饿**形状明显偏离最优：

| 用例 | 默认 Nc | 最优 Nc | 增益 |
|---|--:|--:|--:|
| common_r18_28 | 0（默认） | 32 | +23% |
| custom_4 | 0（默认） | 64 | +6.5% |
| common_thr_n8 | 0（默认） | 32 强制 | **−43%**（回归） |

因此实现为**受约束的调优**：
- 阶段 2 在阶段 1 选出的 `(Nwin,Nf)` 上搜索 `Nc ∈ {默认,16,32,64}`；
- **仅对 `N × 窗tile < 线程数` 的形状**启用（非饥饿形状直接跳过，避免上表最后一行那种回归）；
- 每个候选测 5 次取最小，**快 ≥3% 才替换默认**（3% 门槛用于吸收噪声）。

同时把阶段 1 的采样从 min-of-2 提到 **min-of-5**（本机噪声 ±4%，min-of-2 会让同一形状
在不同进程里选出不同配置，实测 `r18_14` 波动 184–275 GFLOPS）。

效果（交错 6 轮中位数，t32，同二进制对比改前）：

| 用例 | 改前 | 改后 | 提升 |
|---|--:|--:|--:|
| common_r18_14 | 224 | **277** | **+24%** |
| common_r50_ds56 | 586 | **656** | **+12%** |
| common_r50_ds28 | 310 | **338** | **+9%** |
| 对照 `common_r18_56` / `custom_4` | — | — | −4% / −3%（噪声底线） |

全部用例对 naive 参考逐元素校验 **PASS**（任意 tiling 都不改变语义）。

### 3.6 MLIR 管线：快速输入打包 M1（`lib/LowerToPack.cpp` + `runtime/pack_input_tile.c`，2026-11）

目标：把 MLIR 路径里逐元素标量 gather 的 `linalg.generic` 输入打包换成
运行时行分段拷贝（与 `lower.to_blas` 同构）。

实现三件套：
1. **运行时 helper** `pack_input_tile()`：按 `(ic,fh,fw)` 把「同一输出行内连续」的
   窗口段整段 `memcpy`/向量化搬走，避免逐元素 `tensor.extract` + div/mod 地址计算。
2. **新 transform op** `transform.lower.to_pack`：匹配 `packing="input"` 的
   `linalg.generic`，用「把 `linalg.index` 归零后重算既有 affine 链」得到 tile 首元素
   偏移，再发 `llvm.call`。几何信息由 `sconv.pack_geom` 属性显式携带
   （`[Ow,Iw,strideH,strideW,dilH,dilW]`），不依赖从 affine map 反推。
3. **打包目标 buffer 上提复用**（`applyInputPacking`）：`tensor.empty` 形状循环不变，
   提到最外层循环外，使 one-shot bufferize 只分配一次（原为 3200 次/迭代）。

| 口径 | M1 前 | M1 后 |
|---|--:|--:|
| custom_0 单线程 | 156.6 ms | **115.7 ms（-26%）** |
| 其中打包相关（残差法） | ≈52.6 ms | ≈16 ms（helper 独立微基准 15.5 ms / 2.85 GB/s） |

- 收益的大头其实来自第 3 点（约 28 ms）：每迭代新建打包目标 buffer 会让每次写落到
  **冷页**。只换 helper、不hoist 时只省 17 ms。
- 未追到 13 GB/s 的原因：`Nwin=16` 每次只搬 64 B，且按 `(ic,fh,fw)` 重读输入行约 9×；
  继续优化是可选后续，收益有限（见可行性论证 §4 杠杆 1）。
- 验收：`scripts/convbench/test_fastpack.py --stage all`（IR / 逐元素正确性 / 性能）。
  `ir` 检查不再有标量 input-packing generic、出现 `pack_input_tile`；`correct` 对
  custom_0/custom_5 做 rtol/atol 1e-3 比对；`perf` 预算 **≤125 ms/iter**。

### 3.7 MLIR 管线：微内核 tile 放大 M2（`inputMultipackingOpt` + `lower.to_pack`，2026-11）

M2 本想把微内核从 `m16/n8/k216` 放大到更宽的 `n`（Nf 8→32）。实测发现两件事：

1. **`Nf>8` 必然切到 WS + multipacking**：只要 `mK_info=[16,Nf]` 里 Nf>8，CSA 就选
   WS，输入打包变成 multipacking（形状 `[1, Ti, Nc, Fh, Fw, Nwin]`）。M1 的 helper
   只认单打包，所以 `lower.to_pack` 必须先支持 multipacking。
2. **放大 tile 的收益受「内存带宽」限制，不是 GEMM 算力**：warm-cache 微基准里
   `m16/n32/k108` 只要 21 ms（201 GF），但 in-situ 的 GEMM 仍是 ~50–60 ms——
   打包后的 A、输出 C 都放不进 L2（768 KB），每轮都在 DRAM 上读写。

已实现：
- `inputMultipackingOpt` 也写 `sconv.pack_geom`，并把 multipacked scratch 上提到最外层复用；
- `lower.to_pack` 支持 `multipacking=true`，目标形状 `[1, Ti, Nc, Fh, Fw, Nwin]`；
- `pack_input_tile` 增加 `ni`（窗口 tile 数）：以 `t` 为外层、`w=t*Nwin+nwin` 复用同一
  行分段公式，目标块对每个 `t` 连续。

custom_0 实测（单线程，node0，ms/iter）：

| 配置 | 总时 | 说明 |
|---|--:|---|
| M1 `[16,8]` | 115.7 | IS，单打包 |
| Nf=32 `[16,32]` | 111.5 | WS，multipacking |
| Nf=64 `[16,64]` | 102.7 | |
| **`[64,32]`** | **≈84–87** | 最优 |
| `[64,64]` / `[80,32]` | 244 / 117 | 变差 |

`[64,32]` 分解：循环/index 地板 ≈33.5、打包 ≈8.8、GEMM ≈46（warm-cache 本可 ~25）。

> **结论（重要）**：`≤75 ms` 这个目标**没有达到**，且单纯调 tile 到不了：
> 残差是「每微内核迭代的 index 计算 ≈33 ms」+「内存带宽受限的 GEMM ≈46 ms」。
> 要再往下，需要 (a) 把每迭代的 `extract_strided_metadata`/`affine.apply` 提到循环外，
> 或 (b) 改循环序/分块让 A、C 留在 L2。这两项都不是 mK 调参能解决的。
>
> **且 `[64,32]` 不能全局写死**：它要求 `Nwin=64` 整除输出空间尺寸、`Nf=32` 整除
> 滤波器数，否则落入 edge 路径，实测 custom_5 从 229 → **6227 ms**、custom_7 从
> 36 → 216 ms。§3.8 用「最大整除因子」的按形状自动选择解决这个问题：默认快路径
> 现在传 `mK_info = [0, 0]`，规则形状自动选到 `[64,32]`，不规则形状自动退到能整除的
> 较窄 tile，11/11 形状均优于旧的 `[16,8]`。

### 3.8 MLIR 管线：形状自适应 mK（`mK_info = [0, 0]`，2026-11）

M2 的结论是「宽 tile 很快，但写死会翻车」。本节把它变成默认安全的行为，思路对齐
`native_impl` 的「按形状自适应」，但**不用运行时搜索**，只用一条可解释的整除规则。

**设计哲学：精确性优先，而不是拟合代价模型。**
- 宽 tile 的收益只在 tiling **精确**时成立。`Ohw % Nwin != 0` 或 `fn % Nf != 0` 时，
  管线走 edge/split 路径，额外开销远大于宽 tile 省下的微内核调用（custom_5
  `Nf=32` 对 `fn=48`：229 → 6227 ms，50× 级）。
- 所以这里要回答的不是「哪个 tile 平均最快」，而是「在当前形状上，哪个 tile 既宽
  又不会触发 edge 路径」。这是一个**约束满足**问题，最大整除因子就是它的最优解。
- 失败安全：找不到整除的宽候选项就退回历史默认（`Nwin=16, Nf=8`）。这两个值本来就是
  edge handler 的基准粒度；若连它们也不能整除，则与改动前一样走 edge 路径，不会更糟。
- 显式值仍然优先：自动调优只填补 0 槽位，想强制 `[64,32]` 的基准测试仍可照旧传
  显式值（`sconv-blas-fastp-tuned.mlir`）。

**实现**（`lib/SConv.cpp`、`include/SConv.td`）：
1. `mK_info` 中任一槽位为 0 表示该槽位「自动」。`[0,0]` 即两槽全自动。
2. `SConvOp::apply` 在 `createCSAPass` **之前**、对**每个卷积**解析成 `convMK`：
   `Nwin = {64,32,16}` 中能整除 `Oh*Ow` 的最大者（且 `Ohw >= Nwin`），否则 16；
   `Nf = {32,16,8}` 中能整除 `fn` 的最大者，否则 8。
3. `hasEdgeCase` 与后续 CSA/tiling 全部使用解析后的 `convMK`，保证「判断是否走 edge」
   与实际 tiling 用的是同一组 tile。
4. 按卷积解析而不是按 transform 解析：一个 `transform.structured.sconv` 可以匹配多个
   不同形状的卷积，tile 必须随形状走。

**实测（11 个 custom 形状，单线程，ms/iter）**：

| 形状 | Ohw / fn | 自动选择 | `[16,8]` | 自动 | 变化 |
|---|---|--:|--:|--:|--:|
| custom_0 | 1600 / 192 | `[64,32]` | 116 | **87** | -25% |
| custom_1 | 1600 / 768 | `[64,32]` | 2056 | **1403** | -32% |
| custom_2 | 6400 / 96 | `[64,32]` | 159 | **113** | -29% |
| custom_3 | 400 / 768 | `[16,32]` | 498 | **351** | -30% |
| custom_4 | 6400 / 384 | `[64,32]` | 2026 | **1561** | -23% |
| custom_5 | 25600 / 48 | `[64,16]` | 229 | **178** | -22% |
| custom_6 | 25600 / 192 | `[64,32]` | 1437 | **1165** | -19% |
| custom_7 | 400 / 192 | `[16,32]` | 36 | **29** | -19% |
| custom_8 | 1600 / 384 | `[64,32]` | 508 | **376** | -26% |
| custom_9 | 6400 / 96 | `[64,32]` | 534 | **370** | -31% |
| custom_10 | 1600 / 96 | `[64,32]` | 268 | **195** | -27% |

- **11/11 改善，没有回退**；custom_0 从 M1 的 115.7 → ≈87 ms（相对原始 156.6 为 -44%）。
- custom_5 / custom_7 正是写死 `[64,32]` 会崩的两个形状，自动选择分别落到
  `Nf=16` 和 `Nwin=16`，因此既拿到宽 tile 的大部分收益，又完全避开 edge 路径。
- 与 M2 原型表逐项一致（±4–7% 机器噪声内），说明「解析 tile」与「显式传同一 tile」
  在编译结果上是等价的。

**验收**：
- `scripts/convbench/test_mk_auto.py --stage ir`：对 4 个代表形状断言
  **auto 产生的 IR 与显式传入选定 tile 的 IR 完全相同**（脚本只差一个属性），且
  IR 中已出现 `pack_input_tile`、不再有标量 input-packing generic。
- 同一脚本 `--stage perf`：custom_0 自动路径 **≤105 ms/iter**（实测 ≈87）。
- `scripts/convbench/test_fastpack.py --stage all`：默认快路径的 IR / 正确性 /
  性能门禁；正确性用例增加 custom_7（`Nwin` 回退）与 custom_5（`Nf` 回退）作为
  回归守卫。

**边界与代价**：
- 规则只看整除性，不看 cache 大小/线程数；`arch_info` 仍独立生效。若将来要跨机器
  自适应，应把「候选集」换成本机微基准测得的表，而不是继续堆启发式。
- 候选集是固定的三个窗口/三个滤波器 tile；这覆盖了实测中的最优区间
  （`[64,32]` 最优，`[80,32]`/`[64,64]` 已变差），没有必要扩成在线搜索。
- 不做全量 autotune 的原因：编译期搜索需要为每个候选真正 lower + 计时（分钟级/形状），
  而整除规则把 11 个形状全部选对，收益/成本明显更好；真要做全量搜索，应放到
  `native_impl` 那类离线调优器里生成表。

---

## 4. 被否决 / 被证伪的方向（同样是结论）

| 方向 | 做法 | 结果 | 判定依据 |
|---|---|---|---|
| stride==1 打包向量化 | 内层换 `std::copy` 特化 | 0.91–1.06× | 噪声内；clang 已向量化且带宽受限 |
| 打包微优化（3 变体） | 标量 / `memcpy` / 合并写三份独立微基准 | 13.17 / 13.33 / 12.39 GB/s | 三种实现同速 ⇒ 带宽 roofline |
| 扩大 SME 覆盖 | `c0/c2/c5/c7` 路由到 SME 直接法 | kernel-only 920/1130/1058/273 GF，均低于 GEMM 2150/2270/1913/1264 | 全面更慢 |
| 放大 `Nf` | 扫 `Nf ∈ {16,32,64,128,256}` | `{16,32}` 即最优 | 无单调收益 |
| 无条件搜 `Nc` | 所有形状都搜 | `thr_n8` −43% | 必须加「饥饿形状 + 3% 门槛」约束 |

> 注：`common_r18_7`（0.23 GFLOP，t16–t32 落后）经同样评估后**不路由 SME**：
> 端到端 SME t32 126 GF 仍略低于 GEMM 132 GF。属结构性，接受现状。

---

## 5. 当前性能与结构的关系（怎么读那些数字）

1. **t1 的 GFLOPS 主要由 GEMM 效率决定**（打包占比 9%–49%，随通道数增大而升高）。
2. **t32 的 GFLOPS 受两个上限夹击**：`N·窗tile` 的任务数（并行度）× 打包带宽（单核 13 GB/s，多核共享）。
   这解释了 `r18_7`/`r50_ds28` 为什么在 t32 也只有 100–400 GFLOPS 量级。
3. **不是所有用例都该线性扩展**：`custom_1/custom_3` 这类大通道用例 t1 就有 2.5×+ 优势，
   随线程增大会逐渐收敛到 torch 的水平（torch 的多线程扩展更好），这是 t32 加速比回落的来源。
4. 加速比矩阵（after/torch）与自身提升矩阵（after/before）分别在
   `docs/PERFORMANCE_VS_PYTORCH.md` §3.2 / §3.3，**不要混用**。

---

## 6. 测量口径与已知陷阱（重要的诚实性声明）

### 6.1 t ≥ 35 是慢模式，不是噪声

node0 有 38 核，但 t≥35 会稳定进入一个不可复现的慢模式（`c9` t32/t34 ≈ 2695/2709 GFLOPS，
t36 稳定 1650–1760，IQR ±3–5%）。已排除 pool 原语、绑核、DVFS。
**故线程扫描截到 t32**，≥35 档不作为结论（`PERFORMANCE_VS_PYTORCH.md` §1.4）。

### 6.2 ConvBench 的 GFLOPS 曾漏乘 batch（已修复）

`scripts/convbench/gen_payload.py` 的 FLOPS 公式原为
`2·HO·WO·DO·CI·HK·WK`，漏了输出 batch `NO`。对 batch=4 的自定义卷积，
**打印的 GFLOPS 被低估 4×**；`ms` 与**加速比**不受影响（两侧同乘一个常数）。

- 修正后：`custom_0` 的 SConv 由 6.78 → **27.1 GFLOPS**（4.25 GFLOP / 156.6 ms）。
- 脚本已修复并按新口径重新生成 payload；`docs/PERFORMANCE_REPORT.md` 里 2026-09 的历史表
  中 batch=4 行的 GFLOPS 列请按 ×4 理解（加速比列无需修正）。

### 6.3 自动调优本身是「有噪声的决策器」

同一形状跨进程可能选到不同 `(Nwin,Nf,Nc)`（min-of-5 已缓解但未消除）。
因此**不要用单次运行比较两个版本**；也不要把调优选出的参数当作「最优参数」写进论文。

---

## 7. 代码地图（按「改哪里」索引）

| 想改什么 | 文件 | 关键位置 / 开关 |
|---|---|---|
| 算法级 tiling / packing / 微内核替换 | `lib/SConv.cpp`、`lib/LowerToBLAS.cpp` | `applyTileTo()`、`scf::tileUsingSCF`、`LowerToBLASOp::apply()` |
| 快速输入打包（M1/M2） | `lib/LowerToPack.cpp`、`scripts/convbench/runtime/pack_input_tile.c`、`test/lowering/sconv-blas-fastp*.mlir` | `LowerToPackOp::apply`、`applyInputPacking`/`inputMultipackingOpt`（buffer 上提）、`pack_input_tile()` |
| 打包验收门槛 | `scripts/convbench/test_fastpack.py` | `--stage ir\|correct\|perf`；`PERF_BUDGET_MS` |
| SME 路由决策 | `lib/LowerToSME.cpp` | `SMESelectOp::apply`、`min_ic`/`max_oc` 门槛 |
| 参考实现的并行/调优 | `native_impl/src/sconv.cpp` | `build_Bc`、`run_pass`、自适应块、`SCONV_*` 环境变量 |
| CSA 启发式 | `native_impl/src/sconv_csa.h`、`lib/CSA.cpp` | `halfHeuristic`（Nc/K2/K3） |
| SME 端到端 shim | `sme/runtime/sconv_sme_shim.c` | `pool_get`/`pool_run`、`task_packpad`、`task_compute` |
| ConvBench 测速/校验 | `scripts/convbench/` | `run_convbench.py`、`gen_payload.py`、`runtime/sgemm_blas_kernel.c` |
| vs PyTorch 三方对比 | `scripts/bench_vs_torch/` | `compare_vs_pytorch.py --threads 1,2,4,8,16,32` |

环境变量（参考实现）：`SCONV_VERBOSE=1` 打印选中的 `Nwin/Nf/Nc`；
`SCONV_TIMING=1` 分阶段计时；`SCONV_NO_AUTOTUNE=1` 关闭自适应。

---

## 8. 复现

```bash
source env.sh                       # clang 19.1.7 + kblas

# 参考实现（含自动调优）
cd native_impl && cmake -B build -S . -USE_KBLAST=ON -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -j
numactl --cpunodebind=0 --membind=0 ./build/sconv_bench --filter common \
  --skip-naive --runs 8 --warmup 3 --nwin 0 --nf 0 --threads 32

# MLIR 管线（ConvBench：正确性 + 性能）
export SCONV_ROOT=$PWD
python3 scripts/convbench/run_convbench.py --csv scripts/convbench/samples/custom_conv.csv \
  --num 1 --runs 3 --warmup 2

# vs PyTorch 全量
python3 scripts/bench_vs_torch/compare_vs_pytorch.py --threads 1,2,4,8,16,32
```

---

## 9. 相关文档

| 文档 | 回答什么问题 |
|---|---|
| `docs/PERFORMANCE_VS_PYTORCH.md` | **权威性能数据**：18 用例 × 6 线程档、瓶颈证据、复现方法 |
| 本文 | 设计理念、实现地图、优化项的「为什么/效果/证伪」 |
| `docs/FEASIBILITY_VECTORIZED_PACK_MULTICORE.md` | 「向量化打包 + 多核」值不值得做、怎么做、风险与验收 |
| `docs/OPTIMIZATION_OPPORTUNITIES.md` | 更早的优化机会清单（含 SME/流水线/NHWC 等方案设想）与当前落地状态 |
| `docs/PERFORMANCE_REPORT.md` | 2026-09 的 ConvBench 初版结果（注意 §6.2 的口径修正） |
| `docs/INTRODUCTION.md`、`docs/TARGET_MACHINE_ANALYSIS.md` | 算法与目标机背景 |
