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
- 现状（custom_0）：**156.6 ms/iter，27.1 GFLOPS**（单线程）。
  注意：ConvBench 表格里打印的 6.78 GFLOPS 是旧口径（FLOPs 漏乘 batch，见 §6.2），已修正为 4×。
- 结构性问题：**微内核极窄**（`C[16,8] += A[216,16]^T B[216,8]`，每次 0.055 MFLOP，每轮卷积 **76,800 次调用**），
  打包是逐元素标量搬运（按「总时间 − 实测 GEMM 下界」反推，等效约 **0.5 GB/s**，见可行性论证 §2.3）。

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
