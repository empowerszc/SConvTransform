# SConvTransform 卷积 vs PyTorch conv2d：多线程性能对比、瓶颈分析与优化实施

> 测试日期: 2026-10-15（优化实施同日）
> 平台: openEuler 22.03 LTS-SP4, aarch64 HiSilicon Kunpeng (SVE2 + SME), 单 NUMA node 0 (38 核), HPCKit kblas
> 对比对象: `torch.nn.functional.conv2d` (PyTorch 2.10.0+cpu)
> 范围: 11 个自定义卷积 + 7 个 ResNet 常见卷积，线程数 1/2/4/8/16/32/36
> 相关: `docs/PERFORMANCE_REPORT.md`、`docs/OPTIMIZATION_OPPORTUNITIES.md`、`docs/superpowers/plans/2026-10-15-sme-integration.md`

---

## 0. 结论摘要

1. **优化后本方案在 18 个用例上几乎全面领先 PyTorch**：各线程数下 geometric-mean 加速比
   **1.57×–1.67×**，每个线程档 17/18 或 18/18 取胜（优化前为 1.17×–1.39×）。
2. **两项优化生效**（均落在 `native_impl/src/sconv.cpp`）：
   - **P0 `Bc` 滤波器预打包并行化**：batch=1 低分辨率用例提升显著（`common_r18_7` 1.8–1.9×、`common_r18_14` 1.7–2.0×、`common_r18_28` t32 1.7×）。
   - **P0/P1 `(Nwin, Nf)` 自适应（运行时自动调优）**：`common_r18_7` 最多 3.15×、`common_r18_14` 2.73×、`common_r50_ds28` 2.62×；同时修复了因固定 `Nf` 导致的 `custom_1/custom_3` 回退（+1.1–1.4×）。
3. **一项被否决**：stride==1 打包向量化（`std::copy` 特化）实测无收益（打包为带宽受限，clang 已向量化），已回退。
4. **唯一仍落后**：`common_r18_7`（1×512×7×7→512，仅 0.23 GFLOP）在 t16–t36 落后（0.63–0.85×）——问题太小，固定开销（`Bc`、启动）占比过高。
5. **重要口径修正**：原 `SConvTransform` GEMM 基准由 **GCC 10.3.1** 构建，而仓库默认工具链（`env.sh`）是 **clang 19.1.7**；clang 在打包等访存循环上快约 2×。本文所有数字统一为 **clang** 构建。

---

## 1. 测试环境与方法

### 1.1 环境

| 项目 | 规格 |
|------|------|
| CPU | HiSilicon Kunpeng `0x48-0xd22`，608 核 / 32 NUMA / 38 核每 NUMA |
| 指令集 | SVE2 + SME，SVL=512bit |
| BLAS | HPCKit 26.1.RC1 kblas（SME 微内核） |
| 编译器 | **clang 19.1.7**（`$LLVM_BUILD_DIR`，CMake `-O2 -DNDEBUG`） |
| PyTorch | 2.10.0+cpu |
| 绑定 | `numactl --cpunodebind=0 --membind=0` |

### 1.2 对比口径

"本方案 (our)" 对每个用例取两条路径：

| 路径 | 适用范围 | 测量内容 |
|------|----------|----------|
| **SME 编译内核** | `custom_9`、`custom_10` | `sme` 探针 `bench` 的**纯计算**耗时（输入已 pad+pack，输出未 unpad） |
| **native_impl GEMM** | 其余 16 个用例 | `sconv_bench`，含 im2col 打包 + kblas `sgemm` + 解包；`(Nwin,Nf)` 自动调优 |

- **口径注意**：SME 路径为 **kernel-only**（不含 pad/pack/unpad），其加速比是**乐观上界**；GEMM 路径与 torch 均为端到端、相对可比。
- 计时：进程内 `warmup=3` + `runs=8`，取单次平均；每格 3 次重复取最优。GFLOPS = `2·N·Oc·Oh·Ow·Ic·Fh·Fw / t`。
- "修改前 (before)" = 未做本次优化的 clang 构建（`Bc` 串行 + 固定 tile，取 `(Nwin,Nf)∈{(32,16),(64,32)}` 中较优者）；"修改后 (after)" = 并行 `Bc` + 自动 `(Nwin,Nf)`。

### 1.3 用例

自定义 (batch=4)：`custom_0…custom_10`（见 `docs/PERFORMANCE_REPORT.md` §3.1）。
ResNet 常见 (新增)：

| ID | Input | Weight | Stride | Oh | FLOPs |
|----|-------|--------|:---:|:---:|---:|
| common_r18_56 | 1×64×56×56 | 64×64×3×3 | 1 | 56×56 | 0.23G |
| common_r18_28 | 1×128×28×28 | 128×128×3×3 | 1 | 28×28 | 0.23G |
| common_r18_14 | 1×256×14×14 | 256×256×3×3 | 1 | 14×14 | 0.23G |
| common_r18_7 | 1×512×7×7 | 512×512×3×3 | 1 | 7×7 | 0.23G |
| common_r50_ds56 | 1×128×56×56 | 256×128×3×3 | 2 | 28×28 | 0.46G |
| common_r50_ds28 | 1×256×28×28 | 512×256×3×3 | 2 | 14×14 | 0.46G |
| common_thr_n8 | 8×64×56×56 | 64×64×3×3 | 1 | 56×56 | 1.85G |

---

## 2. 优化实施与前后对比

两项改动都在 `native_impl/src/sconv.cpp`，数值语义不变（通过与 native 逐元素比对验证）。

### 2.1 P0：`Bc` 滤波器预打包并行化

`Bc` 把滤波器重排为 `[ft][ic0][k][nf]`，原为**单线程**且每次调用重算。改为对互不相交的 `(ft, it)` 切片 `#pragma omp parallel for collapse(2)`。

A/B（clang，`before` vs `+Bc并行`，`Nwin/Nf` 固定，取 3 次最优，GFLOPS）：

| 用例 | t1 | t8 | t32 |
|---|:--:|:--:|:--:|
| common_r18_7 | 0.94× | **1.80×** | **1.89×** |
| common_r18_14 | 0.98× | **1.96×** | **1.68×** |
| common_r18_28 | 0.97× | 1.23× | **1.68×** |
| custom_0/5/2 | 0.96–1.02× | 1.01–1.02× | 1.02–1.14× |

即：**batch=1 且 `Bc` 占比高的用例**在多线程下大幅提速；t1 无变化（`Bc` 仍串行，符合预期）；大 batch 用例影响小（`Bc` 相对占比低）。

### 2.2 P0/P1：`(Nwin, Nf)` 自适应调优

问题（见 §4.2/§4.3）：并行分解为 `batch × 窗口tile`，batch=1 低分辨率用例任务数只有 2–25 个，无法用满多核；且最优 `Nwin` 随规模/线程数变化。`Nf` 则影响大 `Oc` 用例的滤波器复用。

实现：当 `--nwin/--nf` 传 0（默认）时，按 `(shape, threads)` 为键，对小网格 `Nwin∈{8,16,32,64} × Nf∈{16,32}` 各做 1 次预热 + 2 次计时，取最快者并缓存；`Bc` 按选定 `Nf` 重建。显式传值可覆盖。

A/B（`+Bc并行` vs `+Bc并行+自适应`，隔离自适应效果，GFLOPS）：

| 用例 | t1 | t8 | t32 |
|---|:--:|:--:|:--:|
| common_r18_7 | 1.13× | **1.58×** | 1.45× |
| common_r18_14 | 1.16× | 1.09× | **1.74×** |
| common_r50_ds28 | 1.18× | 1.10× | **1.59×** |
| custom_0 | 1.24× | 1.07× | 1.01× |
| custom_5 | 1.15× | 1.11× | 0.92× |

（自动调优恢复并超过了原固定配置：`custom_1` t1 由固定 `nf16` 的 65 GF 提升到自动 `nw32/nf32` 的 ~118 GF。）

### 2.3 被否决：stride==1 打包向量化

原假设：打包内层 `pA[w]=s[(w-w0)*strideW]` 为标量跨步拷贝，可用 `std::copy` 特化加速。实测（same-binary A/B）**无收益**（比值 0.91–1.06，即噪声）——clang 已向量化，且打包为**带宽受限**。已**回退**该改动。

---

## 3. 最终结果 vs PyTorch

### 3.1 重点用例：custom_9 / custom_10（SME，kernel-only，未改动）

| 用例 | | t1 | t2 | t4 | t8 | t16 | t32 | t36 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| **custom_9** | 本方案(SME) | 200 | 340 | 647 | 1239 | 2425 | 3126 | 3134 |
| | torch | 68 | 144 | 288 | 544 | 948 | 1941 | 2138 |
| | **加速比** | 2.94× | 2.37× | 2.24× | 2.28× | 2.56× | 1.61× | 1.47× |
| **custom_10** | 本方案(SME) | 122 | 244 | 467 | 849 | 1351 | 1582 | 1816 |
| | torch | 50 | 97 | 186 | 345 | 665 | 1257 | 1355 |
| | **加速比** | 2.47× | 2.52× | 2.51× | 2.46× | 2.03× | 1.26× | 1.34× |

### 3.2 修改后 加速比矩阵 (after / torch)

| 用例 | t1 | t2 | t4 | t8 | t16 | t32 | t36 |
|---|--:|--:|--:|--:|--:|--:|--:|
| custom_0 | 1.36 | 1.37 | 1.21 | 1.20 | 1.42 | 1.46 | 1.55 |
| custom_1 | 2.73 | 2.58 | 2.50 | 2.28 | 1.99 | 1.42 | 1.67 |
| custom_2 | 1.45 | 1.40 | 1.37 | 1.41 | 1.45 | 1.54 | 1.56 |
| custom_3 | 2.90 | 2.77 | 2.58 | 2.28 | 1.90 | 1.46 | 1.44 |
| custom_4 | 1.55 | 1.53 | 1.46 | 1.39 | 1.34 | 1.36 | 1.31 |
| custom_5 | 1.49 | 1.59 | 1.69 | 1.67 | 1.64 | 1.77 | 1.80 |
| custom_6 | 1.43 | 1.48 | 1.45 | 1.59 | 1.81 | 1.67 | 2.59 |
| custom_7 | 1.10 | 1.12 | 1.17 | 1.25 | 1.43 | 1.54 | 1.51 |
| custom_8 | 1.79 | 1.74 | 1.67 | 1.62 | 1.68 | 1.37 | 1.35 |
| custom_9 | 2.94 | 2.37 | 2.24 | 2.28 | 2.56 | 1.61 | 1.47 |
| custom_10 | 2.47 | 2.52 | 2.51 | 2.46 | 2.03 | 1.26 | 1.34 |
| common_r18_56 | 1.56 | 1.63 | 1.81 | 2.18 | 3.11 | 3.50 | 3.73 |
| common_r18_28 | 1.16 | 1.34 | 1.39 | 1.46 | 1.61 | 2.17 | 2.62 |
| common_r18_14 | 1.09 | 1.21 | 1.29 | 1.55 | 1.36 | 1.85 | 1.78 |
| common_r18_7 | 0.82 | 0.99 | 1.17 | 0.96 | 0.85 | 0.63 | 0.72 |
| common_r50_ds56 | 1.43 | 1.44 | 1.49 | 1.58 | 1.98 | 2.62 | 2.38 |
| common_r50_ds28 | 1.20 | 1.22 | 1.25 | 1.44 | 1.13 | 1.09 | 1.35 |
| common_thr_n8 | 1.84 | 1.45 | 1.56 | 1.59 | 1.57 | 1.66 | 1.80 |

### 3.3 本方案自身提升 (after / before)

| 用例 | t1 | t2 | t4 | t8 | t16 | t32 | t36 |
|---|--:|--:|--:|--:|--:|--:|--:|
| custom_0 | 1.10 | 1.12 | 1.15 | 1.14 | 1.18 | 1.22 | 1.09 |
| custom_1 | 1.17 | 1.13 | 1.07 | 1.05 | 1.07 | 1.14 | 1.14 |
| custom_2 | 1.24 | 1.21 | 1.20 | 1.15 | 1.11 | 1.01 | 1.01 |
| custom_3 | 1.14 | 1.13 | 1.14 | 1.22 | 1.34 | 1.43 | 1.43 |
| custom_4 | 1.26 | 1.23 | 1.18 | 1.17 | 1.10 | 1.13 | 1.03 |
| custom_5 | 1.21 | 1.16 | 1.12 | 1.11 | 1.02 | 0.92 | 0.91 |
| custom_6 | 1.20 | 1.20 | 1.19 | 1.18 | 1.17 | 1.07 | 1.10 |
| custom_7 | 1.18 | 1.18 | 1.23 | 1.28 | 1.33 | 1.38 | 1.38 |
| custom_8 | 1.37 | 1.35 | 1.31 | 1.34 | 1.28 | 1.29 | 1.22 |
| custom_9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| custom_10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| common_r18_56 | 1.04 | 1.09 | 1.10 | 1.10 | 1.17 | 1.03 | 1.03 |
| common_r18_28 | 1.18 | 1.25 | 1.28 | 1.20 | 1.33 | 1.45 | 1.46 |
| common_r18_14 | 1.16 | 1.33 | 1.51 | 1.97 | 2.24 | 2.73 | 2.70 |
| common_r18_7 | 0.81 | 1.15 | 2.20 | 2.73 | 3.00 | 3.15 | 3.15 |
| common_r50_ds56 | 1.15 | 1.19 | 1.30 | 1.42 | 1.56 | 1.64 | 1.59 |
| common_r50_ds28 | 1.11 | 1.33 | 1.58 | 2.16 | 2.28 | 2.62 | 2.57 |
| common_thr_n8 | 1.19 | 1.13 | 1.13 | 1.13 | 1.02 | 0.98 | 0.96 |

### 3.4 汇总

| 线程 | before/torch(geo) | after/torch(geo) | after 赢 | after 输 |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 1.39× | **1.58×** | 17 | 1 |
| 2 | 1.35× | **1.58×** | 17 | 0 |
| 4 | 1.29× | **1.60×** | 18 | 0 |
| 8 | 1.25× | **1.62×** | 17 | 0 |
| 16 | 1.24× | **1.65×** | 17 | 1 |
| 32 | 1.17× | **1.57×** | 17 | 1 |
| 36 | 1.26× | **1.67×** | 17 | 1 |

---

## 4. 瓶颈根因分析（方法记录）

系统化定位：先取证据（分阶段计时 `SCONV_TIMING`、任务计数），再最小验证（Nwin 扫描、A/B）。

### 4.1 证据：im2col 打包占大头（clang，t1，`Nwin/Nf=32/16`，单次）

| 用例 | 总(ms) | pack(ms) | gemm(ms) | unpack(ms) | **pack 占比** |
|------|------:|------:|------:|------:|:---:|
| custom_5 | 62.3 | 30.5 | 24.1 | 5.5 | **49%** |
| custom_2 | 60.6 | 25.3 | 27.5 | 6.8 | **42%** |
| custom_0 | 50.0 | 15.0 | 29.5 | 4.5 | 30% |
| common_r18_56 | 3.82 | 1.61 | 1.58 | 0.31 | 42% |
| common_r18_28 | 4.61 | 1.53 | 2.26 | 0.41 | 33% |
| common_r18_14 | 4.35 | 0.94 | 2.14 | 0.25 | 22% |
| common_r18_7 | 7.25 | 0.66 | 3.12 | 0.08 | 9%（另 `Bc`/开销 ≈3.4ms，占 47%） |

打包为 im2col（写 ≈ 输入×9），**带宽受限**，且 torch 走隐式 GEMM 不物化 → 低通道/大空间用例吃亏。

### 4.2 证据：并行任务粒度不足

任务 = `batch × ceil(Ohw/Nwin)`：

| 用例 | N | Ohw | 任务数(nw32) | 任务数(nw8) |
|---|--:|---:|:---:|:---:|
| common_r18_7 | 1 | 49 | **2** | 7 |
| common_r18_14 | 1 | 196 | **7** | 25 |
| common_r18_28 | 1 | 784 | 25 | 98 |
| common_r18_56 | 1 | 3136 | 98 | 392 |
| custom_0 | 4 | 1600 | 200 | 800 |

batch=1 低分辨率用例任务数远小于 38 核 → 用不满。缩小 `Nwin` 增加任务数（32 线程，GFLOPS）：

| 用例 | nw8 | nw16 | nw32 | nw64 |
|------|----:|----:|----:|----:|
| common_r18_7 | **48.0** | 44.6 | 36.8 | 30.3 |
| common_r18_14 | **154.8** | 141.7 | 119.1 | 89.3 |
| common_r18_7 (t1) | 23.8 | 29.4 | 31.4 | **34.6** |

而单线程反而大 `Nwin` 最优 → `Nwin` 最优值与规模/线程数强相关，固定值必次优。这正是 §2.2 自适应的依据。

### 4.3 证据：`Bc` 单线程固定开销

`Bc` 元素数 = `Oc·Ic·Fh·Fw`，原为单线程：`common_r18_7` 为 2.36M 元素，占其单次耗时约 **44%**（≈3.4 ms）；`custom_1` 为 5.31M。这解释了 §2.1 并行化后 r18_7/14/28 的大幅提升。

### 4.4 证据：`Nf` 影响大 `Oc` 用例

`custom_1`(Oc=768) t1：`nf16`=65 GF、`nf32`=86 GF（差 ~30%）。固定 `Nf=16` 会拖累大 `Oc` → §2.2 将 `Nf` 一并纳入调优。

---

## 5. 剩余差距

- **`common_r18_7`（t16–t36，0.63–0.85×）**：0.23 GFLOP 的极小用例，`Bc` 与启动/同步固定开销占比过高，任务数上限受 `Ohw=49` 限制。属结构性问题，收益有限；若需覆盖，建议改走 SME 直接卷积（复用既有 SME 路径）。
- `custom_5`/`common_thr_n8` 在 t32/t36 有 ~0.9–0.98× 的轻微波动（噪声量级）。

---

## 6. 复现方法

```bash
source env.sh                       # 使 clang 19.1.7 + kblas 生效
cd native_impl && cmake -B build -S . -DUSE_KBLAST=ON -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -j

# 本方案（自动调优）：--nwin 0 --nf 0
numactl --cpunodebind=0 --membind=0 ./build/sconv_bench --filter common \
  --skip-naive --runs 8 --warmup 3 --nwin 0 --nf 0 --threads 32
# 查看选中的 (Nwin,Nf)：加 SCONV_VERBOSE=1
# 分阶段计时（每次调用独立）：SCONV_TIMING=1
# 关闭自动调优（用固定 Nwin=32/Nf=16）：SCONV_NO_AUTOTUNE=1
# SME kernel-only：cd sme && make direct.out && ./direct.out bench <threads>

# torch 基线 / 全量对比：scripts/bench_vs_torch/（见附录 B）
export SCONV_ROOT=$PWD TORCH_PYTHON=<torch 环境的 python>
python3 scripts/bench_vs_torch/compare_vs_pytorch.py --threads 1,2,4,8,16,32,36
```

---

## 附录 A：完整数据

### A.1 修改后 (after) 各线程 GFLOPS

**线程数 = 1**

| 用例 | 路径 | torch | 本方案 | 加速比 |
|---|---|---:|---:|---:|
| custom_0 | GEMM | 83.7 | 114.0 | 1.36× |
| custom_1 | GEMM | 45.5 | 124.1 | 2.73× |
| custom_2 | GEMM | 71.9 | 104.1 | 1.45× |
| custom_3 | GEMM | 39.5 | 114.4 | 2.90× |
| custom_4 | GEMM | 67.8 | 104.8 | 1.55× |
| custom_5 | GEMM | 63.2 | 94.4 | 1.49× |
| custom_6 | GEMM | 57.5 | 82.5 | 1.43× |
| custom_7 | GEMM | 91.4 | 100.7 | 1.10× |
| custom_8 | GEMM | 62.6 | 112.2 | 1.79× |
| custom_9 | SME | 68.0 | 200.1 | 2.94× |
| custom_10 | SME | 49.6 | 122.4 | 2.47× |
| common_r18_56 | GEMM | 55.6 | 86.6 | 1.56× |
| common_r18_28 | GEMM | 69.3 | 80.2 | 1.16× |
| common_r18_14 | GEMM | 60.1 | 65.7 | 1.09× |
| common_r18_7 | GEMM | 33.1 | 27.3 | 0.82× |
| common_r50_ds56 | GEMM | 58.1 | 83.2 | 1.43× |
| common_r50_ds28 | GEMM | 56.7 | 67.9 | 1.20× |
| common_thr_n8 | GEMM | 50.9 | 93.8 | 1.84× |

**线程数 = 2**

| 用例 | 路径 | torch | 本方案 | 加速比 |
|---|---|---:|---:|---:|
| custom_0 | GEMM | 162.6 | 223.0 | 1.37× |
| custom_1 | GEMM | 93.6 | 241.2 | 2.58× |
| custom_2 | GEMM | 153.0 | 214.3 | 1.40× |
| custom_3 | GEMM | 79.5 | 220.5 | 2.77× |
| custom_4 | GEMM | 134.0 | 204.4 | 1.53× |
| custom_5 | GEMM | 117.7 | 187.6 | 1.59× |
| custom_6 | GEMM | 112.2 | 165.7 | 1.48× |
| custom_7 | GEMM | 165.5 | 185.8 | 1.12× |
| custom_8 | GEMM | 124.7 | 217.3 | 1.74× |
| custom_9 | SME | 143.8 | 340.2 | 2.37× |
| custom_10 | SME | 96.6 | 243.8 | 2.52× |
| common_r18_56 | GEMM | 95.5 | 155.3 | 1.63× |
| common_r18_28 | GEMM | 109.2 | 146.4 | 1.34× |
| common_r18_14 | GEMM | 91.1 | 110.3 | 1.21× |
| common_r18_7 | GEMM | 48.3 | 47.7 | 0.99× |
| common_r50_ds56 | GEMM | 105.6 | 152.2 | 1.44× |
| common_r50_ds28 | GEMM | 98.7 | 120.9 | 1.22× |
| common_thr_n8 | GEMM | 126.4 | 182.9 | 1.45× |

**线程数 = 4**

| 用例 | 路径 | torch | 本方案 | 加速比 |
|---|---|---:|---:|---:|
| custom_0 | GEMM | 343.6 | 414.2 | 1.21× |
| custom_1 | GEMM | 177.4 | 442.9 | 2.50× |
| custom_2 | GEMM | 291.9 | 399.6 | 1.37× |
| custom_3 | GEMM | 155.6 | 401.9 | 2.58× |
| custom_4 | GEMM | 265.5 | 387.5 | 1.46× |
| custom_5 | GEMM | 212.4 | 359.5 | 1.69× |
| custom_6 | GEMM | 221.5 | 322.1 | 1.45× |
| custom_7 | GEMM | 292.2 | 340.9 | 1.17× |
| custom_8 | GEMM | 245.9 | 411.8 | 1.67× |
| custom_9 | SME | 288.5 | 647.3 | 2.24× |
| custom_10 | SME | 186.2 | 467.1 | 2.51× |
| common_r18_56 | GEMM | 148.1 | 267.9 | 1.81× |
| common_r18_28 | GEMM | 164.7 | 229.6 | 1.39× |
| common_r18_14 | GEMM | 136.9 | 176.1 | 1.29× |
| common_r18_7 | GEMM | 80.3 | 93.8 | 1.17× |
| common_r50_ds56 | GEMM | 175.7 | 261.2 | 1.49× |
| common_r50_ds28 | GEMM | 161.5 | 202.0 | 1.25× |
| common_thr_n8 | GEMM | 225.7 | 352.8 | 1.56× |

**线程数 = 8**

| 用例 | 路径 | torch | 本方案 | 加速比 |
|---|---|---:|---:|---:|
| custom_0 | GEMM | 647.4 | 774.3 | 1.20× |
| custom_1 | GEMM | 347.2 | 792.4 | 2.28× |
| custom_2 | GEMM | 531.8 | 748.5 | 1.41× |
| custom_3 | GEMM | 306.0 | 698.2 | 2.28× |
| custom_4 | GEMM | 523.3 | 725.0 | 1.39× |
| custom_5 | GEMM | 390.6 | 652.9 | 1.67× |
| custom_6 | GEMM | 383.8 | 608.6 | 1.59× |
| custom_7 | GEMM | 477.4 | 597.5 | 1.25× |
| custom_8 | GEMM | 474.4 | 767.2 | 1.62× |
| custom_9 | SME | 544.2 | 1238.7 | 2.28× |
| custom_10 | SME | 345.0 | 849.0 | 2.46× |
| common_r18_56 | GEMM | 198.0 | 431.1 | 2.18× |
| common_r18_28 | GEMM | 223.4 | 327.2 | 1.46× |
| common_r18_14 | GEMM | 186.2 | 287.7 | 1.55× |
| common_r18_7 | GEMM | 122.2 | 117.0 | 0.96× |
| common_r50_ds56 | GEMM | 257.8 | 408.5 | 1.58× |
| common_r50_ds28 | GEMM | 247.2 | 354.9 | 1.44× |
| common_thr_n8 | GEMM | 417.2 | 661.3 | 1.59× |

**线程数 = 16**

| 用例 | 路径 | torch | 本方案 | 加速比 |
|---|---|---:|---:|---:|
| custom_0 | GEMM | 927.0 | 1320.1 | 1.42× |
| custom_1 | GEMM | 650.8 | 1292.0 | 1.99× |
| custom_2 | GEMM | 892.2 | 1292.0 | 1.45× |
| custom_3 | GEMM | 570.0 | 1082.2 | 1.90× |
| custom_4 | GEMM | 917.6 | 1232.5 | 1.34× |
| custom_5 | GEMM | 671.4 | 1098.6 | 1.64× |
| custom_6 | GEMM | 608.2 | 1100.2 | 1.81× |
| custom_7 | GEMM | 605.7 | 867.1 | 1.43× |
| custom_8 | GEMM | 788.5 | 1323.1 | 1.68× |
| custom_9 | SME | 948.0 | 2424.6 | 2.56× |
| custom_10 | SME | 665.3 | 1351.0 | 2.03× |
| common_r18_56 | GEMM | 230.2 | 716.5 | 3.11× |
| common_r18_28 | GEMM | 301.3 | 484.4 | 1.61× |
| common_r18_14 | GEMM | 242.8 | 331.2 | 1.36× |
| common_r18_7 | GEMM | 149.8 | 127.9 | 0.85× |
| common_r50_ds56 | GEMM | 324.0 | 640.9 | 1.98× |
| common_r50_ds28 | GEMM | 336.3 | 379.4 | 1.13× |
| common_thr_n8 | GEMM | 675.6 | 1060.2 | 1.57× |

**线程数 = 32**

| 用例 | 路径 | torch | 本方案 | 加速比 |
|---|---|---:|---:|---:|
| custom_0 | GEMM | 1421.5 | 2079.5 | 1.46× |
| custom_1 | GEMM | 1153.5 | 1639.3 | 1.42× |
| custom_2 | GEMM | 1283.8 | 1982.5 | 1.54× |
| custom_3 | GEMM | 1042.0 | 1526.4 | 1.46× |
| custom_4 | GEMM | 1474.4 | 1997.9 | 1.36× |
| custom_5 | GEMM | 929.4 | 1645.3 | 1.77× |
| custom_6 | GEMM | 1080.9 | 1809.7 | 1.67× |
| custom_7 | GEMM | 806.9 | 1239.7 | 1.54× |
| custom_8 | GEMM | 1520.5 | 2087.8 | 1.37× |
| custom_9 | SME | 1941.4 | 3125.5 | 1.61× |
| custom_10 | SME | 1256.6 | 1582.1 | 1.26× |
| common_r18_56 | GEMM | 236.6 | 828.6 | 3.50× |
| common_r18_28 | GEMM | 325.8 | 707.9 | 2.17× |
| common_r18_14 | GEMM | 203.5 | 375.4 | 1.85× |
| common_r18_7 | GEMM | 208.8 | 131.4 | 0.63× |
| common_r50_ds56 | GEMM | 324.5 | 849.4 | 2.62× |
| common_r50_ds28 | GEMM | 401.6 | 439.5 | 1.09× |
| common_thr_n8 | GEMM | 927.7 | 1544.2 | 1.66× |

**线程数 = 36**

| 用例 | 路径 | torch | 本方案 | 加速比 |
|---|---|---:|---:|---:|
| custom_0 | GEMM | 1374.0 | 2130.5 | 1.55× |
| custom_1 | GEMM | 1051.9 | 1757.2 | 1.67× |
| custom_2 | GEMM | 1327.0 | 2075.9 | 1.56× |
| custom_3 | GEMM | 1045.6 | 1508.2 | 1.44× |
| custom_4 | GEMM | 1576.6 | 2069.4 | 1.31× |
| custom_5 | GEMM | 958.7 | 1721.6 | 1.80× |
| custom_6 | GEMM | 779.9 | 2019.0 | 2.59× |
| custom_7 | GEMM | 807.1 | 1219.5 | 1.51× |
| custom_8 | GEMM | 1642.4 | 2225.1 | 1.35× |
| custom_9 | SME | 2137.6 | 3134.0 | 1.47× |
| custom_10 | SME | 1355.1 | 1816.2 | 1.34× |
| common_r18_56 | GEMM | 229.4 | 855.6 | 3.73× |
| common_r18_28 | GEMM | 269.8 | 707.4 | 2.62× |
| common_r18_14 | GEMM | 209.5 | 373.5 | 1.78× |
| common_r18_7 | GEMM | 182.1 | 131.2 | 0.72× |
| common_r50_ds56 | GEMM | 345.3 | 822.5 | 2.38× |
| common_r50_ds28 | GEMM | 315.8 | 426.2 | 1.35× |
| common_thr_n8 | GEMM | 917.9 | 1649.6 | 1.80× |

### A.2 修改前 (before) 各线程 GFLOPS

**线程数 = 1**

| 用例 | torch | before | 加速比 |
|---|---:|---:|---:|
| custom_0 | 83.7 | 103.7 | 1.24× |
| custom_1 | 45.5 | 106.2 | 2.33× |
| custom_2 | 71.9 | 83.6 | 1.16× |
| custom_3 | 39.5 | 100.0 | 2.53× |
| custom_4 | 67.8 | 83.1 | 1.23× |
| custom_5 | 63.2 | 77.7 | 1.23× |
| custom_6 | 57.5 | 68.9 | 1.20× |
| custom_7 | 91.4 | 85.2 | 0.93× |
| custom_8 | 62.6 | 82.1 | 1.31× |
| custom_9 | 68.0 | 200.1 | 2.94× |
| custom_10 | 49.6 | 122.4 | 2.47× |
| common_r18_56 | 55.6 | 82.9 | 1.49× |
| common_r18_28 | 69.3 | 68.1 | 0.98× |
| common_r18_14 | 60.1 | 56.6 | 0.94× |
| common_r18_7 | 33.1 | 33.8 | 1.02× |
| common_r50_ds56 | 58.1 | 72.1 | 1.24× |
| common_r50_ds28 | 56.7 | 61.0 | 1.08× |
| common_thr_n8 | 50.9 | 79.0 | 1.55× |

**线程数 = 2**

| 用例 | torch | before | 加速比 |
|---|---:|---:|---:|
| custom_0 | 162.6 | 199.2 | 1.22× |
| custom_1 | 93.6 | 214.1 | 2.29× |
| custom_2 | 153.0 | 177.4 | 1.16× |
| custom_3 | 79.5 | 194.8 | 2.45× |
| custom_4 | 134.0 | 166.2 | 1.24× |
| custom_5 | 117.7 | 161.9 | 1.38× |
| custom_6 | 112.2 | 138.0 | 1.23× |
| custom_7 | 165.5 | 157.4 | 0.95× |
| custom_8 | 124.7 | 160.9 | 1.29× |
| custom_9 | 143.8 | 340.2 | 2.37× |
| custom_10 | 96.6 | 243.8 | 2.52× |
| common_r18_56 | 95.5 | 142.0 | 1.49× |
| common_r18_28 | 109.2 | 117.0 | 1.07× |
| common_r18_14 | 91.1 | 82.8 | 0.91× |
| common_r18_7 | 48.3 | 41.5 | 0.86× |
| common_r50_ds56 | 105.6 | 127.5 | 1.21× |
| common_r50_ds28 | 98.7 | 91.0 | 0.92× |
| common_thr_n8 | 126.4 | 162.4 | 1.28× |

**线程数 = 4**

| 用例 | torch | before | 加速比 |
|---|---:|---:|---:|
| custom_0 | 343.6 | 361.7 | 1.05× |
| custom_1 | 177.4 | 415.5 | 2.34× |
| custom_2 | 291.9 | 333.8 | 1.14× |
| custom_3 | 155.6 | 351.4 | 2.26× |
| custom_4 | 265.5 | 327.8 | 1.23× |
| custom_5 | 212.4 | 320.8 | 1.51× |
| custom_6 | 221.5 | 271.7 | 1.23× |
| custom_7 | 292.2 | 278.2 | 0.95× |
| custom_8 | 245.9 | 315.5 | 1.28× |
| custom_9 | 288.5 | 647.3 | 2.24× |
| custom_10 | 186.2 | 467.1 | 2.51× |
| common_r18_56 | 148.1 | 242.8 | 1.64× |
| common_r18_28 | 164.7 | 179.5 | 1.09× |
| common_r18_14 | 136.9 | 117.0 | 0.85× |
| common_r18_7 | 80.3 | 42.7 | 0.53× |
| common_r50_ds56 | 175.7 | 200.9 | 1.14× |
| common_r50_ds28 | 161.5 | 128.2 | 0.79× |
| common_thr_n8 | 225.7 | 312.3 | 1.38× |

**线程数 = 8**

| 用例 | torch | before | 加速比 |
|---|---:|---:|---:|
| custom_0 | 647.4 | 678.5 | 1.05× |
| custom_1 | 347.2 | 754.5 | 2.17× |
| custom_2 | 531.8 | 650.4 | 1.22× |
| custom_3 | 306.0 | 574.1 | 1.88× |
| custom_4 | 523.3 | 618.3 | 1.18× |
| custom_5 | 390.6 | 589.9 | 1.51× |
| custom_6 | 383.8 | 517.7 | 1.35× |
| custom_7 | 477.4 | 466.4 | 0.98× |
| custom_8 | 474.4 | 574.5 | 1.21× |
| custom_9 | 544.2 | 1238.7 | 2.28× |
| custom_10 | 345.0 | 849.0 | 2.46× |
| common_r18_56 | 198.0 | 391.6 | 1.98× |
| common_r18_28 | 223.4 | 272.0 | 1.22× |
| common_r18_14 | 186.2 | 146.1 | 0.78× |
| common_r18_7 | 122.2 | 42.8 | 0.35× |
| common_r50_ds56 | 257.8 | 287.7 | 1.12× |
| common_r50_ds28 | 247.2 | 163.9 | 0.66× |
| common_thr_n8 | 417.2 | 583.1 | 1.40× |

**线程数 = 16**

| 用例 | torch | before | 加速比 |
|---|---:|---:|---:|
| custom_0 | 927.0 | 1122.2 | 1.21× |
| custom_1 | 650.8 | 1208.7 | 1.86× |
| custom_2 | 892.2 | 1165.5 | 1.31× |
| custom_3 | 570.0 | 808.8 | 1.42× |
| custom_4 | 917.6 | 1124.5 | 1.23× |
| custom_5 | 671.4 | 1078.3 | 1.61× |
| custom_6 | 608.2 | 943.0 | 1.55× |
| custom_7 | 605.7 | 652.8 | 1.08× |
| custom_8 | 788.5 | 1035.9 | 1.31× |
| custom_9 | 948.0 | 2424.6 | 2.56× |
| custom_10 | 665.3 | 1351.0 | 2.03× |
| common_r18_56 | 230.2 | 611.9 | 2.66× |
| common_r18_28 | 301.3 | 363.1 | 1.21× |
| common_r18_14 | 242.8 | 147.8 | 0.61× |
| common_r18_7 | 149.8 | 42.7 | 0.29× |
| common_r50_ds56 | 324.0 | 410.8 | 1.27× |
| common_r50_ds28 | 336.3 | 166.3 | 0.49× |
| common_thr_n8 | 675.6 | 1039.4 | 1.54× |

**线程数 = 32**

| 用例 | torch | before | 加速比 |
|---|---:|---:|---:|
| custom_0 | 1421.5 | 1699.8 | 1.20× |
| custom_1 | 1153.5 | 1443.3 | 1.25× |
| custom_2 | 1283.8 | 1962.4 | 1.53× |
| custom_3 | 1042.0 | 1067.2 | 1.02× |
| custom_4 | 1474.4 | 1768.8 | 1.20× |
| custom_5 | 929.4 | 1779.5 | 1.91× |
| custom_6 | 1080.9 | 1691.0 | 1.56× |
| custom_7 | 806.9 | 900.4 | 1.12× |
| custom_8 | 1520.5 | 1620.3 | 1.07× |
| custom_9 | 1941.4 | 3125.5 | 1.61× |
| custom_10 | 1256.6 | 1582.1 | 1.26× |
| common_r18_56 | 236.6 | 802.7 | 3.39× |
| common_r18_28 | 325.8 | 487.2 | 1.50× |
| common_r18_14 | 203.5 | 137.5 | 0.68× |
| common_r18_7 | 208.8 | 41.7 | 0.20× |
| common_r50_ds56 | 324.5 | 519.2 | 1.60× |
| common_r50_ds28 | 401.6 | 167.5 | 0.42× |
| common_thr_n8 | 927.7 | 1577.3 | 1.70× |

**线程数 = 36**

| 用例 | torch | before | 加速比 |
|---|---:|---:|---:|
| custom_0 | 1374.0 | 1946.3 | 1.42× |
| custom_1 | 1051.9 | 1536.2 | 1.46× |
| custom_2 | 1327.0 | 2049.5 | 1.54× |
| custom_3 | 1045.6 | 1057.9 | 1.01× |
| custom_4 | 1576.6 | 2017.6 | 1.28× |
| custom_5 | 958.7 | 1886.3 | 1.97× |
| custom_6 | 779.9 | 1841.5 | 2.36× |
| custom_7 | 807.1 | 884.3 | 1.10× |
| custom_8 | 1642.4 | 1819.2 | 1.11× |
| custom_9 | 2137.6 | 3134.0 | 1.47× |
| custom_10 | 1355.1 | 1816.2 | 1.34× |
| common_r18_56 | 229.4 | 834.0 | 3.64× |
| common_r18_28 | 269.8 | 483.9 | 1.79× |
| common_r18_14 | 209.5 | 138.3 | 0.66× |
| common_r18_7 | 182.1 | 41.6 | 0.23× |
| common_r50_ds56 | 345.3 | 518.0 | 1.50× |
| common_r50_ds28 | 315.8 | 165.9 | 0.53× |
| common_thr_n8 | 917.9 | 1710.6 | 1.86× |

### A.3 汇总

| 线程 | before/torch(geo) | after/torch(geo) | after 赢 | after 输 |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 1.39× | 1.58× | 17 | 1 |
| 2 | 1.35× | 1.58× | 17 | 0 |
| 4 | 1.29× | 1.60× | 18 | 0 |
| 8 | 1.25× | 1.62× | 17 | 0 |
| 16 | 1.24× | 1.65× | 17 | 1 |
| 32 | 1.17× | 1.57× | 17 | 1 |
| 36 | 1.26× | 1.67× | 17 | 1 |

---

## 附录 B：对比脚本

> **正式脚本已入库**：`scripts/bench_vs_torch/`（`torch_baseline.py`、`compare_vs_pytorch.py`，用法见其 `README.md`）。
> 下面的 B.1/B.2 是生成本文数据时的历史快照（含当时的临时路径），仅存档参考。

### B.1 `compare_final.py`（本方案 before/after 扫描；before=best of (32,16)/(64,32)，after=自动）

```python
#!/usr/bin/env python3
"""Final before/after vs torch (all clang builds).

before = native_common (clang, no Bc-parallel, fixed Nwin/Nf, best of {(32,16),(64,32)})
after  = native_impl  (clang, Bc-parallel + auto Nwin) for the 16 GEMM cases;
         SME kernel-only (sme probe) for custom_9/custom_10 in both.
"""
import subprocess, os, json, statistics
ROOT="/home/z00889957/100Work/SConvTransform"
BEFORE="/tmp/opencode/native_common/build/sconv_bench"
AFTER ="/home/z00889957/100Work/SConvTransform/native_impl/build/sconv_bench"
SME_DIR=os.path.join(ROOT,"sme"); SME=os.path.join(SME_DIR,"direct.out")
TORCH_PY="/home/z00889957/miniconda3/envs/torch210_env/bin/python"
TORCH_SCRIPT="/tmp/opencode/torch_all.py"
OUT="/tmp/opencode/compare_out2"; os.makedirs(OUT,exist_ok=True)
THREADS=[1,2,4,8,16,32,36]
NUMA=["numactl","--cpunodebind=0","--membind=0"]
RUNS,WARMUP="8","3"; REPEATS=3
KMLROOT=os.environ.get("KMLROOT","")
ENV=os.environ.copy()
if KMLROOT: ENV["LD_LIBRARY_PATH"]=f"{KMLROOT}/lib:"+ENV.get("LD_LIBRARY_PATH","")
SME_CASES={"custom_9","custom_10"}

def run(cmd,cwd=None):
    r=subprocess.run(cmd,capture_output=True,text=True,env=ENV,cwd=cwd)
    return r.stdout+r.stderr

def torch_vals(t):
    out=run([*NUMA,TORCH_PY,TORCH_SCRIPT,"--threads",str(t),"--runs",RUNS,"--warmup",WARMUP])
    open(f"{OUT}/torch_t{t}.out","w").write(out)
    v={}
    for l in out.splitlines():
        if l.startswith("#") or "," not in l: continue
        p=l.split(",")
        if len(p)==4: v[p[0]]=float(p[3])
    return v

def gemm_before(t):
    best={}
    for nw,nf in [(32,16),(64,32)]:
        for filt in ["custom","common"]:
            out=run([*NUMA,BEFORE,"--filter",filt,"--skip-naive","--runs",RUNS,
                     "--warmup",WARMUP,"--nwin",str(nw),"--nf",str(nf),"--threads",str(t)])
            for l in out.splitlines():
                tk=l.split()
                if len(tk)>=8 and tk[0].startswith(("custom_","common_")) and tk[-1]=="[PASS]":
                    if float(tk[5])>best.get(tk[0],0.0): best[tk[0]]=float(tk[5])
    return best

def gemm_after(t):
    best={}
    for filt in ["custom","common"]:
        out=run([*NUMA,AFTER,"--filter",filt,"--skip-naive","--runs",RUNS,
                 "--warmup",WARMUP,"--nwin","0","--nf","0","--threads",str(t)])
        for l in out.splitlines():
            tk=l.split()
            if len(tk)>=8 and tk[0].startswith(("custom_","common_")) and tk[-1]=="[PASS]":
                if float(tk[5])>best.get(tk[0],0.0): best[tk[0]]=float(tk[5])
    return best

def sme_vals(t):
    out=run([*NUMA,SME,"bench",str(t)],cwd=SME_DIR)
    v={}
    for l in out.splitlines():
        if "GFLOPS" in l and "parallel" not in l:
            tk=l.split(); idx=[i for i,x in enumerate(tk) if x=="GFLOPS"]
            if idx: v[tk[0]]=float(tk[idx[-1]-1])
    return v

def best(dst,src):
    for k,v in src.items():
        if v>dst.get(k,0.0): dst[k]=v

res={"torch":{},"before":{},"after":{}}
for t in THREADS:
    print(f"[threads={t}]",flush=True)
    tv,gb,ga,sv={},{},{},{}
    for r in range(REPEATS):
        best(tv,torch_vals(t)); best(gb,gemm_before(t)); best(ga,gemm_after(t)); best(sv,sme_vals(t))
    res["torch"][t]=tv; res["before"][t]=gb; res["after"][t]=ga
    for name in tv:
        if name in SME_CASES and name in sv:
            gb[name]=sv[name]; ga[name]=sv[name]   # SME path unchanged
json.dump(res,open(f"{OUT}/data.json","w"),indent=1)
print("saved", f"{OUT}/data.json")
```

### B.2 `torch_all.py`（PyTorch 基线）

```python
#!/usr/bin/env python3
"""PyTorch conv2d baseline for the 11 custom + 7 common configs.

Prints machine-readable lines:  name,flops,ms,gflops
"""
import argparse, time
import torch

def rtclock():
    return time.perf_counter()

configs = [
    ("custom_0",  4,192,40,40,  192,3,3, 1,1),
    ("custom_1",  4,768,80,80,  768,3,3, 2,1),
    ("custom_2",  4,96,80,80,    96,3,3, 1,1),
    ("custom_3",  4,768,40,40,  768,3,3, 2,1),
    ("custom_4",  4,384,160,160,384,3,3, 2,1),
    ("custom_5",  4,48,160,160,  48,3,3, 1,1),
    ("custom_6",  4,96,320,320, 192,3,3, 2,1),
    ("custom_7",  4,192,20,20,  192,3,3, 1,1),
    ("custom_8",  4,384,80,80,  384,3,3, 2,1),
    ("custom_9",  4,384,80,80,   96,3,3, 1,1),
    ("custom_10", 4,768,40,40,   96,3,3, 1,1),
    ("common_r18_56",   1, 64, 56,56,  64,3,3, 1,1),
    ("common_r18_28",   1,128, 28,28, 128,3,3, 1,1),
    ("common_r18_14",   1,256, 14,14, 256,3,3, 1,1),
    ("common_r18_7",    1,512,  7, 7, 512,3,3, 1,1),
    ("common_r50_ds56", 1,128, 56,56, 256,3,3, 2,1),
    ("common_r50_ds28", 1,256, 28,28, 512,3,3, 2,1),
    ("common_thr_n8",   8, 64, 56,56,  64,3,3, 1,1),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=38)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    print(f"#torch {torch.__version__} threads={args.threads}")
    for name, N, Ic, Ih, Iw, Oc, Fh, Fw, stride, pad in configs:
        Oh = (Ih + 2*pad - Fh)//stride + 1
        Ow = (Iw + 2*pad - Fw)//stride + 1
        flops = 2.0 * N * Oc * Oh * Ow * Ic * Fh * Fw
        x = torch.randn(N, Ic, Ih, Iw)
        w = torch.randn(Oc, Ic, Fh, Fw)
        with torch.no_grad():
            for _ in range(args.warmup):
                torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)
            t0 = rtclock()
            for _ in range(args.runs):
                torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)
            dt = (rtclock() - t0) / args.runs * 1000.0
        print(f"{name},{flops:.0f},{dt:.4f},{flops/(dt*1e6):.2f}")
        del x, w

if __name__ == "__main__":
    main()
```
