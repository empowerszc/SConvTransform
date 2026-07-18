# SConvTransform 性能测试报告

> 测试日期: 2025-07-18
> 测试平台: Apple Mac mini M4 (ARM64, SME)
> LLVM 版本: 19.1.7
> 报告: SConvTransform (CSA + tiling + packing + OpenBLAS microkernel) vs Native Lowering

---

## 1. 测试环境

| 项目 | 规格 |
|------|------|
| 机器 | Apple Mac mini (2024) |
| CPU | Apple M4 (ARMv9.4-A, 10 核) |
| 矩阵扩展 | SME (Scalable Matrix Extension), SVE2 |
| L1/L2/L3 缓存 | 192KiB / 16384KiB shared / — |
| 内存 | 32GB |
| 操作系统 | macOS (Darwin) |
| LLVM | 19.1.7 (预编译 release) |
| OpenBLAS | 0.3.32 (Homebrew, 通用 ARM64, 无 SME 专用微内核) |
| 编译器 | Apple Clang 21.0 |
| 构建类型 | Release + Assertions ON |

## 2. 测试方法

### 对比方案

| 方案 | 说明 |
|------|------|
| **Baseline** | Native Lowering: `linalg.conv_2d_nchw_fchw` 经 `mlir-opt` 标准 pass 直接降为标量 affine 循环 (无优化) |
| **SConv+BLAS** | SConvTransform: CSA 分块分析 → 两级 tiling → packing/multipacking → `lower.to_blas` 替换微内核为 OpenBLAS `sgemm` 调用 |

### 计时方式

- 计时由 payload 内部的 `rtclock()` (基于 `clock_gettime(CLOCK_MONOTONIC)`) 实现
- 每个卷积配置执行流程: **warmup 预热 (不计时) → 正式计时迭代**
- 报告的时间为**单次迭代平均时间** = 总计时 / 迭代次数
- GFLOPS = (2 × OH × OW × DO × CI × HK × WK) / 单次时间
- 正确性: baseline 与 SConv+BLAS 输出经 numpy `allclose(rtol=1e-3)` 比对

### 测试参数

| 卷积集 | warmup | runs | 说明 |
|--------|:---:|:---:|------|
| 小卷积采样 (5 个) | 5 | 10 | 计算量 < 0.03 GFLOP, 迭代快, 多次取均值 |
| 自定义卷积 (11 个) | 1 | 1 | 计算量 1~68 GFLOP, baseline 单次即数秒~百秒 |

> 注: 自定义大卷积的 baseline (标量循环) 极慢 (最高 99 秒/次), 多次迭代不现实。warmup=1 保证 cache 预热后计时。

## 3. 测试配置

### 3.1 自定义卷积 (11 个)

| ID | Input Shape | Weight Shape | Stride | Pad | FLOPs |
|----|-------------|-------------|:---:|:---:|---:|
| custom_0 | 4×192×40×40 | 192×192×3×3 | 1×1 | 1×1 | 4.25G |
| custom_1 | 4×768×80×80 | 768×768×3×3 | 2×2 | 1×1 | 67.95G |
| custom_2 | 4×96×80×80 | 96×96×3×3 | 1×1 | 1×1 | 4.25G |
| custom_3 | 4×768×40×40 | 768×768×3×3 | 2×2 | 1×1 | 16.99G |
| custom_4 | 4×384×160×160 | 384×384×3×3 | 2×2 | 1×1 | 67.95G |
| custom_5 | 4×48×160×160 | 48×48×3×3 | 1×1 | 1×1 | 4.25G |
| custom_6 | 4×96×320×320 | 192×96×3×3 | 2×2 | 1×1 | 33.97G |
| custom_7 | 4×192×20×20 | 192×192×3×3 | 1×1 | 1×1 | 1.06G |
| custom_8 | 4×384×80×80 | 384×384×3×3 | 2×2 | 1×1 | 16.99G |
| custom_9 | 4×384×80×80 | 96×384×3×3 | 1×1 | 1×1 | 16.99G |
| custom_10 | 4×768×40×40 | 96×768×3×3 | 1×1 | 1×1 | 8.49G |

### 3.2 小卷积采样 (ConvBench, 5 个)

| ID | Input Shape | Weight Shape | Stride | FLOPs |
|----|-------------|-------------|:---:|---:|
| conv_1800 | 1×18×14×14 | 144×18×3×3 | 2×2 | 2.3M |
| conv_1812 | 1×16×14×14 | 128×16×3×3 | 2×2 | 1.8M |
| conv_3557 | 1×48×10×10 | 64×48×2×2 | 2×2 | 0.6M |
| conv_3565 | 1×96×10×10 | 128×96×2×2 | 2×2 | 2.5M |
| conv_3580 | 1×48×10×10 | 64×48×3×3 | 2×2 | 1.4M |

## 4. 正确性验证

| 卷积集 | 数量 | 通过 | 失败 |
|--------|:---:|:---:|:---:|
| 自定义卷积 | 11 | 11 | 0 |
| 小卷积采样 | 5 | 5 | 0 |
| **合计** | **16** | **16** | **0** |

> 全部 16 个卷积配置, SConv+BLAS 输出与 native lowering 基线逐元素数值一致 (rtol=1e-3)。

## 5. 性能结果

### 5.1 自定义卷积 (11 个, warmup=1, runs=1)

| 卷积 | FLOPs | Baseline (ms) | SConv+BLAS (ms) | Baseline GFLOPS | SConv+BLAS GFLOPS | 加速比 |
|------|------:|---:|---:|---:|---:|---:|
| custom_0 | 4.25G | 6,265.8 | 77.8 | 0.17 | 13.65 | 80.6x |
| custom_1 | 67.95G | 98,954.0 | 1,127.4 | 0.17 | 15.07 | **87.8x** |
| custom_2 | 4.25G | 6,199.6 | 90.4 | 0.17 | 11.74 | 68.5x |
| custom_3 | 16.99G | 24,226.1 | 285.5 | 0.18 | 14.87 | **84.9x** |
| custom_4 | 67.95G | 98,559.7 | 1,177.2 | 0.17 | 14.43 | 83.7x |
| custom_5 | 4.25G | 5,959.4 | 115.3 | 0.18 | 9.21 | 51.7x |
| custom_6 | 33.97G | 49,219.3 | 701.4 | 0.17 | 12.11 | 70.2x |
| custom_7 | 1.06G | 1,557.0 | 20.1 | 0.17 | 13.20 | 77.5x |
| custom_8 | 16.99G | 24,819.8 | 297.3 | 0.17 | 14.28 | 83.5x |
| custom_9 | 16.99G | 24,650.7 | 340.8 | 0.17 | 12.46 | 72.3x |
| custom_10 | 8.49G | 12,380.3 | 169.7 | 0.17 | 12.51 | 73.0x |
| **总计/均值** | — | **352,792ms** | **4,403ms** | — | — | **75.8x** |

> Baseline 累计 352.8 秒 → SConv+BLAS 累计 4.4 秒, 整体加速 **80 倍**。

### 5.2 小卷积采样 (5 个, warmup=5, runs=10)

| 卷积 | FLOPs | Baseline (ms) | SConv+BLAS (ms) | Baseline GFLOPS | SConv+BLAS GFLOPS | 加速比 |
|------|------:|---:|---:|---:|---:|---:|
| conv_1800 | 2.3M | 2.746 | 0.142 | 0.83 | 16.11 | 19.4x |
| conv_1812 | 1.8M | 2.117 | 0.119 | 0.85 | 15.23 | 17.9x |
| conv_3557 | 0.6M | 0.769 | 0.112 | 0.80 | 5.49 | 6.9x |
| conv_3565 | 2.5M | 3.665 | 0.391 | 0.67 | 6.29 | 9.4x |
| conv_3580 | 1.4M | 1.904 | 0.525 | 0.73 | 2.63 | 3.6x |
| **均值** | — | — | — | — | — | **11.4x** |

## 6. 分析与结论

### 6.1 正确性

全部 16 个卷积 (含 batch=4 大卷积和 batch=1 小卷积) 的 SConv+BLAS 输出与
native lowering 逐元素一致, 证明 SConvTransform 的 tiling + packing +
BLAS 微内核替换流水线在保证语义正确性的前提下完成了卷积优化。

### 6.2 性能加速

| 卷积规模 | FLOPs 范围 | 平均加速比 | SConv+BLAS GFLOPS |
|----------|-----------|:---:|:---:|
| 小卷积 (batch=1) | 0.6~2.5M | 11.4x | 2.6~16.1 |
| 大卷积 (batch=4) | 1~68G | 75.8x | 9.2~15.1 |

大卷积加速比远高于小卷积, 原因:
- 小卷积的 BLAS 调用开销 (函数调用 + 矩阵尺寸计算) 占比高, 数据量不足以充分利用缓存
- 大卷积的 tiling 后每个 tile 足够大, OpenBLAS `sgemm` 能高效利用 cache 和 SIMD

### 6.3 与论文结论对比

| 指标 | 论文 (Apple M4 SME) | 本测试 |
|------|:---:|:---:|
| 正确性 | 7922 卷积全通过 | 16 卷积全通过 ✅ |
| 跨架构 | M4 / Intel / Power10 | M4 ARM64 ✅ |
| Baseline vs SConv 加速 | — | 11x (小) ~ 88x (大) |
| SConv+BLAS 峰值占比 | median 15.5%, best 59.6% (502 GFLOPS) | ~3% (15 GFLOPS / 502) |

本测试 SConv+BLAS 仅达 ~15 GFLOPS (论文 ~78 GFLOPS), 差距来自:
1. **OpenBLAS 后端**: Homebrew 通用 ARM64 版无 SME 专用微内核; 论文使用 SME 优化的 OpenBLAS
2. **外层 tiling 未优化**: SConvTransform 当前版本的 macrokernel 仍有 repacking 开销
3. **测试迭代数**: 大卷积仅 1 次, 结果方差较大

### 6.4 在 SVE/SME 服务器上的预期

在支持 SVE512 + SME 的 OpenEuler aarch64 服务器上, 若使用 SVE/SME 优化的
OpenBLAS 构建, 预期:
- SConv+BLAS GFLOPS 可提升 3~5x (SME 矩阵指令加速)
- 加速比进一步提升 (baseline 不变, SConv 端更快)
- 部分大卷积有望接近论文报告的 50%~60% 峰值

## 7. 复现方法

```bash
# 环境设置
export LLVM_BUILD_DIR=/path/to/llvm-19.1.7
export SCONV_ROOT=/path/to/SConvTransform
export PYTHONUNBUFFERED=1

# 编译
cd $SCONV_ROOT && make configure && make build

# 跑自定义卷积 (大卷积, warmup=1, runs=1)
python3 -u scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/custom_conv.csv --num 100 \
  --runs 1 --warmup 1

# 跑小卷积采样 (多迭代取均值)
python3 -u scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/small_regular.csv --num 5 \
  --runs 10 --warmup 5
```
