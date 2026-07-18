# ConvBench 测试使用指南

## 概述

ConvBench 是 SConvTransform 的卷积基准测试套件, 包含:
- **正确性测试**: 对比 native lowering 与 SConv+OpenBLAS 的输出是否数值一致
- **性能测试**: 对比 native lowering 与 SConv+OpenBLAS 的 GFLOPS/s 加速比

## 前置条件

1. **已编译 SConvTransform** (`build/bin/sconv-opt` 存在)
2. **LLVM 19.1.x** 预编译包 (含 `mlir-opt` 和 `mlir-cpu-runner`)
3. **OpenBLAS** (`brew install openblas` / `apt install libopenblas-dev`)
4. **Python 3 + numpy** (`pip install numpy`)

## 文件结构

```
scripts/convbench/
├── run_convbench.py        # 测试运行脚本 (主入口)
├── gen_payload.py          # 从 CSV 生成 MLIR payload
├── payloads/
│   ├── correctness_payload.mlir   # 正确性测试模板 (含 main + printMemrefF32)
│   └── performance_payload.mlir   # 性能测试模板 (含 main + rtclock + printFlops)
├── runtime/
│   ├── rtclock.c           # rtclock() 计时函数 (printFlops 依赖)
│   └── sgemm_blas_kernel.c # BLAS 微内核包装 (调用 cblas_sgemm)
├── samples/
│   └── small_regular.csv  # 5 个最小 regular 卷积采样集
└── datasets/               # 完整 ConvBench 数据集 (7922 卷积)
    ├── regular_timm+pt2.csv
    ├── elementwise_timm+pt2.csv
    └── rectangle_timm_pt2.csv
```

## 快速开始

```bash
# 1. 设置环境变量
export LLVM_BUILD_DIR=/path/to/LLVM-19.1.7
export SCONV_ROOT=/path/to/SConvTransform

# 2. (首次) 编译 SConvTransform
cd $SCONV_ROOT && make configure && make build

# 3. 运行采样测试 (5 个小卷积, 正确性 + 性能)
python3 scripts/convbench/run_convbench.py

# 输出示例:
#   Building runtime libraries...
#   ============ Correctness Test: 5 convolutions ============
#     PASS conv_1800.mlir
#     PASS conv_1812.mlir
#     ...
#   Result: 5 passed, 0 failed
#   ============ Performance Test: 5 convolutions ============
#     conv_1800       0.82      17.02   20.63x
#     ...
#   Average speedup: 12.96x (5 convs)
```

## 命令行选项

```bash
# 用采样集, 仅跑正确性
python3 scripts/convbench/run_convbench.py --correctness-only

# 用完整 regular 数据集, 取前 20 个
python3 scripts/convbench/run_convbench.py \
  --csv scripts/convbench/datasets/regular_timm+pt2.csv --num 20

# 用 elementwise (1x1) 数据集
python3 scripts/convbench/run_convbench.py \
  --csv scripts/convbench/datasets/elementwise_timm+pt2.csv --num 10
```

## 运行时库说明

`run_convbench.py` 首次运行时会自动编译两个运行时库到 `build/convbench_runtime/`:

| 库 | 源码 | 作用 |
|----|------|------|
| `librtclock.{dylib,so}` | `runtime/rtclock.c` | `rtclock()` 高精度计时 (clock_gettime) |
| `libsgemm_blas_kernel.{dylib,so}` | `runtime/sgemm_blas_kernel.c` | 包装 `cblas_sgemm` 为 SConv 的 `sgemm_blas_kernel` 接口 |

### sgemm_blas_kernel 接口

SConv 的 `lower.to_blas` 操作创建对 `sgemm_blas_kernel` 的调用:

```c
int sgemm_blas_kernel(long m, long n, long k, float alpha,
                      float *A, float *B, float *C, long ldc);
```

- **A** = packed input tile [K, M] (row-major, lda = M)
- **B** = packed filter tile [K, N] (row-major, ldb = N)
- **C** = output tile [N, M] (row-major, ldc given)
- 计算: `C = B^T * A + C` (accumulate, beta = 1.0)

在 CBLAS 中等价于:
```c
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            n, m, k, alpha, B, n, A, m, 1.0f, C, ldc);
```

## 测试流程

```
对每个卷积 conv_N.mlir:

1. Baseline:
   mlir-opt conv_N.mlir  --one-shot-bufferize ... --llvm-legalize-for-export
   mlir-cpu-runner conv_N.llvm.mlir -e main -shared-libs=...
   → 捕获输出 (memref 数据 + GFLOPS/s)

2. SConv+BLAS:
   sconv-opt -transform=sconv-blas.mlir conv_N.mlir
   mlir-opt conv_N.tf.mlir --one-shot-bufferize ... --llvm-legalize-for-export
   mlir-cpu-runner conv_N.tf.llvm.mlir -e main -shared-libs=...
   → 捕获输出 (memref 数据 + GFLOPS/s)

3. 正确性: numpy.allclose(baseline_data, sconv_data, rtol=1e-3)
4. 性能: speedup = sconv_gflops / baseline_gflops
```

## 预期结果 (Apple M4, ARM64)

- **正确性**: 全部 PASS (输出数值一致)
- **性能**: 相比 native scalar lowering, SConv+BLAS 平均 **5-20x 加速**
  (小卷积) 或 **50-88x 加速** (大卷积, batch=4)

> 论文使用 SME 优化的 OpenBLAS 构建, 达到 median 15.5% / best 59.6% 峰值。
> 要复现论文性能, 需用 SME 后端的 OpenBLAS。

## 在 OpenEuler aarch64 服务器上部署

### 1. 安装依赖

```bash
# OpenEuler 包管理
sudo dnf install cmake ninja-build clang lld openblas-devel python3-numpy

# 或从源码安装 OpenBLAS (若需 SVE/SME 优化)
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make TARGET=ARMV8 DYNAMIC_ARCH=1 USE_OPENMP=1 -j$(nproc)
sudo make PREFIX=/opt/openblas install
```

### 2. 获取 LLVM 19.1.7

```bash
# 从 GitHub releases 下载预编译包
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/LLVM-19.1.7-aarch64-linux-gnu.tar.xz
tar xf LLVM-19.1.7-aarch64-linux-gnu.tar.xz
export LLVM_BUILD_DIR="$(pwd)/LLVM-19.1.7-aarch64-linux-gnu"
```

### 3. 编译 SConvTransform

```bash
git clone -b dev https://github.com/empowerszc/SConvTransform.git
cd SConvTransform
export SCONV_ROOT="$(pwd)"
export LLVM_BUILD_DIR=/path/to/LLVM-19.1.7-aarch64-linux-gnu
make configure && make build
```

### 4. 运行自定义卷积测试

```bash
# 用户自定义 11 个大卷积
python3 scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/custom_conv.csv --num 100 --runs 1
```

### 5. 如需 SVE/SME 优化的 OpenBLAS

默认系统 OpenBLAS 可能不含 SVE/SME 指令优化。要获得更高性能:

```bash
# 编译含 SVE 优化的 OpenBLAS
cd OpenBLAS
make TARGET=ARMV8SVE DYNAMIC_ARCH=1 -j$(nproc)
sudo make PREFIX=/opt/openblas-sve install

# 运行时指定库路径
export LD_LIBRARY_PATH=/opt/openblas-sve/lib:$LD_LIBRARY_PATH
# run_convbench.py 会自动检测 /opt/homebrew 或 /usr 下的 OpenBLAS,
# 如需自定义路径, 修改 runtime/sgemm_blas_kernel.c 的编译参数:
#   clang -shared -o build/convbench_runtime/libsgemm_blas_kernel.so \
#     scripts/convbench/runtime/sgemm_blas_kernel.c \
#     -I/opt/openblas-sve/include -L/opt/openblas-sve/lib -lopenblas
```

### 注意事项 (Linux vs macOS)

| 差异 | macOS | Linux (OpenEuler) |
|------|-------|-------------------|
| 动态库后缀 | `.dylib` | `.so` (脚本自动检测) |
| JIT 运行器 | `mlir-cpu-runner` | `mlir-cpu-runner` |
| OpenBLAS 路径 | `/opt/homebrew/opt/openblas` | `/usr` 或 `/opt/openblas` |
| 编译器 | Apple Clang | GCC 或 Clang |
| lld 链接器 | 不可用 (用 Apple ld) | 需安装 `lld` 或跳过 |
