# SConv 独立 C++ 实现 (native_impl)

不依赖 MLIR/LLVM, 用纯 C++ 实现 SConv 的 CSA 分析 + 分块 + 打包 + OpenBLAS 微内核。
方便在目标机器上快速编译运行, 验证算法实际性能。

## 构建

```bash
cd native_impl
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

### 依赖

| 依赖 | 必需? | 安装 |
|------|:---:|------|
| C++17 编译器 | 是 | gcc 9+ / clang 10+ |
| CMake ≥ 3.12 | 是 | `apt install cmake` |
| OpenBLAS | 否 (有 fallback) | macOS: `brew install openblas`; Linux: `apt install libopenblas-dev` |
| OpenMP | 否 (可选, 未来多核) | `apt install libomp-dev` |

### aarch64 特殊编译

在 ARM 机器上, CMake 自动检测并启用 SVE/SME 编译选项:
```
-march=armv9-a+sve+sme -O3
```

如果编译器不支持 SME, 改为:
```
cmake .. -DCMAKE_CXX_FLAGS="-march=armv8.2-a+sve -O3"
```

## 运行

```bash
# 跑全部配置 (11 个自定义 + 2 个小卷积)
./sconv_bench

# 只跑小卷积 (快速验证正确性)
./sconv_bench --filter small

# 只跑自定义大卷积
./sconv_bench --filter custom

# 控制迭代次数和 warmup
./sconv_bench --runs 3 --warmup 1 --filter custom

# 用 NUMA 绑定 (目标服务器)
numactl --cpunodebind=0 --membind=0 ./sconv_bench --filter custom
```

### 输出示例

```
SConv Benchmark (Nwin=16, Nf=8, OpenBLAS=YES)
warmup=1, runs=1

Conv                  FLOPs    Naive(ms)    SConv(ms)   Naive GF   SConv GF    Speedup
----                   ----         ----         ----       ----       ----       ----
custom_0             4.25G     1773.198      351.399       2.39      12.09      5.05x  [PASS]
custom_1            67.95G    32698.600     5683.100       2.08      11.96      5.75x  [PASS]
...

Correctness: 11/11 passed
```

## 文件结构

```
native_impl/
├── CMakeLists.txt        # 构建配置 (自动检测 OpenBLAS, SVE/SME flags)
└── src/
    ├── sconv_csa.h       # CSA 分析 (从 lib/CSA.cpp 移植: IS/WS 代价模型, Nc/K2/K3 推导)
    └── sconv.cpp        # 主程序: naive conv + SConv conv + 正确性验证 + 性能计时
```

## 算法流程

```
1. CSA 分析 (sconv_csa.h)
   输入: 卷积形状 + 机器缓存参数 + 微内核形状 (Nwin, Nf)
   输出: IS/WS 调度, Nc (L1), K2 (L2), K3 (L3/HBM)
   方法: 半值启发式 + 代价模型比较 IS vs WS

2. 分块 (sconv.cpp: conv2d_sconv)
   外层: batch → 输入通道 (step Nc) → 空间 tile → 滤波器 tile
   内层: 空间 (step Nwin) → 滤波器 (step Nf)

3. 打包 (sconv.cpp: conv2d_sconv 内联)
   输入: 从原始 tensor 逐元素提取, 填入 [K, Nwin] packedA (含复制)
   滤波器: 从原始 weight 重排, 填入 [K, Nf] packedB (无复制)

4. 微内核 (sconv.cpp: conv2d_sconv 内联)
   C[Nf, Nwin] += B[K, Nf]^T × A[K, Nwin]
   用 cblas_sgemm(RowMajor, Trans, NoTrans, Nf, Nwin, K, ...) 实现

5. 解包: 从 C[Nf, Nwin] 写回 output[N, Oc, Oh, Ow]
```

## 与 MLIR 版本的区别

| | MLIR 版本 | C++ 版本 |
|---|---|---|
| CSA | `lib/CSA.cpp` | `src/sconv_csa.h` (移植) |
| 分块 | `scf::tileUsingSCF` (MLIR IR 变换) | C++ for 循环 |
| 打包 | `linalg.generic` + affine map | 手动 for 循环 + `cblas_sgemm` |
| 微内核 | `lower.to_blas` → `LLVM::CallOp` | 直接调 `cblas_sgemm` |
| 边界处理 | `linalg::splitOp` 递归 | `std::min` 裁剪 |
| 多重打包 | `filterMultipackingOpt` / `inputMultipackingOpt` | 未实现 (每次重新打包) |
| 并行 | 无 | 无 (预留 OpenMP) |
| 正确性 | `numpy.allclose` | `std::abs(diff) < tol` |

> C++ 版本是 MLIR 版本的简化, 省略了 multipacking 和精细边界处理,
> 但核心算法 (CSA + tiling + packing + BLAS microkernel) 一致。

## 在目标机器 (OpenEuler aarch64) 上运行

```bash
# 安装依赖
sudo dnf install cmake gcc-c++ openblas-devel

# 编译
cd native_impl
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# 运行 (绑定 NUMA 0)
numactl --cpunodebind=0 --membind=0 ./sconv_bench --filter custom --runs 3 --warmup 2
```

如果机器有 HBM (通过 memkind):
```bash
sudo dnf install memkind-devel
# 编译时加 memkind 支持 (需修改 CMakeLists.txt)
cmake .. -DCMAKE_CXX_FLAGS="-DUSE_HBM" -DCMAKE_EXE_LINKER_FLAGS="-lmemkind"
```
