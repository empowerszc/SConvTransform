# SConvTransform: 论文解读与构建依赖分析

> 论文: *Using MLIR Transform to Design Sliced Convolution Algorithm*
> (Victor Ferrari, Lucas Alvarenga, Gustavo Leite, Marcio Pereira, Guido Araujo;
> 预印本, 2025-10-30, Campinas/SP, Brazil)
> 仓库: SConvTransform — MLIR Transform Dialect 扩展

---

## 一、论文详解

### 1.1 研究目标与定位

SConvTransform 是 MLIR **Transform Dialect (变换方言)** 的一个扩展, 目标是把
`linalg.conv_2d_nchw_fchw` (NCHW/FCHW 布局的 2D 卷积) **完全以声明式方式**降低
(lowering) 为一个**分块 + 打包 + 调用微内核 (microkernel)** 的循环嵌套。

它在前作 SConv [Ferrari et al., ACM TACO 2023] 的基础上做了根本性改造:

| 维度 | 旧 SConv (2023) | SConvTransform (2025, 本文) |
|------|----------------|------------------------------|
| 实现形式 | 命令式 C++ rewrite patterns, 针对特定后端硬编码 | **声明式 MLIR Transform Dialect 操作** `transform.structured.sconv` |
| 复用性 | 低, 难演进 | 高, 可组合, 可分析 |
| 目标 | 单一架构 | 跨架构 (Apple M4/SME, Intel AVX512, IBM Power10/MMA) |
| 微内核 | 自带生成 | 复用 OpenBLAS 的 sgemm 微内核 |

核心理念: **把卷积优化的算法逻辑下沉到编译器 IR 层**, 让"分块/打包/调度"这些
原本库内部的策略变成 Transform Dialect 中可观察、可替换、可组合的变换。

### 1.2 核心概念

#### 1.2.1 CSA (Convolution Slicing Analysis, 卷积切片分析)

CSA 是一个**静态代价模型驱动的缓存分块分析**, 输入三类信息:

- **ConvInfo**: 卷积形状 (输入通道 `Ic`、输入宽 `Iw`、输出高/宽 `Oh/Ow`、核 `Fh/Fw`、
  滤波器数 `Oc`、stride/dilation)。
- **ArchInfo**: 架构参数 (L1/L2/L3 缓存大小、各级延迟、cache line 大小)。
- **mKInfo**: 微内核形状 `Nwin × Nf` (每次微内核调用处理的窗口数与滤波器数,
  如 Power10 MMA 为 16×8)。

输出一个 **CSAStrategy**:

- `schedule`: IS (Input Stationary, 输入驻留) 或 WS (Weight Stationary, 权重驻留) —
  通过代价模型比较两种策略的内存流量择优。
- `Nc`: L1 级输入通道分块大小 (使单个 input+filter+output tile 同时放进 L1)。
- `K2`, `K3`: L2/L3 级沿"非归约维度"(窗口数与滤波器数)的分块数; 两种调度下 K2/K3
  对应的维度互换。
- `R_Nc, R_K2, R_K3`: 各分块维度的余数 (dimension mod tile size), 非零即表示存在
  边界 (edge) case。

#### 1.2.2 五层宏内核 (Macrokernel) 结构

CSO (Convolution Slicing Optimization) 把卷积展开为围绕微内核的五层循环 (论文 Fig.3):

```
Layer5: Ic 通道分块 (Nc, 由 L1 决定)
Layer4: K3 个窗口 tile (L3 驱动)
Layer3: K2 个滤波器 tile (L2 驱动)
Layer2: 内层 K2 × Nf (每个 tile 的滤波器组)
Layer1: 内层 K3 × Nwin (每个 tile 的窗口组)
  └── Microkernel: Nwin × Nf 外积
```

调度策略 IS/WS 决定 Layer3/Layer4 中"输入 tile"与"滤波器 tile"谁驻留在哪一级缓存。

### 1.3 SConvTransform 五阶段流水线

论文 Section 5 + Fig.4 把 SConvOp 的 `apply()` 分为 5 个 stage:

#### Stage 1 — 归一化 (Normalization)
- **1a Generalize**: 把 `linalg.conv_2d_nchw_fchw` 泛化为 `linalg.generic`。
- **1b Linearization**: 把 H×W 空间维度**塌缩 (collapse)** 为单一线性维度
  (因为 SConv 按窗口序列而非独立行列处理; MLIR 的 tiler 对多维独立分块支持有限)。
- **1c CSA**: 运行卷积切片分析, 得到 schedule + (Nc, K2, K3) 及其余数。

#### Stage 2 — 边界处理 (Edge Splits)
处理两类边界:
- **2a 结构性边界**: `Oh×Ow` 不能被 `Nwin` 整除, 或 `Oc` 不能被 `Nf` 整除 →
  把卷积 split 成 main + remainder 两个 `linalg.generic`。
- **2b CSA 余数边界**: `K2/K3/Nc` 产生非零余数 → 递归地按 K2 → K3 → Nc
  顺序 split, 每个 split 出来的 kernel 各自独立走完整 tiling+packing 流水线。

split 用 `linalg::splitOp` 实现 (沿某一维度切成两段, 返回两个 TilingInterface op)。

#### Stage 3 — 两级分块 (Two-level Tiling)
- **3a 外层 tiling**: 用 `scf::tileUsingSCF` 按 CSA 的 (Nc, K2, K3) 切宏块。
- **3b 内层 tiling**: 再按 (Nwin, Nf) 切微块, 形成微内核调用点。
- **3c 循环修正**: MLIR 的 `tileUsingSCF` 在 IS 调度下 `setInterchange` 对**内层**
  不生效 (论文 5.7 节指出的 Transform Dialect 缺陷), 因此代码里手写了
  `swapInductionVars()` 显式交换归纳变量与循环界, 重排成期望的嵌套顺序。
  另外对塌缩后的线性空间维度做了 affine 表达式重写 (`promoteOpsOfTile` /
  `createLinearizedAffineApply`), 以正确恢复 2D 行列索引。

#### Stage 4 — 打包 (Packing) + 多重打包 (Multipacking)
- **4a Packing**: 用 `linalg.generic` (而非 `tensor.transpose`) 实现数据重排,
  生成论文 §4 的 affine 索引公式:
  - 滤波器打包: `[Nf, Ic, Fh, Fw] → [Ic, Fh, Fw, Nf]` (转置式, 无复制)。
  - 输入打包: `[N, Ic, Fh, Fw, Nwin]` (含跨窗口的**复制**, 因为滑窗有重叠)。
- **4b Multipacking**: 把内层 packing 提升到外层, 一次打包 `K2` 个 tile:
  - IS 调度: 滤波器多重打包 (外层多组滤波器复用同一输入)。
  - WS 调度: 输入多重打包 (外层多组输入复用同一滤波器)。
- **4c Delinearization**: 把塌缩的 H×W 维度恢复成原始 NCHW 形状, 以兼容下游。

#### Stage 5 — 微内核降低 (Microkernel Lowering)
- **5a Bufferize**: `transform.bufferization.one_shot_bufferize`
  (含 `bufferize_function_boundaries=true`), 再叠加
  `extract_address_computations` + `expand_strided_metadata` 模式, 把 tensor
  降为 memref 并暴露基地址/offset/strides。
- **5b 微内核替换**: 匹配带 `microkernel` 属性的 `linalg.generic`, 用新操作
  `transform.lower.to_blas` 把它替换成对 OpenBLAS `sgemm_blas_kernel` 的
  `LLVM::CallOp` 调用 (动态创建 `LLVMFuncOp` 声明)。
- **5c Native Lowering**: 用 `mlir-opt` 一串标准 pass (见 `scripts/run.sh` 与
  论文 Listing 12) 把 LLVMIR dialect 降到可执行, 再由 `mlir-runner` JIT 跑起来。

### 1.4 打包的 Affine 建模 (论文 §4)

论文用一组参数化 affine 方程统一描述打包, 而非硬编码:

**滤波器打包** (无复制, 纯转置):
- 源 tile: `[Nf, Ic, Fh, Fw]` → 目标: `[Ic, Fh, Fw, Nf]`
- 多重打包引入 tile 维 `Nt`, 用 `iTf = iNt * Nf + iNf` (Eq.3) 索引原滤波器集。

**输入打包** (有复制, 因为滑窗重叠):
- 源 tile: `[Ic, Fh, (Nwin+Fw-1)]` → 目标: `[Ic, Fh, Fw, Nwin]`
- 关键索引 `iHw` 由 `ILstart = IOout + IOin + Ss` (Eq.9/13/15) 与
  `iOhw = ILstart + Nwin` (Eq.10/14) 经 floordiv/mod 拆出行列 (Eq.11/12),
  再组合 stride/dilation。`Ss` 是边界 split 的偏移 `Eoff`。

### 1.5 实验结果

**完整性 (Completeness)**: 在 ConvBench 的 7922 个卷积 (6391 pointwise + 1500 regular
+ 31 non-squared) 与 5 个 CNN 模型 (LeNet/AlexNet/VGG19/SqueezeNet/ConvNext) 上,
SConvTransform + OpenBLAS 输出全部与 Native Lowering 基线一致 (数值正确)。
其中 SqueezeNet 的 `tensor.concat` 用 `decompose_concat` 拆, ConvNext 的
`math.erf`/`math.rsqrt` 用 `convert-math-to-llvm` + `convert-math-to-libm` 组合处理。

**通用性 (Generability)**: 在三套架构上均跑通并产生正确结果:
- Apple M4 (ARMv9.4-A, SME 矩阵扩展)
- Intel i7-11700K (x86-64, AVX512)
- IBM Power10 (PPC64le, MMA, KVM 虚拟机)

**性能 (初步, 非首要目标)**:
- Apple M4: regular 卷积中位数 15.5% 峰值, 最佳 59.6%
- Intel AVX512: 中位数 28.7%, 最佳 67%
- 性能瓶颈: 外层 tiling 未优化, 每个 macro tile 仍做 repacking 与边界处理

### 1.6 论文指出的 Transform Dialect 局限 (§5.7)

1. **内层 loop interchange 失效**: `tileUsingSCF` 的 `setInterchange` 只作用于
   外层, IS 调度下内层顺序不对 → 代码用 `swapInductionVars` 手动绕过。
2. **塌缩维度的 affine 表达式错位**: tiling 后索引打乱原始布局, 需两层修正
   (offset 传播 + 相对索引)。
3. **变换可复用性不一致**: `split` 等接口好用, `generalize` 等难组合, 需部分重写。
4. **affine/extract_slice 组合缺抽象**: 暴露底层细节, 开发者门槛高。

### 1.7 未来工作

- 自动 padding 使边界 tile 走完整流水线 (目前小 remainder 只调 affine)。
- 微内核在 MLIR 内部经 Vector dialect → 架构 intrinsic 生成, 取消外部 BLAS 调用。
- Vector-based Packing (前作 [5] 的向量化打包) 减少输入复制内存压力。
- 边界处理整合进 tiling 过程, 消除重复 packing 逻辑。

---

## 二、代码结构导读

```
SConvTransform/
├── CMakeLists.txt          # 顶层: find_package(LLVM/MLIR), add_llvm_executable(sconv-opt)
├── configure.sh            # cmake 配置: 需要 $LLVM_BUILD_DIR, 用 clang+lld+ninja
├── Makefile                # make configure / make build / make clean
├── include/
│   ├── SConv.td            # ODS: 定义 transform.structured.sconv 与 transform.lower.to_blas
│   ├── SConv.h             # 声明 + 工具 (CreateNameLoc/SetNameLoc, registerSConv)
│   ├── CSA.h               # CSA 的 C 结构体 (ConvInfo/ArchInfo/mKInfo/CSAStrategy)
│   └── CMakeLists.txt      # mlir_tablegen 生成 SConv.h.inc / SConv.cpp.inc
├── lib/
│   ├── SConv.cpp           # ★ 核心 (2480 行): SConvOp::apply 五阶段全流程
│   ├── CSA.cpp             # CSA 代价模型 (InputStationary / WeightStationary)
│   ├── LowerToBLAS.cpp     # LowerToBlasOp::apply (memref→指针→LLVM::CallOp)
│   └── CMakeLists.txt      # add_mlir_library(SConvDialect ...) 链接 MLIRTransformDialect 等
├── sconv-opt/
│   └── sconv-opt.cpp       # ★ 驱动工具: 解析 payload+transform, 调 applyTransforms
├── test/                   # 各场景 .mlir payload/transform 文件
├── scripts/
│   ├── opt.sh              # sconv-opt -transform=... payload.mlir
│   ├── run.sh              # mlir-opt 一串 pass → mlir-runner JIT 执行
│   ├── opt_and_run.sh      # opt + run 组合
│   ├── test_transform.sh   # 基线 vs 优化 输出 diff (调 verify.py 数值比对)
│   ├── verify.py           # numpy allclose (rtol=1e-4) 校验
│   └── convbench/          # ConvBench: 7922 卷积数据集 + 生成/批量测试脚本
└── docs/assets/sconvtransform-2025.pdf
```

### 2.1 关键源码映射 (论文 ↔ 代码)

| 论文章节 | 代码位置 | 说明 |
|---------|---------|------|
| §5.1 归一化 | `SConv.cpp` `SConvOp::apply` 开头 | `tensor.collapse_shape` + 建 `linalg.generic` |
| §5.1.1 CSA | `CSA.cpp` `Strategies::compute` | IS/WS 两子类各自算 K2/K3 与代价 |
| §5.2 边界 | `treatEdgeTileConvolution` / `splitAndTileConvolution` | 递归 split + `adjustSecondOpIndexingMap` |
| §5.3 tiling | `applyTileTo` | 两次 `scf::tileUsingSCF` + `swapInductionVars` |
| §5.3 修正 | `promoteOpsOfTile` / `createLinearizedAffineApply` | 修复 affine 索引与 extract_slice 提升 |
| §5.4 打包 | `applyFilterPacking` / `applyInputPacking` | `linalg.generic` + `tensor.collapse_shape` |
| §4.2.1 多重打包 | `filterMultipackingOpt` (IS) / `inputMultipackingOpt` (WS) | 提升 packing 到外层 |
| §4 索引公式 | `computeLinearInputIndices` / `computeMultiPackInputIndices` | 实现 Eq.8/11/12/14 |
| §5.5 微内核属性 | `applyTileTo` 末尾 | 设 `microkernel`/`schedule` 属性 |
| §5.6 BLAS 降低 | `LowerToBLAS.cpp` `lowerKernelToBlas` | `memref.extract_strided_metadata` → 指针 → `LLVM::CallOp` |

### 2.2 构建流程

```bash
# 1. 先有带 MLIR 的 LLVM build (见下节)
export LLVM_BUILD_DIR="/path/to/llvm-project/build"

# 2. 配置 (需 clang, lld, ninja)
make configure          # = ./configure.sh
#   等价于 cmake -S . -B build -G Ninja \
#     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
#     -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_LLD=ON \
#     -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
#     -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir

# 3. 编译, 产出 build/bin/sconv-opt
make build

# 4. 用法
build/bin/sconv-opt -transform=transform.mlir payload.mlir > output.mlir

# 5. 完整跑通 (需 PATH 里有 mlir-opt / mlir-runner, 以及 LLVM_LIB_PATH)
./scripts/opt_and_run.sh payload.mlir transform.mlir -O
```

运行时还需:
- `mlir-opt` 和 `mlir-runner` (来自同一个 LLVM build 的 `bin/`)
- `libmlir_c_runner_utils.{so,dylib}` 与 `libmlir_runner_utils.{so,dylib}` (在 `LLVM_LIB_PATH`)
- 可选: OpenBLAS 的 `sgemm_blas_kernel` (通过 `SCONV_EXTRA_LIB` 环境变量传入)

---

## 三、LLVM 依赖与版本要求

### 3.1 结论 (TL;DR)

> **是, 本项目强依赖 LLVM/MLIR**。它不是一个独立的纯应用代码, 而是 MLIR Transform
> Dialect 的 C++ 扩展, 必须链接到一份**预先编译好且启用 MLIR** 的 LLVM build。

**推荐版本:**

| 优先级 | LLVM 版本 | 理由 |
|--------|----------|------|
| ★ 首选 | **LLVM 20.x** (2025-03 发布) | 项目主力开发期 (2025-05 ~ 2025-10) 的最新稳定版, API 最匹配 |
| 备选 | **LLVM 21.x** (2025-09 发布) | 与论文成稿日期 (2025-10-30) 同期, 后期提交可能基于此 |
| 最低 | **LLVM 19.x** (2024-09 发布) | 所有必需 API 已就绪; 但 `detail::` 内部 API 在 20 之后有微调, 不保证零修改 |
| 不推荐 | LLVM ≤ 18.x | 缺 `linalg::splitOp` / `registerAllExtensions` / 部分 transform 模式 |

**建议直接拉取 `llvm-project` 的 `release/20.x` 分支** 自行编译; 若失败再退到
`release/19.1.x`, 或前进到 `release/21.x`。

### 3.2 判定依据 (代码中使用的版本敏感 API)

以下 API 在代码中出现, 它们把版本下限锁在 **LLVM ≥ 19**:

| API / 用法 | 出现位置 | 引入版本 |
|-----------|---------|---------|
| `linalg::splitOp(rewriter, op, dim, splitPoint)` 返回 `pair<TilingInterface,TilingInterface>` | `lib/SConv.cpp:1831` `performSplit` | LLVM 19 (2024) |
| `mlir::registerAllExtensions(registry)` (与 `registerAllDialects` 分离) | `sconv-opt/sconv-opt.cpp:320` | LLVM 19 |
| `scf::SCFTilingResult` 含 `.loops` / `.tiledOps` 字段; `scf::tileUsingSCF` 返回 `FailureOr<SCFTilingResult>` | `lib/SConv.cpp:1621-1650` | LLVM 18, 19 定型 |
| `scf::SCFTilingOptions::LoopType::ForOp` (`setLoopType`) | `lib/SConv.cpp:1618,1646` | LLVM 18+ |
| `transform.foreach` op (带 block arg) | `test/lowering/sconv-blas.mlir:52` | LLVM 18+ |
| `transform.apply_patterns.memref.extract_address_computations` | `test/lowering/sconv-blas.mlir:27` | LLVM 19~20 |
| `transform.apply_patterns.memref.expand_strided_metadata` | 同上 :28 | LLVM 18+, 19 定型 |
| `transform.apply_patterns to %op { ... }` 块语法 | 同上 :26 | LLVM 18+ |
| `MLIRContext(registry, MLIRContext::Threading::DISABLED)` (枚举而非 bool) | `sconv-opt/sconv-opt.cpp:249` | LLVM 18+ |
| `rewriter.create<LLVM::CallOp>(loc, callee, ValueRange{...})` (传 `LLVMFuncOp` 而非符号名) | `lib/LowerToBLAS.cpp:204` | LLVM 19+ |
| `detail::findTransformEntryPoint(root, ModuleOp{}, entryName)` 三参形式 | `sconv-opt/sconv-opt.cpp:296` | LLVM 19 内部 API |
| `detail::mergeSymbolsInto(*transformRoot, std::move(libraryModule))` | `sconv-opt/sconv-opt.cpp:284` | LLVM 19 内部 API |
| `convOp.getDpsInputs()` / `getDpsInits()` (DPS 接口新命名) | `lib/SConv.cpp:2328-2329` | LLVM 19+ (旧名 `getInputs/getOutputs` 仍并存) |
| `memref::ExtractAlignedPointerAsIndexOp` | `lib/LowerToBLAS.cpp:128` | LLVM 18+ |

> 注: `sconv-opt.cpp` 直接调用了 `mlir::transform::detail::` 命名空间下的内部函数
> (`findTransformEntryPoint` / `mergeSymbolsInto`)。这些是非公开 API, **LLVM 小版本
> 之间都可能改签名**, 因此实际编译时, 选用与开发者同时期的 LLVM 版本最稳妥。

### 3.3 项目开发时间线 (来自 git log)

| 时间 | 事件 | 对应 LLVM |
|------|------|-----------|
| 2024-09-11 | 首次提交 "Initial version" | LLVM 19 刚发布 (2024-09-17) |
| 2024-10 | 用 `tileUsingSCF` 重写 tiling | LLVM 19 |
| 2025-05-14 | "Lower micro-kernel to BLAS call" | LLVM 20 已发布 (2025-03-07) |
| 2025-06 | 多边界支持、affine 修正 | LLVM 20.x |
| 2025-10-30 | 论文定稿 | LLVM 21 将发布 (2025-09) |
| 2025-11-03 | 最新提交 (Makefile/README) | LLVM 20.1.x 或 21.x |

### 3.4 编译 LLVM 的推荐配置

```bash
# 拉取 (推荐 release/20.x 分支)
git clone --depth 1 --branch release/20.x https://github.com/llvm/llvm-project.git
cd llvm-project

cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS="mlir;clang"  \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;ARM;AArch64;PowerPC;WebAssembly" \
  -DLLVM_ENABLE_LLD=ON

cmake --build build -j$(nproc)
# 产出: build/bin/{mlir-opt,mlir-runner,sconv-opt 链接所需库}
#        build/lib/libmlir_*.{a,so,dylib}, cmake 配置在 build/lib/cmake/{llvm,mlir}/
```

要点:
- **必须启用 MLIR** (`LLVM_ENABLE_PROJECTS` 含 `mlir`); 默认不开。
- 建议同时编 `clang` (项目 `configure.sh` 指定 `clang/clang++` 且 `-fuse-ld=lld`)。
- 目标架构按需选; Apple Silicon 选 `AArch64`, Intel 选 `X86`, Power10 选 `PowerPC`。
- 编译耗时较长 (Release 约 30~60 min 视核数), 磁盘约 20~40 GB。

### 3.5 环境变量与 PATH

编译完 LLVM 后, 跑 SConvTransform 还需让 shell 找到 `mlir-opt` / `mlir-runner` 与
运行时库:

```bash
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export PATH="$LLVM_BUILD_DIR/bin:$PATH"
export LLVM_LIB_PATH="$LLVM_BUILD_DIR/lib"
# 可选: 若要用 OpenBLAS 微内核
export SCONV_EXTRA_LIB="/path/to/libopenblas.{so,dylib}"
```

### 3.6 宿主工具链要求

`configure.sh` 硬性要求:
- **CMake ≥ 3.22** (`CMakeLists.txt:1`)
- **Ninja** (`-G Ninja`)
- **clang / clang++** (显式指定为编译器)
- **lld** (`-fuse-ld=lld` 在 `CMakeLists.txt:15-17` 强制开启)

macOS 上可用 Homebrew 的 `llvm`/`ninja`/`cmake`; Linux 上用对应包管理器。
