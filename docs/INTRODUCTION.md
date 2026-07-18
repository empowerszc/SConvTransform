# SConvTransform 通俗导读

> 写给了解 MLIR 但不熟悉本项目的人。
> 结合论文、代码和实际跑通的测试例子, 用大白话讲清楚"这个项目到底干了什么"。

---

## 一句话概括

**SConvTransform 是 MLIR Transform Dialect 的一个扩展, 它能把一个普通的 `linalg.conv_2d` 卷积操作, 自动拆成"分块 + 数据重排 + 调用 OpenBLAS 矩阵乘微内核"的循环嵌套, 让卷积跑得快几十倍。**

---

## 1. 背景: 卷积为什么慢?

卷积是 CNN (卷积神经网络) 里最耗时的操作。一个简单的 2D 卷积:

```
输入: 1×128×66×66 (1张图, 128通道, 66×66像素)
权重: 256×128×3×3 (256个滤波器, 每个128通道×3×3)
输出: 1×256×64×64
```

这个卷积的计算量 = 2 × 64 × 64 × 256 × 128 × 3 × 3 ≈ **2.4 亿次浮点运算**。

如果用 MLIR 的 native lowering (把 `linalg.conv_2d` 直接展开成标量循环), 就是一层套一层的 `for` 循环, 每次只算一个乘加, 完全没有利用缓存和 SIMD 指令。在我们的 M4 机器上, 这种"裸跑"只有 **0.17 GFLOPS** —— 也就是说上面这个卷积要跑 **1.4 秒**。

而 OpenBLAS 的矩阵乘 (`sgemm`) 在同样的机器上能跑到 **15 GFLOPS**, 快了近 100 倍。

**问题来了: 卷积不是矩阵乘, 怎么用 BLAS?**

这就是 SConvTransform 要解决的核心问题。

---

## 2. 核心思想: 卷积可以变成矩阵乘

如果你做过深度学习, 可能听说过 **im2col**: 把卷积的输入"展开"成一个大矩阵, 然后用一次 GEMM (矩阵乘) 算完整个卷积。但 im2col 的缺点是会**大量复制数据** (每个滑窗的输入数据都复制一份), 内存开销巨大。

SConv 用了一个更聪明的方法: **不是把所有数据一次性展开, 而是切成小块, 每次只打包一小块数据, 算完再切下一块。** 这样既利用了 BLAS 的高效矩阵乘, 又控制了内存开销。

用切蛋糕打个比方:

- **Native lowering**: 整个蛋糕一口一口啃, 每口只咬一粒蛋糕屑 (标量循环)
- **im2col**: 把整个蛋糕切成粒, 重新摆成一个大长条, 然后一口吞掉 (大量复制)
- **SConv**: 把蛋糕切成巴掌大的块, 每块打包好, 一块一块地用机器吃 (分块+打包+微内核)

---

## 3. SConvTransform 的五步流水线

论文里把这五步叫 Stage 1~5, 代码里对应 `lib/SConv.cpp` 的 `SConvOp::apply()` 方法。我们用一个真实例子走一遍。

### 测试例子

```
输入:  1×128×66×66  (1张图, 128通道, 66×66)
权重:  256×128×3×3  (256个3×3滤波器)
输出:  1×256×64×64  (stride=1, padding=1)
```

对应的 MLIR payload:
```mlir
%res = linalg.conv_2d_nchw_fchw
  {dilations = dense<1>, strides = dense<1>}
  ins(%in, %wei : tensor<1x128x66x66xf32>, tensor<256x128x3x3xf32>)
  outs(%out : tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32>
```

---

### Stage 1: 归一化 (把卷积变成 linalg.generic)

**论文 §5.1 / 代码 `SConv.cpp` 第 2320 行附近**

SConv 不直接处理 `linalg.conv_2d_nchw_fchw`, 而是先把它"翻译"成更通用的 `linalg.generic` 操作。这一步做两件事:

1. **Generalize**: 把 named op 变成 generic op, 暴露出所有的循环结构和 affine 索引
2. **Linearization (塌缩)**: 把 H×W 两个空间维度合并成一个维度

为什么要塌缩? 因为 SConv 是按"窗口序列"来分块的, 不是按行/列独立分块。MLIR 的 tiler 对多维独立分块支持有限, 合并成一个维度后就好处理了。

塌缩后:
```mlir
// 原来的输出: tensor<1x256x64x64>  (N, Oc, Oh, Ow)
// 塌缩后:     tensor<1x256x4096>   (N, Oc, Oh*Ow)
```

代码里这步在 `SConv.cpp:2359-2368`, 用 `tensor.collapse_shape` 实现。

---

### Stage 1c: CSA 分析 (决定怎么切)

**论文 §2.2, §5.1.1 / 代码 `lib/CSA.cpp`**

CSA (Convolution Slicing Analysis) 是整个流程的"大脑"。它根据三个输入决定怎么分块:

1. **卷积形状** (ConvInfo): 输入/输出通道数、空间尺寸、kernel 大小
2. **机器架构** (ArchInfo): L1/L2/L3 缓存大小、延迟、cache line 大小
3. **微内核形状** (mKInfo): 每次微内核调用处理多少个窗口 (Nwin) 和多少个滤波器 (Nf)

对例子中的卷积 (128通道, 64×64输出, 3×3 kernel), 默认参数:
- Nwin = 16, Nf = 8 (微内核每次处理 16 个窗口 × 8 个滤波器)

CSA 的输出是一个 **CSAStrategy**:

| 参数 | 值 | 含义 |
|------|---|------|
| schedule | IS | 输入驻留 (输入 tile 放 L1, 滤波器 tile 换着用) |
| Nc | 32 | 每次处理 32 个输入通道 (L1 能放下) |
| K2 | 32 | L2 层放 32 组滤波器 tile |
| K3 | 128 | L3 层放 128 组窗口 tile |

CSA 还检查这些值能不能整除, 不能的话就有"边界" (edge case), 需要额外处理。

CSA 的代价模型很简单: 分别算 IS 和 WS 两种调度策略的内存访问量, 选代价小的那个。代码在 `CSA.cpp` 的 `InputStationary::cost_model()` 和 `WeightStationary::cost_model()`。

---

### Stage 2: 边界处理 (切不整怎么办?)

**论文 §5.2 / 代码 `SConv.cpp` 的 `treatEdgeTileConvolution` 函数**

比如输出空间维度 Oh×Ow = 64×64 = 4096, 而 Nwin = 16:
- 4096 / 16 = 256, 整除! 不需要处理边界

但如果 Oh×Ow = 75×75 = 5625:
- 5625 / 16 = 351 余 9
- 需要把卷积 split 成: 主部分 (5616 个窗口) + 边界部分 (9 个窗口)
- 主部分走完整 tiling 流水线; 边界部分太小, 只做 affine 索引修正

代码用 `linalg::splitOp` 实现切分, 沿某一维度把一个 op 变成两个。

---

### Stage 3: 两级分块 (Tiling)

**论文 §5.3 / 代码 `SConv.cpp` 的 `applyTileTo` 函数**

分块就是把大的循环切成小的。SConv 用两级分块:

**第一级 (外层, CSA 驱动)**:
```
for N (batch=1)          ← 批次
  for Nc=32 (input channels)  ← L1 缓存能放下的通道数
    for K3=128 (window tiles)  ← L3 驱动的窗口复用
      for K2=32 (filter tiles)  ← L2 驱动的滤波器复用
```

**第二级 (内层, 微内核驱动)**:
```
        for Nwin=16 (windows per uK call)  ← 微内核每次处理 16 个窗口
          for Nf=8 (filters per uK call)    ← 微内核每次处理 8 个滤波器
            → 调用微内核 (linalg.generic)
```

内层每次迭代就是一次微内核调用: 处理 16 个窗口 × 8 个滤波器, 内部对 K (=32×3×3=288 个输入元素) 做归约。

代码用 `scf::tileUsingSCF` 实现分块。但有个坑: MLIR 的 tiler 在 IS 调度下内层循环顺序不对 (`setInterchange` 对内层不生效), 论文 §5.7 专门吐槽了这个问题。代码里手写了 `swapInductionVars()` 函数, 强行交换内外循环的归纳变量和边界来修复。**这个函数有个 bug: 交换后外循环引用了定义在自身 body 内的常量, 违反 SSA 支配关系。我们在编译时发现并修了 (把内循环界常量提升到外循环之前)。**

---

### Stage 4: 打包 (Packing)

**论文 §4, §5.4 / 代码 `SConv.cpp` 的 `applyFilterPacking` 和 `applyInputPacking` 函数**

打包是 SConv 最精巧的部分。微内核期望输入和权重以特定的内存布局存放, 打包就是把原始数据重新排列成这个布局。

**滤波器打包** (简单, 只是转置):
```
原始: [Nf=8, Ic=32, Fh=3, Fw=3]  → 排列成 → [Ic=32, Fh=3, Fw=3, Nf=8]
然后 collapse 成 2D: [288, 8]  (= 32*3*3, 8)
```
滤波器没有数据复制, 只是把维度顺序换一下。

**输入打包** (复杂, 有数据复制):
```
原始输入 tile: [Ic=32, (Nwin + Fw - 1)]  (因为滑窗有重叠)
打包后:       [Ic=32, Fh=3, Fw=3, Nwin=16]  (每个窗口的 3×3 邻域都复制进来)
然后 collapse 成 3D: [288, 16]  (= 32*3*3, 16)
```

为什么输入要复制? 因为滑窗之间有重叠 (stride < kernel size), 相邻窗口共享大部分输入数据。打包时把每个窗口需要的 3×3 邻域都"铺平"摆好, 微内核就能连续读取, 不用做复杂的索引计算。

打包的索引计算用 affine map 实现, 论文 §4 给了一组参数化方程。比如输入打包的索引:
```
iHw = ((ILstart + nwin) / Ow - ILstart / Ow) * strideH * Iw
     + ((ILstart + nwin) % Ow - ILstart % Ow) * strideW
     + fh * dilationH * Iw + fw * dilationW
```
这个公式把"窗口序号"翻译回原始输入张量的二维坐标。代码里对应 `computeLinearInputIndices()` 函数。

**Multipacking (多重打包)**: 把内层打包提升到外层, 一次打包 K2 个 tile, 让外层循环也能复用打包数据。

---

### Stage 5: 微内核替换 (Lower to BLAS)

**论文 §5.6 / 代码 `lib/LowerToBLAS.cpp`**

经过 Stage 1~4, 原来的卷积变成了:

```mlir
scf.for ... {  // 外层 tiling
  // 打包好的输入: memref<288x16xf32>  (= [K, M])
  // 打包好的权重: memref<288x8xf32>   (= [K, N])
  scf.for ... {  // 内层 tiling
    linalg.generic {  // 微内核: C[M,N] += A[K,M]^T * B[K,N]
      indexing_maps = [lhs: (d0,d1,d2,d3) -> (d0,d3,d2),
                       rhs: (d0,d1,d2,d3) -> (d3,d1),
                       out: (d0,d1,d2,d3) -> (d0,d1,d2)]
      iterator_types = [parallel, parallel, parallel, reduction]
    }
  }
}
```

这个 `linalg.generic` 本质上是一个小矩阵乘: `C += A^T * B`。但 native lowering 还是会把它展开成标量循环。

Stage 5 做的事: **把这个 `linalg.generic` 替换成对 OpenBLAS `sgemm` 的函数调用**。

具体步骤:
1. **Bufferize** (5a): 把 tensor 变成 memref (有内存地址的类型)
2. **提取指针** (5b): 用 `memref.extract_strided_metadata` 拿到 tile 的基地址和 strides, 算出实际指针
3. **创建函数调用** (5b): 声明 `sgemm_blas_kernel(m, n, k, alpha, A, B, C, ldc)`, 把微内核替换掉

我们写的 `sgemm_blas_kernel` 包装函数:
```c
// SConv 微内核算的是 C[N,M] += B[K,N]^T * A[K,M]
// 在 CBLAS 里等价于:
int sgemm_blas_kernel(long m, long n, long k, float alpha,
                      float *A, float *B, float *C, long ldc) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, m, k, alpha, B, n, A, m, 1.0f, C, ldc);
    return 0;
}
```

注意 SConv 的微内核做的是 `C[n][m] += A[k][m] * B[k][n]`, 不是标准的 `C = A*B`, 需要转置 B。这个细节论文没写, 是我们从代码和输出对比中推导出来的。

---

## 4. 完整的执行流程 (我们实际跑的例子)

用我们测试的 `custom_0` (4×192×40×40 → 192×192×3×3, stride=1) 为例:

```
原始 MLIR payload (含 main 函数, 生成数据, 调用卷积)
    │
    ▼ sconv-opt -transform=sconv-blas.mlir
Stage 1-4: 归一化 → CSA → 边界处理 → 两级 tiling → packing
    │   (产出: 嵌套 scf.for + 打包的 linalg.generic 微内核)
    ▼
Stage 5a: bufferization (tensor → memref)
Stage 5b: lower.to_blas (linalg.generic → LLVM::CallOp sgemm_blas_kernel)
    │   (产出: 嵌套 scf.for + func.call @sgemm_blas_kernel)
    ▼
mlir-opt (标准 lowering passes)
    │   (scf → cf, linalg → affine loops, memref → LLVM, func → LLVM)
    ▼
mlir-cpu-runner (JIT 执行)
    │   (链接 libmlir_runner_utils + librtclock + libsgemm_blas_kernel + libopenblas)
    ▼
输出: 打印 memref 数据 + 计时 (ms) + GFLOPS
```

Baseline 路径 (无 transform):
```
原始 MLIR payload → mlir-opt (标准 lowering) → mlir-cpu-runner
    (linalg.conv 直接展开成标量 for 循环, 无优化)
```

---

## 5. 实测结果 (Apple M4)

### 大卷积 (batch=4, 自定义配置)

| 卷积 | FLOPs | Baseline | SConv+BLAS | 加速比 |
|------|------:|---:|---:|---:|
| custom_0 (192ch, 40²) | 4.25G | 6,266ms | 78ms | 81x |
| custom_1 (768ch, 80²→40²) | 67.95G | 98,954ms | 1,127ms | **88x** |
| custom_4 (384ch, 160²→80²) | 67.95G | 98,560ms | 1,177ms | 84x |
| custom_7 (192ch, 20²) | 1.06G | 1,557ms | 20ms | 77x |
| ... | ... | ... | ... | ... |
| **总计** | — | **352.8s** | **4.4s** | **80x** |

### 小卷积 (batch=1, ConvBench 采样, warmup=5, runs=10)

| 卷积 | FLOPs | Baseline | SConv+BLAS | 加速比 |
|------|------:|---:|---:|---:|
| conv_1800 | 2.3M | 2.75ms | 0.14ms | 19x |
| conv_3557 | 0.6M | 0.77ms | 0.11ms | 7x |
| ... | ... | ... | ... | ... |

大卷积加速 50~88 倍, 小卷积加速 4~19 倍。差异来自小卷积的 BLAS 调用开销占比更高。

### 正确性

全部 16 个卷积配置, SConv+BLAS 输出与 baseline 逐元素一致 (numpy `allclose`, rtol=1e-3)。

---

## 6. 代码结构速查

如果你想读代码或改代码, 按这个顺序看:

```
include/SConv.td          ← 定义两个 transform op: structured.sconv 和 lower.to_blas
include/CSA.h             ← CSA 的数据结构 (ConvInfo, ArchInfo, mKInfo, CSAStrategy)
lib/CSA.cpp               ← CSA 代价模型 (IS vs WS, 算 K2/K3/Nc)
lib/SConv.cpp (2480行)    ← 核心流水线, 关键函数:
  - SConvOp::apply()           入口, 编排 5 个 stage
  - treatEdgeTileConvolution()  Stage 2: 边界 split
  - applyTileTo()              Stage 3: 两级 tiling + Stage 4: packing
  - applyFilterPacking()       滤波器打包
  - applyInputPacking()        输入打包 (含 affine 索引公式)
  - swapInductionVars()        修复 MLIR tiler 的 loop interchange bug
  - filterMultipackingOpt()    IS 多重打包
  - inputMultipackingOpt()     WS 多重打包
  - adjustLinalgOps()          重写微内核为外积形式
lib/LowerToBLAS.cpp       ← Stage 5: 微内核 → BLAS 调用
sconv-opt/sconv-opt.cpp   ← 命令行工具 (解析 payload + transform, 调 applyTransforms)
scripts/convbench/        ← ConvBench 测试框架
  - run_convbench.py          自动化测试 (正确性 + 性能)
  - runtime/sgemm_blas_kernel.c  OpenBLAS 包装函数
  - runtime/rtclock.c         计时函数
  - payloads/*.mlir           测试 payload 模板
  - samples/custom_conv.csv   用户自定义卷积配置
```

---

## 7. 关键设计决策 (论文的亮点)

### 7.1 为什么用 Transform Dialect 而不是 Pass?

传统的 MLIR 优化用 Pass (编译通道), 但 Pass 是黑盒: 你只能看到输入和输出, 中间过程不可见、不可组合、不可复用。

SConvTransform 用 Transform Dialect, 把整个优化流程写成**声明式**的 MLIR 脚本:
```mlir
%ukernels, %loops = transform.structured.sconv %convs
  { mK_info = [16, 8] }
```

好处:
- 可读: 一眼看出做了什么
- 可组合: 可以和其他 transform 操作串联
- 可分析: 中间 IR 始终合法, 能打印、能验证

### 7.2 为什么不直接用 Im2Col + GEMM?

Im2Col 把整个输入展开成大矩阵, 内存膨胀严重 (一个 128×66×66 的输入, im2col 后变成 128×3×3 × 4096 = 4.7M 元素, 复制了 ~12 倍)。

SConv 按需打包 (packing-on-demand), 每次只打包一个 tile (288×16 = 4608 元素), 用完就丢, 内存开销小得多。

### 7.3 为什么用 OpenBLAS 而不是 MLIR 自带的 Vector dialect?

论文说得很坦率: MLIR 的 Vector dialect 在某些架构 (如 Power10) 上还没有成熟的后端。而 OpenBLAS 在所有主流架构上都有高度优化的微内核。用 OpenBLAS 可以"一次编写, 到处能跑"。

代价是引入了外部库依赖, 且 OpenBLAS 的微内核不是公开 API (需要自己写包装函数)。

---

## 8. 局限性和坑

1. **只支持 NCHW 布局**: 不支持 NHWC, 不支持 grouped convolution
2. **边界处理不完美**: 小于 tile 大小的边界部分跳过 tiling/packing, 只做 affine 修正, 性能不如主部分
3. **swapInductionVars 有 SSA bug**: 原始代码交换循环界后没提升常量, 我们修了
4. **强依赖 LLVM 19**: LLVM 20 的 TransformResults API 变了, 不兼容
5. **setAttrs 语法**: LLVM 19 不接受初始化列表, 需要改成逐个 setAttr
6. **外层 tiling 未优化**: macrokernel 仍有 repacking 开销, 性能未达峰值
7. **OpenBLAS 后端**: 需要架构专用的 OpenBLAS 构建 (如 SME 优化版) 才能达到论文性能

---

## 9. 如何复现

```bash
# 1. 准备 LLVM 19.1.7 (下载预编译包或自编译)
export LLVM_BUILD_DIR=/path/to/LLVM-19.1.7
export SCONV_ROOT=/path/to/SConvTransform

# 2. 编译 SConvTransform
cd $SCONV_ROOT && make configure && make build

# 3. 跑测试 (需 OpenBLAS + numpy)
export PYTHONUNBUFFERED=1
python3 -u scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/custom_conv.csv \
  --num 100 --runs 1 --warmup 1

# 4. 只跑小卷积 (快, 多次迭代)
python3 -u scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/small_regular.csv \
  --num 5 --runs 10 --warmup 5
```

详细的环境搭建和排错指南见 `docs/BUILD_AND_TROUBLESHOOTING.md`。

---

## 10. 术语速查

| 术语 | 大白话解释 |
|------|-----------|
| CSA (Convolution Slicing Analysis) | 根据卷积大小和机器缓存, 算出"切多大"的分析器 |
| CSO (Convolution Slicing Optimization) | 根据切法, 生成循环嵌套的代码生成器 |
| Tiling (分块) | 把大循环切成小块, 让每块数据能放进缓存 |
| Packing (打包) | 把数据按微内核喜欢的布局重新排列 |
| Multipacking | 一次打包多个 tile, 提升外层循环的数据复用 |
| IS (Input Stationary) | 输入 tile 驻留 L1, 滤波器 tile 换着用 |
| WS (Weight Stationary) | 滤波器 tile 驻留 L1, 输入 tile 换着用 |
| Nwin | 微内核每次处理的窗口数 (如 16) |
| Nf | 微内核每次处理的滤波器数 (如 8) |
| Nc | 每次处理的输入通道数 (L1 驱动) |
| K2 / K3 | L2 / L3 层的 tile 复用数 |
| Microkernel | 最内层的矩阵乘计算, 调用 OpenBLAS sgemm |
| Transform Dialect | MLIR 的"声明式变换"方言, 用脚本描述优化 |
| linalg.generic | MLIR 的通用线性代数操作, 可自定义 indexing map |
