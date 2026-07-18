# SConvTransform 通俗导读 (含公式详解)

> 写给了解 MLIR 但不熟悉本项目的人。
> 结合论文、代码和实际跑通的测试, 用大白话 + 公式讲清楚每一步。

---

## 一句话概括

**SConvTransform 是 MLIR Transform Dialect 的扩展, 把 `linalg.conv_2d` 卷积自动拆成"分块 + 数据重排 + OpenBLAS 矩阵乘微内核"的循环嵌套, 让卷积快几十倍。**

---

## 1. 卷积的数学定义

### 1.1 2D 卷积公式

给定输入张量 `I[N, Ic, Ih, Iw]` 和权重 `W[Oc, Ic, Fh, Fw]`, 输出 `O[N, Oc, Oh, Ow]`:

```
O[n, oc, oh, ow] = Σ_{ic=0}^{Ic-1} Σ_{fh=0}^{Fh-1} Σ_{fw=0}^{Fw-1}
    I[n, ic, oh*strideH + fh*dilationH - padH_top,
         ow*strideW + fw*dilationW - padW_left]
  × W[oc, ic, fh, fw]
```

其中:
- `Oh = (Ih + padH_top + padH_bottom - (Fh-1)*dilationH - 1) / strideH + 1`
- `Ow = (Iw + padW_left + padW_right - (Fw-1)*dilationW - 1) / strideW + 1`

### 1.2 我们的测试例子

```
输入 I: [1, 128, 66, 66]    (N=1, Ic=128, Ih=66, Iw=66)
权重 W: [256, 128, 3, 3]    (Oc=256, Fh=3, Fw=3)
stride = (1,1), dilation = (1,1), pad = (1,1)
输出 O: [1, 256, 64, 64]    (Oh=64, Ow=64)
```

验证: `Oh = (66 + 1 + 1 - (3-1)*1 - 1) / 1 + 1 = 65/1 + 1 = 64` ✓

计算量 = `2 × N × Oc × Oh × Ow × Ic × Fh × Fw = 2 × 1 × 256 × 64 × 64 × 128 × 3 × 3 = 2,415,919,104 ≈ 2.4 GFLOP`

### 1.3 卷积 ≈ 矩阵乘

卷积的本质是: 对每个输出位置, 做一次加权求和。如果把每个窗口的输入数据
"铺平"成一列, 滤波器"铺平"成一行, 卷积就变成了矩阵乘。

**展开成矩阵乘**:
- 把输入的每个窗口 `[Ic, Fh, Fw]` 铺平成向量 `a` (长度 `Ic×Fh×Fw = K`)
- 把每个滤波器 `[Ic, Fh, Fw]` 铺平成向量 `b` (长度 `K`)
- 输出 `O[oc, window] = a[window] · b[oc]`

即 `O = A × B^T`, 其中 A 是 `[Oh×Ow, K]`, B 是 `[Oc, K]`。

**Im2Col 的问题**: A 矩阵大小 = `4096 × 1152 × 4 bytes ≈ 18.4 MB`, 而原始输入
只有 `1 × 128 × 66 × 66 × 4 ≈ 2.2 MB`, 数据膨胀了 8 倍。

**SConv 的做法**: 不一次全部展开, 而切成小块逐块打包, 控制内存开销。

---

## 2. 五步流水线详解

### Stage 1: 归一化 — 把卷积变成 linalg.generic

**代码: `SConv.cpp:2320-2411`**

SConv 不直接处理 named op, 先泛化成 `linalg.generic`, 暴露所有循环和索引。
这一步做两件事:

#### 1a. Generalize — 构造 affine map

泛化后的 `linalg.generic` 有 6 个循环维度:

| 维度 | 含义 | 类型 |
|------|------|------|
| d0 | batch (N) | parallel |
| d1 | output channel (Oc) | parallel |
| d2 | output spatial (Oh×Ow, 已塌缩) | parallel |
| d3 | input channel (Ic) | reduction |
| d4 | filter height (Fh) | reduction |
| d5 | filter width (Fw) | reduction |

三个 affine map (代码 `SConv.cpp:2402-2404`):

```mlir
// 输入索引: 从输出坐标反推输入坐标
// d2 是塌缩后的 (oh*Ow + ow), 用 floordiv/mod 拆回 oh, ow
lhs_map = affine_map<(d0, d1, d2, d3, d4, d5) ->
    (d0,  d3,  (d2 floordiv Ow) * strideH + d4 * dilationH) * Iw
        + (d2 mod Ow) * strideW + d5 * dilationW)>

// 权重索引: 直接索引
rhs_map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4, d5)>

// 输出索引: 直接索引
result_map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
```

输入索引的公式拆解:
```
input_idx = (oh * strideH + fh * dilationH) * Iw + ow * strideW + fw * dilationW
```
其中 `oh = d2 floordiv Ow`, `ow = d2 mod Ow` (从塌缩维度恢复二维坐标)。

#### 1b. Linearization — 塌缩空间维度

```mlir
// 塌缩前:
tensor<1x128x66x66xf32>  → 输入
tensor<1x256x64x64xf32>  → 输出

// 塌缩后 (H×W 合并):
tensor<1x128x4356xf32>  → 输入 (66×66=4356)
tensor<1x256x4096xf32>  → 输出 (64×64=4096)
```

用 `tensor.collapse_shape` 实现。塌缩后 d2 是一个一维索引, 表示"第几个窗口"。

---

### Stage 1c: CSA — 决定怎么切

**代码: `lib/CSA.cpp`**

#### 输入参数

CSA 接收三组参数:

**ConvInfo** (卷积形状):
```
Ic=128, Iw=66, Oh=64, Ow=64, Fh=3, Fw=3, Oc=256, data_size=4 (float32)
```

**ArchInfo** (机器参数, 代码 `SConv.cpp:2240-2244` 默认值):
```
L1 = 32768 * 0.9 = 29491 bytes   (~29 KB)
L2 = 1048576 * 0.9 = 943718 bytes  (~922 KB)
L3 = 4194304 * 0.9 = 3774873 bytes (~3.6 MB)
L1_latency=2, L2_latency=10, L3_latency=30, mem_latency=300
cache_line = 128 bytes
```

**mKInfo** (微内核形状):
```
Nwin=16 (每次微内核处理16个窗口)
Nf=8    (每次微内核处理8个滤波器)
Noutput=128 (=16×8, 每次输出16×8=128个标量)
```

#### 基础量计算

```
in_size  = Nwin × Fh × Fw × data_size = 16 × 3 × 3 × 4 = 576 bytes
w_size   = Nf × Fh × Fw × data_size = 8 × 3 × 3 × 4 = 288 bytes
out_size = Noutput × data_size = 128 × 4 = 512 bytes
```

这些是微内核每次调用所需的输入、权重、输出 tile 大小。

#### Nc 的推导 (L1 约束)

**目标**: 让一个 input tile + filter tile + output tile 同时放进 L1。

IS 调度下 L1 约束 (代码 `CSA.cpp:164-168`):
```
tileSizeL1(Nc) = in_size × Nc + w_size × Nc + out_size
               = 576 × Nc + 288 × Nc + 512
               = 864 × Nc + 512 ≤ L1 = 29491
```

解得: `Nc ≤ (29491 - 512) / 864 = 33.5`

用半值启发式 (从 Ic=128 开始不断除 2 直到满足约束):
```
Nc=128: 864×128+512 = 111104 > 29491 → 太大
Nc=64:  864×64+512  = 55808  > 29491 → 太大
Nc=32:  864×32+512  = 28160  ≤ 29491 → 满足! Nc=32
```

**结果: Nc = 32** (每次处理 32 个输入通道)

通道分块数: `tCH = Ic / Nc = 128 / 32 = 4` (余数 `extra_tCH = 0`, 无边界)

每通道层的 tile 数:
```
in_tiles_per_tch = (Oh × Ow) / Nwin = 4096 / 16 = 256
w_tiles_per_tch  = Oc / Nf = 256 / 8 = 32
```

#### K2 的推导 (L2 约束, IS 调度)

IS 调度下 L2 放滤波器 tile (代码 `CSA.cpp:170-171`):
```
tileSizeL2(K2) = in_size + K2 × w_size + K2 × out_size
              = 576 + K2 × 288 + K2 × 512
              = 576 + K2 × 800 ≤ L2 = 943718
```

从 `w_tiles_per_tch=32` 开始: `576 + 32×800 = 26176 ≤ 943718` → **K2 = 32**

余数: `extra_k2 = 32 % 32 = 0`, 无边界

#### K3 的推导 (L3 约束, IS 调度)

IS 调度下 L3 放输入 tile (代码 `CSA.cpp:173-174`):
```
tileSizeL3(K3) = K3 × in_size + K2 × w_size + K2 × K3 × out_size
              = K3 × 576 + 32 × 288 + 32 × K3 × 512
              = K3 × (576 + 16384) + 9216
              = K3 × 16960 + 9216 ≤ L3 = 3774873
```

从 `in_tiles_per_tch=256` 开始:
```
K3=256: 256×16960+9216 = 4350976 > 3774873 → 太大
K3=128: 128×16960+9216 = 2180096 ≤ 3774873 → 满足! K3=128
```

余数: `extra_k3 = 256 % 128 = 0`, 无边界

#### IS vs WS 的代价模型

CSA 同时算 IS 和 WS 的代价, 取较小的。代价 = 加权内存访问量 (代码 `CSA.cpp:196-269`):

**IS 代价模型** (Input Stationary):
```
EQ1: mem = ceil((in_tiles_total × in_size + w_tiles_total × w_size) / cache_line)
     = 第一次从内存加载所有 input 和 weight tile 的 cache line 数

EQ2: mem += tCH × ceil(w_fit × in_fit × w_tiles_per_tch × w_size / cache_line)
     = 当滤波器 tile 数 > K2 且输入 tile 数 > K3 时, 额外的内存访问

EQ3: l3 = tCH × ceil(w_fit × in_tiles_per_tch × in_size / cache_line)
     = 滤波器 tile 组超出 L2 时, 输入 tile 从 L3 重复加载

EQ4: l2 = tCH × ceil(ntiles × w_tiles_per_tch × w_size / cache_line)
     = 除第一个输入 tile 外, 滤波器 tile 从 L2 重复加载

EQ5: l1 = 2 × total_flops - (l3 + l2 + mem)
     = L1 命中的访问量 (总访问减去其他层级)

EQ6: (tCH > 1 时) 跨通道层的输出回读, 根据访问距离分配到对应缓存层级

总代价 = l1 × L1_latency + l2 × L2_latency + l3 × L3_latency + mem × mem_latency
```

**WS 的公式类似**, 但 K2 对应输入 tile (不是滤波器), K3 对应滤波器 tile。

代码比较两个代价, 选小的: `if (cost_ws > cost_is) return IS; else return WS;`

#### 对例子的最终 CSA 结果

| 参数 | IS 调度 | 含义 |
|------|:---:|------|
| Nc = 32 | | 每次处理 32 个输入通道 |
| K2 = 32 | | L2 复用 32 组滤波器 tile |
| K3 = 128 | | L3 复用 128 组输入 tile |
| extra_k2 = 0, extra_k3 = 0, extra_tCH = 0 | | 无边界 |

---

### Stage 2: 边界处理

**代码: `SConv.cpp` 的 `treatEdgeTileConvolution`**

CSA 可能产生不能整除的余数:
```
RX = dimension_size mod tile_size   (非零则有边界)
```

对例子: `4096 % 16 = 0` (输入), `256 % 8 = 0` (滤波器), CSA 余数全 0 → **无边界**。

但如果有边界 (如 `Oh×Ow=5625, Nwin=16` → 余数 9):
- 沿 d2 维度 split 成: main (5616) + remainder (9)
- main 部分走完整 tiling+packing
- remainder 部分太小 (< Nwin), 只做 affine 索引修正

CSA 的 `extra_k2/extra_k3/extra_tCH` 非零时, 递归 split, 每个 split 出来的 kernel
独立走 tiling+packing。

---

### Stage 3: 两级分块

**代码: `SConv.cpp` 的 `applyTileTo` (第 1590 行)**

#### 外层 tiling (CSA 驱动)

IS 调度的 tile sizes (代码 `SConv.cpp:1606`):
```
tileSize = [1,                        // batch (不切)
            Nf × K2 = 8 × 32 = 256,   // output channels
            Nwin × K3 = 16 × 128 = 2048,  // spatial (Oh×Ow)
            Nc = 32,                   // input channels
            0, 0]                      // Fh, Fw (不切)
```

tile interchange (循环顺序, IS 调度, 代码 `SConv.cpp:1614`):
```
[0, 3, 2, 1]  →  batch, input_channels, spatial, output_channels
```

即外层循环顺序:
```
for batch (1)                     ← d0
  for input_channels (Nc=32)     ← d3
    for spatial (Nwin×K3=2048)  ← d2
      for output_channels (Nf×K2=256)  ← d1
```

#### 内层 tiling (微内核驱动)

```
innerTileSize = [0, Nf=8, Nwin=16, 0, 0, 0]
```

即在每个外层 tile 内部再切:
```
for spatial_inner (Nwin=16)   ← 每次处理 16 个窗口
  for filters_inner (Nf=8)    ← 每次处理 8 个滤波器
    → 微内核调用 (linalg.generic)
```

#### 循环交换修复

MLIR 的 `tileUsingSCF` 在 IS 调度下, `setInterchange` 对内层不生效。
代码 `swapInductionVars()` 手动交换内外循环的归纳变量和边界。

#### 整体循环结构 (IS 调度, 我们的例子)

```
for n = 0 to 1                          // batch (外层, 1次)
  for ic = 0 to 128 step 32             // input channels (4次)
    for sp = 0 to 4096 step 2048        // spatial outer (2次)
      for oc = 0 to 256 step 256        // output channels (1次)
        for sp_i = 0 to 2048 step 16    // spatial inner (128次)
          for flt_i = 0 to 256 step 8   // filter inner (32次)
            → 微内核: 16 windows × 8 filters × 288 reduction
```

每次微内核调用: `C[16, 8] += A[288, 16]^T × B[288, 8]`
(即 288 维归约, 16 个窗口 × 8 个滤波器的输出)

---

### Stage 4: 打包 (Packing)

**代码: `SConv.cpp` 的 `applyFilterPacking` / `applyInputPacking`**

打包把原始数据重新排列成微内核期望的内存布局。微内核做的是矩阵乘
`C[N, M] += B[K, N]^T × A[K, M]`, 所以 A 和 B 要按行连续存放。

#### 4a. 滤波器打包

**论文 Eq.1-2 / 代码 `applyFilterPacking` (第 751 行)**

原始滤波器 tile shape: `[Nf, Nc, Fh, Fw]` = `[8, 32, 3, 3]`
打包后 shape: `[Nc, Fh, Fw, Nf]` = `[32, 3, 3, 8]`
collapse 后: `[288, 8]` (= `[Nc×Fh×Fw, Nf]` = `[K, N]`)

**索引方程 (论文 Eq.3, 多重打包时)**:
```
iTf = iNt × Nf + iNf
```
其中 `iNt` 是 tile 维索引, `iNf` 是滤波器维内索引。

打包用 `linalg.generic` 实现, 遍历输出维度, 用 affine map 从原始 tensor.extract:
```mlir
// 遍历 [Nc, Fh, Fw, Nf], 从原始 [Nf, Nc, Fh, Fw] 提取
linalg.generic {
  indexing_maps = [identity_map],  // 输出恒等映射
  iterator_types = [parallel, parallel, parallel, parallel]
} outs(%packed : tensor<32x3x3x8xf32>) {
  ^bb0(%out: f32):
    %ic = linalg.index 0; %fh = linalg.index 1; %fw = linalg.index 2; %nf = linalg.index 3
    %val = tensor.extract %original_filter[%nf, %ic, %fh, %fw]
    linalg.yield %val
}
```

滤波器打包**无数据复制**, 只是维度重排 (转置)。

#### 4b. 输入打包

**论文 Eq.4-12 / 代码 `applyInputPacking` (第 873 行)**

原始输入 tile shape: `[N, Nc, (Nwin + Fw - 1)]` (考虑滑窗重叠)
打包后 shape: `[N, Nc, Fh, Fw, Nwin]`
collapse 后: `[N, K, Nwin]` (= `[N, Nc×Fh×Fw, Nwin]` = `[N, K, M]`)

输入打包**有数据复制**, 因为滑窗之间有重叠, 相同的输入元素会被复制到多个窗口。

**索引方程 (论文 Eq.6-8, 简单情况 — 窗口在同一行内)**:

```
iT_h = iFh × dilationH                                    (Eq.6)
iT_w = iNwin × strideW + iFw × dilationH                   (Eq.7)
iT_hw = iT_h × Tw + iT_w                                   (Eq.8)
```

其中 `iFh, iFw, iNwin` 是打包 tensor 的迭代变量, `Tw` 是 tile 内宽度。

**索引方程 (论文 Eq.9-12, 一般情况 — 窗口跨行)**:

当 tile 内的窗口序列跨越原始输入的行边界时, 需要知道当前窗口在完整输出中的绝对位置:

```
iTs = iOout + iOin + Ss          (Eq.9, 绝对窗口起始位置)
                                    Ss = edge offset (边界偏移, 无边界时为0)
iOhw = iTs + iNwin               (Eq.10, 当前窗口绝对位置)

iTh = (iOhw floordiv Ow - iTs floordiv Ow) × strideH + iFh × dilationH  (Eq.11)
iTw = (iOhw mod Ow - iTs mod Ow) × strideW + iFw × dilationW            (Eq.12)

iT_hw = iTh × Iw + iTw          (Eq.8, 原始输入的线性索引)
```

**关键理解**: `iOhw floordiv Ow` 把一维窗口序号转换成行号, `iOhw mod Ow` 转换成列号。
`iTs floordiv Ow` 是 tile 起始的行号。两者之差乘 stride 就是相对行偏移。

代码对应 `computeLinearInputIndices()` 函数 (`SConv.cpp:245`), 用 affine map 实现:

```mlir
// ILstart = IOin + IOout + Ss
ILstart = affine_map<(d0) -> (d0 + s0 + Ss)>(IOin, IOout)

// iHw = ((ILstart + nwin) floordiv Ow - ILstart floordiv Ow) * strideH * Iw
//     + ((ILstart + nwin) mod Ow - ILstart mod Ow) * strideW
//     + fh * dilationH * Iw + fw * dilationW
iHw_map = affine_map<(d0,d1,d2)[s0] ->
    (((s0 + d2) floordiv Ow - s0 floordiv Ow) * strideH + d0 * dilationH) * Iw
   + (((s0 + d2) mod Ow - s0 mod Ow) * strideW + d1 * dilationW)>
```

#### 4c. 多重打包 (Multipacking)

**论文 §4.2.1 / 代码 `inputMultipackingOpt` (WS) / `filterMultipackingOpt` (IS)**

把内层 packing 提升到外层, 一次打包 K2 个 tile:

**IS 调度 → 滤波器多重打包**:
```
packed_filter shape: [K2, Nc, Fh, Fw, Nf]  → collapse → [K2, K, Nf]
索引: iTf = iNt × Nf + iNf  (Eq.3)
```

**WS 调度 → 输入多重打包**:
```
packed_input shape: [N, K2, Nc, Fh, Fw, Nwin]  → collapse → [N, K2, K, Nwin]
索引: iOhw = iOout + iNt × Nwin + iNwin  (Eq.14)
```

#### 4d. Delinearization

打包完成后, 把塌缩的空间维度恢复成原始 NCHW 形状, 以兼容下游 pass。

---

### Stage 5: 微内核替换

**代码: `lib/LowerToBLAS.cpp`**

#### 5a. 微内核的结构 (adjustLinalgOps)

打包后, 微内核 (`linalg.generic`) 被 `adjustLinalgOps()` 重写为外积形式
(代码 `SConv.cpp:669-673`):

```mlir
linalg.generic {
  // 4 个循环维度: batch(par), filter(par), window(par), reduction(red)
  indexing_maps = [
    affine_map<(d0,d1,d2,d3) -> (d0, d3, d2)>,   // 输入: [N, K, M] → (batch, red, window)
    affine_map<(d0,d1,d2,d3) -> (d3, d1)>,        // 权重: [K, N] → (red, filter)
    affine_map<(d0,d1,d2,d3) -> (d0, d1, d2)>      // 输出: [N, N, M] → (batch, filter, window)
  ],
  iterator_types = [parallel, parallel, parallel, reduction]
} {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %mul, %c : f32
    linalg.yield %add : f32
}
```

计算: `C[batch, filter, window] += Σ_k A[batch, k, window] × B[k, filter]`

对 batch=0: `C[n, m] += Σ_k A[k, m] × B[k, n]`

这是一个矩阵乘, 但输出布局是 `[N, M]` (filter × window, 不是 window × filter)。

#### 5b. BLAS 调用替换

`lower.to_blas` 操作 (`LowerToBLAS.cpp`) 把 `linalg.generic` 替换为:

```mlir
// 从 memref 提取基地址和偏移
%base_in  = memref.extract_aligned_pointer_as_index %in_tile
%offset_in = memref.extract_strided_metadata %in_tile  →  offset
%ptr_in  = inttoptr(%base_in + offset_in × 4) : i64 to !llvm.ptr

// 同理提取 weight 和 output 的指针

// 调用 sgemm_blas_kernel
%call = func.call @sgemm_blas_kernel(
    %m,    %n,    %k,    %alpha,
    %A,    %B,    %C,    %ldc) :
    (i64, i64, i64, f32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
```

参数对应 (代码 `LowerToBLAS.cpp:196-203`):
```
m = inShape[2] = Nwin = 16      (窗口数 = GEMM 的 M)
n = fsShape[1] = Nf = 8         (滤波器数 = GEMM 的 N)
k = fsShape[0] = K = 288        (归约维度 = Ic×Fh×Fw)
A = input tile 指针              (布局 [K, M], row-major)
B = filter tile 指针             (布局 [K, N], row-major)
C = output tile 指针             (布局 [N, M], row-major)
ldc = output strides[1] = M     (输出 leading dimension)
alpha = 1.0
```

#### 5c. sgemm_blas_kernel 包装函数

SConv 微内核计算: `C[n][m] += Σ_k A[k][m] × B[k][n]`

在矩阵符号中: `C = B^T × A + C` (不是标准的 `C = A × B`)

因此 cblas_sgemm 调用 (我们的包装函数 `runtime/sgemm_blas_kernel.c`):
```c
cblas_sgemm(CblasRowMajor,    // 行主序
            CblasTrans,        // B 转置: B[K,N] → B^T[N,K]
            CblasNoTrans,      // A 不转置: A[K,M]
            n, m, k,           // 输出维度: N×M, 归约 K
            alpha,             // 1.0
            B, n,              // B^T 的 leading dim = N (原始 B 的列数)
            A, m,              // A 的 leading dim = M
            1.0f,              // beta = 1.0 (累加)
            C, ldc);           // C 的 leading dimension
```

**验证**: `cblas_sgemm` 计算 `C[i,j] = alpha × Σ_k B^T[i,k] × A[k,j] + beta × C[i,j]`
= `alpha × Σ_k B[k,i] × A[k,j] + beta × C[i,j]`
= `Σ_k B[k,n] × A[k,m] + C[n,m]` ✓ (i=n, j=m, alpha=1, beta=1)

---

## 3. 完整执行流程 (实际跑的例子)

用 `custom_0` (4×192×40×40 → 192×192×3×3, stride=1) 为例:

```
Step 1: 生成 payload
  gen_payload.py 从 CSV 生成 conv_custom_0.mlir
  (含 main 函数: 生成数据 → warmup → 计时 → 打印)

Step 2: SConv transform
  sconv-opt -transform=sconv-blas.mlir conv_custom_0.mlir
  → Stage 1-4: tiling + packing
  → Stage 5a: bufferize
  → Stage 5b: lower.to_blas (linalg.generic → func.call @sgemm_blas_kernel)
  → 输出: conv_custom_0.tf.mlir

Step 3: Native lowering
  mlir-opt conv_custom_0.tf.mlir [标准 lowering passes] -o tf.llvm.mlir
  (scf → cf, linalg → affine, memref → LLVM, func → LLVM)

Step 4: JIT 执行
  mlir-cpu-runner tf.llvm.mlir -e main -shared-libs=...
  (链接 libmlir_runner_utils + librtclock + libsgemm_blas_kernel + libopenblas)

Step 5: 输出
  stderr: "6265.806 ms" (printTime)
  stderr: "0.17 GFLOPS" (printFlops, baseline)
  stdout: memref data (printMemrefF32)
```

Baseline 路径 (跳过 Step 2, 直接 Step 1 → 3 → 4 → 5):
```
mlir-opt conv_custom_0.mlir [标准 lowering] → mlir-cpu-runner
→ linalg.conv 直接展开成标量 for 循环 (无优化)
```

---

## 4. 测试结果

### 自定义大卷积 (batch=4, warmup=1, runs=1)

| 卷积 | FLOPs | Baseline(ms) | SConv+BLAS(ms) | 加速比 |
|------|------:|---:|---:|---:|
| custom_0 (192ch,40²) | 4.25G | 6,266 | 78 | 81x |
| custom_1 (768ch,80²→40²) | 67.95G | 98,954 | 1,127 | **88x** |
| custom_4 (384ch,160²→80²) | 67.95G | 98,560 | 1,177 | 84x |
| custom_7 (192ch,20²) | 1.06G | 1,557 | 20 | 77x |
| custom_5 (48ch,160²) | 4.25G | 5,959 | 115 | 52x |
| ... | ... | ... | ... | ... |
| **总计** | — | **352.8s** | **4.4s** | **~80x** |

### 小卷积 (batch=1, warmup=5, runs=10)

| 卷积 | FLOPs | Baseline(ms) | SConv+BLAS(ms) | 加速比 |
|------|------:|---:|---:|---:|
| conv_1800 | 2.3M | 2.75 | 0.14 | 19x |
| conv_3557 | 0.6M | 0.77 | 0.11 | 7x |
| ... | ... | ... | ... | ... |

全部 16 个卷积**正确性验证通过** (numpy allclose, rtol=1e-3)。

---

## 5. 代码结构速查

```
include/SConv.td           ← 定义 transform.structured.sconv 和 transform.lower.to_blas
include/CSA.h              ← CSA 数据结构 (ConvInfo, ArchInfo, mKInfo, CSAStrategy)
lib/CSA.cpp                ← 代价模型: IS/WS 的缓存层次分析
lib/SConv.cpp (2493行)     ← 核心流水线:
  SConvOp::apply()              入口 (第 2232 行)
  treatEdgeTileConvolution()    边界处理 (第 2111 行)
  splitAndTileConvolution()     递归 split (第 1942 行)
  applyTileTo()                 两级 tiling + packing 入口 (第 1590 行)
  swapInductionVars()           修复 MLIR tiler 的 loop interchange bug (第 1015 行)
  promoteOpsOfTile()            提升 affine.apply 和 extract_slice (第 420 行)
  applyFilterPacking()          滤波器打包 (第 751 行)
  applyInputPacking()           输入打包 (第 873 行)
  adjustLinalgOps()             重写微内核为外积形式 (第 614 行)
  filterMultipackingOpt()      IS 多重打包 (第 1357 行)
  inputMultipackingOpt()       WS 多重打包 (第 1092 行)
  computeLinearInputIndices()   输入打包的 affine 索引公式 (第 245 行)
  computeMultiPackInputIndices() 多重打包的索引公式 (第 297 行)
lib/LowerToBLAS.cpp        ← 微内核 → BLAS 调用
sconv-opt/sconv-opt.cpp    ← 命令行工具
scripts/convbench/         ← 测试框架
  run_convbench.py             自动化测试
  runtime/sgemm_blas_kernel.c  OpenBLAS 包装
  runtime/rtclock.c            计时函数
```

---

## 6. 关键设计决策

### 为什么用 Transform Dialect 而不是 Pass?

Pass 是黑盒; Transform Dialect 是声明式的, 中间 IR 始终合法可打印。

### 为什么不直接用 Im2Col?

Im2Col 内存膨胀 ~8x; SConv 按需打包, 每次 tile 只有 KB 级。

### 为什么用 OpenBLAS 而不是 MLIR Vector dialect?

MLIR Vector dialect 在某些架构 (Power10) 上没有成熟后端; OpenBLAS 到处都有。

---

## 7. 局限性

1. 只支持 NCHW, 不支持 NHWC 和 grouped conv
2. 边界 tile 跳过 tiling/packing, 性能不如主部分
3. swapInductionVars 有 SSA bug (我们修了)
4. 强依赖 LLVM 19 (LLVM 20 API 不兼容)
5. 外层 tiling 未优化, repacking 有开销

---

## 8. 术语表

| 术语 | 含义 |
|------|------|
| CSA | 根据卷积大小和机器缓存, 算出 tile 大小的分析器 |
| CSO | 根据 CSA 结果, 生成循环嵌套的代码生成器 |
| Tiling | 把大循环切成小块, 让数据放进缓存 |
| Packing | 把数据按微内核期望的布局重新排列 |
| Multipacking | 一次打包多个 tile, 提升外层循环的数据复用 |
| IS | 输入 tile 驻留 L1, 滤波器 tile 换着用 |
| WS | 滤波器 tile 驻留 L1, 输入 tile 换着用 |
| Nwin | 微内核每次处理的窗口数 (默认 16) |
| Nf | 微内核每次处理的滤波器数 (默认 8) |
| Nc | 每次处理的输入通道数 (L1 驱动) |
| K2 / K3 | L2 / L3 层的 tile 复用数 |
| Microkernel | 最内层矩阵乘, 调用 OpenBLAS sgemm |
