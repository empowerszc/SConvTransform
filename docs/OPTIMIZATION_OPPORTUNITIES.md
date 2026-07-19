# SConvTransform 在 aarch64 + SVE/SME 上的优化机会

> 本文探讨在支持 SVE (Scalable Vector Extension) 和 SME (Scalable Matrix Extension) 的
> aarch64 CPU 上, SConvTransform 还有哪些优化空间, 以及大致的实现思路。

---

## 现状: 当前 SConvTransform 的性能瓶颈

在 Apple M4 (ARM64, SME) 上的实测数据:

| 指标 | 值 | 说明 |
|------|---|------|
| SConv+BLAS 最高 GFLOPS | ~15 GFLOPS | Homebrew OpenBLAS, 无 SME 优化 |
| M4 SME 峰值 | 502 GFLOPS | 论文用微基准测得 (32 条 FMOPA 指令) |
| 峰值占比 | ~3% | 差距巨大 |
| Baseline (native lowering) | ~0.17 GFLOPS | 标量循环, 无优化 |

论文报告的最佳结果是峰值 59.6% (≈300 GFLOPS), 差距来自:

1. **OpenBLAS 后端**: 我们用的 Homebrew 通用版没有 SME 专用微内核
2. **单线程**: 整个计算是 `scf.for` 串行循环, 没有多核并行
3. **打包开销**: 每次打包用标量 `tensor.extract`, 没有 SIMD 加速
4. **外层 tiling 未优化**: 每个 macro tile 都做 repacking, 无流水线

下面逐项分析优化机会和思路。

---

## 1. 多核并行 (最直接的加速)

### 问题

当前代码全部用 `scf.for` (串行循环), 没有任何并行。M4 有 10 个 CPU 核, 只用了 1 个。

### 机会

卷积的外层循环天然可并行:

```
for batch (N=4)            ← 各 batch 独立, 可并行
  for ic_tile (tCH=4)      ← 各通道段独立 (但会共享 L2/L3)
    for sp_tile (K3=128)   ← 各空间 tile 独立 (共享 L3)
      for oc_tile (K2=32) ← 各滤波器 tile 独立 (共享 L2)
        ...微内核...
```

最安全的并行维度是 **batch** 和 **sp_tile (空间 tile)**, 因为它们之间没有数据依赖。

### 实现思路

**方案 A: MLIR `scf.parallel` (最简单)**

把最外层的 `scf.for` 替换为 `scf.parallel`:

```mlir
// 当前 (串行):
scf.for %n = 0 to 4 step 1 {
  // ... 每个 batch 的卷积计算
}

// 优化后 (并行):
scf.parallel (%n) = 0 to 4 step 1 {
  // ... 每个 batch 的卷积计算
}
```

`scf.parallel` 在 lowering 到 LLVM 时, 可以用 OpenMP runtime 或 pthread 实现。

代码改动位置: `SConv.cpp` 的 `applyTileTo()` 中, 把 `scf::tileUsingSCF` 生成的最外层 `scf.for` 改为 `scf.parallel`。或者更优雅地, 在 Transform IR 里加一个 `transform.structured.tile_to_parallel` 操作。

**方案 B: OpenMP (编译时指定)**

在 `mlir-opt` lowering 阶段加 `--convert-scf-to-openmp`:

```bash
mlir-opt input.mlir \
  --one-shot-bufferize=... \
  --convert-linalg-to-affine-loops \
  --convert-scf-to-openmp    # ← 新增
  ...
```

需要在 payload 的 `scf.for` 上加 `loop_schedule = "static"` 等属性, 或在 Transform IR 里用 `transform.loop.coalesce` + `transform.loop.parallelize`。

**方案 C: 手写线程池 (绕过 MLIR)**

在 `sgemm_blas_kernel` 包装层实现多线程: 把多个 macro tile 分发到不同线程。但这需要修改 SConv 的 tiling 结构, 在外层循环中插入线程分发逻辑。

### 预期收益

4 核并行 → ~3x 加速 (考虑缓存竞争和同步开销); 10 核 → ~5-7x。

---

## 2. SME 专用微内核 (最大潜力)

### 问题

当前微内核调用 OpenBLAS 的 `cblas_sgemm`, 这是一个通用函数入口, 内部会判断矩阵尺寸选择不同的实现路径。对于 SConv 产生的小 tile (如 288×16 × 288×8), 调用开销占比高, 且 Homebrew OpenBLAS 不含 SME 指令。

### SME 简介

ARM SME (Scalable Matrix Extension) 提供了**矩阵级指令**:

| 指令 | 功能 | 吞吐 |
|------|------|------|
| `FMOPA` | 16×16 外积累加: `ZA += ZA + V1 × V2^T` | 每个 tick 16×16=256 个 FMA |
| `FMOPS` | 16×8 外积累加 | 128 个 FMA |
| `ADDHA` | 矩阵 tile 加法 | — |
| `LD1R` | 广播加载 | — |

SME 的核心优势: **一条指令完成 16×16 矩阵乘加**, 而 SVE2 的 `FMLA` 一条指令只能做 4×4 (128-bit) 或 8×8 (SVE 512-bit)。

### 机会: 用 MLIR Vector dialect → ArmSME dialect

MLIR 已经有 ArmSME dialect (`mlir/Dialect/ArmSME/`), 可以从 Vector dialect 自动 lowering:

```
linalg.generic (微内核)
    → vector.contract (转为向量操作)
    → vector.maskedload/maskedstore (向量化访存)
    → arm_sme.fmopa_32 (SME 矩阵指令)
    → LLVM ARM intrinsics
```

### 实现思路

**Step 1: 把 `linalg.generic` 微内核转为 `vector.contract`**

在 Stage 5 中, 不再调用 `lower.to_blas`, 而是用 MLIR 的向量化:

```mlir
// 当前: 替换为 BLAS 调用
%call = func.call @sgemm_blas_kernel(%m, %n, %k, %alpha, %A, %B, %C, %ldc)

// 优化: 转为 vector.contract, 让 MLIR 自己向量化
%va = vector.transfer_read %A[%k_off, %m_off] : memref<288x16xf32>, vector<288x16xf32>
%vb = vector.transfer_read %B[%k_off, %n_off] : memref<288x8xf32>, vector<288x8xf32>
%vc = vector.transfer_read %C[%n_off, %m_off] : memref<8x16xf32>, vector<8x16xf32>
%res = vector.contract {indexing_maps=..., iterator_types=[parallel,parallel,reduction]}
    %va, %vb, %vc : vector<288x16xf32>, vector<288x8xf32> -> vector<8x16xf32>
vector.transfer_write %res, %C[%n_off, %m_off] : vector<8x16xf32>, memref<8x16xf32>
```

**Step 2: 调整 lowering pipeline**

在 `mlir-opt` 的 lowering pipeline 中加入 ArmSME 相关 pass:

```bash
mlir-opt input.mlir \
  --convert-vector-to-arm-sme       # vector.contract → arm_sme.fmopa
  --convert-arm-sme-to-llvm         # arm_sme → LLVM intrinsics
  --convert-vector-to-llvm          # 剩余 vector ops → LLVM
  ...
```

**Step 3: 调整 Nwin/Nf 匹配 SME tile 大小**

SME 的矩阵 tile 是 16×16 (或可扩展)。当前 Nwin=16, Nf=8, 可以考虑:
- Nf=16 (匹配 SME 16 列): `C[16,16] += A[K,16]^T × B[K,16]`, 每次 FMOPA 正好填满一个 tile
- 或保持 Nf=8, 用 `FMOPS` (16×8 半 tile)

代码改动: `SConvOp::apply()` 中的默认 `mKInfo = {16, 8, 128}`, 改为 `{16, 16, 256}`。

### 预期收益

论文报告 SME 优化后达到 502 GFLOPS 峰值的 ~60% = ~300 GFLOPS。
相比当前 15 GFLOPS, 潜在 **20x 加速**。

---

## 3. 向量化打包 (Vector-based Packing)

### 问题

当前打包用标量 `tensor.extract` 逐元素复制:

```mlir
linalg.generic {
  // 逐元素: 每次复制 1 个 float
  ^bb0(%out: f32):
    %val = tensor.extract %input[%i, %j, %k] : tensor<...>
    linalg.yield %val : f32
}
```

对于输入打包, 滑窗之间有大量数据重叠 (stride < kernel), 逐元素复制效率很低。

### 机会

前作论文 [Ferrari et al., TACO 2023] 提出了 **Vector-based Packing**: 用 SVE 向量操作来减少数据复制:

```
原始: 逐元素 extract (每个元素单独读取)
优化: 用向量移位和拼接, 从上一窗口的向量"滑"出下一窗口

  上一窗口向量: [a b c | d e f g h i]   (3×3 kernel, 一行)
  下一窗口向量: [  b c d | e f g h i j]  (只读 1 个新元素 j, 其余移位复用)
```

SVE 的 `tbl` (table lookup) 和 `splice` (向量拼接) 指令可以实现这种"滑动窗口"式打包。

### 实现思路

**方案 A: 在 MLIR Vector dialect 层实现**

把打包的 `linalg.generic` 替换为向量化版本:

```mlir
// 当前 (标量):
%val = tensor.extract %input[%ic, %ih, %iw] : tensor<...xf32>

// 优化 (向量化):
%vec = vector.transfer_read %input[%ic, %ih, %iw] : vector<16xf32>
%shifted = vector.shuffle %prev_vec, %vec, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
// 16 个元素只需读 1 个新的, 其余从上一窗口移位复用
vector.transfer_write %shifted, %packed[%ic, %fh, %fw, %nwin_off]
```

然后 lowering 到 SVE:
```
vector.shuffle → SVE tbl / ext (extract + concatenate)
vector.transfer_read → SVE ld1
```

**方案 B: 用 SVE intrinsic 手写打包函数**

类似 `sgemm_blas_kernel.c`, 写一个 C 打包函数, 用 `arm_sve.h` intrinsic:

```c
#include <arm_sve.h>
void packed_input_sve(float* dst, float* src, int nwin, int Iw, ...) {
    svfloat32_t prev = svdup_f32(0.0);
    for (int w = 0; w < nwin; w += svcntw()) {
        svfloat32_t curr = svld1_f32(svptrue_b32(), src + w * stride);
        // 移位拼接: 去掉前 stride 个, 补上新元素
        svfloat32_t packed = svext_f32(prev, curr, stride);
        svst1_f32(svptrue_b32(), dst + w, packed);
        prev = curr;
    }
}
```

### 预期收益

打包阶段加速 4-8x (SVE 向量化), 减少内存复制量 50%+ (滑动窗口复用)。

---

## 4. 流水线 (Packing + Compute 重叠)

### 4.0 当前循环结构和 buffer 分配 (ping-pong 可行性分析)

先看当前代码在 IS 调度下, 每次内层迭代实际做了什么:

```
for batch
  for ic_tile (step Nc=32)
    for sp_outer (step K3×Nwin=2048)
      for oc_outer (step K2×Nf=256)
        // --- 滤波器多重打包 (做一次, hoisted 到这层) ---
        // tensor.empty → 分配 buffer: [K2, Nc, Fh, Fw, Nf] = [32, 32, 3, 3, 8]
        //   = 288 KB (放 L2)
        // linalg.generic → 填充数据
        // collapse_shape → [288, 8]... 不对, 是 [K2×288, Nf] = [9216, 8]

        for sp_inner (step Nwin=16)         ← ★ ping-pong 的目标层
          for oc_inner (step Nf=8)
            // --- 输入打包 (每次迭代都做!) ---
            // SConv.cpp:941  tensor.empty() 分配 buffer
            //   shape: [N, Nc, Fh, Fw, Nwin] = [1, 32, 3, 3, 16]
            //   大小: 32×3×3×16×4 = 18,432 bytes = 18 KB  ← 放得进 L1!
            // linalg.generic → 逐元素从原始输入提取数据填入
            // collapse_shape → [1, 288, 16] = [1, K, M]

            // --- 微内核调用 ---
            // linalg.generic → sgemm_blas_kernel
            //   C[16, 8] += A[288, 16]^T × B[288, 8]
            //   读 packed_input (18 KB) + packed_filter, 写 output tile
```

**关键观察**:

1. **每次 `sp_inner` 迭代**: 先打包 (读输入 → 写 buffer), 再计算 (读 buffer → 写输出)
2. **打包是内存密集型**: 逐元素 `tensor.extract`, 读原始输入, 写打包 buffer
3. **微内核是计算密集型**: BLAS `sgemm`, 读打包 buffer + 滤波器, 写输出
4. **两者访问不同数据**: 打包读原始输入, 微内核读打包 buffer — **天然可重叠**
5. **buffer 只有 18 KB**: 两个 ping-pong buffer = 36 KB, 轻松放进 L1 (29 KB×0.9 实际可用... 嗯, 刚好溢出, 但 18 KB 单个是够的)

### 4.1 ping-pong 的核心思路

当前 (串行):
```
迭代 0:  [Pack tile 0] → [Compute tile 0] → [Pack tile 1] → [Compute tile 1] → ...
                    ↑ 串行等待 ↑               ↑ 串行等待 ↑
```

优化后 (ping-pong):
```
迭代 0:  [Pack tile 0 → buf A]
                            ↓
         [Compute tile 0 ← buf A]     [Pack tile 1 → buf B]    ← 同时进行!
                                       ↓
                                   [Compute tile 1 ← buf B]   [Pack tile 2 → buf A]
                                                             ↑ 交替使用两个 buffer
```

打包第 N+1 个 tile 的同时, 用第 N 个 tile 做计算。

### 4.2 具体实现方案

#### 方案 A: MLIR Async dialect (多线程)

把打包和计算分别放进 `async.execute`, 用双缓冲交替:

```mlir
// 在 sp_inner 循环之前: 预分配两个 buffer
%buf_A = memref.alloc() : memref<1x32x3x3x16xf32>   // 18 KB
%buf_B = memref.alloc() : memref<1x32x3x3x16xf32>   // 18 KB

// 迭代 0: 先打包到 buf_A
%token_pack_0 = async.execute {
  pack(%input, %tile_idx_0, %buf_A)
  async.yield %buf_A
}

scf.for %i = 0 to N step 1 {
  // 当前迭代的 packed data (从上一次打包得到)
  %packed_current = async.await %token_pack_current

  // 异步打包下一个 tile 到另一个 buffer
  %next_buf = scf.if (%i even) -> memref<...> { scf.yield %buf_B } else { scf.yield %buf_A }
  %token_pack_next = async.execute {
    pack(%input, %tile_idx_(%i+1), %next_buf)
    async.yield %next_buf
  }

  // 同步: 计算当前 tile (用 packed_current)
  compute_sgemm(%packed_current, %filter, %output_tile)

  // 交换 token
  %token_pack_current = %token_pack_next
}

// 最后一次: await 最后的 pack, 做最后一次 compute
%packed_last = async.await %token_pack_current
compute_sgemm(%packed_last, %filter, %output_tile)
```

需要的 lowering:
```bash
mlir-opt ... --convert-async-to-llvm   # async → LLVM coroutine + 线程池
```

#### 方案 B: 软件流水线 (单线程, 编译器调度)

不用多线程, 而是让编译器把打包指令和计算指令交错排列:

```
原始循环体:
  pack(tile_i)        ← 读输入, 写 buf
  compute(buf)        ← 读 buf, 写输出

软件流水后:
  pack(tile_0) → buf_A                    ← 预热
  loop:
    compute(buf_A)  ||  pack(tile_i+1) → buf_B   ← 交错
    compute(buf_B)  ||  pack(tile_i+2) → buf_A
  compute(buf_last)                       ← 排空
```

在 MLIR 层面, 可以用 `transform.loop.pipeline` 或手动构造 pipeline 形式的 IR。

#### 方案 C: 手写 C 包装层 (最简单, 不改 MLIR)

在 `sgemm_blas_kernel.c` 里实现双缓冲:

```c
// 预分配两个 L1 大小的 buffer
static float pack_buf[2][MAX_K * MAX_M];  // 288×16 = 4608 floats = 18 KB each

// 打包函数 (SVE 加速)
void pack_input_tile(float* src, float* dst, int K, int M, ...);

// 双缓冲计算函数
int sgemm_blas_kernel_pingpong(...) {
    int buf = 0;
    // 预打包第一个 tile
    pack_input_tile(src + tile0_offset, pack_buf[0], K, M, ...);

    for (int i = 0; i < num_tiles; i++) {
        // 异步打包下一个 tile 到另一个 buffer
        #pragma omp task  // 或用另一线程
        pack_input_tile(src + tile(i+1)_offset, pack_buf[!buf], K, M, ...);

        // 用当前 buffer 计算
        cblas_sgemm(..., pack_buf[buf], ...);

        // 等待打包完成
        #pragma omp taskwait
        buf = !buf;  // 交换
    }
}
```

这种方案不改 MLIR 代码, 只改 `sgemm_blas_kernel.c`。但需要把多个 microkernel 调用合并成一个函数 (因为 ping-pong 需要跨迭代管理 buffer)。

### 4.3 代码改动位置

| 方案 | 改动文件 | 改动内容 |
|------|---------|---------|
| A (async) | `SConv.cpp` `applyInputPacking()` + `applyTileTo()` | 把 `tensor.empty` 移到循环外, 加 `async.execute` 包装打包, 加 `async.await` |
| B (软件流水) | Transform IR | 加 `transform.loop.pipeline` 操作 |
| C (C 包装层) | `runtime/sgemm_blas_kernel.c` | 重写为双缓冲版本, 把多个 tile 的打包+计算合并 |

### 4.4 预期收益分析

假设打包占 30% 时间, 计算占 70%:

| 场景 | 打包时间 | 计算时间 | 总时间 | 加速 |
|------|:---:|:---:|:---:|:---:|
| 当前 (串行) | 30% | 70% | 100% | 1x |
| ping-pong (完美隐藏) | 0% (被隐藏) | 70% | 70% | 1.43x |
| ping-pong (部分隐藏) | 10% | 70% | 80% | 1.25x |

如果同时配合向量化打包 (SVE, 把打包时间从 30% 降到 15%):

| 场景 | 打包 | 计算 | 总时间 | 加速 |
|------|:---:|:---:|:---:|:---:|
| 当前 | 30% | 70% | 100% | 1x |
| SVE 打包 + ping-pong | 0% | 70% | 70% | 1.43x |
| SVE 打包, 无 ping-pong | 15% | 70% | 85% | 1.18x |

### 4.5 还有第二层 ping-pong 机会 (多重打包层)

在 `oc_outer` 层, 滤波器多重打包也是每次迭代分配 + 填充:

```
for oc_outer (step K2×Nf):
  // 分配 multipack buffer: [K2, Nc, Fh, Fw, Nf] = 288 KB
  // 填充: 从原始滤波器提取 K2 个 tile
  // 然后用这个 buffer 跑 K3/Nwin 次 sp_inner 迭代
```

这里也可以 ping-pong: 在用当前 multipack buffer 跑 sp_inner 循环的同时, 异步打包下一个 oc_outer 的 multipack。

但 buffer 更大 (288 KB × 2 = 576 KB), 需要确认能放进 L2。对 M4 (L2 = 16 MB shared) 没问题; 对小 L2 的机器可能不够。

| 层级 | buffer 大小 | 两倍 buffer | 能放进 |
|------|:---:|:---:|---|
| sp_inner (输入打包) | 18 KB | 36 KB | L1 |
| oc_outer (滤波器多重打包) | 288 KB | 576 KB | L2 |

**两层 ping-pong 可以叠加**: 外层滤波器 multipack + 内层输入 pack, 各自双缓冲, 理论上能隐藏全部打包时间。

---

## 5. 自动调参 (Auto-tuning)

### 问题

CSA 用固定启发式 (半值法) 选 Nc/K2/K3, 不一定是特定硬件上的最优值。

例如, M4 的 L2 是 16MB shared (很大), 当前 CSA 只用了 1MB 的假想值; 如果用真实的 16MB, K2 可以更大, 减少滤波器 tile 的重复加载。

### 机会

**思路 A: 暴力搜索**

对给定卷积, 搜索所有合理的 (Nc, K2, K3) 组合, 实际跑一次取最快:

```python
for Nc in [16, 32, 64, 128]:
    for K2 in [8, 16, 32, 64]:
        for K3 in [32, 64, 128, 256]:
            time = run_sconv(conv, Nc, K2, K3)
            best = min(best, time)
```

SConv 的 `mK_info` 和 `arch_info` 属性已经支持外部传参, 不用改代码:

```mlir
transform.structured.sconv %conv
  { mK_info = [16, 16],        // Nwin=16, Nf=16 (匹配 SME)
    arch_info = [32768, 16777216, 0, 128],  // L1=32K, L2=16M, L3=0, line=128
    latency = [2, 10, 30, 300] }
```

**思路 B: 代价模型改进**

CSA 当前的代价模型只算 cache line 数量, 不考虑:
- TLB miss (大 tile 可能跨页)
- 预取效果 (连续访问模式)
- BLAS 调用开销 (小 tile 的函数调用开销占比高)

可以在代价模型中加入这些因素, 或者用 ML 预测 (类似 TVM/Ansor)。

### 预期收益

根据硬件特性调参 → **1.2-2x** 加速 (取决于默认参数离最优有多远)。

---

## 6. 内存布局优化 (NHWC)

### 问题

SConv 只支持 NCHW (batch, channel, height, width), 但在 aarch64 上 NHWC 布局更 cache-friendly:

- NCHW: 同一空间位置的多个通道不连续 (跨步大)
- NHWC: 同一空间位置的所有通道连续 (cache line 友好)

### 机会

改用 NHWC 后, 打包时的输入读取可以连续加载, SVE 的 `ld1` 指令可以一次加载整个 cache line。

### 实现思路

需要在 `linalg.conv_2d_nhwc_fhwc` 上工作 (MLIR 有对应的 named op), 修改所有 affine map 和打包公式。

改动量较大: `SConv.cpp` 中所有索引公式都要改。但论文的算法框架不变, 只是坐标计算不同。

### 预期收益

打包阶段访存效率提升 → **1.3-1.5x** 加速。

---

## 7. 边界处理优化

### 问题

当前边界 tile (小于 Nwin 或 Nf 的部分) 跳过 tiling/packing, 只做 affine 修正, 性能远不如主部分。且边界处理在 tiling 之前, 导致重复的 packing 逻辑。

### 机会

**思路: 自动 padding**

在 tiling 前对输入/输出 tensor 做自动 padding, 使所有维度整除 tile 大小:

```
原始: Oh×Ow = 4096, Nwin = 16 → 整除, 无边界
       但 Oh×Ow = 5625, Nwin = 16 → 余 9, 有边界

padding 后: Oh×Ow = 5632 (= 352×16), 整除
  → 所有 tile 走完整 tiling+packing 流水线
  → 计算完成后裁剪掉 padding 部分
```

代码改动: 在 `SConvOp::apply()` 的 tiling 之前, 检测余数, 如果非零则插入 `tensor.pad` 操作。

### 预期收益

消除边界处理开销, 消除重复 packing 逻辑 → **1.1-1.3x** 加速 (取决于边界 tile 占比)。

---

## 优化优先级和预期收益总结

| 优化 | 实现难度 | 预期加速 | 改动位置 | 依赖 |
|------|:---:|:---:|---|---|
| **1. 多核并行** | 低 | 3-7x | `scf.for → scf.parallel` | OpenMP runtime |
| **2. SME 微内核** | 高 | 10-20x | `lower.to_blas → vector.contract → arm_sme` | MLIR ArmSME dialect |
| **3. 向量化打包** | 中 | 2-4x | 打包 `linalg.generic → vector.transfer` | SVE intrinsics |
| **4. 流水线** | 中 | 1.4x | `scf.for → async.execute` | MLIR Async dialect |
| **5. 自动调参** | 低 | 1.2-2x | `arch_info` 参数 | 无 |
| **6. NHWC 布局** | 高 | 1.3-1.5x | 全部 affine map 和打包公式 | `linalg.conv_2d_nhwc` |
| **7. 自动 padding** | 低 | 1.1-1.3x | tiling 前加 `tensor.pad` | 无 |

**如果全部实现, 理论加速比**: 3-7x (并行) × 10-20x (SME) × 2-4x (打包) × 1.4x (流水线) × 1.5x (调参+布局+padding) ≈ **125-588x** 相比当前单线程 OpenBLAS 版本。

实际上优化之间有重叠 (比如 SME 微内核可能已经包含向量化打包的效果), 合理预期:
- **短期 (多核 + 调参)**: 当前 15 GFLOPS → 50-100 GFLOPS
- **中期 (+ SME 微内核)**: → 200-300 GFLOPS (接近论文水平)
- **长期 (+ 流水线 + NHWC + padding)**: → 300-400 GFLOPS

---

## 快速验证 SME 可用性 (OpenEuler aarch64)

在目标机器上检查 SME/SVE 支持:

```bash
# 检查 SVE
cat /proc/cpuinfo | grep -o 'sve[0-9]*' | head
# 检查 SME
cat /proc/cpuinfo | grep -o 'sme' | head
# 检查 SVE 向量长度
prctl --sve-vector-length  # 或: sysreg S3_0_C4_C2_4

# 编译含 SME 指令的测试程序
cat > /tmp/sme_test.c << 'EOF'
#include <arm_sve.h>
#include <stdio.h>
int main() {
    if (svevl1() > 0) {
        printf("SVE active, vector length = %d bits\n", svcntb()*8);
    }
    return 0;
}
EOF
clang -o /tmp/sme_test /tmp/sme_test.c -march=armv9-a+sve+sme && /tmp/sme_test
```

如果 SME 可用, 编译 OpenBLAS 时加 `TARGET=ARMV8SVE` 或 `TARGET=ARMV9SME`:

```bash
make TARGET=ARMV9SME DYNAMIC_ARCH=1 -j$(nproc)
```

这样 OpenBLAS 内部的 `sgemm_kernel` 就会用 SME 指令, SConv 的 `sgemm_blas_kernel` 包装层不用改, 直接受益。
