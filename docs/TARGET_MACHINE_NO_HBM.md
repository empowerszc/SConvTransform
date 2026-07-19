# 目标机器分析 (无 HBM 场景): 608 核 aarch64 SVE512+SME OpenEuler

> **本文分析不使用 HBM 的场景** (仅 L1 + L2 + DDR, 无任何 L3 层)。
> 用于理解无 L3 缓存时 CSA 的行为, 以及为什么需要 HBM。
>
> 含 HBM 的完整分析见 `TARGET_MACHINE_ANALYSIS.md`。
>
> 目标机器: 16 NUMA × 38 核 = 608 核, aarch64, SVE 512-bit + SME, 无 L3 cache。
> 本文推演该机器上 SConv 的 CSA 参数, 分析需修改的代码, 并给出优化策略。

---

## 1. 硬件规格

| 参数 | 值 |
|------|---|
| CPU 架构 | aarch64 (ARMv9) |
| 向量扩展 | SVE 512-bit, SME |
| NUMA 节点数 | 16 |
| 每节点核心数 | 38 |
| 总核心数 | 608 |
| L1 缓存 | 32 KB / 核 (私有) |
| L2 缓存 | 768 KB / 核 (私有) |
| L3 缓存 | **无** |
| 操作系统 | OpenEuler (CentOS 系) |

**关键特征: 无 L3 缓存。** CSA 的三级缓存模型 (L1/L2/L3) 中, L3 层变为 DRAM。

---

## 2. CSA 代码问题与修复

### 2.1 Bug: L3_size=0 时死循环

`lib/CSA.cpp` 的 `halfHeuristic()` 在 `cache_size=0` 时会死循环:

```
solution = 256 → tiles_size > 0 → solution = 128 → ... → solution = 1
→ tiles_size = tileSizeL3(1) > 0 → solution = 0
→ tiles_size = tileSizeL3(0) = k2×w_size > 0 → solution = 0/2 = 0
→ tiles_size = tileSizeL3(0) > 0 → ... → 无限循环!
```

`binarySearchHeuristic()` 在 `cache_size=0` 时返回初始值 (错误, 应为 1)。

### 2.2 修复

已在 `lib/CSA.cpp` 的两个 heuristic 函数开头加守卫:

```cpp
if (cache_size == 0) return 1;  // no cache at this level
```

含义: 该缓存层不存在时, 只能容纳 1 个 tile (从下一级内存加载, 用完丢弃)。

### 2.3 代价模型调整建议

当前代价模型把访问分为 `mem / l3 / l2 / l1` 四级。无 L3 时, 原本命中 L3 的访问
(`l3`) 实际来自 DRAM, 应计入 `mem`。

**当前代码** (`CSA.cpp` IS 的 EQ3):
```cpp
l3 = tCH × ceil(w_fit × in_tiles_per_tch × in_size / cache_line);
```

**建议修改**:
```cpp
if (arch_.l3_size == 0) {
    mem += tCH × ceil(w_fit × in_tiles_per_tch × in_size / cache_line);
} else {
    l3 = tCH × ceil(w_fit × in_tiles_per_tch × in_size / cache_line);
}
```

类似地, WS 的 EQ3 也需要同样处理。本修改不影响正确性 (只是代价估算更准确)。

---

## 3. 参数推演

### 3.1 ArchInfo 参数

```cpp
ArchInfo arch = {
    (uint32_t)(32768 * 0.9),     // L1: 29491 bytes (~29 KB)
    (uint32_t)(768 * 1024 * 0.9), // L2: 706562 bytes (~690 KB)
    (uint32_t)0,                  // L3: 0 (无 L3!)
    4,                            // L1 latency: ~4 cycles (aarch64 典型)
    20,                           // L2 latency: ~20 cycles
    300,                          // L3 latency = mem latency (L3 miss = DRAM)
    300,                          // mem latency: ~300 cycles (NUMA local DRAM)
    128                           // cache line: 128 bytes (需实测确认)
};
```

> 注: cache_line 大小需在目标机器上实测: `getconf LEVEL1_DCACHE_LINESIZE`
> NUMA 远端 DRAM 延迟更高 (~600-1000 cycles), 需避免跨 NUMA 访问。

### 3.2 mKInfo 参数 (两套方案)

**方案 A: 当前配置 (Nwin=16, Nf=8)**

```
mKInfo mK = {16, 8, 128};
```
- 微内核: C[16,8] += A[288,16]^T × B[288,8]
- 每次 36,864 FMA
- SME 只用半 tile (FMOPS 16×8 或 masked FMOPA)

**方案 B: SME 优化配置 (Nwin=16, Nf=16)**

```
mKInfo mK = {16, 16, 256};
```
- 微内核: C[16,16] += A[K,16]^T × B[K,16]
- 每次 73,728 FMA (翻倍)
- SME 全 tile (FMOPA 16×16), 每条指令 256 FMA
- K/16 次 FMOPA 即可完成归约

### 3.3 方案 A 推演 (Nwin=16, Nf=8)

#### 基础量

```
in_size  = Nwin × Fh × Fw × 4 = 16 × 3 × 3 × 4 = 576 bytes
w_size   = Nf × Fh × Fw × 4   = 8 × 3 × 3 × 4   = 288 bytes
out_size = Nwin × Nf × 4      = 16 × 8 × 4       = 512 bytes
```

#### Nc (L1 约束)

```
tileSizeL1(Nc) = (in_size + w_size) × Nc + out_size
              = 864 × Nc + 512 ≤ L1 = 29491

Nc=32: 864×32+512 = 28160 ≤ 29491 → 满足 ✓
Nc=64: 864×64+512 = 55808 > 29491 → 不满足

结果: Nc = 32, tCH = 128/32 = 4
```

#### K2 (L2 约束, IS 调度)

```
tileSizeL2(K2) = in_size + K2 × (w_size + out_size)
              = 576 + K2 × 800 ≤ L2 = 706562

最大 K2 = (706562-576)/800 = 882
但 K2 ≤ w_tiles_per_tch = Oc/Nf = 256/8 = 32

结果: K2 = 32 (数据量限制, 非缓存限制)
```

L2 利用率: `(576 + 32×800) / 706562 = 26176 / 706562 = 3.7%`
→ **L2 大幅未充分利用!** 可以增大 Nf 或 Nwin 来更好利用 L2。

#### K3 (无 L3!)

```
L3_size = 0 → K3 = 1 (代码修复后)

含义: 每个输入 tile 从 DRAM 加载, 无跨 tile 复用。
extra_k3 = in_tiles_per_tch % 1 = 0 (无边界)
```

#### WS 调度下的 K2

```
tileSizeL2(K2) = K2 × in_size + w_size + K2 × out_size
              = K2 × 1088 + 288 ≤ L2 = 706562

最大 K2 = (706562-288)/1088 = 649
K2 ≤ in_tiles_per_tch = (Oh×Ow)/Nwin = 4096/16 = 256

结果: K2 = 256 (全部输入 tile 放进 L2!)
L2 利用率: (256×1088+288) / 706562 = 278656 / 706562 = 39.4%
```

**关键发现**: WS 调度下, 所有 256 个输入 tile 都能放进 L2 (只需 272 KB)!
这意味着输入只需从 DRAM 加载一次, 之后全部从 L2 命中。
而 IS 调度只放 32 个滤波器 tile (26 KB), 利用率仅 3.7%。

#### IS vs WS 代价比较

对这台机器 (大 L2, 无 L3), **WS 大概率更优**:

| | IS | WS |
|---|---|---|
| K2 | 32 (滤波器 tile 放 L2, 26 KB) | 256 (输入 tile 放 L2, 272 KB) |
| K3 | 1 (输入从 DRAM) | 1 (滤波器从 DRAM) |
| L2 利用率 | 3.7% | 39.4% |
| DRAM 访问 | 输入 tile 每次从 DRAM | 滤波器 tile 每次从 DRAM |
| 谁驻留 L1 | 输入 (576 B/tile) | 滤波器 (288 B/tile) |

WS 的优势: 输入 tile 更大 (576 B) 但全部放 L2; 滤波器 tile 更小 (288 B),
从 DRAM 加载的开销更小。输入是"大而少复用"的数据 (每个 tile 只用一次),
滤波器是"小而多复用"的数据 (每个 tile 对所有窗口复用)。

### 3.4 方案 B 推演 (Nwin=16, Nf=16, SME 全 tile)

#### 基础量

```
in_size  = 16 × 3 × 3 × 4 = 576 bytes  (不变)
w_size   = 16 × 3 × 3 × 4 = 576 bytes  (翻倍!)
out_size = 16 × 16 × 4    = 1024 bytes  (翻倍!)
```

#### Nc (L1 约束)

```
tileSizeL1(Nc) = (576 + 576) × Nc + 1024 = 1152 × Nc + 1024 ≤ 29491

Nc=16: 1152×16+1024 = 19456 ≤ 29491 → 满足 ✓
Nc=32: 1152×32+1024 = 37888 > 29491 → 不满足 ✗

结果: Nc = 16 (减半!), tCH = 128/16 = 8
```

Nc 减半意味着通道迭代次数翻倍 (4→8), 但每次微内核调用的输出量也翻倍 (128→256)。

#### K2 (L2 约束, IS 调度)

```
tileSizeL2(K2) = 576 + K2 × (576 + 1024) = 576 + K2 × 1600 ≤ 706562
最大 K2 = (706562-576)/1600 = 441
K2 ≤ w_tiles_per_tch = 256/16 = 16

结果: K2 = 16
```

#### K2 (WS 调度)

```
tileSizeL2(K2) = K2 × 576 + 576 + K2 × 1024 = K2 × 1600 + 576 ≤ 706562
最大 K2 = (706562-576)/1600 = 441
K2 ≤ in_tiles_per_tch = 4096/16 = 256

结果: K2 = 256 (全部输入 tile 仍能放 L2)
L2 利用率: (256×1600+576)/706562 = 409776/706562 = 58.0%
```

### 3.5 两种方案汇总

| 参数 | 方案 A (Nf=8) | 方案 B (Nf=16, SME) |
|------|:---:|:---:|
| Nwin | 16 | 16 |
| Nf | 8 | 16 |
| Nc | 32 | 16 |
| tCH | 4 | 8 |
| K2 (IS) | 32 | 16 |
| K2 (WS) | **256** | **256** |
| K3 | 1 | 1 |
| 微内核 FMA/次 | 36,864 | 73,728 |
| SME 利用 | 半 tile (16×8) | **全 tile (16×16)** |
| L2 利用率 (WS) | 39.4% | 58.0% |

**推荐**: 方案 B + WS 调度。SME 全 tile 利用 + L2 充分利用 + 输入全缓存。

---

## 4. NUMA 并行策略

### 4.1 两级并行

```
NUMA 级 (16 节点):
  每个 NUMA 节点处理独立的卷积分片
  数据驻留在本地 DRAM (避免跨 NUMA 访问)

核级 (每节点 38 核):
  38 个核并行处理同一分片内的不同 tile
  共享 L2 (但 L2 是私有的, 所以各自处理不同 tile)
```

### 4.2 数据分区方案

对 batch=4 的卷积 (如 custom_0: 4×192×40×40):

**方案 1: 按 batch 分区 (4 个 NUMA 节点)**

```
NUMA 0 → batch 0  (1×192×40×40, 输入 ~2.4 MB, 输出 ~2.4 MB)
NUMA 1 → batch 1
NUMA 2 → batch 2
NUMA 3 → batch 3
其余 12 个 NUMA 节点空闲 (或处理其他卷积)
```

每个 NUMA 节点的 38 核并行处理 batch 内的 4096 个窗口 (K3=1, K2=256):
```
每核处理 4096/38 ≈ 108 个窗口
每核工作集: 108 × in_size = 108 × 576 = 62 KB → 放不进 L1 (29 KB)
→ 每核实际处理 ~50 个窗口 (50×576=28.8 KB ≈ L1)
→ 4096/50 ≈ 82 次迭代/核, 38 核并行
```

**方案 2: 按通道分区 (用更多 NUMA 节点)**

```
Ic=192, Nc=16 (方案 B) → tCH=12
12 个 NUMA 节点各处理 1 个通道段 (16 通道)
每节点: 16×3×3×4 = 576 B 输入通道段 × 4096 窗口 = 2.3 MB (放本地 DRAM)
每节点的 38 核并行处理 4096 个窗口
```

**方案 3: 按 batch × 通道 双重分区 (用满 16 NUMA)**

```
4 batch × 4 通道段 = 16 个分片 → 恰好分配到 16 个 NUMA 节点
每节点: 1 batch × 4 通道 = 1×48×40×40 × 4 bytes = 368 KB (本地 DRAM)
每节点 38 核并行处理空间维度
```

### 4.3 NUMA 亲和性绑定

```bash
# OpenEuler 上绑定进程到 NUMA 节点
numactl --cpunodebind=0 --membind=0 ./sconv-opt ...

# 或用 taskset 绑定到特定核
taskset -c 0-37 ./sconv-opt ...   # NUMA 0 的 38 个核
```

如果用 OpenMP:
```c
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int numa_node = tid / 38;
    // 确保线程访问本地内存
    struct bitmask *mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, numa_node);
    numa_bind(mask);
}
```

---

## 5. 完整参数配置

在 Transform IR 中指定:

```mlir
// 方案 B: SME 全 tile + WS 调度
%ukernels, %loops = transform.structured.sconv %convs
  { mK_info = [16, 16],                                      // Nwin=16, Nf=16
    arch_info = [32768, 786432, 0, 128],                     // L1=32K, L2=768K, L3=0, line=128
    latency = [4, 20, 300, 300]                               // L1=4, L2=20, L3=300=mem, mem=300
  }
  : (!transform.op<"linalg.conv_2d_nchw_fchw">)
  -> (!transform.op<"linalg.generic">, !transform.any_op)
```

或在代码中修改默认值 (`SConv.cpp:2239-2244`):

```cpp
mKInfo mK = {16, 16, 256};  // Nwin=16, Nf=16, Noutput=256
ArchInfo arch = {
    (uint32_t)(32768 * 0.9),       // L1
    (uint32_t)(768 * 1024 * 0.9),  // L2 = 690 KB
    (uint32_t)0,                    // L3 = 0 (无 L3!)
    4, 20, 300, 300, 128           // latencies, cache_line
};
```

---

## 6. 预期性能

### 6.1 单核性能估算

| 配置 | 微内核 FMA/次 | SME 指令/次 | 假设 IPC | GFLOPS (单核) |
|------|:---:|:---:|:---:|:---:|
| 方案 A (Nf=8, OpenBLAS) | 36,864 | — | — | ~15 (实测) |
| 方案 B (Nf=16, SME) | 73,728 | 288 FMOPA | 1 FMOPA/cycle | 288×256/1 = 73,728 FMA/cycle... |

更实际的估算:
- SME FMOPA 吞吐: 假设 1 条/cycle, 每条 256 FMA
- 微内核需要 K=288 条 FMOPA → 288 cycles
- 产出: 16×16 = 256 个输出值
- 有效 GFLOPS = 256×2 / 288cycles ≈ 1.78 FMA/cycle
- 按频率 2.5 GHz: 1.78 × 2.5 = **4.4 GFLOPS** (单核)

等等, 这太低了。让我重新算:
- 每个 FMOPA = 16×16 = 256 FMA
- 288 条 FMOPA = 288 × 256 = 73,728 FMA
- 如果 1 FMOPA/cycle, 288 cycles 完成 73,728 FMA
- FLOPS = 73,728 × 2 / 288 = 512 FLOP/cycle (×2 因为 FMA=2 FLOP)
- 按频率 2.5 GHz: 512 × 2.5 = **1280 GFLOPS** (单核峰值)

但这没算访存和打包开销。实际可能只有峰值的 20-40%:
- 单核: 256-512 GFLOPS
- 38 核 (一个 NUMA): ~10-19 TFLOPS
- 608 核: ~155-310 TFLOPS

### 6.2 对比当前 M4 实测

| 平台 | 核心 | 当前 GFLOPS | 预期 GFLOPS | 加速来源 |
|------|:---:|:---:|:---:|---|
| M4 (1核, OpenBLAS) | 1 | ~15 | — | 基准 |
| 目标 (1核, SME) | 1 | — | 256-512 | SME 指令 |
| 目标 (38核, 1 NUMA) | 38 | — | 5,000-10,000 | 多核并行 |
| 目标 (608核, 全机) | 608 | — | 80,000-300,000 | 全机并行 |

> 注: 实际加速受限于 DRAM 带宽、打包开销、NUMA 延迟等, 上述为理论上限。

---

## 7. 实施路线图

| 步骤 | 改动 | 预期单核加速 | 累计 |
|------|------|:---:|:---:|
| 1. 修复 CSA (L3=0) | 已完成 | — | — |
| 2. 设置正确 arch_info | 参数调整 | ~2x (L2 利用) | 2x |
| 3. Nf=16 + SME 微内核 | vector.contract → arm_sme | 10-20x | 20-40x |
| 4. 38 核并行 | scf.parallel / OpenMP | ~30x | 600-1200x |
| 5. 16 NUMA 分区 | numactl + 数据分布 | ~10x | 6000-12000x |
| 6. 向量化打包 + ping-pong | SVE + double buffer | 1.5x | 9000-18000x |

从当前 M4 单核 15 GFLOPS → 目标全机 **90-180 TFLOPS** (理论上限)。
