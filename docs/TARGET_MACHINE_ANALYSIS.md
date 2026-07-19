# 目标机器分析 (含 HBM): 608 核 aarch64 SVE512+SME OpenEuler

> 目标机器: 2 颗芯片, 每芯片 8 NUMA × 38 核 = 304 核, 全机 16 NUMA × 38 = 608 核。
> aarch64, SVE 512-bit + SME, 无硬件 L3 cache, 有 HBM (memkind 管理)。
>
> **不含 HBM 的场景分析** (帮助理解无 L3 时的问题) 见 `TARGET_MACHINE_NO_HBM.md`。

---

## 1. 硬件规格

### 1.1 拓扑结构

```
1 台机器
├── 芯片 0 (8 NUMA, 304 核)
│   ├── NUMA 0 (38 核, L1×38, L2×38, HBM 分区, DDR 分区)
│   ├── NUMA 1
│   ├── ...
│   └── NUMA 7
└── 芯片 1 (8 NUMA, 304 核)
    ├── NUMA 8
    ├── ...
    └── NUMA 15
```

### 1.2 基础规格

| 参数 | 值 |
|------|---|
| CPU 架构 | aarch64 (ARMv9) |
| 向量扩展 | SVE 512-bit (16 floats/vector), SME (16×16 tile) |
| 芯片数 | 2 / 台 |
| NUMA / 芯片 | 8 |
| 核 / NUMA | 38 |
| 总核数 | 2 × 8 × 38 = **608** |
| L1 缓存 | 32 KB / 核 (私有, ~4 cycles) |
| L2 缓存 | 768 KB / 核 (私有, ~20 cycles) |
| 硬件 L3 缓存 | **无** |
| 操作系统 | OpenEuler (CentOS 系) |

### 1.3 HBM 规格

| 参数 | 值 |
|------|---|
| HBM 容量 (每 NUMA) | **4 GB** |
| HLM 总带宽 (每芯片) | **4 TB/s** |
| HBM 带宽 (每 NUMA) | 4 TB/s / 8 = **512 GB/s** |
| HBM 带宽 (每核) | 512 / 38 = **13.5 GB/s** |
| HBM 访问延迟 | ~100-150 cycles |

### 1.4 DDR 规格

| 参数 | 值 |
|------|---|
| DDR 总带宽 (每芯片) | **358.4 GB/s** (8ch × 5.6Gbps × 8B) |
| DDR 带宽 (每 NUMA) | 358.4 / 8 = **44.8 GB/s** |
| DDR 带宽 (每核) | 44.8 / 38 = **1.18 GB/s** |
| DDR 容量 | ~1 TB |
| DDR 访问延迟 (NUMA local) | ~300 cycles |

### 1.5 带宽对比 (每核)

| 层级 | 每核带宽 | 延迟 | vs DDR |
|------|:---:|:---:|:---:|
| L2 cache | ~100+ GB/s | ~20 | ~85x |
| **HBM** | **13.5 GB/s** | ~120 | **~11x** |
| DDR (local) | 1.18 GB/s | ~300 | 1x |

> HBM 带宽是 DDR 的 11x (per core), 远低于 L2。微内核数据应来自 L2, HBM 只用于填充 L2。

### 1.6 典型使用场景

单算子通常使用 **1 个 NUMA (38 核)** + 本地 HBM (4 GB)。下面按单 NUMA 分析。

### 1.7 内存层次模型

```
L1 cache  (32 KB/核, 硬件, ~4 cycles, 带宽极大)
  ↓ miss
L2 cache  (768 KB/核, 硬件, ~20 cycles, ~100+ GB/s per core)
  ↓ miss
HBM       (4 GB/NUMA, memkind 软件管理, ~120 cycles, 13.5 GB/s per core)
  ↓ miss
DDR       (~62.5 GB/NUMA, ~300 cycles, 1.18 GB/s per core)
```

CSA 三级模型映射: L1→硬件L1, L2→硬件L2, L3→**HBM** (4 GB, latency=120)。

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

## 3. 参数推演 (含 HBM)

### 3.1 ArchInfo 参数 (更新: HBM 作为 L3)

```cpp
ArchInfo arch = {
    (uint32_t)(32768 * 0.9),        // L1: 29491 bytes (~29 KB)
    (uint32_t)(768 * 1024 * 0.9),    // L2: 706562 bytes (~690 KB)
    (uint32_t)(4UL * 1024 * 1024 * 1024 * 0.9),  // L3 = HBM: ~3.6 GB!
    4,      // L1 latency: ~4 cycles
    20,     // L2 latency: ~20 cycles
    120,    // L3 = HBM latency: ~120 cycles (远低于 DDR 300!)
    300,    // mem = DDR latency: ~300 cycles (NUMA local)
    128     // cache line: 128 bytes (需实测)
};
```

**关键变化**: L3 不再是 0, 而是 3.6 GB HBM! 延迟 120 cycles (远低于 DDR 300)。
CSA 的 heuristic 不再死循环, K3 可以取到数据量允许的最大值。

### 3.2 mKInfo 参数

**方案 A (当前)**: `mK_info = [16, 8, 128]` — SME 半 tile (16×8)
**方案 B (SME 优化, 推荐)**: `mK_info = [16, 16, 256]` — SME 全 tile (16×16)

下面以方案 B 为主推演, 方案 A 列出对比。

### 3.3 方案 B 推演 (Nwin=16, Nf=16, SME 全 tile)

#### 基础量

```
in_size  = Nwin × Fh × Fw × 4 = 16 × 3 × 3 × 4 = 576 bytes
w_size   = Nf × Fh × Fw × 4   = 16 × 3 × 3 × 4 = 576 bytes
out_size = Nwin × Nf × 4      = 16 × 16 × 4     = 1024 bytes
```

#### Nc (L1 约束)

```
tileSizeL1(Nc) = (in_size + w_size) × Nc + out_size
              = 1152 × Nc + 1024 ≤ L1 = 29491

Nc=16: 1152×16+1024 = 19456 ≤ 29491 → 满足 ✓
Nc=32: 1152×32+1024 = 37888 > 29491 → 不满足 ✗

结果: Nc = 16, tCH = Ic/16
  (对 Ic=128: tCH=8; 对 Ic=192: tCH=12)
```

#### K2 (L2 约束, IS 调度)

```
tileSizeL2(K2) = in_size + K2 × (w_size + out_size)
              = 576 + K2 × 1600 ≤ L2 = 706562

最大 K2 = (706562-576)/1600 = 441
K2 ≤ w_tiles_per_tch = Oc/Nf = Oc/16

对 Oc=192: K2 = min(192/16, 441) = 12
对 Oc=256: K2 = min(256/16, 441) = 16
```

L2 利用率 (Oc=192): `(576 + 12×1600) / 706562 = 19776 / 706562 = 2.8%` — 仍然很低。

#### K2 (WS 调度)

```
tileSizeL2(K2) = K2 × in_size + w_size + K2 × out_size
              = K2 × 1600 + 576 ≤ L2 = 706562

最大 K2 = (706562-576)/1600 = 441
K2 ≤ in_tiles_per_tch = (Oh×Ow)/Nwin

对 Oh×Ow=1600 (custom_0): K2 = min(100, 441) = 100
对 Oh×Ow=4096: K2 = min(256, 441) = 256
```

L2 利用率 (Oh×Ow=1600): `(100×1600+576)/706562 = 160576/706562 = 22.7%` — 好很多。

#### K3 (HBM 约束!)

```
tileSizeL3(K3) = K3 × in_size + K2 × w_size + K2 × K3 × out_size
              = K3 × 576 + K2 × 576 + K2 × K3 × 1024
              = K3 × (576 + K2 × 1024) + K2 × 576 ≤ L3 = 3.6 GB

以 K2=100 (WS, Oh×Ow=1600) 为例:
tileSizeL3(K3) = K3 × (576 + 100×1024) + 100×576
              = K3 × 102976 + 57600 ≤ 3,865,475,072

最大 K3 = (3.6G - 57600) / 102976 ≈ 37,000+

但 K3 ≤ w_tiles_per_tch = Oc/Nf = 192/16 = 12

结果: K3 = 12 (数据量限制, 非 HBM 容量限制)
```

**关键发现**: HBM (3.6 GB) 能容纳的 tile 数远超实际需要 (K3=12)。所有滤波器 tile
都能放进 HBM, 加载一次后全部从 HBM 命中。

HBM 利用率: `(12×102976+57600) / 3.6G ≈ 1.24 MB / 3.6 GB = 0.03%` — 容量绰绰有余。

### 3.4 方案 A 对比 (Nwin=16, Nf=8)

| 参数 | 方案 A (Nf=8) | 方案 B (Nf=16) |
|------|:---:|:---:|
| Nc | 32 (L1 利用率 95%) | 16 (66%) |
| K2 (IS, Oc=192) | 24 (L2: 1.9%) | 12 (L2: 2.8%) |
| K2 (WS, Ohw=1600) | 100 (L2: 15.3%) | 100 (L2: 22.7%) |
| K3 (IS, HBM) | 24 (HBM 几乎无限) | 12 |
| K3 (WS, HBM) | 100+ (HBM 几乎无限) | 100+ |
| 微内核 FMA/次 | 36,864 | 73,728 |
| SME tile 利用 | 50% (16×8) | **100% (16×16)** |

### 3.5 IS vs WS (含 HBM)

| | IS (输入驻留 L1) | WS (权重驻留 L1) |
|---|---|---|
| L2 放什么 | 滤波器 tile (K2 个) | 输入 tile (K2 个) |
| HBM 放什么 | 输入 tile (K3 个) | 滤波器 tile (K3 个) |
| K2 (L2, Oc=192) | 12 (19 KB) | 100 (156 KB) |
| K3 (HBM) | 100+ (57 KB) | 12+ (7 KB) |
| L2 利用率 | 2.8% | **22.7%** |
| HBM 利用率 | 0.001% | 0.0003% |

**结论: WS 调度更优**。原因:
1. L2 利用率: WS 22.7% >> IS 2.8% (输入 tile 更多, 但每个更小)
2. HBM 效果: HBM 容量 (4 GB) 远超任何 tile 集大小, 无论 IS/WS 都能全缓存
3. 带宽利用: HBM 带宽 2.75 TB/s, 数据只需从 HBM 加载一次, 之后全在 L2/L1

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

## 5. HBM 优化策略 (新增)

### 5.1 HBM 作为软件管理的 L3

HBM (4 GB/NUMA, 2.75 TB/s) 通过 memkind 显式分配, 不是硬件 cache。
需要主动把数据放到 HBM, 才能享受高带宽。

### 5.2 哪些数据应该放 HBM?

| 数据 | 大小 (custom_0) | 放哪? | 理由 |
|------|---:|---|------|
| 原始输入 tensor | 4×192×42×42×4 = 5.1 MB | **HBM** | 太大放不进 L2, 被 packing 反复读取 |
| 原始权重 tensor | 192×192×3×3×4 = 1.3 MB | **HBM** | 被 packing 读取 |
| 输出 tensor | 4×192×40×40×4 = 4.7 MB | **HBM** | 累积结果 |
| 打包输入 tile (单次) | 576 B | L1 | 微内核直接读 |
| 打包滤波器 tile (单次) | 576 B | L1 | 微内核直接读 |
| 多重打包 buffer (WS) | 160 KB | **L2** | 放进 L2 复用 |
| 多重打包 buffer (IS) | 19 KB | L2 | 放进 L2 复用 |

> 对 custom_0 (5.1 MB 输入 + 1.3 MB 权重 + 4.7 MB 输出 = 11.1 MB), 4 GB HBM 绰绰有余。

### 5.3 memkind 集成方案

**方案 A: 在 MLIR payload 层分配 HBM**

在 `performance_payload.mlir` 的 main 函数中, 用 C 函数分配 HBM:

```c
// runtime/hbm_alloc.c
#include <hbwmalloc.h>
void* alloc_hbm(size_t size) { 
    void* ptr; hbw_posix_memalign(&ptr, 64, size); return ptr; 
}
void free_hbm(void* ptr) { hbw_free(ptr); }
```

在 MLIR payload 中调用:
```mlir
func.func @main() {
  %size = arith.constant ... : index
  %ptr = func.call @alloc_hbm(%size) : (index) -> !llvm.ptr
  %input = memref.view %ptr ...  // 从 HBM 指针创建 memref
  ...
}
```

**方案 B: 在 sgemm_blas_kernel 包装层使用 HBM**

让打包后的 tile 数据驻留在 HBM:
```c
// 在 sgemm_blas_kernel 初始化时, 把 input/weight 数据拷贝到 HBM
static float* hbm_input = NULL;
static float* hbm_weight = NULL;

void init_sconv_hbm(float* input, float* weight, size_t in_size, size_t w_size) {
    if (!hbm_input) hbw_posix_memalign((void**)&hbm_input, 64, in_size);
    if (!hbm_weight) hbw_posix_memalign((void**)&hbm_weight, 64, w_size);
    memcpy(hbm_input, input, in_size);   // DDR → HBM 一次性拷贝
    memcpy(hbm_weight, weight, w_size);
}
```

**方案 C: 用 LD_PRELOAD 拦截 malloc (最简单, 不改代码)**

```bash
# 用 memkind 的 hbw_malloc 替换所有 malloc
LD_PRELOAD=/usr/lib/libhbw.so.1 HBW_MALLOC_PREFER_HBW=1 ./sconv-opt ...
```

这样所有动态分配 (包括 `memref.alloca` → `malloc`) 都会优先用 HBM。

### 5.4 HBM 带宽分析

**单 NUMA (38 核) 的带宽预算**:

```
HBM 带宽: 2.75 TB/s / 38 cores = 72.4 GB/s per core
L2 带宽:  ~100+ GB/s per core (硬件 cache, 更快)
DDR 带宽: 22.4 GB/s / 38 = 0.59 GB/s per core
```

**微内核算术强度分析** (方案 B, K=288):
```
数据读取: A[K,M]=288×16×4=18KB + B[K,N]=288×16×4=18KB = 36 KB
数据写入: C[M,N]=16×16×4=1 KB
总数据:   37 KB
总 FMA:    2×K×M×N = 2×288×16×16 = 147,456
算术强度: 147,456 / 37,000 = 3.98 FLOP/byte
```

**屋顶线分析** (per core):
```
SME 计算峰值: ~1,280 GFLOPS (估算)
HBM 带宽 (per core): 72.4 GB/s
平衡点算术强度: 1280 / 72.4 = 17.7 FLOP/byte
实际 AI: 3.98 FLOP/byte  ← 远低于平衡点!

→ 单次微内核调用是 HBM 带宽受限的!
```

**如何提高算术强度**:

| 策略 | 效果 | 说明 |
|------|------|------|
| **L2 缓存 multipack** | AI 提升 10x+ | 打包数据放 L2 (768 KB), 微内核从 L2 读 (100+ GB/s) |
| **增大 Nf** | AI ×2 | Nf=32: AI=7.9, 但 L1 放不下 |
| **减少 K (减小 Nc)** | AI 提升 | Nc=8: K=72, 但 channel 迭代增多 |
| **ping-pong + HBM 预取** | 隐藏访存 | 从 HBM 预取下一 tile 到 L2, 同时算当前 L2 中的 tile |

**关键**: 微内核数据访问主要来自 **L2 缓存的 multipack buffer** (100-160 KB),
不是 HBM。L2 带宽 (100+ GB/s) 远高于 HBM per-core (72 GB/s), 所以实际瓶颈在 L2 带宽,
HBM 只在 L2 miss 时才被访问 (加载数据进 L2 的 packing 阶段)。

### 5.5 HBM-aware ping-pong (L2 ↔ HBM 双缓冲)

结合 ping-pong 和 HBM:

```
HBM (4 GB): 原始输入/权重 tensor (一次性加载)
    ↓ 预取 (HBM 带宽 72 GB/s per core)
L2 (768 KB): multipack buffer (打包好的 K2 个 tile)
    ↓ 直接读 (L2 带宽 100+ GB/s per core)
L1 (32 KB): 当前微内核 tile (in + w + out)
    ↓ SME 指令
寄存器: FMOPA 16×16 外积
```

**三层 ping-pong**:
```
Layer 1 (L1 ↔ L2):  微内核计算时, 从 L2 预取下一 input tile 到 L1
                    buffer: 2×18 KB = 36 KB, 放得进 L1 (29 KB)... 刚好溢出
                    → 用单 buffer, 依赖硬件 prefetcher

Layer 2 (L2 ↔ HBM): 从 L2 算时, 从 HBM 预取打包数据到另一 L2 区域
                    buffer: 2×160 KB = 320 KB, 放得进 L2 (690 KB) ✓

Layer 3 (HBM ↔ DDR): 初始数据加载, DDR → HBM (一次性)
                    带宽: 22.4 GB/s → 5.1 MB 加载约 0.2 ms (可忽略)
```

### 5.6 HBM 带宽 vs SME 计算: 谁是瓶颈?

对 custom_0 (4×192×40×40, 4.25 GFLOP) 单核分析:

```
计算时间 (SME 峰值): 4.25G / 1280G = 3.3 ms
数据加载 (HBM, 首次): 11 MB / 72 GB/s = 0.15 ms  ← 远小于计算时间!
数据加载 (L2, 复用): 每次微内核 37 KB / 100 GB/s = 0.37 μs
    × 总微内核次数 (4.25G / 73,728 = 57,640 次) = 21.3 ms

→ 瓶颈: L2 带宽 (21 ms), 不是 HBM (0.15 ms) 也不是 SME 计算 (3.3 ms)
→ 优化方向: 减少微内核次数 (增大 tile) 或减少每次数据读取 (L2 复用 / ping-pong)
```

**38 核并行后**:
```
L2 带宽 (每核独立): 38 × 100 = 3,800 GB/s
计算时间: 21 ms / 38 = 0.55 ms (L2 受限)
HBM 首次加载: 0.15 ms (串行, 只需一次)
总时间: ~0.7 ms
GFLOPS: 4.25G / 0.7ms = 6,071 GFLOPS = 6.1 TFLOPS
```

---

## 6. 完整参数配置 (含 HBM)

在 Transform IR 中指定:

```mlir
// 方案 B: SME 全 tile + WS 调度 + HBM 作为 L3
%ukernels, %loops = transform.structured.sconv %convs
  { mK_info = [16, 16],                                    // Nwin=16, Nf=16 (SME 全 tile)
    arch_info = [32768, 786432, 4294967296, 128],         // L1=32K, L2=768K, L3=4G(HBM), line=128
    latency = [4, 20, 120, 300]                            // L1=4, L2=20, HBM=120, DDR=300
  }
```

或在代码中修改默认值 (`SConv.cpp:2239-2244`):

```cpp
mKInfo mK = {16, 16, 256};           // SME 全 tile
ArchInfo arch = {
    (uint32_t)(32768 * 0.9),          // L1
    (uint32_t)(768 * 1024 * 0.9),     // L2
    (uint32_t)(4UL * 1024 * 1024 * 1024 * 0.9),  // L3 = HBM (4 GB)
    4, 20, 120, 300, 128              // latencies, cache_line
};
```

运行时用 memkind 把数据分配到 HBM:

```bash
# 方式 1: LD_PRELOAD (最简单)
export LD_PRELOAD=/usr/lib/libhbw.so.1
export HBW_MALLOC_PREFER_HBW=1
numactl --cpunodebind=0 --membind=0 ./sconv-opt ...

# 方式 2: 在 payload 中显式调用 alloc_hbm (见 §5.3)
```

---

## 7. 预期性能 (含 HBM)

### 7.1 单 NUMA (38 核) 性能

以 custom_0 (4×192×40×40, 4.25 GFLOP) 为例:

| 环节 | 单核时间 | 38 核并行 | 说明 |
|------|:---:|:---:|------|
| HBM 首次加载 | 0.15 ms | 0.15 ms (串行) | 11 MB / 72 GB/s, 只需一次 |
| L2 packing 读取 | 21 ms | 0.55 ms | 37 KB × 57,640 次 / 100 GB/s / 38 |
| SME 计算 (峰值) | 3.3 ms | 0.09 ms | 4.25G / 1,280G / 38 |
| **总时间 (瓶颈)** | 21 ms (L2 受限) | **~0.7 ms** | HBM + L2 + 计算 |
| **GFLOPS** | 200 | **6,071 (6.1 TFLOPS)** | 4.25G / 0.7ms |

### 7.2 与 M4 对比

| 平台 | 配置 | custom_0 GFLOPS | 加速比 |
|------|------|:---:|:---:|
| M4 (1 核, OpenBLAS) | Nf=8, IS | 13.7 | 1x (基准) |
| M4 (1 核, OpenBLAS) | Nf=8, IS | 13.7 | 1x |
| 目标 (1 核, 无优化) | Nf=8, IS, L3=0 | ~15 (估) | ~1x |
| 目标 (1 核, HBM+WS) | Nf=8, WS, HBM | ~50-100 (估) | 4-7x |
| 目标 (1 核, SME+HBM) | Nf=16, WS, HBM | ~200-400 (估) | 15-30x |
| 目标 (38 核, SME+HBM) | Nf=16, WS, HBM | **~6,000+ (6 TFLOPS)** | **440x+** |

### 7.3 全 11 个自定义卷积预估 (单 NUMA, 38 核)

| 卷积 | FLOPs | 预计时间 (ms) | 预计 GFLOPS |
|------|------:|---:|---:|
| custom_0 (4.25G) | 4.25G | 0.7 | 6,071 |
| custom_1 (67.95G) | 67.95G | 11.2 | 6,067 |
| custom_4 (67.95G) | 67.95G | 11.2 | 6,067 |
| custom_7 (1.06G) | 1.06G | 0.17 | 6,235 |
| ... | ... | ... | ~6,000 |

> 全部 11 个卷积: 预计总时间 ~45 秒 (38 核并行)
> (vs M4 单核 baseline 352 秒 → SConv+BLAS 4.4 秒)

### 7.4 优化实施路线图

| 步骤 | 改动 | 单核加速 | 累计 | 说明 |
|------|------|:---:|:---:|------|
| 0 | CSA L3=0 修复 (已完成) | — | — | 防止死循环 |
| 1 | 设置 HBM 参数 (L3=4G, latency=120) | ~2x | 2x | K3 从 1→256, 全部 tile 缓存 |
| 2 | WS 调度 (L2 放输入) | ~1.5x | 3x | L2 利用率 3%→23% |
| 3 | Nf=16 SME 全 tile | ~2x | 6x | 256 FMA/cycle vs 128 |
| 4 | 38 核 OpenMP 并行 | ~30x | 180x | 空间 tile 级并行 |
| 5 | memkind HBM 数据分配 | ~1.5x | 270x | HBM 带宽 123x DDR |
| 6 | ping-pong (L2↔HBM) | ~1.4x | 380x | 隐藏打包访存 |
| 7 | SVE 向量化打包 | ~1.5x | 570x | 打包加速 |

**理论极限**: 15 GFLOPS (M4 基准) × 570x ≈ **8.5 TFLOPS** (单 NUMA)
实际可能 50-70% 效率: **4-6 TFLOPS** (单 NUMA)

---

## 8. 实施清单

```
[已完成] CSA 代码: cache_size=0 守卫 (防死循环)
[待做]   CSA 代码: 代价模型 L3→mem 归并 (当 L3_size<HBM 时)
[待做]   代码:    修改 SConv.cpp 默认 arch_info (HBM 参数)
[待做]   代码:    修改 mK_info 默认为 {16,16,256} (SME 全 tile)
[待做]   运行时:  编译 memkind/hbw 库, 集成到 sgemm_blas_kernel.c
[待做]   运行时:  实现 ping-pong 双缓冲 (L2↔HBM)
[待做]   并行:    scf.for → scf.parallel (38 核空间并行)
[待做]   SME:     vector.contract → arm_sme.fmopa lowering
[待做]   打包:    SVE 向量化 (tensor.extract → vector.transfer_read + shuffle)
[待做]   部署:    numactl --cpunodebind + membind 绑定
```
