# SConvTransform 优化演进实录（2026-09 → 2026-11-19）

> 本文记录**做过什么、为什么这样做、当时看到了什么、最后怎么决定**，包括失败的尝试和被证伪的假设。
> 它不是结果汇总（结果看 `PERFORMANCE_VS_PYTORCH.md`），也不是设计文档（设计看
> `OPTIMIZATION_DESIGN_AND_RESULTS.md`），而是两者之间的“决策轨迹”。
>
> 口径提醒：2026-11 之前的绝对毫秒是 **1.55 GHz** 时代测的，只用于看相对变化；
> 本文所有最终数字若未特别标注，均为 **2.0 GHz、单 NUMA 节点（node0）** 重测。
> 线程数只报 ≤32；t≥35 是本机的慢模式，不作结论。

---

## 0. 一张时间线

| 日期 | 提交 | 里程碑 | 关键结果（2.0 GHz 口径） |
|---|---|---|---|
| 2026-09 | 早期提交 | `native_impl` 参考实现 | 18 用例 t1–t32 对 PyTorch geomean **1.51×–1.62×** |
| 2026-11 | `f2c4b09` | `native_impl` 的 `Nc` 精调 | 饥饿形状 +5%–23%，只对 `task < threads` 生效 |
| 2026-11 | `54616f3` | **M1** 快速输入打包 | custom_0 **156.6 → 115.7 ms（−26%）** |
| 2026-11 | `2a3b40c` | **M2** multipacking + 微内核 tile 放大 | custom_0 **≈84–87 ms**（原目标 75 未达） |
| 2026-11 | `a4c80b5` | **A** 形状自适应 mK（`[0,0]`） | 11/11 形状优于旧 `[16,8]`，无回退 |
| 2026-11-18 | `2433254` | **A+C+G** 快边界拷贝 + `Nc` 自适应 + `Nwin=160` | custom_0 **33.3 ms / 127.7 GF**；geomean **1.57× vs torch t1** |
| 2026-11-18 | `2b09fc2` | **M3** 单 NUMA 多核 | geomean **2.36×**，最高 **338 GF**，t16≈t32 |
| 2026-11-19 | `922c240` | **M4-c** 多核专用 `Nwin` + 调度保护 | geomean **2.51×**，最高 **386 GF**，11/11（t8/t32） |

---

## 1. 起点：为什么同时存在两条路径

项目一开始面对的是一个“编译器交付物”和“性能上限探针”分离的问题：

- **MLIR 路径（仓库本体）**：交付物是 `sconv-opt` 一条命令。所有优化必须在 IR / transform /
  pass 层完成，不能偷偷塞一个运行时线程池。
- **`native_impl`（参考实现）**：不受编译器抽象约束，用来回答“这个算法在 Kunpeng 上到底能跑多快”。
  它的数字只作为上限参考，不直接算作交付路径的成绩。

这个划分决定了后面的所有取舍：**MLIR 路径优先做“语义安全、可回退、能解释”的改动**；
`native_impl` 可以更激进地做运行时自适应。

---

## 2. native_impl 阶段：先把“上限”摸出来

### 2.1 `Bc` 滤波器预打包并行化

- **问题**：每次调用都重新把滤波器打包成 `[ft][ic0][k][nf]`，单线程占端到端约 44%
  （`common_r18_7` 2.36M 元素，≈3.4 ms）。
- **改动**：`build_Bc` 按 `ft` 维并行；只有 `Nf` 变化时才重建。
- **结果**：大形状 t1 提升明显，且让后续 GEMM 的 B 操作数成为连续切片。
- **留下的教训**：固定开销要按“端到端”看，不能只看 kernel-only。

### 2.2 `(Nwin, Nf)` 运行时自适应

- **问题**：任务粒度是 `batch × 空间窗tile`。`r18_7`（`N=1, Ohw=49`）用 `Nwin=32`
  只有 2 个任务，32 核只能干瞪眼；而单线程反而偏好大 `Nwin`。
- **改动**：运行时小规模搜索 `(Nwin,Nf)`，按实测时间选。
- **结果**：低分辨率/小 batch 形状不再“有核用不上”。
- **代价**：自动调优本身是有噪声的决策器，同一形状跨进程可能选到不同参数（见 §7.3）。

### 2.3 修掉自适应路径的冗余 `memset`

- **问题**：自动调优缓存未命中时会把整块输出清零，但 GEMM 随后会完整覆盖它。
- **改动**：只在缓存未命中时清一次，且只清需要清的范围。
- **结果**：小形状收益最明显；也提醒我们“优化器自己也会制造固定开销”。

### 2.4 SME shim 端到端优化（go/no-go）

- **早期误判**：报的是 kernel-only 上界（输入已 pad+pack、输出未 unpad）。
- **端到端口径暴露的三块固定开销**：单线程 pad、每次 `calloc`、每次
  `pthread_create/join`；`custom_10` t32 一度只有 0.39×。
- **改动**：持久 worker pool、buffer 复用、pad/pack 并行。
- **决策**：SME 只保留给 `custom_9/10` 这类能铺满 16×16 tile 的形状；扩到
  `c0/c2/c5/c7` 实测全面更慢，**筛选器保持窄是设计决策，不是遗漏**。

### 2.5 `Nc` 精调

- **问题**：CSA 的纯 cache 代价模型在“并行饥饿”形状上给出的 `Nc` 偏离最优 5%–23%。
- **改动**：只对 `task < threads` 的形状做受限搜索，并加 3% 门槛。
- **被证伪的版本**：无条件搜 `Nc` 让 `thr_n8` 退化 43%——**搜索必须有约束**。

---

## 3. MLIR 单线程阶段：从 0.66× 追到 1.57×

### 3.1 起点与分解

改动前的 custom_0（t1、`[64,32]`）：**156.6 ms / 27.1 GF**，对 PyTorch t1 只有 **0.66×**。
用 stub 内核做差分分解后发现，最大的两块不是“卷积计算”：

| 组成 | 约耗时 | 性质 |
|---|--:|---|
| 函数边界 tensor→memref 拷贝 | ≈24 ms | ABI 时间，不是算法 |
| 输入 im2col 打包 | ≈52.6 ms | 9× 写放大 |
| filter 打包 | ≈3.7 ms | 可摊销 |
| GEMM | ≈43 ms | 真正的计算 |
| 循环/index/其他 | ≈1 ms | 曾被误判为 33 ms |

两个关键判断：

1. **先把 ABI 时间从算法时间里拆出去**。边界拷贝的 roofline 是 memcpy 带宽（~10 GB/s），
   通用 `memrefCopy` 只跑到 0.45 GB/s——差 20×，收益确定、零语义风险。
2. **打包是带宽问题，不是“写得不够聪明”**。三种写法（标量 / `memcpy` / 合并写）都停在
   13 GB/s/core，说明已经贴顶；继续做 SIMD 化没有空间。

### 3.2 M1：快速输入打包（156.6 → 115.7 ms）

**假设**：逐元素 `linalg.generic` gather 太慢，换成行分段 `memcpy` 就能大幅下降。

**做法三件套**：

1. 运行时 `pack_input_tile()`：按 `(ic,fh,fw)` 把同一输出行内连续的窗口段整段搬走；
2. 新 transform `transform.lower.to_pack`：把 `linalg.index` 归零后重算 affine 链，
   几何信息用 `sconv.pack_geom` 显式携带；
3. 打包目标 buffer 上提复用：从“每次迭代 `tensor.empty`”改为“最外层分配一次”。

**结果**：156.6 → 115.7 ms（−26%）。

**意外发现**：收益的大头其实来自第 3 点（约 28 ms）。只换 helper、不 hoist 只省 17 ms；
原来每次写都落到冷页。**“分配/复用”经常比“算法写法”更值钱。**

### 3.3 M2：微内核 tile 放大（≈84–87 ms）

**假设**：把 `Nf` 从 8 放大到 32，GEMM 调用更少、更宽。

**实测打脸**：

- `Nf>8` 必然切到 WS + multipacking，M1 的 helper 只认单打包，必须先支持 multipacking；
- warm-cache 微基准里 `m16/n32/k108` 只要 21 ms（201 GF），但 in-situ 仍是 50–60 ms：
  打包后的 A、输出 C 放不进 L2（768 KB），每轮都在 DRAM 上读写。

**结果**：≈84–87 ms，没到 75 ms 的目标。**“微基准快”不等于“端到端快”**，这是本阶段最重要的教训。

### 3.4 A：形状自适应 mK（`mK_info = [0,0]`）

**问题**：固定 `[64,32]` 要求 `Nwin` 整除 `Ohw`、`Nf` 整除 `fn`，否则落入 edge 路径。
实测 `custom_5` 从 229 → 6227 ms、`custom_7` 从 36 → 216 ms。

**改动**：`[0,0]` 作为哨兵，按形状选“最大整除因子”：
窗口候选 `{160,128,80,64,32,16}`，filter 候选 `{32,16,8}`。

**结果**：11/11 形状优于旧 `[16,8]`，无回退；也把“固定 tile 写死”这个坑永久封住。

### 3.5 A+C+G：快边界拷贝 + `Nc` 自适应 + `Nwin=160`

| 改动 | 内容 | custom_0 |
|---|---|--:|
| **A** 快速 `memrefCopy` | `fast_memref_copy.c`，`get_libs` 置顶加载 | 63.4 → 63.1 ms（边界拷贝 24 → ~2 ms） |
| **C** `Nc` 自动 | `mK_info` 第三槽 + `computeTileC` 广 tile 规则 | 63 → 40 ms（`Nc=6,k=54` → `Nc=32,k=288`） |
| **G** 窗口 tile 放大 | `selectAutoNwin` 候选扩到 160 | 40 → 41 ms（持平；弱形状大改） |

**最终**：**33.3 ms / 127.7 GF**；11 形状对 torch t1 geomean **1.57×，10/11 ≥1.0×**。

**为什么 G 在这里“持平”却仍然值得做**：强形状已经接近单核 SVE 峰值（≈128 GF），
G 的收益主要体现在弱形状（c6 525→496、c5 79→56、c3 284→265），以及为多核提供更大的
`Nwin` 基线。

**仍然落后的 c7（0.83×）**：`Ohw=400` 只能取 `Nwin=80`，720 次 `m80/n32/k288` 小 GEMM
对 torch 的一次 `m≈1600,n=192,k=1728` 大 GEMM。要赢需要余数 peeling（允许 `Nwin=128`，
最后一块 16），属于 D 类改动，收益 ≈0.03 geomean，优先级最低。

---

## 4. M3 多核：把“并行”留在 IR 里

### 4.1 第一次尝试为什么错

原型把 `Nc`（输入通道块）也标成并行，结果 **8/11 形状错，max_diff 1e6–1e8**。

**根因**：`Nc` 是 GEMM 的**归约维**。同一个输出 tile 被 6 个 `Nc` 迭代累加，
并行后多个线程同时写同一块 C——这不是“调度问题”，是语义错误。

**决策**：只并行“写不相交 C 列、body 是完整工作项”的**最内层窗口 tile 循环**；
filter tile 循环与窗口组循环因体内含 filter 打包（写共享 scratch）而不入链。

### 4.2 正式实现的五个坑

1. `memref.alloca_scope` 会锁死 `convert-scf-to-cf`：新增 `sconv-erase-alloca-scope`；
2. `memref.reinterpret_cast` 的 offset 相对 base buffer：私有化后要加 `tid * slotElements`，
   并把结果类型改成 `strided<[...], offset: ?>`；
3. 不能每迭代 `memref.alloc`：改成 `[64][Nc·Fh·Fw·Nwin]` 的 per-thread 槽，整轮复用；
4. 窗口循环不能按位置选（M4 发现）：`Nwin` 改变时 `scf::tileUsingSCF` 会翻转内层循环顺序，
   `loopOps[size-2]` 会标到 filter-subblock；
5. 不是所有打包都在并行循环里（M4 发现）：被 hoist 的打包只读共享 buffer，不能私有化。

### 4.3 结果与新的硬约束

- **geomean 2.36×**，最高 **338 GF**（custom_1 @t16），11/11 正确（t8、t32）；
- **t16 与 t32 基本持平**：并行宽度只有 **5–20**，线程数超过任务数后只是空转；
- 新的矛盾：单线程要“大 tile 少调用”，多核要“小 tile 多任务”。**这两个目标直接冲突。**

---

## 5. M4-c：多核专用 `Nwin`，以及一次差点做错的规则

### 5.1 假设与第一版规则

**假设**：把 `Nwin` 减半，窗口 tile 循环的任务数翻倍，t16/t32 就能用上更多核。

**第一版规则**：`IC≤64` 保持串行 tile；否则 `min(base,80)`；`Ohw/candidate<8` 时取 40。

**第一次扫描就翻车**：

- custom_1/custom_2/custom_9 在 `Nwin=80` 连构建都过不去：
  “scratch buffer is used outside of an `scf.parallel` loop”；
- 修完构建后，c1/c2/c9 的并行时间反而 **2 倍以上**（c2 t8 22.5 → 50.2 ms）。

### 5.2 三个根因（按发现顺序）

1. **marker 按位置选错了循环**：`loopOps[size-2]` 在 tiling 顺序翻转时是
   filter-subblock 循环。改成“从微内核向上找最近的、`step == nwindows` 的祖先循环”。
2. **打包被 hoist 后不能私有化**：有的形状把输入打包提到窗口循环外，scratch 在并行区
   之前写完、区内只读。只有位于 `sconv.parallel` 循环内的打包才打 `parallel_scratch`。
3. **调度翻转（最隐蔽）**：`Nwin=80` 让 c1/c2/9 的 CSA 从 **IS 翻到 WS**；
   同时 c2/9 的窗口组从 1600 缩到 400，宽度反而 10→5。
   这不是“tile 变小”的直接后果，而是 `k3` 随 `Nwin` 重算后的连锁反应。

**决策**：`selectAutoNwinMulticore` 用 CSA 成本模型分别探测 base 与 candidate 的 `schd`，
不同就保留 base。**多核 tile 选择第一次变成了“调度感知”的决策。**

### 5.3 最终规则

```
IC <= 64                  → 保持串行 tile
candidate = min(base, 80)
Ohw / candidate < 8 且 40 整除 → candidate = 40
CSA(base).schd != CSA(candidate).schd → 保持 base
```

落到 11 个形状：

| 形状 | Nwin | 宽度 | 结果 |
|---|--:|--:|---|
| c0/c4/c6/c8/c10 | 80 | 20 | 提升 12%–20% |
| c3/c7 | 40 | 10 | c3 +8.2%，c7 +5.9% |
| c5（IC=48） | 160 | 20 | 保持（薄通道打包受限） |
| c1/c2/c9 | 160 | 10 | 保持（IS→WS 翻转） |

**最终**：geomean **2.51×**（M3 2.36×），最高 **386 GF**（custom_4），
相对 M3 best-time geomean **1.08×**；11/11 正确（t8、t32）。

### 5.4 为什么停在这里，而不是继续 M4-a/b

计划 §9.6 预先写好了判据：**“如果宽度 40 仍到不了 600 GF，问题在串行项（B/E），
不是宽度，届时应先做 B。”**

M4-c 的实测正好命中这个判据：

- 宽度已到 20–40，最高仍只有 386 GF；
- **t16→t32 只涨 ≤17%**；
- 手工把宽度从 20 加到 40（mc40）不再更快：
  c0 14.5 vs 14.4、c8 58.5 vs 48.9、c10 27.7 vs 26.9。

所以 M4-a（归约环内移）和 M4-b（窗口组入链）只是继续加宽，收益上限有限。
**下一步按计划做 E1（filter 打包去 3/4 冗余）→ E2（输入打包分段）→ B（边界拷贝）。**

---

## 6. 被否决 / 被证伪的方向（同样是结论）

| 方向 | 做法 | 结果 | 判定依据 |
|---|---|---|---|
| stride==1 打包向量化 | 内层换 `std::copy` 特化 | 0.91–1.06× | 噪声内；clang 已向量化且带宽受限 |
| 打包微优化三变体 | 标量 / `memcpy` / 合并写 | 13.17 / 13.33 / 12.39 GB/s | 同速 ⇒ 带宽 roofline |
| 扩大 SME 覆盖 | c0/c2/c5/c7 路由 SME | kernel-only 920/1130/1058/273 GF | 均低于 GEMM 的 2150/2270/1913/1264 |
| 放大 `Nf` | `{16,32,64,128,256}` | `{16,32}` 最优 | 无单调收益 |
| 无条件搜 `Nc` | 所有形状都搜 | `thr_n8` −43% | 必须加“饥饿形状 + 3% 门槛”约束 |
| 固定 `[64,32]` | 全局写死 | c5 229→6227 ms | 不整除就落 edge 路径 |
| M4 第一版“盲目减半” | 所有 IC>64 都降到 80 | c1/c2/9 慢 2× | IS→WS 翻转 + `k3` 缩小 |
| 每迭代 `memref.alloc` | 每次新分配消除共享 | t1 499→820 ms | malloc/缺页成本 |

---

## 7. 决策原则（从这些尝试里沉淀出来的）

### 7.1 先分清“算法时间”和“ABI/harness 时间”

tensor→memref 边界拷贝、benchmark 输出清零、动态 stride 适配，都不是卷积算法的一部分。
它们有独立的 roofline，必须先拆出去，否则会把 ABI 的锅算到算法头上。

### 7.2 用同二进制 A/B + 交错重复，不做跨进程比较

本机是共享节点，t32 噪声约 ±4–7%；自动调优本身也会因为计时噪声选到不同配置。
判定标准固定为：同代码只切目标机制、单/多线程各 3–6 轮交错、取中位数、
未改动用例给噪声底线；收益 < 噪声底线一律视为“无收益”。

### 7.3 自动调优是“有噪声的决策器”

min-of-5 能缓解但不能消除。**不要用单次运行比较两个版本**，也不要把调优选出的参数
当作“最优参数”写进论文。

### 7.4 并行只并行“写不相交 + body 是完整工作项”的循环

归约维不能并行；共享 scratch 的循环不能入链；被 hoist 的只读 buffer 要共享而不是私有化。
这三条现在是 `sconv-parallelize` 的硬边界。

### 7.5 tile 选择要“调度感知”，不能只看任务数

`Nwin` 会影响 `k3`、`tile_c`，进而影响 IS/WS 的选择。多核 tile 的选择必须同时看
“任务数”和“调度是否翻转”，否则会得到“任务更多但更慢”的反直觉结果。

### 7.6 在预先写好的判据处停手

M4-c 完成后按 §9.6 的判据停手，而不是因为“已经投入了”继续做 M4-a/b。
**能解释“为什么不继续”，和能解释“为什么继续”同样重要。**

---

## 8. 现在的位置与下一步

- **单线程**：custom_0 33.3 ms / 127.7 GF，接近单核 SVE 峰值；geomean 1.57× vs torch t1，
  只剩 c7 一个结构性落后形状。
- **多核**：M4-c geomean 2.51×、最高 386 GF；宽度已不是唯一瓶颈。
- **下一步**：E1 filter 打包去冗余 → E2 输入打包分段 → B 边界拷贝；
  D（c7 余数 peeling）仍是最低优先级。

复现入口与门禁：

```bash
# 单线程
python3 scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/custom_conv.csv \
  --transform test/lowering/sconv-blas-fastp.mlir --num 11 --runs 5 --warmup 3

# 多核（M4-c）
build/bin/sconv-opt -transform=test/lowering/sconv-blas-fastp-mc.mlir \
  -sconv-parallelize -sconv-lower-parallel conv.mlir -o conv.mc.llvm.mlir
OMP_NUM_THREADS=16 numactl --cpunodebind=0 --membind=0 \
  mlir-cpu-runner conv.mc.llvm.mlir -e main -entry-point-result=void \
  -shared-libs=...,libomp.so

# 门禁
python3 scripts/convbench/test_fastpack.py --stage all
python3 scripts/convbench/test_mk_auto.py
python3 scripts/convbench/test_multicore.py --stage all
```

---

## 9. 相关文档

| 文档 | 作用 |
|---|---|
| `docs/OPTIMIZATION_DESIGN_AND_RESULTS.md` | 设计理念、代码地图、每个优化的实现与效果 |
| `docs/OPTIMIZATION_PLAN_2026-11-18.md` | 阶段性方案、验收、M3/M4 的完整测量与坑 |
| `docs/PERFORMANCE_VS_PYTORCH.md` | 权威性能数据（含附录 C：MLIR 单线程/多核） |
| `docs/FEASIBILITY_VECTORIZED_PACK_MULTICORE.md` | 为什么做、值不值得做、风险与停止条件 |
| `docs/html/index.html` | 本文档的可视化版本（离线 HTML + SVG 示意图） |
