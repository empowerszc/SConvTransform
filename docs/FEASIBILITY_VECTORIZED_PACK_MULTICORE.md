# 可行性论证：「向量化打包 + 多核」

> 结论前置：**值得做，但只对 MLIR/ConvBench 管线有意义**；`native_impl` 侧已经没有空间。
> 建议按「打包向量化 → 微内核 tile 放大 → 多核」三步走，预计 5–9 人日。
> **2026-11-18 更新（2.0 GHz 重测）**：单线程已从 156.6 ms 降到 **33.3 ms
> （custom_0，≈4.7×）**，11 形状对 PyTorch t1 的 geomean 达 **1.57×**（10/11 ≥1.0×）；
> 多核 M3 已正式落地，单 NUMA **geomean 2.36×**、最高 338 GF。
> 唯一未达的验收是 t32 ≥800 GF，原因是并行宽度只有 5–20（§5、计划文档 §9.4）。
>
> 相关：`docs/OPTIMIZATION_DESIGN_AND_RESULTS.md`（设计理念与已生效优化）、
> `docs/PERFORMANCE_VS_PYTORCH.md`（权威数据）。

---

## 0. 一句话判断

| 维度 | 判断 |
|---|---|
| 方向对不对 | 对。但收益的主体不是「向量化打包」，而是**「打包向量化 + 微内核放大 + 多核」三者叠加**。 |
| 作用对象 | 仅 **MLIR/ConvBench 管线**（当前单线程、打包 ~0.8 GB/s）。`native_impl` 打包已贴 13 GB/s 带宽上限，做了也白做。 |
| 预期收益（已兑现） | M1+M2+A **实测 -44%**（156.6→87 ms）；**A+C+G 再降到 33.3 ms / 127.7 GF**，单线程相对原始基线 **-79%（4.7×）**，geomean **1.57× vs PyTorch t1**；**M3 多核 geomean 2.36×、最高 338 GF**。原投影「t32 0.9–1.7 TFLOPS」**没有兑现**，因为并行宽度只有 5–20。 |
| 主要风险 | MLIR 的 `scf.for → OpenMP` lowering（已解决，见 §6.1 与计划文档 §9.5）；剩下的风险是并行宽度/串行项，属结构性。 |
| 建议 | M1/M2/A/C/G/M3 均已交付。下一步做 M4「加宽」（归约环内移 / 窗口组入链 / 多核专用 `Nwin`），加线程已无收益。 |

---

## 1. 先把「作用在哪条路径」说清楚

仓库有两条路径（详见设计文档 §1）：

| | MLIR/ConvBench 管线 | `native_impl` |
|---|---|---|
| 角色 | 交付物（验证 SConv 变换） | 性能上界探针 |
| 并行 | **无**（串行 `scf.for`） | OpenMP（`batch × 空间tile`） |
| 打包 | 逐元素 `linalg.generic` | 手写 C 循环（行缓存 + `std::copy`） |
| custom_0 现状 | 156.6 ms / **27.1 GFLOPS** | t1 99.5 GF、t32 1737 GF |

同一份算法、同一台机器，**差 3.7×（t1）到 64×（t32）**。
这个差距不是「算法不行」，而是管线里两件事没做：打包的访存效率、并行。

> 注意：ConvBench 表格历史上打印的 `custom_0 = 6.78 GFLOPS` 是 FLOPs 漏乘 batch 的口径 bug
> （已修，见设计文档 §6.2）。真实值 27.1 GFLOPS。加速比（116.78×）不受影响。

---

## 2. 现状量化：钱花在哪了（custom_0）

### 2.1 生成的循环结构

`test/lowering/sconv-blas.mlir` 变换 4×192×42×42 → 192×192×3×3 → 4×192×40×40 得到：

```
batch      : 4                       (4)
ic_tile    : 0..192 step 24          (8)      ← Nc=24
sp_outer   : 0..1600 step 1600       (1)      ← K3·Nwin 覆盖整个 Ohw
oc_outer   : 0..192 step 192         (1)      ← K2·Nf 覆盖整个 Oc
  sp_inner : 0..1600 step 16         (100)    ← Nwin=16
    pack   : linalg.generic, [1,24,3,3,16]（3,456 个 f32 = 13.8 KB）
    oc_inner: 0..192 step 8          (24)     ← Nf=8
      call @sgemm_blas_kernel(m=16, n=8, k=216, ldc=1600)
```

关键点：**微内核是 16×8×216**，只有 55 kFLOP/次，而每轮卷积要调用

```
4 (batch) × 8 (ic) × 100 (sp) × 24 (oc) = 76,800 次
```

另有两个值得注意的现象：

- **外层 tiling 退化**：`sp_outer`/`oc_outer` 的 trip count 都是 1（CSA 给出的 `K3=100`、`K2=24`
  正好覆盖整个 `Ohw` / `Oc`）。所以实际起作用的只有内层 `16×8`，**没有任何跨 cache 层级的复用调度**。
- 打包块只有 13.8 KB（放得进 L1），被 24 个 oc tile 复用 —— 复用结构是对的，
  问题纯粹是**搬运实现太慢**。

### 2.2 用同一份 kblas 实测每次调用成本

独立微基准（`numactl` 绑 node0，与管线同库同编译选项）：

| 形状 | 单次耗时 | 等效 GFLOPS |
|---|--:|--:|
| **m=16, n=8, k=216**（当前 MLIR 形状） | **0.840 µs** | 66.1 |
| m=16, n=8, k=1728（K 合并到全通道） | 5.703 µs | 77.6 |
| m=16, n=32, k=216（Nf 8→32） | 1.344 µs | **164.5** |
| m=16, n=192, k=1728（Nf=Oc、K 合并） | 137.5 µs | 77.2 |
| m=64, n=32, k=1728（`native_impl` 量级） | 41.3 µs | 171.3 |

### 2.3 分解（custom_0，单线程）——用 stub 内核实测，不靠反推

把 `sgemm_blas_kernel` 和 `pack_input_tile` 分别替换成空实现，直接称出各段成本
（`mlir-cpu-runner`，`numactl` node0，去掉输出打印）：

| 配置 | 总时间（ms/iter） | 推出 |
|---|--:|---|
| 真实 GEMM + 标量打包（基线，同口径） | **161.3** | — |
| GEMM 置空 + 标量打包 | 94.6 | ⇒ GEMM ≈ **66.7** |
| GEMM 置空 + 打包置空 | **42.0** | ⇒ 循环/分配「地板」≈ **42** |
| ⇒ 标量打包（残差） | ≈ **52.6** | 44.2 MB ⇒ **≈0.84 GB/s** |

> 口径说明：同一轮 stub 实测的基线是 161.3 ms/iter；ConvBench 常规口径报 156.6，
> 差的约 3% 来自测量方式（打印/预热轮数），不影响分解比例。
>
> 修正早前口径：最初用「总时间 − GEMM 下界」把整段 92 ms 残差都算成打包
> （0.48 GB/s）。stub 实测表明其中约 42 ms 是**循环控制 + 每次迭代的
> `memref.alloc`**，与搬运无关；标量打包本身约 52.6 ms、等效 ~0.84 GB/s。
> 结论方向不变（远低于带宽顶、值得替换），但应读作
> **打包 ≈33%、GEMM ≈41%、循环/分配 ≈26%**，而不是「打包 59%」。
>
> 这里的 42 ms 地板里，很大一块是打包目标 tile **每迭代重新分配**（3200 次/迭代）：
> 后续实测把它上提复用后，整体直接省了 ~28 ms（见 §5 的 M1 实测）。


---

## 3. 为什么同一个「打包」在两个实现里差 27 倍

| | `native_impl` pack | MLIR pack |
|---|---|---|
| 实现 | 行缓存：每个 `(ic,fh,r)` 只 `memcpy` 一行，再按 `Fw` 位移切片 | `linalg.generic` 逐元素 `tensor.extract` + 边界谓词 |
| 访存 | 连续 `ld1/st1`（clang 自动向量化） | 标量、带 div/mod 地址计算 |
| 实测 | **13.2 GB/s/core**（三种写法同速 ⇒ 带宽顶） | **0.48 GB/s** |

因此：

- 对 `native_impl`：向量化打包**没有空间**（三变体微基准 13.17 / 13.33 / 12.39 GB/s，已回退）。
- 对 MLIR：**有 ~26× 的空间**，而且不需要 SVE 黑魔法——把手写 C 的行缓存搬运
  以运行时函数的形式接进 MLIR 即可（和 `sgemm_blas_kernel` 完全同一个套路）。

---

## 4. 三个杠杆与实测依据

### 杠杆 1：打包向量化（标量 gather → 行分段拷贝）

- 基线标量打包 ≈52.6 ms（0.84 GB/s）→ M1 helper 独立微基准 **15.5 ms / 2.85 GB/s**。
- 未到 `native_impl` 的 13 GB/s 的原因：`Nwin=16` 使每次只搬 **64 B**，且按
  `(ic,fh,fw)` 反复读同一输入行（约 9× 重读）。`native_impl` 用「行缓存一次、再按
  `Fw` 切片」避免重读。继续逼近 13 GB/s 是后续可选项，但打包只剩 ~16 ms/115 ms，
  边际收益有限。
- 真正被低估的大头是**打包目标 buffer 每迭代重新分配**（3200 次/迭代）：
  把它提到最外层循环外复用，单次又省 ~28 ms。两项合计 M1 实测 156.6 → **115.7 ms**。
- 实现：`pack_input_tile()` 放 `scripts/convbench/runtime/`；`transform.lower.to_pack`
  把 `packing="input"` 的 `linalg.generic` 换成 `llvm.call`（与 `lower.to_blas` 同构）。

### 杠杆 2：微内核 tile 放大（Nf 8 → 32/64）

| 方案 | 调用次数 | 单次 | GEMM 总时 | 备注 |
|---|--:|--:|--:|---|
| 现状 Nf=8, k=216 | 76,800 | 0.84 µs | 64.5 ms | 66 GFLOPS |
| Nf=32, k=216 | 19,200 | 1.34 µs | **25.8 ms** | **164 GFLOPS** |
| Nf=8, k=1728（只合并 K） | 9,600 | 5.70 µs | 54.7 ms | 77 GFLOPS，**不如放大 Nf** |

> 注意方向：**放大 `N`（输出通道 tile）比合并 `K` 有效得多**。
> 合并 `K` 只减少调用次数，不改善单次内的数据重用；放大 `Nf` 两者都改善。
> 这与 `native_impl` 里 `Nf ∈ {16,32}` 最优、`Nc` 只在饥饿形状有意义（设计文档 §3.5）是一致的。

### 杠杆 3：多核（`scf.for` → 并行）

- 可并行的维度天然充足：`batch 4 × sp_inner 100 × oc_inner 24`，即使 `Nf=32` 也有 `4×100×6 = 2400` 个任务。
- 每个任务只写自己那段输出（`unpack` 后输出区间互不相交），**无跨任务依赖**，
  唯一共享的是只读的 packed 权重 → 并行安全性好。
- 上界参考：`native_impl` 同 conv t32 = **1737 GFLOPS（2.45 ms）**，
  即它已经用经验证明这个形状在 32 核上的可达点。

---

## 5. 现状与预期（M1/M2/A/C/G/M3 均已实测，2.0 GHz）

| 阶段 | 措施 | 单线程 | t32 |
|---|---|--:|--:|
| 现状 | — | 156.6 ms / 27.1 GF | — |
| **M1（已实测）** | 打包 helper + 打包目标 buffer 复用 | **115.7 ms / ≈36.7 GF** | — |
| **M2（已实测）** | + multipacking 支持 + `mK_info=[64,32]` | **≈84–87 ms / ≈49 GF** | — |
| **A（已实测）** | + `mK_info=[0,0]` 形状自适应（最大整除因子） | **≈87 ms / ≈49 GF**，11/11 改善且无回退 | — |
| **A+C+G（已实测，2026-11-18，2.0 GHz 重测）** | 快速 `memrefCopy` + 宽 tile `Nc` 自适应 + 窗口 tile 160 | **33.3 ms / 127.7 GF**；geomean **1.57× vs torch t1**，10/11 ≥1.0× | — |
| **M3（已实测，2026-11-18）** | 窗口 tile 循环 → `scf.parallel` → OpenMP + per-thread scratch | 串行 34.4 ms | **t16 16.2 ms / 269 GF；geomean 2.36×，最高 338 GF（custom_1）** |
| M2 原目标 | （投影） | ≤75 ms（实际 87，已越过） | — |
| M3 原目标 | （投影） | — | 800 GF × 3 形状、≥0.5× `native_impl` t32（**未达**，见下） |

- 表中「现状/M1/M2/A」四行是 **1.55 GHz** 时代的历史值，只用来看相对变化；
  「A+C+G」与「M3」两行是 **2.0 GHz 重测**（A+C+G 在 1.55 GHz 下是 41 ms / 104 GF /
  1.35×，差异来自同频重测，见 `OPTIMIZATION_PLAN_2026-11-18.md` 开头）。
- **M1 实测 115.7 ms（-26%）**。原估算「≈70 ms」偏乐观：残差里约 42 ms 是循环/分配地板，
  单靠打包到不了 70。
- **M2 实测 ≈84–87 ms（-46% vs 原基线）**，但**没到 75**。当时归因（index ≈33 ms）
  已被推翻：真实构成是边界 copy ≈24 ms + 输入打包 ≈7.7 ms + filter 打包 ≈3.7 ms +
  GEMM ≈43 ms（设计文档 §3.9）。**A+C 已把 87 → 40 ms，G 再把它稳在 41 ms**
  （1.55 GHz；2.0 GHz 下终值 33.3 ms）。
- **`[64,32]` 不能全局写死**：要求 `Nwin=64` 整除输出空间尺寸、`Nf=32` 整除滤波器数，
  否则落入 edge 路径（custom_5: 229→6227 ms；custom_7: 36→216 ms）。
  **A 已解决**：默认快路径传 `mK_info=[0,0]`，按形状选最大整除因子；11/11 形状
  优于旧 `[16,8]`，无回退（设计文档 §3.8）。
- M3 实测把 custom_0 从 34.4 ms 压到 16.2 ms（2.18×），但**没到 800 GF**：
  实测显示瓶颈不是线程数而是**并行宽度只有 5–20**（窗口 tile 循环的 trip count），
  t16 与 t32 持平。最高 338 GF（custom_1 @t16），距 `native_impl` t32 的 1.7 TF
  仍差 5×。下一步是「加宽」而不是「加核」（计划文档 §9.6）。

---

## 6. 实现方案比较（打包）

| 方案 | 做法 | 优点 | 缺点 | 建议 |
|---|---|---|---|---|
| **A. 运行时 C helper** | 在 `convbench/runtime/` 加 `pack_input_tile()`，MLIR 用 `func.call` | 1–2 天；可直接复用 `native_impl` 已验证的行缓存逻辑；性能确定（8–13 GB/s） | 打包逻辑「离开」了 IR，不再是纯编译器产物 | **首选**（M1） |
| B. MLIR vector dialect | 打包 `linalg.generic` 改 `vector.transfer_read/write` + `vector.shuffle` | 保持「全部在 IR 里」，更像论文级实现 | 要处理边界谓词、`vector.shuffle` 掩码，调试试错成本高 | M1 达标后再评估 |
| C. 手写 SVE intrinsic | `svld1/svst1/svext` | 理论最高 | 维护成本最高；收益相对 A 很小（A 已接近带宽顶） | 不建议 |

> 结论：**打包是「拷贝」不是「计算」**，带宽顶就是 13 GB/s，
> 方案 A 已能贴顶，没必要为 5% 去写 SVE。

### 6.1 多核的实现路径（按风险从低到高）

1. **`scf.for` → `scf.parallel`，再 lowering 到 OpenMP**（**已采用并落地**）
   （`--convert-scf-to-openmp` / `SCFToOpenMP`，运行时已有 libomp——kblas 内部就在用）。
   实现方式：transform 打 `sconv.parallel` 属性 → `lib/Parallelize.cpp` 折叠 + 私化
   scratch → `sconv-opt -sconv-lower-parallel` 内置 lowering。
   **实际踩到的风险点**：`memref.alloca_scope` 清理（新 pass）、
   `memref.reinterpret_cast` 的 offset 相对 base buffer、per-thread scratch 不能每次 alloc。
   详见 `OPTIMIZATION_PLAN_2026-11-18.md` §9.5。
2. **在 transform 里直接生成 `omp.parallel`/`omp.wsloop`**（绕过 scf→omp）。
   更可控，但要写 MLIR dialect 代码。
3. **兜底：运行时并行**（把整段 tile 循环做成运行时函数，内部用持久线程池，照抄 SME shim 的 `pool_run`）。
   性能可达，但等于「绕过编译器并行化」，作为降级方案而非目标。

---

## 7. 风险清单与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| MLIR `scf.parallel → OpenMP` 在本版本不通 | M3 阻塞 | 先用 §6.1 方案 2；仍不通则走方案 3 兜底，并把它标注为「运行时并行」而非编译器并行 |
| `scf.parallel` 与 bufferization 交互（`memref.alloc` 在并行区内） | 编译失败 / 数据竞争 | 把 `tensor.empty` 提到并行区外（与设计文档 §2.2 的「复用优先」一致），每个 worker 一段 |
| 数值正确性 | 破坏交付物可信度 | ConvBench 11/11 逐元素 `rtol=1e-3` 作为硬门槛，每步都跑 |
| 收益被噪声淹没（±4–7%） | 误判 | A/B 用交错多轮中位数 + 对照组；验收统一按此口径（设计文档 §2.5） |
| 与 `native_impl` 功能重复、双份维护 | 长期成本 | 明确分工：`native_impl` 只当 oracle，不再加功能；MLIR 侧新逻辑集中到 `runtime/` 与 `lib/SConv.cpp` 两处 |
| 并行后与 kblas 内部线程叠加（超订） | 高线程档变慢 | 固定「外层并行 + `OMP_NUM_THREADS=1` 给 kblas」，并在 t32 用 `numactl` 绑定后复测 |

---

## 8. 里程碑与验收标准（可测量，不含主观项）

| 里程碑 | 交付 | 验收（同机、`numactl` node0） |
|---|---|---|
| **M1** 打包向量化（已交付） | 运行时 `pack_input_tile()` + `transform.lower.to_pack` + 打包目标 buffer 上提复用 | custom_0 单线程 **≤125 ms（实测 115.7）**；IR 中不再有标量 input-packing generic；`test_fastpack.py --stage all` PASS |
| **M2** 微内核放大（已交付，未达 75） | multipacking 快打包 + `mK_info=[64,32]` 的 tuned 变体（`sconv-blas-fastp-tuned.mlir`） | custom_0 单线程 **≈84–87 ms**（原定 ≤75 未达）；11/11 PASS（11 个 custom 形状） |
| **A** 形状自适应 mK（已交付） | `mK_info=[0,0]` 哨兵 + 最大整除因子（`selectAutoNwin/Nf`），默认快路径启用 | 11/11 形状优于 `[16,8]` 且无回退；custom_0 **≈87 ms**；`test_mk_auto.py --stage ir`（auto IR == 显式 IR）与 `--stage perf` PASS |
| **M3** 多核（已交付，未达 800 GF） | `sconv.parallel` → `scf.parallel` → OpenMP + per-thread scratch | 11/11 PASS（t8、t32）；geomean **2.36×**、最高 338 GF；**t32 ≥ 800 GF 未达成**，原因是并行宽度 5–20（计划文档 §9.4） |
| 全程 | — | 任一形状相对当前回退 > 5% 即视为失败并回滚 |

> 说明：原表把「≤75 ms」写在 M1，后移到 M2。当时的归因「index ≈33 ms」是**误判**
> （实测 index/循环只有 ≈1 ms，大头是动态 stride 的边界拷贝，见设计文档 §3.9）。
> A+C+G 之后单线程 33.3 ms（2.0 GHz），已越过原 55–65 ms 的投影区间。
>
> **`[64,32]` 只能按形状用**：`Nwin=64` 不整除输出空间时落入 edge 路径，
> 实测 custom_5 229→6227 ms、custom_7 36→216 ms。**A+G 用「最大整除因子」自动选择
> 解决**：默认快路径传 `mK_info=[0,0]`，窗口候选 `{160,128,80,64,32,16}`，
> 规则形状选 160，`Ohw=400` 落到 80；11/11 形状均优于 `[16,8]`，无回退（设计文档 §3.10）。

**停止条件（复核）**：原停止条件是「M3 若 4 人日内无法让 t32 突破 400 GFLOPS，
就停在单线程向量化版」。实测最高 338 GF（custom_1 @t16），**未突破 400**，
因此按原条件应停在「单线程向量化版 + 多核可用版」，把加宽留给 M4；
是否继续投入取决于是否真的需要 800 GF。

---

## 9. 不应该做的事（省时间）

| 不做 | 原因 |
|---|---|
| 继续优化 `native_impl` 的打包 | 三变体同速 13 GB/s ⇒ 带宽 roofline，没有空间 |
| 把 `native_impl` 改成多核 | 它已经是 |
| 扩大 SME 直接法覆盖 | `c0/c2/c5/c7` 的 SME kernel-only（920/1130/1058/273 GF）均低于 GEMM（2150/2270/1913/1264 GF） |
| 靠继续放大 `Nf`/`Nc` 拿收益 | `Nf` 扫 16–256 后 `{16,32}` 最优；`Nc` 仅饥饿形状 +5–23%，已进调优器 |
| 用 SVE 手写打包 | 方案 A 已贴带宽顶，边际收益 <5% |

---

## 10. 建议

1. **M1 + M2 + A + C + G 已完成**：单线程从 27 GF 拉到 127.7 GF（geomean 1.57× vs torch）。
2. **M3 已完成并可用**（geomean 2.36×、11/11）：但 t16 以后不再变快，
   说明继续加线程是浪费；下一笔投入应放在**加宽**（M4：归约环内移 / 窗口组入链 /
   多核专用 `Nwin`），而不是更多核。
3. 无论做到哪一步，**对外结论都以 `native_impl` 的 vs-PyTorch 数据为准**，
   MLIR 管线的数字只作为「变换本身的有效性」证据，两者不要混在同一张表里。
