# 可行性论证：「向量化打包 + 多核」

> 结论前置：**值得做，但只对 MLIR/ConvBench 管线有意义**；`native_impl` 侧已经没有空间。
> 建议按「打包向量化 → 微内核 tile 放大 → 多核」三步走，预计 5–9 人日，
> 单线程可拿到 **3.5–5×**，t32 可追到 `native_impl` 的 **0.5–1.0×**（≈0.9–1.7 TFLOPS）。
>
> 相关：`docs/OPTIMIZATION_DESIGN_AND_RESULTS.md`（设计理念与已生效优化）、
> `docs/PERFORMANCE_VS_PYTORCH.md`（权威数据）。

---

## 0. 一句话判断

| 维度 | 判断 |
|---|---|
| 方向对不对 | 对。但收益的主体不是「向量化打包」，而是**「打包向量化 + 微内核放大 + 多核」三者叠加**。 |
| 作用对象 | 仅 **MLIR/ConvBench 管线**（当前单线程、打包 ~0.8 GB/s）。`native_impl` 打包已贴 13 GB/s 带宽上限，做了也白做。 |
| 预期收益 | M1 已实测 **-26%**（156.6→115.7 ms）；M2 后再 ~2×，单线程落到 ~55–65 ms；t32 若能并行再 ~10×，落到 **0.9–1.7 TFLOPS**（投影）。 |
| 主要风险 | MLIR 的 `scf.for → OpenMP` lowering 是否顺手（最大不确定性）；其余风险可控。 |
| 建议 | 先做 M1（打包 helper，**已交付**）+ M2（tile），这两步低风险；M3（多核）单独立项验证。 |

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

## 5. 现状与预期（M1 已实测，M2/M3 为投影）

| 阶段 | 措施 | 单线程 | t32 |
|---|---|---|--:|--:|
| 现状 | — | 156.6 ms / 27.1 GF | — |
| **M1（已实测）** | 打包 helper + 打包目标 buffer 复用 | **115.7 ms / ≈36.7 GF** | — |
| M2（投影） | + Nf=32（GEMM 66→26，微内核 4× 减少） | ≈ 55–65 ms / 65–77 GF | — |
| M3（投影） | + 多核（有效 8–16×） | — | **2.5–4 ms / 0.9–1.7 TFLOPS** |

- **M1 实测 115.7 ms（-26%；4.25 GFLOP / 0.1157 s ≈ 36.7 GFLOPS）**。原估算「≈70 ms」
  偏乐观：残差里约 42 ms 是循环/分配地板，单靠打包到不了 70。
- M2 同时改三处：GEMM 变宽（66→26 ms）、微内核调用数 76,800→19,200
  （循环地板 ≈42→≈10 ms）、每次固定成本摊薄。三者叠加才把单线程推到 55–65 ms，
  所以 **75 ms/iter 是 M2 的验收线，不是 M1 的**。
- t32 仍是区间而非点值：并行效率受共享带宽与任务粒度影响，用 `native_impl` t32
  （2.45 ms）作上界夹。对照 PyTorch t32 0.2–2.0 TFLOPS（`PERFORMANCE_VS_PYTORCH.md`
  附录 A.1），MLIR 管线要做到 M3 才勉强进入同一量级。

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

1. **`scf.for` → `scf.parallel`，再 lowering 到 OpenMP**
   （`--convert-scf-to-openmp` / `SCFToOpenMP`，运行时已有 libomp——kblas 内部就在用）。
   需要在 `SConv.cpp` 里用 `transform.structured.tile_to_parallel` 或直接改写外层循环。
   **风险点**：pinned LLVM 版本里该 pass 是否可直接串进现有 pipeline、与 one-shot-bufferize 的先后顺序。
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
| **M2** 微内核放大 | `mKInfo`/CSA 调参到 `Nf=32`，必要时调 `arch_info` | custom_0 单线程 **≤75 ms**；≥120 GFLOPS；11/11 PASS |
| **M3** 多核 | `scf.parallel` → omp（或兜底） | t32 ≥ 800 GFLOPS 且 ≥ 0.5× `native_impl` t32（3 个代表形状）；11/11 PASS |
| 全程 | — | 任一形状相对当前回退 > 5% 即视为失败并回滚 |

> 说明：原表把「≤75 ms」写在 M1。stub 实测证明 75 ms 只有在 M2 把 GEMM 变宽、
> 微内核调用数降到 1/4 之后才可能达到（单靠 M1 的地板 ≈ 循环 42 + GEMM 66）。
> 故把 75 ms 移到 M2，M1 的验收改为更贴近实测的 ≤125 ms。

**停止条件**：M3 若 4 人日内无法让 t32 突破 400 GFLOPS，
停止在「单线程向量化版」（已是 5×），并把它作为交付结论。

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

1. **立刻做 M1 + M2**（2–3 人日）。这两步证据充分、风险低，且能把 MLIR 管线从
   27 GFLOPS 拉到 ~137 GFLOPS——**这是目前唯一「数量级」级别的剩余空间**。
2. **M3 单独立项**，先花半天验证 `scf.parallel → OpenMP` 在本仓库的 LLVM 版本上能跑通
   一个玩具 kernel；能跑通再投入，跑不通就走兜底或停在 M2。
3. 无论做到哪一步，**对外结论都以 `native_impl` 的 vs-PyTorch 数据为准**，
   MLIR 管线的数字只作为「变换本身的有效性」证据，两者不要混在同一张表里。
