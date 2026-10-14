# SME 直接卷积路径设计方案（MLIR ArmSME 直写）

- 状态：Draft（待用户评审）
- 日期：2026-10-14
- 分支：dev
- 关联：`docs/PERFORMANCE_REPORT.md`、`deepseek提供的方案说明.md`、`deepseek提供的直接法代码示意.cpp`

## 1. 背景与问题

SConvTransform 当前的卷积实现走 **GEMM 路径**：`transform.structured.sconv`
把 `linalg.conv_2d_nchw_fchw` 经两级 tiling + VB-packing 转成外积式
`linalg.generic` 微内核，再由 `transform.lower.to_blas` 替换成对运行时函数
`sgemm_blas_kernel` 的调用，最终落到厂商 kblas（SME/SVE 优化的 CBLAS）。

在 38 线程（numactl node0）下，该路径已在 11 个 custom 配置中的 9 个超过
PyTorch conv2d。**剩余两个配置仍落后**：

| 配置 | N, Ic, Ih, Iw | Oc, Fh, Fw | Oh, Ow | stride, pad | ours (GFLOPS) | PyTorch | 比值 |
|------|----------------|-----------|--------|-------------|---------------|---------|------|
| custom_9  | 4, 384, 80, 80 | 96, 3, 3 | 80, 80 | 1, 1 | 1620 | 2131 | 0.76 |
| custom_10 | 4, 768, 40, 40 | 96, 3, 3 | 40, 40 | 1, 1 | 957  | 1116 | 0.86 |

共同特征：**高 Ic、低 Oc、stride1、3×3、SAME padding**。此类配置下
im2col/VB-packing 的相对开销高、GEMM 的 N 维（Oc）小、算术强度利用率低。

## 2. 目标与非目标

### 目标
- 对 custom_9 / custom_10 这类配置，提供一条 **SME 直接卷积**路径：
  直接在原始 NCHW 输入上滑窗计算，跳过 packing。
- 数值正确；性能 **GFLOPS ≥ 现有 GEMM 路径**（并尽量超过 PyTorch）。
- **选择机制通用化**：用可复用的谓词 + 代价模型决定是否走 SME（见 §8），
  不枚举/写死具体配置；新增配置无需改代码。kernel 形状参数化、支持尾块。

### 非目标（首版）
- stride > 1、非 3×3 kernel（这是*支持集*上限，非选择机制）。
- 替换现有 GEMM 路径（其余 9 个配置保持 GEMM 不变）。

## 3. 关键前置发现

### 3.1 工具链（重要）
本机 LLVM 19.1.7 与 HPCKit BiSheng 5.2 的 **C 前端不支持 SME 的 ACLE
ZA/流式状态属性**（`arm_inout_za`、`__arm_new_za`、`arm_streaming` 等），
因此**无法用 C 内建函数正确编译 SME kernel**（编译通过但运行时 SIGILL）。

**但 MLIR 路径可用**：`mlir/Dialect/ArmSME` 方言 + `--enable-arm-streaming`
会生成正确的 IR 级函数属性，映射为：
- `arm_locally_streaming` → `"aarch64_pstate_sm_body"`
- `arm_new_za`           → `"aarch64_new_za"`

后端据此正确生成 `smstart sm` / `smstart za` / `zero {za}` / `smstop`，链接
`libclang_rt.builtins.a` 后可正常运行（已端到端验证，exit 0）。

### 3.2 硬件参数
- SVL（streaming vector length）= **512 bit = 64 字节 = 16 个 f32**。
- 因此 f32 的 ZA tile = **16 × 16**；4 个 ZA tile（za0–za3）= 16 oc × 64 列。

### 3.3 为什么必须做“整卷积替换”而非“微内核替换”
现有 `to_blas` 作用的微内核已经是 packing 后的 GEMM 外积。若在**微内核级**
接 SME，本质是重造一个 SME GEMM，去对标厂商 kblas，几乎不可能赢。
直接法的价值在于**跳过 packing、直接在原始输入上算**，因此必须在**整卷积
粒度**替换。

## 4. 总体架构

分为编译期与运行期两层：

```
        AOT 编译期                                 运行期 / 管线
┌───────────────────────────┐          ┌──────────────────────────────┐
│ sme/sconv_sme_direct.mlir │          │ transform.structured.sconv_sme│
│  (arm_sme 方言直写)        │          │   匹配目标配置 → CallOp       │
│        │                  │          │      │                       │
│  mlir-opt 流水线           │          │      ▼                       │
│        ▼                  │          │ LLVM::CallOp @sconv_sme_direct│
│ sconv_sme_direct.o        │  link    │                              │
│  (含 SME ABI)             │─────────▶│  运行时库                     │
└───────────────────────────┘          └──────────────────────────────┘
```

### 组件清单
1. `sme/sconv_sme_direct.mlir` — arm_sme 方言直写的直接卷积 kernel。
2. `sme/build_kernel.sh` + CMake 规则 — 编译 MLIR kernel 为 `.o`。
3. `lib/LowerToSME.cpp` — 新 transform op，生成 `CallOp`（仿 `lib/LowerToBLAS.cpp`）。
4. `include/SConv.td` — 新增 transform op 定义。
5. `test/lowering/sconv-sme.mlir` — 管线测试脚本。

## 5. Kernel 算法（stride1 / 3×3，形状参数化 + 尾块掩码）

> 通用化原则：除 stride/dilation/kernel 尺寸等**算法结构**外，所有
> `N/Ic/Oc/Oh/Ow` 与 tile 行数 `R`（由运行时 SVL 决定）都作为**运行时参数**，
> 不用常量写死；`Oc`、列尾块由掩码处理，不要求整倍数。

### 5.1 数据布局
- 设 `R = SVL/32`（f32 下 ZA tile 的行/列数；`SVL=512` 时 `R=16`）。
- 输入 `in`：NCHW，`[N][Ic][Ih][Iw]`。
- 权重打包 `wpack`：`[ceil(Oc/R)][Ic*9][R]`，最后一维是连续 `R` 个 oc 的权重，
  中间维按 `ic*9 + kh*3 + kw` 排列（对应 deepseek 的
  `[OB][IC*9][16]`）。尾部 `ocb` 块不足 `R` 时用 0 填充。
- 输出 `out`：NCHW，`[N][Oc][Oh][Ow]`。

### 5.2 循环结构
```
R = SVL/32                      # 运行时查询（首版可固定为 16）
COLS = 4 * R                    # 4 个 ZA tile 覆盖的输出列数（R=16 → 64）
for n in 0..N:
  for ocb in 0..ceil(Oc/R):                        # 运行时上界
    for col0 in 0..Oh*Ow step COLS:                # 运行时上界
      # col0 为展平的输出位置：oh = col0 / Ow, ow = col0 % Ow
      zero za0..za3
      for ic in 0..Ic:                             # 运行时
        for kh in 0..Fh:
          in_row = oh + kh - ph                    # 越界行按 0 处理
          for kw in 0..Fw:
            w = load wpack[ocb][ic*9+kh*3+kw][0..R-1]     # 1 条连续 load
            # 4 个 tile 覆盖 col0..col0+COLS-1 的 COLS 个输出列
            x_t = load in[n][ic][in_row][col0..] with shift (kw-pw)  # 掩码处理
            fmopa za_t += w ⊗ x_t   for t in 0..3              # 4 次 FMOPA / (ic,kh,kw)
      store za0..za3 → out[n][ocb*R .. +R-1][col0 .. col0+COLS-1]   # 掩码处理尾块
```

- 每个 `(ocb, col0)` 块 FMOPA 次数 = `Ic × Fh × Fw × 4`；每个块计算 `R × COLS` 个输出。
- 每次 FMOPA 累加一个 `R×R` tile（`R²` MAC）。
- **尾块处理**：
  - `ocb` 尾块（`Oc % R != 0`）：权重补 0，存储时按 `Oc - ocb*R` 掩码。
  - 列尾块（`Oh*Ow % COLS != 0`）：加载/存储用谓词掩码，避免越界写。

### 5.3 权重访问
每个 `(ic, kh, kw)` 需要 `R` 个 oc 的权重作为 FMOPA 的第一操作数（行方向）。
打包成连续 `R` 个 float 后一次 `ldr z` 载入；尾块不足 `R` 时补 0。

### 5.4 输入访问与 stride 处理（首版 stride1）
- 一个输入行上连续 `COLS` 个元素即覆盖目标输出列的滑窗（stride1），
  对 `kw=0..Fw-1` 是 `Fw` 个相邻错位窗口，存在重叠（可后续用 SVE 移位优化）。
- 输入行 `in_row` 越界（`<0` 或 `>=Ih`）时整行视为 0。

### 5.5 边界（SAME pad + 尾块）
- 首/末输出行、列对应的输入越界部分补 0。
- 统一用**谓词掩码**在同一套 kernel 里处理，而非特判不同形状：
  - pad 越界：对 `x_t` 的加载/外积用 operand mask（`arm_sme.outerproduct`
    支持 operand mask），越界元素不参与累加。
  - 列尾块：掩码限制有效列数。
  - `ocb` 尾块：掩码限制有效行数（存储时）。
- 备选（若掩码路径不可行或过慢）：在 transform/运行时侧先把输入 pad 到
  `[Ih+2ph][Iw+2pw]` 的 0 填充 buffer，kernel 内不处理 pad（仍处理尾块）。

## 6. MLIR 表达（arm_sme 直写）

### 6.1 流式/ZA 属性注入
kernel 函数由 `--enable-arm-streaming=streaming-mode=streaming-locally,za-mode=new-za`
打上 `arm_locally_streaming` + `arm_new_za` 属性（无需在源码里写）。

### 6.2 使用的方言与 op
- `arm_sme.outerproduct`（→ `llvm.aarch64.sme.mopa`）：核心累加，
  操作数为 `vector<[4]xf32>`（= 16 f32），结果/累加器为 SME tile
  `vector<[4]x[4]xf32>`（= 16×16）。
- `arm_sme.get_tile` / tile 初始化算子（`zero`）。
- tile 切片存储（`st1w.horiz` 系列）写回输出。
- SVE 载入：`vector.load` / `arm_sve` / `llvm.aarch64.sve.ld1` 系列。
- `scf.for` 表达循环，`memref`/`llvm.ptr` 表达寻址。
- `arm_sme.streaming_vl`（或 `rdsvl`）在运行时查询 `R = SVL/32`，
  使 kernel 适配不同 SME 实现，而非固定 16。

### 6.3 说明
- 首版 kernel 采用“arm_sme 直写”，即把算法 1:1 映射到 arm_sme op，
  类似用 MLIR 写汇编。目的是完全掌控布局与性能。

## 7. 编译流程

```
mlir-opt sme/sconv_sme_direct.mlir \
  --enable-arm-streaming="streaming-mode=streaming-locally za-mode=new-za" \
  --arm-sme-vector-legalization \
  --convert-vector-to-arm-sme \
  --arm-sme-outer-product-fusion \
  --convert-to-llvm \
  | mlir-translate --mlir-to-llvmir > /tmp/kernel.ll

clang -march=armv9-a+sme -O3 -c /tmp/kernel.ll -o sconv_sme_direct.o
```

- 链接运行时需追加 `libclang_rt.builtins.a`（SME ABI：`__arm_get_current_vg`、
  `__arm_tpidr2_save` 等）。
- `mlir-opt` 运行需 `LD_LIBRARY_PATH` 含 `miniconda3/lib` 与 LLVM `lib`
  （否则报 `GLIBCXX_3.4.30 not found`）。

## 8. 运行时接口与集成

### 8.1 签名
```c
void sconv_sme_direct(
    int64_t N,  int64_t Ic, int64_t Ih, int64_t Iw,
    int64_t Oc, int64_t Oh, int64_t Ow,
    int64_t ph, int64_t pw,              // SAME pad (ph=pw=1)
    const float* in,                     // NCHW 原始输入
    const float* wpack,                  // [ceil(Oc/R)][Ic*9][R]
    float* out);                         // NCHW 输出
```

### 8.2 transform op（通用判定，非配置表）

新增 `transform.structured.sconv_sme`（**不改动现有 `transform.structured.sconv`**）。
匹配所有 `linalg.conv_2d_nchw_fchw`，对每个 op 执行**两层通用判定**，
只对胜出者生成 `LLVM::CallOp`，其余保持不动（交给原 `sconv`/`to_blas` 流程）。

**第一层 · 结构可行性 `eligible(conv)`**（读 op 属性/shape，非配置名单）：
```cpp
dilation_h == 1 && dilation_w == 1 &&
stride_h == 1 && stride_w == 1 &&          // 首版；stride 后续可扩
Fh == 3 && Fw == 3 &&                       // 支持集首版 {3x3}，设计成可扩展集合
形状为静态（否则回退 GEMM） &&
SME 可用（运行时/目标特性）
```
— 注意：**不再要求 `Oc % R == 0`**，尾块由 kernel 掩码处理。

**第二层 · 性能收益 `useSme(conv)`**（代价模型，含可调阈值）：
```cpp
// 解析估计：直接法 vs GEMM 的相对收益
//   复用度 low  → 直接法占优   sme_score 大
//   Oc 小、Ic 大、空间大 → GEMM 打包难摊薄、微内核 N 维利用不足
double gemm_score = f_gemm (N, Ic, Oc, Oh, Ow, Fh, Fw);   // 含 packing 体积项
double sme_score  = f_sme  (N, Ic, Oc, Oh, Ow, Fh, Fw);   // FLOPs / SME 有效峰值
bool use_sme = sme_score > gemm_score;
```
- `f_*` 为解析式，系数/阈值由**已有 11 个配置的实测值标定一次**，之后对未见
  配置外推 → **新增配置无需改代码**。
- 通过 op 选项暴露可调参数（如 `force`、`min_ic`、`max_oc`、`smem_bias`），
  便于调参与强制开关；`force` 可绕过代价模型（用于测试/对照）。

**执行动作**（仅对 `eligible && use_sme` 的 conv）：
- 生成 `LLVM::CallOp` 调用 `sconv_sme_direct`，传原始张量指针 + 几何参数。
- **权重打包**在 transform 侧生成一次性 packing 循环（写入 `wpack` 临时 buffer），
  避免每块重复打包。
- 移除原 conv 计算（与其关联的 packing/tiling 不再生成）。

### 8.3 选择策略（数据驱动，无硬编码）
- 决策完全由 8.2 的 `eligible + useSme` 函数给出，**不枚举具体配置**。
- 管线形态：脚本先跑 `sconv_sme`（部分替换），剩余 conv 再走标准
  `sconv` + `to_blas`；`sconv_sme` 对不满足的 op 是 no-op。
- 代价模型是"启发式估计 + 一次标定"，不追求精确；重点是**趋势正确**：
  打包开销相对计算越大、GEMM N 维越小，越倾向 SME 直接法。

## 9. 验证与验收

### 9.1 数值正确性
- 与 naive 实现 / PyTorch 逐元素比对，采用相对+绝对容差（fp32 累加，
  容差 ~1e-3 相对）。
- 覆盖：custom_9、custom_10；首次单一 `(N=1, 单 ocb, 单 col0)` 单元测试。
- **通用性用例**：额外构造非整倍数 `Oc`（尾行掩码）、`Oh*Ow` 非 `COLS` 整数倍
  （尾列掩码）、`Nh/Ic` 变化等形状，验证参数化与掩码正确。

### 9.2 性能
- 环境：38 线程，`numactl --cpunodebind=0 --membind=0`。
- 指标：custom_9 / custom_10 上 GFLOPS **≥ 现有 GEMM 路径**。
- 对照：现有 GEMM 路径、PyTorch conv2d。

### 9.3 回归
- 其余 9 个配置仍走 GEMM 路径，性能与正确性不变。

## 10. 风险与待验证项（由探针回答）

1. **性能上限未知**：手写 kernel 能否追平/超过厂商 kblas 是核心风险，
   探针必须首先回答。
2. **arm_sme 直写工作量大**：寻址、边界、切片存储易错。
3. **MLIR 流水线细节**：`arm_sme.outerproduct` 的累加器用法、
   是否需要 `convert-vector-to-arm-sme`、`vector<[4]xf32>` 与 tile 类型的
   正确构造，均需实测。
4. **权重打包开销**：kernel 内打包可能吃掉收益 → 倾向 transform 侧打包。
5. **边界处理**：掩码路径（operand mask）与预 pad 路径的正确性/性能取舍。
6. **代价模型标定**：`f_gemm/f_sme` 的形式与系数能否从现有测量中稳定拟合，
   以及它对未见配置的外推是否可靠（趋势正确即可）。

## 11. 执行顺序（探针优先）

**阶段 1（探针，answering “能否实现”）**
1. 确认 SVL（已确认 = 512）。
2. 跑通最小 MLIR ArmSME 程序（outerproduct → 运行、验证数值）。
3. 手写 custom_9 形状的 `sconv_sme_direct.mlir`（stride1、pad1）。
4. 走完整流水线编译运行，与 naive 比对正确性。
5. 测 GFLOPS，与现有 GEMM 路径对照。

**阶段 2（集成，仅当阶段 1 达标）**
6. 新增 `LowerToSME` transform op + `SConv.td` 定义，实现
   `eligible()` + 代价模型 `useSme()`。
7. 用已有 11 个配置的实测数据**标定代价模型系数**，并验证其在
   custom_9/10 上选出 SME、在其余 9 个上回退 GEMM。
8. 加 kernel 构建规则（CMake/sh）。
9. 接入 `test/lowering/sconv-sme.mlir`，跑通端到端。
10. 全量回归 + 性能验收（含 9.1 的通用性用例）。

## 12. 已定 / 待定

**已定（本 spec 采纳）**
- 选择机制：op 内部判定 = 结构谓词 + 可调代价模型（方案 A），不枚举配置。
- kernel：形状参数化、支持尾块掩码、运行时查询 SVL。
- 边界：优先掩码路径；预 pad 作为备选。
- 权重打包：transform 侧一次性完成。

**待定**
- `f_gemm/f_sme` 的具体解析式与系数（阶段 2 第 7 步标定）。
- 掩码路径在目标硬件上的性能是否可接受（探针/集成期实测）。
- stride>1、其它 kernel 尺寸的扩展时机（非首版）。
