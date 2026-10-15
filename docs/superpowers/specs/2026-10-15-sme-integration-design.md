# SME 直接卷积集成方案（transform 接入 + 运行时）设计方案

- 状态：Draft（待用户评审）
- 日期：2026-10-15
- 分支：dev
- 关联：`docs/superpowers/specs/2026-10-14-sme-direct-conv-design.md`（探针设计，已实现并达标）
- 前序：探针阶段（Task 1–5）已完成，GO/NO-GO **通过**（见 §1）。

## 1. 背景与探针结论

`2026-10-14-sme-direct-conv-design.md` 提出的 SME 直接卷积探针已实现：

- 用 `arm_sme` 方言直写的 kernel `sme/kernel/sconv_sme_direct.mlir`，
  经 `--test-lower-to-arm-sme` 流水线 AOT 编译，链接 `libclang_rt.builtins.a` 后可运行。
- 正确性与 naive NCHW SAME-pad 参考在 3 组形状（含 `Oc/Oh/Ow` 尾块）逐元素一致；
  并行结果与串行结果完全一致（0 diff）。
- 38 线程（`numactl --cpunodebind=0 --membind=0`）实测（useful GFLOPS =
  `2*N*Oh*Ow*Oc*Ic*9/t`）：

  | 配置 | SME 直接 | GEMM 路径(最优) | PyTorch | 门限 |
  |------|----------|------------------|---------|------|
  | custom_9  | 2050 | 1620 (nw32nf16) | 2131 | PASS |
  | custom_10 | 1972 | 1025 (nw64nf32) | 1116 | PASS |

  → 门限通过，本方案解决“如何把它接入编译器管线与 ConvBench”。

本方案是**集成**设计，不改变 kernel 算法本身（算法见前序 spec §5–§6）。

## 2. 目标与非目标

### 目标
- 提供管线入口 `sconv-opt -transform=test/lowering/sconv-sme.mlir`，对每个
  `linalg.conv_2d_nchw_fchw` **自动**决定走 SME 直接法（custom_9/10 一类）
  还是既有 GEMM（`sconv`+`to_blas`）路径。
- 选择机制 = 结构谓词 `eligible()` + 代价模型 `useSme()`（前序 spec §8），
  系数由已有 11 配置实测标定一次；**不枚举配置**。
- 数值正确（ConvBench `np.allclose(rtol=atol=1e-3)` 全过）；custom_9/10 上
  GFLOPS ≥ GEMM；其余 9 个配置走 GEMM 且行为不变（无回归）。

### 非目标
- 不改动 kernel 的数值算法（沿用探针 kernel）。
- 不支持 stride>1、非 3×3（首版支持集）。
- 不替换/改动现有 GEMM 路径与其结果。

## 3. 关键约束（决定集成形态）

1. **`transform.structured.sconv` 只接受 tensor 形态**（`lib/SConv.cpp` 内部构造
   `tensor::CollapseShapeOp` 等，见 `sconv-blas.mlir:11-21`），必须在
   `one_shot_bufferize` **之前**运行。
2. **`transform.lower.to_blas` 只接受 memref 形态**（`lib/LowerToBLAS.cpp:155-175`
   断言 operand 为 memref），必须在 bufferize **之后**运行。
3. 因此 SME 的“选择”必须在 tensor 阶段完成并**打标记**，真正的
   调用生成放到 bufferize **之后**——沿用 `microkernel` 标记的既有模式。
4. **ConvBench 运行器（`mlir-cpu-runner`）是单线程的**（`run_convbench.py:105-106`
   未传线程选项），GEMM 的并行来自 kblas 内部 OpenMP。故 SME 路径必须自备
   多线程，否则与 kblas 比较不公平。

## 4. 总体架构

```
        AOT 编译期（kernel）                        运行期（shim + 线程）
┌────────────────────────────┐          ┌────────────────────────────────────┐
│ sme/kernel/sconv_sme_direct.mlir│      │ sconv_sme_direct(N..Ow,ph,pw,       │
│   @sconv_sme_compute       │          │                  in,filter,out)     │
│   (arm_sme；预 pad 输入/输出)│          │   ├─ pad 输入 → 预 pad buffer        │
│        │ mlir-opt 流水线    │          │   ├─ pack 权重 → wpack               │
│        ▼                   │  link    │   ├─ pthreads 分任务调用             │
│ sconv_sme_compute.o        │─────────▶│   │    @sconv_sme_compute(T0,TN)     │
└────────────────────────────┘          │   └─ unpad 输出 → NCHW               │
                                         └────────────────────────────────────┘
        transform（编译期）                       链接为 libsconv_sme_direct.so
┌──────────────────────────────────────────────────────────────┐
│ sme_select(tensor) → 标记 sme.direct；to_sme(memref) → CallOp  │
│ 其余 conv 走 sconv + to_blas（不变）                            │
└──────────────────────────────────────────────────────────────┘
```

**分工原则**：MLIR kernel 只做**计算**（预 pad 输入/输出 + wpack，任务区间
`[T0,TN)`）；**pad/unpad 与线程调度**放在 C shim。理由：线程需要“pad 一次 →
多线程算 → unpad 一次”，若 kernel 自含 pad 则无法作为被线程化的单元（每个
线程会重复 pad）。C shim 复用探针 driver 已验证的逻辑。

## 5. 组件清单与接口

### 5.1 计算 kernel（沿用探针，改名）
- 文件：`sme/kernel/sconv_sme_direct.mlir`，函数名 `sconv_sme_direct`
  → **改名 `sconv_sme_compute`**（避免与 C shim 同名）。
- 签名（保持探针不变，预 pad 输入/输出）：
  ```c
  void sconv_sme_compute(
      int64_t N, int64_t Ic, int64_t IwP, int64_t OcP,
      int64_t OhP, int64_t OwP, int64_t T0, int64_t TN,
      /*in:*/     rank-2 memref<?x?xf32> [N*Ic][IhP*IwP],
      /*wpack:*/  rank-1 memref<?xf32>  [(Ic*9)*OcP],
      /*out:*/    rank-2 memref<?x?xf32> [N*OcP][OhP*OwP]);
  ```
  其中 `IhP=OhP+2, IwP=OwP+2, OcP=ceil(Oc/R)*R, OhP=ceil(Oh/4)*4,
  OwP=ceil(Ow/R)*R, R=arm_sme.streaming_vl<word>`。任务展平
  `task=((n*Ocb)+ocb)*OhB+ohb`，`Ocb=OcP/R, OhB=OhP/4`。
- ABI：MLIR 将每个 memref 参数降为**扁平化描述符标量**（rank-1→5、
  rank-2→7），这与 C 侧 `MemRef2D/MemRef1D`（`sme/test/harness.h`）一致——
  **这是 shim 与 kernel 之间的强约定**。

### 5.2 C shim（新增）
- 文件：`sme/runtime/sconv_sme_shim.c`（或 `scripts/convbench/runtime/sconv_sme_direct.c`）。
- 对外签名（transform 侧按此生成 `llvm.func` 声明，**裸指针**，仿 `sgemm_blas_kernel`）：
  ```c
  int sconv_sme_direct(
      int64_t N,  int64_t Ic, int64_t Ih, int64_t Iw,
      int64_t Oc, int64_t Oh, int64_t Ow,
      int64_t ph, int64_t pw,
      const float* in,      // NCHW 原始输入
      const float* filter,  // [Oc][Ic][Fh][Fw] 原始权重
      float* out);          // NCHW 输出
  ```
- 行为：
  1. 计算 `OcP/OhP/OwP/IhP/IwP`；`malloc` 预 pad 输入、`calloc` wpack、`calloc` 预 pad 输出。
  2. 输入 pad：`pin[ic][r+ph][c+pw] = in[...]`（零填充边界）。
  3. 权重打包：`wpack[(ic*9+kh*3+kw)*OcP + oc] = filter[oc][ic][kh][kw]`，尾部 `oc∈[Oc,OcP)` 为 0。
  4. 线程：任务总数 `T = N*Ocb*OhB`；用 `pthreads` 把 `[0,T)` 切块，
     每线程按 `[T0,TN)` 调用 `sconv_sme_compute`（构造 memref 描述符）。
  5. 输出 unpad：`out[n][oc][oh][ow] = pout[n*OcP+oc][oh*OwP+ow]`（`oc<Oc,oh<Oh,ow<Ow`）。
  6. `free`。
- 线程数：默认 = 在线 CPU 数（`sysconf(_SC_NPROCESSORS_ONLN)`）；可用环境变量
  `SCONV_SME_THREADS` 覆盖。运行器在 `numactl` 下启动，故绑定到 node0。

### 5.3 transform op（新增两个，`include/SConv.td` + `lib/LowerToSME.cpp`）
两 op 由 `registerTransformOps<GET_OP_LIST>` 自动注册（`lib/SConv.cpp:126-129`），
`lib/CMakeLists.txt` 追加 `LowerToSME.cpp`。

**(a) `transform.lower.sme_select`** — tensor 阶段执行：
```
transform.lower.sme_select %convs {force}
  : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.op<"linalg.conv_2d_nchw_fchw">
```
- 参数：`TransformHandleTypeInterface:$target`；可选 `UnitAttr:$force` + 阈值
  （`min_ic`/`max_oc`/`sme_bias` 等，attr-dict）。不持有 `name`（它不生成调用）。
- 逻辑：对句柄内每个 `linalg.conv_2d_nchw_fchw` 求 `eligible(conv)` 与
  `useSme(conv)`（§6）；命中者打 unit attr **`sme.direct`**，未命中者不动。
- 结果：返回**未命中**的 conv 句柄 `$remaining`，供脚本交给
  `transform.structured.sconv`（从而 SConv 天然跳过已选中的 op，无需改 SConv.cpp）。

**(b) `transform.lower.to_sme`** — bufferize 之后执行：
```
transform.lower.to_sme "sconv_sme_direct", %smec
  : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.any_op
```
- 参数：`StrAttr:$name`、`$target`。
- 逻辑：对每个带 `sme.direct` 的（已 bufferize 的）conv：
  - 复用 `createBlasCallOp` 中的 `createPtrWithOffsetFromMemRef`，从 conv 的
    operand 0/1/2 取 `!llvm.ptr`：`in/filter/out`；
  - 从静态 shape 取标量 `N,Ic,Ih,Iw,Oc,Oh,Ow`，`ph/pw` 取 conv 的 padding 属性；
  - 声明 `llvm.func @<name>(i64×9, ptr, ptr, ptr) -> i32`（若模块中已存在则复用）；
  - 生成 `LLVM::CallOp`，随后 `eraseOp(conv)`。
- 说明：与 `LowerToBlasOp` 实现同构，仅参数列表不同（裸指针 + 几何标量）。

### 5.4 transform 脚本（新增 `test/lowering/sconv-sme.mlir`）
```
%convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
%rest  = transform.lower.sme_select %convs {force}
%uk, %loops = transform.structured.sconv %rest
%m = transform.bufferization.one_shot_bufferize %arg0 {bufferize_function_boundaries = true}
%funcs = transform.structured.match ops{["func.func"]} in %m
apply_patterns (extract_address_computations, expand_strided_metadata)
apply_cse / apply_dce
%smec = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} attributes {sme.direct} in %m
transform.foreach %smec : ... { transform.lower.to_sme "sconv_sme_direct", %op }
%uk2 = transform.structured.match ops{["linalg.generic"]} attributes {microkernel} in %m
transform.foreach %uk2 : ... { transform.lower.to_blas "sgemm_blas_kernel", %op }
```

### 5.5 构建（`run_convbench.py` + `sme/`）
- `sme/build.sh` 编译 kernel → `sconv_sme_compute.o`。
- 编译 shim → `sconv_sme_direct.o`（含 `pthread`）。
- 链接为 `build/convbench_runtime/libsconv_sme_direct.so`（`clang -shared -lpthread`）。
- `run_convbench.py`：`build_runtime_libs()` 增加上述构建；`get_libs()` 追加该 `.so`；
  新增 `--transform` 选项选择 `sconv-sme.mlir`。

## 6. 选择机制（复用前序 spec §8，不枚举配置）

**第一层 · 结构谓词 `eligible(conv)`**：`dilation==1`、`stride==1`、`Fh==Fw==3`、
形状静态、pad 对称（ph==pw）。不要求 `Oc%R==0`（kernel 以 pad 处理尾块）。

**第二层 · 代价模型 `useSme(conv)`**：比较 `f_gemm` 与 `f_sme`（均为
`N,Ic,Oc,Oh,Ow,Fh,Fw` 的解析式，`f_gemm` 含 packing 体积项）。系数由 11 个配置
实测**标定一次**；`force` 选项可绕过，阈值可由 attr-dict 覆盖。判定
`use_sme = force || (sme_score > gemm_score)`。

标定目标（趋势正确即可）：custom_9/10 选 SME，其余 9 个回落 GEMM。

## 7. 验证与验收

1. **选择正确性**：对包含 11 个 conv 的模块运行 `sconv-opt -transform=sconv-sme.mlir`，
   断言 custom_9/10 被替换为 `llvm.call @sconv_sme_direct`，其余 9 个被替换为
   `llvm.call @sgemm_blas_kernel`（并保留 `microkernel` 标记的中间形态可对照）。
2. **数值正确性**：ConvBench `sconv-sme` 全 11 配置 `np.allclose(rtol=atol=1e-3)` PASS。
3. **性能**：38 线程 node0；custom_9/10 的 GFLOPS ≥ GEMM；其余 9 个与既有持平。
4. **无回归**：既有 `sconv-blas.mlir` 路径与其测试不变、仍通过。

## 8. 风险与缓解

1. **runner 单线程**：已由 C shim 的 pthreads 解决；线程数默认全核，可调。
2. **memref 描述符 ABI 一致性**：shim 与 kernel 必须同一 ABI（探针已验证）。
   缓解：shim 复用 `harness.h` 的 `MemRef1D/2D` 构造；kernel 只改函数名。
3. **transform 阶段耦合**：`sme_select`/`to_sme` 依赖 `sme.direct` 标记跨 bufferize
   存活——既定模式（`microkernel`）已证明可行。
4. **代价模型外推**：只要求趋势正确；`force` 兜底。
5. **pad 开销**：输入 pad 一次（N*Ic*Ih*Iw 拷贝）在计算面前可忽略；输出 unpad
   同理。对 custom_9/10 其空间维为 R 的整倍数，无额外尾块浪费。

## 9. 与前序 spec 的差异（探针期间的设计变更）

- **边界**：由“优先掩码”改为**预 pad**（前序 spec §5.5 的备选路径）——kernel
  不变，正确性已由 naive 三形状验证覆盖。
- **线程**：引入 C shim 做 pad + pthreads；MLIR kernel 保持“计算 + 任务区间”。
- **接口**：kernel 计算入口为预 pad 的 `sconv_sme_compute`；对外入口
  `sconv_sme_direct` 在 C shim，接受原始 NCHW，与 transform 生成的裸指针调用一致。
- **权重打包**：由 C shim 承担（等价于前序 spec §8 的 transform 侧打包，位置不同）。

## 10. 执行顺序

1. 扩展 kernel：函数改名 `sconv_sme_compute`，保持探针逻辑；扩展 build.sh 产出 `.o`。
2. 新增 C shim，接入探针 driver 逻辑（pad/pack/线程/unpad），单独测试正确性。
3. 新增 `LowerToSMEOp` + `lib/LowerToSME.cpp`（先 `to_sme`，后 `sme_select`）。
4. 标定代价模型，编写 `test/lowering/sconv-sme.mlir`，验证选择。
5. 接入 `run_convbench.py`（构建 + 加载 + `--transform`），跑全 11 配置回归与性能。
