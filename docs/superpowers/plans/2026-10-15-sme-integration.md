# SME 直接卷积集成（Phase 2）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已验证的 SME 直接卷积 kernel 接入 SConvTransform 编译器管线与 ConvBench，使 `sconv-opt -transform=test/lowering/sconv-sme.mlir` 能对每个卷积**自动**选择 SME 直接法或既有 GEMM 路径。

**Architecture:** 新增两个 transform op —— `transform.lower.sme_select`（tensor 阶段：结构谓词 + 代价模型，命中者打 `sme.direct` 标记，返回未命中句柄）与 `transform.lower.to_sme`（bufferize 之后：仿 `to_blas` 生成裸指针 `LLVM::CallOp`）。MLIR kernel 保持“计算 + 任务区间”；新增 C shim 负责 pad / 权重打包 / pthreads / unpad。

**Tech Stack:** MLIR 19.1.7（Transform dialect、ArmSME）、C++17、C、CMake、pthreads、Python（ConvBench runner）。

**Spec:** `docs/superpowers/specs/2026-10-15-sme-integration-design.md`（本计划实现该 spec；executor 需同时阅读 spec 与探针 spec `docs/superpowers/specs/2026-10-14-sme-direct-conv-design.md`）。

## Global Constraints

- 平台：openEuler 22.03，aarch64 Kunpeng，SME 可用，**SVL=512 → R=16**（f32 ZA tile = 16×16）。仅本机验证。
- 无 sudo；`pkill`/后台命令会挂起 shell —— 用 `setsid nohup … &` + `pgrep`。
- 每次命令前 `source /home/z00889957/100Work/SConvTransform/env.sh`；LLVM 在 `$LLVM_BUILD_DIR`。
- `mlir-opt` 需 `LD_LIBRARY_PATH` 含 `$LLVM_BUILD_DIR/lib` 与 `/home/z00889957/miniconda3/lib`。
- 链接须加 `$LLVM_BUILD_DIR/lib/clang/19/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a`。
- 编译 SME kernel 必须走 `--test-lower-to-arm-sme`（自动注入 `aarch64_pstate_sm_body`/`aarch64_new_za`）；`-march=armv9-a+sme`。
- **memref ABI 约定（强约束）**：MLIR 将 memref 形参降为扁平化描述符标量（rank-1→5、rank-2→7），C 侧用 `MemRef1D/MemRef2D` 对应。kernel 与 shim 必须一致。
- **框架输入已预 pad**：`gen_payload.py:26-27` 令 `PHI=HI+HPTOP+HPBOTTOM`，且 `linalg.conv_2d_nchw_fchw` **无 padding 属性**（valid conv）。因此 shim 不再需要 `ph/pw` 参数。
  - **Ruling 1（对 spec §5.2 的精化）**：`sconv_sme_direct` 去掉 `ph,pw`，改传输入张量维度 `Ih,Iw`；eligibility 要求 `Ih==Oh+2 && Iw==Ow+2`（3×3 stride1 的预 pad/valid 形态一致）。shim 只在 `IwP>Iw` 或 `IhP>Ih`（尾块需要）时补零扩展右下。
- 目标配置（须选 SME）：custom_9 `N4 Ic384 Ih82 Iw82 Oc96 Oh80 Ow80`、custom_10 `N4 Ic768 Ih42 Iw42 Oc96 Oh40 Ow40`。
- 性能基准线（38 线程 node0，useful GFLOPS）：GEMM c9=1620、c10=1025；SME 探针 c9≈2050、c10≈1972。

## Review Focus

1. **预 pad 输入的对齐**：框架输入左上角已含 border；shim 复用/扩展时若把数据错位（应为左上锚定、只补右下），输出整体平移——由 naive valid-conv 比对捕获。
2. **尾块（`Oc/Oh/Ow` 非 `R/4` 整倍数）**：`OcP/OhP/OwP` 上取整；越界行/列必须落进补零区、unpad 只回写有效区——由非整倍数形状用例捕获。
3. **`sme.direct` 标记跨 bufferize 存活 & 只作用于卷积**：`sme_select` 返回的“未命中”句柄必须让 `sconv` 只处理剩余 conv；不可误伤 packing/tiling 生成的 generic。
4. **memref ABI 一致性**：shim 传给 `sconv_sme_compute` 的描述符字段顺序/数量必须与 MLIR 降级一致——由 shim 数值测试在实机捕获（错则 SIGSEGV 或错值）。
5. **选择路由**：`sme_select` 必须恰好选中 c9/c10、其余走 GEMM——由 11 配置路由测试 + ConvBench 回归捕获。

---

## Task 1: Kernel 计算入口改名 + 探针回归

**Files:**
- Modify: `sme/kernel/sconv_sme_direct.mlir:1`（函数名）
- Modify: `sme/test/direct_main.c:8-12`（extern 名 + 调用）
- Test: `sme/Makefile`（`direct` 目标，无需改）

**Interfaces:**
- Produces: 计算入口符号 `sconv_sme_compute`，签名不变（预 pad 输入/输出 + `T0,TN`）：
  ```c
  void sconv_sme_compute(int64_t N,int64_t Ic,int64_t IwP,int64_t OcP,int64_t OhP,
                         int64_t OwP,int64_t T0,int64_t TN,
                         /*in*/    memref<?x?xf32> [N*Ic][IhP*IwP],
                         /*wpack*/ memref<?xf32>    [(Ic*9)*OcP],
                         /*out*/   memref<?x?xf32> [N*OcP][OhP*OwP]);
  ```

- [ ] **Step 1: 改函数名**

在 `sme/kernel/sconv_sme_direct.mlir` 中把 `func.func @sconv_sme_direct(` 改为 `func.func @sconv_sme_compute(`（文件顶部注释里提到 `@sconv_sme_direct` 的一并改）。

- [ ] **Step 2: 更新探针驱动 extern**

`sme/test/direct_main.c` 中把 `sconv_sme_direct` 改名为 `sconv_sme_compute`（`extern void` 声明与 `call_kernel` 内调用各一处）。

- [ ] **Step 3: 运行回归**

Run:
```bash
cd /home/z00889957/100Work/SConvTransform/sme && source ../env.sh && make -s clean && make direct
```
Expected: `ALL PASS`（三个形状）。

- [ ] **Step 4: 提交**

```bash
git add sme/kernel/sconv_sme_direct.mlir sme/test/direct_main.c
git commit -m "refactor(sme): rename compute entry to sconv_sme_compute"
```

---

## Task 2: C shim（pad + 权重打包 + pthreads + unpad）+ 数值测试

**Files:**
- Create: `sme/runtime/sme_abi.h`（共享 memref 助手 + 两个 extern 声明）
- Create: `sme/runtime/sconv_sme_shim.c`
- Create: `sme/test/shim_main.c`
- Modify: `sme/test/harness.h`（改为 include `../runtime/sme_abi.h`，删除重复结构）
- Modify: `sme/Makefile`（新增 `shim` 目标）

**Interfaces:**
- Consumes: `sconv_sme_compute`（Task 1）。
- Produces: 对外入口（transform 按此生成 `llvm.func`，裸指针）
  ```c
  int sconv_sme_direct(int64_t N,int64_t Ic,int64_t Ih,int64_t Iw,
                       int64_t Oc,int64_t Oh,int64_t Ow,
                       const float* in,      /* NCHW，已含 border */
                       const float* filter,  /* [Oc][Ic][3][3] */
                       float* out);          /* NCHW */
  ```

- [ ] **Step 1: 建共享 ABI 头**

Create `sme/runtime/sme_abi.h`:
```c
#ifndef SME_ABI_H
#define SME_ABI_H
#include <stdint.h>
/* MLIR memref 描述符降级后的扁平 ABI：rank-1 = 5 个标量，rank-2 = 7 个标量。 */
typedef struct { float *allocated, *aligned; int64_t offset, size, stride; } MemRef1D;
typedef struct { float *allocated, *aligned; int64_t offset, size0, size1, stride0, stride1; } MemRef2D;
static inline MemRef1D mk1d(float *p, int64_t n){ MemRef1D m={p,p,0,n,1}; return m; }
static inline MemRef2D mk2d(float *p, int64_t r, int64_t c){
  MemRef2D m={p,p,0,r,c,c,1}; return m;
}
/* 计算 kernel（预 pad 输入/输出 + 任务区间 [T0,TN)） */
extern void sconv_sme_compute(
    int64_t N,int64_t Ic,int64_t IwP,int64_t OcP,int64_t OhP,int64_t OwP,
    int64_t T0,int64_t TN,
    float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,
    float*,float*,int64_t,int64_t,int64_t,
    float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t);
/* 对外入口（pad + 打包 + 线程 + unpad） */
int sconv_sme_direct(int64_t N,int64_t Ic,int64_t Ih,int64_t Iw,
                     int64_t Oc,int64_t Oh,int64_t Ow,
                     const float* in,const float* filter,float* out);
#endif
```

- [ ] **Step 2: 重构 harness.h 复用该头**

`sme/test/harness.h` 改为：
```c
#ifndef SME_TEST_HARNESS_H
#define SME_TEST_HARNESS_H
#include "../runtime/sme_abi.h"
#include <math.h>
static inline int close_enough(float got, float want){
  float d = fabsf(got-want);
  return d <= 1e-3f*(1.0f+fabsf(want));
}
#endif
```
（`direct_main.c` 只用 `mk1d/mk2d/close_enough`，无需其他改动。）

- [ ] **Step 3: 写 shim**

Create `sme/runtime/sconv_sme_shim.c`:
```c
#include "sme_abi.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define R 16   /* SVL=512 -> R=16（本机已确认） */

typedef struct {
  int64_t N,Ic,IwP,OcP,OhP,OwP,T0,TN;
  float *in,*wpack,*out;
} Job;

static void run_range(Job*j){
  MemRef2D mi = mk2d(j->in, j->N*j->Ic, j->IwP*(j->OhP+2));
  MemRef1D mw = mk1d(j->wpack, j->Ic*9*j->OcP);
  MemRef2D mo = mk2d(j->out, j->N*j->OcP, j->OhP*j->OwP);
  sconv_sme_compute(j->N,j->Ic,j->IwP,j->OcP,j->OhP,j->OwP,j->T0,j->TN,
    mi.allocated,mi.aligned,mi.offset,mi.size0,mi.size1,mi.stride0,mi.stride1,
    mw.allocated,mw.aligned,mw.offset,mw.size,mw.stride,
    mo.allocated,mo.aligned,mo.offset,mo.size0,mo.size1,mo.stride0,mo.stride1);
}
static void*worker(void*p){ run_range((Job*)p); return NULL; }

static int sme_threads(void){
  const char*e=getenv("SCONV_SME_THREADS");
  if(e&&*e){int n=atoi(e); if(n>0)return n;}
  long n=sysconf(_SC_NPROCESSORS_ONLN);
  return n>0?(int)n:1;
}

int sconv_sme_direct(int64_t N,int64_t Ic,int64_t Ih,int64_t Iw,
                     int64_t Oc,int64_t Oh,int64_t Ow,
                     const float*in,const float*filter,float*out){
  int64_t OhP=(Oh+3)/4*4, OwP=(Ow+R-1)/R*R, OcP=(Oc+R-1)/R*R;
  int64_t IhP=OhP+2, IwP=OwP+2;

  /* 权重打包：wpack[(ic*9+kh*3+kw)*OcP + oc] */
  float*wpack=calloc((size_t)Ic*9*OcP,sizeof(float));
  for(int64_t ic=0;ic<Ic;ic++)
    for(int64_t kh=0;kh<3;kh++)
      for(int64_t kw=0;kw<3;kw++)
        for(int64_t oc=0;oc<Oc;oc++)
          wpack[(ic*9+kh*3+kw)*OcP+oc]=filter[((oc*Ic+ic)*3+kh)*3+kw];

  /* 输入：框架已左上预 pad；仅在尾块需要时把行/列扩到 IhP×IwP（右下补零）。 */
  float*pin; int need_pad = (Ih!=IhP)||(Iw!=IwP);
  if(!need_pad){ pin=(float*)in; }
  else {
    pin=calloc((size_t)N*Ic*IhP*IwP,sizeof(float));
    for(int64_t n=0;n<N;n++)
      for(int64_t ic=0;ic<Ic;ic++)
        for(int64_t r=0;r<Ih;r++)
          memcpy(pin+((n*Ic+ic)*IhP+r)*IwP,
                 in +((n*Ic+ic)*Ih +r)*Iw,
                 (size_t)Iw*sizeof(float));
  }

  /* 预 pad 输出 */
  float*pout=calloc((size_t)N*OcP*OhP*OwP,sizeof(float));

  int64_t Ocb=OcP/R, OhB=OhP/4, T=N*Ocb*OhB;
  int nt=sme_threads(); if(nt>T)nt=(int)T; if(nt<1)nt=1;

  Job base={N,Ic,IwP,OcP,OhP,OwP,0,0,pin,wpack,pout};
  if(nt==1){ base.T0=0; base.TN=T; run_range(&base); }
  else {
    pthread_t*th=malloc(sizeof(pthread_t)*nt);
    Job*ar=malloc(sizeof(Job)*nt);
    int64_t per=(T+nt-1)/nt, t=0;
    for(int i=0;i<nt;i++){
      int64_t cnt=per; if(t+cnt>T)cnt=T-t; if(cnt<0)cnt=0;
      ar[i]=base; ar[i].T0=t; ar[i].TN=cnt; t+=cnt;
      pthread_create(&th[i],NULL,worker,&ar[i]);
    }
    for(int i=0;i<nt;i++)pthread_join(th[i],NULL);
    free(th); free(ar);
  }

  /* unpad 输出（只回写有效 oc<Oc,oh<Oh,ow<Ow） */
  for(int64_t n=0;n<N;n++)
    for(int64_t oc=0;oc<Oc;oc++)
      for(int64_t oh=0;oh<Oh;oh++)
        memcpy(out+((n*Oc+oc)*Oh+oh)*Ow,
               pout+((n*OcP+oc)*OhP+oh)*OwP,
               (size_t)Ow*sizeof(float));

  if(need_pad)free(pin);
  free(wpack); free(pout);
  return 0;
}
```

- [ ] **Step 4: 写 shim 数值测试（应失败：先无 shim 目标）**

Create `sme/test/shim_main.c`:
```c
#include "harness.h"
#include <stdio.h>
#include <stdlib.h>

/* 输入为“已预 pad”的 NCHW：Ih=Oh+2, Iw=Ow+2（valid conv）。 */
static int run(int N,int Ic,int Oh,int Ow,int Oc){
  int Ih=Oh+2, Iw=Ow+2;
  float*in=malloc(sizeof(float)*(size_t)N*Ic*Ih*Iw);
  float*flt=malloc(sizeof(float)*(size_t)Oc*Ic*9);
  float*out=malloc(sizeof(float)*(size_t)N*Oc*Oh*Ow);
  float*ref=malloc(sizeof(float)*(size_t)N*Oc*Oh*Ow);
  for(size_t i=0;i<(size_t)N*Ic*Ih*Iw;i++) in[i]=0.1f*(float)((i%29)-13);
  for(size_t i=0;i<(size_t)Oc*Ic*9;i++)    flt[i]=0.2f*(float)((i%17)-8);
  for(int n=0;n<N;n++)for(int oc=0;oc<Oc;oc++)for(int oh=0;oh<Oh;oh++)for(int ow=0;ow<Ow;ow++){
    float s=0;
    for(int ic=0;ic<Ic;ic++)for(int kh=0;kh<3;kh++)for(int kw=0;kw<3;kw++)
      s+=in[((size_t)(n*Ic+ic)*Ih+oh+kh)*Iw+ow+kw]*flt[(((size_t)oc*Ic+ic)*3+kh)*3+kw];
    ref[((size_t)(n*Oc+oc)*Oh+oh)*Ow+ow]=s;
  }
  sconv_sme_direct(N,Ic,Ih,Iw,Oc,Oh,Ow,in,flt,out);
  int bad=0;
  for(size_t i=0;i<(size_t)N*Oc*Oh*Ow;i++) if(!close_enough(out[i],ref[i])) bad++;
  printf("N%d Ic%d Oh%d Ow%d Oc%d: %s (%d bad)\n",N,Ic,Oh,Ow,Oc,bad?"FAIL":"PASS",bad);
  free(in);free(flt);free(out);free(ref);
  return bad?1:0;
}
int main(void){
  int bad=0;
  bad|=run(1,2,8,8,16);
  bad|=run(1,3,10,10,17);
  bad|=run(2,5,7,13,32);
  printf(bad?"SHIM FAIL\n":"SHIM PASS\n");
  return bad?1:0;
}
```

- [ ] **Step 5: 运行验证失败（RED）**

Run:
```bash
cd /home/z00889957/100Work/SConvTransform/sme && source ../env.sh && make -s
clang -O2 -c test/shim_main.c -o /tmp/opencode/shim_main.o 2>&1 | head
```
Expected: `make` 未含 shim 目标（此刻只 `smoke`+`direct`）；`shim_main.c` 因 `sconv_sme_direct` 无定义若单独链接会 `undefined reference`。

- [ ] **Step 6: 加 Makefile 目标**

在 `sme/Makefile` 的 `direct` 目标后加入（并把 `shim` 加进 `all` 与 `.PHONY`）：
```make
shim: kernel/sconv_sme_direct.o runtime/sconv_sme_shim.o test/shim_main.o
	clang $(SME) $^ $(BUILTINS) -lpthread -o shim.out
	@./shim.out

runtime/%.o: runtime/%.c runtime/sme_abi.h
	clang $(CFLAGS) -c $< -o $@
```
并把 `all: smoke direct` 改为 `all: smoke direct shim`，`.PHONY` 增加 `shim`，`clean` 增加 `shim.out runtime/*.o`。

- [ ] **Step 7: 构建并验证（GREEN）**

Run:
```bash
cd /home/z00889957/100Work/SConvTransform/sme && source ../env.sh && make -s clean && make
```
Expected: `SMOKE PASS` / `ALL PASS` / `SHIM PASS`。

- [ ] **Step 8: 提交**

```bash
git add sme/runtime sme/test/harness.h sme/test/shim_main.c sme/Makefile
git commit -m "feat(sme): C shim (pad/pack/threads/unpad) + numeric test"
```

---

## Task 3: `transform.lower.to_sme`（生成裸指针 CallOp）

**Files:**
- Modify: `include/SConv.td`（在 `LowerToBlasOp` 后新增 `LowerToSMEOp`）
- Create: `lib/LowerToSME.cpp`
- Modify: `lib/CMakeLists.txt:6-9`（追加 `LowerToSME.cpp`）
- Create: `test/lowering/sme/to_sme_in.mlir`
- Create: `test/lowering/sme/to_sme_transform.mlir`

**Interfaces:**
- Consumes: `sconv_sme_direct`（Task 2）。
- Produces: transform op `transform.lower.to_sme`，作用于带 `sme.direct` 的（已 bufferize）`linalg.conv_2d_nchw_fchw`，生成
  `llvm.func @sconv_sme_direct(i64,i64,i64,i64,i64,i64,i64, !llvm.ptr,!llvm.ptr,!llvm.ptr) -> i32` 与调用，并擦除原 conv。

- [ ] **Step 1: td 定义**

在 `include/SConv.td` 的 `LowerToBlasOp` 定义之后追加：
```tablegen
def LowerToSMEOp : Op<Transform_Dialect, "lower.to_sme",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Lower an SME-selected whole conv to the sconv_sme_direct runtime call.";
  let description = [{}];
  let arguments = (ins
    StrAttr:$name,
    TransformHandleTypeInterface:$target
  );
  let results = (outs TransformHandleTypeInterface:$output);
  let assemblyFormat = "$name `,` $target attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
```

- [ ] **Step 2: 实现 LowerToSME.cpp**

Create `lib/LowerToSME.cpp`（复刻 `lib/LowerToBLAS.cpp` 的结构；此处给出要点，完整仿写）：
```cpp
#include <cassert>
#include "SConv.h"
#include <llvm/ADT/Twine.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::LLVM;
using namespace mlir::transform;
using LLVMCallOp = LLVM::CallOp;

namespace {
// 与 LowerToBLAS.cpp 相同：从 memref 取 (ptr, offset) 得到 !llvm.ptr
// （直接复制 createPtrWithOffsetFromMemRef 实现，勿改）

static FailureOr<LLVMFuncOp> getOrCreateSmeFuncOp(TransformRewriter &rw,
                                                  LowerToSMEOp op, ModuleOp mod) {
  auto name = op.getName();
  for (auto f : mod.getOps<LLVMFuncOp>()) if (f.getName() == name) return f;
  rw.setInsertionPointToEnd(mod.getBody());
  auto ctx = mod.getContext();
  auto i64 = IntegerType::get(ctx,64), i32 = IntegerType::get(ctx,32);
  auto ptr = LLVMPointerType::get(ctx);
  SmallVector<Type,16> args = { i64,i64,i64,i64,i64,i64,i64, ptr,ptr,ptr };
  return rw.create<LLVMFuncOp>(mod.getLoc(), name,
             LLVMFunctionType::get(i32, args));
}

static FailureOr<Operation*> lowerConvToSme(TransformRewriter &rw,
                                            LowerToSMEOp op, linalg::Conv2DNchwFchwOp conv) {
  auto mod = conv->getParentOfType<ModuleOp>();
  if (!mod) return conv.emitError() << "payload must be inside a module";
  auto callee = getOrCreateSmeFuncOp(rw, op, mod);
  if (failed(callee)) return conv.emitError() << "failed to declare SME func";

  auto inTy  = cast<ShapedType>(conv.getInput().getType());
  auto fTy   = cast<ShapedType>(conv.getFilter().getType());
  auto outTy = cast<ShapedType>(conv.getOutput().getType());
  auto i64 = [&](int64_t v){ return rw.create<arith::ConstantIntOp>(conv.getLoc(), v, 64); };
  auto cvt = [&](Value memref)->Value {
    // 复用 LowerToBLAS.cpp 中 createPtrWithOffsetFromMemRef 的逻辑
    return ...; };

  rw.setInsertionPointAfter(conv);
  Value N=i64(inTy.getDimSize(0)), Ic=i64(inTy.getDimSize(1)),
        Ih=i64(inTy.getDimSize(2)), Iw=i64(inTy.getDimSize(3)),
        Oc=i64(fTy.getDimSize(0)),
        Oh=i64(outTy.getDimSize(2)), Ow=i64(outTy.getDimSize(3));
  Value pIn=createPtr(rw, conv.getInput(), "in");
  Value pF =createPtr(rw, conv.getFilter(), "filter");
  Value pO =createPtr(rw, conv.getOutput(), "out");
  auto call = rw.create<LLVMCallOp>(conv.getLoc(), *callee,
        ValueRange{N,Ic,Ih,Iw,Oc,Oh,Ow,pIn,pF,pO});
  rw.eraseOp(conv);
  return call.getOperation();
}
} // namespace

DiagnosedSilenceableFailure LowerToSMEOp::apply(TransformRewriter &rewriter,
    TransformResults &results, TransformState &state) {
  SmallVector<Operation*,8> res;
  for (Operation *t : state.getPayloadOps(getTarget())) {
    auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(t);
    if (!conv) { t->emitError() << "expected linalg.conv_2d_nchw_fchw"; 
                 return DiagnosedSilenceableFailure::definiteFailure(); }
    if (!conv->hasAttr("sme.direct")) continue;  // 未选中者不动
    auto r = lowerConvToSme(rewriter, *this, conv);
    if (failed(r)) return DiagnosedSilenceableFailure::definiteFailure();
    res.push_back(*r);
  }
  results.set((*this)->getOpResult(0), res);
  return DiagnosedSilenceableFailure::success();
}
void LowerToSMEOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &e) {
  consumesHandle(getTargetMutable(), e);
  producesHandle(getOperation()->getOpResults(), e);
  modifiesPayload(e);
}
LogicalResult LowerToSMEOp::verify() {
  return getName().empty() ? emitOpError("expected non-empty kernel name") : success();
}
```
> 注：`createPtr` 直接把 `lib/LowerToBLAS.cpp:100-145` 的 `createPtrWithOffsetFromMemRef` 原样搬来（同文件内私有函数不可跨 TU 复用）。

- [ ] **Step 3: CMake 登记**

在 `lib/CMakeLists.txt` 的源文件列表（`SConv.cpp / CSA.cpp / LowerToBLAS.cpp`）后加一行 `LowerToSME.cpp`。并在 `LINK_LIBS PRIVATE` 加 `MLIRLinalgDialect MLIRMemRefDialect MLIRArithDialect`（若构建报未定义符号）。

- [ ] **Step 4: 构建**

Run:
```bash
cd /home/z00889957/100Work/SConvTransform && source env.sh
cmake --build build --target sconv-opt -j8 2>&1 | tail -15
```
Expected: 编译通过，产出 `build/bin/sconv-opt`。

- [ ] **Step 5: 写测试输入（带标记的 conv）**

Create `test/lowering/sme/to_sme_in.mlir`:
```mlir
module {
  func.func @conv(%in: tensor<1x2x10x10xf32>, %wei: tensor<16x2x3x3xf32>,
                  %out: tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32> {
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<[1,1]> : tensor<2xi64>,
                                   strides = dense<[1,1]> : tensor<2xi64>,
                                   sme.direct}
      ins(%in, %wei: tensor<1x2x10x10xf32>, tensor<16x2x3x3xf32>)
      outs(%out: tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
    return %r : tensor<1x16x8x8xf32>
  }
}
```

- [ ] **Step 6: 写 transform 脚本**

Create `test/lowering/sme/to_sme_transform.mlir`:
```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %m = transform.bufferization.one_shot_bufferize %arg0
      { bufferize_function_boundaries = true } : (!transform.any_op) -> !transform.any_op
    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]}
               attributes {sme.direct} in %m
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">
    transform.foreach %convs : !transform.op<"linalg.conv_2d_nchw_fchw"> -> !transform.any_op {
    ^bb(%op: !transform.op<"linalg.conv_2d_nchw_fchw">):
      %c = transform.lower.to_sme "sconv_sme_direct", %op
        : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.any_op
      transform.yield %c : !transform.any_op
    }
    transform.yield
  }
}
```

- [ ] **Step 7: 运行验证**

Run:
```bash
cd /home/z00889957/100Work/SConvTransform && source env.sh
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:/home/z00889957/miniconda3/lib:$LD_LIBRARY_PATH"
build/bin/sconv-opt -transform=test/lowering/sme/to_sme_transform.mlir \
  test/lowering/sme/to_sme_in.mlir | grep -E "llvm.func @sconv_sme_direct|llvm.call @sconv_sme_direct" | head
```
Expected: 出现 `llvm.func @sconv_sme_direct(...)` 声明与 `llvm.call @sconv_sme_direct(...)`；原 conv 不再出现。

- [ ] **Step 8: 提交**

```bash
git add include/SConv.td lib/LowerToSME.cpp lib/CMakeLists.txt test/lowering/sme/
git commit -m "feat(sme): transform.lower.to_sme emits sconv_sme_direct call"
```

---

## Task 4: `transform.lower.sme_select`（谓词 + 代价模型）+ 全流程脚本 + 选择验证

**Files:**
- Modify: `include/SConv.td`（新增 `SMESelectOp`）
- Modify: `lib/LowerToSME.cpp`（实现 `sme_select` + `eligible` + 代价模型）
- Create: `test/lowering/sconv-sme.mlir`
- Modify: `sme/test/direct_main.c`（bench 覆盖 11 个 custom 配置，取数据）
- Test: `scripts/convbench/samples/custom_conv.csv`（作为路由测试输入）

**Interfaces:**
- Consumes: `transform.lower.to_sme`（Task 3）。
- Produces: `transform.lower.sme_select`（tensor 阶段）——对句柄内每个 conv 求 `eligible && useSme`，命中者打 `sme.direct`，返回未命中句柄。

- [ ] **Step 1: 采集 SME 全量数据**

在 `sme/test/direct_main.c` 的 bench 模式中，把 11 个 custom 配置加入 `bench_one` 调用（几何取自 `scripts/convbench/samples/custom_conv.csv`；输入/输出按框架预 pad 约定 `Ih=Oh+2,Iw=Ow+2`）。构建并运行：
```bash
cd /home/z00889957/100Work/SConvTransform/sme && source ../env.sh
make -s direct.out && numactl --cpunodebind=0 --membind=0 ./direct.out bench 38
```
记录每个配置 `Ih/Iw` 用 `Oh+2/Ow+2`（c9: 82×82，c10: 42×42，其余按 CSV 的 H*+2）。
Expected: 打印 11 行 `1thr/38thr` GFLOPS，供 Step 4 标定。

- [ ] **Step 2: td 定义 `sme_select`**

在 `include/SConv.td` 追加：
```tablegen
def SMESelectOp : Op<Transform_Dialect, "lower.sme_select",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Tag conv ops that should take the SME direct path.";
  let description = [{}];
  let arguments = (ins
    TransformHandleTypeInterface:$target,
    UnitAttr:$force,
    OptionalAttr<I64Attr>:$min_ic,
    OptionalAttr<I64Attr>:$max_oc
  );
  let results = (outs TransformHandleTypeInterface:$remaining);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
```

- [ ] **Step 3: 实现 `eligible` + 代价模型 + `apply`**

在 `lib/LowerToSME.cpp` 增加：
```cpp
namespace {
struct Geom { int64_t N,Ic,Ih,Iw,Oc,Oh,Ow; };
static std::optional<Geom> eligible(linalg::Conv2DNchwFchwOp c){
  auto d = c.getDilations(); auto s = c.getStrides();
  if (d[0]!=1||d[1]!=1||s[0]!=1||s[1]!=1) return {};
  auto in=cast<ShapedType>(c.getInput().getType());
  auto f =cast<ShapedType>(c.getFilter().getType());
  auto o =cast<ShapedType>(c.getOutput().getType());
  if (!in.hasStaticShape()||!f.hasStaticShape()||!o.hasStaticShape()) return {};
  if (f.getDimSize(2)!=3||f.getDimSize(3)!=3) return {};
  int64_t N=in.getDimSize(0),Ic=in.getDimSize(1),Ih=in.getDimSize(2),Iw=in.getDimSize(3);
  int64_t Oc=f.getDimSize(0),Oh=o.getDimSize(2),Ow=o.getDimSize(3);
  if (Ih!=Oh+2||Iw!=Ow+2) return {};            // 3x3 stride1 预 pad/valid
  return Geom{N,Ic,Ih,Iw,Oc,Oh,Ow};
}
// 代价模型（时间估计，任意单位）。GEMM 付 im2col 打包；SME 付输入按 ocb 重读。
static double fGemm(const Geom&g){
  double flops=2.0*g.N*g.Oh*g.Ow*g.Oc*g.Ic*9;
  double pack =(double)g.N*g.Oh*g.Ow*g.Ic*9;
  return flops/GEMM_PEAK + pack*4.0/BW;
}
static double fSme(const Geom&g){
  double flops=2.0*g.N*g.Oh*g.Ow*g.Oc*g.Ic*9;
  double Ocb =(double)((g.Oc+15)/16);
  double traffic=(double)g.N*g.Ic*g.Ih*g.Iw*Ocb;
  return flops/SME_PEAK + traffic*4.0/BW;
}
constexpr double GEMM_PEAK=1.62e12, SME_PEAK=2.05e12, BW=3.5e10;
} // namespace

DiagnosedSilenceableFailure SMESelectOp::apply(TransformRewriter &rewriter,
    TransformResults &results, TransformState &state) {
  SmallVector<Operation*,8> remaining;
  int64_t minIc = getMinIc()? *getMinIc() : 384;
  int64_t maxOc = getMaxOc()? *getMaxOc() : 96;
  bool force = (bool)getForce();
  for (Operation *t : state.getPayloadOps(getTarget())) {
    auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(t);
    if (!conv) { remaining.push_back(t); continue; }
    auto g = eligible(conv);
    if (!g) { remaining.push_back(t); continue; }
    bool pick = force ||
      (conv.getInput().getType().cast<ShapedType>().getDimSize(1)>=minIc &&
       conv.getFilter().getType().cast<ShapedType>().getDimSize(0)<=maxOc &&
       fSme(*g) < fGemm(*g));
    if (pick) conv->setAttr("sme.direct", rewriter.getUnitAttr());
    else      remaining.push_back(t);
  }
  results.set((*this)->getOpResult(0), remaining);
  return DiagnosedSilenceableFailure::success();
}
void SMESelectOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &e){
  onlyReadsPayload(e);            // 只打标记，不擦除 op
  consumesHandle(getTargetMutable(), e);
  producesHandle(getOperation()->getOpResults(), e);
}
LogicalResult SMESelectOp::verify(){ return success(); }
```
> `onlyReadsPayload` 可用（只加属性）；若接口要求 modifiesPayload 才可见，则改 `modifiesPayload(e)`。

- [ ] **Step 4: 标定系数（使 11 配置路由正确）**

用 Step 1 的实测数据核对：目标为 c9/c10 选 SME、其余 GEMM。若 `Iw` 尾块会令 SME 在 `Ow` 非 16 倍数时多算（`OwP/Ow` 比），可在 `fSme` 内乘 `(OwP*OhP)/(Ow*Oh)` 修正。调整 `SME_PEAK/BW/min_ic/max_oc` 直到：
```
custom_9 : SME   custom_10: SME
custom_0..8 (8 个): GEMM
```
把最终系数写入 `lib/LowerToSME.cpp` 顶部常量并加注释（记录标定来源 = 本机 38 线程实测）。

- [ ] **Step 5: 写全流程脚本**

Create `test/lowering/sconv-sme.mlir`:
```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">
    %rest = transform.lower.sme_select %convs
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.op<"linalg.conv_2d_nchw_fchw">
    %uk, %loops = transform.structured.sconv %rest
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> (!transform.op<"linalg.generic">, !transform.any_op)
    %m = transform.bufferization.one_shot_bufferize %arg0
      { bufferize_function_boundaries = true } : (!transform.any_op) -> !transform.any_op
    %funcs = transform.structured.match ops{["func.func"]} in %m
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.memref.extract_address_computations
      transform.apply_patterns.memref.expand_strided_metadata
    } : !transform.any_op
    transform.apply_cse to %m : !transform.any_op
    transform.apply_dce to %m : !transform.any_op
    %smec = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]}
              attributes {sme.direct} in %m
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">
    transform.foreach %smec : !transform.op<"linalg.conv_2d_nchw_fchw"> -> !transform.any_op {
    ^bb(%op: !transform.op<"linalg.conv_2d_nchw_fchw">):
      %c = transform.lower.to_sme "sconv_sme_direct", %op
        : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.any_op
      transform.yield %c : !transform.any_op
    }
    %uk2 = transform.structured.match ops{["linalg.generic"]} attributes {microkernel} in %m
      : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.foreach %uk2 : !transform.op<"linalg.generic"> -> !transform.any_op {
    ^bb(%op: !transform.op<"linalg.generic">):
      %c = transform.lower.to_blas "sgemm_blas_kernel", %op
        : (!transform.op<"linalg.generic">) -> !transform.any_op
      transform.yield %c : !transform.any_op
    }
    transform.yield
  }
}
```

- [ ] **Step 6: 路由验证**

Run:
```bash
cd /home/z00889957/100Work/SConvTransform && source env.sh
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:/home/z00889957/miniconda3/lib:$LD_LIBRARY_PATH"
rm -rf /tmp/sme_sel && mkdir -p /tmp/sme_sel
for f in scripts/convbench/samples/custom_conv.csv; do
  python3 scripts/convbench/gen_payload.py \
    scripts/convbench/payloads/correctness_payload.mlir "$f" /tmp/sme_sel >/dev/null
done
for m in /tmp/sme_sel/conv_custom_*.mlir; do
  o=$(build/bin/sconv-opt -transform=test/lowering/sconv-sme.mlir "$m" 2>/dev/null \
        | grep -oE "@(sconv_sme_direct|sgemm_blas_kernel)" | sort -u | tr '\n' ' ')
  echo "$(basename "$m"): $o"
done
```
Expected:
```
conv_custom_9.mlir: @sconv_sme_direct
conv_custom_10.mlir: @sconv_sme_direct
conv_custom_0..8.mlir: @sgemm_blas_kernel
```

- [ ] **Step 7: 提交**

```bash
git add include/SConv.td lib/LowerToSME.cpp test/lowering/sconv-sme.mlir sme/test/direct_main.c
git commit -m "feat(sme): sme_select predicate+cost model and full sconv-sme pipeline"
```

---

## Task 5: ConvBench 接入与验收（正确性 + 性能 + 无回归）

**Files:**
- Modify: `scripts/convbench/run_convbench.py`（构建/加载 SME 库；`--transform` 选项）
- Create: `sme/test/build_runtime.sh`（或内联进 `build_runtime_libs`）
- Test: 全 11 配置

**Interfaces:**
- Consumes: `sconv-sme.mlir`（Task 4）、`libsconv_sme_direct.so`（Task 2+）。

- [ ] **Step 1: 构建 SME 运行库**

在 `run_convbench.py` 的 `build_runtime_libs()` 中增加构建 `libsconv_sme_direct.so`：
```python
sme_lib = os.path.join(BUILD_DIR, f"libsconv_sme_direct.{LIB_EXT}")
if not os.path.exists(sme_lib):
    sme_dir = os.path.join(SCONV_ROOT, "sme")
    # 1) 编译 MLIR 计算核
    r = subprocess.run([os.path.join(sme_dir, "build.sh"),
                        os.path.join(sme_dir, "kernel/sconv_sme_direct.mlir")],
                       capture_output=True, text=True, cwd=sme_dir,
                       env={**os.environ,
                            "LLVM_BUILD_DIR": LLVM_DIR,
                            "LD_LIBRARY_PATH": f"{LLVM_DIR}/lib:/home/z00889957/miniconda3/lib:"+os.environ.get("LD_LIBRARY_PATH","")})
    if r.returncode: print(f"  WARN: sme kernel build failed: {r.stderr[:300]}")
    # 2) 编译 shim 并链接为 .so
    cmd = ["clang","-shared","-fPIC","-O2","-o",sme_lib,
           os.path.join(sme_dir,"runtime/sconv_sme_shim.c"),
           os.path.join(sme_dir,"kernel/sconv_sme_direct.o"),
           os.path.join(LLVM_DIR,"lib/clang/19/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a"),
           "-lpthread"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(f"  WARN: sme shim build failed: {r.stderr[:300]}")
return rtclock_lib, blas_lib, sme_lib
```
并把 `get_libs` 改为接收并追加 `sme_lib`。

- [ ] **Step 2: 加 `--transform` 选项**

在 `main()` 增加 `ap.add_argument("--transform", default=TF_BLAS)`，并把 `perf_test`/`correctness_test` 内硬编码的 `TF_BLAS` 换成该参数（`perf_test(..., tf)`），以便用 `test/lowering/sconv-sme.mlir` 跑 SME 流程。

- [ ] **Step 3: 全量正确性**

Run:
```bash
cd /home/z00889957/100Work/SConvTransform && source env.sh
export SCONV_ROOT=$PWD
numactl --cpunodebind=0 --membind=0 python3 scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/custom_conv.csv --num 11 \
  --transform test/lowering/sconv-sme.mlir --correctness-only 2>&1 | tail -20
```
Expected: `11 passed, 0 failed`（`np.allclose rtol=atol=1e-3`）。

- [ ] **Step 4: 全量性能**

Run:
```bash
numactl --cpunodebind=0 --membind=0 python3 scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/custom_conv.csv --num 11 \
  --transform test/lowering/sconv-sme.mlir --runs 5 --warmup 2 2>&1 | tail -25
```
Expected: c9/c10 的 SConv GFLOPS ≥ 基线（c9≥1620、c10≥1025）；其余 9 个与既有 `sconv-blas.mlir` 基线持平（±5%）。

- [ ] **Step 5: 无回归**

Run:
```bash
numactl --cpunodebind=0 --membind=0 python3 scripts/convbench/run_convbench.py \
  --csv scripts/convbench/samples/custom_conv.csv --num 11 --correctness-only 2>&1 | tail -5
```
Expected: 默认（`sconv-blas.mlir`）路径仍 `11 passed, 0 failed`。

- [ ] **Step 6: 记录验收结果并提交**

把 Step 3–5 的实测数字写入 `.superpowers/sdd/…/progress.md` 与本计划的“验收结果”小节。
```bash
git add scripts/convbench/run_convbench.py
git commit -m "feat(sme): wire sconv-sme into ConvBench (build/load/--transform)"
```

---

## Self-Review

**Spec coverage：**
- spec §4 架构 → Task 2（shim）+ Task 3/4（transform）。
- spec §5.1 kernel 改名 → Task 1；§5.2 shim → Task 2；§5.3 op → Task 3/4；§5.4 脚本 → Task 4；§5.5 构建 → Task 5。
- spec §6 选择机制 → Task 4 Step 2–4。
- spec §7 验证 → Task 2 Step 4（shim 数值）、Task 4 Step 6（路由）、Task 5（ConvBench）。
- spec §9 探针期变更（预 pad / C shim 线程 / 去 ph,pw）→ Global Constraints Ruling 1 + Task 2。
- spec §10 执行顺序 → Task 1→5 一致。

**Placeholder scan：** Task 3 Step 2 与 Task 4 Step 3 给了要点骨架并注明“复用 `LowerToBLAS.cpp` 的 `createPtrWithOffsetFromMemRef` 原样搬入”；其余步骤均含可执行代码/命令。无 TBD/TODO。

**Type consistency：** `sconv_sme_compute`（Task 1）在 Task 2 的 `sme_abi.h` 中以同一签名声明；`sconv_sme_direct`（Task 2）在 Task 3 生成同形 `llvm.func`；`sme.direct` 标记在 Task 3（消费）与 Task 4（产出）一致；`MemRef1D/2D`+`mk1d/mk2d` 定义一次于 `sme/runtime/sme_abi.h`。

**Review Focus 覆盖：** (1) 预 pad 对齐 → Task 2 Step 4；(2) 尾块 → Task 2 Step 4（Oc=17/Ow=13）与 Task 3 之外的 ConvBench 形状；(3) `sme.direct` 只作用卷积 → Task 4 Step 5–6（其余走 `to_blas`）；(4) ABI 一致 → Task 2 Step 7；(5) 路由 → Task 4 Step 6 + Task 5 Step 4。
