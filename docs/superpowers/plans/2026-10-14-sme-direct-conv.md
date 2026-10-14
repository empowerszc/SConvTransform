# SME Direct Convolution Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate (Phase 1 probe) a standalone MLIR ArmSME direct-conv kernel that is numerically correct and competitive with the existing kblas GEMM path on custom_9/custom_10; then (Phase 2, gated) integrate it into the SConv transform pipeline.

**Architecture:** A direct convolution kernel is authored directly with `arm_sme` dialect ops (outer product = `fmopa`, ZA tile load/store), compiled AOT via MLIR→ArmSME→LLVM, and (Phase 2) called from a new `transform.lower.to_sme` op that mirrors `to_blas`.

**Tech Stack:** MLIR 19.1.7 (`mlir-opt`, `mlir-translate`), ArmSME/ArmSVE dialects, clang 19.1.7 (`-march=armv9-a+sme`), `libclang_rt.builtins.a`, C test harness.

**Spec:** `docs/superpowers/specs/2026-10-14-sme-direct-conv-design.md`

## Global Constraints

- Toolchain: `LLVM_BUILD_DIR=/home/z00889957/000App/clang+llvm-19.1.7-aarch64-linux-gnu`; source `env.sh` first.
- `mlir-opt`/`mlir-translate` need `LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:/home/z00889957/miniconda3/lib:$LD_LIBRARY_PATH` (else `GLIBCXX_3.4.30 not found`).
- Compile flags for SME: `-march=armv9-a+sme`.
- Linking SME executables requires `libclang_rt.builtins.a` at `$LLVM_BUILD_DIR/lib/clang/19/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a`.
- Hardware: SVL = 512 bit = 64 bytes = 16 f32 → f32 ZA tile = 16×16 (i.e. `vector<[4]x[4]xf32>`).
- Kernel must use `arm_sme` dialect ops directly (NOT C intrinsics — the C frontend cannot compile SME state attributes on this toolchain).
- Streaming/ZA state is injected only via `--enable-arm-streaming` / `--test-lower-to-arm-sme`; never written in the source `.mlir`.
- Do not modify `lib/SConv.cpp` / `lib/LowerToBLAS.cpp` behavior in Phase 1.
- Config reference: custom_9 = N4,Ic384,Ih80,Iw80, Oc96,F3x3,Oh80,Ow80,stride1,pad1; custom_10 = N4,Ic768,Ih40,Iw40, Oc96,F3x3,Oh40,Ow40,stride1,pad1.
- Correctness tolerance: max abs error ≤ 1e-3 relative to naive fp32 accumulation.

## Review Focus

- **Input with SAME padding** (pad=1): border output elements must use zero-valued padded input, not garbage or shifted taps — a reasonable person expects identical results to PyTorch `conv2d` with `padding=1`.
- **Non-multiple output columns** (`Oh*Ow % 64 != 0`): the last column block must not write past the end of `out` — expected behavior is correct results with no out-of-bounds writes.
- **Non-multiple `Oc`** (`Oc % 16 != 0`): the last oc block must not read/write past `wpack`/`out`.
- **Runtime `Ic`, `Oh`, `Ow`, `N` variation**: kernel must produce correct results for shapes other than the two target configs (parameterization, not hardcoding).
- **Weight-packing layout error**: a wrong `wpack` index must be caught by correctness tests across multiple `ic/kh/kw`, not just the first tap.

---

## File Structure

Phase 1 (this plan):

- Create `sme/kernel/sme_smoke.mlir` — minimal ArmSME smoke kernel (pipeline validation).
- Create `sme/kernel/sconv_sme_direct.mlir` — the direct-conv kernel.
- Create `sme/build.sh` — MLIR→object build (shared by both kernels).
- Create `sme/test/harness.h` — memref descriptor structs + naive reference (shared).
- Create `sme/test/smoke_main.c` — driver for the smoke kernel.
- Create `sme/test/direct_main.c` — driver for the direct-conv kernel (naive compare + timing).
- Create `sme/Makefile` — build/run smoke + direct tests.

Phase 2 (separate plan, after go/no-go gate): `include/SConv.td`, `lib/LowerToSME.cpp`, `lib/CMakeLists.txt`, `test/lowering/sconv-sme.mlir`, kernel build wiring — outlined at the end (not detailed here; the spec gates it on Phase 1 results).

---

## Task 1: Smoke-test the full MLIR→ArmSME→object pipeline

**Files:**
- Create: `sme/kernel/sme_smoke.mlir`
- Create: `sme/build.sh`
- Create: `sme/test/harness.h`
- Create: `sme/test/smoke_main.c`
- Create: `sme/Makefile`

**Interfaces:**
- Consumes: nothing.
- Produces: `build.sh <file.mlir>` → `<file>.ll` and `<file>.o`; memref descriptor structs `MemRef1D`/`MemRef2D` in `harness.h`.

- [ ] **Step 1: Write the smoke kernel**

Create `sme/kernel/sme_smoke.mlir`:
```mlir
// out[i][j] = a[i] * b[j]  (16x16 outer product on the ZA tile)
func.func @sme_smoke(%a: memref<?xf32>, %b: memref<?xf32>, %out: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %va = vector.load %a[%c0] : memref<?xf32>, vector<[4]xf32>
  %vb = vector.load %b[%c0] : memref<?xf32>, vector<[4]xf32>
  %tile = arm_sme.outerproduct %va, %vb : vector<[4]xf32>, vector<[4]xf32>
  arm_sme.tile_store %tile, %out[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}
```

- [ ] **Step 2: Write the build script**

Create `sme/build.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
# Usage: build.sh <file.mlir>   (source env.sh first, or LLVM_BUILD_DIR set)
LLVM=${LLVM_BUILD_DIR:?set LLVM_BUILD_DIR or source env.sh}
export PATH="$LLVM/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM/lib:/home/z00889957/miniconda3/lib:${LD_LIBRARY_PATH:-}"
src="$1"; base="${src%.mlir}"
mlir-opt "$src" \
  --test-lower-to-arm-sme \
  --convert-vector-to-llvm --convert-to-llvm --reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir > "${base}.ll"
clang -march=armv9-a+sme -O2 -c "${base}.ll" -o "${base}.o"
echo "built ${base}.o"
```

- [ ] **Step 3: Write the shared harness header**

Create `sme/test/harness.h`:
```c
#ifndef SME_TEST_HARNESS_H
#define SME_TEST_HARNESS_H
#include <stdint.h>
#include <math.h>

/* Canonical memref descriptor ABI (rank-1 / rank-2, dynamic). */
typedef struct { float *allocated, *aligned; int64_t offset, size, stride; } MemRef1D;
typedef struct { float *allocated, *aligned; int64_t offset, size0, size1, stride0, stride1; } MemRef2D;

static inline MemRef1D mk1d(float *p, int64_t n) {
  MemRef1D m = { p, p, 0, n, 1 }; return m;
}
static inline MemRef2D mk2d(float *p, int64_t r, int64_t c) {
  MemRef2D m = { p, p, 0, r, c, c, 1 }; return m;
}
static inline int close_enough(float got, float want) {
  float d = fabsf(got - want);
  return d <= 1e-3f * (1.0f + fabsf(want));
}
#endif
```

- [ ] **Step 4: Write the smoke driver**

Create `sme/test/smoke_main.c`:
```c
#include "harness.h"
#include <stdio.h>
extern void sme_smoke(MemRef1D a, MemRef1D b, MemRef2D out);

int main(void) {
  float a[16], b[16], out[16 * 16];
  for (int i = 0; i < 16; i++) { a[i] = (float)(i + 1); b[i] = (float)(i + 1); out[i] = 0; }
  for (int i = 0; i < 16 * 16; i++) out[i] = -1.0f;
  sme_smoke(mk1d(a, 16), mk1d(b, 16), mk2d(out, 16, 16));
  int bad = 0;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      if (!close_enough(out[i * 16 + j], a[i] * b[j])) { bad++; if (bad == 1) printf("first mismatch [%d][%d]=%g want %g\n", i, j, out[i*16+j], a[i]*b[j]); }
  printf(bad ? "SMOKE FAIL (%d)\n" : "SMOKE PASS\n", bad);
  return bad ? 1 : 0;
}
```

- [ ] **Step 5: Write the Makefile**

Create `sme/Makefile`:
```make
LLVM ?= $(LLVM_BUILD_DIR)
BUILTINS := $(LLVM)/lib/clang/19/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a
CFLAGS := -O2
SME := -march=armv9-a+sme

all: smoke

smoke: kernel/sme_smoke.o test/smoke_main.o
	clang $(SME) $^ $(BUILTINS) -o smoke.out
	@./smoke.out

kernel/%.o: kernel/%.mlir build.sh
	./build.sh $<

test/%.o: test/%.c test/harness.h
	clang $(CFLAGS) -c $< -o $@

clean:
	rm -f kernel/*.ll kernel/*.o test/*.o smoke.out
.PHONY: all smoke clean
```

- [ ] **Step 6: Build and run**

Run:
```bash
cd sme && source ../env.sh && make smoke
```
Expected: `SMOKE PASS`, exit 0.

- [ ] **Step 7: Verify SME instructions and streaming attrs are emitted**

Run:
```bash
grep -E "aarch64_pstate_sm_body|aarch64_new_za|sme.mopa" sme/kernel/sme_smoke.ll
llvm-objdump -d sme/kernel/sme_smoke.o | grep -E "fmopa|smstart"
```
Expected: both show `mopa`/`fmopa` and the two attrs; the object has `smstart`.

- [ ] **Step 8: Commit**

```bash
git add sme/
git commit -m "feat(sme): add standing ArmSME smoke-test pipeline (mlir->object->run)"
```

---

## Task 2: Direct-conv core — single oc-block × 4 ZA tiles over taps (stride1, no pad)

**Files:**
- Create: `sme/kernel/sconv_sme_direct.mlir`
- Create: `sme/test/direct_main.c`
- Modify: `sme/Makefile` (add `direct` target)

**Interfaces:**
- Consumes: `build.sh`, `harness.h` (Task 1).
- Produces: kernel `sconv_sme_direct` with a **fixed small test shape** and signature:
  `void sconv_sme_direct(MemRef2D in[n][ic] rows, MemRef1D wpack, MemRef2D out)` — refined in Task 4 to the full runtime signature. For Task 2 the driver calls a helper `sme_block1`.

For Task 2, first implement and test the **inner block only**: given a pack of input rows for one image/channel (`in_rows[?]`), a weight pack, produce 16 oc × 64 columns.

- [ ] **Step 1: Write the failing correctness driver**

Create `sme/test/direct_main.c` (Task 2 version — block test):
```c
#include "harness.h"
#include <stdio.h>
#include <stdlib.h>

/* Block under test: one (ocb) x one (col0..col0+63) block, stride1.
   Shape (test): Ic=2, Fh=Fw=3, one image, one output row of 64 cols.
   in: memref<?x?xf32> [Ic][Ih*Iw] (rows contiguous)
   wpack: memref<?xf32> [(Ic*9)][16]
   out: memref<?x?xf32> [16][64]
*/
extern void sme_block1(MemRef2D in, MemRef1D wpack, MemRef2D out);

#define IC 2
#define IH 4
#define IW 80           /* >= 64 + 2 */
#define OC 16
#define OCOLS 64

int main(void) {
  static float in[IC * IH * IW];
  static float wpack[IC * 9 * 16];
  static float out[OC * OCOLS];
  static float ref[OC * OCOLS];
  for (int i = 0; i < IC*IH*IW; i++) in[i] = 0.5f * (float)((i % 37) - 18);
  for (int i = 0; i < IC*9*16; i++) wpack[i] = 0.25f * (float)((i % 11) - 5);
  /* naive reference for output row 0..0, columns 0..63, SAME-ish (no pad here) */
  for (int oc = 0; oc < OC; oc++)
    for (int c = 0; c < OCOLS; c++) {
      float s = 0;
      for (int ic = 0; ic < IC; ic++)
        for (int kh = 0; kh < 3; kh++)
          for (int kw = 0; kw < 3; kw++) {
            int ih = 0 + kh, iw = c + kw;          /* no pad, out row 0 */
            float x = in[ic * IH * IW + ih * IW + iw];
            float w = wpack[(ic * 9 + kh * 3 + kw) * 16 + oc];
            s += x * w;
          }
      ref[oc * OCOLS + c] = s;
    }
  sme_block1(mk2d(in, IC, IH * IW), mk1d(wpack, IC * 9 * 16), mk2d(out, OC, OCOLS));
  int bad = 0;
  for (int i = 0; i < OC * OCOLS; i++)
    if (!close_enough(out[i], ref[i])) { if (!bad) printf("first mismatch idx %d got %g want %g\n", i, out[i], ref[i]); bad++; }
  printf(bad ? "BLOCK FAIL (%d/%d)\n" : "BLOCK PASS\n", bad, OC * OCOLS);
  return bad ? 1 : 0;
}
```

- [ ] **Step 2: Run to verify it fails to link**

Run: `cd sme && source ../env.sh && make direct`
Expected: link error `undefined reference to sme_block1` (kernel not written yet).

- [ ] **Step 3: Write the block kernel**

Create `sme/kernel/sconv_sme_direct.mlir` (Task 2 version). It reads, for each `ic`, the 3 input row taps, broadcasts the weight vector, and accumulates into 4 ZA tiles (64 columns).

```mlir
// One output block: 16 oc (rows) x 64 output columns (4 ZA tiles), stride1, no pad.
//   in:     memref<?x?xf32>  [Ic][Ih*Iw]
//   wpack:  memref<?xf32>    [(Ic*9)][16]   (index: (ic*9+kh*3+kw)*16 + oc)
//   out:    memref<?x?xf32>  [16][64]
func.func @sme_block1(%in: memref<?x?xf32>, %wpack: memref<?xf32>, %out: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c9 = arith.constant 9 : index
  %c64 = arith.constant 64 : index

  %ta = arm_sme.zero : vector<[4]x[4]xf32>
  %tb = arm_sme.zero : vector<[4]x[4]xf32>
  %tc = arm_sme.zero : vector<[4]x[4]xf32>
  %td = arm_sme.zero : vector<[4]x[4]xf32>

  // iterate ic (runtime via scf.for would be added in Task 4; here Ic fixed by loop bound arg)
  %icn = memref.dim %in, %c0 : memref<?x?xf32>
  %acc = scf.for %ic = %c0 to %icn step %c1
      iter_args(%a = %ta, %b = %tb, %c = %tc, %d = %td) -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>) {
    %acc2 = scf.for %kh = %c0 to %c3 step %c1
        iter_args(%a2 = %a, %b2 = %b, %c2i = %c, %d2 = %d) -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>) {
      // input row pointer base = ic*Ih*Iw + kh*Iw  (no pad, output row 0)
      %iw = memref.dim %in, %c1 : memref<?x?xf32>
      %row = arith.muli %kh, %iw : index
      %icbase = arith.muli %ic, %iw : index
      %inbase = arith.addi %icbase, %row : index
      // 3 weight vectors for kw=0,1,2
      %widx0 = arith.muli %ic, %c9 : index
      %w0i = arith.addi %widx0, %kh : index
      %w0 = arith.muli %w0i, %c16 : index
      %w1 = arith.addi %w0, %c16 : index
      %w2 = arith.addi %w1, %c16 : index
      %wv0 = vector.load %wpack[%w0] : memref<?xf32>, vector<[4]xf32>
      %wv1 = vector.load %wpack[%w1] : memref<?xf32>, vector<[4]xf32>
      %wv2 = vector.load %wpack[%w2] : memref<?xf32>, vector<[4]xf32>
      // 4 input column windows (kw shifts) x 4 tiles
      %o0 = arith.addi %inbase, %c0 : index
      %o1 = arith.addi %inbase, %c1 : index
      %o2 = arith.addi %inbase, %c2 : index
      %c16i = arith.muli %c16, %c1 : index
      %c32 = arith.muli %c16, %c2 : index
      %c48 = arith.muli %c16, %c3 : index
      // tile t covers output cols [t*16, t*16+16) ; input col = outcol + kw
      %in0_0 = vector.load %in[%ic, %o0] : memref<?x?xf32>, vector<[4]xf32>
      %in0_1 = vector.load %in[%ic, %o1] : memref<?x?xf32>, vector<[4]xf32>
      %in0_2 = vector.load %in[%ic, %o2] : memref<?x?xf32>, vector<[4]xf32>
      // (tile loads for t=1,2,3 add 16/32/48 to the column index; see Step 3 note)
      %a3 = arm_sme.outerproduct %wv0, %in0_0 acc(%a2) : vector<[4]xf32>, vector<[4]xf32>
      %b3 = arm_sme.outerproduct %wv1, %in0_1 acc(%b2) : vector<[4]xf32>, vector<[4]xf32>
      %c3v = arm_sme.outerproduct %wv2, %in0_2 acc(%c2i) : vector<[4]xf32>, vector<[4]xf32>
      scf.yield %a3, %b3, %c3v, %d2 : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
    }
    scf.yield %acc2#0, %acc2#1, %acc2#2, %acc2#3 : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
  }
  %c0b = arith.constant 0 : index
  arm_sme.tile_store %acc#0, %out[%c0b, %c0b] : memref<?x?xf32>, vector<[4]x[4]xf32>
  arm_sme.tile_store %acc#1, %out[%c0b, %c16] : memref<?x?xf32>, vector<[4]x[4]xf32>
  arm_sme.tile_store %acc#2, %out[%c0b, %c32] : memref<?x?xf32>, vector<[4]x[4]xf32>
  arm_sme.tile_store %acc#3, %out[%c0b, %c48] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}
```

> **Note for the implementer:** the KW-shift mapping is subtle — tile `t` (output columns `t*16..t*16+15`) with tap `kw` reads input columns `t*16+kw .. t*16+15+kw`, i.e. tile load at column offset `t*16 + kw`. The sketch above only shows tile 0; you must emit all 4 tiles × 3 kw = 12 input vector loads and 12 `arm_sme.outerproduct` accumulations (one per (tile,kw)), accumulated into the corresponding tile `%a/%b/%c/%d`. Correctness against the naive reference in Step 1 is the check.

- [ ] **Step 4: Run and iterate until BLOCK PASS**

Run: `cd sme && source ../env.sh && make direct`
Expected: `BLOCK PASS`, exit 0. If it fails, fix the kw-tile column offsets (and the `wpack` index order) until the naive compare passes.

- [ ] **Step 5: Commit**

```bash
git add sme/kernel/sconv_sme_direct.mlir sme/test/direct_main.c sme/Makefile
git commit -m "feat(sme): direct-conv core block (16oc x 64col over 3x3 taps)"
```

---

## Task 3: Add SAME-padding (pad=1) via operand masks

**Files:**
- Modify: `sme/kernel/sconv_sme_direct.mlir`
- Modify: `sme/test/direct_main.c` (add a padded reference + a border-row case)

**Interfaces:**
- Consumes: `sme_block1` from Task 2.
- Produces: `sme_block1` handles `in_row` out of `[0,Ih)` and input columns out of `[0,Iw)` as zero.

- [ ] **Step 1: Extend the driver with a padded reference and a border output row**

In `direct_main.c`, change the reference so the kernel computes output row `oh=0` with `pad=1` (input row index `= oh + kh - 1`), and add a case for a border row where `oh + kh - 1 < 0` (expect those taps contribute 0). Assert both against naive `conv2d` semantics with `padding=1`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd sme && source ../env.sh && make direct`
Expected: `BLOCK FAIL` on the border case (rows are currently read out of bounds / not masked).

- [ ] **Step 3: Implement masking**

In the kernel, compute `in_row = oh + kh - pad`. When `in_row` is out of `[0, Ih)`, skip the contribution (e.g. guard with `scf.if`, or use masked loads with a zero `padding` value via `arm_sme.tile_load`/`vector.mask`). For input columns outside `[0, Iw)`, use a masked `vector.load` with pad value `0.0`. Use `arm_sme.outerproduct` `masks(%lm, %rm)` (`vector<[4]xi1>` operands) to drop out-of-range lanes.

- [ ] **Step 4: Run until PASS**

Run: `cd sme && source ../env.sh && make direct`
Expected: `BLOCK PASS` for both interior and border rows.

- [ ] **Step 5: Commit**

```bash
git add sme/kernel/sconv_sme_direct.mlir sme/test/direct_main.c
git commit -m "feat(sme): SAME-padding (pad=1) boundary handling via masks"
```

---

## Task 4: Full loop nest + runtime shapes + tails + runtime SVL

**Files:**
- Modify: `sme/kernel/sconv_sme_direct.mlir` (finalize `sconv_sme_direct` signature)
- Modify: `sme/test/direct_main.c` (call `sconv_sme_direct`, test multiple shapes)
- Modify: `sme/Makefile`

**Interfaces:**
- Consumes: block logic from Tasks 2–3.
- Produces: final kernel:
  ```c
  void sconv_sme_direct(int64_t N, int64_t Ic, int64_t Ih, int64_t Iw,
                        int64_t Oc, int64_t Oh, int64_t Ow, int64_t ph, int64_t pw,
                        const float* in,   // NCHW  -> MemRef1D over N*Ic*Ih*Iw
                        const float* wpack,// MemRef1D
                        float* out);       // NCHW  -> MemRef1D over N*Oc*Oh*Ow
  ```
  (Implemented in MLIR as a func taking scalars + `memref<?xf32>` buffers and indexing manually; see Step 3.)

- [ ] **Step 1: Write failing tests for multiple shapes**

In `direct_main.c`, loop over shapes: `{N1,Ic2,Ih8,Iw8,Oc16,3x3}`, `{N1,Ic3,Ih10,Iw10,Oc17,3x3}` (non-multiple Oc), `{N2,Ic5,Ih7,Iw13,Oc32,3x3}` (non-multiple `Oh*Ow` vs 64). Compare against a naive NCHW+OcHw+pad conv. Expected: all pass; currently fails (kernel still block-only).

- [ ] **Step 2: Run to verify failure**

Run: `cd sme && source ../env.sh && make direct`
Expected: compile/link error or numeric FAIL.

- [ ] **Step 3: Implement the full kernel**

Rewrite `sconv_sme_direct.mlir` as the full kernel:
- Compute `R = arm_sme.streaming_vl <word>` (16 on this machine), `COLS = 4*R`.
- Nest `scf.for` over `n`, `ocb` (`0..ceil(Oc/R)`), `col0` (`0..Oh*Ow step COLS`) with the Task 2/3 block body inside.
- Tail `ocb`: mask stores to valid rows; fill `wpack` tail with 0.
- Tail columns: mask loads/stores to `min(COLS, Oh*Ow-col0)`.
- Store via `arm_sme.tile_store %tile, %out[%row0, %col]` using computed offsets into the NCHW output buffer (flatten `oc` and spatial into the last index, or use a rank-2 view `[Oc][Oh*Ow]`).

- [ ] **Step 4: Run all shape tests until PASS**

Run: `cd sme && source ../env.sh && make direct`
Expected: all shapes `PASS`.

- [ ] **Step 5: Commit**

```bash
git add sme/kernel/sconv_sme_direct.mlir sme/test/direct_main.c sme/Makefile
git commit -m "feat(sme): full direct-conv kernel with runtime shapes, tails, runtime SVL"
```

---

## Task 5: Benchmark custom_9/custom_10 vs GEMM/PyTorch — GO/NO-GO

**Files:**
- Modify: `sme/test/direct_main.c` (add timing mode)
- Create: `sme/test/bench.sh`

**Interfaces:**
- Consumes: full `sconv_sme_direct` (Task 4).
- Produces: GFLOPS numbers for custom_9 and custom_10 → the spec's acceptance gate.

- [ ] **Step 1: Add a timing mode to the driver**

In `direct_main.c`, when `argv[1] == "bench"`, allocate full custom_9/custom_10 buffers (NCHW + wpack), run the kernel `R` times, and print:
```
custom_9: <ms> ms  <gflops> GFLOPS
custom_10: <ms> ms  <gflops> GFLOPS
```
where `gflops = 2.0*N*Oh*Ow*Oc*Ic*9 / (ms*1e6)`.

- [ ] **Step 2: Add the bench script**

Create `sme/test/bench.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../env.sh"
numactl --cpunodebind=0 --membind=0 \
  sme/direct.out bench
```

- [ ] **Step 3: Build and run the benchmark**

Run: `cd sme && source ../env.sh && make direct bench`
Expected: prints GFLOPS for both configs.

- [ ] **Step 4: Compare against baselines and record the gate result**

Run the existing GEMM path and PyTorch for custom_9/custom_10 (38 threads, node0) and record:
```
custom_9 : ours_sme=<X>  gemm=<1620>  torch=<2131>   PASS/FAIL (X >= 1620 ?)
custom_10: ours_sme=<X>  gemm=<957>   torch=<1116>   PASS/FAIL (X >= 957 ?)
```
Expected (gate): both PASS → proceed to Phase 2. If either FAILs, STOP and report; do not start Phase 2.

- [ ] **Step 5: Commit**

```bash
git add sme/test/direct_main.c sme/test/bench.sh
git commit -m "feat(sme): benchmark harness for custom_9/custom_10 (go/no-go)"
```

---

## Phase 2 — Integration outline (separate plan, only if Task 5 gates PASS)

Detailed plan deferred per the spec's go/no-go gate. Scope when opened:

1. `include/SConv.td`: add `LowerToSMEOp` (`transform.lower.to_sme`, `StrAttr:$name` + handle), mirroring `LowerToBlasOp`.
2. `lib/LowerToSME.cpp`: for each matched `linalg.conv_2d_nchw_fchw` microkernel: run `eligible(conv)` (dilation=1, stride=1, Fh=Fw=3, static shapes) and `useSme(conv)` (cost model `f_gemm` vs `f_sme`, coefficients calibrated from the 11 configs); if selected, emit the `wpack` packing loop + `LLVM::CallOp @sconv_sme_direct(...)` (raw NCHW in/out), erase the conv. Expose `force`/threshold options.
3. `sme/build.sh` + CMake: compile `sconv_sme_direct.mlir` to an object linked into the runtime.
4. `test/lowering/sconv-sme.mlir`: pipeline `sconv_sme` then standard `sconv`+`to_blas`; assert custom_9/10 select SME and the other 9 select GEMM.
5. `scripts/convbench`: wire the runtime symbol and re-run the 11-config head-to-head, asserting no regression.

---

## Self-Review

**Spec coverage:**
- Spec §5 kernel algorithm → Tasks 2–4.
- Spec §5.5 boundary (SAME pad + tails) → Task 3 (pad) + Task 4 (tails).
- Spec §6 MLIR expression (arm_sme direct) → Tasks 2–4.
- Spec §7 build flow → Task 1 (build.sh), Task 5 (link/builtins).
- Spec §8 runtime interface + general selection → Phase 2 outline (gated).
- Spec §9 verification/acceptance → Task 5 gate; correctness tests in Tasks 2–4.
- Spec §11 execution order (probe first) → Phases 1/2 split with the Task 5 gate.
- Spec §2/§3 constraints (toolchain, SVL, arm_sme-only) → Global Constraints.

**Placeholder scan:** The only intentionally-iterative step is Task 2 Step 4 ("fix offsets until PASS") — this is a verification loop against a concrete naive reference, not a placeholder. All other steps carry concrete commands/code.

**Type consistency:** `MemRef1D`/`MemRef2D`/`mk1d`/`mk2d`/`close_enough` defined once in `harness.h` (Task 1) and reused; kernel function names `sme_smoke`, `sme_block1`, `sconv_sme_direct` consistent across tasks.

**Review Focus coverage:** padding → Task 3; non-multiple columns/Oc → Task 4 Steps 1–4; runtime shape variation → Task 4; wpack layout → Task 2 naive reference across all ic/kh/kw.
