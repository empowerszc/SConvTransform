#include "harness.h"
#include <stdio.h>

/* MLIR lowers each memref argument to flattened descriptor fields:
   rank-1: (allocated, aligned, offset, size, stride)
   rank-2: (allocated, aligned, offset, size0, size1, stride0, stride1) */
extern void sme_smoke(float *, float *, int64_t, int64_t, int64_t,
                      float *, float *, int64_t, int64_t, int64_t,
                      float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);

int main(void) {
  float a[16], b[16], out[16 * 16];
  for (int i = 0; i < 16; i++) { a[i] = (float)(i + 1); b[i] = (float)(i + 1); }
  for (int i = 0; i < 16 * 16; i++) out[i] = -1.0f;
  MemRef1D ma = mk1d(a, 16), mb = mk1d(b, 16);
  MemRef2D mo = mk2d(out, 16, 16);
  sme_smoke(ma.allocated, ma.aligned, ma.offset, ma.size, ma.stride,
            mb.allocated, mb.aligned, mb.offset, mb.size, mb.stride,
            mo.allocated, mo.aligned, mo.offset, mo.size0, mo.size1,
            mo.stride0, mo.stride1);
  int bad = 0;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 16; j++)
      if (!close_enough(out[i * 16 + j], a[i] * b[j])) {
        if (bad == 0) printf("first mismatch [%d][%d]=%g want %g\n", i, j, out[i*16+j], a[i]*b[j]);
        bad++;
      }
  printf(bad ? "SMOKE FAIL (%d)\n" : "SMOKE PASS\n", bad);
  return bad ? 1 : 0;
}
