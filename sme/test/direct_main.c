#include "harness.h"
#include <stdio.h>
#include <stdlib.h>

/* Block under test: one ocb (16 oc) x one col0..col0+63 block, stride1, no pad.
   in:    [Ic][Ih][Iw]   (rank-3)
   wpack: [(Ic*9)][16]   (rank-1, index (ic*9+kh*3+kw)*16 + oc)
   out:   [16][64]       (rank-2)

   MLIR passes memref args as flattened descriptor fields:
     rank-3 -> 9 scalars, rank-1 -> 5, rank-2 -> 7. */
extern void sme_block1(
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);

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
  for (int i = 0; i < IC * IH * IW; i++) in[i] = 0.5f * (float)((i % 37) - 18);
  for (int i = 0; i < IC * 9 * 16; i++) wpack[i] = 0.25f * (float)((i % 11) - 5);
  for (int i = 0; i < OC * OCOLS; i++) out[i] = -1.0f;

  /* naive reference: output row 0, columns 0..63, no pad -> input row kh */
  for (int oc = 0; oc < OC; oc++)
    for (int c = 0; c < OCOLS; c++) {
      float s = 0;
      for (int ic = 0; ic < IC; ic++)
        for (int kh = 0; kh < 3; kh++)
          for (int kw = 0; kw < 3; kw++) {
            int ih = kh, iw = c + kw;
            float x = in[ic * IH * IW + ih * IW + iw];
            float w = wpack[(ic * 9 + kh * 3 + kw) * 16 + oc];
            s += x * w;
          }
      ref[oc * OCOLS + c] = s;
    }

  MemRef3D mi = mk3d(in, IC, IH, IW);
  MemRef1D mw = mk1d(wpack, IC * 9 * 16);
  MemRef2D mo = mk2d(out, OC, OCOLS);
  sme_block1(mi.allocated, mi.aligned, mi.offset, mi.size0, mi.size1, mi.size2,
             mi.stride0, mi.stride1, mi.stride2,
             mw.allocated, mw.aligned, mw.offset, mw.size, mw.stride,
             mo.allocated, mo.aligned, mo.offset, mo.size0, mo.size1,
             mo.stride0, mo.stride1);

  int bad = 0;
  for (int i = 0; i < OC * OCOLS; i++)
    if (!close_enough(out[i], ref[i])) {
      if (!bad) printf("first mismatch idx %d got %g want %g\n", i, out[i], ref[i]);
      bad++;
    }
  printf(bad ? "BLOCK FAIL (%d/%d)\n" : "BLOCK PASS\n", bad, OC * OCOLS);
  return bad ? 1 : 0;
}
