#include "harness.h"
#include <stdio.h>
#include <stdlib.h>

/* Block under test: one ocb (16 oc) x one output row, columns 0..63, stride1,
   SAME padding (ph=pw=1). The driver PRE-PADS the input into [Ic][Ih+2][Iw+2]
   (spec §5.5 fallback), so the kernel indexes the padded buffer directly with
   no out-of-range access: padded row = oh+kh, padded col = col+kw.
   in:    [Ic][Ih+2][Iw+2]  (rank-3, pre-padded)
   wpack: [(Ic*9)][16]      (rank-1)
   out:   [16][64]          (rank-2) */
extern void sme_block1(
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);

#define IC 2
#define IH 4
#define IW 80
#define OC 16
#define OCOLS 64
#define PH 1
#define PW 1

int main(void) {
  static float in[IC * IH * IW];
  static float pin[IC * (IH + 2) * (IW + 2)];
  static float wpack[IC * 9 * 16];
  static float out[OC * OCOLS];
  static float ref[OC * OCOLS];
  const int PHIW = IW + 2;
  const int PHIH = IH + 2;

  for (int i = 0; i < IC * IH * IW; i++) in[i] = 0.5f * (float)((i % 37) - 18);
  for (int i = 0; i < IC * 9 * 16; i++) wpack[i] = 0.25f * (float)((i % 11) - 5);
  for (int i = 0; i < OC * OCOLS; i++) out[i] = -1.0f;

  /* zero-pad into pin[ic][r][c] = in at (r-PH, c-PW) */
  for (int i = 0; i < IC * PHIH * PHIW; i++) pin[i] = 0.0f;
  for (int ic = 0; ic < IC; ic++)
    for (int r = 0; r < IH; r++)
      for (int c = 0; c < IW; c++)
        pin[ic * PHIH * PHIW + (r + PH) * PHIW + (c + PW)] = in[ic * IH * IW + r * IW + c];

  /* naive reference for output row 0 with SAME pad: padded row = 0+kh */
  for (int oc = 0; oc < OC; oc++)
    for (int c = 0; c < OCOLS; c++) {
      float s = 0;
      for (int ic = 0; ic < IC; ic++)
        for (int kh = 0; kh < 3; kh++)
          for (int kw = 0; kw < 3; kw++) {
            float x = pin[ic * PHIH * PHIW + kh * PHIW + (c + kw)];
            float w = wpack[(ic * 9 + kh * 3 + kw) * 16 + oc];
            s += x * w;
          }
      ref[oc * OCOLS + c] = s;
    }

  MemRef3D mp = mk3d(pin, IC, PHIH, PHIW);
  MemRef1D mw = mk1d(wpack, IC * 9 * 16);
  MemRef2D mo = mk2d(out, OC, OCOLS);
  sme_block1(mp.allocated, mp.aligned, mp.offset, mp.size0, mp.size1, mp.size2,
             mp.stride0, mp.stride1, mp.stride2,
             mw.allocated, mw.aligned, mw.offset, mw.size, mw.stride,
             mo.allocated, mo.aligned, mo.offset, mo.size0, mo.size1,
             mo.stride0, mo.stride1);

  int bad = 0;
  for (int i = 0; i < OC * OCOLS; i++)
    if (!close_enough(out[i], ref[i])) {
      if (!bad) printf("first mismatch idx %d got %g want %g\n", i, out[i], ref[i]);
      bad++;
    }
  printf(bad ? "PAD BLOCK FAIL (%d/%d)\n" : "PAD BLOCK PASS\n", bad, OC * OCOLS);
  return bad ? 1 : 0;
}
