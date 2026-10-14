#include "harness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Full direct conv: stride1, 3x3, SAME pad, pre-padded I/O.
   The kernel works on pre-padded buffers (driver pads here); R is the streaming
   vector length. On this machine SVL=512 -> R=16 (confirmed). */
extern void sconv_sme_direct(
    int64_t N, int64_t Ic, int64_t IwP, int64_t OcP, int64_t OhP, int64_t OwP,
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);

#define R 16

static int run_shape(int N, int Ic, int Ih, int Iw, int Oc, int Oh, int Ow) {
  int OhP = (Oh + 3) / 4 * 4;
  int OwP = (Ow + R - 1) / R * R;
  int OcP = (Oc + R - 1) / R * R;
  int IhP = OhP + 2;
  int IwP = OwP + 2;

  float *in   = malloc(sizeof(float) * (size_t)N * Ic * Ih * Iw);
  float *win  = malloc(sizeof(float) * (size_t)Oc * Ic * 9);
  float *pin  = calloc((size_t)N * Ic * IhP * IwP, sizeof(float));
  float *wpack= calloc((size_t)Ic * 9 * OcP, sizeof(float));
  float *out  = calloc((size_t)N * OcP * OhP * OwP, sizeof(float));
  float *ref  = malloc(sizeof(float) * (size_t)N * Oc * Oh * Ow);
  if (!in || !win || !pin || !wpack || !out || !ref) { printf("OOM\n"); exit(2); }

  for (size_t i = 0; i < (size_t)N * Ic * Ih * Iw; i++) in[i] = 0.1f * (float)((i % 29) - 13);
  for (size_t i = 0; i < (size_t)Oc * Ic * 9; i++) win[i] = 0.2f * (float)((i % 17) - 8);

  /* pad input: P[n][ic][r][c] = in[n][ic][r-1][c-1] */
  for (int n = 0; n < N; n++)
    for (int ic = 0; ic < Ic; ic++)
      for (int r = 0; r < Ih; r++)
        for (int c = 0; c < Iw; c++) {
          size_t pi = ((size_t)(n * Ic + ic) * IhP + (r + 1)) * IwP + (c + 1);
          pin[pi] = in[((size_t)(n * Ic + ic) * Ih + r) * Iw + c];
        }

  /* pack weights: wpack[(ic*9+kh*3+kw)*OcP + oc] = win[oc][ic][kh][kw] */
  for (int ic = 0; ic < Ic; ic++)
    for (int kh = 0; kh < 3; kh++)
      for (int kw = 0; kw < 3; kw++)
        for (int oc = 0; oc < Oc; oc++)
          wpack[((size_t)(ic * 9 + kh * 3 + kw) * OcP) + oc] =
              win[(((size_t)oc * Ic + ic) * 3 + kh) * 3 + kw];

  /* naive NCHW reference with SAME pad */
  for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Oc; oc++)
      for (int oh = 0; oh < Oh; oh++)
        for (int ow = 0; ow < Ow; ow++) {
          float s = 0;
          for (int ic = 0; ic < Ic; ic++)
            for (int kh = 0; kh < 3; kh++)
              for (int kw = 0; kw < 3; kw++) {
                int ih = oh + kh - 1, iw = ow + kw - 1;
                if (ih < 0 || ih >= Ih || iw < 0 || iw >= Iw) continue;
                float x = in[((size_t)(n * Ic + ic) * Ih + ih) * Iw + iw];
                float w = win[(((size_t)oc * Ic + ic) * 3 + kh) * 3 + kw];
                s += x * w;
              }
          ref[((size_t)(n * Oc + oc) * Oh + oh) * Ow + ow] = s;
        }

  MemRef2D min = mk2d(pin, (int64_t)N * Ic, (int64_t)IhP * IwP);
  MemRef1D mw  = mk1d(wpack, (int64_t)Ic * 9 * OcP);
  MemRef2D mout = mk2d(out, (int64_t)N * OcP, (int64_t)OhP * OwP);
  sconv_sme_direct(N, Ic, IwP, OcP, OhP, OwP,
                   min.allocated, min.aligned, min.offset, min.size0, min.size1, min.stride0, min.stride1,
                   mw.allocated, mw.aligned, mw.offset, mw.size, mw.stride,
                   mout.allocated, mout.aligned, mout.offset, mout.size0, mout.size1, mout.stride0, mout.stride1);

  int bad = 0;
  for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Oc; oc++)
      for (int oh = 0; oh < Oh; oh++)
        for (int ow = 0; ow < Ow; ow++) {
          float got = out[((size_t)(n * OcP + oc) * OhP + oh) * OwP + ow];
          float want = ref[((size_t)(n * Oc + oc) * Oh + oh) * Ow + ow];
          if (!close_enough(got, want)) {
            if (!bad) printf("  first mismatch n%d oc%d oh%d ow%d got %g want %g\n",
                             n, oc, oh, ow, got, want);
            bad++;
          }
        }
  printf("shape N%d Ic%d %dx%d Oc%d 3x3 SAME: %s (%d bad)\n",
         N, Ic, Ih, Iw, Oc, bad ? "FAIL" : "PASS", bad);

  free(in); free(win); free(pin); free(wpack); free(out); free(ref);
  return bad ? 1 : 0;
}

int main(void) {
  int bad = 0;
  bad |= run_shape(1, 2, 8, 8, 16, 8, 8);
  bad |= run_shape(1, 3, 10, 10, 17, 10, 10);
  bad |= run_shape(2, 5, 7, 13, 32, 7, 13);
  printf(bad ? "ALL FAIL\n" : "ALL PASS\n");
  return bad ? 1 : 0;
}
