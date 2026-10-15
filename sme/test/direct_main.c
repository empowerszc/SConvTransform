#include "harness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

/* Full direct conv: stride1, 3x3, SAME pad, pre-padded I/O. Work is flattened
   into tasks over (n, ocb, 4-row block); the caller runs tasks [T0, T0+TN). */
extern void sconv_sme_compute(
    int64_t N, int64_t Ic, int64_t IwP, int64_t OcP, int64_t OhP, int64_t OwP,
    int64_t T0, int64_t TN,
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t,
    float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);

#define R 16

typedef struct { int N, Ic, Ih, Iw, Oc, Oh, Ow; } Cfg;
typedef struct {
  int OhP, OwP, OcP, IhP, IwP;
  float *in, *win, *pin, *wpack, *out;
  int64_t tasks, ocb, ohb;
} Buf;

static void padded_dims(Cfg c, int *OhP, int *OwP, int *OcP, int *IhP, int *IwP) {
  *OhP = (c.Oh + 3) / 4 * 4;
  *OwP = (c.Ow + R - 1) / R * R;
  *OcP = (c.Oc + R - 1) / R * R;
  *IhP = *OhP + 2;
  *IwP = *OwP + 2;
}

static void prep_buffers(Cfg c, Buf *b) {
  padded_dims(c, &b->OhP, &b->OwP, &b->OcP, &b->IhP, &b->IwP);
  b->in    = malloc(sizeof(float) * (size_t)c.N * c.Ic * c.Ih * c.Iw);
  b->win   = malloc(sizeof(float) * (size_t)c.Oc * c.Ic * 9);
  b->pin   = calloc((size_t)c.N * c.Ic * b->IhP * b->IwP, sizeof(float));
  b->wpack = calloc((size_t)c.Ic * 9 * b->OcP, sizeof(float));
  b->out   = calloc((size_t)c.N * b->OcP * b->OhP * b->OwP, sizeof(float));
  b->ocb = b->OcP / R;
  b->ohb = b->OhP / 4;
  b->tasks = (int64_t)c.N * b->ocb * b->ohb;

  for (size_t i = 0; i < (size_t)c.N * c.Ic * c.Ih * c.Iw; i++)
    b->in[i] = 0.1f * (float)((i % 29) - 13);
  for (size_t i = 0; i < (size_t)c.Oc * c.Ic * 9; i++)
    b->win[i] = 0.2f * (float)((i % 17) - 8);

  for (int n = 0; n < c.N; n++)
    for (int ic = 0; ic < c.Ic; ic++)
      for (int r = 0; r < c.Ih; r++)
        for (int col = 0; col < c.Iw; col++) {
          size_t pi = ((size_t)(n * c.Ic + ic) * b->IhP + (r + 1)) * b->IwP + (col + 1);
          b->pin[pi] = b->in[((size_t)(n * c.Ic + ic) * c.Ih + r) * c.Iw + col];
        }
  for (int ic = 0; ic < c.Ic; ic++)
    for (int kh = 0; kh < 3; kh++)
      for (int kw = 0; kw < 3; kw++)
        for (int oc = 0; oc < c.Oc; oc++)
          b->wpack[((size_t)(ic * 9 + kh * 3 + kw) * b->OcP) + oc] =
              b->win[(((size_t)oc * c.Ic + ic) * 3 + kh) * 3 + kw];
}

static void call_kernel(Cfg c, Buf *b, int64_t t0, int64_t tn) {
  MemRef2D min  = mk2d(b->pin, (int64_t)c.N * c.Ic, (int64_t)b->IhP * b->IwP);
  MemRef1D mw   = mk1d(b->wpack, (int64_t)c.Ic * 9 * b->OcP);
  MemRef2D mout = mk2d(b->out, (int64_t)c.N * b->OcP, (int64_t)b->OhP * b->OwP);
  sconv_sme_compute(c.N, c.Ic, b->IwP, b->OcP, b->OhP, b->OwP, t0, tn,
                   min.allocated, min.aligned, min.offset, min.size0, min.size1, min.stride0, min.stride1,
                   mw.allocated, mw.aligned, mw.offset, mw.size, mw.stride,
                   mout.allocated, mout.aligned, mout.offset, mout.size0, mout.size1, mout.stride0, mout.stride1);
}

static int run_shape(Cfg c) {
  Buf b;
  prep_buffers(c, &b);
  float *ref = malloc(sizeof(float) * (size_t)c.N * c.Oc * c.Oh * c.Ow);

  for (int n = 0; n < c.N; n++)
    for (int oc = 0; oc < c.Oc; oc++)
      for (int oh = 0; oh < c.Oh; oh++)
        for (int ow = 0; ow < c.Ow; ow++) {
          float s = 0;
          for (int ic = 0; ic < c.Ic; ic++)
            for (int kh = 0; kh < 3; kh++)
              for (int kw = 0; kw < 3; kw++) {
                int ih = oh + kh - 1, iw = ow + kw - 1;
                if (ih < 0 || ih >= c.Ih || iw < 0 || iw >= c.Iw) continue;
                s += b.in[((size_t)(n * c.Ic + ic) * c.Ih + ih) * c.Iw + iw] *
                     b.win[(((size_t)oc * c.Ic + ic) * 3 + kh) * 3 + kw];
              }
          ref[((size_t)(n * c.Oc + oc) * c.Oh + oh) * c.Ow + ow] = s;
        }

  call_kernel(c, &b, 0, b.tasks);

  int bad = 0;
  for (int n = 0; n < c.N; n++)
    for (int oc = 0; oc < c.Oc; oc++)
      for (int oh = 0; oh < c.Oh; oh++)
        for (int ow = 0; ow < c.Ow; ow++) {
          float got = b.out[((size_t)(n * b.OcP + oc) * b.OhP + oh) * b.OwP + ow];
          float want = ref[((size_t)(n * c.Oc + oc) * c.Oh + oh) * c.Ow + ow];
          if (!close_enough(got, want)) {
            if (!bad) printf("  first mismatch n%d oc%d oh%d ow%d got %g want %g\n",
                             n, oc, oh, ow, got, want);
            bad++;
          }
        }
  printf("shape N%d Ic%d %dx%d Oc%d: %s (%d bad)\n", c.N, c.Ic, c.Ih, c.Iw, c.Oc,
         bad ? "FAIL" : "PASS", bad);
  free(b.in); free(b.win); free(b.pin); free(b.wpack); free(b.out); free(ref);
  return bad ? 1 : 0;
}

/* ---- multithreaded bench ---- */
typedef struct { Cfg c; Buf *b; int64_t t0, tn; } TArg;
static void *worker(void *p) {
  TArg *a = p;
  call_kernel(a->c, a->b, a->t0, a->tn);
  return NULL;
}
static void run_parallel(Cfg c, Buf *b, int nt) {
  pthread_t *th = malloc(sizeof(pthread_t) * nt);
  TArg *ar = malloc(sizeof(TArg) * nt);
  int64_t per = (b->tasks + nt - 1) / nt, t = 0;
  for (int i = 0; i < nt; i++) {
    int64_t cnt = per;
    if (t + cnt > b->tasks) cnt = b->tasks - t;
    if (cnt < 0) cnt = 0;
    ar[i] = (TArg){ c, b, t, cnt };
    t += cnt;
    pthread_create(&th[i], NULL, worker, &ar[i]);
  }
  for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
  free(th); free(ar);
}

static double now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void bench_one(const char *name, Cfg c, int nt) {
  Buf b; prep_buffers(c, &b);
  double gflop = 2.0 * c.N * c.Oh * c.Ow * c.Oc * c.Ic * 9 / 1e9;

  call_kernel(c, &b, 0, b.tasks);                       /* warm */
  int iters = 5;
  double t0 = now_ms();
  for (int i = 0; i < iters; i++) call_kernel(c, &b, 0, b.tasks);
  double ms1 = (now_ms() - t0) / iters;

  double msm = 0;
  if (nt > 1) {
    size_t nsz = (size_t)c.N * b.OcP * b.OhP * b.OwP;
    float *ref = malloc(sizeof(float) * nsz);
    memcpy(ref, b.out, sizeof(float) * nsz);            /* serial result */
    memset(b.out, 0, sizeof(float) * nsz);
    run_parallel(c, &b, nt);
    int bad = 0;
    for (size_t i = 0; i < nsz; i++) if (ref[i] != b.out[i]) bad++;
    printf("  %s parallel-vs-serial: %s (%d diff)\n", name, bad ? "FAIL" : "OK", bad);
    free(ref);
    t0 = now_ms();
    for (int i = 0; i < iters; i++) run_parallel(c, &b, nt);
    msm = (now_ms() - t0) / iters;
  }
  printf("%-10s 1thr %8.2f ms %8.1f GFLOPS", name, ms1, gflop / (ms1 / 1e3));
  if (nt > 1) printf(" | %dthr %8.2f ms %8.1f GFLOPS", nt, msm, gflop / (msm / 1e3));
  printf("\n");
  free(b.in); free(b.win); free(b.pin); free(b.wpack); free(b.out);
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "bench") == 0) {
    int nt = argc > 2 ? atoi(argv[2]) : 38;
    /* ConvBench custom 配置中 stride1/pad1/3x3 的 6 个 SME 候选（几何取自
       scripts/convbench/samples/custom_conv.csv 的 unpadded 输入尺寸）。
       stride2 的 c1/c3/c4/c6/c8 不适用 SME，未列入。 */
    bench_one("custom_0",  (Cfg){ 4, 192,  40,  40, 192,  40,  40 }, nt);
    bench_one("custom_2",  (Cfg){ 4,  96,  80,  80,  96,  80,  80 }, nt);
    bench_one("custom_5",  (Cfg){ 4,  48, 160, 160,  48, 160, 160 }, nt);
    bench_one("custom_7",  (Cfg){ 4, 192,  20,  20, 192,  20,  20 }, nt);
    bench_one("custom_9",  (Cfg){ 4, 384,  80,  80,  96,  80,  80 }, nt);
    bench_one("custom_10", (Cfg){ 4, 768,  40,  40,  96,  40,  40 }, nt);
    return 0;
  }
  int bad = 0;
  bad |= run_shape((Cfg){ 1, 2, 8, 8, 16, 8, 8 });
  bad |= run_shape((Cfg){ 1, 3, 10, 10, 17, 10, 10 });
  bad |= run_shape((Cfg){ 2, 5, 7, 13, 32, 7, 13 });
  printf(bad ? "ALL FAIL\n" : "ALL PASS\n");
  return bad ? 1 : 0;
}
