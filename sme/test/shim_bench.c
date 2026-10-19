#include "harness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* End-to-end SME bench: times the public entry `sconv_sme_direct`, which
   includes weight packing + input pad + threading + output unpad (i.e. the
   honest "our conv" number, unlike the kernel-only `direct.out bench`).
   Input is pre-padded by the framework: Ih=Oh+2, Iw=Ow+2. Threads via
   SCONV_SME_THREADS (falls back to process CPU affinity). */

typedef struct { const char *name; int N, Ic, Oh, Ow, Oc; } Cfg;

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void bench_one(Cfg c) {
  int Ih = c.Oh + 2, Iw = c.Ow + 2;
  float *in  = malloc(sizeof(float) * (size_t)c.N * c.Ic * Ih * Iw);
  float *flt = malloc(sizeof(float) * (size_t)c.Oc * c.Ic * 9);
  float *out = malloc(sizeof(float) * (size_t)c.N * c.Oc * c.Oh * c.Ow);
  for (size_t i = 0; i < (size_t)c.N * c.Ic * Ih * Iw; i++) in[i]  = 0.1f * (float)((i % 29) - 13);
  for (size_t i = 0; i < (size_t)c.Oc * c.Ic * 9; i++)     flt[i] = 0.2f * (float)((i % 17) - 8);
  double gflop = 2.0 * c.N * c.Oh * c.Ow * c.Oc * c.Ic * 9 / 1e9;

  sconv_sme_direct(c.N, c.Ic, Ih, Iw, c.Oc, c.Oh, c.Ow, in, flt, out); /* warm */
  int iters = 5;
  double t0 = now_ms();
  for (int i = 0; i < iters; i++)
    sconv_sme_direct(c.N, c.Ic, Ih, Iw, c.Oc, c.Oh, c.Ow, in, flt, out);
  double ms = (now_ms() - t0) / iters;
  printf("%-16s e2e %10.3f ms %9.1f GFLOPS\n", c.name, ms, gflop / (ms / 1e3));
  free(in); free(flt); free(out);
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "bench") == 0) {
    if (argc > 2) setenv("SCONV_SME_THREADS", argv[2], 1);
    /* stride1/pad1/3x3 SME-eligible shapes (unpadded input sizes). */
    bench_one((Cfg){"custom_0",      4, 192,  40,  40, 192});
    bench_one((Cfg){"custom_2",      4,  96,  80,  80,  96});
    bench_one((Cfg){"custom_5",      4,  48, 160, 160,  48});
    bench_one((Cfg){"custom_7",      4, 192,  20,  20, 192});
    bench_one((Cfg){"custom_9",      4, 384,  80,  80,  96});
    bench_one((Cfg){"custom_10",     4, 768,  40,  40,  96});
    bench_one((Cfg){"common_r18_56", 1,  64,  56,  56,  64});
    bench_one((Cfg){"common_r18_28", 1, 128,  28,  28, 128});
    bench_one((Cfg){"common_r18_14", 1, 256,  14,  14, 256});
    bench_one((Cfg){"common_r18_7",  1, 512,   7,   7, 512});
    bench_one((Cfg){"common_thr_n8", 8,  64,  56,  56,  64});
    return 0;
  }
  printf("usage: %s bench [threads]\n", argv[0]);
  return 1;
}
