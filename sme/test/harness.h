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
