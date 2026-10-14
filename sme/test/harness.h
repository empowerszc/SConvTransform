#ifndef SME_TEST_HARNESS_H
#define SME_TEST_HARNESS_H
#include <stdint.h>
#include <math.h>

/* Canonical memref descriptor ABI (dynamic ranks). MLIR lowers each memref
   argument to flattened descriptor fields:
     rank-1: (allocated, aligned, offset, size, stride)                     -> 5
     rank-2: (allocated, aligned, offset, size0, size1, stride0, stride1)   -> 7
     rank-3: (... offset, size0..2, stride0..2)                             -> 9 */
typedef struct { float *allocated, *aligned; int64_t offset, size, stride; } MemRef1D;
typedef struct { float *allocated, *aligned; int64_t offset, size0, size1, stride0, stride1; } MemRef2D;
typedef struct { float *allocated, *aligned; int64_t offset,
                 size0, size1, size2, stride0, stride1, stride2; } MemRef3D;

static inline MemRef1D mk1d(float *p, int64_t n) {
  MemRef1D m = { p, p, 0, n, 1 }; return m;
}
static inline MemRef2D mk2d(float *p, int64_t r, int64_t c) {
  MemRef2D m = { p, p, 0, r, c, c, 1 }; return m;
}
static inline MemRef3D mk3d(float *p, int64_t d0, int64_t d1, int64_t d2) {
  MemRef3D m = { p, p, 0, d0, d1, d2, d1 * d2, d2, 1 }; return m;
}
static inline int close_enough(float got, float want) {
  float d = fabsf(got - want);
  return d <= 1e-3f * (1.0f + fabsf(want));
}
#endif
