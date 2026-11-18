// Faster drop-in replacement for MLIR's runner-utils `memrefCopy`.
//
// Why: when a memref has dynamic strides (which is how tensors are passed
// across a bufferized function boundary), the stock memrefCopy falls back to an
// element-by-element walk. On the ConvBench payloads that runs at ~0.45 GB/s,
// and it accounted for ~24 ms of custom_0's 85 ms/iter before this change
// (docs/OPTIMIZATION_PLAN_2026-11-18.md).
//
// This implementation keeps the ABI and semantics identical, but copies the
// innermost unit-stride run with memmove and only walks the outer dimensions.
// For the boundary copies (4D, row-contiguous) that turns 24 ms into ~2 ms and
// gives custom_0 50.9 -> 67.6 GFLOPS with bit-identical results.
//
// ABI (mlir/ExecutionEngine/CRunnerUtils.h):
//   void memrefCopy(int64_t elemSize, UnrankedMemRefType<char> *src,
//                   UnrankedMemRefType<char> *dst);
// UnrankedMemRefType = { int64_t rank; void *descriptor; }
// descriptor = { void *allocated; void *aligned; int64_t offset;
//                int64_t sizes[rank]; int64_t strides[rank]; }
//
// The library must be loaded before libmlir_runner_utils.so so that this
// definition wins symbol resolution (run_convbench.get_libs puts it first).

#include <stdint.h>
#include <string.h>

typedef struct {
  int64_t rank;
  void *descriptor;
} Unranked;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
} Head;

static inline int64_t *sizes_of(void *desc) {
  return (int64_t *)((char *)desc + sizeof(Head));
}

static inline int64_t *strides_of(void *desc, int64_t rank) {
  return sizes_of(desc) + rank;
}

// Copy one innermost run, then recurse over the outer dimensions. Recursion
// depth equals the memref rank, which is small (<= 4 in the SConv pipeline).
static void copy_nd(char *src, char *dst, int64_t elemSize, int64_t rank,
                    const int64_t *sizes, const int64_t *srcStrides,
                    const int64_t *dstStrides, int64_t dim) {
  if (dim == rank - 1) {
    int64_t n = sizes[dim];
    if (n <= 0)
      return;
    if (srcStrides[dim] == 1 && dstStrides[dim] == 1) {
      memmove(dst, src, (size_t)(n * elemSize));
      return;
    }
    for (int64_t i = 0; i < n; ++i)
      memmove(dst + i * dstStrides[dim] * elemSize,
              src + i * srcStrides[dim] * elemSize, (size_t)elemSize);
    return;
  }
  int64_t n = sizes[dim];
  int64_t ss = srcStrides[dim] * elemSize;
  int64_t ds = dstStrides[dim] * elemSize;
  for (int64_t i = 0; i < n; ++i)
    copy_nd(src + i * ss, dst + i * ds, elemSize, rank, sizes, srcStrides,
            dstStrides, dim + 1);
}

void memrefCopy(int64_t elemSize, Unranked *srcArg, Unranked *dstArg) {
  if (!srcArg || !dstArg || srcArg->rank != dstArg->rank)
    return;
  int64_t rank = srcArg->rank;

  Head *s = (Head *)srcArg->descriptor;
  Head *d = (Head *)dstArg->descriptor;
  if (!s || !d || !s->aligned || !d->aligned)
    return;

  char *src = (char *)s->aligned + s->offset * elemSize;
  char *dst = (char *)d->aligned + d->offset * elemSize;

  if (rank == 0) {
    memmove(dst, src, (size_t)elemSize);
    return;
  }

  int64_t *sizes = sizes_of(srcArg->descriptor);
  int64_t *srcStrides = strides_of(srcArg->descriptor, rank);
  int64_t *dstStrides = strides_of(dstArg->descriptor, rank);
  copy_nd(src, dst, elemSize, rank, sizes, srcStrides, dstStrides, 0);
}
