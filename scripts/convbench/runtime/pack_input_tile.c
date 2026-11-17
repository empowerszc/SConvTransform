/* Fast input packing for SConvTransform (M1).
 *
 * Replaces the scalar `linalg.generic` gather emitted by `applyInputPacking`
 * (one `memref.reinterpret_cast` + `memref.load` per element, ~0.5 GB/s on the
 * target machine) with row-segmented contiguous copies (~13 GB/s).
 *
 * Semantics reproduced exactly (see lib/SConv.cpp computeLinearInputIndices):
 *
 *   L          = ILstart + w
 *   plane(w)   = (L / Ow) * strideH * Iw + fh * dilationH * Iw
 *              + (L % Ow) * strideW     + fw * dilationW
 *
 * Folded over the tile: with `crossed = L / Ow - ILstart / Ow` and
 * `col = L % Ow`, the relative offset is
 *
 *   plane(w) - plane(0) = crossed * (strideH * Iw - Ow * strideW) + w * strideW
 *
 * so each maximal run that stays inside one output row is a contiguous copy
 * when strideW == 1.
 *
 * `src` already points at the tile's first element (n, ic0, oh0, ow0); `off0`
 * is its element offset from the buffer base, used to recover `ow0` and to
 * bounds-check the reads (the original generic performs the same, possibly
 * out-of-range, reads at the epilogue; we clamp instead of faulting).
 */
#include <stddef.h>
#include <string.h>

int pack_input_tile(float *dst, const float *src,
                    long off0, long nc, long Fh, long Fw, long nwin,
                    long Ow, long Iw,
                    long strideH, long strideW,
                    long dilationH, long dilationW,
                    long srcCStride, long srcRemain) {
  if (nc <= 0 || nwin <= 0 || Fh <= 0 || Fw <= 0 || Ow <= 0 || Iw <= 0)
    return 0;

  /* Column of the first window. off0 % Iw removes whole input rows, which are
   * multiples of Iw for the contiguous packed source; dividing by strideW
   * undoes the horizontal stride (ow0 * strideW < Iw holds for any valid
   * convolution, since the last window must be covered). */
  const long ow0 = (off0 % Iw) / strideW;
  const long rowJump = strideH * Iw - Ow * strideW;

  for (long ic = 0; ic < nc; ++ic) {
    const float *cbase = src + ic * srcCStride;
    for (long fh = 0; fh < Fh; ++fh) {
      const long fhOff = fh * dilationH * Iw;
      for (long fw = 0; fw < Fw; ++fw) {
        float *d = dst + ((ic * Fh + fh) * Fw + fw) * nwin;
        const float *row = cbase + fhOff + fw * dilationW;
        long col = ow0, crossed = 0, done = 0;
        while (done < nwin) {
          long run = Ow - col;
          if (run > nwin - done)
            run = nwin - done;
          const float *s = row + crossed * rowJump + done * strideW;
          const long sOff = (long)(s - src);
          if (strideW == 1 && sOff >= 0 && sOff + run <= srcRemain) {
            /* The runs are short (Nwin elements, i.e. 64 B for the common
             * 16-window microkernel), where the libc memcpy call overhead
             * dominates. A plain loop lets the compiler vectorize it. */
            if (run >= 32) {
              memcpy(d + done, s, (size_t)run * sizeof(float));
            } else {
              for (long k = 0; k < run; ++k)
                d[done + k] = s[k];
            }
          } else if (strideW == 1) {
            for (long k = 0; k < run; ++k) {
              const long so = sOff + k;
              d[done + k] = (so >= 0 && so < srcRemain) ? s[k] : 0.0f;
            }
          } else {
            for (long k = 0; k < run; ++k) {
              const long so = sOff + k * strideW;
              d[done + k] =
                  (so >= 0 && so < srcRemain) ? s[k * strideW] : 0.0f;
            }
          }
          done += run;
          col += run;
          if (col >= Ow) {
            col -= Ow;
            ++crossed;
          }
        }
      }
    }
  }
  return 0;
}
