/* Fast input packing for SConvTransform (M1 + M2).
 *
 * Replaces the scalar `linalg.generic` gather emitted by `applyInputPacking` /
 * `inputMultipackingOpt` (one `memref.reinterpret_cast` + `memref.load` per
 * element, ~0.5 GB/s on the target machine) with row-segmented copies.
 *
 * Semantics reproduced exactly (see lib/SConv.cpp computeLinearInputIndices and
 * computeMultiPackInputIndices):
 *
 *   L          = ILstart + w,          w = t*Nwin + nwin   (w in [0, Ni*Nwin))
 *   plane(w)   = (L / Ow) * strideH * Iw + fh * dilationH * Iw
 *              + (L % Ow) * strideW     + fw * dilationW
 *
 * Folded relative to the first window (w = 0):
 *
 *   plane(w) - plane(0) = crossed * (strideH * Iw - Ow * strideW) + w * strideW
 *
 * where `crossed = L / Ow - ILstart / Ow`. So each maximal run that stays inside
 * one output row is a contiguous copy when strideW == 1.
 *
 * Destination layouts (both fully contiguous when `t` is the outer loop):
 *   single packing:     [Nc, Fh, Fw, Nwin]              (ni == 1)
 *   multi  packing:     [Ni, Nc, Fh, Fw, Nwin]
 * i.e. dst index (((t*Nc + ic)*Fh + fh)*Fw + fw)*Nwin + nwin.
 *
 * `src` already points at the first element (n, ic0, ILstart + 0); `off0` is its
 * element offset from the buffer base, used to recover the first column and to
 * bounds-check reads (the original generic performs the same, possibly
 * out-of-range, reads at the epilogue; we clamp instead of faulting).
 */
#include <stddef.h>
#include <string.h>

int pack_input_tile(float *dst, const float *src,
                    long off0, long nc, long Fh, long Fw, long ni, long nwin,
                    long Ow, long Iw,
                    long strideH, long strideW,
                    long dilationH, long dilationW,
                    long srcCStride, long srcRemain) {
  if (nc <= 0 || nwin <= 0 || ni <= 0 || Fh <= 0 || Fw <= 0 || Ow <= 0 ||
      Iw <= 0)
    return 0;

  /* Column of the first window. off0 % Iw removes whole input rows, which are
   * multiples of Iw for the contiguous packed source; dividing by strideW
   * undoes the horizontal stride (ow0 * strideW < Iw holds for any valid
   * convolution, since the last window must be covered). */
  const long ow0 = (off0 % Iw) / strideW;
  const long rowJump = strideH * Iw - Ow * strideW;
  const long dstTile = nc * Fh * Fw * nwin; /* stride between t blocks */

  for (long t = 0; t < ni; ++t) {
    const long w0 = t * nwin;
    /* Starting column and crossed rows for window w0. */
    long col = (ow0 + w0) % Ow;
    long crossed = (ow0 + w0) / Ow;
    const long wShift = w0 * strideW; /* window part already advanced */

    for (long ic = 0; ic < nc; ++ic) {
      const float *cbase = src + ic * srcCStride;
      /* t is the outer loop, so this block is contiguous in dst. */
      float *dTile = dst + t * dstTile;
      for (long fh = 0; fh < Fh; ++fh) {
        const long fhOff = fh * dilationH * Iw;
        for (long fw = 0; fw < Fw; ++fw) {
          float *d = dTile + ((ic * Fh + fh) * Fw + fw) * nwin;
          const float *row = cbase + fhOff + fw * dilationW + wShift;
          /* `col`/`crossed` are the same for every (ic, fh, fw) at this t. */
          long c = col, cr = crossed, done = 0;
          while (done < nwin) {
            long run = Ow - c;
            if (run > nwin - done)
              run = nwin - done;
            const float *s = row + cr * rowJump + done * strideW;
            const long sOff = (long)(s - src);
            if (strideW == 1 && sOff >= 0 && sOff + run <= srcRemain) {
              /* Runs are short (Nwin elements, i.e. 64 B for the common
               * 16-window microkernel), where libc memcpy call overhead can
               * dominate; a plain loop lets the compiler vectorize it. */
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
            c += run;
            if (c >= Ow) {
              c -= Ow;
              ++cr;
            }
          }
        }
      }
    }
  }
  return 0;
}
