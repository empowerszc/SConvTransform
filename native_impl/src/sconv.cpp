// sconv.cpp — Standalone SConv convolution implementation
// Implements: naive conv, CSA-guided tiling + packing + OpenBLAS microkernel
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make
// Run:   ./sconv_bench [--csv file] [--runs N] [--warmup N]

#include "sconv_csa.h"
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#if defined(USE_OPENBLAS)
#include <cblas.h>
#elif defined(USE_KBLAST)
#include <kblas.h>
#endif

// ---- Tensor helpers ----
struct Tensor {
    std::vector<float> data;
    int N, C, H, W;
    Tensor(int n, int c, int h, int w) : data((size_t)n*c*h*w, 0), N(n), C(c), H(h), W(w) {}
    float& at(int n, int c, int h, int w) {
        return data[((size_t)n*C+c)*H*W + (size_t)h*W + w];
    }
    float at(int n, int c, int h, int w) const {
        return data[((size_t)n*C+c)*H*W + (size_t)h*W + w];
    }
    size_t size_bytes() const { return data.size() * sizeof(float); }
};

// ---- Naive reference convolution (NCHW + FCHW) ----
void conv2d_naive(const Tensor& input, const Tensor& weight, Tensor& output,
                  int strideH, int strideW, int padTop, int padLeft) {
    int N = input.N, Ic = input.C, Ih = input.H, Iw = input.W;
    int Oc = weight.N, Fh = weight.H, Fw = weight.W;
    int Oh = output.H, Ow = output.W;
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < Oc; oc++)
    for (int oh = 0; oh < Oh; oh++)
    for (int ow = 0; ow < Ow; ow++) {
        float sum = 0.0f;
        for (int ic = 0; ic < Ic; ic++)
        for (int fh = 0; fh < Fh; fh++)
        for (int fw = 0; fw < Fw; fw++) {
            int ih = oh * strideH + fh - padTop;
            int iw = ow * strideW + fw - padLeft;
            if (ih >= 0 && ih < Ih && iw >= 0 && iw < Iw)
                sum += input.at(n, ic, ih, iw) * weight.at(oc, ic, fh, fw);
        }
        output.at(n, oc, oh, ow) = sum;
    }
}

// ---- SConv: tiled + packed convolution with OpenBLAS microkernel ----
//
// The algorithm mirrors the MLIR SConvTransform pipeline:
//   1. CSA: compute Nc, K2, K3, IS/WS schedule
//   2. Collapse spatial dims (Oh*Ow → 1D)
//   3. Two-level tiling: outer (Nc, K2, K3) + inner (Nwin, Nf)
//   4. Pack input tile [Nc, Fh, Fw, Nwin] and filter tile [Nc, Fh, Fw, Nf]
//   5. Microkernel: C[Nf, Nwin] += B[K, Nf]^T * A[K, Nwin]  (via cblas_sgemm)

void conv2d_sconv(const Tensor& input, const Tensor& weight, Tensor& output,
                  int strideH, int strideW, int padTop, int padLeft,
                  int Nwin, int Nf, int num_threads) {
    int N = input.N, Ic = input.C, Ih = input.H, Iw = input.W;
    int Oc = weight.N, Fh = weight.H, Fw = weight.W;
    int Oh = output.H, Ow = output.W;
    int Ohw = Oh * Ow;  // collapsed spatial

    // CSA
    ConvInfo ci{(int64_t)Ic, (int64_t)Iw, (int64_t)Oh, (int64_t)Ow,
                 (int64_t)Fh, (int64_t)Fw, (int64_t)Oc, 0, 0, 4};
    ArchInfo arch{
        (uint32_t)(32768 * 0.9),
        (uint32_t)(768 * 1024 * 0.9),
        0,  // L3=0 (no L3; set to HBM size if available)
        4, 20, 300, 300, 128
    };
    mKInfo mK{(uint8_t)Nwin, (uint8_t)Nf, (uint16_t)(Nwin * Nf)};
    CSA csa(arch, ci, mK);
    CSAStrategy strat = csa.run();

    int Nc = strat.tile_c;
    int K2 = strat.k2;
    int K3 = strat.k3;
    int K = Nc * Fh * Fw;

    // Per-thread packed buffers (no reallocation across runs)
    static thread_local std::vector<float> packedA, packedC;
    static thread_local std::vector<float> inrow;
    static thread_local std::vector<int> row_w0, row_w1;
    static thread_local long long tl_pack_ns = 0, tl_gemm_ns = 0, tl_calls = 0;
    static thread_local long long tl_unpack_ns = 0;
    bool timing = getenv("SCONV_TIMING") != nullptr;
    // Zero this thread's counters so each call reports its own breakdown
    // (counters are static and otherwise accumulate across calls/configs).
    auto reset_timing = [&]() {
        if (timing) { tl_pack_ns = 0; tl_gemm_ns = 0; tl_unpack_ns = 0; tl_calls = 0; }
    };

    // Filter pre-pack: Bc[ft][ic0][k=(ic_local*Fh+fh)*Fw+fw][nf], contiguous
    // slices (ldb = nf_str) → fast CblasTrans B operand for every sgemm.
    // Rebuilt per oc-tile width so the auto-tuner can try different Nf values.
    int nIc = (Ic + Nc - 1) / Nc;
    std::vector<float> Bc;   // shared, read-only inside the parallel region
    auto build_Bc = [&](int nf_str) -> int {
        int nFlt = (Oc + nf_str - 1) / nf_str;
        Bc.assign((size_t)nFlt * nIc * Nc * Fh * Fw * nf_str, 0.0f);
        size_t tileStride = (size_t)Nc * Fh * Fw * nf_str;
        // Embarrassingly parallel over the disjoint (ft, it) slices.
#pragma omp parallel for collapse(2) schedule(static) num_threads(num_threads)
        for (int ft = 0; ft < nFlt; ft++) {
            for (int it = 0; it < nIc; it++) {
                int nf_start = ft * nf_str;
                int nf = std::min(nf_str, Oc - nf_start);
                int ic0 = it * Nc;
                int nc = std::min(Nc, Ic - ic0);
                float* pB = &Bc[(size_t)ft * nIc * tileStride + (size_t)it * tileStride];
                for (int ic = 0; ic < nc; ic++)
                for (int fh = 0; fh < Fh; fh++)
                for (int fw = 0; fw < Fw; fw++)
                    for (int f = 0; f < nf; f++)
                        pB[((size_t)(ic * Fh + fh) * Fw + fw) * nf_str + f] =
                            weight.at(nf_start + f, ic0 + ic, fh, fw);
            }
        }
        return nFlt;
    };
    // One task per (batch, window tile): each task packs the full-channel A
    // slice once, then runs all oc-tile GEMMs. Output regions are disjoint
    // across window tiles and batches → race-free, zero redundant packing.
    //
    // `run_pass` runs one full convolution for a given window width; keeping it
    // as a lambda lets us auto-tune Nwin per shape/thread-count (see below).
    auto run_pass = [&](int nw_str, int nf_str) {
    int nFlt = (Oc + nf_str - 1) / nf_str;
    int nWinTilesL = (Ohw + nw_str - 1) / nw_str;
    int totalL = N * nWinTilesL;

    auto run_tile = [&](int n, int wt) {
        // Per-thread buffers must be sized inside the parallel region.
        int nwin = std::min(nw_str, Ohw - wt * nw_str);
        packedA.resize((size_t)Ic * Fh * Fw * nw_str);
        packedC.resize((size_t)nf_str * nw_str);

        // --- Pack full-channel input tile A[k_global][nwin], k=(ic*Fh+fh)*Fw+fw ---
        // Row-cache: read each input row once per (ic, fh, r) and reuse it across
        // the Fw kernel offsets as shifted slices.
        auto tp0 = timing ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
        int base_pos = wt * nw_str;
        int r0 = base_pos / Ow;
        int r1 = (base_pos + nwin - 1) / Ow;
        int nrows = r1 - r0 + 1;
        row_w0.resize(nrows); row_w1.resize(nrows);
        for (int r = 0; r < nrows; r++) {
            int rb = (r0 + r) * Ow;
            row_w0[r] = std::max(0, rb - base_pos);
            row_w1[r] = std::min(nwin, (rb + Ow) - base_pos);
        }
        int maxlen = (nwin - 1) * strideW + Fw;
        if ((int)inrow.size() < maxlen) inrow.resize(maxlen);
        for (int ic = 0; ic < Ic; ic++) {
            const float* im = input.data.data() +
                ((size_t)n * Ic + ic) * (size_t)(Ih * Iw);
            for (int fh = 0; fh < Fh; fh++) {
                for (int r = 0; r < nrows; r++) {
                    int ih = (r0 + r) * strideH + fh - padTop;
                    int w0 = row_w0[r], w1 = row_w1[r];
                    int len = (w1 - w0 - 1) * strideW + Fw;
                    int iw_lo = (base_pos - (r0 + r) * Ow + w0) * strideW - padLeft;
                    int iw_hi = iw_lo + len;
                    if (ih < 0 || ih >= Ih) {
                        std::fill(inrow.data(), inrow.data() + len, 0.0f);
                    } else {
                        const float* rp = im + (size_t)ih * Iw;
                        if (iw_lo >= 0 && iw_hi <= Iw) {
                            std::copy(rp + iw_lo, rp + iw_hi, inrow.data());
                        } else {
                            std::fill(inrow.data(), inrow.data() + len, 0.0f);
                            int s0 = std::max(0, iw_lo);
                            int s1 = std::min(Iw, iw_hi);
                            if (s0 < s1)
                                std::copy(rp + s0, rp + s1,
                                          inrow.data() + (s0 - iw_lo));
                        }
                    }
                    for (int fw = 0; fw < Fw; fw++) {
                        float* pA = &packedA[((size_t)(ic * Fh + fh) * Fw + fw) * nwin];
                        const float* s = inrow.data() + fw;
                        for (int w = w0; w < w1; w++)
                            pA[w] = s[(w - w0) * strideW];
                    }
                }
            }
        }
        auto tp1 = timing ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
        if (timing)
            tl_pack_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tp1 - tp0).count();

        // --- GEMM over oc-tiles, accumulating C across channel tiles ---
        for (int ft = 0; ft < nFlt; ft++) {
            int nf_start = ft * nf_str;
            int nf = std::min(nf_str, Oc - nf_start);
            auto t0 = timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
#if defined(USE_OPENBLAS) || defined(USE_KBLAST)
            for (int ic0 = 0; ic0 < Ic; ic0 += Nc) {
                int nc = std::min(Nc, Ic - ic0);
                int k_dim = nc * Fh * Fw;
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            nf, nwin, k_dim,
                            1.0f,
                            &Bc[(size_t)ft * nIc * (size_t)Nc * Fh * Fw * nf_str +
                                (size_t)(ic0 / Nc) * (size_t)Nc * Fh * Fw * nf_str], nf_str,
                            &packedA[(size_t)ic0 * Fh * Fw * nwin], nwin,
                            (ic0 == 0 ? 0.0f : 1.0f),
                            packedC.data(), nwin);
            }
#else
            for (int fi = 0; fi < nf; fi++)
            for (int wi = 0; wi < nwin; wi++) {
                float s = 0.0f;
                for (int ic0 = 0; ic0 < Ic; ic0 += Nc)
                for (int ki = 0; ki < std::min(Nc, Ic - ic0) * Fh * Fw; ki++)
                    s += Bc[(size_t)ft * nIc * (size_t)Nc * Fh * Fw * nf_str +
                            (size_t)(ic0 / Nc) * (size_t)Nc * Fh * Fw * nf_str + (size_t)ki * nf_str + fi] *
                         packedA[(size_t)(ic0 + ki / (Fh * Fw)) * Fh * Fw * nwin + ki % (Fh * Fw) * nwin + wi];
                packedC[fi * nwin + wi] = s;
            }
#endif
            auto t1 = timing ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
            if (timing) {
                tl_gemm_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                tl_calls += 1;
            }

            // --- Unpack C → output, row-segmented (no div/mod) ---
            auto tu0 = timing ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
            for (int f = 0; f < nf; f++) {
                const float* pc = &packedC[f * nwin];
                float* outbase = output.data.data() +
                    ((size_t)n * Oc + (nf_start + f)) * (size_t)Ohw;
                for (int r = 0; r < nrows; r++) {
                    int w0 = row_w0[r], w1 = row_w1[r];
                    int oh = r0 + r;
                    float* o = outbase + (size_t)oh * Ow;
                    int col0 = base_pos + w0 - (r0 + r) * Ow;
                    for (int w = w0; w < w1; w++)
                        o[col0 + w - w0] += pc[w];
                }
            }
            auto tu1 = timing ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
            if (timing)
                tl_unpack_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tu1 - tu0).count();
        } // oc tiles
    };

        if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads)
            {
                reset_timing();
#pragma omp for schedule(dynamic)
                for (int task = 0; task < totalL; task++) {
                    run_tile(task / nWinTilesL, task % nWinTilesL);
                }
            }
        } else {
            reset_timing();
            for (int task = 0; task < totalL; task++) {
                run_tile(task / nWinTilesL, task % nWinTilesL);
            }
        }
    };

    // --- Adaptive tiling (Nwin, Nf) ---------------------------------------
    // Too-large Nwin starves parallelism when N*nWinTiles is small (batch=1,
    // low resolution); too-small Nwin hurts GEMM efficiency; Nf trades filter
    // reuse vs tile count. Pick (Nwin, Nf) per (shape, threads) by timing a
    // small grid once, then cache it. Bc is rebuilt for the chosen Nf.
    int nw = Nwin, nf = Nf;
    bool autotune = (Nwin <= 0 || Nf <= 0);
    if (autotune && getenv("SCONV_NO_AUTOTUNE")) {
        if (nw <= 0) nw = 32;
        if (nf <= 0) nf = 16;
        autotune = false;
    }
    if (autotune) {
        struct Tune { int nw, nf, nc; };
        static thread_local std::map<std::string, Tune> tile_cache;
        std::string key = std::to_string(N) + "_" + std::to_string(Ic) + "_" +
                          std::to_string(Ohw) + "_" + std::to_string(Oc) + "_" +
                          std::to_string(strideH) + "_" + std::to_string(Fh) + "_" +
                          std::to_string(Fw) + "_" + std::to_string(num_threads);
        auto itc = tile_cache.find(key);
        if (itc != tile_cache.end()) {
            nw = itc->second.nw;
            nf = itc->second.nf;
            Nc = itc->second.nc;
            nIc = (Ic + Nc - 1) / Nc;
        } else {
            int hiw = std::max(8, ((Ohw + 7) / 8) * 8);
            int hif = std::max(8, Oc);
            const int candsW[4] = {8, 16, 32, 64};
            const int candsF[2] = {16, 32};
            double best_ms = 1e300;
            int best_nw = 16, best_nf = std::min(16, hif);
            bool saved = timing;
            timing = false;  // don't pollute SCONV_TIMING with tuning passes
            // Stage 1: (Nwin, Nf) grid at the CSA-default Nc.
            for (int fi = 0; fi < 2; fi++) {
                int cf = std::min(candsF[fi], hif);
                if (fi > 0 && cf == std::min(candsF[fi - 1], hif)) continue;
                build_Bc(cf);
                int prevw = -1;
                for (int wi = 0; wi < 4; wi++) {
                    int cw = std::min(candsW[wi], hiw);
                    if (cw == prevw) continue;
                    prevw = cw;
                    std::memset(output.data.data(), 0, output.size_bytes());
                    run_pass(cw, cf);  // warm
                    double c_ms = 1e300;
                    for (int rep = 0; rep < 5; rep++) {
                        std::memset(output.data.data(), 0, output.size_bytes());
                        auto s0 = std::chrono::steady_clock::now();
                        run_pass(cw, cf);
                        double ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - s0).count();
                        if (ms < c_ms) c_ms = ms;
                    }
                    if (c_ms < best_ms) { best_ms = c_ms; best_nw = cw; best_nf = cf; }
                }
            }
            // Stage 2: Nc refinement, only for parallel-starved shapes (where the
            // window-tile count < threads, so the channel block matters most).
            // The CSA tile_c heuristic uses a generic cache model; on starved
            // shapes the real optimum swings +5..+23%. Require a 3% win to switch
            // (protects against timing noise), and always keep the default.
            int best_nc = Nc;
            int winTiles = (Ohw + best_nw - 1) / best_nw;
            bool starved = (long)N * winTiles < num_threads;
            if (starved) {
                int base_nc = Nc;
                const int candsC[4] = {base_nc, 16, 32, 64};
                for (int ci = 0; ci < 4; ci++) {
                    int cc = std::min(candsC[ci], Ic);
                    if (cc < 1) cc = 1;
                    bool dup = false;
                    for (int pj = 0; pj < ci; pj++)
                        if (std::min(candsC[pj], Ic) == cc) dup = true;
                    if (dup) continue;
                    Nc = cc;
                    nIc = (Ic + Nc - 1) / Nc;
                    build_Bc(best_nf);
                    std::memset(output.data.data(), 0, output.size_bytes());
                    run_pass(best_nw, best_nf);  // warm
                    double c_ms = 1e300;
                    for (int rep = 0; rep < 5; rep++) {
                        std::memset(output.data.data(), 0, output.size_bytes());
                        auto s0 = std::chrono::steady_clock::now();
                        run_pass(best_nw, best_nf);
                        double ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - s0).count();
                        if (ms < c_ms) c_ms = ms;
                    }
                    if (c_ms < best_ms * 0.97) { best_ms = c_ms; best_nc = cc; }
                }
            }
            Nc = best_nc;
            nIc = (Ic + Nc - 1) / Nc;
            timing = saved;
            nw = best_nw;
            nf = best_nf;
            tile_cache[key] = Tune{nw, nf, Nc};
            // Clear tuner leftovers so the (accumulating) run below starts clean.
            // Only needed on a cache miss; doing it every call costs ~0.3ms.
            std::memset(output.data.data(), 0, output.size_bytes());
        }
    }
    build_Bc(nf);
    if (autotune && getenv("SCONV_VERBOSE"))
        fprintf(stderr, "  [auto] N=%d Ic=%d Ohw=%d Oc=%d -> Nwin=%d Nf=%d Nc=%d (threads=%d)\n",
                N, Ic, Ohw, Oc, nw, nf, Nc, num_threads);
    run_pass(nw, nf);
    if (timing) {
        long long pack_ns = 0, gemm_ns = 0, calls = 0, unpack_ns = 0;
#pragma omp parallel reduction(+ : pack_ns, gemm_ns, calls, unpack_ns) num_threads(num_threads)
        {
            pack_ns += tl_pack_ns;
            gemm_ns += tl_gemm_ns;
            calls += tl_calls;
            unpack_ns += tl_unpack_ns;
        }
        fprintf(stderr, "  [timing] pack=%.2fms gemm=%.2fms unpack=%.2fms calls=%lld (%.3fus/call)\n",
                pack_ns * 1e-6, gemm_ns * 1e-6, unpack_ns * 1e-6, calls,
                calls > 0 ? (double)gemm_ns / calls * 1e-3 : 0.0);
    }
}

// ---- Verification ----
bool verify(const Tensor& a, const Tensor& b, float tol = 1e-3f) {
    if (a.data.size() != b.data.size()) return false;
    for (size_t i = 0; i < a.data.size(); i++) {
        if (std::abs(a.data[i] - b.data[i]) > tol * (1.0f + std::abs(a.data[i]))) {
            printf("  MISMATCH at %zu: %f vs %f (diff=%e)\n", i, a.data[i], b.data[i],
                   std::abs(a.data[i] - b.data[i]));
            return false;
        }
    }
    return true;
}

// ---- Timing ----
double rtclock() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// ---- Test configurations ----
struct ConvConfig {
    const char* name;
    int N, Ic, Ih, Iw, Oc, Fh, Fw, Oh, Ow, stride, pad;
};

ConvConfig configs[] = {
    // Custom user configs (batch=4)
    {"custom_0",  4,192,40,40,  192,3,3, 40,40,   1,1},
    {"custom_1",  4,768,80,80,  768,3,3, 40,40,   2,1},
    {"custom_2",  4,96,80,80,    96,3,3, 80,80,   1,1},
    {"custom_3",  4,768,40,40,  768,3,3, 20,20,   2,1},
    {"custom_4",  4,384,160,160,384,3,3, 80,80,   2,1},
    {"custom_5",  4,48,160,160,  48,3,3, 160,160, 1,1},
    {"custom_6",  4,96,320,320, 192,3,3, 160,160, 2,1},
    {"custom_7",  4,192,20,20,  192,3,3, 20,20,   1,1},
    {"custom_8",  4,384,80,80,  384,3,3, 40,40,   2,1},
    {"custom_9",  4,384,80,80,   96,3,3, 80,80,   1,1},
    {"custom_10", 4,768,40,40,   96,3,3, 40,40,   1,1},
    // Common ResNet-style 3x3 configs (batch=1 unless noted)
    {"common_r18_56",   1, 64, 56,56,  64,3,3, 56,56,   1,1},
    {"common_r18_28",   1,128, 28,28, 128,3,3, 28,28,   1,1},
    {"common_r18_14",   1,256, 14,14, 256,3,3, 14,14,   1,1},
    {"common_r18_7",    1,512,  7, 7, 512,3,3,  7, 7,   1,1},
    {"common_r50_ds56", 1,128, 56,56, 256,3,3, 28,28,   2,1},
    {"common_r50_ds28", 1,256, 28,28, 512,3,3, 14,14,   2,1},
    {"common_thr_n8",   8, 64, 56,56,  64,3,3, 56,56,   1,1},
    // Small configs (batch=1, for quick verification)
    {"small_0",   1,18,14,14,  144,3,3, 7,7,     2,1},
    {"small_1",   1,48,10,10,   64,2,2, 5,5,     2,0},
};

int main(int argc, char** argv) {
    int runs = 3, warmup = 1;
    int Nwin = 0, Nf = 0;   // <=0 => auto-tune (Nwin, Nf)
    int threads = 1;
    bool skip_naive = false;
    const char* filter = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i+1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i+1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--filter") == 0 && i+1 < argc) filter = argv[++i];
        else if (strcmp(argv[i], "--nwin") == 0 && i+1 < argc) Nwin = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nf") == 0 && i+1 < argc) Nf = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip-naive") == 0) skip_naive = true;
    }

    bool has_blas =
#if defined(USE_OPENBLAS) || defined(USE_KBLAST)
        true;
#else
        false;
#endif

    printf("SConv Benchmark (Nwin=%d, Nf=%d, threads=%d, OpenBLAS=%s)\n", Nwin, Nf, threads, has_blas ? "YES" : "NO");
    printf("warmup=%d, runs=%d\n\n", warmup, runs);

    printf("%-16s %10s %12s %12s %10s %10s %10s\n",
           "Conv", "FLOPs", "Naive(ms)", "SConv(ms)", "Naive GF", "SConv GF", "Speedup");
    printf("%-16s %10s %12s %12s %10s %10s %10s\n",
           "----", "----", "----", "----", "----", "----", "----");

    int n_configs = sizeof(configs) / sizeof(configs[0]);
    int n_pass = 0, n_total = 0;

    for (int ci = 0; ci < n_configs; ci++) {
        auto& c = configs[ci];
        if (filter && !strstr(c.name, filter)) continue;
        n_total++;

        int N=c.N, Ic=c.Ic, Ih=c.Ih, Iw=c.Iw, Oc=c.Oc, Fh=c.Fh, Fw=c.Fw;
        int Oh=c.Oh, Ow=c.Ow, stride=c.stride, pad=c.pad;
        int padH = (Ih + 2*pad - (Fh-1) - 1) / stride + 1; // verify
        double flops = 2.0 * N * Oc * Oh * Ow * Ic * Fh * Fw;

        Tensor input(N, Ic, Ih, Iw);
        Tensor weight(Oc, Ic, Fh, Fw);
        Tensor out_naive(N, Oc, Oh, Ow);
        Tensor out_sconv(N, Oc, Oh, Ow);

        // Init data (deterministic)
        for (size_t i = 0; i < input.data.size(); i++) input.data[i] = (float)(i % 1000);
        for (size_t i = 0; i < weight.data.size(); i++) weight.data[i] = (float)(i % 100);

        // Correctness: run once, compare
        bool ok = true;
        if (!skip_naive) {
            conv2d_naive(input, weight, out_naive, stride, stride, pad, pad);
            memset(out_sconv.data.data(), 0, out_sconv.size_bytes());
            conv2d_sconv(input, weight, out_sconv, stride, stride, pad, pad, Nwin, Nf, threads);
            ok = verify(out_naive, out_sconv);
            if (ok) n_pass++;
        }

        // Warmup (sconv only; skip naive when --skip-naive)
        for (int w = 0; w < warmup; w++) {
            if (!skip_naive)
                conv2d_naive(input, weight, out_naive, stride, stride, pad, pad);
            memset(out_sconv.data.data(), 0, out_sconv.size_bytes());
            conv2d_sconv(input, weight, out_sconv, stride, stride, pad, pad, Nwin, Nf, threads);
        }

        // Time naive
        double t_naive = 0.0;
        if (!skip_naive) {
            double t0 = rtclock();
            for (int r = 0; r < runs; r++)
                conv2d_naive(input, weight, out_naive, stride, stride, pad, pad);
            t_naive = (rtclock() - t0) / runs * 1000.0; // ms
        }

        // Time sconv
        double t0 = rtclock();
        for (int r = 0; r < runs; r++) {
            memset(out_sconv.data.data(), 0, out_sconv.size_bytes());
            conv2d_sconv(input, weight, out_sconv, stride, stride, pad, pad, Nwin, Nf, threads);
        }
        double t_sconv = (rtclock() - t0) / runs * 1000.0;

        double naive_gf = flops / (t_naive * 1e6);
        double sconv_gf = flops / (t_sconv * 1e6);
        double speedup = t_naive / t_sconv;

        const char* status = ok ? "PASS" : "FAIL";
        if (skip_naive)
            printf("%-16s %8.2fG %12s %12.3f %10s %10.2f %9s  [%s]\n",
                   c.name, flops/1e9, "--", t_sconv, "--", sconv_gf, "--", status);
        else
            printf("%-16s %8.2fG %12.3f %12.3f %10.2f %10.2f %9.2fx  [%s]\n",
                   c.name, flops/1e9, t_naive, t_sconv, naive_gf, sconv_gf, speedup, status);
    }

    if (skip_naive)
        printf("\nCorrectness: skipped (--skip-naive)\n");
    else
        printf("\nCorrectness: %d/%d passed\n", n_pass, n_total);
    return 0;
}
