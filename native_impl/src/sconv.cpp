// sconv.cpp — Standalone SConv convolution implementation
// Implements: naive conv, CSA-guided tiling + packing + OpenBLAS microkernel
//
// Build: cmake .. -DCMAKE_BUILD_TYPE=Release && make
// Run:   ./sconv_bench [--csv file] [--runs N] [--warmup N]

#include "sconv_csa.h"
#include <vector>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#ifdef USE_OPENBLAS
#include <cblas.h>
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
                  int Nwin, int Nf) {
    int N = input.N, Ic = input.C, Ih = input.H, Iw = input.W;
    int Oc = weight.N, Fh = weight.H, Fw = weight.W;
    int Oh = output.H, Ow = output.W;
    int Ohw = Oh * Ow;  // collapsed spatial
    int K = 0; // = Nc * Fh * Fw, computed per Nc

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
    int tCH = (Ic + Nc - 1) / Nc;
    int K2 = strat.k2;
    int K3 = strat.k3;
    bool is_IS = (strat.schd == IS);

    // Allocate packed buffers
    // Packed input: [Nc*Fh*Fw, Nwin] = [K, M]  (reused per tile)
    // Packed filter: [Nc*Fh*Fw, Nf]  = [K, N]
    K = Nc * Fh * Fw;
    std::vector<float> packedA((size_t)K * Nwin);
    std::vector<float> packedB((size_t)K * Nf);
    std::vector<float> packedC((size_t)Nf * Nwin);

    int nWinTiles = (Ohw + Nwin - 1) / Nwin;
    int nFltTiles = (Oc + Nf - 1) / Nf;

    for (int n = 0; n < N; n++) {
        for (int ic_start = 0; ic_start < Ic; ic_start += Nc) {
            int nc = std::min(Nc, Ic - ic_start);
            int k_dim = nc * Fh * Fw;

            // Pack filter tile for this channel range
            // [nc, Fh, Fw, Nf] → packed [k_dim, Nf]
            for (int nf_start = 0; nf_start < Oc; nf_start += Nf) {
                int nf = std::min(Nf, Oc - nf_start);

                // Outer loop: K2 groups (filter tiles), inner: K3 groups (window tiles)
                // For simplicity, process all tiles sequentially (CSA guides ordering in MLIR)
                for (int wt = 0; wt < nWinTiles; wt++) {
                    int nwin = std::min(Nwin, Ohw - wt * Nwin);

                    // Pack input: [nc, Fh, Fw, nwin] → [k_dim, nwin]
                    for (int ic = 0; ic < nc; ic++)
                    for (int fh = 0; fh < Fh; fh++)
                    for (int fw = 0; fw < Fw; fw++)
                    for (int w = 0; w < nwin; w++) {
                        int ohw = wt * Nwin + w;
                        int oh = ohw / Ow;
                        int ow = ohw % Ow;
                        int ih = oh * strideH + fh - padTop;
                        int iw = ow * strideW + fw - padLeft;
                        float val = 0.0f;
                        if (ih >= 0 && ih < Ih && iw >= 0 && iw < Iw)
                            val = input.at(n, ic_start + ic, ih, iw);
                        // packedA[k_idx][w] where k_idx = (ic*Fh+fh)*Fw+fw
                        packedA[((size_t)(ic * Fh + fh) * Fw + fw) * nwin + w] = val;
                    }

                    // Pack filter: [nc, Fh, Fw, nf] → [k_dim, nf]
                    for (int ic = 0; ic < nc; ic++)
                    for (int fh = 0; fh < Fh; fh++)
                    for (int fw = 0; fw < Fw; fw++)
                    for (int f = 0; f < nf; f++) {
                        packedB[((size_t)(ic * Fh + fh) * Fw + fw) * nf + f] =
                            weight.at(nf_start + f, ic_start + ic, fh, fw);
                    }

                    // Microkernel: C[nf, nwin] += B[k_dim, nf]^T * A[k_dim, nwin]
                    memset(packedC.data(), 0, (size_t)nf * nwin * sizeof(float));
#ifdef USE_OPENBLAS
                    // C = B^T * A + C  (our wrapper: same as SConv's sgemm_blas_kernel)
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                nf, nwin, k_dim,
                                1.0f,
                                packedB.data(), nf,    // B is [k_dim, nf], ldb=nf
                                packedA.data(), nwin,   // A is [k_dim, nwin], lda=nwin
                                1.0f,
                                packedC.data(), nwin); // C is [nf, nwin], ldc=nwin
#else
                    // Fallback: naive GEMM (no OpenBLAS)
                    for (int fi = 0; fi < nf; fi++)
                    for (int wi = 0; wi < nwin; wi++) {
                        float s = 0.0f;
                        for (int ki = 0; ki < k_dim; ki++)
                            s += packedB[ki * nf + fi] * packedA[ki * nwin + wi];
                        packedC[fi * nwin + wi] = s;
                    }
#endif
                    // Unpack C → output
                    for (int f = 0; f < nf; f++)
                    for (int w = 0; w < nwin; w++) {
                        int ohw = wt * Nwin + w;
                        int oh = ohw / Ow;
                        int ow = ohw % Ow;
                        output.at(n, nf_start + f, oh, ow) += packedC[f * nwin + w];
                    }
                } // window tiles
            } // filter tiles
        } // channel tiles
    } // batch
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
    // Small configs (batch=1, for quick verification)
    {"small_0",   1,18,14,14,  144,3,3, 7,7,     2,1},
    {"small_1",   1,48,10,10,   64,2,2, 5,5,     2,0},
};

int main(int argc, char** argv) {
    int runs = 3, warmup = 1;
    const char* filter = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i+1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i+1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--filter") == 0 && i+1 < argc) filter = argv[++i];
    }

    int Nwin = 16, Nf = 8;
    bool has_blas =
#ifdef USE_OPENBLAS
        true;
#else
        false;
#endif

    printf("SConv Benchmark (Nwin=%d, Nf=%d, OpenBLAS=%s)\n", Nwin, Nf, has_blas ? "YES" : "NO");
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
        conv2d_naive(input, weight, out_naive, stride, stride, pad, pad);
        memset(out_sconv.data.data(), 0, out_sconv.size_bytes());
        conv2d_sconv(input, weight, out_sconv, stride, stride, pad, pad, Nwin, Nf);

        bool ok = verify(out_naive, out_sconv);
        if (ok) n_pass++;

        // Warmup
        for (int w = 0; w < warmup; w++) {
            conv2d_naive(input, weight, out_naive, stride, stride, pad, pad);
            memset(out_sconv.data.data(), 0, out_sconv.size_bytes());
            conv2d_sconv(input, weight, out_sconv, stride, stride, pad, pad, Nwin, Nf);
        }

        // Time naive
        double t0 = rtclock();
        for (int r = 0; r < runs; r++)
            conv2d_naive(input, weight, out_naive, stride, stride, pad, pad);
        double t_naive = (rtclock() - t0) / runs * 1000.0; // ms

        // Time sconv
        t0 = rtclock();
        for (int r = 0; r < runs; r++) {
            memset(out_sconv.data.data(), 0, out_sconv.size_bytes());
            conv2d_sconv(input, weight, out_sconv, stride, stride, pad, pad, Nwin, Nf);
        }
        double t_sconv = (rtclock() - t0) / runs * 1000.0;

        double naive_gf = flops / (t_naive * 1e6);
        double sconv_gf = flops / (t_sconv * 1e6);
        double speedup = t_naive / t_sconv;

        const char* status = ok ? "PASS" : "FAIL";
        printf("%-16s %8.2fG %12.3f %12.3f %10.2f %10.2f %9.2fx  [%s]\n",
               c.name, flops/1e9, t_naive, t_sconv, naive_gf, sconv_gf, speedup, status);
    }

    printf("\nCorrectness: %d/%d passed\n", n_pass, n_total);
    return 0;
}
