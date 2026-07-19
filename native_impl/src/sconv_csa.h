#ifndef SCONV_CSA_H
#define SCONV_CSA_H

#include <cstdint>
#include <cmath>
#include <algorithm>

enum Scheduling { IS = 1, WS };

struct ConvInfo {
    int64_t input_channels;
    int64_t input_cols;
    int64_t output_rows;
    int64_t output_cols;
    int64_t kernel_rows;
    int64_t kernel_cols;
    int64_t num_filters;
    int64_t split_size_windows;
    int64_t split_size_filters;
    uint8_t data_size; // bytes -- 4 (float32)
};

struct ArchInfo {
    uint32_t l1_size;
    uint32_t l2_size;
    uint32_t l3_size;
    uint32_t l1_latency;
    uint32_t l2_latency;
    uint32_t l3_latency;
    uint32_t mem_latency;
    uint32_t cache_line;
};

struct mKInfo {
    uint8_t nwindows;
    uint8_t num_filters;
    uint16_t noutput;
};

struct CSAStrategy {
    Scheduling schd;
    uint32_t k2;
    uint32_t extra_k2;
    uint32_t k3;
    uint32_t extra_k3;
    uint32_t tile_c;
    uint32_t extra_tile_c;
};

// ---- CSA (ported from lib/CSA.cpp) ----

class CSA {
public:
    CSA(const ArchInfo &arch, const ConvInfo &conv, mKInfo mK)
        : arch_(arch), conv_(conv), mK_(mK) {}

    CSAStrategy run() {
        in_size_  = mK_.nwindows * conv_.kernel_rows * conv_.kernel_cols * conv_.data_size;
        w_size_   = mK_.num_filters * conv_.kernel_rows * conv_.kernel_cols * conv_.data_size;
        out_size_ = mK_.noutput * conv_.data_size;

        // Nc: L1 constraint
        tile_c_ = halfHeuristic(conv_.input_channels, &CSA::tileSizeL1, arch_.l1_size);
        if (tile_c_ == 0) tile_c_ = 1;

        in_size_  *= tile_c_;
        w_size_   *= tile_c_;

        tCH_ = conv_.input_channels / tile_c_;
        extra_tCH_ = conv_.input_channels % tile_c_;

        in_tiles_per_tch_ = (uint32_t)((conv_.output_rows * conv_.output_cols) / (double)mK_.nwindows);
        w_tiles_per_tch_  = (uint32_t)(conv_.num_filters / (double)mK_.num_filters);

        // IS and WS costs
        k2_is_ = halfHeuristic(w_tiles_per_tch_, &CSA::tileSizeL2_IS, arch_.l2_size);
        if (k2_is_ == 0) { k2_is_ = 1; extra_k2_is_ = 0; }
        else extra_k2_is_ = w_tiles_per_tch_ % k2_is_;

        k3_is_ = halfHeuristic(in_tiles_per_tch_, &CSA::tileSizeL3_IS, arch_.l3_size);
        if (k3_is_ == 0) { k3_is_ = 1; extra_k3_is_ = 0; }
        else extra_k3_is_ = in_tiles_per_tch_ % k3_is_;

        k2_ws_ = halfHeuristic(in_tiles_per_tch_, &CSA::tileSizeL2_WS, arch_.l2_size);
        if (k2_ws_ == 0) { k2_ws_ = 1; extra_k2_ws_ = 0; }
        else extra_k2_ws_ = in_tiles_per_tch_ % k2_ws_;

        k3_ws_ = halfHeuristic(w_tiles_per_tch_, &CSA::tileSizeL3_WS, arch_.l3_size);
        if (k3_ws_ == 0) { k3_ws_ = 1; extra_k3_ws_ = 0; }
        else extra_k3_ws_ = w_tiles_per_tch_ % k3_ws_;

        uint64_t cost_is = costModel_IS();
        uint64_t cost_ws = costModel_WS();

        CSAStrategy res;
        if (cost_ws > cost_is) {
            res.schd = IS;
            res.k2 = k2_is_; res.extra_k2 = extra_k2_is_;
            res.k3 = k3_is_; res.extra_k3 = extra_k3_is_;
        } else {
            res.schd = WS;
            res.k2 = k2_ws_; res.extra_k2 = extra_k2_ws_;
            res.k3 = k3_ws_; res.extra_k3 = extra_k3_ws_;
        }
        res.tile_c = tile_c_;
        res.extra_tile_c = extra_tCH_;
        return res;
    }

private:
    const ArchInfo &arch_;
    const ConvInfo &conv_;
    mKInfo mK_;
    uint32_t in_size_, w_size_, out_size_;
    uint32_t tile_c_, tCH_, extra_tCH_;
    uint32_t in_tiles_per_tch_, w_tiles_per_tch_;
    uint32_t k2_is_, extra_k2_is_, k3_is_, extra_k3_is_;
    uint32_t k2_ws_, extra_k2_ws_, k3_ws_, extra_k3_ws_;

    uint32_t halfHeuristic(uint32_t initial, uint32_t (CSA::*func)(uint32_t), uint32_t cache_size) {
        if (cache_size == 0) return 1;
        uint32_t sol = initial;
        while ((this->*func)(sol) > cache_size) {
            sol /= 2;
            if (sol == 0) return 1;
        }
        return sol;
    }

    // L1 tile size (same for IS and WS): in*Nc + w*Nc + out
    uint32_t tileSizeL1(uint32_t Nc) {
        return in_size_ * Nc + w_size_ * Nc + out_size_;
    }

    // IS: L2 holds filter tiles. L2 = in + K2*w + K2*out
    uint32_t tileSizeL2_IS(uint32_t k2) {
        return in_size_ + k2 * w_size_ + k2 * out_size_;
    }

    // IS: L3 holds input tiles. L3 = K3*in + K2*w + K2*K3*out
    uint32_t tileSizeL3_IS(uint32_t k3) {
        return k3 * in_size_ + k2_is_ * w_size_ + k2_is_ * k3 * out_size_;
    }

    // WS: L2 holds input tiles. L2 = K2*in + w + K2*out
    uint32_t tileSizeL2_WS(uint32_t k2) {
        return k2 * in_size_ + w_size_ + k2 * out_size_;
    }

    // WS: L3 holds filter tiles. L3 = K2*in + K3*w + K2*K3*out
    uint32_t tileSizeL3_WS(uint32_t k3) {
        return k2_ws_ * in_size_ + k3 * w_size_ + k2_ws_ * k3 * out_size_;
    }

    // Simplified cost model (focus on memory traffic)
    uint64_t costModel_IS() {
        uint64_t mem = (uint64_t)ceil((double)(in_tiles_per_tch_ * in_size_ * tCH_ +
                                               w_tiles_per_tch_ * w_size_ * tCH_) / arch_.cache_line);
        int w_fit = std::min((int)(w_tiles_per_tch_ / k2_is_) - 1, 1);
        int in_fit = (int)(in_tiles_per_tch_ / k3_is_) - 1;
        mem += tCH_ * (uint64_t)ceil((double)(w_fit * in_fit * w_tiles_per_tch_ * w_size_) / arch_.cache_line);
        uint64_t l3 = tCH_ * (uint64_t)ceil((double)(((w_tiles_per_tch_/k2_is_)-1) * in_tiles_per_tch_ * in_size_) / arch_.cache_line);
        uint64_t l2 = tCH_ * (uint64_t)ceil((double)((in_tiles_per_tch_-1) * w_tiles_per_tch_ * w_size_) / arch_.cache_line);
        if (arch_.l3_size == 0) mem += l3; // no L3 → goes to DRAM
        return l2 * arch_.l2_latency + (arch_.l3_size == 0 ? mem : l3) * (arch_.l3_size == 0 ? arch_.mem_latency : arch_.l3_latency) + mem * arch_.mem_latency;
    }

    uint64_t costModel_WS() {
        uint64_t mem = (uint64_t)ceil((double)(in_tiles_per_tch_ * in_size_ * tCH_ +
                                               w_tiles_per_tch_ * w_size_ * tCH_) / arch_.cache_line);
        int in_fit = std::min((int)(in_tiles_per_tch_ / k2_ws_) - 1, 1);
        int w_fit = (int)(w_tiles_per_tch_ / k3_ws_) - 1;
        mem += tCH_ * (uint64_t)ceil((double)(in_fit * w_fit * in_tiles_per_tch_ * in_size_) / arch_.cache_line);
        uint64_t l3 = tCH_ * (uint64_t)ceil((double)(((w_tiles_per_tch_/k3_ws_)-1) * in_tiles_per_tch_ * in_size_) / arch_.cache_line);
        uint64_t l2 = tCH_ * (uint64_t)ceil((double)((w_tiles_per_tch_-1) * in_tiles_per_tch_ * in_size_) / arch_.cache_line);
        if (arch_.l3_size == 0) mem += l3;
        return l2 * arch_.l2_latency + (arch_.l3_size == 0 ? mem : l3) * (arch_.l3_size == 0 ? arch_.mem_latency : arch_.l3_latency) + mem * arch_.mem_latency;
    }
};

#endif // SCONV_CSA_H
