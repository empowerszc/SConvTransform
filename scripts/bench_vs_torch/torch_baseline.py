#!/usr/bin/env python3
"""PyTorch conv2d baseline for the native_impl benchmark configs.

Prints machine-readable lines:  name,flops,ms,gflops
Used by compare_vs_pytorch.py; can also be run standalone:
    python3 torch_baseline.py --threads 32 --runs 8 --warmup 3
"""
import argparse, time
import torch


def rtclock():
    return time.perf_counter()


configs = [
    ("custom_0",  4, 192,  40,  40, 192, 3, 3, 1, 1),
    ("custom_1",  4, 768,  80,  80, 768, 3, 3, 2, 1),
    ("custom_2",  4,  96,  80,  80,  96, 3, 3, 1, 1),
    ("custom_3",  4, 768,  40,  40, 768, 3, 3, 2, 1),
    ("custom_4",  4, 384, 160, 160, 384, 3, 3, 2, 1),
    ("custom_5",  4,  48, 160, 160,  48, 3, 3, 1, 1),
    ("custom_6",  4,  96, 320, 320, 192, 3, 3, 2, 1),
    ("custom_7",  4, 192,  20,  20, 192, 3, 3, 1, 1),
    ("custom_8",  4, 384,  80,  80, 384, 3, 3, 2, 1),
    ("custom_9",  4, 384,  80,  80,  96, 3, 3, 1, 1),
    ("custom_10", 4, 768,  40,  40,  96, 3, 3, 1, 1),
    ("common_r18_56",   1,  64,  56, 56,  64, 3, 3, 1, 1),
    ("common_r18_28",   1, 128,  28, 28, 128, 3, 3, 1, 1),
    ("common_r18_14",   1, 256,  14, 14, 256, 3, 3, 1, 1),
    ("common_r18_7",    1, 512,   7,  7, 512, 3, 3, 1, 1),
    ("common_r50_ds56", 1, 128,  56, 56, 256, 3, 3, 2, 1),
    ("common_r50_ds28", 1, 256,  28, 28, 512, 3, 3, 2, 1),
    ("common_thr_n8",   8,  64,  56, 56,  64, 3, 3, 1, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=38)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    print(f"#torch {torch.__version__} threads={args.threads}")
    for name, N, Ic, Ih, Iw, Oc, Fh, Fw, stride, pad in configs:
        Oh = (Ih + 2 * pad - Fh) // stride + 1
        Ow = (Iw + 2 * pad - Fw) // stride + 1
        flops = 2.0 * N * Oc * Oh * Ow * Ic * Fh * Fw
        x = torch.randn(N, Ic, Ih, Iw)
        w = torch.randn(Oc, Ic, Fh, Fw)
        with torch.no_grad():
            for _ in range(args.warmup):
                torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)
            t0 = rtclock()
            for _ in range(args.runs):
                torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)
            dt = (rtclock() - t0) / args.runs * 1000.0
        print(f"{name},{flops:.0f},{dt:.4f},{flops / (dt * 1e6):.2f}")
        del x, w


if __name__ == "__main__":
    main()
