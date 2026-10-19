#!/usr/bin/env python3
"""Compare this project's convolution against PyTorch conv2d.

Three variants are collected (all GFLOPS, per thread count):

  torch  : torch.nn.functional.conv2d baseline (via torch_baseline.py)
  after  : native_impl build (parallel Bc prepack + adaptive (Nwin,Nf));
           custom_9/custom_10 use the SME end-to-end bench (sme/e2e.out, incl.
           pad/pack/unpad) when available, else the kernel-only probe.
  before : OPTIONAL pre-optimization build (--before-binary); omitted if not given.

Results are written to <out>/data.json as
    { "torch": {t: {name: gf}}, "after": {...}, "before": {...} }

Example:
    export SCONV_ROOT=/path/to/SConvTransform
    export TORCH_PYTHON=/path/to/envs/torch210_env/bin/python
    python3 scripts/bench_vs_torch/compare_vs_pytorch.py \
        --before-binary /tmp/native_common/build/sconv_bench \
        --threads 1,2,4,8,16,32,36
"""
import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SME_CASES = {"custom_9", "custom_10"}


def run(cmd, env, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    return r.stdout + r.stderr


def parse_bench(out):
    """native_impl 'Conv ... GFLOPS ... [PASS]' table -> {name: gflops}."""
    v = {}
    for l in out.splitlines():
        tk = l.split()
        if len(tk) >= 8 and tk[0].startswith(("custom_", "common_")) and tk[-1] == "[PASS]":
            try:
                gf = float(tk[5])
            except ValueError:
                continue
            if gf > v.get(tk[0], 0.0):
                v[tk[0]] = gf
    return v


def parse_torch(out):
    v = {}
    for l in out.splitlines():
        if l.startswith("#") or "," not in l:
            continue
        p = l.split(",")
        if len(p) == 4:
            v[p[0]] = float(p[3])
    return v


def parse_sme(out):
    """sme probe 'name ... GFLOPS' -> {name: gflops}."""
    v = {}
    for l in out.splitlines():
        if "GFLOPS" in l and "parallel" not in l:
            tk = l.split()
            idx = [i for i, x in enumerate(tk) if x == "GFLOPS"]
            if idx:
                v[tk[0]] = float(tk[idx[-1] - 1])
    return v


def merge_best(dst, src):
    for k, val in src.items():
        if val > dst.get(k, 0.0):
            dst[k] = val


def main():
    root = os.environ.get("SCONV_ROOT", os.path.dirname(os.path.dirname(HERE)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=root)
    ap.add_argument("--after-binary", default=None,
                    help="default $root/native_impl/build/sconv_bench")
    ap.add_argument("--before-binary", default=None,
                    help="optional pre-optimization sconv_bench")
    ap.add_argument("--sme-probe", default=None, help="default $root/sme/direct.out")
    ap.add_argument("--sme-e2e", default=None,
                    help="SME end-to-end bench (incl. pad/pack/unpad); default $root/sme/e2e.out")
    ap.add_argument("--torch-python", default=os.environ.get("TORCH_PYTHON", "python3"))
    ap.add_argument("--torch-script", default=os.path.join(HERE, "torch_baseline.py"))
    ap.add_argument("--threads", default="1,2,4,8,16,32,36")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3, help="best-of-N repeats per cell")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--numa", default="0", help="numactl node, or '' to disable")
    args = ap.parse_args()

    after = args.after_binary or os.path.join(args.root, "native_impl/build/sconv_bench")
    sme = args.sme_probe or os.path.join(args.root, "sme/direct.out")
    sme_e2e = args.sme_e2e or os.path.join(args.root, "sme/e2e.out")
    sme_bin = sme_e2e if os.path.exists(sme_e2e) else sme
    print(f"  sme: {sme_bin} ({'end-to-end' if sme_bin == sme_e2e else 'kernel-only'})", flush=True)
    threads = [int(x) for x in args.threads.split(",")]
    numa = ["numactl", f"--cpunodebind={args.numa}", f"--membind={args.numa}"] if args.numa else []
    os.makedirs(args.out, exist_ok=True)

    env = os.environ.copy()
    kml = env.get("KMLROOT", "")
    if kml:
        env["LD_LIBRARY_PATH"] = f"{kml}/lib:" + env.get("LD_LIBRARY_PATH", "")

    def bench(binary, t, nw, nf):
        out = {}
        for filt in ("custom", "common"):
            cmd = [*numa, binary, "--filter", filt, "--skip-naive",
                   "--runs", str(args.runs), "--warmup", str(args.warmup),
                   "--nwin", str(nw), "--nf", str(nf), "--threads", str(t)]
            merge_best(out, parse_bench(run(cmd, env)))
        return out

    res = {"torch": {}, "after": {}, "before": {}}
    for t in threads:
        print(f"[threads={t}]", flush=True)
        tv, gb, ga, sv = {}, {}, {}, {}
        for _ in range(args.repeats):
            merge_best(tv, parse_torch(run(
                [*numa, args.torch_python, args.torch_script,
                 "--threads", str(t), "--runs", str(args.runs), "--warmup", str(args.warmup)], env)))
            merge_best(ga, bench(after, t, 0, 0))
            if args.before_binary:
                for nw, nf in ((32, 16), (64, 32)):
                    merge_best(gb, bench(args.before_binary, t, nw, nf))
            if os.path.exists(sme_bin):
                merge_best(sv, parse_sme(run([*numa, sme_bin, "bench", str(t)], env,
                                             cwd=os.path.dirname(sme_bin))))
        for name in SME_CASES:
            if name in sv:
                ga[name] = sv[name]
                if args.before_binary:
                    gb[name] = sv[name]
        res["torch"][t] = tv
        res["after"][t] = ga
        res["before"][t] = gb
    path = os.path.join(args.out, "data.json")
    json.dump(res, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
