#!/usr/bin/env python3
"""Acceptance test for the shape-based mK autotuner (stage A).

`mK_info = [0, 0]` means "auto": the compiler picks

    Nwin = largest of {64, 32, 16} that divides Oh*Ow (and is <= Oh*Ow), else 16
    Nf   = largest of {32, 16, 8} that divides the filter count, else 8

The whole point is that the choice must be *equivalent* to passing those tiles
explicitly, so the strongest check is an exact IR comparison: the transform
script differs only in the attribute; the emitted IR must be identical.

Why largest divisors: the tuned [64, 32] tile is only fast when it divides the
shape. Otherwise the pipeline falls into the edge path and regresses badly
(custom_5: 229 -> 6227 ms with Nf=32 against fn=48; custom_7: 36 -> 216 ms with
Nwin=64 against Ohw=400). Choosing the largest divisor keeps the wide tile on
regular shapes and degrades gracefully on irregular ones.

Usage:
  python3 scripts/convbench/test_mk_auto.py [--stage ir|perf|all]
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_convbench as R  # noqa: E402

CSV = os.path.join(R.SCONV_ROOT, "scripts/convbench/samples/custom_conv.csv")
BASE_TF = os.path.join(R.SCONV_ROOT, "test/lowering/sconv-blas-fastp.mlir")
SENTINEL = "      { mK_info = [0, 0] }\n"

# Expected choice per case, derived from the shape table and stated
# independently of the implementation (do not import the heuristic here).
EXPECTED = {
    "custom_0": (64, 32),   # Ohw=1600, fn=192
    "custom_3": (16, 32),   # Ohw=400,  fn=768  (400 % 32 != 0)
    "custom_5": (64, 16),   # Ohw=25600, fn=48  (48 % 32 != 0)
    "custom_7": (16, 32),   # Ohw=400,  fn=192
}
PERF_CASE = "custom_0"
# Auto should land on [64,32] and ~86 ms/iter; 105 leaves room for machine noise
# while still failing loudly if the heuristic silently falls back to [16,8].
PERF_BUDGET_MS = 105.0


def fail(msg):
    print(f"FAIL: {msg}")
    return 1


def _write_tf(path, mK):
    src = open(BASE_TF).read()
    if SENTINEL not in src:
        raise SystemExit(f"{BASE_TF}: anchor {SENTINEL!r} not found")
    # Replace the auto sentinel with the explicit tile. The rest of the script
    # is identical, so the emitted IR must be identical too.
    src = src.replace(
        SENTINEL, f"      {{ mK_info = [{mK[0]}, {mK[1]}] }}\n", 1)
    with open(path, "w") as f:
        f.write(src)


def _payloads(tag, num, runs, warmup):
    tmp = f"/tmp/sconvtest_mk_auto/{tag}"
    R.gen_payloads(CSV, R.PERFORMANCE_TPL, tmp, num, runs, warmup)
    return tmp


def stage_ir():
    tmp = _payloads("ir", None, 1, 1)
    tf_dir = "/tmp/sconvtest_mk_auto/tf"
    os.makedirs(tf_dir, exist_ok=True)
    bad = 0
    for case, mk in EXPECTED.items():
        src = os.path.join(tmp, f"conv_{case}.mlir")
        outs = {}
        for tag, tile in (("auto", (0, 0)), ("explicit", mk)):
            tf = os.path.join(tf_dir, f"{case}_{tag}.mlir")
            _write_tf(tf, tile)
            r = R.sh([R.SCONV, "-transform=" + tf, src])
            if r.returncode:
                bad += fail(f"{case}/{tag}: sconv-opt failed:\n{r.stderr[:500]}")
                continue
            outs[tag] = r.stdout
        if len(outs) != 2:
            continue
        if "pack_input_tile" not in outs["auto"]:
            bad += fail(f"{case}: auto IR has no pack_input_tile call")
            continue
        if 'packing = "input"' in outs["auto"]:
            bad += fail(f"{case}: auto IR left the input-packing generic")
            continue
        if outs["auto"] != outs["explicit"]:
            bad += fail(f"{case}: auto IR != explicit {list(mk)} IR")
            continue
        print(f"PASS: ir — {case} auto == explicit [{mk[0]}, {mk[1]}]")
    return bad


def stage_perf():
    tmp = _payloads("perf", 1, 3, 2)
    libs = R.get_libs(*R.build_runtime_libs())
    tf = "/tmp/sconvtest_mk_auto/tf/custom_0_auto_perf.mlir"
    os.makedirs(os.path.dirname(tf), exist_ok=True)
    _write_tf(tf, (0, 0))
    src = os.path.join(tmp, f"conv_{PERF_CASE}.mlir")
    r = R.sh([R.SCONV, "-transform=" + tf, src])
    if r.returncode:
        return fail(f"transform failed:\n{r.stderr[:500]}")
    lowered = src[:-5] + "_auto.mlir"
    with open(lowered, "w") as f:
        f.write(r.stdout)
    out, err, _ = R.lower_run(lowered, libs)
    os.path.exists(lowered) and os.unlink(lowered)
    if err:
        return fail(f"run failed: {err[:300]}")
    total = R.get_time_ms(out)
    if total is None:
        return fail("no time printed")
    per_iter = total / 3.0
    if per_iter > PERF_BUDGET_MS:
        return fail(f"{PERF_CASE} {per_iter:.1f} ms/iter > {PERF_BUDGET_MS} "
                    f"(heuristic likely fell back to a narrow tile)")
    print(f"PASS: perf — {PERF_CASE} {per_iter:.1f} ms/iter "
          f"(<= {PERF_BUDGET_MS})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["ir", "perf", "all"])
    a = ap.parse_args()
    stages = ["ir", "perf"] if a.stage == "all" else [a.stage]
    rc = 0
    for s in stages:
        rc |= {"ir": stage_ir, "perf": stage_perf}[s]()
    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
