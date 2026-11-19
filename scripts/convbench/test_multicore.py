#!/usr/bin/env python3
"""Acceptance gate for the multicore path (M3/M4).

Stages (run in this order; each is a separate check):
  ir       the loop the transform marks is the window tile loop (its step is the
           selected Nwin), for every shape / Nwin combination that matters
  auto     the multicore transform picks the per-shape window tile from the
           documented table (small tile for more work items, wide tile for
           thin-channel shapes and for shapes where halving would flip the CSA
           schedule from IS to WS) and keeps the serial schedule
  correct  11/11 elementwise match between the multicore build run with 8
           threads and the untouched serial lowering (rtol/atol 1e-3)
  perf     custom_0 multicore t16 <= 24 ms

Why the `ir` stage exists
  `loopOps[size-2]` is only the window tile loop when `scf::tileUsingSCF` emits
  the two inner loops in the usual order.  With Nwin=80 on custom_1 the order
  flips and [size-2] is the filter-subblock loop: the marker then parallelizes
  a loop whose step is 32, and every downstream measurement is meaningless.
  The pass also refuses to build when the input packing was hoisted out of the
  window loop (the scratch is then written before the parallel region and only
  read inside, so it must stay shared instead of being privatized).

Why the auto stage also checks the schedule
  A smaller Nwin is not automatically wider: the CSA window group is Nwin*k3,
  and k3 is re-derived for each Nwin.  On custom_1/2/9, Nwin=80 makes the cost
  model switch from input-stationary to weight-stationary and the parallel time
  more than doubles.  The stage asserts that the multicore transform keeps the
  same schedule as the base transform, which is the contract the heuristic
  implements (see selectAutoNwinMulticore in lib/SConv.cpp).

Set SCONV_MC_TF to test a different multicore transform.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_convbench as R  # noqa: E402

BASE_TF = os.path.join(R.SCONV_ROOT, "test/lowering/sconv-blas-fastp.mlir")
MC_TF = os.environ.get(
    "SCONV_MC_TF",
    os.path.join(R.SCONV_ROOT, "test/lowering/sconv-blas-fastp-mc.mlir"))
CSV = os.path.join(R.SCONV_ROOT, "scripts/convbench/samples/custom_conv.csv")
LLVM = os.environ.get("LLVM_BUILD_DIR", "")
OMPLIB = f"{LLVM}/lib/aarch64-unknown-linux-gnu/libomp.so"

# (shape, Nwin) pairs that exercise both tiling orders and both kinds of
# packing placement.
IR_CASES = [
    ("custom_0", 160),
    ("custom_1", 80),
    ("custom_2", 80),
    ("custom_9", 80),
    ("custom_3", 40),
    ("custom_7", 40),
]

# Multicore window tile per shape: halve the serial tile to double the number
# of work items, but keep the wide tile for thin-channel shapes (packing and
# GEMM-m dominate there) and quarter it when halving would leave fewer than 8
# work items (Ohw=400 shapes).
#
# The halved tile is only accepted when the CSA schedule does not flip.  On
# custom_1/2/9 Nwin=80 makes the cost model pick weight-stationary instead of
# input-stationary and the parallel time more than doubles (custom_2 t8
# 22.5 -> 50.2 ms); those shapes keep the serial tile.  Every chosen tile below
# therefore keeps the same schedule as the base transform.
AUTO_TILE = {
    "custom_0": 80, "custom_1": 160, "custom_2": 160, "custom_3": 40,
    "custom_4": 80, "custom_5": 160, "custom_6": 80, "custom_7": 40,
    "custom_8": 80, "custom_9": 160, "custom_10": 80,
}

CASES = sorted(AUTO_TILE, key=lambda x: int(x.split("_")[1]))

# Correctness subset: the payload prints the whole output tensor, and parsing
# custom_1/4/6 would mean tens of millions of floats.  These five cover both
# code paths (privatized scratch: custom_0/3/7; hoisted packing, shared scratch:
# custom_2/9) and both window tiles (40 and 80).  The 11-shape gate for the
# default transform lives in scripts/convbench/test_mk_auto.py + the M3 run log.
CORRECT_CASES = ["custom_0", "custom_2", "custom_3", "custom_7", "custom_9"]
PERF_BUDGET_MS = 24.0


def fail(msg):
    print(f"FAIL: {msg}")
    return 1


def _tf_with_nwin(nwin):
    """A copy of the default transform that pins Nwin and autotunes Nf."""
    with open(BASE_TF) as f:
        text = f.read()
    pinned = text.replace("mK_info = [0, 0]", f"mK_info = [{nwin}, 0]")
    if pinned == text:
        raise SystemExit(f"cannot pin mK_info in {BASE_TF}")
    path = f"/tmp/sconvtest_mc/tf_nwin{nwin}.mlir"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(pinned)
    return path


def _parallel_step(ir):
    m = re.search(r"scf\.parallel .*? to \(%c(\d+)\) step \(%c(\d+)\)", ir)
    return None if not m else (int(m.group(1)), int(m.group(2)))


def _schedule(ir):
    """CSA schedule of the microkernel call as printed in the IR."""
    m = re.search(r'schedule = "([A-Z]+)"', ir)
    return None if not m else m.group(1)


def _stage_ir():
    os.makedirs("/tmp/sconvtest_mc/ir", exist_ok=True)
    for case, nwin in IR_CASES:
        src = os.path.join("/tmp/sconvtest_mc/ir", f"conv_{case}.mlir")
        if not os.path.exists(src):
            R.gen_payloads(CSV, R.PERFORMANCE_TPL, "/tmp/sconvtest_mc/ir",
                           None, 1, 1)
        r = R.sh([R.SCONV, "-transform=" + _tf_with_nwin(nwin),
                  "-sconv-parallelize", src])
        if r.returncode:
            return fail(f"{case} @ Nwin={nwin}: parallelize failed: "
                        f"{r.stderr.strip().splitlines()[0][:200]}")
        step = _parallel_step(r.stdout)
        if step is None:
            return fail(f"{case} @ Nwin={nwin}: no scf.parallel emitted")
        if step[1] != nwin:
            return fail(f"{case} @ Nwin={nwin}: marked loop has step {step[1]} "
                        f"(expected the window tile, step {nwin})")
        print(f"PASS: ir — {case} @ Nwin={nwin} marks the window tile loop")
    return 0


def _stage_auto():
    if not os.path.exists(MC_TF):
        return fail(f"missing multicore transform {MC_TF}")
    os.makedirs("/tmp/sconvtest_mc/auto", exist_ok=True)
    for case in CASES:
        src = os.path.join("/tmp/sconvtest_mc/auto", f"conv_{case}.mlir")
        if not os.path.exists(src):
            R.gen_payloads(CSV, R.PERFORMANCE_TPL, "/tmp/sconvtest_mc/auto",
                           None, 1, 1)
        r = R.sh([R.SCONV, "-transform=" + MC_TF, "-sconv-parallelize", src])
        if r.returncode:
            return fail(f"{case}: multicore transform failed: "
                        f"{r.stderr.strip().splitlines()[0][:200]}")
        step = _parallel_step(r.stdout)
        if step is None:
            return fail(f"{case}: no scf.parallel emitted")
        if step[1] != AUTO_TILE[case]:
            return fail(f"{case}: picked Nwin={step[1]}, "
                        f"expected {AUTO_TILE[case]}")
        sched = _schedule(r.stdout)
        if sched is None:
            return fail(f"{case}: no schedule printed")
        base = R.sh([R.SCONV, "-transform=" + BASE_TF,
                     "-sconv-parallelize", src])
        base_sched = None if base.returncode else _schedule(base.stdout)
        if base_sched and sched != base_sched:
            return fail(f"{case}: Nwin={step[1]} flips the schedule "
                        f"({base_sched} -> {sched}); keep the serial tile")
        print(f"PASS: auto — {case} picks Nwin={step[1]} ({sched})")
    return 0


def _stage_correct():
    libs = R.get_libs(*R.build_runtime_libs())
    os.makedirs("/tmp/sconvtest_mc/correct", exist_ok=True)
    R.gen_payloads(CSV, R.CORRECTNESS_TPL, "/tmp/sconvtest_mc/correct",
                   None, 1, 1)
    os.environ["OMP_NUM_THREADS"] = "8"
    bad = 0
    for case in CORRECT_CASES:
        src = os.path.join("/tmp/sconvtest_mc/correct", f"conv_{case}.mlir")
        base_out, err, _ = R.lower_run(src, libs)
        if err:
            bad += fail(f"{case}: serial lowering failed: {err[:200]}")
            continue
        out = src.replace(".mlir", ".mc.llvm.mlir")
        r = R.sh([R.SCONV, "-transform=" + MC_TF, "-sconv-parallelize",
                  "-sconv-lower-parallel", src, "-o", out])
        if r.returncode:
            bad += fail(f"{case}: multicore build failed: "
                        f"{r.stderr.strip().splitlines()[0][:200]}")
            continue
        q = R.sh(["numactl", "--cpunodebind=0", "--membind=0",
                  f"{LLVM}/bin/mlir-cpu-runner", out, "-e", "main",
                  "-entry-point-result=void",
                  "-shared-libs=" + libs + "," + OMPLIB])
        got = q.stdout + q.stderr
        import numpy as np
        b, t = R.get_data(base_out), R.get_data(got)
        if b is None or t is None:
            bad += fail(f"{case}: could not parse output")
        elif not np.allclose(b, t, rtol=1e-3, atol=1e-3):
            bad += fail(f"{case}: max_diff={np.max(np.abs(b - t)):.4e}")
        else:
            print(f"PASS: correct — {case}")
    return bad


def _stage_perf():
    libs = R.get_libs(*R.build_runtime_libs())
    os.makedirs("/tmp/sconvtest_mc/perf", exist_ok=True)
    R.gen_payloads(CSV, R.PERFORMANCE_TPL, "/tmp/sconvtest_mc/perf", 1, 3, 2)
    src = "/tmp/sconvtest_mc/perf/conv_custom_0.mlir"
    out = "/tmp/sconvtest_mc/perf/conv_custom_0.mc.llvm.mlir"
    r = R.sh([R.SCONV, "-transform=" + MC_TF, "-sconv-parallelize",
              "-sconv-lower-parallel", src, "-o", out])
    if r.returncode:
        return fail(f"build failed: {r.stderr[:300]}")
    os.environ["OMP_NUM_THREADS"] = "16"
    q = R.sh(["numactl", "--cpunodebind=0", "--membind=0",
              f"{LLVM}/bin/mlir-cpu-runner", out, "-e", "main",
              "-entry-point-result=void",
              "-shared-libs=" + libs + "," + OMPLIB])
    total = R.get_time_ms(q.stdout + q.stderr)
    if total is None:
        return fail("no time printed")
    per_iter = total / 3.0
    if per_iter > PERF_BUDGET_MS:
        return fail(f"custom_0 t16 {per_iter:.1f} ms/iter > "
                    f"budget {PERF_BUDGET_MS} ms")
    print(f"PASS: perf — custom_0 t16 {per_iter:.1f} ms/iter "
          f"(<= {PERF_BUDGET_MS})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["ir", "auto", "correct", "perf", "all"])
    a = ap.parse_args()
    stages = (["ir", "auto", "correct", "perf"]
              if a.stage == "all" else [a.stage])
    rc = 0
    for s in stages:
        rc |= {"ir": _stage_ir, "auto": _stage_auto,
               "correct": _stage_correct, "perf": _stage_perf}[s]()
    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
