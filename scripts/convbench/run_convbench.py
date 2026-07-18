#!/usr/bin/env python3
"""ConvBench test runner: correctness + performance comparison.

Usage:
  export LLVM_BUILD_DIR=/path/to/llvm-19.1.7
  export SCONV_ROOT=/path/to/SConvTransform

  # Run correctness + performance on small sample
  python3 scripts/convbench/run_convbench.py

  # Run on a custom CSV with performance payload
  python3 scripts/convbench/run_convbench.py --csv my_dataset.csv --num 20

  # Only correctness, no performance
  python3 scripts/convbench/run_convbench.py --correctness-only
"""
import subprocess, sys, os, ast, platform
import numpy as np

# --- Config from environment ---
LLVM_DIR = os.environ.get("LLVM_BUILD_DIR", "")
if not LLVM_DIR:
    print("Error: set LLVM_BUILD_DIR first"); sys.exit(1)
SCONV_ROOT = os.environ.get("SCONV_ROOT", os.path.dirname(os.path.abspath(__file__)) + "/../..")
os.chdir(SCONV_ROOT)

# --- Platform detection ---
IS_MAC = platform.system() == "Darwin"
LIB_EXT = "dylib" if IS_MAC else "so"

# --- Paths ---
SCONV = os.path.join(SCONV_ROOT, "build/bin/sconv-opt")
MOPT = f"{LLVM_DIR}/bin/mlir-opt"
RUNNER = f"{LLVM_DIR}/bin/mlir-cpu-runner"
RUNTIME_DIR = os.path.join(SCONV_ROOT, "scripts/convbench/runtime")
BUILD_DIR = os.path.join(SCONV_ROOT, "build/convbench_runtime")
TF_BLAS = os.path.join(SCONV_ROOT, "test/lowering/sconv-blas.mlir")
CORRECTNESS_TPL = os.path.join(SCONV_ROOT, "scripts/convbench/payloads/correctness_payload.mlir")
PERFORMANCE_TPL = os.path.join(SCONV_ROOT, "scripts/convbench/payloads/performance_payload.mlir")
SAMPLE_CSV = os.path.join(SCONV_ROOT, "scripts/convbench/samples/small_regular.csv")
GEN_PAYLOAD = os.path.join(SCONV_ROOT, "scripts/convbench/gen_payload.py")

LOPTS = ("--one-shot-bufferize=bufferize-function-boundaries "
    "--convert-linalg-to-affine-loops --canonicalize --cse "
    "--expand-strided-metadata --lower-affine --convert-scf-to-cf "
    "--normalize-memrefs --memref-expand --finalize-memref-to-llvm "
    "--lower-affine --convert-func-to-llvm --convert-arith-to-llvm "
    "--convert-cf-to-llvm --canonicalize --cse --symbol-dce "
    "--llvm-legalize-for-export").split()

def build_runtime_libs():
    """Compile rtclock and sgemm_blas_kernel if not already built."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    rtclock_lib = os.path.join(BUILD_DIR, f"librtclock.{LIB_EXT}")
    blas_lib = os.path.join(BUILD_DIR, f"libsgemm_blas_kernel.{LIB_EXT}")
    # rtclock (no deps)
    if not os.path.exists(rtclock_lib):
        src = os.path.join(RUNTIME_DIR, "rtclock.c")
        r = subprocess.run(["clang", "-shared", "-o", rtclock_lib, src],
                           capture_output=True, text=True)
        if r.returncode: print(f"  WARN: rtclock build failed: {r.stderr[:200]}")
    # sgemm_blas_kernel (links OpenBLAS)
    if not os.path.exists(blas_lib):
        src = os.path.join(RUNTIME_DIR, "sgemm_blas_kernel.c")
        openblas_inc = "/opt/homebrew/opt/openblas/include" if IS_MAC else "/usr/include/openblas"
        openblas_lib = "/opt/homebrew/opt/openblas/lib" if IS_MAC else "/usr/lib"
        cmd = ["clang", "-shared", "-o", blas_lib, src,
               f"-I{openblas_inc}", f"-L{openblas_lib}", "-lopenblas"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode: print(f"  WARN: sgemm_blas_kernel build failed: {r.stderr[:200]}")
    return rtclock_lib, blas_lib

def get_libs(rtclock_lib, blas_lib):
    return ",".join([
        f"{LLVM_DIR}/lib/libmlir_c_runner_utils.{LIB_EXT}",
        f"{LLVM_DIR}/lib/libmlir_runner_utils.{LIB_EXT}",
        rtclock_lib, blas_lib,
    ])

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)

def lower_run(mlir, libs):
    tmp = mlir.replace(".mlir", ".llvm.mlir")
    r = sh([MOPT, mlir] + LOPTS + ["-o", tmp])
    if r.returncode: return None, r.stderr[:300]
    r = sh([RUNNER, tmp, "-e", "main", "-entry-point-result=void",
            "-shared-libs=" + libs])
    if os.path.exists(tmp): os.unlink(tmp)
    if r.returncode: return None, r.stderr[:300]
    return r.stdout + r.stderr, None

def get_flops(out):
    for l in reversed(out.strip().split("\n")):
        l = l.strip()
        if "GFLOPS" in l:
            for p in l.split():
                try: return float(p)
                except: pass
    return None

def get_data(out):
    lines = out.strip().split("\n")
    start = -1
    for i, l in enumerate(lines):
        if "data =" in l:
            start = i; break
    if start < 0: return None
    data_str = "\n".join(lines[start+1:])
    end = data_str.rfind("]")
    if end < 0: return None
    data_str = data_str[:end+1].strip()
    data_str = data_str.replace(" ", "").replace("\n", "").replace("\t", "")
    try: return np.array(ast.literal_eval(data_str))
    except: return None

def gen_payloads(csv, template, out_dir, num=None):
    cmd = [sys.executable, GEN_PAYLOAD, template, csv, out_dir]
    if num: cmd += ["--num", str(num)]
    subprocess.run(cmd, capture_output=True, text=True)

def correctness_test(payload_dir, n, libs):
    print(f"\n{'='*60}")
    print(f"  Correctness Test: {n} convolutions")
    print(f"{'='*60}")
    passed = 0; failed = 0
    for f in sorted(os.listdir(payload_dir))[:n]:
        if not f.endswith(".mlir"): continue
        mlir = os.path.join(payload_dir, f)
        base_out, err = lower_run(mlir, libs)
        if err:
            print(f"  SKIP {f}: baseline error"); failed += 1; continue
        tf_tmp = mlir.replace(".mlir", ".tf.mlir")
        r = sh([SCONV, "-transform=" + TF_BLAS, mlir])
        if r.returncode:
            print(f"  SKIP {f}: transform error"); failed += 1; continue
        with open(tf_tmp, "w") as of: of.write(r.stdout)
        tf_out, err = lower_run(tf_tmp, libs)
        if os.path.exists(tf_tmp): os.unlink(tf_tmp)
        if err:
            print(f"  FAIL {f}: run error"); failed += 1; continue
        b = get_data(base_out); t = get_data(tf_out)
        if b is not None and t is not None and np.allclose(b, t, rtol=1e-3, atol=1e-3):
            print(f"  PASS {f}"); passed += 1
        elif b is not None and t is not None:
            print(f"  FAIL {f}: max_diff={np.max(np.abs(b-t)):.4e}"); failed += 1
        else:
            print(f"  SKIP {f}: output parse failed"); failed += 1
    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed

def perf_test(payload_dir, n, libs):
    print(f"\n{'='*60}")
    print(f"  Performance Test: {n} convolutions")
    print(f"{'='*60}")
    print(f"  {'Conv':<20} {'Baseline':>12} {'SConv+BLAS':>12} {'Speedup':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*8}")
    results = []
    for f in sorted(os.listdir(payload_dir))[:n]:
        if not f.endswith(".mlir"): continue
        mlir = os.path.join(payload_dir, f)
        b_out, err = lower_run(mlir, libs)
        if err: continue
        b_flops = get_flops(b_out)
        tf_tmp = mlir.replace(".mlir", ".tf.mlir")
        r = sh([SCONV, "-transform=" + TF_BLAS, mlir])
        if r.returncode: continue
        with open(tf_tmp, "w") as of: of.write(r.stdout)
        t_out, err = lower_run(tf_tmp, libs)
        if os.path.exists(tf_tmp): os.unlink(tf_tmp)
        if err: continue
        t_flops = get_flops(t_out)
        name = f.replace(".mlir", "")
        if b_flops and t_flops:
            su = t_flops / b_flops
            print(f"  {name:<20} {b_flops:>10.2f} {t_flops:>12.2f} {su:>7.2f}x")
            results.append((name, b_flops, t_flops, su))
        else:
            print(f"  {name:<20} {'FAIL':>12} {'FAIL':>12}")
    if results:
        avg = sum(r[3] for r in results) / len(results)
        print(f"\n  Average speedup: {avg:.2f}x ({len(results)} convs)")
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=SAMPLE_CSV, help="CSV dataset file")
    ap.add_argument("--num", type=int, default=5, help="Number of convolutions")
    ap.add_argument("--correctness-only", action="store_true")
    args = ap.parse_args()

    print("Building runtime libraries...")
    rtclock_lib, blas_lib = build_runtime_libs()
    libs = get_libs(rtclock_lib, blas_lib)
    print(f"  rtclock: {rtclock_lib}")
    print(f"  blas:   {blas_lib}")

    # Generate payloads
    tmp = "/tmp/sconvtest_convbench"
    corr_dir = os.path.join(tmp, "correctness")
    perf_dir = os.path.join(tmp, "performance")
    gen_payloads(args.csv, CORRECTNESS_TPL, corr_dir, args.num)
    gen_payloads(args.csv, PERFORMANCE_TPL, perf_dir, args.num)

    p, f = correctness_test(corr_dir, args.num, libs)
    if not args.correctness_only:
        perf_test(perf_dir, args.num, libs)
    print(f"\n{'='*60}")
    print(f"  Done. Correctness: {p} passed, {f} failed")
    print(f"{'='*60}")
