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
TF_SME = os.path.join(SCONV_ROOT, "test/lowering/sconv-sme.mlir")
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
    # sgemm_blas_kernel: OpenBLAS or HPCKit kblas
    if not os.path.exists(blas_lib):
        src = os.path.join(RUNTIME_DIR, "sgemm_blas_kernel.c")
        kml = os.environ.get("KMLROOT",
            "/home/envs/HPCKit_26.1.RC1/HPCKit/latest/kml/bisheng")
        kblas_variant = os.environ.get("KBLAS_VARIANT", "multi")
        if os.path.exists(os.path.join(kml, "include", "kblas.h")):
            # Prefer HPCKit kblas if it has the SME/matrix variant for Kunpeng
            for sub in ["matrix/kblas", "sve/kblas"]:
                if os.path.exists(os.path.join(kml, "lib", sub, kblas_variant, "libkblas.so")):
                    kblas_libdir = os.path.join(kml, "lib", sub, kblas_variant)
                    break
            cmd = ["clang", "-shared", "-o", blas_lib, src,
                   f"-I{kml}/include", f"-L{kblas_libdir}", "-lkblas", "-DUSE_KBLAST"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(f"  WARN: sgemm_blas_kernel (kblas) build failed: {r.stderr[:300]}")
        else:
            openblas_inc = "/opt/homebrew/opt/openblas/include" if IS_MAC else "/usr/include/openblas"
            openblas_lib = "/opt/homebrew/opt/openblas/lib" if IS_MAC else "/usr/lib"
            cmd = ["clang", "-shared", "-o", blas_lib, src,
                   f"-I{openblas_inc}", f"-L{openblas_lib}", "-lopenblas"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode: print(f"  WARN: sgemm_blas_kernel build failed: {r.stderr[:200]}")
    # pack_input_tile: row-segmented input packing used by `lower.to_pack`.
    pack_lib = os.path.join(BUILD_DIR, f"libpack_input_tile.{LIB_EXT}")
    pack_src = os.path.join(RUNTIME_DIR, "pack_input_tile.c")
    if (not os.path.exists(pack_lib)
            or os.path.getmtime(pack_lib) < os.path.getmtime(pack_src)):
        r = subprocess.run(["clang", "-shared", "-fPIC", "-O2", "-o", pack_lib, pack_src],
                           capture_output=True, text=True)
        if r.returncode: print(f"  WARN: pack_input_tile build failed: {r.stderr[:200]}")
    # fast_memref_copy: row-segmented memrefCopy. Loaded before the MLIR runner
    # utils so its definition of `memrefCopy` wins (see get_libs).
    fastcopy_lib = os.path.join(BUILD_DIR, f"libfast_memref_copy.{LIB_EXT}")
    fastcopy_src = os.path.join(RUNTIME_DIR, "fast_memref_copy.c")
    if (not os.path.exists(fastcopy_lib)
            or os.path.getmtime(fastcopy_lib) < os.path.getmtime(fastcopy_src)):
        r = subprocess.run(["clang", "-shared", "-fPIC", "-O3",
                            "-o", fastcopy_lib, fastcopy_src],
                           capture_output=True, text=True)
        if r.returncode: print(f"  WARN: fast_memref_copy build failed: {r.stderr[:200]}")
    # sconv_sme_direct: ArmSME direct-conv kernel (MLIR) + C shim (pad/pack/threads).
    sme_lib = os.path.join(BUILD_DIR, f"libsconv_sme_direct.{LIB_EXT}")
    sme_dir = os.path.join(SCONV_ROOT, "sme")
    sme_mlir = os.path.join(sme_dir, "kernel/sconv_sme_direct.mlir")
    sme_obj = os.path.join(sme_dir, "kernel/sconv_sme_direct.o")
    sme_shim = os.path.join(sme_dir, "runtime/sconv_sme_shim.c")
    builtins = f"{LLVM_DIR}/lib/clang/19/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a"
    newest_src = max(os.path.getmtime(sme_mlir), os.path.getmtime(sme_shim))
    if not os.path.exists(sme_lib) or os.path.getmtime(sme_lib) < newest_src:
        env = {**os.environ, "LLVM_BUILD_DIR": LLVM_DIR,
               "LD_LIBRARY_PATH": f"{LLVM_DIR}/lib:/home/z00889957/miniconda3/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
        r = subprocess.run([os.path.join(sme_dir, "build.sh"), sme_mlir],
                           capture_output=True, text=True, cwd=sme_dir, env=env)
        if r.returncode:
            print(f"  WARN: sme kernel build failed: {r.stderr[:300]}")
        else:
            cmd = ["clang", "-shared", "-fPIC", "-O2", "-march=armv9-a+sme",
                   "-o", sme_lib, sme_shim, sme_obj, builtins, "-lpthread"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode: print(f"  WARN: sme shim build failed: {r.stderr[:300]}")
    return rtclock_lib, blas_lib, sme_lib, pack_lib, fastcopy_lib

def get_libs(rtclock_lib, blas_lib, sme_lib, pack_lib=None, fastcopy_lib=None):
    libs = []
    # Must be loaded first so its `memrefCopy` wins symbol resolution over the
    # stock one in libmlir_runner_utils.
    if fastcopy_lib:
        libs.append(fastcopy_lib)
    libs += [
        f"{LLVM_DIR}/lib/libmlir_c_runner_utils.{LIB_EXT}",
        f"{LLVM_DIR}/lib/libmlir_runner_utils.{LIB_EXT}",
        rtclock_lib, blas_lib, sme_lib,
    ]
    if pack_lib:
        libs.append(pack_lib)
    return ",".join(libs)

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

def lower_run(mlir, libs):
    """Lower MLIR to LLVM and JIT-run. Returns (output, error, wall_seconds)."""
    import time
    tmp = mlir.replace(".mlir", ".llvm.mlir")
    r = sh([MOPT, mlir] + LOPTS + ["-o", tmp])
    if r.returncode: return None, r.stderr[:300], 0
    t0 = time.time()
    r = sh([RUNNER, tmp, "-e", "main", "-entry-point-result=void",
            "-shared-libs=" + libs])
    elapsed = time.time() - t0
    if os.path.exists(tmp): os.unlink(tmp)
    if r.returncode: return None, r.stderr[:300], elapsed
    return r.stdout + r.stderr, None, elapsed

def get_flops(out):
    for l in reversed(out.strip().split("\n")):
        l = l.strip()
        if "GFLOPS" in l:
            for p in l.split():
                try: return float(p)
                except: pass
    return None

def get_time_ms(out):
    """Parse 'X ms' printed by printTime() to stderr."""
    for l in out.strip().split("\n"):
        l = l.strip()
        if l.endswith(" ms"):
            try: return float(l.replace(" ms", "").strip())
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

def gen_payloads(csv, template, out_dir, num=None, runs=5, warmup=3):
    cmd = [sys.executable, GEN_PAYLOAD, template, csv, out_dir]
    if num: cmd += ["--num", str(num)]
    cmd += ["--runs", str(runs), "--warmup", str(warmup)]
    subprocess.run(cmd, capture_output=True, text=True)

def correctness_test(payload_dir, n, libs, tf=TF_BLAS):
    print(f"\n{'='*60}")
    print(f"  Correctness Test: {n} convolutions")
    print(f"{'='*60}")
    passed = 0; failed = 0
    for f in sorted(os.listdir(payload_dir))[:n]:
        if not f.endswith(".mlir"): continue
        mlir = os.path.join(payload_dir, f)
        base_out, err, _ = lower_run(mlir, libs)
        if err:
            print(f"  SKIP {f}: baseline error"); failed += 1; continue
        tf_tmp = mlir.replace(".mlir", ".tf.mlir")
        r = sh([SCONV, "-transform=" + tf, mlir])
        if r.returncode:
            print(f"  SKIP {f}: transform error"); failed += 1; continue
        with open(tf_tmp, "w") as of: of.write(r.stdout)
        tf_out, err, _ = lower_run(tf_tmp, libs)
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

def perf_test(payload_dir, n, libs, runs=5, warmup=3, tf=TF_BLAS):
    print(f"\n{'='*85}")
    print(f"  Performance Test: {n} convolutions (warmup={warmup}, runs={runs})")
    print(f"  Time = average per-iteration ms (total_time / runs)")
    print(f"{'='*85}")
    hdr = f"  {'Conv':<18} {'Base(ms)':>9} {'SConv(ms)':>9} {'Base GFLOPS':>11} {'SConv GFLOPS':>12} {'Speedup':>7}"
    sep = f"  {'-'*18} {'-'*9} {'-'*9} {'-'*11} {'-'*12} {'-'*7}"
    print(hdr); print(sep)
    results = []
    for f in sorted(os.listdir(payload_dir))[:n]:
        if not f.endswith(".mlir"): continue
        mlir = os.path.join(payload_dir, f)
        b_out, err, _ = lower_run(mlir, libs)
        if err: continue
        b_gflops = get_flops(b_out)
        b_total_ms = get_time_ms(b_out)
        tf_tmp = mlir.replace(".mlir", ".tf.mlir")
        r = sh([SCONV, "-transform=" + tf, mlir])
        if r.returncode: continue
        with open(tf_tmp, "w") as of: of.write(r.stdout)
        t_out, err, _ = lower_run(tf_tmp, libs)
        if os.path.exists(tf_tmp): os.unlink(tf_tmp)
        if err: continue
        t_gflops = get_flops(t_out)
        t_total_ms = get_time_ms(t_out)
        name = f.replace(".mlir", "")
        if b_total_ms is not None and t_total_ms is not None and b_gflops and t_gflops:
            b_ms = b_total_ms / runs  # per-iteration
            t_ms = t_total_ms / runs
            su = t_gflops / b_gflops
            print(f"  {name:<18} {b_ms:>9.3f} {t_ms:>9.3f} {b_gflops:>9.2f} {t_gflops:>12.2f} {su:>6.2f}x")
            results.append((name, b_ms, t_ms, b_gflops, t_gflops, su))
        else:
            print(f"  {name:<18} {'FAIL':>9} {'FAIL':>9}")
    if results:
        avg_su = sum(r[5] for r in results) / len(results)
        tot_b = sum(r[1] for r in results)
        tot_t = sum(r[2] for r in results)
        print(f"\n  Total per-iter time:  baseline {tot_b:.1f}ms  |  SConv+BLAS {tot_t:.1f}ms")
        print(f"  Average speedup: {avg_su:.2f}x ({len(results)} convs)")
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=SAMPLE_CSV, help="CSV dataset file")
    ap.add_argument("--num", type=int, default=5, help="Number of convolutions")
    ap.add_argument("--runs", type=int, default=5, help="Timed iterations (after warmup)")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup iterations (untimed)")
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--transform", default=TF_BLAS,
                    help="Transform script for the SConv path (default: sconv-blas.mlir)")
    args = ap.parse_args()
    tf = os.path.abspath(args.transform)

    print("Building runtime libraries...")
    rtclock_lib, blas_lib, sme_lib, pack_lib, fastcopy_lib = build_runtime_libs()
    libs = get_libs(rtclock_lib, blas_lib, sme_lib, pack_lib, fastcopy_lib)
    print(f"  rtclock: {rtclock_lib}")
    print(f"  blas:   {blas_lib}")
    print(f"  sme:    {sme_lib}")
    print(f"  pack:   {pack_lib}")
    print(f"  copy:   {fastcopy_lib}")
    print(f"  transform: {tf}")

    # Generate payloads
    tmp = "/tmp/sconvtest_convbench"
    corr_dir = os.path.join(tmp, "correctness")
    perf_dir = os.path.join(tmp, "performance")
    gen_payloads(args.csv, CORRECTNESS_TPL, corr_dir, args.num, args.runs, args.warmup)
    gen_payloads(args.csv, PERFORMANCE_TPL, perf_dir, args.num, args.runs, args.warmup)

    p, f = correctness_test(corr_dir, args.num, libs, tf)
    if not args.correctness_only:
        perf_test(perf_dir, args.num, libs, args.runs, args.warmup, tf)
    print(f"\n{'='*60}")
    print(f"  Done. Correctness: {p} passed, {f} failed")
    print(f"{'='*60}")
