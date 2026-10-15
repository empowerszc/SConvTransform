#!/usr/bin/env bash
# Regression test for the SME routing pipeline (transform.lower.sme_select and
# transform.lower.to_sme). Usage:  source env.sh && test/lowering/sme/run_tests.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
cd "$root"

OPT="${SCONV_OPT:-build/bin/sconv-opt}"
SEL="$here/select_only.mlir"
PIPE="test/lowering/sconv-sme.mlir"
LLVM="${LLVM_BUILD_DIR:-/home/z00889957/000App/clang+llvm-19.1.7-aarch64-linux-gnu}"
export LD_LIBRARY_PATH="$LLVM/lib:/home/z00889957/miniconda3/lib:${LD_LIBRARY_PATH:-}"

pass=0; fail=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "PASS $1"; pass=$((pass+1));
  else echo "FAIL $1: expected [$2] got [$3]"; fail=$((fail+1)); fi
}
tagged() { # file -> yes/no (does sme_select apply the sme.direct marker?)
  local n
  n=$("$OPT" -transform="$SEL" "$1" 2>/dev/null | grep -c "sme.direct")
  if [ "$n" -gt 0 ]; then echo yes; else echo no; fi
}
callname() { # file -> emitted runtime symbol via the full pipeline
  "$OPT" -transform="$PIPE" "$1" 2>/dev/null \
    | grep -oE "@(sconv_sme_direct|sgemm_blas_kernel)" | sort -u | tr -d '\n'
}

check "select_sme tagged"           yes  "$(tagged "$here/select_sme.mlir")"
check "select_gemm_ic not tagged"   no   "$(tagged "$here/select_gemm_ic.mlir")"
check "select_gemm_tiny not tagged" no   "$(tagged "$here/select_gemm_tiny.mlir")"
check "pipeline sme emits call"     @sconv_sme_direct "$(callname "$here/select_sme.mlir")"
check "pipeline gemm emits blas"    @sgemm_blas_kernel "$(callname "$here/select_gemm_ic.mlir")"

echo "routing tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
