#!/usr/bin/env bash
set -euo pipefail
# Usage: build.sh <file.mlir>   (source env.sh first, or LLVM_BUILD_DIR set)
LLVM=${LLVM_BUILD_DIR:?set LLVM_BUILD_DIR or source env.sh}
export PATH="$LLVM/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM/lib:/home/z00889957/miniconda3/lib:${LD_LIBRARY_PATH:-}"
src="$1"; base="${src%.mlir}"
mlir-opt "$src" \
  --test-lower-to-arm-sme \
  --convert-vector-to-llvm --convert-to-llvm --reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir > "${base}.ll"
clang -march=armv9-a+sme -O2 -c "${base}.ll" -o "${base}.o"
echo "built ${base}.o"
