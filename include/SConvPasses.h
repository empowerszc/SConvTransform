//===- SConvPasses.h - Post-transform passes for SConv ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// @brief Buffer-level passes used after the SConv transform script has run.
//
// `createSConvParallelizePass` turns the tile loops that the SConv transform
// tagged with `sconv.parallel` into `scf.parallel`, and gives the scratch
// buffers tagged with `sconv.parallel_scratch` one slot per OpenMP thread.
//
// See docs/OPTIMIZATION_PLAN_2026-11-18.md (section 9) for the design.
//
//===----------------------------------------------------------------------===//

#ifndef SCONV_PASSES_H
#define SCONV_PASSES_H

#include <memory>

namespace mlir {
class Pass;

/// Create the pass that rewrites `sconv.parallel` tile loops into
/// `scf.parallel` and privatizes tagged scratch buffers per thread.
std::unique_ptr<Pass> createSConvParallelizePass();

/// Create the pass with an explicit number of per-thread scratch slots.
std::unique_ptr<Pass> createSConvParallelizePass(unsigned maxThreads);

/// Create the pass that inlines `memref.alloca_scope` regions without stack
/// allocations (needed before the CFG lowering inside a parallel region).
std::unique_ptr<Pass> createSConvEraseAllocaScopePass();

/// Register all SConv passes with the global pass registry, so that they can be
/// referenced from a textual pass pipeline.
void registerSConvPasses();

} // namespace mlir

#endif // SCONV_PASSES_H
