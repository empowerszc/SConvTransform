// Multicore SConv fast path.
//
// Same pipeline as sconv-blas-fastp.mlir, plus the `multicore` hint: the
// automatic window tile is chosen by selectAutoNwinMulticore() instead of
// selectAutoNwin(). The window tile loop is the parallel dimension of
// `-sconv-parallelize`, so multicore prefers a smaller tile (more work items)
// while thin-channel shapes keep the wide tile (packing-bandwidth bound) and
// the Ohw=400 shapes quarter it (halving would leave only 5 work items).
//
// Use it only together with -sconv-parallelize: single-thread numbers are
// 0-15% worse by design. Measured gains on the parallel path are 10-20% on
// custom_0/4/6/8/10; see docs/OPTIMIZATION_PLAN_2026-11-18.md section 9.6.
//
// mK_info = [0, 0] still means "autotune"; explicit values override the
// multicore heuristic.

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    //===------------------------------------------------------------------===//
    // APPLY SCONV
    //===------------------------------------------------------------------===//

    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    %ukernels_0, %loops = transform.structured.sconv %convs
      { mK_info = [0, 0], multicore }
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) 
      -> (!transform.op<"linalg.generic">, !transform.any_op)

    //===------------------------------------------------------------------===//
    // BUFFERIZE
    //===------------------------------------------------------------------===//

    %bufferized = transform.bufferization.one_shot_bufferize %arg0
      { bufferize_function_boundaries = true }
      : (!transform.any_op) -> !transform.any_op

    %funcs = transform.structured.match ops{["func.func"]} in %bufferized 
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %funcs {
      transform.apply_patterns.memref.extract_address_computations 
      transform.apply_patterns.memref.expand_strided_metadata 
    } : !transform.any_op

    //===------------------------------------------------------------------===//
    // SIMPLIFY
    //===------------------------------------------------------------------===//

    // Apply common sub-expression elimination
    transform.apply_cse to %bufferized : !transform.any_op
    // Apply dead-code elimination
    transform.apply_dce to %bufferized : !transform.any_op

    //===------------------------------------------------------------------===//
    // FAST INPUT PACKING (M1)
    //===------------------------------------------------------------------===//

    // The scalar input-packing gather is replaced by a row-segmented runtime
    // call (`pack_input_tile` in scripts/convbench/runtime/). Semantics are
    // unchanged: the call is handed the offset of the tile's first element
    // (evaluated from the existing affine chain with the tile indices zeroed)
    // plus the relative geometry recorded in `sconv.pack_geom`.
    %packs = transform.structured.match
      ops{["linalg.generic"]}
      attributes {packing = "input"} in %bufferized
      : (!transform.any_op) -> !transform.op<"linalg.generic">

    transform.foreach %packs : !transform.op<"linalg.generic"> -> !transform.any_op {
    ^bb0(%pack: !transform.op<"linalg.generic">):
      %packed = transform.lower.to_pack "pack_input_tile", %pack
        : (!transform.op<"linalg.generic">) -> (!transform.any_op)
      transform.yield %packed : !transform.any_op
    }

    //===------------------------------------------------------------------===//
    // LOWER
    //===------------------------------------------------------------------===//

    // Get another handle to the kernels because the last one was invalidated
    // by one-shot bufferize.
    %ukernels_1 = transform.structured.match
      ops{["linalg.generic"]}
      attributes {microkernel} in %bufferized
      : (!transform.any_op) -> !transform.op<"linalg.generic">
 
    // Lower each microkernel individually
    transform.foreach %ukernels_1 : !transform.op<"linalg.generic"> -> !transform.any_op {
    ^bb1(%op: !transform.op<"linalg.generic">):
      %lowered = transform.lower.to_blas "sgemm_blas_kernel", %op
        : (!transform.op<"linalg.generic">) -> (!transform.any_op)
      transform.yield %lowered : !transform.any_op
    }

    transform.yield
  }
}

// vim: sts=2 sw=2 et
