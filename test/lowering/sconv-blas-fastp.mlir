// Default SConv fast path.
//
// mK_info = [0, 0] enables the shape-based autotuner: the compiler picks the
// widest window/filter tile that divides the output spatial size and the filter
// count (see selectAutoNwin/selectAutoNf in lib/SConv.cpp). For regular shapes
// this selects [Nwin=64, Nf=32], which measured 115.7 -> ~87 ms/iter on
// custom_0; for shapes where 64/32 would hit the edge path it degrades to
// [16,32]/[64,16]/[16,8] instead of regressing. All 11 custom shapes improve
// over the old fixed [16,8] tile (docs/OPTIMIZATION_DESIGN_AND_RESULTS.md §3.8).
//
// test/lowering/sconv-blas-fastp-tuned.mlir pins [64,32] explicitly as an
// escape hatch for benchmarking.

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    //===------------------------------------------------------------------===//
    // APPLY SCONV
    //===------------------------------------------------------------------===//

    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    %ukernels_0, %loops = transform.structured.sconv %convs
      { mK_info = [0, 0] }
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
