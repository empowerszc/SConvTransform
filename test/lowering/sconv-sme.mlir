// Full SConv pipeline with SME routing:
//   1. tag SME-eligible-and-profitable convs with `sme.direct` (sme_select)
//   2. run the SConv/GEMM pipeline on the remaining convs
//   3. bufferize, simplify
//   4. lower `sme.direct` convs to `sconv_sme_direct`, microkernels to BLAS
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    //===------------------------------------------------------------------===//
    // SELECT SME CANDIDATES
    //===------------------------------------------------------------------===//

    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    %rest = transform.lower.sme_select %convs
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    //===------------------------------------------------------------------===//
    // APPLY SCONV ON THE GEMM REMAINDER
    //===------------------------------------------------------------------===//

    %ukernels_0, %loops = transform.structured.sconv %rest
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

    transform.apply_cse to %bufferized : !transform.any_op
    transform.apply_dce to %bufferized : !transform.any_op

    //===------------------------------------------------------------------===//
    // LOWER SME CONVS
    //===------------------------------------------------------------------===//

    %sme_convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]}
      attributes {sme.direct} in %bufferized
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    transform.foreach %sme_convs : !transform.op<"linalg.conv_2d_nchw_fchw"> -> !transform.any_op {
    ^bb(%op: !transform.op<"linalg.conv_2d_nchw_fchw">):
      %lowered = transform.lower.to_sme "sconv_sme_direct", %op
        : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> (!transform.any_op)
      transform.yield %lowered : !transform.any_op
    }

    //===------------------------------------------------------------------===//
    // LOWER GEMM MICROKERNELS
    //===------------------------------------------------------------------===//

    // Get another handle to the kernels because the last one was invalidated
    // by one-shot bufferize.
    %ukernels_1 = transform.structured.match
      ops{["linalg.generic"]}
      attributes {microkernel} in %bufferized
      : (!transform.any_op) -> !transform.op<"linalg.generic">

    transform.foreach %ukernels_1 : !transform.op<"linalg.generic"> -> !transform.any_op {
    ^bb(%op: !transform.op<"linalg.generic">):
      %lowered = transform.lower.to_blas "sgemm_blas_kernel", %op
        : (!transform.op<"linalg.generic">) -> (!transform.any_op)
      transform.yield %lowered : !transform.any_op
    }

    transform.yield
  }
}

// vim: sts=2 sw=2 et
