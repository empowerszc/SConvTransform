module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %m = transform.bufferization.one_shot_bufferize %arg0
      { bufferize_function_boundaries = true } : (!transform.any_op) -> !transform.any_op
    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]}
               attributes {sme.direct} in %m
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">
    transform.foreach %convs : !transform.op<"linalg.conv_2d_nchw_fchw"> -> !transform.any_op {
    ^bb(%op: !transform.op<"linalg.conv_2d_nchw_fchw">):
      %c = transform.lower.to_sme "sconv_sme_direct", %op
        : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.any_op
      transform.yield %c : !transform.any_op
    }
    transform.yield
  }
}
