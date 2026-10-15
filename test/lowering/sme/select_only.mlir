// Run only `sme_select` over the payload's convs, so the test can observe the
// `sme.direct` marker without needing the rest of the pipeline to succeed.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">
    %rest = transform.lower.sme_select %convs
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.op<"linalg.conv_2d_nchw_fchw">
    transform.yield
  }
}
