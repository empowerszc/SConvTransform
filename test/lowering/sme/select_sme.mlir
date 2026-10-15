// Eligible (3x3, stride1, Ih==Oh+2) AND profitable: Ic=384>=384, Oc=96<=96,
// area 80x80 large enough -> must be tagged `sme.direct`.
module {
  func.func @conv(%in: tensor<1x384x82x82xf32>, %wei: tensor<96x384x3x3xf32>,
                  %out: tensor<1x96x80x80xf32>) -> tensor<1x96x80x80xf32> {
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<[1,1]> : tensor<2xi64>,
                                   strides = dense<[1,1]> : tensor<2xi64>}
      ins(%in, %wei: tensor<1x384x82x82xf32>, tensor<96x384x3x3xf32>)
      outs(%out: tensor<1x96x80x80xf32>) -> tensor<1x96x80x80xf32>
    return %r : tensor<1x96x80x80xf32>
  }
}
