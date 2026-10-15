// Eligible by shape but Ic=96 < min_ic(384) -> must stay on the GEMM path.
module {
  func.func @conv(%in: tensor<1x96x82x82xf32>, %wei: tensor<96x96x3x3xf32>,
                  %out: tensor<1x96x80x80xf32>) -> tensor<1x96x80x80xf32> {
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<[1,1]> : tensor<2xi64>,
                                   strides = dense<[1,1]> : tensor<2xi64>}
      ins(%in, %wei: tensor<1x96x82x82xf32>, tensor<96x96x3x3xf32>)
      outs(%out: tensor<1x96x80x80xf32>) -> tensor<1x96x80x80xf32>
    return %r : tensor<1x96x80x80xf32>
  }
}
