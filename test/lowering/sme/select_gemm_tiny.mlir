// Eligible and passes the Ic/Oc gates, but the output area (4x4=16) is far too
// small for the SME kernel's per-task overhead to pay off -> must stay GEMM.
module {
  func.func @conv(%in: tensor<1x384x6x6xf32>, %wei: tensor<16x384x3x3xf32>,
                  %out: tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32> {
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<[1,1]> : tensor<2xi64>,
                                   strides = dense<[1,1]> : tensor<2xi64>}
      ins(%in, %wei: tensor<1x384x6x6xf32>, tensor<16x384x3x3xf32>)
      outs(%out: tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
    return %r : tensor<1x16x4x4xf32>
  }
}
