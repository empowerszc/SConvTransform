module {
  func.func @conv(%in: tensor<1x2x10x10xf32>, %wei: tensor<16x2x3x3xf32>,
                  %out: tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32> {
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<[1,1]> : tensor<2xi64>,
                                   strides = dense<[1,1]> : tensor<2xi64>,
                                   sme.direct}
      ins(%in, %wei: tensor<1x2x10x10xf32>, tensor<16x2x3x3xf32>)
      outs(%out: tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
    return %r : tensor<1x16x8x8xf32>
  }
}
