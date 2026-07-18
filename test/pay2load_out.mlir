// sconv-opt -transform=test/sconv_mK.mlir test/pay2load.mlir
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
#map3 = affine_map<(d0, d1, d2)[s0] -> (((d2 + s0) floordiv 56 - s0 floordiv 56 + d0) * 56 + (d2 + s0) mod 56 - s0 mod 56 + d1)>
#map4 = affine_map<(d0) -> (d0 floordiv 8)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @conv_2d_nchw_fchw(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<24x96x1x1xf32>, %arg2: tensor<144x24x1x1xf32>, %arg3: tensor<1x24x56x56xf32>, %arg4: tensor<1x144x56x56xf32>) -> tensor<1x144x56x56xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<1x96x56x56xf32> into tensor<1x96x3136xf32>
  %collapsed_0 = tensor.collapse_shape %arg3 [[0], [1], [2, 3]] : tensor<1x24x56x56xf32> into tensor<1x24x3136xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_1 = arith.constant 1 : index
  %0 = scf.for %arg5 = %c0 to %c1 step %c1_1 iter_args(%arg6 = %collapsed_0) -> (tensor<1x24x3136xf32>) {
    %c0_8 = arith.constant 0 : index
    %c96 = arith.constant 96 : index
    %c96_9 = arith.constant 96 : index
    %2 = scf.for %arg7 = %c0_8 to %c96 step %c96_9 iter_args(%arg8 = %arg6) -> (tensor<1x24x3136xf32>) {
      %c0_10 = arith.constant 0 : index
      %c3136 = arith.constant 3136 : index
      %c3136_11 = arith.constant 3136 : index
      %3 = scf.for %arg9 = %c0_10 to %c3136 step %c3136_11 iter_args(%arg10 = %arg8) -> (tensor<1x24x3136xf32>) {
        %c0_12 = arith.constant 0 : index
        %c24 = arith.constant 24 : index
        %c24_13 = arith.constant 24 : index
        %4 = scf.for %arg11 = %c0_12 to %c24 step %c24_13 iter_args(%arg12 = %arg10) -> (tensor<1x24x3136xf32>) {
          %extracted_slice = tensor.extract_slice %collapsed[%arg5, %arg7, %arg9] [1, 96, 3136] [1, 1, 1] : tensor<1x96x3136xf32> to tensor<1x96x3136xf32>
          %extracted_slice_14 = tensor.extract_slice %arg1[%arg11, %arg7, 0, 0] [24, 96, 1, 1] [1, 1, 1, 1] : tensor<24x96x1x1xf32> to tensor<24x96x1x1xf32>
          %extracted_slice_15 = tensor.extract_slice %arg12[%arg5, %arg11, %arg9] [1, 24, 3136] [1, 1, 1] : tensor<1x24x3136xf32> to tensor<1x24x3136xf32>
          %c0_16 = arith.constant 0 : index
          %c24_17 = arith.constant 24 : index
          %c8 = arith.constant 8 : index
          %c0_18 = arith.constant 0 : index
          %c3136_19 = arith.constant 3136 : index
          %c16 = arith.constant 16 : index
          %5 = tensor.empty() : tensor<3x96x1x1x8xf32>
          %6 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%5 : tensor<3x96x1x1x8xf32>) attrs =  {multipacking = true, packing = "filter"} {
          ^bb0(%out: f32):
            %8 = linalg.index 0 : index
            %9 = linalg.index 1 : index
            %10 = linalg.index 2 : index
            %11 = linalg.index 3 : index
            %12 = linalg.index 4 : index
            %13 = affine.apply #map1(%8, %12)
            %extracted = tensor.extract %extracted_slice_14[%13, %9, %10, %11] : tensor<24x96x1x1xf32>
            linalg.yield %extracted : f32
          } -> tensor<3x96x1x1x8xf32>
          %7 = scf.for %arg13 = %c0_18 to %c3136_19 step %c16 iter_args(%arg14 = %extracted_slice_15) -> (tensor<1x24x3136xf32>) {
            %extracted_slice_20 = tensor.extract_slice %extracted_slice[0, 0, %arg13] [1, 96, 16] [1, 1, 1] : tensor<1x96x3136xf32> to tensor<1x96x16xf32>
            %8 = tensor.empty() : tensor<1x96x1x1x16xf32>
            %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%8 : tensor<1x96x1x1x16xf32>) attrs =  {multipacking = false, packing = "input"} {
            ^bb0(%out: f32):
              %11 = linalg.index 0 : index
              %12 = linalg.index 1 : index
              %13 = linalg.index 2 : index
              %14 = linalg.index 3 : index
              %15 = linalg.index 4 : index
              %16 = affine.apply #map2(%arg13)[%arg9]
              %17 = affine.apply #map3(%13, %14, %15)[%16]
              %extracted = tensor.extract %extracted_slice_20[%11, %12, %17] : tensor<1x96x16xf32>
              linalg.yield %extracted : f32
            } -> tensor<1x96x1x1x16xf32>
            %collapsed_21 = tensor.collapse_shape %9 [[0], [1, 2, 3], [4]] : tensor<1x96x1x1x16xf32> into tensor<1x96x16xf32>
            %10 = scf.for %arg15 = %c0_16 to %c24_17 step %c8 iter_args(%arg16 = %arg14) -> (tensor<1x24x3136xf32>) {
              %11 = affine.apply #map4(%arg15)
              %extracted_slice_22 = tensor.extract_slice %6[%11, 0, 0, 0, 0] [1, 96, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<3x96x1x1x8xf32> to tensor<1x96x1x1x8xf32>
              %collapsed_23 = tensor.collapse_shape %extracted_slice_22 [[0, 1, 2, 3], [4]] : tensor<1x96x1x1x8xf32> into tensor<96x8xf32>
              %extracted_slice_24 = tensor.extract_slice %arg16[0, %arg15, %arg13] [1, 8, 16] [1, 1, 1] : tensor<1x24x3136xf32> to tensor<1x8x16xf32>
              %12 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_21, %collapsed_23 : tensor<1x96x16xf32>, tensor<96x8xf32>) outs(%extracted_slice_24 : tensor<1x8x16xf32>) attrs =  {microkernel, schedule = "IS"} {
              ^bb0(%in: f32, %in_26: f32, %out: f32):
                %13 = arith.mulf %in, %in_26 : f32
                %14 = arith.addf %13, %out : f32
                linalg.yield %14 : f32
              } -> tensor<1x8x16xf32>
              %inserted_slice_25 = tensor.insert_slice %12 into %arg16[0, %arg15, %arg13] [1, 8, 16] [1, 1, 1] : tensor<1x8x16xf32> into tensor<1x24x3136xf32>
              scf.yield %inserted_slice_25 : tensor<1x24x3136xf32>
            }
            scf.yield %10 : tensor<1x24x3136xf32>
          }
          %inserted_slice = tensor.insert_slice %7 into %arg12[%arg5, %arg11, %arg9] [1, 24, 3136] [1, 1, 1] : tensor<1x24x3136xf32> into tensor<1x24x3136xf32>
          scf.yield %inserted_slice : tensor<1x24x3136xf32>
        }
        scf.yield %4 : tensor<1x24x3136xf32>
      }
      scf.yield %3 : tensor<1x24x3136xf32>
    }
    scf.yield %2 : tensor<1x24x3136xf32>
  }
  %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] output_shape [1, 24, 56, 56] : tensor<1x24x3136xf32> into tensor<1x24x56x56xf32>
  %collapsed_2 = tensor.collapse_shape %expanded [[0], [1], [2, 3]] : tensor<1x24x56x56xf32> into tensor<1x24x3136xf32>
  %collapsed_3 = tensor.collapse_shape %arg4 [[0], [1], [2, 3]] : tensor<1x144x56x56xf32> into tensor<1x144x3136xf32>
  %c0_4 = arith.constant 0 : index
  %c1_5 = arith.constant 1 : index
  %c1_6 = arith.constant 1 : index
  %1 = scf.for %arg5 = %c0_4 to %c1_5 step %c1_6 iter_args(%arg6 = %collapsed_3) -> (tensor<1x144x3136xf32>) {
    %c0_8 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c24_9 = arith.constant 24 : index
    %2 = scf.for %arg7 = %c0_8 to %c24 step %c24_9 iter_args(%arg8 = %arg6) -> (tensor<1x144x3136xf32>) {
      %c0_10 = arith.constant 0 : index
      %c3136 = arith.constant 3136 : index
      %c3136_11 = arith.constant 3136 : index
      %3 = scf.for %arg9 = %c0_10 to %c3136 step %c3136_11 iter_args(%arg10 = %arg8) -> (tensor<1x144x3136xf32>) {
        %c0_12 = arith.constant 0 : index
        %c144 = arith.constant 144 : index
        %c144_13 = arith.constant 144 : index
        %4 = scf.for %arg11 = %c0_12 to %c144 step %c144_13 iter_args(%arg12 = %arg10) -> (tensor<1x144x3136xf32>) {
          %extracted_slice = tensor.extract_slice %collapsed_2[%arg5, %arg7, %arg9] [1, 24, 3136] [1, 1, 1] : tensor<1x24x3136xf32> to tensor<1x24x3136xf32>
          %extracted_slice_14 = tensor.extract_slice %arg2[%arg11, %arg7, 0, 0] [144, 24, 1, 1] [1, 1, 1, 1] : tensor<144x24x1x1xf32> to tensor<144x24x1x1xf32>
          %extracted_slice_15 = tensor.extract_slice %arg12[%arg5, %arg11, %arg9] [1, 144, 3136] [1, 1, 1] : tensor<1x144x3136xf32> to tensor<1x144x3136xf32>
          %c0_16 = arith.constant 0 : index
          %c144_17 = arith.constant 144 : index
          %c8 = arith.constant 8 : index
          %c0_18 = arith.constant 0 : index
          %c3136_19 = arith.constant 3136 : index
          %c16 = arith.constant 16 : index
          %5 = tensor.empty() : tensor<18x24x1x1x8xf32>
          %6 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%5 : tensor<18x24x1x1x8xf32>) attrs =  {multipacking = true, packing = "filter"} {
          ^bb0(%out: f32):
            %8 = linalg.index 0 : index
            %9 = linalg.index 1 : index
            %10 = linalg.index 2 : index
            %11 = linalg.index 3 : index
            %12 = linalg.index 4 : index
            %13 = affine.apply #map1(%8, %12)
            %extracted = tensor.extract %extracted_slice_14[%13, %9, %10, %11] : tensor<144x24x1x1xf32>
            linalg.yield %extracted : f32
          } -> tensor<18x24x1x1x8xf32>
          %7 = scf.for %arg13 = %c0_18 to %c3136_19 step %c16 iter_args(%arg14 = %extracted_slice_15) -> (tensor<1x144x3136xf32>) {
            %extracted_slice_20 = tensor.extract_slice %extracted_slice[0, 0, %arg13] [1, 24, 16] [1, 1, 1] : tensor<1x24x3136xf32> to tensor<1x24x16xf32>
            %8 = tensor.empty() : tensor<1x24x1x1x16xf32>
            %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%8 : tensor<1x24x1x1x16xf32>) attrs =  {multipacking = false, packing = "input"} {
            ^bb0(%out: f32):
              %11 = linalg.index 0 : index
              %12 = linalg.index 1 : index
              %13 = linalg.index 2 : index
              %14 = linalg.index 3 : index
              %15 = linalg.index 4 : index
              %16 = affine.apply #map2(%arg13)[%arg9]
              %17 = affine.apply #map3(%13, %14, %15)[%16]
              %extracted = tensor.extract %extracted_slice_20[%11, %12, %17] : tensor<1x24x16xf32>
              linalg.yield %extracted : f32
            } -> tensor<1x24x1x1x16xf32>
            %collapsed_21 = tensor.collapse_shape %9 [[0], [1, 2, 3], [4]] : tensor<1x24x1x1x16xf32> into tensor<1x24x16xf32>
            %10 = scf.for %arg15 = %c0_16 to %c144_17 step %c8 iter_args(%arg16 = %arg14) -> (tensor<1x144x3136xf32>) {
              %11 = affine.apply #map4(%arg15)
              %extracted_slice_22 = tensor.extract_slice %6[%11, 0, 0, 0, 0] [1, 24, 1, 1, 8] [1, 1, 1, 1, 1] : tensor<18x24x1x1x8xf32> to tensor<1x24x1x1x8xf32>
              %collapsed_23 = tensor.collapse_shape %extracted_slice_22 [[0, 1, 2, 3], [4]] : tensor<1x24x1x1x8xf32> into tensor<24x8xf32>
              %extracted_slice_24 = tensor.extract_slice %arg16[0, %arg15, %arg13] [1, 8, 16] [1, 1, 1] : tensor<1x144x3136xf32> to tensor<1x8x16xf32>
              %12 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_21, %collapsed_23 : tensor<1x24x16xf32>, tensor<24x8xf32>) outs(%extracted_slice_24 : tensor<1x8x16xf32>) attrs =  {microkernel, schedule = "IS"} {
              ^bb0(%in: f32, %in_26: f32, %out: f32):
                %13 = arith.mulf %in, %in_26 : f32
                %14 = arith.addf %13, %out : f32
                linalg.yield %14 : f32
              } -> tensor<1x8x16xf32>
              %inserted_slice_25 = tensor.insert_slice %12 into %arg16[0, %arg15, %arg13] [1, 8, 16] [1, 1, 1] : tensor<1x8x16xf32> into tensor<1x144x3136xf32>
              scf.yield %inserted_slice_25 : tensor<1x144x3136xf32>
            }
            scf.yield %10 : tensor<1x144x3136xf32>
          }
          %inserted_slice = tensor.insert_slice %7 into %arg12[%arg5, %arg11, %arg9] [1, 144, 3136] [1, 1, 1] : tensor<1x144x3136xf32> into tensor<1x144x3136xf32>
          scf.yield %inserted_slice : tensor<1x144x3136xf32>
        }
        scf.yield %4 : tensor<1x144x3136xf32>
      }
      scf.yield %3 : tensor<1x144x3136xf32>
    }
    scf.yield %2 : tensor<1x144x3136xf32>
  }
  %expanded_7 = tensor.expand_shape %1 [[0], [1], [2, 3]] output_shape [1, 144, 56, 56] : tensor<1x144x3136xf32> into tensor<1x144x56x56xf32>
  return %expanded_7 : tensor<1x144x56x56xf32>
}
