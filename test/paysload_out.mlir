// sconv-opt -transform=test/sconv_mK.mlir test/paysload.mlir
#map = affine_map<(d0) -> ((d0 floordiv 28) * 30 + d0 mod 28)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map3 = affine_map<(d0)[s0] -> (((d0 + s0) floordiv 28 - s0 floordiv 28) * 30 + (d0 + s0) mod 28 - s0 mod 28)>
#map4 = affine_map<(d0)[s0] -> (d0 + s0)>
#map5 = affine_map<(d0, d1, d2)[s0] -> (((d2 + s0) floordiv 28 - s0 floordiv 28 + d0) * 30 + (d2 + s0) mod 28 - s0 mod 28 + d1)>
#map6 = affine_map<(d0) -> (d0 floordiv 8)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @conv_2d_nchw_fchw(%arg0: tensor<1x332x30x30xf32>, %arg1: tensor<336x332x3x3xf32>, %arg2: tensor<1x336x28x28xf32>) -> tensor<1x336x28x28xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<1x332x30x30xf32> into tensor<1x332x900xf32>
  %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<1x336x28x28xf32> into tensor<1x336x784xf32>
  %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0] [1, 320, 900] [1, 1, 1] : tensor<1x332x900xf32> to tensor<1x320x900xf32>
  %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, 0] [336, 320, 3, 3] [1, 1, 1, 1] : tensor<336x332x3x3xf32> to tensor<336x320x3x3xf32>
  %extracted_slice_2 = tensor.extract_slice %collapsed_0[0, 0, 0] [1, 336, 784] [1, 1, 1] : tensor<1x336x784xf32> to tensor<1x336x784xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_3 = arith.constant 1 : index
  %0 = scf.for %arg3 = %c0 to %c1 step %c1_3 iter_args(%arg4 = %extracted_slice_2) -> (tensor<1x336x784xf32>) {
    %c0_11 = arith.constant 0 : index
    %c320 = arith.constant 320 : index
    %c20 = arith.constant 20 : index
    %2 = scf.for %arg5 = %c0_11 to %c320 step %c20 iter_args(%arg6 = %arg4) -> (tensor<1x336x784xf32>) {
      %c0_12 = arith.constant 0 : index
      %c784 = arith.constant 784 : index
      %c784_13 = arith.constant 784 : index
      %3 = scf.for %arg7 = %c0_12 to %c784 step %c784_13 iter_args(%arg8 = %arg6) -> (tensor<1x336x784xf32>) {
        %c0_14 = arith.constant 0 : index
        %c336 = arith.constant 336 : index
        %c336_15 = arith.constant 336 : index
        %4 = scf.for %arg9 = %c0_14 to %c336 step %c336_15 iter_args(%arg10 = %arg8) -> (tensor<1x336x784xf32>) {
          %5 = affine.apply #map(%arg7)
          %extracted_slice_16 = tensor.extract_slice %extracted_slice[%arg3, %arg5, %5] [1, 20, 900] [1, 1, 1] : tensor<1x320x900xf32> to tensor<1x20x900xf32>
          %extracted_slice_17 = tensor.extract_slice %extracted_slice_1[%arg9, %arg5, 0, 0] [336, 20, 3, 3] [1, 1, 1, 1] : tensor<336x320x3x3xf32> to tensor<336x20x3x3xf32>
          %extracted_slice_18 = tensor.extract_slice %arg10[%arg3, %arg9, %arg7] [1, 336, 784] [1, 1, 1] : tensor<1x336x784xf32> to tensor<1x336x784xf32>
          %c0_19 = arith.constant 0 : index
          %c336_20 = arith.constant 336 : index
          %c8 = arith.constant 8 : index
          %c0_21 = arith.constant 0 : index
          %c784_22 = arith.constant 784 : index
          %c16 = arith.constant 16 : index
          %6 = tensor.empty() : tensor<42x20x3x3x8xf32>
          %7 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%6 : tensor<42x20x3x3x8xf32>) attrs =  {multipacking = true, packing = "filter"} {
          ^bb0(%out: f32):
            %9 = linalg.index 0 : index
            %10 = linalg.index 1 : index
            %11 = linalg.index 2 : index
            %12 = linalg.index 3 : index
            %13 = linalg.index 4 : index
            %14 = affine.apply #map2(%9, %13)
            %extracted = tensor.extract %extracted_slice_17[%14, %10, %11, %12] : tensor<336x20x3x3xf32>
            linalg.yield %extracted : f32
          } -> tensor<42x20x3x3x8xf32>
          %8 = scf.for %arg11 = %c0_21 to %c784_22 step %c16 iter_args(%arg12 = %extracted_slice_18) -> (tensor<1x336x784xf32>) {
            %9 = affine.apply #map3(%arg11)[%arg7]
            %extracted_slice_24 = tensor.extract_slice %extracted_slice_16[0, 0, %9] [1, 20, 78] [1, 1, 1] : tensor<1x20x900xf32> to tensor<1x20x78xf32>
            %10 = tensor.empty() : tensor<1x20x3x3x16xf32>
            %11 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%10 : tensor<1x20x3x3x16xf32>) attrs =  {multipacking = false, packing = "input"} {
            ^bb0(%out: f32):
              %13 = linalg.index 0 : index
              %14 = linalg.index 1 : index
              %15 = linalg.index 2 : index
              %16 = linalg.index 3 : index
              %17 = linalg.index 4 : index
              %18 = affine.apply #map4(%arg11)[%arg7]
              %19 = affine.apply #map5(%15, %16, %17)[%18]
              %extracted = tensor.extract %extracted_slice_24[%13, %14, %19] : tensor<1x20x78xf32>
              linalg.yield %extracted : f32
            } -> tensor<1x20x3x3x16xf32>
            %collapsed_25 = tensor.collapse_shape %11 [[0], [1, 2, 3], [4]] : tensor<1x20x3x3x16xf32> into tensor<1x180x16xf32>
            %12 = scf.for %arg13 = %c0_19 to %c336_20 step %c8 iter_args(%arg14 = %arg12) -> (tensor<1x336x784xf32>) {
              %13 = affine.apply #map6(%arg13)
              %extracted_slice_26 = tensor.extract_slice %7[%13, 0, 0, 0, 0] [1, 20, 3, 3, 8] [1, 1, 1, 1, 1] : tensor<42x20x3x3x8xf32> to tensor<1x20x3x3x8xf32>
              %collapsed_27 = tensor.collapse_shape %extracted_slice_26 [[0, 1, 2, 3], [4]] : tensor<1x20x3x3x8xf32> into tensor<180x8xf32>
              %extracted_slice_28 = tensor.extract_slice %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x336x784xf32> to tensor<1x8x16xf32>
              %14 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_25, %collapsed_27 : tensor<1x180x16xf32>, tensor<180x8xf32>) outs(%extracted_slice_28 : tensor<1x8x16xf32>) attrs =  {microkernel, schedule = "IS"} {
              ^bb0(%in: f32, %in_30: f32, %out: f32):
                %15 = arith.mulf %in, %in_30 : f32
                %16 = arith.addf %15, %out : f32
                linalg.yield %16 : f32
              } -> tensor<1x8x16xf32>
              %inserted_slice_29 = tensor.insert_slice %14 into %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x8x16xf32> into tensor<1x336x784xf32>
              scf.yield %inserted_slice_29 : tensor<1x336x784xf32>
            }
            scf.yield %12 : tensor<1x336x784xf32>
          }
          %inserted_slice_23 = tensor.insert_slice %8 into %arg10[%arg3, %arg9, %arg7] [1, 336, 784] [1, 1, 1] : tensor<1x336x784xf32> into tensor<1x336x784xf32>
          scf.yield %inserted_slice_23 : tensor<1x336x784xf32>
        }
        scf.yield %4 : tensor<1x336x784xf32>
      }
      scf.yield %3 : tensor<1x336x784xf32>
    }
    scf.yield %2 : tensor<1x336x784xf32>
  }
  %inserted_slice = tensor.insert_slice %0 into %collapsed_0[0, 0, 0] [1, 336, 784] [1, 1, 1] : tensor<1x336x784xf32> into tensor<1x336x784xf32>
  %extracted_slice_4 = tensor.extract_slice %collapsed[0, 320, 0] [1, 12, 900] [1, 1, 1] : tensor<1x332x900xf32> to tensor<1x12x900xf32>
  %extracted_slice_5 = tensor.extract_slice %arg1[0, 320, 0, 0] [336, 12, 3, 3] [1, 1, 1, 1] : tensor<336x332x3x3xf32> to tensor<336x12x3x3xf32>
  %extracted_slice_6 = tensor.extract_slice %inserted_slice[0, 0, 0] [1, 336, 784] [1, 1, 1] : tensor<1x336x784xf32> to tensor<1x336x784xf32>
  %c0_7 = arith.constant 0 : index
  %c1_8 = arith.constant 1 : index
  %c1_9 = arith.constant 1 : index
  %1 = scf.for %arg3 = %c0_7 to %c1_8 step %c1_9 iter_args(%arg4 = %extracted_slice_6) -> (tensor<1x336x784xf32>) {
    %c0_11 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c12_12 = arith.constant 12 : index
    %2 = scf.for %arg5 = %c0_11 to %c12 step %c12_12 iter_args(%arg6 = %arg4) -> (tensor<1x336x784xf32>) {
      %c0_13 = arith.constant 0 : index
      %c784 = arith.constant 784 : index
      %c784_14 = arith.constant 784 : index
      %3 = scf.for %arg7 = %c0_13 to %c784 step %c784_14 iter_args(%arg8 = %arg6) -> (tensor<1x336x784xf32>) {
        %c0_15 = arith.constant 0 : index
        %c336 = arith.constant 336 : index
        %c336_16 = arith.constant 336 : index
        %4 = scf.for %arg9 = %c0_15 to %c336 step %c336_16 iter_args(%arg10 = %arg8) -> (tensor<1x336x784xf32>) {
          %5 = affine.apply #map(%arg7)
          %extracted_slice_17 = tensor.extract_slice %extracted_slice_4[%arg3, %arg5, %5] [1, 12, 900] [1, 1, 1] : tensor<1x12x900xf32> to tensor<1x12x900xf32>
          %extracted_slice_18 = tensor.extract_slice %extracted_slice_5[%arg9, %arg5, 0, 0] [336, 12, 3, 3] [1, 1, 1, 1] : tensor<336x12x3x3xf32> to tensor<336x12x3x3xf32>
          %extracted_slice_19 = tensor.extract_slice %arg10[%arg3, %arg9, %arg7] [1, 336, 784] [1, 1, 1] : tensor<1x336x784xf32> to tensor<1x336x784xf32>
          %c0_20 = arith.constant 0 : index
          %c336_21 = arith.constant 336 : index
          %c8 = arith.constant 8 : index
          %c0_22 = arith.constant 0 : index
          %c784_23 = arith.constant 784 : index
          %c16 = arith.constant 16 : index
          %6 = tensor.empty() : tensor<42x12x3x3x8xf32>
          %7 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%6 : tensor<42x12x3x3x8xf32>) attrs =  {multipacking = true, packing = "filter"} {
          ^bb0(%out: f32):
            %9 = linalg.index 0 : index
            %10 = linalg.index 1 : index
            %11 = linalg.index 2 : index
            %12 = linalg.index 3 : index
            %13 = linalg.index 4 : index
            %14 = affine.apply #map2(%9, %13)
            %extracted = tensor.extract %extracted_slice_18[%14, %10, %11, %12] : tensor<336x12x3x3xf32>
            linalg.yield %extracted : f32
          } -> tensor<42x12x3x3x8xf32>
          %8 = scf.for %arg11 = %c0_22 to %c784_23 step %c16 iter_args(%arg12 = %extracted_slice_19) -> (tensor<1x336x784xf32>) {
            %9 = affine.apply #map3(%arg11)[%arg7]
            %extracted_slice_25 = tensor.extract_slice %extracted_slice_17[0, 0, %9] [1, 12, 78] [1, 1, 1] : tensor<1x12x900xf32> to tensor<1x12x78xf32>
            %10 = tensor.empty() : tensor<1x12x3x3x16xf32>
            %11 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%10 : tensor<1x12x3x3x16xf32>) attrs =  {multipacking = false, packing = "input"} {
            ^bb0(%out: f32):
              %13 = linalg.index 0 : index
              %14 = linalg.index 1 : index
              %15 = linalg.index 2 : index
              %16 = linalg.index 3 : index
              %17 = linalg.index 4 : index
              %18 = affine.apply #map4(%arg11)[%arg7]
              %19 = affine.apply #map5(%15, %16, %17)[%18]
              %extracted = tensor.extract %extracted_slice_25[%13, %14, %19] : tensor<1x12x78xf32>
              linalg.yield %extracted : f32
            } -> tensor<1x12x3x3x16xf32>
            %collapsed_26 = tensor.collapse_shape %11 [[0], [1, 2, 3], [4]] : tensor<1x12x3x3x16xf32> into tensor<1x108x16xf32>
            %12 = scf.for %arg13 = %c0_20 to %c336_21 step %c8 iter_args(%arg14 = %arg12) -> (tensor<1x336x784xf32>) {
              %13 = affine.apply #map6(%arg13)
              %extracted_slice_27 = tensor.extract_slice %7[%13, 0, 0, 0, 0] [1, 12, 3, 3, 8] [1, 1, 1, 1, 1] : tensor<42x12x3x3x8xf32> to tensor<1x12x3x3x8xf32>
              %collapsed_28 = tensor.collapse_shape %extracted_slice_27 [[0, 1, 2, 3], [4]] : tensor<1x12x3x3x8xf32> into tensor<108x8xf32>
              %extracted_slice_29 = tensor.extract_slice %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x336x784xf32> to tensor<1x8x16xf32>
              %14 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_26, %collapsed_28 : tensor<1x108x16xf32>, tensor<108x8xf32>) outs(%extracted_slice_29 : tensor<1x8x16xf32>) attrs =  {microkernel, schedule = "IS"} {
              ^bb0(%in: f32, %in_31: f32, %out: f32):
                %15 = arith.mulf %in, %in_31 : f32
                %16 = arith.addf %15, %out : f32
                linalg.yield %16 : f32
              } -> tensor<1x8x16xf32>
              %inserted_slice_30 = tensor.insert_slice %14 into %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x8x16xf32> into tensor<1x336x784xf32>
              scf.yield %inserted_slice_30 : tensor<1x336x784xf32>
            }
            scf.yield %12 : tensor<1x336x784xf32>
          }
          %inserted_slice_24 = tensor.insert_slice %8 into %arg10[%arg3, %arg9, %arg7] [1, 336, 784] [1, 1, 1] : tensor<1x336x784xf32> into tensor<1x336x784xf32>
          scf.yield %inserted_slice_24 : tensor<1x336x784xf32>
        }
        scf.yield %4 : tensor<1x336x784xf32>
      }
      scf.yield %3 : tensor<1x336x784xf32>
    }
    scf.yield %2 : tensor<1x336x784xf32>
  }
  %inserted_slice_10 = tensor.insert_slice %1 into %inserted_slice[0, 0, 0] [1, 336, 784] [1, 1, 1] : tensor<1x336x784xf32> into tensor<1x336x784xf32>
  %expanded = tensor.expand_shape %inserted_slice_10 [[0], [1], [2, 3]] output_shape [1, 336, 28, 28] : tensor<1x336x784xf32> into tensor<1x336x28x28xf32>
  return %expanded : tensor<1x336x28x28xf32>
}
