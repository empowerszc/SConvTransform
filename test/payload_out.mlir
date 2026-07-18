// sconv-opt -transform=test/sconv_mK.mlir test/payload.mlir
#map = affine_map<(d0) -> ((d0 floordiv 64) * 66 + d0 mod 64)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map3 = affine_map<(d0)[s0] -> (((d0 + s0) floordiv 64 - s0 floordiv 64) * 66 + (d0 + s0) mod 64 - s0 mod 64)>
#map4 = affine_map<(d0)[s0] -> (d0 + s0)>
#map5 = affine_map<(d0, d1, d2)[s0] -> (((d2 + s0) floordiv 64 - s0 floordiv 64 + d0) * 66 + (d2 + s0) mod 64 - s0 mod 64 + d1)>
#map6 = affine_map<(d0) -> (d0 floordiv 8)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<1x128x66x66xf32>, %arg1: tensor<256x128x3x3xf32>, %arg2: tensor<1x256x64x64xf32>) -> tensor<1x256x64x64xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<1x128x66x66xf32> into tensor<1x128x4356xf32>
    %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<1x256x64x64xf32> into tensor<1x256x4096xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %0 = scf.for %arg3 = %c0 to %c1 step %c1_1 iter_args(%arg4 = %collapsed_0) -> (tensor<1x256x4096xf32>) {
      %c0_2 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %1 = scf.for %arg5 = %c0_2 to %c128 step %c32 iter_args(%arg6 = %arg4) -> (tensor<1x256x4096xf32>) {
        %c0_3 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c1024 = arith.constant 1024 : index
        %2 = scf.for %arg7 = %c0_3 to %c4096 step %c1024 iter_args(%arg8 = %arg6) -> (tensor<1x256x4096xf32>) {
          %c0_4 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c256_5 = arith.constant 256 : index
          %3 = scf.for %arg9 = %c0_4 to %c256 step %c256_5 iter_args(%arg10 = %arg8) -> (tensor<1x256x4096xf32>) {
            %4 = affine.apply #map(%arg7)
            %extracted_slice = tensor.extract_slice %collapsed[%arg3, %arg5, %4] [1, 32, 1188] [1, 1, 1] : tensor<1x128x4356xf32> to tensor<1x32x1188xf32>
            %extracted_slice_6 = tensor.extract_slice %arg1[%arg9, %arg5, 0, 0] [256, 32, 3, 3] [1, 1, 1, 1] : tensor<256x128x3x3xf32> to tensor<256x32x3x3xf32>
            %extracted_slice_7 = tensor.extract_slice %arg10[%arg3, %arg9, %arg7] [1, 256, 1024] [1, 1, 1] : tensor<1x256x4096xf32> to tensor<1x256x1024xf32>
            %c0_8 = arith.constant 0 : index
            %c256_9 = arith.constant 256 : index
            %c8 = arith.constant 8 : index
            %c0_10 = arith.constant 0 : index
            %c1024_11 = arith.constant 1024 : index
            %c16 = arith.constant 16 : index
            %5 = tensor.empty() : tensor<32x32x3x3x8xf32>
            %6 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%5 : tensor<32x32x3x3x8xf32>) attrs =  {multipacking = true, packing = "filter"} {
            ^bb0(%out: f32):
              %8 = linalg.index 0 : index
              %9 = linalg.index 1 : index
              %10 = linalg.index 2 : index
              %11 = linalg.index 3 : index
              %12 = linalg.index 4 : index
              %13 = affine.apply #map2(%8, %12)
              %extracted = tensor.extract %extracted_slice_6[%13, %9, %10, %11] : tensor<256x32x3x3xf32>
              linalg.yield %extracted : f32
            } -> tensor<32x32x3x3x8xf32>
            %7 = scf.for %arg11 = %c0_10 to %c1024_11 step %c16 iter_args(%arg12 = %extracted_slice_7) -> (tensor<1x256x1024xf32>) {
              %8 = affine.apply #map3(%arg11)[%arg7]
              %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, 0, %8] [1, 32, 150] [1, 1, 1] : tensor<1x32x1188xf32> to tensor<1x32x150xf32>
              %9 = tensor.empty() : tensor<1x32x3x3x16xf32>
              %10 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%9 : tensor<1x32x3x3x16xf32>) attrs =  {multipacking = false, packing = "input"} {
              ^bb0(%out: f32):
                %12 = linalg.index 0 : index
                %13 = linalg.index 1 : index
                %14 = linalg.index 2 : index
                %15 = linalg.index 3 : index
                %16 = linalg.index 4 : index
                %17 = affine.apply #map4(%arg11)[%arg7]
                %18 = affine.apply #map5(%14, %15, %16)[%17]
                %extracted = tensor.extract %extracted_slice_12[%12, %13, %18] : tensor<1x32x150xf32>
                linalg.yield %extracted : f32
              } -> tensor<1x32x3x3x16xf32>
              %collapsed_13 = tensor.collapse_shape %10 [[0], [1, 2, 3], [4]] : tensor<1x32x3x3x16xf32> into tensor<1x288x16xf32>
              %11 = scf.for %arg13 = %c0_8 to %c256_9 step %c8 iter_args(%arg14 = %arg12) -> (tensor<1x256x1024xf32>) {
                %12 = affine.apply #map6(%arg13)
                %extracted_slice_14 = tensor.extract_slice %6[%12, 0, 0, 0, 0] [1, 32, 3, 3, 8] [1, 1, 1, 1, 1] : tensor<32x32x3x3x8xf32> to tensor<1x32x3x3x8xf32>
                %collapsed_15 = tensor.collapse_shape %extracted_slice_14 [[0, 1, 2, 3], [4]] : tensor<1x32x3x3x8xf32> into tensor<288x8xf32>
                %extracted_slice_16 = tensor.extract_slice %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x256x1024xf32> to tensor<1x8x16xf32>
                %13 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_13, %collapsed_15 : tensor<1x288x16xf32>, tensor<288x8xf32>) outs(%extracted_slice_16 : tensor<1x8x16xf32>) attrs =  {microkernel, schedule = "IS"} {
                ^bb0(%in: f32, %in_18: f32, %out: f32):
                  %14 = arith.mulf %in, %in_18 : f32
                  %15 = arith.addf %14, %out : f32
                  linalg.yield %15 : f32
                } -> tensor<1x8x16xf32>
                %inserted_slice_17 = tensor.insert_slice %13 into %arg14[0, %arg13, %arg11] [1, 8, 16] [1, 1, 1] : tensor<1x8x16xf32> into tensor<1x256x1024xf32>
                scf.yield %inserted_slice_17 : tensor<1x256x1024xf32>
              }
              scf.yield %11 : tensor<1x256x1024xf32>
            }
            %inserted_slice = tensor.insert_slice %7 into %arg10[%arg3, %arg9, %arg7] [1, 256, 1024] [1, 1, 1] : tensor<1x256x1024xf32> into tensor<1x256x4096xf32>
            scf.yield %inserted_slice : tensor<1x256x4096xf32>
          }
          scf.yield %3 : tensor<1x256x4096xf32>
        }
        scf.yield %2 : tensor<1x256x4096xf32>
      }
      scf.yield %1 : tensor<1x256x4096xf32>
    }
    %expanded = tensor.expand_shape %0 [[0], [1], [2, 3]] output_shape [1, 256, 64, 64] : tensor<1x256x4096xf32> into tensor<1x256x64x64xf32>
    return %expanded : tensor<1x256x64x64xf32>
  }
}
