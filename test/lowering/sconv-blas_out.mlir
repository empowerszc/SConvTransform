// sconv-opt -transform=test/lowering/sconv-blas.mlir test/payload.mlir
#map = affine_map<()[s0, s1, s2] -> (s0 * 1048576 + s1 * 4096 + s2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<()[s0, s1, s2, s3, s4] -> (s0 + s1 * s2 + s3 * s4)>
#map3 = affine_map<(d0, d1)[s0, s1, s2, s3, s4, s5, s6, s7] -> ((d0 * 8 + d1) * s1 + s0 + s2 * s3 + s4 * s5 + s6 * s7)>
#map4 = affine_map<(d0)[s0, s1] -> (d0 + s0 * 557568 + s1 * 4356 + (d0 floordiv 64) * 2)>
#map5 = affine_map<(d0)[s0, s1] -> (d0 + s0 + ((d0 + s1) floordiv 64) * 2 - (s1 floordiv 64) * 2)>
#map6 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * 66 + d1 + d2 + s0 + s1 * 557568 + s2 * 4356 + ((d2 + d3 + s3) floordiv 64) * 2 - ((d3 + s3) floordiv 64) * 2)>
#map7 = affine_map<(d0) -> ((d0 floordiv 8) * 2304)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 4096 + s2)>
#map9 = affine_map<()[s0, s1] -> (s0 + s1 * 4)>
module {
  func.func @conv_2d_nchw_fchw(%arg0: memref<1x128x66x66xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<256x128x3x3xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg2: memref<1x256x64x64xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x256x64x64xf32> {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x66x66xf32>
    memref.copy %arg0, %alloc : memref<1x128x66x66xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x128x66x66xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x256x64x64xf32>
    memref.copy %arg2, %alloc_0 : memref<1x256x64x64xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x256x64x64xf32>
    %reinterpret_cast = memref.reinterpret_cast %alloc_0 to offset: [0], sizes: [1, 256, 4096], strides: [1048576, 4096, 1] : memref<1x256x64x64xf32> to memref<1x256x4096xf32>
    %0 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg4 = %reinterpret_cast) -> (memref<1x256x4096xf32>) {
      %1 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %arg4) -> (memref<1x256x4096xf32>) {
        %2 = scf.for %arg7 = %c0 to %c4096 step %c1024 iter_args(%arg8 = %arg6) -> (memref<1x256x4096xf32>) {
          %3 = scf.for %arg9 = %c0 to %c256 step %c256 iter_args(%arg10 = %arg8) -> (memref<1x256x4096xf32>) {
            %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg10 : memref<1x256x4096xf32> -> memref<f32>, index, index, index, index, index, index, index
            %4 = affine.apply #map()[%arg3, %arg9, %arg7]
            %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [%4], sizes: [1, 256, 1024], strides: [1048576, 4096, 1] : memref<f32> to memref<1x256x1024xf32, strided<[1048576, 4096, 1], offset: ?>>
            %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32x32x3x3x8xf32>
            linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%alloc_7 : memref<32x32x3x3x8xf32>) attrs =  {multipacking = true, packing = "filter"} {
            ^bb0(%out: f32):
              %6 = linalg.index 0 : index
              %7 = linalg.index 1 : index
              %8 = linalg.index 2 : index
              %9 = linalg.index 3 : index
              %10 = linalg.index 4 : index
              %base_buffer_8, %offset_9, %sizes_10:4, %strides_11:4 = memref.extract_strided_metadata %arg1 : memref<256x128x3x3xf32, strided<[?, ?, ?, ?], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index, index, index
              %11 = affine.apply #map2()[%offset_9, %arg9, %strides_11#0, %arg5, %strides_11#1]
              %12 = affine.apply #map3(%6, %10)[%11, %strides_11#0, %7, %strides_11#1, %8, %strides_11#2, %9, %strides_11#3]
              %reinterpret_cast_12 = memref.reinterpret_cast %base_buffer_8 to offset: [%12], sizes: [1, 1, 1, 1], strides: [%strides_11#0, %strides_11#1, %strides_11#2, %strides_11#3] : memref<f32> to memref<1x1x1x1xf32, strided<[?, ?, ?, ?], offset: ?>>
              %13 = memref.load %reinterpret_cast_12[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32, strided<[?, ?, ?, ?], offset: ?>>
              linalg.yield %13 : f32
            }
            %5 = scf.for %arg11 = %c0 to %c1024 step %c16 iter_args(%arg12 = %reinterpret_cast_6) -> (memref<1x256x1024xf32, strided<[1048576, 4096, 1], offset: ?>>) {
              %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x32x3x3x16xf32>
              linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%alloc_8 : memref<1x32x3x3x16xf32>) attrs =  {multipacking = false, packing = "input"} {
              ^bb0(%out: f32):
                %7 = linalg.index 0 : index
                %8 = linalg.index 1 : index
                %9 = linalg.index 2 : index
                %10 = linalg.index 3 : index
                %11 = linalg.index 4 : index
                %12 = affine.apply #map4(%arg7)[%arg3, %arg5]
                %13 = affine.apply #map5(%arg11)[%12, %arg7]
                %14 = affine.apply #map6(%9, %10, %11, %arg11)[%13, %7, %8, %arg7]
                %reinterpret_cast_10 = memref.reinterpret_cast %alloc to offset: [%14], sizes: [1, 1, 1], strides: [557568, 4356, 1] : memref<1x128x66x66xf32> to memref<1x1x1xf32, strided<[557568, 4356, 1], offset: ?>>
                %15 = memref.load %reinterpret_cast_10[%c0, %c0, %c0] : memref<1x1x1xf32, strided<[557568, 4356, 1], offset: ?>>
                linalg.yield %15 : f32
              }
              %reinterpret_cast_9 = memref.reinterpret_cast %alloc_8 to offset: [0], sizes: [1, 288, 16], strides: [4608, 16, 1] : memref<1x32x3x3x16xf32> to memref<1x288x16xf32>
              %6 = scf.for %arg13 = %c0 to %c256 step %c8 iter_args(%arg14 = %arg12) -> (memref<1x256x1024xf32, strided<[1048576, 4096, 1], offset: ?>>) {
                %7 = affine.apply #map7(%arg13)
                %reinterpret_cast_10 = memref.reinterpret_cast %alloc_7 to offset: [%7], sizes: [288, 8], strides: [8, 1] : memref<32x32x3x3x8xf32> to memref<288x8xf32, strided<[8, 1], offset: ?>>
                %base_buffer_11, %offset_12, %sizes_13:3, %strides_14:3 = memref.extract_strided_metadata %arg14 : memref<1x256x1024xf32, strided<[1048576, 4096, 1], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
                %8 = affine.apply #map8()[%offset_12, %arg13, %arg11]
                %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [%8], sizes: [1, 8, 16], strides: [1048576, 4096, 1] : memref<f32> to memref<1x8x16xf32, strided<[1048576, 4096, 1], offset: ?>>
                %c16_i64 = arith.constant 16 : i64
                %c8_i64 = arith.constant 8 : i64
                %c288_i64 = arith.constant 288 : i64
                %base_buffer_16, %offset_17, %sizes_18:3, %strides_19:3 = memref.extract_strided_metadata %reinterpret_cast_9 : memref<1x288x16xf32> -> memref<f32>, index, index, index, index, index, index, index
                %intptr = memref.extract_aligned_pointer_as_index %reinterpret_cast_9 : memref<1x288x16xf32> -> index
                %9 = affine.apply #map9()[%intptr, %offset_17]
                %10 = arith.index_cast %9 : index to i64
                %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
                %base_buffer_20, %offset_21, %sizes_22:2, %strides_23:2 = memref.extract_strided_metadata %reinterpret_cast_10 : memref<288x8xf32, strided<[8, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
                %intptr_24 = memref.extract_aligned_pointer_as_index %reinterpret_cast_10 : memref<288x8xf32, strided<[8, 1], offset: ?>> -> index
                %12 = affine.apply #map9()[%intptr_24, %offset_21]
                %13 = arith.index_cast %12 : index to i64
                %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
                %base_buffer_25, %offset_26, %sizes_27:3, %strides_28:3 = memref.extract_strided_metadata %reinterpret_cast_15 : memref<1x8x16xf32, strided<[1048576, 4096, 1], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index
                %intptr_29 = memref.extract_aligned_pointer_as_index %reinterpret_cast_15 : memref<1x8x16xf32, strided<[1048576, 4096, 1], offset: ?>> -> index
                %15 = affine.apply #map9()[%intptr_29, %offset_26]
                %16 = arith.index_cast %15 : index to i64
                %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
                %c4096_i64 = arith.constant 4096 : i64
                %cst = arith.constant 1.000000e+00 : f32
                %18 = llvm.call @sgemm_blas_kernel(%c16_i64, %c8_i64, %c288_i64, %cst, %11, %14, %17, %c4096_i64) {schedule = "IS"} : (i64, i64, i64, f32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
                memref.copy %reinterpret_cast_15, %reinterpret_cast_15 : memref<1x8x16xf32, strided<[1048576, 4096, 1], offset: ?>> to memref<1x8x16xf32, strided<[1048576, 4096, 1], offset: ?>>
                scf.yield %arg14 : memref<1x256x1024xf32, strided<[1048576, 4096, 1], offset: ?>>
              }
              scf.yield %6 : memref<1x256x1024xf32, strided<[1048576, 4096, 1], offset: ?>>
            }
            memref.copy %5, %reinterpret_cast_6 : memref<1x256x1024xf32, strided<[1048576, 4096, 1], offset: ?>> to memref<1x256x1024xf32, strided<[1048576, 4096, 1], offset: ?>>
            scf.yield %arg10 : memref<1x256x4096xf32>
          }
          scf.yield %3 : memref<1x256x4096xf32>
        }
        scf.yield %2 : memref<1x256x4096xf32>
      }
      scf.yield %1 : memref<1x256x4096xf32>
    }
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %0 : memref<1x256x4096xf32> -> memref<f32>, index, index, index, index, index, index, index
    %reinterpret_cast_1 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [1, 256, 64, 64], strides: [1048576, 4096, 64, 1] : memref<f32> to memref<1x256x64x64xf32>
    return %reinterpret_cast_1 : memref<1x256x64x64xf32>
  }
  llvm.func @sgemm_blas_kernel(i64, i64, i64, f32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
}
