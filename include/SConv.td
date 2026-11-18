//===-- SConv.td - Transform dialect SConv -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the SConv.
//
//===----------------------------------------------------------------------===//

#ifndef SCONV
#define SCONV

include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

// Define the sconv operation. By convention, prefix its name with the name of the 
// extension, "structured.". The full op name will be further prefixed with "transform.".
def SConvOp : Op<Transform_Dialect, "structured.sconv",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  // Provide a brief and a full description of SConvOp. It is recommended describing 
  // the effects on the operands and how the operation processes various failure modes.
  let summary = "Improving Direct Convolution through Tensor Slicing, VB Packing.";
  let description = [{
    SConv is a direct-convolution algorithm based on an MLIR/LLVM code-generation
    toolchain that can be integrated into machine-learning compilers.
    This algorithm introduces: (a) Convolution Slicing Analysis (CSA) — a convolution-
    specific 3D cache-blocking analysis pass that focuses on tile reuse over the cache
    hierarchy; and (b) Convolution Slicing Optimization (CSO) — a transform pass that
    uses CSA to generate a tiled and packed direct-convolution micro-kernel.
  }];

  // The argument include the handle to the payload operation, with optional mK_info, 
  // arch_info & latency configurations:
  //   mK_info = { nwindows, num_filters [, nc] }
  //   arch_info = { l1size, l2size, l3size, cache_line }
  //   latency = { l1_latency, l2_latency, l3_latency, mem_latency }
  //
  // A zero in `mK_info` selects the value automatically for each convolution
  // from its shape: `mK_info = [0, 0]` picks the widest window/filter tile that
  // divides the output spatial size and the filter count (window candidates
  // 160/128/80/64/32/16, filter candidates 32/16/8). The optional third value
  // overrides the input-channel tile (0/absent = auto: wide tiles target a
  // longer K that still fits L2). Explicit values are used as given.

  // The handle must implement TransformHandleTypeInterface.   

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    OptionalAttr<ArrayAttr>:$mK_info,
    OptionalAttr<ArrayAttr>:$arch_info,
    OptionalAttr<ArrayAttr>:$latency
  );

  let results = (outs
    TransformHandleTypeInterface:$output_convs,     // container of uKernels
    Variadic<TransformHandleTypeInterface>:$loops   // container of associated loops
  );

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let hasVerifier = 1;

}

def LowerToBlasOp : Op<Transform_Dialect, "lower.to_blas",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Lower micro-kernels to BLAS kernel routines.";
  let description = [{}];
  let arguments = (ins
    StrAttr:$name,
    TransformHandleTypeInterface:$target
  );
  let results = (outs TransformHandleTypeInterface:$output);
  let assemblyFormat = "$name `,` $target attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LowerToPackOp : Op<Transform_Dialect, "lower.to_pack",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Lower an input-packing generic to a fast runtime packing call.";
  let description = [{
    Replaces a bufferized input-packing `linalg.generic` (the one tagged with
    `packing = "input"`, `multipacking = false` and `sconv.pack_geom`) by a call
    to a row-segmented C implementation (`pack_input_tile`) that reaches
    memory-copy throughput instead of the scalar gather emitted by the generic.

    The generic's per-element index chain is preserved exactly: the call takes
    the offset of the tile's first element (evaluated by zeroing the tile
    indices of the existing affine apply chain) plus the relative geometry from
    `sconv.pack_geom`.
  }];
  let arguments = (ins
    StrAttr:$name,
    TransformHandleTypeInterface:$target
  );
  let results = (outs TransformHandleTypeInterface:$output);
  let assemblyFormat = "$name `,` $target attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def LowerToSMEOp : Op<Transform_Dialect, "lower.to_sme",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Lower an SME-selected whole conv to the sconv_sme_direct runtime call.";
  let description = [{}];
  let arguments = (ins
    StrAttr:$name,
    TransformHandleTypeInterface:$target
  );
  let results = (outs TransformHandleTypeInterface:$output);
  let assemblyFormat = "$name `,` $target attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def SMESelectOp : Op<Transform_Dialect, "lower.sme_select",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Tag conv ops that should take the SME direct path.";
  let description = [{
    Upon each `linalg.conv_2d_nchw_fchw` in `target`, decides whether the SME
    direct kernel should handle it (static 3x3, stride 1, unit dilation,
    pre-padded input `Ih==Oh+2 && Iw==Ow+2`) and, if so, whether the calibrated
    cost model plus the `min_ic`/`max_oc` gates prefer SME over GEMM. Selected
    convs get the `sme.direct` marker attribute. Returns the handle of the
    convs that were NOT selected (the GEMM remainder).
  }];
  let arguments = (ins
    TransformHandleTypeInterface:$target,
    UnitAttr:$force,
    OptionalAttr<I64Attr>:$min_ic,
    OptionalAttr<I64Attr>:$max_oc
  );
  let results = (outs TransformHandleTypeInterface:$remaining);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // SCONV
