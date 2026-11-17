//===-- LowerToPack.cpp - Lower input packing to a fast runtime call ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the `transform.lower.to_pack` operation of the SConv
// transform dialect extension. It replaces a bufferized *input-packing*
// `linalg.generic` (the one produced by `applyInputPacking`, tagged with
// `packing = "input"`, `multipacking = false` and `sconv.pack_geom`) with a call
// to a row-segmented C implementation.
//
// Why: the generic is a scalar gather with per-element affine index math
// (floordiv/floormod by the output width, a `memref.reinterpret_cast` and a
// `memref.load` per element). On the target machine it runs at ~0.5 GB/s while
// the same copies reach 13 GB/s with contiguous row moves.
//
// Correctness strategy: rather than re-deriving the absolute address from
// geometry (padding, epilogue split, schedule), the call is given the *offset of
// the tile's first element*, obtained by re-evaluating the existing affine apply
// chain with the tile indices (linalg.index) set to zero. The C side then only
// applies the *relative* geometry, which is recorded on the op as
// `sconv.pack_geom = [Ow, Iw, strideH, strideW, dilationH, dilationW]`.
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <optional>

#include "SConv.h"
#include <llvm/ADT/Twine.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlir::linalg;
using namespace mlir::LLVM;
using namespace mlir::memref;
using namespace mlir::transform;

using LLVMCallOp = LLVM::CallOp;

#define DEBUG_TYPE "sconv-lower-to-pack"

namespace {

/// Number of i64 scalars passed to the runtime packing helper.
constexpr unsigned kNumScalars = 13;

static FailureOr<LLVMFuncOp> getOrCreatePackFuncOp(TransformRewriter &rewriter,
                                                   LowerToPackOp lowering,
                                                   ModuleOp module) {
  auto funcName = lowering.getName();
  for (auto func : module.getOps<LLVMFuncOp>())
    if (func.getName() == funcName)
      return func;

  rewriter.setInsertionPointToEnd(module.getBody());
  auto ctx = module.getContext();
  auto loc = module.getLoc();
  auto ptrTy = LLVMPointerType::get(ctx);

  SmallVector<Type, 16> argumentTys = {/*dst:*/ ptrTy, /*src:*/ ptrTy};
  for (unsigned i = 0; i < kNumScalars; ++i)
    argumentTys.push_back(IntegerType::get(ctx, 64));

  auto funcOp = rewriter.create<LLVMFuncOp>(
      loc, funcName,
      LLVMFunctionType::get(IntegerType::get(ctx, 32), argumentTys));
  return funcOp;
}

/// Clone the chain that computes `v` with every `linalg.index` replaced by 0.
/// Loop induction variables and other outer values are kept as-is. This yields
/// the address of the tile's first element (ic = fh = fw = w = 0).
static Value cloneWithZeroTileIndices(Value v, OpBuilder &builder,
                                      DenseMap<Value, Value> &memo) {
  if (Value known = memo.lookup(v))
    return known;

  Value result = v;
  Operation *def = v.getDefiningOp();
  if (def) {
    if (isa<linalg::IndexOp>(def)) {
      result = builder.create<arith::ConstantIndexOp>(v.getLoc(), 0);
    } else if (auto apply = dyn_cast<AffineApplyOp>(def)) {
      SmallVector<Value> operands;
      for (Value operand : apply.getMapOperands())
        operands.push_back(cloneWithZeroTileIndices(operand, builder, memo));
      result = builder.create<AffineApplyOp>(apply.getLoc(),
                                             apply.getAffineMap(), operands);
    } else if (def->getNumResults() == 1 && def->getNumOperands() > 0 &&
               isa<ArithDialect>(def->getDialect()) &&
               isMemoryEffectFree(def)) {
      Operation *clone = builder.clone(*def);
      for (auto [idx, operand] : llvm::enumerate(def->getOperands()))
        clone->setOperand(idx,
                          cloneWithZeroTileIndices(operand, builder, memo));
      result = clone->getResult(0);
    }
  }

  memo[v] = result;
  return result;
}

/// Materialize the offset of a `memref.reinterpret_cast` view as a value.
static Value getReinterpretCastOffset(OpBuilder &builder,
                                      ReinterpretCastOp rc) {
  if (auto dynamic = rc.getOffsets(); !dynamic.empty())
    return dynamic.front();
  if (auto staticOffsets = rc.getStaticOffsets();
      staticOffsets.size() == 1 && !ShapedType::isDynamic(staticOffsets[0]))
    return builder.create<arith::ConstantIndexOp>(rc.getLoc(),
                                                  staticOffsets[0]);
  return {};
}

/// Find the `memref.reinterpret_cast` whose (single) element is loaded by the
/// packing body. Returns null if the body does not have the expected shape.
static ReinterpretCastOp findLoadedReinterpretCast(GenericOp packingOp) {
  ReinterpretCastOp found = nullptr;
  packingOp.getRegion().walk([&](memref::LoadOp load) {
    Value memref = load.getMemRef();
    if (auto rc = memref.getDefiningOp<ReinterpretCastOp>())
      found = rc;
  });
  return found;
}

/// Create `ptr = base + offsetInElems * elemsize` from a memref value.
static Value createPtrAtElementOffset(OpBuilder &builder, Value memref,
                                      Value offsetInElems) {
  auto loc = memref.getLoc();
  auto ctx = memref.getContext();
  auto memrefTy = cast<MemRefType>(memref.getType());
  int64_t elemBytes = memrefTy.getElementTypeBitWidth() / 8;

  Value base = builder.create<ExtractAlignedPointerAsIndexOp>(loc, memref);
  Value memrefOffset =
      builder.create<ExtractStridedMetadataOp>(loc, memref).getOffset();
  Value total = builder.create<arith::AddIOp>(loc, memrefOffset,
                                              offsetInElems);
  Value bytes = builder.create<arith::MulIOp>(
      loc, total, builder.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value addr = builder.create<arith::AddIOp>(loc, base, bytes);
  Value asI64 = builder.create<IndexCastOp>(loc, builder.getI64Type(), addr);
  return builder.create<IntToPtrOp>(loc, LLVMPointerType::get(ctx), asI64);
}

static FailureOr<Operation *> lowerPackingToCall(TransformRewriter &rewriter,
                                                 LowerToPackOp lowering,
                                                 GenericOp packingOp) {
  auto module = packingOp->getParentOfType<ModuleOp>();
  if (!module)
    return packingOp.emitError() << "payload must be inside a module";

  if (!packingOp->hasAttr("packing"))
    return packingOp.emitError() << "expected `packing` attribute";
  if (packingOp->getAttrOfType<StringAttr>("packing").getValue() != "input")
    return packingOp.emitError() << "only `packing = \"input\"` is supported";
  if (auto mp = packingOp->getAttrOfType<BoolAttr>("multipacking");
      mp && mp.getValue())
    return packingOp.emitError()
           << "multipacking is not supported by `lower.to_pack`";

  auto geomAttr = packingOp->getAttrOfType<DenseI64ArrayAttr>("sconv.pack_geom");
  if (!geomAttr || geomAttr.size() != 6)
    return packingOp.emitError()
           << "expected `sconv.pack_geom = [Ow, Iw, strideH, strideW, "
              "dilationH, dilationW]`";
  ArrayRef<int64_t> geom = geomAttr.asArrayRef();
  int64_t Ow = geom[0], Iw = geom[1];
  int64_t strideH = geom[2], strideW = geom[3];
  int64_t dilationH = geom[4], dilationW = geom[5];

  // Destination tile: [1, Nc, Fh, Fw, Nwin].
  Value dst = packingOp.getOutputs().front();
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!dstTy || dstTy.getRank() != 5 || !dstTy.hasStaticShape())
    return packingOp.emitError()
           << "expected a rank-5 static destination tile [1, Nc, Fh, Fw, Nwin]";
  ArrayRef<int64_t> dstShape = dstTy.getShape();
  if (dstShape[0] != 1)
    return packingOp.emitError() << "expected destination batch dim == 1";
  int64_t Nc = dstShape[1], Fh = dstShape[2], Fw = dstShape[3];
  int64_t Nwin = dstShape[4];

  // Source: the padded input plane read by the generic.
  ReinterpretCastOp rc = findLoadedReinterpretCast(packingOp);
  if (!rc)
    return packingOp.emitError()
           << "expected the body to load through `memref.reinterpret_cast`";
  ArrayRef<int64_t> sizes = rc.getStaticSizes();
  ArrayRef<int64_t> strides = rc.getStaticStrides();
  if (sizes.size() != 3 || strides.size() != 3)
    return packingOp.emitError() << "expected a rank-3 source view";
  for (int64_t s : sizes)
    if (s != 1)
      return packingOp.emitError() << "expected unit source view sizes";
  int64_t srcCStride = strides[1];

  Value srcMemRef = rc.getSource();
  auto srcTy = dyn_cast<MemRefType>(srcMemRef.getType());
  if (!srcTy || !srcTy.hasStaticShape())
    return packingOp.emitError() << "expected a static source memref";
  int64_t srcNumElemsConstant = 1;
  for (int64_t dim : srcTy.getShape())
    srcNumElemsConstant *= dim;

  // Declare the callee first: creating the declaration moves the insertion
  // point to the end of the module.
  auto callee = getOrCreatePackFuncOp(rewriter, lowering, module);
  if (failed(callee))
    return packingOp.emitError()
           << "failed to create declaration to packing function";
  rewriter.setInsertionPointAfter(packingOp);

  // Offset of the tile's first element, with all linalg.index set to zero.
  Value rcOffset = getReinterpretCastOffset(rewriter, rc);
  if (!rcOffset)
    return packingOp.emitError() << "expected a dynamic or static offset view";
  DenseMap<Value, Value> memo;
  Value off0Idx = cloneWithZeroTileIndices(rcOffset, rewriter, memo);
  Value off0 = rewriter.create<IndexCastOp>(packingOp.getLoc(),
                                            rewriter.getI64Type(), off0Idx);

  Value srcNumElems = rewriter.create<arith::ConstantIntOp>(
      packingOp.getLoc(), srcNumElemsConstant, 64);
  Value srcRemain =
      rewriter.create<arith::SubIOp>(packingOp.getLoc(), srcNumElems, off0);
  Value srcPtr = createPtrAtElementOffset(rewriter, srcMemRef, off0Idx);
  Value dstPtr = createPtrAtElementOffset(
      rewriter, dst,
      rewriter.create<arith::ConstantIndexOp>(packingOp.getLoc(), 0));

  auto i64 = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(packingOp.getLoc(), value, 64);
  };

  auto call = rewriter.create<LLVMCallOp>(
      packingOp.getLoc(), *callee,
      ValueRange{dstPtr, srcPtr, off0, i64(Nc), i64(Fh), i64(Fw), i64(Nwin),
                 i64(Ow), i64(Iw), i64(strideH), i64(strideW), i64(dilationH),
                 i64(dilationW), i64(srcCStride), srcRemain});

  rewriter.eraseOp(packingOp);
  return call.getOperation();
}

} // namespace

DiagnosedSilenceableFailure LowerToPackOp::apply(TransformRewriter &rewriter,
                                                 TransformResults &results,
                                                 TransformState &state) {
  SmallVector<Operation *, 4> resultOps;
  for (Operation *targetOp : state.getPayloadOps(getTarget())) {
    auto packing = dyn_cast<GenericOp>(targetOp);
    if (!packing) {
      targetOp->emitError() << "expected `linalg.generic` operation";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    auto result = lowerPackingToCall(rewriter, *this, packing);
    if (failed(result)) {
      emitError() << "failed to lower input packing";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    resultOps.push_back(*result);
  }
  results.set((*this)->getOpResult(0), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void LowerToPackOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult LowerToPackOp::verify() {
  if (this->getName().size() == 0)
    return emitOpError("expected non-empty kernel name");
  return success();
}
