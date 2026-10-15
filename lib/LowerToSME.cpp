//===-- LowerToSME.cpp - Lower SME-selected conv to runtime call ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the `transform.lower.to_sme` operation of the SConv
// transform dialect extension. It lowers a (bufferized) conv that carries the
// `sme.direct` marker attribute into a call to the `sconv_sme_direct` C shim
// entry point.
//
//===----------------------------------------------------------------------===//
#include <cassert>

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

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlir::linalg;
using namespace mlir::LLVM;
using namespace mlir::memref;
using namespace mlir::transform;

using LLVMCallOp = LLVM::CallOp;

//===----------------------------------------------------------------------===//
// Lowering to SME
//===----------------------------------------------------------------------===//

namespace {

/// Create a LLVM function declaration for the SME direct-conv shim. The
/// function has the following signature:
///
///   int sconv_sme_direct(long N, long Ic, long Ih, long Iw, long Oc, long Oh,
///                        long Ow, float *in, float *filter, float *out)
///
/// The name of the function comes from the `name` attribute in the transform
/// operation.
static FailureOr<LLVMFuncOp> getOrCreateSmeFuncOp(TransformRewriter &rewriter,
                                                  LowerToSMEOp lowering,
                                                  ModuleOp module) {
  auto funcName = lowering.getName();

  for (auto func : module.getOps<LLVMFuncOp>())
    if (func.getName() == funcName)
      return func;

  rewriter.setInsertionPointToEnd(module.getBody());

  auto ctx = module.getContext();
  auto loc = module.getLoc();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto ptrTy = LLVMPointerType::get(ctx);

  SmallVector<Type, 16> argumentTys = {
      /*N:*/ i64Ty,  /*Ic:*/ i64Ty,     /*Ih:*/ i64Ty, /*Iw:*/ i64Ty,
      /*Oc:*/ i64Ty, /*Oh:*/ i64Ty,     /*Ow:*/ i64Ty,
      /*in:*/ ptrTy, /*filter:*/ ptrTy, /*out:*/ ptrTy,
  };

  return rewriter.create<LLVMFuncOp>(
      loc, funcName, LLVMFunctionType::get(i32Ty, argumentTys));
}

/// Given a value of type Memref, create a value of type LLVM pointer. Mirrors
/// `createPtrWithOffsetFromMemRef` from LowerToBLAS.cpp.
static Value createPtrWithOffsetFromMemRef(OpBuilder builder, Value value,
                                           Twine &name) {
  assert(isa<MemRefType>(value.getType()) && "Expected memref value");

  auto ctx = value.getContext();
  auto loc = value.getLoc();
  Type i64Ty = builder.getI64Type();
  Type ptrTy = LLVMPointerType::get(ctx);
  auto valTy = cast<MemRefType>(value.getType());

  value << name.concat("_tile");

  AffineExpr s_base, s_offset;
  bindSymbols(ctx, s_base, s_offset);

  auto tileAddrMap = AffineMap::get(
      0, 2, s_base + s_offset * (valTy.getElementTypeBitWidth() / 8));

  Value offset =
      builder.create<ExtractStridedMetadataOp>(loc, value).getOffset();

  Value base = builder.create<ExtractAlignedPointerAsIndexOp>(loc, value);
  base << name.concat("_base_ptr");

  Value index_ptr =
      builder.create<AffineApplyOp>(loc, tileAddrMap, ValueRange{base, offset});
  index_ptr << name.concat("_index_ptr");

  Value i64_ptr = builder.create<IndexCastOp>(loc, i64Ty, index_ptr);
  i64_ptr << name.concat("_i64_ptr");

  Value llvm_ptr = builder.create<IntToPtrOp>(loc, ptrTy, i64_ptr);
  llvm_ptr << name.concat("_llvm_ptr");

  return llvm_ptr;
}

static FailureOr<Operation *>
lowerConvToSme(TransformRewriter &rewriter, LowerToSMEOp lowering,
               Conv2DNchwFchwOp conv) {
  auto module = conv->getParentOfType<ModuleOp>();
  if (!module)
    return conv.emitError() << "payload must be inside a module";

  auto callee = getOrCreateSmeFuncOp(rewriter, lowering, module);
  if (failed(callee))
    return conv.emitError()
           << "failed to create declaration to SME function";

  Value inVal = conv.getInputs()[0];
  Value filterVal = conv.getInputs()[1];
  Value outVal = conv.getOutputs()[0];

  auto inTy = cast<ShapedType>(inVal.getType());
  auto fTy = cast<ShapedType>(filterVal.getType());
  auto outTy = cast<ShapedType>(outVal.getType());

  auto createI64 = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(conv.getLoc(), value, 64);
  };
  auto createPtr = [&](Value source, Twine name) -> Value {
    return createPtrWithOffsetFromMemRef(rewriter, source, name);
  };

  rewriter.setInsertionPointAfter(conv);

  Value N = createI64(inTy.getDimSize(0)) << "N";
  Value Ic = createI64(inTy.getDimSize(1)) << "Ic";
  Value Ih = createI64(inTy.getDimSize(2)) << "Ih";
  Value Iw = createI64(inTy.getDimSize(3)) << "Iw";
  Value Oc = createI64(fTy.getDimSize(0)) << "Oc";
  Value Oh = createI64(outTy.getDimSize(2)) << "Oh";
  Value Ow = createI64(outTy.getDimSize(3)) << "Ow";
  Value pIn = createPtr(inVal, "in");
  Value pFilter = createPtr(filterVal, "filter");
  Value pOut = createPtr(outVal, "out");

  auto call = rewriter.create<LLVMCallOp>(
      conv.getLoc(), *callee,
      ValueRange{N, Ic, Ih, Iw, Oc, Oh, Ow, pIn, pFilter, pOut});

  rewriter.eraseOp(conv);

  return call.getOperation();
}

} // namespace

DiagnosedSilenceableFailure LowerToSMEOp::apply(TransformRewriter &rewriter,
                                                TransformResults &results,
                                                TransformState &state) {
  SmallVector<Operation *, 6> resultOps;

  for (Operation *targetOp : state.getPayloadOps(getTarget())) {
    auto conv = dyn_cast<Conv2DNchwFchwOp>(targetOp);
    if (!conv) {
      targetOp->emitError() << "expected `linalg.conv_2d_nchw_fchw` operation";
      return DiagnosedSilenceableFailure::definiteFailure();
    }

    if (!conv->hasAttr("sme.direct"))
      continue;

    auto result = lowerConvToSme(rewriter, *this, conv);
    if (failed(result))
      return DiagnosedSilenceableFailure::definiteFailure();

    resultOps.push_back(*result);
  }

  results.set((*this)->getOpResult(0), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void LowerToSMEOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult LowerToSMEOp::verify() {
  if (this->getName().size() == 0)
    return emitOpError("expected non-empty kernel name");
  return success();
}
