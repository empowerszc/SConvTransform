//===- Parallelize.cpp - Multicore rewrite for SConv tile loops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// @brief Turn the tile loops tagged by the SConv transform into `scf.parallel`
//        and privatize the packing scratch buffers per OpenMP thread.
//
// WHY
//   The window tile loop of the SConv schedule carries all the work of one
//   output tile (input packing + micro-kernel), but it only has 5-20
//   iterations and it writes a scratch buffer that is allocated outside the
//   loop nest.  A naive `omp parallel for` on that loop is therefore both
//   narrow and racy (see docs/OPTIMIZATION_PLAN_2026-11-18.md, section 9).
//
// HOW
//   1. The SConv transform marks the innermost tiling loop -- the window tile
//      of the micro-kernel schedule -- with `sconv.parallel`.  It is the only
//      loop whose iterations write disjoint output tiles and whose body is a
//      whole work item.  The input-channel loop is the *reduction* dimension
//      of the micro-kernel (its iterations accumulate into the same output
//      tile), and the filter tile / window group loops also contain the filter
//      packing that writes a shared scratch buffer.  Earlier prototypes
//      parallelized the first and the third of those loops and produced wrong
//      results; the rule is recorded in section 9.4 of the plan.
//   2. This pass replaces the longest chain of directly nested marked
//      `scf.for` ops with one multi-dimensional `scf.parallel`.  With a single
//      marked loop the chain has length one; the machinery still matters for
//      schedules that expose more than one nestable loop.
//   3. `memref.alloc` ops tagged `sconv.parallel_scratch` are rewritten as an
//      array with one slot per thread; inside the parallel loop a
//      `memref.subview` indexed by `omp_get_thread_num()` replaces the buffer.
//      The slot is stable for the whole parallel region, so the packing /
//      micro-kernel pair within one iteration always sees the same memory.
//
//   Loop-carried values are only supported when they are trivially threaded
//   (the yielded value is the block argument itself), which is what the SConv
//   tiling produces: the loops only thread the output memref through.  Any
//   other shape leaves the loop serial.
//
//===----------------------------------------------------------------------===//

#include "SConvPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr StringLiteral kParallelAttr = "sconv.parallel";
constexpr StringLiteral kScratchAttr = "sconv.parallel_scratch";
constexpr StringLiteral kThreadNumFn = "omp_get_thread_num";

bool isMarkedParallel(Operation *op) { return op->hasAttr(kParallelAttr); }

/// Return the marked `scf.for` that is nested directly below `loop`, i.e. not
/// behind another (unmarked) loop.  Returns a null op when there is none.
scf::ForOp findInnerMarkedLoop(scf::ForOp loop) {
  scf::ForOp found;
  loop.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (auto inner = dyn_cast<scf::ForOp>(op)) {
      if (isMarkedParallel(inner)) {
        found = inner;
        return WalkResult::interrupt();
      }
      // Do not look through loops that are not part of the parallel chain.
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Map every loop result that merely forwards its initial value (possibly
/// through the result of a nested loop) to that initial value.  Computed for
/// all loops of the function, innermost first, so that chains reaching through
/// an unmarked loop (the filter loop) are still recognized.
DenseMap<Value, Value> computeTrivialForwarding(Operation *root) {
  SmallVector<scf::ForOp> loops;
  root->walk<WalkOrder::PreOrder>(
      [&](scf::ForOp loop) { loops.push_back(loop); });

  DenseMap<Value, Value> forwardedTo;
  for (scf::ForOp loopRef : llvm::reverse(loops)) {
    scf::ForOp loop = loopRef;
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield)
      continue;
    Block::BlockArgListType args = loop.getRegionIterArgs();
    OperandRange yielded = yield.getResults();
    if (args.size() != yielded.size())
      continue;

    bool trivial = true;
    for (unsigned i = 0, e = args.size(); i != e && trivial; ++i) {
      Value value = yielded[i];
      if (auto it = forwardedTo.find(value); it != forwardedTo.end())
        value = it->second;
      trivial = value == args[i];
    }
    if (!trivial)
      continue;
    for (auto [result, init] :
         llvm::zip(loop.getResults(), loop.getInitArgs()))
      forwardedTo[result] = init;
  }
  return forwardedTo;
}

/// True when the chain only threads values through itself, given the
/// forwarding map above.
bool hasOnlyTrivialCarriedValues(ArrayRef<scf::ForOp> chain,
                                 const DenseMap<Value, Value> &forwardedTo) {
  for (scf::ForOp loopRef : chain) {
    scf::ForOp loop = loopRef;
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield)
      return false;
    Block::BlockArgListType args = loop.getRegionIterArgs();
    OperandRange yielded = yield.getResults();
    if (args.size() != yielded.size())
      return false;
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      Value value = yielded[i];
      if (auto it = forwardedTo.find(value); it != forwardedTo.end())
        value = it->second;
      if (value != args[i])
        return false;
    }
  }
  return true;
}

/// Three loops are collapsed below, so the trip counts must be static.
std::optional<int64_t> constantTripCount(scf::ForOp loop) {
  std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub < *lb)
    return std::nullopt;
  return llvm::divideCeil(*ub - *lb, *step);
}

/// Replace a validated chain of nested marked loops by one `scf.parallel`.
FailureOr<scf::ParallelOp>
collapseChain(ArrayRef<scf::ForOp> chain,
              const DenseMap<Value, Value> &forwardedTo) {
  assert(!chain.empty() && "expected at least one loop");

  // The outer loops of the chain must contain nothing but the next loop, so
  // that collapsing them cannot drop intermediate work.
  for (unsigned i = 0, e = chain.size() - 1; i != e; ++i) {
    scf::ForOp outer = chain[i];
    scf::ForOp next = chain[i + 1];
    Block *body = outer.getBody();
    if (body->getOperations().size() != 2 ||
        &body->front() != next.getOperation())
      return failure();
  }

  for (scf::ForOp loopRef : chain) {
    scf::ForOp loop = loopRef;
    if (!constantTripCount(loop))
      return failure();
  }
  if (!hasOnlyTrivialCarriedValues(chain, forwardedTo))
    return failure();

  scf::ForOp outermost = chain.front();
  Location loc = outermost.getLoc();
  OpBuilder builder(outermost);

  SmallVector<Value> lowerBounds, upperBounds, steps;
  for (scf::ForOp loop : chain) {
    lowerBounds.push_back(loop.getLowerBound());
    upperBounds.push_back(loop.getUpperBound());
    steps.push_back(loop.getStep());
  }

  // Build the parallel op with an empty body that already has its terminator,
  // then splice the innermost loop body in front of it.  Arguments are named
  // to keep the callback signature explicit.
  scf::ParallelOp parallel = builder.create<scf::ParallelOp>(
      loc, lowerBounds, upperBounds, steps,
      [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange) {
        nestedBuilder.create<scf::ReduceOp>(nestedLoc);
      });

  Block *parallelBody = parallel.getBody();
  auto terminator = cast<scf::ReduceOp>(parallelBody->getTerminator());

  // Move the work of the innermost loop into the parallel body.
  scf::ForOp innermost = chain.back();
  Block *innermostBody = innermost.getBody();
  parallelBody->getOperations().splice(
      Block::iterator(terminator), innermostBody->getOperations(),
      innermostBody->begin(), std::prev(innermostBody->end()));

  // Rewire induction variables, loop-carried values and results, then drop
  // the original loops (innermost first keeps the nesting valid).
  SmallVector<Value> inductionVars = parallel.getInductionVars();
  for (auto [index, loopRef] : llvm::enumerate(chain)) {
    scf::ForOp loop = loopRef;
    loop.getInductionVar().replaceAllUsesWith(inductionVars[index]);
    for (auto [arg, init] :
         llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
      arg.replaceAllUsesWith(init);
    for (auto [result, init] : llvm::zip(loop.getResults(), loop.getInitArgs()))
      result.replaceAllUsesWith(init);
  }
  for (scf::ForOp loopRef : llvm::reverse(chain)) {
    scf::ForOp loop = loopRef;
    loop.erase();
  }

  return parallel;
}

/// Create (or reuse) `func.func private @omp_get_thread_num() -> i32` and emit
/// a call to it, cast to `index`.
Value buildThreadIndex(OpBuilder &builder, Location loc, ModuleOp module) {
  auto i32Type = builder.getI32Type();
  if (!module.lookupSymbol<func::FuncOp>(kThreadNumFn)) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto decl = builder.create<func::FuncOp>(
        loc, kThreadNumFn, builder.getFunctionType({}, {i32Type}));
    decl.setPrivate();
  }
  auto call = builder.create<func::CallOp>(loc, kThreadNumFn,
                                           TypeRange{i32Type}, ValueRange{});
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            call.getResult(0));
}

/// Return the offset that `cast` applies to its base buffer, as one index value.
///
/// A `memref.reinterpret_cast` either lists one "linearized" offset or one
/// offset per dimension; the linearized form is what the SConv lowerings emit,
/// and the full-rank form is folded into a scalar product here.
static FailureOr<Value> getCastOffset(OpBuilder &builder,
                                      memref::ReinterpretCastOp cast) {
  Location loc = cast.getLoc();
  SmallVector<OpFoldResult> offsets = cast.getMixedOffsets();
  if (offsets.size() == 1)
    return getValueOrCreateConstantIndexOp(builder, loc, offsets.front());

  SmallVector<OpFoldResult> strides = cast.getMixedStrides();
  if (offsets.size() != strides.size())
    return failure();
  Value linear = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (auto [offset, stride] : llvm::zip(offsets, strides)) {
    Value offsetValue = getValueOrCreateConstantIndexOp(builder, loc, offset);
    Value strideValue = getValueOrCreateConstantIndexOp(builder, loc, stride);
    Value product = builder.create<arith::MulIOp>(loc, offsetValue, strideValue);
    linear = builder.create<arith::AddIOp>(loc, linear, product);
  }
  return linear;
}

/// Add `delta` elements to the offset of `cast` and return the rewritten op.
///
/// Rationale: the address of a `memref.reinterpret_cast` is relative to the
/// *base buffer* of its source, i.e. the op resets the offset instead of
/// inheriting it.  The per-thread slot of a privatized scratch buffer is
/// carried by a `memref.subview` offset, so every reinterpret_cast that derives
/// from that subview has to add the slot start itself.  Uses that read the
/// metadata directly (`memref.extract_strided_metadata`) inherit the offset and
/// need no adjustment.
static FailureOr<memref::ReinterpretCastOp>
shiftCastOffset(memref::ReinterpretCastOp cast, Value delta) {
  OpBuilder builder(cast);
  FailureOr<Value> offset = getCastOffset(builder, cast);
  if (failed(offset))
    return cast.emitError() << "cannot linearize the offsets of a "
                               "memref.reinterpret_cast";
  Location loc = cast.getLoc();
  Value shifted = builder.create<arith::AddIOp>(loc, *offset, delta);

  // The shifted offset is dynamic, so the result layout has to say so: a fully
  // static layout (offset 0) is treated as the authoritative metadata and makes
  // `memref.extract_strided_metadata` drop the operand.
  MemRefType oldType = cast.getType();
  SmallVector<int64_t> strides(cast.getStaticStrides());
  if (llvm::is_contained(strides, ShapedType::kDynamic))
    return cast.emitError() << "expected a memref.reinterpret_cast with static "
                               "strides";
  auto layout = StridedLayoutAttr::get(cast.getContext(), ShapedType::kDynamic,
                                      strides);
  auto resultType = MemRefType::get(oldType.getShape(),
                                    oldType.getElementType(), layout,
                                    oldType.getMemorySpace());

  auto rewritten = builder.create<memref::ReinterpretCastOp>(
      loc, resultType, cast.getSource(), OpFoldResult(shifted),
      cast.getMixedSizes(), cast.getMixedStrides());
  cast.replaceAllUsesWith(rewritten.getResult());
  cast.erase();
  return rewritten;
}

/// Add the per-thread slot base to every `memref.reinterpret_cast` that derives
/// from the privatized scratch buffer.
static LogicalResult addThreadBaseToCasts(Value scratchView, Value threadBase) {
  SmallVector<Value> worklist{scratchView};
  DenseSet<Value> visited;
  DenseSet<Operation *> patched;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
      Operation *owner = use.getOwner();
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(owner)) {
        if (!patched.insert(owner).second)
          continue;
        FailureOr<memref::ReinterpretCastOp> shifted =
            shiftCastOffset(cast, threadBase);
        if (failed(shifted))
          return failure();
        // The old cast is gone; walk the results of its replacement instead.
        owner = shifted->getOperation();
        patched.insert(owner);
      }
      for (Value result : owner->getResults())
        if (isa<MemRefType>(result.getType()))
          worklist.push_back(result);
    }
  }
  return success();
}

/// Rewrite one tagged scratch allocation into `[maxThreads][...shape...]` and
/// replace its uses inside the parallel region by a thread-indexed subview.
LogicalResult privatizeScratchBuffer(memref::AllocOp alloc,
                                     unsigned maxThreads) {
  MemRefType oldType = alloc.getType();
  if (!oldType.hasStaticShape())
    return alloc.emitError() << "expected a statically shaped scratch buffer";

  // All uses must live in the same parallel loop, otherwise a single slot per
  // thread is not enough (and, if a use sits outside, the subview would not
  // dominate it).
  scf::ParallelOp parallel;
  for (OpOperand &use : alloc->getUses()) {
    auto user = use.getOwner()->getParentOfType<scf::ParallelOp>();
    if (!user)
      return alloc.emitError()
             << "scratch buffer is used outside of an `scf.parallel` loop";
    if (!parallel)
      parallel = user;
    else if (parallel != user)
      return alloc.emitError()
             << "scratch buffer is shared by several `scf.parallel` loops";
  }
  if (!parallel)
    return success();

  MLIRContext *context = alloc.getContext();
  Location loc = alloc.getLoc();
  OpBuilder builder(alloc);

  SmallVector<int64_t> expandedShape{static_cast<int64_t>(maxThreads)};
  llvm::append_range(expandedShape, oldType.getShape());
  auto expandedType =
      MemRefType::get(expandedShape, oldType.getElementType(),
                      oldType.getLayout(), oldType.getMemorySpace());

  builder.setInsertionPoint(alloc);
  auto expanded = builder.create<memref::AllocOp>(loc, expandedType);
  if (IntegerAttr alignment = alloc.getAlignmentAttr())
    expanded.setAlignmentAttr(alignment);

  // Compute the row-major strides of the original (contiguous) tile so that
  // the subview has an explicit, static layout.
  SmallVector<int64_t> strides(oldType.getRank(), 1);
  for (int64_t i = oldType.getRank() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * oldType.getShape()[i + 1];
  auto resultType = MemRefType::get(
      oldType.getShape(), oldType.getElementType(),
      StridedLayoutAttr::get(context, ShapedType::kDynamic, strides),
      oldType.getMemorySpace());

  // Build [tid, 0, .., 0] -> [1, d0, .., dn] view at the start of the region.
  builder.setInsertionPointToStart(parallel.getBody());
  Value threadIndex = buildThreadIndex(builder, loc, alloc->getParentOfType<ModuleOp>());
  builder.setInsertionPointAfter(threadIndex.getDefiningOp());

  SmallVector<OpFoldResult> offsets, sizes, subStrides;
  offsets.push_back(threadIndex);
  sizes.push_back(builder.getIndexAttr(1));
  subStrides.push_back(builder.getIndexAttr(1));
  for (int64_t dim : oldType.getShape()) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(builder.getIndexAttr(dim));
    subStrides.push_back(builder.getIndexAttr(1));
  }

  auto subview = builder.create<memref::SubViewOp>(
      loc, resultType, expanded, offsets, sizes, subStrides);

  // Element count of one slot: the subview's `[tid, 0, .., 0]` offset is
  // `tid * slotElements` deep into the expanded buffer.
  int64_t slotElements = 1;
  for (int64_t dim : oldType.getShape())
    slotElements *= dim;
  Value threadBase = builder.create<arith::MulIOp>(
      loc, threadIndex,
      builder.create<arith::ConstantIndexOp>(loc, slotElements));

  alloc.replaceAllUsesWith(subview.getResult());
  alloc.erase();

  return addThreadBaseToCasts(subview.getResult(), threadBase);
}

/// Collapse the marked tile loops of one function, then privatize scratch.
LogicalResult parallelizeFunction(func::FuncOp function, unsigned maxThreads) {
  SmallVector<scf::ForOp> markedLoops;
  function.walk([&](scf::ForOp loop) { markedLoops.push_back(loop); });
  if (markedLoops.empty())
    return success();

  // A root is a marked loop that has no marked loop above it.  Nested marked
  // loops are reached while walking a chain, not as separate roots.
  SmallVector<scf::ForOp> roots;
  for (scf::ForOp loop : markedLoops) {
    if (!isMarkedParallel(loop))
      continue;
    bool hasMarkedAncestor = false;
    for (Operation *parent = loop->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto parentLoop = dyn_cast<scf::ForOp>(parent);
      if (parentLoop && isMarkedParallel(parentLoop)) {
        hasMarkedAncestor = true;
        break;
      }
    }
    if (!hasMarkedAncestor)
      roots.push_back(loop);
  }

  DenseMap<Value, Value> forwardedTo = computeTrivialForwarding(function);

  for (scf::ForOp root : roots) {
    SmallVector<scf::ForOp> chain{root};
    while (scf::ForOp inner = findInnerMarkedLoop(chain.back()))
      chain.push_back(inner);

    FailureOr<scf::ParallelOp> parallel =
        collapseChain(chain, forwardedTo);
    if (failed(parallel)) {
      for (scf::ForOp loopRef : chain) {
        scf::ForOp loop = loopRef;
        loop->removeAttr(kParallelAttr);
      }
      root.emitRemark()
          << "sconv-parallelize: loop chain is not parallelizable, left serial";
    }
  }

  SmallVector<memref::AllocOp> scratchBuffers;
  function.walk([&](memref::AllocOp alloc) {
    if (alloc->hasAttr(kScratchAttr))
      scratchBuffers.push_back(alloc);
  });
  for (memref::AllocOp alloc : scratchBuffers)
    if (failed(privatizeScratchBuffer(alloc, maxThreads)))
      return failure();

  return success();
}

struct SConvParallelizePass
    : public PassWrapper<SConvParallelizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SConvParallelizePass)

  SConvParallelizePass() = default;
  explicit SConvParallelizePass(unsigned threads) : maxThreads(threads) {}

  /// Number of per-thread scratch slots to allocate.  Kept as a plain member
  /// (rather than a pass `Option`) so that `Pass::clonePass` keeps working
  /// without a hand-written copy constructor.
  unsigned maxThreads = 64;

  StringRef getArgument() const final { return "sconv-parallelize"; }
  StringRef getDescription() const final {
    return "Collapse `sconv.parallel` tile loops into `scf.parallel` and give "
           "tagged scratch buffers one slot per thread";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (failed(parallelizeFunction(function, maxThreads)))
        return signalPassFailure();
  }
};

/// `convert-scf-to-openmp` wraps every parallel body in a `memref.alloca_scope`
/// (so that stack allocations made by the body are released per iteration).
/// That scope is single-block by construction, which the CFG lowering inside
/// the parallel body later violates, so it has to be inlined before
/// `convert-scf-to-cf` runs.  Inlining is safe for the SConv loops because the
/// bodies only use heap allocations; a scope that actually contains a
/// `memref.alloca` is left untouched (its lifetime would change).
struct SConvEraseAllocaScopePass
    : public PassWrapper<SConvEraseAllocaScopePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SConvEraseAllocaScopePass)

  StringRef getArgument() const final { return "sconv-erase-alloca-scope"; }
  StringRef getDescription() const final {
    return "Inline `memref.alloca_scope` regions that contain no stack "
           "allocations, so that CFG lowering can run inside parallel regions";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<memref::AllocaScopeOp> scopes;
    module.walk(
        [&](memref::AllocaScopeOp scope) { scopes.push_back(scope); });

    IRRewriter rewriter(module.getContext());
    for (memref::AllocaScopeOp scope : scopes) {
      Block *body = scope.getBody();
      if (!body)
        continue;
      bool hasStackAlloca = false;
      scope.getBodyRegion().walk(
          [&](memref::AllocaOp) { hasStackAlloca = true; });
      if (hasStackAlloca)
        continue;

      auto terminator =
          dyn_cast<memref::AllocaScopeReturnOp>(body->getTerminator());
      if (!terminator || terminator.getNumOperands() != scope.getNumResults()) {
        scope.emitRemark()
            << "sconv-erase-alloca-scope: unexpected terminator, scope kept";
        continue;
      }
      for (auto [result, value] :
           llvm::zip(scope.getResults(), terminator.getOperands()))
        result.replaceAllUsesWith(value);

      Block *parent = scope->getBlock();
      parent->getOperations().splice(Block::iterator(scope),
                                     body->getOperations(), body->begin(),
                                     std::prev(body->end()));
      rewriter.eraseOp(scope);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createSConvParallelizePass() {
  return std::make_unique<SConvParallelizePass>();
}

std::unique_ptr<Pass> mlir::createSConvParallelizePass(unsigned maxThreads) {
  return std::make_unique<SConvParallelizePass>(maxThreads);
}

std::unique_ptr<Pass> mlir::createSConvEraseAllocaScopePass() {
  return std::make_unique<SConvEraseAllocaScopePass>();
}

void mlir::registerSConvPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<Pass> { return createSConvParallelizePass(); });
  mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return createSConvEraseAllocaScopePass();
  });
}
