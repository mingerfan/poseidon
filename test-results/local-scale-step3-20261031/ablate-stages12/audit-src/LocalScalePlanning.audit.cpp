#include "hecate/Dialect/Earth/Transforms/LocalScalePlanning.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>

using namespace mlir;
using namespace hecate;
using namespace hecate::earth;

namespace {
constexpr StringLiteral editableAttr = "earth.local_scale_editable";
constexpr StringLiteral flexAttr = "earth.local_scale_flex";
constexpr StringLiteral forecastAttr = "earth.local_scale_forecast";
constexpr StringLiteral managedAttr = "earth.local_scale_managed";

LocalScalePlanning policy(double waterline) {
  return {EarthDialect::nativeScaleChain,
          EarthDialect::nativeLevelAnchor,
          waterline,
          bootstrapQBudget(),
          static_cast<unsigned>(EarthDialect::nativeBootInputLevel),
          EarthDialect::nativeBootInputScale};
}

bool editable(Operation *op) {
  return !op->hasAttr("sm_fixed") &&
         (op->hasAttr(editableAttr) || op->hasAttr(managedAttr));
}

void cacheFlex(Value value, LocalScaleValue state, OpBuilder &builder) {
  if (auto *op = value.getDefiningOp())
    op->setAttr(flexAttr, builder.getDenseF64ArrayAttr(
                              {state.minShift, state.maxShift,
                               static_cast<double>(state.edits)}));
}

LocalScaleKind kindOf(Operation *op) {
  if (auto mul = dyn_cast<MulOp>(op))
    return mul.isSingle() ? LocalScaleKind::MulCP : LocalScaleKind::MulCC;
  if (auto add = dyn_cast<AddOp>(op))
    return add.isSingle() ? LocalScaleKind::AddCP : LocalScaleKind::AddCC;
  return LocalScaleKind::Unary;
}

// Retune only the bounded private tree described by the same flex state used
// in simulation. Constants are cloned; shared ciphertext producers are
// barriers.
LogicalResult retune(Value value, double delta, double waterline,
                     OpBuilder &builder, unsigned &edits) {
  if (delta == 0)
    return success();
  auto *op = value.getDefiningOp();
  if (!op || !editable(op) || ++edits > LocalScalePlanning::maxEdits)
    return failure();
  auto infer = [&]() -> LogicalResult {
    op->removeAttr("earth.actual_scale");
    op->removeAttr(flexAttr);
    SmallVector<Type> types;
    auto inference = cast<InferTypeOpInterface>(op);
    if (failed(inference.inferReturnTypes(
            op->getContext(), op->getLoc(), op->getOperands(),
            op->getAttrDictionary(), op->getPropertiesStorage(),
            op->getRegions(), types)))
      return failure();
    op->getResult(0).setType(types.front());
    actualScale(value);
    return success();
  };
  if (auto mul = dyn_cast<MulOp>(op)) {
    if (mul.isSingle()) {
      unsigned index = getScaleType(mul.getLhs()).isCipher() ? 1 : 0;
      auto constant = mul->getOperand(index).getDefiningOp<ConstantOp>();
      if (!constant)
        return failure();
      double scale = actualScale(constant.getResult()) + delta;
      if (scale < waterline)
        return failure();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(mul);
      auto copy = cast<ConstantOp>(builder.clone(*constant));
      copy->getResult(0).setType(
          getScaleType(copy.getResult()).switchScale(std::ceil(scale)));
      copy->setAttr("earth.encoding_scale", builder.getF64FloatAttr(scale));
      mul->setOperand(index, copy.getResult());
    } else {
      auto a = localScaleValue(mul.getLhs(), waterline);
      auto b = localScaleValue(mul.getRhs(), waterline);
      unsigned side =
          a.flexible() && (!b.flexible() || a.edits <= b.edits) ? 0 : 1;
      if (failed(
              retune(mul->getOperand(side), delta, waterline, builder, edits)))
        return failure();
    }
  } else if (auto add = dyn_cast<AddOp>(op)) {
    for (Value operand : add->getOperands())
      if (getScaleType(operand).isCipher() &&
          failed(retune(operand, delta, waterline, builder, edits)))
        return failure();
    if (add.isSingle()) {
      unsigned plain = getScaleType(add.getLhs()).isCipher() ? 1 : 0;
      auto constant = add->getOperand(plain).getDefiningOp<ConstantOp>();
      if (!constant)
        return failure();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(add);
      auto copy = cast<ConstantOp>(builder.clone(*constant));
      copy->getResult(0).setType(
          getScaleType(add->getOperand(1 - plain)).toPlain());
      add->setOperand(plain, copy.getResult());
    }
  } else if (auto up = dyn_cast<UpscaleOp>(op)) {
    auto current = up->getAttrOfType<FloatAttr>("earth.scale_delta");
    double scale =
        (current ? current.getValueAsDouble() : up.getUpFactor()) + delta;
    if (scale < auditUnitFloor(waterline))
      return failure();
    up.setUpFactor(std::ceil(scale));
    up->setAttr("earth.scale_delta", builder.getF64FloatAttr(scale));
  } else if (isa<RescaleOp, ModswitchOp, RotateOp, NegateOp>(op)) {
    if (failed(retune(op->getOperand(0), delta, waterline, builder, edits)))
      return failure();
  } else
    return failure();
  return infer();
}
} // namespace

void hecate::earth::prepareLocalScalePlanning(func::FuncOp func) {
  if (!usesLocalScalePlanning() || func->hasAttr("earth.local_scale_hints"))
    return;
  OpBuilder builder(func);
  func->setAttr("earth.local_scale_hints", builder.getUnitAttr());
  func->setAttr("earth.local_scale_policy", builder.getUnitAttr());
  // Only 15 possible three-step forecasts are interned, rather than storing
  // a separate list of future instruction IDs on every node.
  for (auto &op : llvm::reverse(func.front())) {
    if (op.getNumResults() != 1 || isa<ConstantOp>(op))
      continue;
    Value value = op.getResult(0);
    if (value.hasOneUse() && !op.hasAttr("sm_fixed"))
      op.setAttr(editableAttr, builder.getUnitAttr());
    std::array<int64_t, 3> best{-1, -1, -1};
    auto score = [](const auto &a) {
      return std::make_pair(
          std::count(a.begin(), a.end(), int64_t{1}),
          std::count_if(a.begin(), a.end(), [](int64_t x) { return x >= 0; }));
    };
    for (Operation *user : value.getUsers()) {
      std::array<int64_t, 3> next{-1, -1, -1};
      if (auto forecast = user->getAttrOfType<DenseI64ArrayAttr>(forecastAttr))
        std::copy(forecast.asArrayRef().begin(), forecast.asArrayRef().end(),
                  next.begin());
      if (auto mul = dyn_cast<MulOp>(user)) {
        if (!mul.isSingle() && mul.getLhs() != mul.getRhs())
          continue;
        next = {mul.isSingle() ? 0 : 1, next[0], next[1]};
      } else if (!isa<AddOp, RotateOp, NegateOp>(user))
        continue;
      // Crossing an Add estimates its scale pressure; legality always uses
      // the actual two operand states when that Add is reached.
      if (score(next) > score(best))
        best = next;
    }
    op.setAttr(forecastAttr, builder.getDenseI64ArrayAttr(best));
  }
}

LocalScaleValue hecate::earth::localScaleValue(Value value, double waterline) {
  auto type = getScaleType(value);
  LocalScaleValue state{static_cast<int64_t>(type.getLevel()),
                        actualScale(value),
                        static_cast<int64_t>(type.getScale())};
  auto *op = value.getDefiningOp();
  if (!op || !editable(op))
    return state;
  if (auto cached = op->getAttrOfType<DenseF64ArrayAttr>(flexAttr)) {
    auto data = cached.asArrayRef();
    state.minShift = data[0];
    state.maxShift = data[1];
    state.edits = data[2];
    return state;
  }
  if (auto mul = dyn_cast<MulOp>(op); mul && mul.isSingle()) {
    Value plain =
        getScaleType(mul.getLhs()).isCipher() ? mul.getRhs() : mul.getLhs();
    if (!plain.getDefiningOp<ConstantOp>())
      return state;
    state.minShift = waterline - actualScale(plain);
    state.maxShift = EarthDialect::nativeScaleChain.capacity(
                         EarthDialect::nativeLevelAnchor - state.level) -
                     state.scale - 1;
    state.edits = 1;
  } else if (isa<RescaleOp, ModswitchOp>(op)) {
    auto source = localScaleValue(op->getOperand(0), waterline);
    if (source.flexible() && source.edits + 1 <= LocalScalePlanning::maxEdits) {
      state.minShift = source.minShift;
      state.maxShift = std::min(
          source.maxShift, EarthDialect::nativeScaleChain.capacity(
                               EarthDialect::nativeLevelAnchor - state.level) -
                               state.scale - 1);
      state.edits = source.edits + 1;
    }
  } else if (auto up = dyn_cast<UpscaleOp>(op)) {
    auto delta = up->getAttrOfType<FloatAttr>("earth.scale_delta");
    double plain = delta ? delta.getValueAsDouble() : up.getUpFactor();
    state.minShift = auditUnitFloor(waterline) - plain;
    state.maxShift = EarthDialect::nativeScaleChain.capacity(
                         EarthDialect::nativeLevelAnchor - state.level) -
                     state.scale - 1;
    state.edits = 1;
  } else if (isa<MulOp, AddOp, RotateOp, NegateOp>(op)) {
    SmallVector<LocalScaleValue> operands;
    for (Value input : op->getOperands())
      if (getScaleType(input).isCipher())
        operands.push_back(localScaleValue(input, waterline));
    if (operands.size() == 2 && isa<AddOp>(op) && operands[0].flexible() &&
        operands[1].flexible()) {
      state.minShift = std::max(operands[0].minShift, operands[1].minShift);
      state.maxShift = std::min(operands[0].maxShift, operands[1].maxShift);
      state.edits = operands[0].edits + operands[1].edits + 1;
    } else if (!operands.empty()) {
      unsigned side = 0;
      if (operands.size() == 2 && isa<MulOp>(op))
        side =
            operands[0].flexible() && (!operands[1].flexible() ||
                                       operands[0].edits <= operands[1].edits)
                ? 0
                : 1;
      if (operands[side].flexible() &&
          (operands.size() == 1 || isa<MulOp>(op))) {
        state.minShift = operands[side].minShift;
        state.maxShift = operands[side].maxShift;
        state.edits = operands[side].edits + 1;
      }
    }
    state.maxShift = std::min(
        state.maxShift, EarthDialect::nativeScaleChain.capacity(
                            EarthDialect::nativeLevelAnchor - state.level) -
                            state.scale - 1 - isa<AddOp>(op));
    if (state.edits > LocalScalePlanning::maxEdits)
      state.edits = 0;
  }
  OpBuilder builder(op);
  cacheFlex(value, state, builder);
  return state;
}

std::optional<LocalScalePlan>
hecate::earth::planLocalScale(Operation *op, ArrayRef<LocalScaleValue> operands,
                              double waterline) {
  std::array<int64_t, 3> forecast{-1, -1, -1};
  if (auto data = op->getAttrOfType<DenseI64ArrayAttr>(forecastAttr))
    std::copy(data.asArrayRef().begin(), data.asArrayRef().end(),
              forecast.begin());
  if (operands.empty())
    return std::nullopt;
  return policy(waterline).choose(
      kindOf(op), operands[0],
      operands.size() == 2 ? operands[1] : LocalScaleValue{}, editable(op),
      !op->hasAttr("sm_fixed"), forecast);
}

FailureOr<Value> hecate::earth::applyLocalScale(ForwardMgmtInterface forward,
                                                int64_t waterline,
                                                bool replaceUses) {
  Operation *op = forward.getOperation();
  OpBuilder builder(op);
  SmallVector<LocalScaleValue> operands;
  SmallVector<unsigned> cipherIndices;
  for (auto [index, value] : llvm::enumerate(op->getOperands())) {
    if (getScaleType(value).isCipher()) {
      operands.push_back(localScaleValue(value, waterline));
      cipherIndices.push_back(index);
    } else if (auto constant = value.getDefiningOp<ConstantOp>()) {
      if (!value.hasOneUse())
        op->setOperand(index, builder.clone(*constant)->getResult(0));
    }
  }
  auto plan = planLocalScale(op, operands, waterline);
  if (!plan)
    return failure();
  auto kind = kindOf(op);
  if (kind == LocalScaleKind::MulCP) {
    unsigned plain = 1 - cipherIndices.front();
    auto constant = op->getOperand(plain).getDefiningOp<ConstantOp>();
    if (!constant)
      return op->emitError(
                 "local scale planning requires locally encoded coefficients"),
             failure();
    constant->getResult(0).setType(
        getScaleType(op->getOperand(cipherIndices.front()))
            .toPlain()
            .switchScale(std::ceil(plan->encodingScale)));
    constant->setAttr("earth.encoding_scale",
                      builder.getF64FloatAttr(plan->encodingScale));
  }
  for (auto [i, index] : llvm::enumerate(cipherIndices)) {
    Value value = op->getOperand(index);
    const auto &action = plan->inputs[i];
    if (action.retune && action.shift != 0) {
      unsigned edits = 0;
      if (failed(retune(value, action.shift, waterline, builder, edits)))
        return op->emitError(
                   "local coefficient retuning disagrees with its state plan"),
               failure();
    }
    if (action.modswitch) {
      value =
          builder.create<ModswitchOp>(op->getLoc(), value, action.modswitch);
      value.getDefiningOp()->setAttr(managedAttr, builder.getUnitAttr());
    }
    if (action.unit) {
      auto up = builder.create<UpscaleOp>(
          op->getLoc(), value, static_cast<int64_t>(std::ceil(action.shift)));
      up->setAttr("earth.scale_delta", builder.getF64FloatAttr(action.shift));
      up->setAttr(managedAttr, builder.getUnitAttr());
      value = up;
    }
    for (unsigned r = 0; r < action.rescales; ++r) {
      value = builder.create<RescaleOp>(op->getLoc(), value);
      value.getDefiningOp()->setAttr(managedAttr, builder.getUnitAttr());
    }
    op->setOperand(index, value);
  }
  if (kind == LocalScaleKind::AddCP) {
    unsigned plain = 1 - cipherIndices.front();
    op->getOperand(plain).setType(
        getScaleType(op->getOperand(cipherIndices.front())).toPlain());
  }
  op->removeAttr("earth.actual_scale");
  op->removeAttr(flexAttr);
  SmallVector<Type> types;
  auto inference = cast<InferTypeOpInterface>(op);
  if (failed(inference.inferReturnTypes(
          op->getContext(), op->getLoc(), op->getOperands(),
          op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(),
          types)))
    return failure();
  op->getResult(0).setType(types.front());
  builder.setInsertionPointAfter(op);
  Value value = op->getResult(0);
  IRRewriter rewriter(builder);
  for (unsigned r = 0; r < plan->resultRescales; ++r) {
    auto rs = builder.create<RescaleOp>(op->getLoc(), value);
    rs->setAttr(managedAttr, builder.getUnitAttr());
    if (replaceUses)
      rewriter.replaceAllUsesExcept(value, rs, rs);
    value = rs;
  }
  cacheFlex(value, plan->result, builder);
  actualScale(value);
  return value;
}

void hecate::earth::clearLocalScaleAttributes(func::FuncOp func) {
  func.walk([](Operation *op) {
    op->removeAttr(editableAttr);
    op->removeAttr(flexAttr);
    op->removeAttr(forecastAttr);
    op->removeAttr(managedAttr);
  });
  func->removeAttr("earth.local_scale_hints");
}
