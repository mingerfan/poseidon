// Expand already-placed bootstrap boundaries before scheduling/placement.
// This pass does not choose bootstrap locations or approximate a native circuit.
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

namespace hecate::ckks {
#define GEN_PASS_DEF_EXPANDNATIVEBOOTSTRAP
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
}
using namespace mlir;

namespace {
struct ExpandNativeBootstrapPass
    : hecate::ckks::impl::ExpandNativeBootstrapBase<ExpandNativeBootstrapPass> {
  ExpandNativeBootstrapPass() = default;
  explicit ExpandNativeBootstrapPass(
      const hecate::ckks::ExpandNativeBootstrapOptions &options) {
    templateFile = options.templateFile;
    expectedCount = options.expectedCount;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::ckks::CKKSDialect, func::FuncDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() override {
    auto function = getOperation();
    SmallVector<hecate::ckks::BootstrapCOp> boundaries;
    function.walk([&](hecate::ckks::BootstrapCOp op) { boundaries.push_back(op); });
    auto reject = [&](const llvm::Twine &message) {
      function.emitError(message);
      signalPassFailure();
    };
    if (expectedCount <= 0 || int64_t(boundaries.size()) != expectedCount) {
      reject("bootstrap boundary count differs from expected-count"); return;
    }
    auto file = llvm::MemoryBuffer::getFile(templateFile);
    if (!file) { reject("cannot read explicit bootstrap template"); return; }
    auto module = parseSourceString<ModuleOp>((*file)->getBuffer(), &getContext());
    if (!module || failed(verify(*module))) {
      reject("invalid explicit bootstrap template module"); return;
    }
    auto body = module->lookupSymbol<func::FuncOp>("native_bootstrap_template");
    if (!body || body.isExternal() || !body.getBody().hasOneBlock() ||
        body.getNumArguments() != 1 || body.getNumResults() != 1) {
      reject("template requires one single-block function, one input and one result");
      return;
    }
    auto terminator = dyn_cast<func::ReturnOp>(body.getBody().front().getTerminator());
    if (!terminator || terminator.getNumOperands() != 1) {
      reject("template requires one returned ciphertext"); return;
    }
    int64_t primitiveCount = 0;
    for (Operation &op : body.getBody().front().without_terminator()) {
      if (op.getNumRegions() || isa<hecate::ckks::BootstrapCOp>(op) ||
          (!isa<tensor::EmptyOp>(op) &&
           (!isa<hecate::ckks::HEVMOpInterface>(op) ||
            op.getName().getDialectNamespace() != "ckks"))) {
        reject("template contains an opaque, nested, or unsupported operation");
        return;
      }
      primitiveCount += !isa<tensor::EmptyOp>(op);
    }
    if (!primitiveCount || terminator.getOperand(0) == body.getArgument(0)) {
      reject("identity/empty template cannot replace a bootstrap"); return;
    }
    // Validate every boundary before changing the original graph.
    for (auto op : boundaries) {
      if (op.getSrc().getType() != body.getArgument(0).getType() ||
          op.getResult().getType() != body.getResultTypes()[0]) {
        reject("bootstrap/template input or result type mismatch"); return;
      }
    }
    const auto hash = llvm::SHA256::hash(llvm::arrayRefFromStringRef((*file)->getBuffer()));
    const auto hashText = llvm::toHex(hash, true);
    OpBuilder builder(&getContext());
    int64_t group = 0;
    for (auto boundary : boundaries) {
      builder.setInsertionPoint(boundary);
      IRMapping mapping;
      mapping.map(body.getArgument(0), boundary.getSrc());
      int64_t index = 0;
      for (Operation &op : body.getBody().front().without_terminator()) {
        auto *clone = builder.clone(op, mapping);
        clone->setAttr("poseidon.bootstrap_group", builder.getI64IntegerAttr(group));
        clone->setAttr("poseidon.bootstrap_local_index", builder.getI64IntegerAttr(index++));
        clone->setAttr("poseidon.bootstrap_template_sha256", builder.getStringAttr(hashText));
      }
      boundary.getResult().replaceAllUsesWith(mapping.lookup(terminator.getOperand(0)));
      boundary.erase();
      ++group;
    }
    function->setAttr("poseidon.expanded_bootstraps", builder.getI64IntegerAttr(group));
    function->setAttr("poseidon.bootstrap_primitives_per_group",
                      builder.getI64IntegerAttr(primitiveCount));
    function->setAttr("poseidon.bootstrap_template_sha256", builder.getStringAttr(hashText));
  }
};
} // namespace
