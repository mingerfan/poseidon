// Reattach the compiler's saved cut/target schedule to the original constants.
// Only this ResNet experiment's cut==target schedules are accepted.
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <stdexcept>

using namespace mlir;
namespace earth = hecate::earth;
int main(int argc, char **argv) {
  if (argc != 5) return 2; // original bytecode, config, schedule attribute, output
  earth::EarthDialect::setCKKSParameters(argv[2]);
  MLIRContext context;
  context.getOrLoadDialect<earth::EarthDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module) return 1;
  auto function = *module->getOps<func::FuncOp>().begin();
  if (std::next(module->getOps<func::FuncOp>().begin()) != module->getOps<func::FuncOp>().end()) return 1;
  // Same dead-code removal as BypassDetection before assigning target IDs.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(function.front())))
    if (isOpTriviallyDead(&op)) op.erase();
  std::ifstream input(argv[3]);
  std::string text((std::istreambuf_iterator<char>(input)), {});
  auto schedule = dyn_cast_or_null<ArrayAttr>(parseAttribute(text, &context));
  if (!schedule) return 1;
  function->setAttr("btp_schedule", schedule);
  SmallVector<Value> values{Value{}};
  for (auto argument : function.getArguments()) values.push_back(argument);
  for (auto &op : function.front()) for (auto result : op.getResults()) values.push_back(result);
  OpBuilder builder(&context);
  int64_t previous = 0;
  for (auto item : schedule) {
    auto entry = cast<DenseI64ArrayAttr>(item).asArrayRef();
    if (entry.size() != 2 || entry[0] != entry[1] || entry[0] <= previous ||
        entry[0] >= static_cast<int64_t>(values.size())) return 1;
    previous = entry[0];
    auto source = values[entry[0]];
    auto boundary = source.getDefiningOp();
    if (!boundary || !earth::getScaleType(source).isCipher()) return 1;
    auto type = earth::getScaleType(source);
    builder.setInsertionPointAfter(boundary);
    auto boot = builder.create<earth::BootstrapOp>(source.getLoc(), source,
                                                   type.getLevel(), type.getScale());
    boot->setAttr("planned_scale", builder.getI64IntegerAttr(45));
    boot->setAttr("boot_target", builder.getI64IntegerAttr(entry[1]));
    boot->setAttr("opid", builder.getI64IntegerAttr(entry[0]));
    source.replaceUsesWithIf(boot.getResult(), [&](OpOperand &use) {
      return use.getOwner() != boot.getOperation() && boundary->isBeforeInBlock(use.getOwner());
    });
  }
  if (failed(verify(*module))) return 1;
  std::error_code error;
  llvm::raw_fd_ostream out(argv[4], error, llvm::sys::fs::OF_None);
  if (error || failed(writeBytecodeToFile(*module, out))) return 1;
  llvm::outs() << "Reattached " << schedule.size() << " compiler-selected Boots\n";
}
