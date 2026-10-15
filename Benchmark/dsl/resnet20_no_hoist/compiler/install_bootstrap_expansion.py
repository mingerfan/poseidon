"""Install the opt-in real Hecate pass without replacing existing scale patches.

The pass needs a separately verified native primitive template. Installing it
alone does not produce or validate that template, or change legacy pipelines.
"""
from pathlib import Path
import argparse
import shutil

PASS = '''
def ExpandNativeBootstrap : Pass<"expand-native-bootstrap", "::mlir::func::FuncOp"> {
  let summary = "Expand fixed bootstrap boundaries into explicit native CKKS primitives";
  let options = [
    Option<"templateFile", "template-file", "std::string", [{""}], "Verified native bootstrap template MLIR">,
    Option<"expectedCount", "expected-count", "int64_t", [{18}], "Required original bootstrap boundary count">
  ];
}
'''

def install(root):
    folder = root / 'lib/Dialect/CKKS/Transforms'
    source = Path(__file__).with_name('ExpandNativeBootstrap.cpp')
    changes = {
        root / 'include/hecate/Dialect/CKKS/Transforms/Passes.td':
            ('#endif', PASS + '\n#endif', 'def ExpandNativeBootstrap'),
        folder / 'CMakeLists.txt':
            ('  EmitHEVM.cpp', '  EmitHEVM.cpp\n  ExpandNativeBootstrap.cpp', '  ExpandNativeBootstrap.cpp'),
        root / 'include/hecate/Dialect/CKKS/IR/CKKSOps.td':
            ('#endif', 'include "hecate/Dialect/CKKS/IR/NativeBootstrapOps.td"\n\n#endif', 'include "hecate/Dialect/CKKS/IR/NativeBootstrapOps.td"'),
    }
    for file, (old, new, marker) in changes.items():
        text = file.read_text()
        if marker in text:
            continue
        if text.count(old) != 1:
            raise ValueError(f'Unexpected compiler source layout: {file}')
        backup = file.with_name(file.name + '.before-bootstrap-expansion')
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(file, backup)
        file.write_text(text.replace(old, new))
    shutil.copy2(source, folder / source.name)
    shutil.copy2(source.with_name('NativeBootstrapOps.td'),
                 root / 'include/hecate/Dialect/CKKS/IR/NativeBootstrapOps.td')
    emitter = folder / 'EmitHEVM.cpp'
    text = emitter.read_text()
    if 'POSEIDON_HEVM_NATIVE_PRIMITIVE_V1' not in text:
        backup = emitter.with_name(emitter.name + '.before-bootstrap-expansion')
        if backup.exists():
            raise FileExistsError(backup)
        changes = [
            ('    bool invalid_native_scale = false;', '''    bool invalid_native_scale = false;
    std::vector<std::array<uint64_t, 3>> primitive_params, bootstrap_groups;
    const bool expanded = func->hasAttr("poseidon.expanded_bootstraps");'''),
            ('        if (auto scale = op->getAttrOfType<FloatAttr>("poseidon_native_scale")) {', '''        if (auto scaleOp = dyn_cast<hecate::ckks::ScaleSetCOp>(op)) {
          double value = scaleOp.getScaleAttr().getValueAsDouble();
          uint64_t bits; std::memcpy(&bits, &value, sizeof(bits));
          primitive_params.push_back({insts.size(), 13, bits});
        }
        if (auto group = op->getAttrOfType<IntegerAttr>("poseidon.bootstrap_group")) {
          auto local = op->getAttrOfType<IntegerAttr>("poseidon.bootstrap_local_index");
          if (!local || group.getInt() < 0 || local.getInt() < 0) {
            op->emitError("invalid expanded bootstrap provenance");
            invalid_native_scale = true;
          } else bootstrap_groups.push_back({insts.size(), uint64_t(group.getInt()), uint64_t(local.getInt())});
        }
        if (auto scale = op->getAttrOfType<FloatAttr>("poseidon_native_scale")) {'''),
            ('    });\n\n    if (invalid_native_scale)', '''      else if (expanded && !isa<func::FuncOp, func::ReturnOp>(op)) {
        op->emitError("expanded bootstrap program contains unsupported operation; refusing to drop it");
        invalid_native_scale = true;
      }
    });

    if (invalid_native_scale)'''),
            ('  void getDependentDialects(DialectRegistry &registry) const override {', '''  void writeExpandedMetadata(const std::string &path,
      llvm::ArrayRef<HEVMOperation> insts,
      const std::vector<std::array<uint64_t, 3>> &rows,
      const char *suffix, const char *magic) {
    if (rows.empty()) return;
    uint64_t hash=14695981039346656037ULL;
    std::ifstream file(path, std::ios::binary); char byte;
    while(file.get(byte)){hash^=static_cast<unsigned char>(byte);hash*=1099511628211ULL;}
    std::ofstream out(path+suffix);
    out << magic << ' ' << insts.size() << ' ' << std::hex << hash << std::dec << '\\n';
    for(const auto &row:rows)
      out << row[0] << ' ' << row[1] << ' ' << std::hex << row[2] << std::dec << '\\n';
    if(!file.eof() || !out){getOperation().emitError("cannot bind expanded bootstrap metadata");signalPassFailure();}
  }
  void getDependentDialects(DialectRegistry &registry) const override {'''),
        ]
        for old, new in changes:
            if text.count(old) != 1:
                raise ValueError(f'Unexpected EmitHEVM layout: {old[:70]}')
            text = text.replace(old, new)
        # Place metadata writes after both HEVM and the optional scale sidecar.
        old = '\n  }\n\n  void writeExpandedMetadata'
        new = '''
    writeExpandedMetadata(std::string(printpath), insts, primitive_params, ".native_primitives", "POSEIDON_HEVM_NATIVE_PRIMITIVE_V1");
    writeExpandedMetadata(std::string(printpath), insts, bootstrap_groups, ".bootstrap_groups", "POSEIDON_HEVM_BOOTSTRAP_GROUPS_V1");
  }

  void writeExpandedMetadata'''
        if text.count(old) != 1:
            raise ValueError('Unexpected emitter metadata write position')
        text = text.replace(old, new)
        shutil.copy2(emitter, backup)
        emitter.write_text(text)
    print('Installed opt-in expand-native-bootstrap pass; native template still required')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hecate_root', type=Path)
    install(parser.parse_args().hecate_root)
