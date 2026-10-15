"""Add opt-in 32-bit HEVM serialization without changing legacy runtime ABI."""
import argparse
import difflib
from pathlib import Path


def apply(root, backup):
    paths = ['include/hecate/Support/HEVMHeader.h',
             'include/hecate/Dialect/CKKS/IR/PolyTypeInterface.td',
             'include/hecate/Dialect/CKKS/IR/CKKSOps.td',
             'include/hecate/Dialect/CKKS/IR/NativeBootstrapOps.td',
             'lib/Dialect/CKKS/Transforms/EmitHEVM.cpp']
    backup.mkdir(parents=True, exist_ok=False)
    patches = []
    for name in paths:
        path = root / name
        old = path.read_text()
        if name.endswith('HEVMHeader.h'):
            new = old.replace('struct HEVMOperation {', '''// Compiler-side instruction; retain full signed attributes until checked output.
struct HEVMInstruction {
  int64_t opcode = 0, dst = 0, lhs = 0, rhs = 0;
};
// Opt-in file magic 0x48455632; header/config unchanged, instructions 16 bytes.
struct HEVMWideOperation {
  uint32_t opcode, dst, lhs, rhs;
};
struct HEVMOperation {''')
        else:
            # Change the returned type, not the getHEVMOperation method name.
            new = old.replace('"HEVMOperation"', '"HEVMInstruction"')
            new = new.replace('HEVMOperation ', 'HEVMInstruction ')
            new = new.replace('<HEVMOperation', '<HEVMInstruction')
            new = new.replace('getHEVMInstruction', 'getHEVMOperation')
            if name.endswith('NativeBootstrapOps.td'):
                new = new.replace('op.getMultiplier()<=65535', 'op.getMultiplier()<=4294967295LL')
            if name.endswith('EmitHEVM.cpp'):
                new = new.replace('const bool expanded =', 'const bool wide = func->hasAttr("poseidon.hevm_wide");\n    const bool expanded =')
                new = new.replace('else if (expanded &&', 'else if ((expanded || wide) &&')
                needle = '    std::ofstream of(printpath, std::ios::binary);'
                assert needle in new
                new = new.replace(needle, '''    // Validate before opening any artifact: never silently truncate indices.
    const int64_t maxField = wide ? 4294967295LL : 65535;
    for (const auto &inst : insts) {
      if (inst.opcode == -1) continue;
      const int64_t minRhs = inst.opcode == 1 ? (wide ? -2147483648LL : -32768) : 0;
      if (inst.opcode < 0 || inst.opcode > maxField || inst.dst < 0 ||
          inst.dst > maxField || inst.lhs < 0 || inst.lhs > maxField ||
          inst.rhs < minRhs || inst.rhs > maxField) {
        func.emitError("HEVM field overflow; use poseidon.hevm_wide for 32-bit instructions");
        signalPassFailure(); return;
      }
    }
    header.magic_number = wide ? 0x48455632 : 0x4845564D;
''' + needle)
                needle = '    of.write((char *)insts.data(), insts.size() * sizeof(HEVMOperation));'
                assert needle in new
                new = new.replace(needle, '''    for (const auto &inst : insts) {
      if (wide) {
        HEVMWideOperation packed{uint32_t(inst.opcode), uint32_t(inst.dst),
                                 uint32_t(inst.lhs), uint32_t(inst.rhs)};
        of.write(reinterpret_cast<const char *>(&packed), sizeof(packed));
      } else {
        HEVMOperation packed{uint16_t(inst.opcode), uint16_t(inst.dst),
                             uint16_t(inst.lhs), uint16_t(inst.rhs)};
        of.write(reinterpret_cast<const char *>(&packed), sizeof(packed));
      }
    }''')
        assert new != old, name
        dest = backup / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(old)
        path.write_text(new)
        patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                       fromfile='a/' + name, tofile='b/' + name))
    (backup / 'wide-hevm.patch').write_text(''.join(patches))
    print('Wide HEVM compiler source prepared; requires real build and regression tests')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=Path)
    parser.add_argument('backup', type=Path)
    args = parser.parse_args()
    apply(args.compiler, args.backup)
