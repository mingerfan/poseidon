
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <filesystem>
#include <fstream>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>

#include "hecate/Support/HEVMHeader.h"

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_EMITHEVM
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct EmitHEVMPass : public hecate::ckks::impl::EmitHEVMBase<EmitHEVMPass> {
  EmitHEVMPass() {}
  EmitHEVMPass(hecate::ckks::EmitHEVMOptions ops) { this->prefix = ops.prefix; }

  void runOnOperation() override {
    markAllAnalysesPreserved();
    auto &&func = getOperation();
    HEVMHeader header;
    ConfigBody config_body;
    header.hevm_header_size = sizeof(HEVMHeader);
    header.config_header.arg_length = func.getNumArguments();
    header.config_header.res_length = func.getNumResults();

    config_body.config_body_length = sizeof(ConfigBody);
    SmallVector<uint64_t, 4> config_body_ints;
    SmallVector<HEVMInstruction, 4> insts;
    // Only explicitly annotated physical schedules get a scale sidecar.
    std::vector<std::array<uint64_t, 4>> native_scales;
    bool invalid_native_scale = false;
    std::vector<std::array<uint64_t, 3>> primitive_params, bootstrap_groups;
    const bool wide = func->hasAttr("poseidon.hevm_wide");
    const bool expanded = func->hasAttr("poseidon.expanded_bootstraps");

    int64_t cipher_registers = 0;
    int64_t plain_registers = 0;
    llvm::DenseMap<mlir::Value, int64_t> cipher_register_file;
    llvm::DenseMap<mlir::Value, int64_t> plain_register_file;
    for (auto &&arg : func.getArguments()) {
      auto tt = arg.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
      if (tt.getNumPoly() == 1) {
        plain_register_file.insert({arg, plain_registers++});
      } else {
        cipher_register_file.insert({arg, cipher_registers++});
      }
    }
    func.walk([&](mlir::Operation *op) {
      if (auto alloc = dyn_cast<mlir::tensor::EmptyOp>(op)) {
        auto tt = alloc.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
        HEVMInstruction heops;
        heops.opcode = -1;
        insts.push_back(heops);

        if (tt.getNumPoly() == 1) {
          plain_register_file.insert({alloc, plain_registers++});
        } else {
          cipher_register_file.insert({alloc, cipher_registers++});
        }
      } else if (auto ops = dyn_cast<hecate::ckks::HEVMOpInterface>(op)) {
        HEVMInstruction heops =
            ops.getHEVMOperation(plain_register_file, cipher_register_file);
        if (auto scaleOp = dyn_cast<hecate::ckks::ScaleSetCOp>(op)) {
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
        if (auto scale = op->getAttrOfType<FloatAttr>("poseidon_native_scale")) {
          const double value = scale.getValueAsDouble();
          if (heops.opcode != 0 || !(value > 0) || !std::isfinite(value)) {
            op->emitError("native scale requires Encode and positive finite f64");
            invalid_native_scale = true;
          } else {
            uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            native_scales.push_back({insts.size(), heops.lhs,
                                     uint64_t(heops.rhs >> 10), bits});
          }
        }
        insts.push_back(heops);

        if (heops.opcode > 0) {
          cipher_register_file.insert({op->getResult(0), heops.dst});
        } else {
          plain_register_file.insert({op->getResult(0), heops.dst});
        }
      }
      else if ((expanded || wide) && !isa<func::FuncOp, func::ReturnOp>(op)) {
        op->emitError("expanded bootstrap program contains unsupported operation; refusing to drop it");
        invalid_native_scale = true;
      }
    });

    if (invalid_native_scale) {
      signalPassFailure();
      return;
    }
    SmallVector<int64_t> ret_dst;
    auto retOp =
        dyn_cast<func::ReturnOp>(func.getBlocks().front().getTerminator());
    for (auto arg : retOp.getOperands()) {
      ret_dst.push_back(cipher_register_file[arg]);
    }

    auto arg_scale_array =
        func->getAttrOfType<DenseI64ArrayAttr>("arg_scale").asArrayRef();
    auto arg_level_array =
        func->getAttrOfType<DenseI64ArrayAttr>("arg_level").asArrayRef();
    auto res_scale_array =
        func->getAttrOfType<DenseI64ArrayAttr>("res_scale").asArrayRef();
    auto res_level_array =
        func->getAttrOfType<DenseI64ArrayAttr>("res_level").asArrayRef();

    config_body.num_operations = insts.size();
    config_body.num_ctxt_buffer = cipher_registers;
    config_body.num_ptxt_buffer = plain_registers;
    config_body.init_level =
        func->getAttrOfType<IntegerAttr>("init_level").getInt();

    config_body_ints.append(arg_scale_array.begin(), arg_scale_array.end());
    config_body_ints.append(arg_level_array.begin(), arg_level_array.end());
    config_body_ints.append(res_scale_array.begin(), res_scale_array.end());
    config_body_ints.append(res_level_array.begin(), res_level_array.end());
    config_body_ints.append(ret_dst.begin(), ret_dst.end());

    config_body.config_body_length +=
        config_body_ints.size() * sizeof(uint64_t);

    std::filesystem::path printpath(prefix.getValue());

    printpath = std::string(printpath) + "." + func.getName().str() + ".hevm";

    // Validate before opening any artifact: never silently truncate indices.
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
    std::ofstream of(printpath, std::ios::binary);
    of.write((char *)&header, sizeof(HEVMHeader));
    of.write((char *)&config_body, sizeof(ConfigBody));
    of.write((char *)config_body_ints.data(),
             config_body_ints.size() * sizeof(uint64_t));
    for (const auto &inst : insts) {
      if (wide) {
        HEVMWideOperation packed{uint32_t(inst.opcode), uint32_t(inst.dst),
                                 uint32_t(inst.lhs), uint32_t(inst.rhs)};
        of.write(reinterpret_cast<const char *>(&packed), sizeof(packed));
      } else {
        HEVMOperation packed{uint16_t(inst.opcode), uint16_t(inst.dst),
                             uint16_t(inst.lhs), uint16_t(inst.rhs)};
        of.write(reinterpret_cast<const char *>(&packed), sizeof(packed));
      }
    }
    of.close();
    if (!native_scales.empty()) {
      // FNV binds metadata to these exact emitted bytes (integrity, not security).
      uint64_t hash = 14695981039346656037ULL;
      std::ifstream emitted(printpath, std::ios::binary);
      char byte;
      while (emitted.get(byte)) {
        hash ^= static_cast<unsigned char>(byte);
        hash *= 1099511628211ULL;
      }
      std::ofstream scales(std::string(printpath) + ".native_scales");
      scales << "POSEIDON_HEVM_NATIVE_SCALE_V1 " << insts.size()
             << ' ' << std::hex << hash << std::dec << '\n';
      for (const auto &row : native_scales)
        scales << row[0] << ' ' << row[1] << ' ' << row[2] << ' '
               << std::hex << row[3] << std::dec << '\n';
      if (!scales) {
        func.emitError("cannot write native scale metadata");
        signalPassFailure();
      }
    }
    writeExpandedMetadata(std::string(printpath), insts, primitive_params, ".native_primitives", "POSEIDON_HEVM_NATIVE_PRIMITIVE_V1");
    writeExpandedMetadata(std::string(printpath), insts, bootstrap_groups, ".bootstrap_groups", "POSEIDON_HEVM_BOOTSTRAP_GROUPS_V1");
  }

  void writeExpandedMetadata(const std::string &path,
      llvm::ArrayRef<HEVMInstruction> insts,
      const std::vector<std::array<uint64_t, 3>> &rows,
      const char *suffix, const char *magic) {
    if (rows.empty()) return;
    uint64_t hash=14695981039346656037ULL;
    std::ifstream file(path, std::ios::binary); char byte;
    while(file.get(byte)){hash^=static_cast<unsigned char>(byte);hash*=1099511628211ULL;}
    std::ofstream out(path+suffix);
    out << magic << ' ' << insts.size() << ' ' << std::hex << hash << std::dec << '\n';
    for(const auto &row:rows)
      out << row[0] << ' ' << row[1] << ' ' << std::hex << row[2] << std::dec << '\n';
    if(!file.eof() || !out){getOperation().emitError("cannot bind expanded bootstrap metadata");signalPassFailure();}
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::ckks::CKKSDialect>();
  }
};
} // namespace
