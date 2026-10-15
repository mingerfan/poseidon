// Primitive operations absent from the legacy HEVM dialect.
// These are not bootstrap/EvalMod/DFT callbacks.
class NativeUnary<string name, int opcode, dag args, code rhs, code condition>
    : CKKS_Op<name, [SameOperandsAndResultShape]> {
  let arguments = args;
  code extraClassDefinition = arithClassDefinition # [{
    ::mlir::LogicalResult $cppClass::inferReturnTypes(
        ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
        ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
        ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type> &types) {
      auto op = $cppClass::Adaptor(operands, attributes, properties, regions);
      auto dst = hecate::ckks::getPolyType(op.getDst());
      auto src = hecate::ckks::getPolyType(op.getSrc());
      if (!(}] # condition # [{)) return ::mlir::failure();
      types.push_back(op.getDst().getType());
      return ::mlir::success();
    }
    HEVMOperation $cppClass::getHEVMOperation(
        llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap) {
      HEVMOperation op;
      op.opcode = }] # opcode # [{;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getSrc()];
      op.rhs = }] # rhs # [{;
      return op;
    }
  }];
}

def ConjugateCOp : NativeUnary<"conjugatec", 11,
    (ins PolyTensor:$dst, PolyTensor:$src), [{0}],
    [{dst.getNumPoly()==2 && src.getNumPoly()==2 && dst.getLevel()==src.getLevel()}]>;
def ModRaiseCOp : NativeUnary<"modraisec", 12,
    (ins PolyTensor:$dst, PolyTensor:$src, I64Attr:$level), [{getLevel()}],
    [{dst.getNumPoly()==2 && src.getNumPoly()==2 && dst.getLevel()>src.getLevel() && dst.getLevel()==op.getLevel()}]>;
// rhs is reserved; EmitHEVM writes the exact f64 in a required sidecar.
def ScaleSetCOp : NativeUnary<"setscalec", 13,
    (ins PolyTensor:$dst, PolyTensor:$src, F64Attr:$scale), [{0}],
    [{dst.getNumPoly()==src.getNumPoly() && dst.getLevel()==src.getLevel() && op.getScaleAttr().getValueAsDouble()>0.0 && op.getScaleAttr().getValueAsDouble()<=1.7976931348623157e308}]>;
def MultiplyScalarCOp : NativeUnary<"multiplyscalarc", 14,
    (ins PolyTensor:$dst, PolyTensor:$src, I64Attr:$multiplier), [{getMultiplier()}],
    [{dst.getNumPoly()==src.getNumPoly() && dst.getLevel()==src.getLevel() && op.getMultiplier()>=0 && op.getMultiplier()<=65535}]>;
def RelinearizeCOp : NativeUnary<"relinearizec", 16,
    (ins PolyTensor:$dst, PolyTensor:$src), [{0}],
    [{dst.getNumPoly()==2 && src.getNumPoly()==3 && dst.getLevel()==src.getLevel()}]>;
def RescaleManyCOp : NativeUnary<"rescalemanyc", 17,
    (ins PolyTensor:$dst, PolyTensor:$src, I64Attr:$count), [{getCount()}],
    [{dst.getNumPoly()==src.getNumPoly() && op.getCount()>0 && op.getCount()<=src.getLevel() && dst.getLevel()+op.getCount()==src.getLevel()}]>;

def RawMulCCOp : CKKS_Op<"rawmulcc", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
  code extraClassDefinition = arithClassDefinition # [{
    ::mlir::LogicalResult $cppClass::inferReturnTypes(
        ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
        ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
        ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type> &types) {
      auto op = RawMulCCOpAdaptor(operands, attributes, properties, regions);
      auto dst=hecate::ckks::getPolyType(op.getDst());
      auto lhs=hecate::ckks::getPolyType(op.getLhs());
      auto rhs=hecate::ckks::getPolyType(op.getRhs());
      if (dst.getNumPoly()!=3 || lhs.getNumPoly()!=2 || rhs.getNumPoly()!=2 ||
          dst.getLevel()!=lhs.getLevel() || lhs.getLevel()!=rhs.getLevel())
        return ::mlir::failure();
      types.push_back(op.getDst().getType());
      return ::mlir::success();
    }
    HEVMOperation $cppClass::getHEVMOperation(
        llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap) {
      HEVMOperation op; op.opcode=15; op.dst=cipherMap[getDst()];
      op.lhs=cipherMap[getLhs()]; op.rhs=cipherMap[getRhs()]; return op;
    }
  }];
}
