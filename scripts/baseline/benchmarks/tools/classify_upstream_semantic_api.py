"""Classify the pinned upstream API by reviewed source, never by execution success.

Draft data for later explicit adoption by benchmark_semantics; does not expose
new imports/calls to candidate programs or modify any active runtime contract.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from benchmark_graph import digest

# Exhaustive names, deliberately not prefix fallbacks: new upstream symbols fail.
GROUPS = {
 "Func.py": {
  "public_fhe_helper": "HE_BN HE_MPBN HE_Conv HE_ConvBN HE_MaxPad HE_Max HE_Avg HE_DS HE_Pool HE_Linear HE_ReshapeLinear HE_DwConv HE_Concat HE_ReLU HE_SiLU",
  "bootstrap_helper_closure": "HE_MaxPad.maximum HE_Max.maximum HE_ReLU.sign",
 },
 "MPCB.py": {
  "public_integer_shape_utility": "cint fint",
  "mutable_maximum_hook": "maximum",
  "cipher_rotation_wrapper": "roll",
  "polynomial_factory_and_closure": "GenPoly GenPoly.polynomial",
  "packing_closure_factory": "shapeClosure",
  "public_tensor_pack_unpack": "shapeClosure.MultParPack shapeClosure.OutPack",
  "public_weight_and_mask_transform": "shapeClosure.ParMultWgt shapeClosure.DwMultWgt shapeClosure.Selecting shapeClosure.ParBNConst shapeClosure.ParInBNConst shapeClosure.DownSelecting shapeClosure.ParMPDA shapeClosure.ParMPDM shapeClosure.PoolSelecting shapeClosure.ConcatSelecting abstractBN Reshape",
  "upstream_unimplemented_stub": "shapeClosure.AvgMidSelecting",
  "cipher_reduction_dependency": "shapeClosure.SumSlots",
  "cipher_helper_closure": "shapeClosure.Downsamp shapeClosure.AvgPool shapeClosure.AvgMidPool shapeClosure.Concat shapeClosure.DwConvBN shapeClosure.MultParBN shapeClosure.MultParConv shapeClosure.MultParConvBN",
  "bootstrap_maximum_closure": "shapeClosure.MaxPool shapeClosure.MaxPoolPad",
  "cipher_dense_affine_dependency": "Linear BN",
  "public_packing_shape_propagation": "InferShapes CascadeConcat CascadeConv CascadeMax CascadeDS CascadePool",
 },
 "Poly.py": {
  "fixed_polynomial_factory": "GenPoly",
  "standalone_approximation_composition": "sign genRelu6 relua",
  "bootstrap_maximum_composition": "maxx maxx.sign",
  "plaintext_diagnostic_utility": "ReLU rms nprelu",
 },
 "expr.py": {
  "trusted_framework_io": "save removeCtxt",
  "bootstrap_frontend_factory": "unaryFactory unaryFactory.unaryMethod unaryFactory.unaryMethod.apply",
  "frontend_metaclass_and_operator_factory": "hecateMetaBase hecateMetaBase.__new__ hecateMetaBinary hecateMetaBinary.__new__ hecateMetaBinary.__new__.binaryFactory hecateMetaBinary.__new__.innerFactory",
  "copy_rejection": "hecateMetaBase.__new__.raiser",
  "binary_operand_order": "hecateMetaBinary.__new__.binaryFactory.binaryMethod hecateMetaBinary.__new__.binaryFactory.binaryReverseMethod",
  "defined_but_unregistered_inplace": "hecateMetaBinary.__new__.binaryFactory.binaryInplaceMethod",
  "cipher_negation_implementation": "hecateMetaBinary.__new__.innerFactory.innerMethod",
  "cipher_rotation_implementation": "hecateMetaBinary.__new__.rotate",
  "trusted_framework_introspection": "getProperFrame",
  "legacy_public_literal_utility": "recType flatten",
  "implicit_public_operand_conversion": "resolveType",
  "frontend_expression_handle": "Expr Expr.__init__",
  "plaintext_expression_constructor": "Plain Plain.__init__",
  "empty_accumulator_semantics": "Empty Empty.__init__ Empty.__add__ Empty.__radd__ Empty.__iadd__ Empty.__sub__ Empty.__rsub__ Empty.__isub__",
  "native_function_tracing": "func func.generateFunc Func Func.__init__ Func.eval Func.__call__",
 },
}

NOTES = {
 "public_fhe_helper": "Candidate access only via a specific checked helper profile; source presence does not broaden supported parameters.",
 "bootstrap_helper_closure": "Calls real hc.bootstrap; keep blocked, never replace it with decrypt/re-encrypt.",
 "public_integer_shape_utility": "Ceil/floor of public shape arithmetic; not encrypted rounding.",
 "mutable_maximum_hook": "Default torch.maximum is replaced by HE_Max/HE_MaxPad; distinguish default plaintext hook from bootstrap encrypted path.",
 "cipher_rotation_wrapper": "Maps roll(A,i) to A.rotate(-i); direction is part of the contract.",
 "polynomial_factory_and_closure": "Builds a fixed-coefficient Chebyshev evaluation closure; trees, coefficients and scale must be bound.",
 "packing_closure_factory": "Captures packing geometry and returns keyed helper closures; construction alone is not numeric coverage.",
 "public_tensor_pack_unpack": "Rearranges public torch tensors for input/output packing; not arbitrary encrypted reshape.",
 "public_weight_and_mask_transform": "Public constants/layout transformation used by checked helper paths; every branch is not individually traced.",
 "upstream_unimplemented_stub": "Body is pass and has no returned operation; do not claim functional coverage.",
 "cipher_reduction_dependency": "Rotate/add reduction handles power-of-two and remainder branches; requires packing-sensitive contexts.",
 "cipher_helper_closure": "Real MPCB operation over expression containers; bounded adapter mappings determine supported layouts.",
 "bootstrap_maximum_closure": "Uses mutable maximum hook; HE_Max/HE_MaxPad supply bootstrap-dependent implementation.",
 "cipher_dense_affine_dependency": "Fixed nt affine operation; actual helper adapters bind layout and public parameters.",
 "public_packing_shape_propagation": "Public dictionary/model shape propagation only; CascadeMax itself does not execute Max or bootstrap.",
 "fixed_polynomial_factory": "Uses source-pinned coefficient/tree data and MPCB.GenPoly, including actual coefficient structure.",
 "standalone_approximation_composition": "Publicly named polynomial composition not directly exposed by current helper profiles; requires distinct capacity/approximation checks.",
 "bootstrap_maximum_composition": "Nested sign calls bootstrap; exact source operation must remain visible as blocked.",
 "plaintext_diagnostic_utility": "Uses NumPy maximum/sqrt/mean on plaintext; not evidence for encrypted ReLU or sqrt support.",
 "trusted_framework_io": "Owned by trusted tracing lifecycle; generated programs may not choose arbitrary paths or teardown the context.",
 "bootstrap_frontend_factory": "Creates unary opcode 0 and container mutation behavior; frontend tracing is not real backend bootstrap.",
 "frontend_metaclass_and_operator_factory": "Trusted dynamic operator installation, not permission for user-defined metaclasses.",
 "copy_rejection": "Installs copy/deepcopy raisers; intentional rejection, not a missing numerical operator.",
 "binary_operand_order": "Creates binary IR with explicit left/right operand order; __i*__ is installed as binaryMethod.",
 "defined_but_unregistered_inplace": "Function mutates self.obj but is never installed; actual __i*__ uses binaryMethod, so do not attribute mutation semantics to this unused definition.",
 "cipher_negation_implementation": "Installed inner unary negation opcode 13; distinguish object-array outer operations.",
 "cipher_rotation_implementation": "Creates frontend rotation with literal offset; key/layout restrictions remain caller-bound.",
 "trusted_framework_introspection": "Finds source location in call stack for trusted frontend; candidate introspection remains rejected.",
 "legacy_public_literal_utility": "Recursive public-list shape/flatten utilities; not encrypted tensor reshape and not direct candidate imports.",
 "implicit_public_operand_conversion": "Converts allowed public operands to Plain; upstream capability is broader than the restricted candidate allowlist.",
 "frontend_expression_handle": "Wraps frontend object handle; candidates cannot manufacture or mutate raw C++ handles.",
 "plaintext_expression_constructor": "Creates a public constant via ctypes; fixed candidate constants and allocation bounds still apply.",
 "empty_accumulator_semantics": "Identity-like accumulation; all subtraction methods return resolveType(other) in this pinned source. Do not assume ordinary zero-minus-x semantics.",
 "native_function_tracing": "Signature binding, return container preservation, forward calls, recursion rejection and failed-context rejection.",
}

BOOTSTRAP = {"HE_MaxPad", "HE_Max", "HE_ReLU"}

def nodes(path):
    result = {}
    def visit(body, prefix=""):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                name = prefix+n.name
                result[name] = n
                visit(n.body, name+".")
    visit(ast.parse(path.read_bytes()).body)
    return result

def build(ledger, root=ROOT):
    mappings = {}
    for file, groups in GROUPS.items():
        table = {}
        for role, names in groups.items():
            if role not in NOTES:
                raise ValueError("Missing reviewed role explanation")
            for name in names.split():
                if name in table:
                    raise ValueError("Duplicate reviewed symbol")
                table[name] = role
        mappings[file] = table
    actual = {}
    for row in ledger["upstream_api"]:
        file = Path(row["source"]).name
        actual.setdefault(file, set()).add(row["symbol"])
    if actual != {file:set(names) for file,names in mappings.items()}:
        raise ValueError("Unreviewed upstream API drift")
    parsed = {source:nodes(root/source) for source in {r["source"] for r in ledger["upstream_api"]}}
    records = []
    for row in ledger["upstream_api"]:
        file, symbol = Path(row["source"]).name, row["symbol"]
        path = root/row["source"]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if ledger["upstream_source_hashes"].get(row["source"]) != sha:
            raise ValueError("Pinned source drift")
        node = parsed[row["source"]][symbol]
        if node.lineno != row["line"]:
            raise ValueError("Source location drift")
        role = mappings[file][symbol]
        blocked = (file=="Func.py" and symbol.split(".")[0] in BOOTSTRAP) or role in (
            "bootstrap_maximum_closure", "bootstrap_maximum_composition",
            "bootstrap_frontend_factory", "bootstrap_helper_closure")
        candidate_surface = ("bound_helper_only" if role=="public_fhe_helper" else
                             "restricted_frontend_syntax" if role in (
                              "binary_operand_order","cipher_negation_implementation",
                              "cipher_rotation_implementation","plaintext_expression_constructor",
                              "empty_accumulator_semantics","native_function_tracing") else
                             "no_direct_candidate_access")
        record = dict(source=row["source"], source_sha256=sha, symbol=symbol,
                      line=node.lineno, end_line=node.end_lineno,
                      ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
                      role=role, source_review=NOTES[role], candidate_surface=candidate_surface,
                      execution_evidence="not_individually_audited",
                      acceptance_not_inferred_from_helper_success=True,
                      backend_blocker="real_bootstrap" if blocked else
                          "upstream_stub" if role=="upstream_unimplemented_stub" else None)
        if role=="public_fhe_helper":
            record["related_requirement_ids"]=["helper."+symbol]
        else:
            record["related_requirement_ids"]=[]
        records.append(record)
    result = dict(schema=1,kind="reviewed_upstream_api_classification_draft",
                  authoritative_ledger=False, changes_candidate_permissions=False,
                  classification_is_not_execution_coverage=True,
                  records=records, unclassified=0, paid_api_calls=0)
    result["classification_sha256"]=digest(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Preserve existing classification")
    ledger=json.loads((a.suite/"coverage.json").read_text())
    result=build(ledger)
    result["generator_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["ledger_sha256"]=hashlib.sha256((a.suite/"coverage.json").read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    from collections import Counter
    print(json.dumps(dict(records=len(result["records"]),roles=dict(Counter(r["role"] for r in result["records"])),
                          unclassified=0,execution_claimed=False)))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
