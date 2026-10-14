"""Arithmetic intake, provenance and retained-version regressions."""
import copy,hashlib,json,operator,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from benchmark_graph import digest
from pytorch_arithmetic_normalization import normalize,rename_record,bind_record,check_record
from pytorch_static_trace import trace,INDEX_POLICY
from pytorch_model_capture import capture
import pytorch_model_intake as intake

PROFILE="general-tensors-v1"
ROOT=Path(__file__).resolve().parents[2]
CASES=ROOT/"scripts/baseline/cases/trusted-pytorch-arithmetic-v8"

class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("public",torch.tensor([[.125,-.25],[.375,-.5]],dtype=torch.float64))
    def forward(self,x,y):
        return self.public-x+y

def captured():
    signature=[dict(name="first",shape=[2,2]),dict(name="second",shape=[2,2])]
    original=trace(Source().eval(),signature,resource_profile=PROFILE,policy=INDEX_POLICY)
    normalized,record=normalize(original,resource_profile=PROFILE)
    canonical=trace(normalized,signature,resource_profile=PROFILE,policy=INDEX_POLICY)
    record=rename_record(record,normalized,canonical)
    high,_,_=capture(canonical,signature,["result"],resource_profile=PROFILE,
                    dtype_policy="strict-float64-v1",preserve_output_structure=True,
                    capture_policy=INDEX_POLICY,matrix_policy="static-tensor-matmul-v1")
    return high,bind_record(record,high,captured_fx=canonical),normalized,canonical

def resign(record,graph):
    record["captured_graph_sha256"]=digest(graph)
    record.pop("binding_sha256",None);record["binding_sha256"]=digest(record)
    return record

class ProvenanceTests(unittest.TestCase):
    def test_attribute_read_reorder_is_accepted_with_exact_dependencies(self):
        high,record,normalized,canonical=captured()
        self.assertNotEqual([n.op for n in normalized.graph.nodes],[n.op for n in canonical.graph.nodes])
        check_record(record,high)
        self.assertEqual(record["capture_name_map"]["sub"],"add")
        self.assertEqual(record["fx_inputs"],["x","y"])
        entry=record["entries"][0]
        self.assertEqual(entry["action"],"negate_cipher_then_add_public")
        self.assertEqual(entry["source_operands"][1],dict(kind="fx_value",name="x"))

    def test_retrace_changed_operation_is_rejected(self):
        _,_,normalized,canonical=captured()
        changed=copy.deepcopy(canonical)
        next(n for n in changed.graph.nodes if n.op=="call_function" and n.target is operator.add).target=operator.sub
        _,record=normalize(normalized,resource_profile=PROFILE)
        with self.assertRaisesRegex(ValueError,"retrace semantics"):
            rename_record(record,normalized,changed)

    def test_retrace_changed_output_is_rejected(self):
        _,_,normalized,canonical=captured()
        changed=copy.deepcopy(canonical)
        output=next(n for n in changed.graph.nodes if n.op=="output")
        output.args=(next(n for n in changed.graph.nodes if n.op=="placeholder"),)
        _,record=normalize(normalized,resource_profile=PROFILE)
        with self.assertRaisesRegex(ValueError,"retrace semantics"):
            rename_record(record,normalized,changed)

    def test_retrace_changed_public_buffer_is_rejected(self):
        _,_,normalized,canonical=captured()
        changed=copy.deepcopy(canonical)
        changed.public.add_(.125)
        _,record=normalize(normalized,resource_profile=PROFILE)
        with self.assertRaisesRegex(ValueError,"public tensor changed"):
            rename_record(record,normalized,changed)

    def test_cipher_operands_cannot_be_swapped_by_rebinding_report(self):
        high,record,_,_=captured()
        changed=copy.deepcopy(high)
        node=changed["nodes"][-1]
        node["inputs"]=node["inputs"][::-1]
        with self.assertRaisesRegex(ValueError,"cipher operand binding"):
            check_record(resign(copy.deepcopy(record),changed),changed)

    def test_negation_cannot_bind_the_other_input(self):
        high,record,_,_=captured()
        changed=copy.deepcopy(high)
        next(n for n in changed["nodes"] if n["op"]=="negate")["inputs"]=["second"]
        with self.assertRaisesRegex(ValueError,"arithmetic negation"):
            check_record(resign(copy.deepcopy(record),changed),changed)

    def test_wrong_source_operand_mapping_is_rejected(self):
        high,record,_,_=captured()
        changed=copy.deepcopy(record)
        changed["entries"][0]["source_operands"][1]["name"]="y"
        with self.assertRaisesRegex(ValueError,"source operand mapping"):
            check_record(resign(changed,high),high)

    def test_public_content_change_is_rejected_even_with_new_report_hash(self):
        high,record,_,_=captured()
        changed=copy.deepcopy(high)
        name=next(n for n in changed["nodes"] if n["op"]=="add")["inputs"][1]
        changed["constants"][name][0][0]+=.125
        with self.assertRaisesRegex(ValueError,"public tensor binding"):
            check_record(resign(copy.deepcopy(record),changed),changed)

    def test_source_text_hash_and_action_are_checked(self):
        high,record,_,_=captured()
        for key,value in (("source_fx_text",record["source_fx_text"]+" "),("source_graph_preserved",False)):
            changed=copy.deepcopy(record);changed[key]=value
            with self.assertRaises(ValueError):check_record(resign(changed,high),high)
        changed=copy.deepcopy(record);changed["entries"][0]["action"]="canonical_call"
        with self.assertRaisesRegex(ValueError,"entry action"):
            check_record(resign(changed,high),high)

    def test_split_port_operand_is_bound(self):
        class Split(torch.nn.Module):
            def forward(self,x,y):
                a,b=torch.split(x,[2,2],dim=1)
                return a-y,b+y
        signature=[dict(name="x",shape=[2,4]),dict(name="y",shape=[2,2])]
        raw=trace(Split().eval(),signature,resource_profile=PROFILE,policy=INDEX_POLICY)
        normalized,record=normalize(raw,resource_profile=PROFILE)
        canonical=trace(normalized,signature,resource_profile=PROFILE,policy=INDEX_POLICY)
        record=rename_record(record,normalized,canonical)
        high,_,_=capture(canonical,signature,["left","right"],resource_profile=PROFILE,
                        dtype_policy="strict-float64-v1",preserve_output_structure=True,
                        capture_policy=INDEX_POLICY,matrix_policy="static-tensor-matmul-v1")
        bound=bind_record(record,high,captured_fx=canonical);check_record(bound,high)
        self.assertEqual(len(bound["split_aliases"]),2)
        changed=copy.deepcopy(high)
        n=next(n for n in changed["nodes"] if n["op"]=="subtract")
        n["inputs"][0]=next(n for n in changed["nodes"] if n["op"]=="split")["outputs"][1]
        with self.assertRaisesRegex(ValueError,"cipher operand binding"):
            check_record(resign(copy.deepcopy(bound),changed),changed)

class FacadeTests(unittest.TestCase):
    def test_all_legacy_versions_keep_their_package_format(self):
        for version in range(1,8):
            spec=dict(format="poseidon-trusted-pytorch-v"+str(version))
            with patch.object(intake.legacy,"package_format",return_value="old") as delegated:
                self.assertEqual(intake.package_format(spec),"old")
                delegated.assert_called_once_with(spec)
    def test_legacy_prepare_and_run_delegate_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"manifest.json"
            path.write_text(json.dumps(dict(format="poseidon-trusted-pytorch-v7")))
            with patch.object(intake.legacy,"prepare",return_value="old") as prepare:
                self.assertEqual(intake.prepare(path),"old");prepare.assert_called_once_with(path)
            with patch.object(intake.legacy,"prepare",return_value=({},{})),patch.object(intake.legacy,"run",return_value="old") as run:
                self.assertEqual(intake.run(path,Path(tmp)/"out",trusted=True),"old")
                run.assert_called_once()
    def test_v8_manifest_retains_explicit_version(self):
        path=CASES/"v8_subtract_tensor_left_operator/manifest.json"
        before=path.read_bytes();spec,_=intake.prepare(path)
        self.assertEqual(spec["format"],intake.FORMAT)
        self.assertEqual(intake.shim(spec)["format"],intake.legacy.SOURCE_V7)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(intake.package_format(spec),intake.PACKAGE_FORMAT)
    def test_source_identity_keeps_all_seventeen_legacy_hashes(self):
        identity=intake.source_identity()
        self.assertEqual(len(identity),19)
        for name in intake.legacy.IMPLEMENTATION:
            self.assertEqual(identity[name],hashlib.sha256((ROOT/"scripts/baseline"/name).read_bytes()).hexdigest())
    def test_flat_and_reserved_files_rejected_before_source_read(self):
        base=json.loads((CASES/"v8_subtract_tensor_left_operator/manifest.json").read_text())
        for name in ("../model.py",".env","pytorch_model_intake.py"):
            spec=copy.deepcopy(base);spec["files"]={name:"0"*64}
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"manifest.json";path.write_text(json.dumps(spec))
                with patch.object(intake,"regular_bytes",side_effect=AssertionError("Source read must not happen")) as read:
                    with self.assertRaises(ValueError):intake.prepare(path)
                    read.assert_not_called()
    def test_manifest_budget_is_checked_before_copy(self):
        value=dict(format=intake.FORMAT,padding="x"*131073)
        with self.assertRaisesRegex(ValueError,"manifest data/budget"):intake.shim(value)
    def test_application_identity_binds_the_exact_isolated_loader(self):
        from application_identity import source_closure,DYNAMIC
        pair=("pytorch_model_intake.py","importlib.import_module(module_name)")
        self.assertIn(pair,DYNAMIC)
        closure=source_closure(ROOT/"scripts/baseline",["pytorch_model_intake.py"])
        self.assertIn("pytorch_model_intake.py",closure["sources"])
        self.assertIn("pytorch_arithmetic_normalization.py",closure["sources"])
        denied=dict(DYNAMIC);denied.pop(pair)
        with self.assertRaisesRegex(ValueError,"Unbounded dynamic dependency"):
            source_closure(ROOT/"scripts/baseline",["pytorch_model_intake.py"],dynamic_authorities=denied)
    def test_metadata_requires_the_arithmetic_record(self):
        spec=json.loads((CASES/"v8_subtract_tensor_left_operator/manifest.json").read_text())
        with self.assertRaisesRegex(ValueError,"Arithmetic capture metadata"):
            intake.check_capture_metadata({}, {},spec,resource_profile=PROFILE)

if __name__=="__main__":unittest.main()
