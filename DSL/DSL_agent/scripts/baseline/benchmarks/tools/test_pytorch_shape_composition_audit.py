"""Developer audit guards; these tests do not claim model/backend support."""
import ast,copy,tempfile,unittest
from pathlib import Path
from benchmark_graph import digest
from benchmarks.tools.generate_pytorch_shape_composition_sources import generate,source
from benchmarks.tools.audit_pytorch_shape_composition import classify,checked,preservation

class ShapeAuditGuards(unittest.TestCase):
    def case(self,**changes):
        row=dict(kind="positive",style="method",query="result_shape",rejection=None)
        row.update(changes);return row
    def log(self,terminal="ValueError: Static shape unavailable for this operation"):
        return 'traced=trace(module,spec["inputs"],resource_profile=profile,policy=INDEX_POLICY)\n'+terminal
    def test_initial_trace_matches_precise_method_context(self):
        for query in ("result_shape","result_size","result_numel"):
            self.assertEqual(classify(self.case(query=query),self.log())["code"],
                             "TENSOR_ARITHMETIC_RESULT_METADATA_MISSING")
    def test_shape_diagnostic_requires_stage_and_fixture(self):
        for row in (self.case(style="operator"),self.case(query="literal"),self.case(kind="rejection")):
            self.assertIsNone(classify(row,self.log()))
        self.assertIsNone(classify(self.case(),"ValueError: Static shape unavailable for this operation"))
        self.assertIsNone(classify(self.case(),self.log()+"\nRuntimeError: environment"))
        self.assertIsNone(classify(self.case(),self.log()+" trailing"))
    def test_environment_failure_is_not_source_rejection(self):
        for reason in ("encrypted_branch","inplace","out_mutation"):
            row=self.case(kind="rejection",rejection=reason)
            self.assertIsNone(classify(row,self.log("RuntimeError: Pinned capture runtime")))
    def test_bindings_and_inventories_reject_tampering(self):
        body=dict(planned_positive=33);body["binding"]=digest(body);checked(body)
        changed=copy.deepcopy(body);changed["planned_positive"]=32
        with self.assertRaisesRegex(ValueError,"binding"):checked(changed)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"frozen.py";p.write_text("original")
            import hashlib
            proposal=dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()},
                          case_hashes={},tool_hashes={})
            preservation(proposal);p.write_text("changed")
            with self.assertRaisesRegex(ValueError,"identity changed"):preservation(proposal)
    def test_source_fixture_inventory_and_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/"first";second=Path(tmp)/"second"
            a=generate(first);b=generate(second)
            self.assertEqual(a,b)
            self.assertEqual((len(a["cases"]),a["mathematical_groups"]),(36,3))
            self.assertEqual(len([r for r in a["cases"] if r["kind"]=="positive"]),33)
            self.assertEqual(len([r for r in a["cases"] if r["kind"]=="rejection"]),3)
            for name in first.rglob("*"):
                if name.is_file():self.assertEqual(name.read_bytes(),(second/name.relative_to(first)).read_bytes())
            with self.assertRaisesRegex(ValueError,"Fresh"):generate(first)
    def test_generated_sources_are_static_and_separate(self):
        for op in ("add","subtract","multiply"):
            for style in ("operator","function","method"):
                ast.parse(source(op,style,"result_shape"))
        self.assertIn("if x.sum()>0",source("add","function","literal","encrypted_branch"))
        self.assertIn("add_",source("add","function","literal","inplace"))
        self.assertIn("out=x",source("add","function","literal","out_mutation"))

if __name__=="__main__":unittest.main()
