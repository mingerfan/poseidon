"""Read-only checks against retained real compiler evidence, no new FHE execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("compiler_contexts",HERE/"bind_compiler_context_examples.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from workspace_paths import RESULTS
AUDIT=RESULTS/"upstream-chunks-r29-compiler-evidence.json"

class CompilerContextBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=module.read(AUDIT)
        cls.records={r["evidence"]:r for r in cls.audit["records"]}
    def record(self,name):
        return copy.deepcopy(self.records[self.audit["observed_requirements"]["compiler."+name][0]])
    def test_five_partitions_three_real_topologies_each(self):
        result=module.build(AUDIT,RESULTS)
        self.assertEqual(result["new_encrypted_executions"],0)
        for name,examples in result["examples"].items():
            self.assertEqual(len(examples),3)
            self.assertEqual(len({e["topology"] for e in examples}),3)
            self.assertTrue(all(e["actual_encrypted_execution_recorded"] for e in examples))
    def test_comparison_cannot_be_rewritten_with_still_passing_values(self):
        r=self.record("rescale")
        r["comparison"]["actual"][0][0]+=1e-12
        with self.assertRaisesRegex(ValueError,"Comparison record drift"):
            module.verified_example(r,"compiler.rescale",RESULTS)
    def test_operation_count_cannot_be_invented(self):
        r=self.record("rescale");r["compiler_evidence"]["rescale"]["artifact_operations"]+=1
        with self.assertRaisesRegex(ValueError,"operation count mismatch"):
            module.verified_example(r,"compiler.rescale",RESULTS)
    def test_fused_relinearization_not_per_instruction_runtime_trace(self):
        r=self.record("relinearization");r["compiler_evidence"]["relinearization"]["per_operation_runtime_observed"]=True
        with self.assertRaisesRegex(ValueError,"overclaimed runtime observation"):
            module.verified_example(r,"compiler.relinearization",RESULTS)
    def test_security_record_must_match_executed_report(self):
        r=self.record("security_parameters")
        r["compiler_evidence"]["security_parameters"]["actual_generated_parameters"]["polynomial_degree"]=65536
        with self.assertRaisesRegex(ValueError,"Security parameter record drift"):
            module.verified_example(r,"compiler.security_parameters",RESULTS)
    def test_repeated_artifact_is_not_three_topologies(self):
        changed=copy.deepcopy(self.audit)
        paths=changed["observed_requirements"]["compiler.rescale"]
        changed["observed_requirements"]["compiler.rescale"]=[paths[0]]*3
        with tempfile.TemporaryDirectory(prefix="poseidon-compiler-context-test-") as tmp:
            p=Path(tmp)/"audit.json";p.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"Fewer than three verified topologies"):
                module.build(p,RESULTS)
if __name__=="__main__":unittest.main()
