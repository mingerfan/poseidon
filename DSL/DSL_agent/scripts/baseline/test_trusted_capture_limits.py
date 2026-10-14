"""Shared limits in trusted capture: import boundaries and fresh identity checks."""
import copy,hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from trusted_pytorch_import import IMPLEMENTATION,source_identity,check_reference,REFERENCE_FORMAT,runtime_identity
from benchmark_graph import digest
from model_resources import identity

class TrustedCaptureLimitsTests(unittest.TestCase):
    def test_shared_limit_bytes_are_in_capture_identity(self):
        self.assertIn("process_resource_limits.py",IMPLEMENTATION)
        current=source_identity()
        self.assertEqual(current["process_resource_limits.py"],
            hashlib.sha256(Path(__file__).parent.joinpath("process_resource_limits.py").read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in IMPLEMENTATION:(root/name).write_bytes(Path(__file__).parent.joinpath(name).read_bytes())
            before=source_identity(root)
            (root/"process_resource_limits.py").write_text("# isolated drift fixture\n")
            after=source_identity(root)
            self.assertEqual({n for n in before if before[n]!=after[n]},{"process_resource_limits.py"})

    def test_reference_identity_rejects_missing_or_drifted_limits(self):
        original={"value":"fixed"};lowered={"value":"fixed"};spec={"format":"poseidon-trusted-pytorch-v1"}
        report=dict(format=REFERENCE_FORMAT,mode="isolated_trusted_pytorch_reference",
            original_sha256=digest(original),model_sha256=digest(lowered),manifest_sha256=digest(spec),
            source_hashes=source_identity(),runtime=runtime_identity(),probes=16,resource_profile=identity("legacy-v1"),
            reference_status="passed",original_module_checked=True,pure_forward_checked=True,
            encrypted_execution=False,agent_generated=False,paid_calls=0,runtime_layout_validated=False,
            approximation_accuracy_certified=False,comparison_tolerance=dict(atol=1e-12,rtol=1e-12),
            max_decomposition_error=0,max_original_module_error=0,max_original_function_approximation_error=0,
            max_reference_tolerance_ratio=0)
        self.assertIs(check_reference(report,original,lowered,spec),report)
        for missing in (True,False):
            bad=copy.deepcopy(report)
            if missing:bad["source_hashes"].pop("process_resource_limits.py")
            else:bad["source_hashes"]["process_resource_limits.py"]="0"*64
            with self.subTest(missing=missing),self.assertRaisesRegex(ValueError,"Trusted reference identity"):
                check_reference(bad,original,lowered,spec)
        drift=source_identity();drift["process_resource_limits.py"]="0"*64
        with patch("trusted_pytorch_import.source_identity",return_value=drift):
            with self.assertRaisesRegex(ValueError,"Trusted reference identity"):
                check_reference(report,original,lowered,spec)

    def test_run_preparation_does_not_import_historical_sandbox_or_provider(self):
        script="""
import importlib.abc,sys
from unittest.mock import patch
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in {'candidate_sandbox','sandbox_policy','upstream_helper_probe',
                                     'benchmark_runner','deepseek_provider','deepseek_http_worker'}:
            raise AssertionError('forbidden capture import: '+fullname)
sys.meta_path.insert(0,Guard())
import trusted_pytorch_import
class ReachedPreparation(Exception):pass
with patch.object(trusted_pytorch_import,'prepare',side_effect=ReachedPreparation):
    try:trusted_pytorch_import.run('unused.json','unused',resource_profile='general-tensors-v1',trusted=True)
    except ReachedPreparation:pass
    else:raise AssertionError('Capture preparation was not reached')
assert 'process_resource_limits' in sys.modules
assert not any(name in sys.modules for name in ('candidate_sandbox','sandbox_policy','upstream_helper_probe','deepseek_provider'))
"""
        script="import sys;sys.path.insert(0,"+repr(str(Path(__file__).parent))+")\n"+script
        child=subprocess.run([sys.executable,"-B","-c",script],env=dict(os.environ),
                             capture_output=True,text=True,timeout=30)
        self.assertEqual(child.returncode,0,child.stderr)

    def test_explicit_trust_check_precedes_new_capture_imports(self):
        from trusted_pytorch_import import run
        with patch("trusted_pytorch_import.prepare",side_effect=AssertionError("No preparation without trust")):
            with self.assertRaisesRegex(ValueError,"Explicit"):
                run("unused.json","unused",trusted=False)

if __name__=="__main__":unittest.main()
