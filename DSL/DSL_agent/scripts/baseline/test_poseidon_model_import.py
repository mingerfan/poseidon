"""Poseidon import environment, real FX reference and inert package provenance."""
import copy,hashlib,importlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import candidate_bundle as bundle
import trusted_pytorch_import as trusted
from benchmark_graph import digest
from restricted_model_python import parse
from hecate_python_env import ROOT,WORK
import model_import_cli

SOURCE=ROOT/"scripts/baseline/cases/trusted-pytorch-v1"
RESTRICTED=ROOT/"scripts/baseline/cases/operator-decomposition-v1"

class ImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(dir=WORK/"results")
        cls.capture=Path(cls.temporary.name)/"capture"
        cls.result=trusted.run(SOURCE/"manifest.json",cls.capture,trusted=True)
        read=lambda n:json.loads((cls.capture/n).read_text())
        cls.model=read("model.json")
        cls.value=dict(original=read("original.json"),decomposition=read("decomposition.json"),
            reference_report=read("reference-report.json"),trusted_pytorch=dict(
                format=trusted.PACKAGE_FORMAT,manifest=read("trusted-python/manifest.json"),
                files={n:(cls.capture/"trusted-python"/n).read_text() for n in read("trusted-python/manifest.json")["files"]}))
    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_real_module_and_independent_references(self):
        r=self.value["reference_report"]
        self.assertTrue(self.result["reference_verified"])
        self.assertEqual(r["probes"],16);self.assertTrue(r["original_module_checked"])
        self.assertTrue(r["pure_forward_checked"]);self.assertFalse(r["encrypted_execution"])
        self.assertLessEqual(r["max_reference_tolerance_ratio"],1)
        self.assertEqual(r["runtime"]["numpy"],"1.25.2")
        self.assertIsNone(r["runtime"]["cuda"])

    def test_valid_source_provenance_is_inert(self):
        with patch.object(importlib,"import_module",side_effect=AssertionError("Source must not execute")) as importing:
            self.assertEqual(bundle.check_provenance(self.value,self.model),self.value["decomposition"])
            importing.assert_not_called()

    def test_independent_reaudit_of_import(self):bundle.audit_provenance(self.value,self.model)

    def test_tampered_source_bytes_rejected(self):
        bad=copy.deepcopy(self.value);bad["trusted_pytorch"]["files"]["model.py"]+="\nraise RuntimeError()\n"
        with self.assertRaisesRegex(ValueError,"hash"):bundle.check_provenance(bad,self.model)

    def test_rehashed_source_changes_reference_binding(self):
        bad=copy.deepcopy(self.value);package=bad["trusted_pytorch"]
        package["files"]["model.py"]+="\n# changed source\n"
        package["manifest"]["files"]["model.py"]=hashlib.sha256(package["files"]["model.py"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError,"reference identity"):bundle.check_provenance(bad,self.model)

    def test_tampered_original_weights_rejected(self):
        bad=copy.deepcopy(self.value);key=next(iter(bad["original"]["constants"]))
        bad["original"]["constants"][key][0][0]+=1
        with self.assertRaises(ValueError):bundle.check_provenance(bad,self.model)

    def test_tampered_capture_implementation_rejected(self):
        bad=copy.deepcopy(self.value);bad["decomposition"]["capture_implementation"]["pytorch_model_capture.py"]="0"*64
        with self.assertRaisesRegex(ValueError,"provenance mismatch"):bundle.check_provenance(bad,self.model)

    def test_missing_original_module_check_rejected(self):
        bad=copy.deepcopy(self.value);bad["reference_report"]["original_module_checked"]=False
        with self.assertRaisesRegex(ValueError,"acceptance scope"):bundle.check_provenance(bad,self.model)

    def test_approximation_accuracy_is_not_certified(self):
        bad=copy.deepcopy(self.value);bad["reference_report"]["approximation_accuracy_certified"]=True
        with self.assertRaisesRegex(ValueError,"acceptance scope"):bundle.check_provenance(bad,self.model)

    def test_reference_runtime_and_probe_count(self):
        for field,value in (("probes",4),("runtime",{"torch":"different"})):
            bad=copy.deepcopy(self.value);bad["reference_report"][field]=value
            with self.assertRaisesRegex(ValueError,"runtime/probes"):bundle.check_provenance(bad,self.model)

    def test_reference_failure_nonfinite_and_scope_not_accepted(self):
        for field,value in (("max_reference_tolerance_ratio",1.1),("max_original_module_error",float("nan")),
                            ("paid_calls",1),("encrypted_execution",True)):
            bad=copy.deepcopy(self.value);bad["reference_report"][field]=value
            with self.assertRaises(ValueError):bundle.check_provenance(bad,self.model)

    def test_input_output_binding(self):
        bad=copy.deepcopy(self.value);bad["trusted_pytorch"]["manifest"]["outputs"]=["changed"]
        with self.assertRaisesRegex(ValueError,"input/output binding"):bundle.check_provenance(bad,self.model)

    def test_mixed_python_provenance_rejected(self):
        bad=copy.deepcopy(self.value);bad["python"]={}
        with self.assertRaisesRegex(ValueError,"Ambiguous"):bundle.check_provenance(bad,self.model)

    def test_source_inventory_and_traversal(self):
        for name in ("../outside.py","/absolute.py","sub/module.py"):
            package=copy.deepcopy(self.value["trusted_pytorch"]);text=package["files"].pop("model.py")
            package["files"][name]=text;package["manifest"]["files"]={name:hashlib.sha256(text.encode()).hexdigest()}
            with self.assertRaisesRegex(ValueError,"Flat"):trusted.validate_package(package["manifest"],package["files"])

    def test_legacy_static_python_provenance_preserved(self):
        manifest=json.loads((RESTRICTED/"mlp-python.json").read_text())
        files={n:(RESTRICTED/n).read_text() for n in manifest["files"]}
        original,model,binding=parse(manifest,files)
        value=dict(original=original,decomposition=binding,python=dict(manifest=manifest,files=files),
            reference_report=dict(original_sha256=digest(original),model_sha256=digest(model),
                reference_status="passed",approximation_accuracy_certified=False))
        self.assertEqual(bundle.check_provenance(value,model),binding)

    def test_every_import_reentry_selects_poseidon_cpu(self):
        modes=[["--python-manifest",str(RESTRICTED/"mlp-python.json")],
               ["--graph",str(RESTRICTED/"rmsnorm.json")],
               ["--trusted-pytorch-manifest",str(SOURCE/"manifest.json"),"--trust-local-code"]]
        for mode in modes:
            with tempfile.TemporaryDirectory(dir=WORK/"results") as t:
                args=["prepare_model.py",*mode,"--write","--output",str(Path(t)/"new")]
                with patch("sys.argv",args),patch("hecate_python_env.enter_nix",return_value=0) as entering:
                    self.assertEqual(model_import_cli.main(),0)
                    self.assertEqual(entering.call_args.kwargs["environment"],"poseidon-cpu")

    def test_capture_keeps_files_and_reference(self):
        manifest=self.value["trusted_pytorch"]["manifest"]
        self.assertEqual(set(manifest["files"]),{"model.py"})
        self.assertEqual(trusted.prepare(self.capture/"trusted-python/manifest.json")[0],manifest)
        self.assertEqual(self.result["model_sha256"],digest(self.model))

    def test_capture_worker_filesystem_network_and_environment_isolation(self):
        with tempfile.TemporaryDirectory(dir=WORK/"results") as t:
            root=Path(t);host=root/"host-only.txt";host.write_text("synthetic private marker")
            code="import torch,os,socket,json\nfrom pathlib import Path\nclass Model(torch.nn.Module):\n def forward(self,x): return x+x\ndef make():\n seen=Path("+repr(str(host))+").exists()\n try:\n  s=socket.create_connection(('127.0.0.1',9),timeout=.2);s.close();network=True\n except OSError: network=False\n Path('/out/isolation-probe.json').write_text(json.dumps(dict(host_file_seen=seen,network=network,environment=os.environ.get('POSEIDON_SYNTHETIC_PRIVATE_VALUE') is not None)))\n return Model()\n"
            (root/"probe.py").write_text(code)
            manifest=dict(format="poseidon-trusted-pytorch-v1",files={"probe.py":hashlib.sha256(code.encode()).hexdigest()},
                entry="probe.make",inputs=[dict(name="x",shape=[2])],outputs=["y"],approximations={})
            (root/"manifest.json").write_text(json.dumps(manifest))
            with patch.dict(os.environ,{"POSEIDON_SYNTHETIC_PRIVATE_VALUE":"synthetic-canary"}):
                result=trusted.run(root/"manifest.json",root/"captured",trusted=True)
            self.assertTrue(result["reference_verified"])
            observed=json.loads((root/"captured/isolation-probe.json").read_text())
            self.assertEqual(observed,dict(host_file_seen=False,network=False,environment=False))

    def test_source_inventory_refused_before_reading_code(self):
        spec=copy.deepcopy(self.value["trusted_pytorch"]["manifest"])
        spec["files"]={str(i)+".py":"0"*64 for i in range(9)}
        raw=json.dumps(spec).encode()
        with patch.object(trusted,"regular_bytes",return_value=raw) as reading:
            with self.assertRaisesRegex(ValueError,"Source inventory"):trusted.prepare("unused/manifest.json")
            self.assertEqual(reading.call_count,1)

    def test_source_total_budget_is_enforced_before_second_payload(self):
        spec=copy.deepcopy(self.value["trusted_pytorch"]["manifest"])
        spec["files"]={"first.py":hashlib.sha256(b" "*65530).hexdigest(),"second.py":"0"*64};seen=[]
        def reading(path,limit):
            seen.append((Path(path).name,limit))
            if Path(path).name=="manifest.json":return json.dumps(spec).encode()
            if Path(path).name=="first.py":return b" "*65530
            raise ValueError("Synthetic source exceeds remaining six-byte budget")
        with patch.object(trusted,"regular_bytes",side_effect=reading):
            with self.assertRaisesRegex(ValueError,"remaining six-byte"):trusted.prepare("unused/manifest.json")
        self.assertEqual(seen[-1],("second.py",6))

    def test_duplicate_and_nonfinite_manifest_refused_before_code(self):
        for raw in (b'{"format":0,"format":1}',b'{"bad":NaN}'):
            with patch.object(trusted,"regular_bytes",return_value=raw) as reading:
                with self.assertRaises(ValueError):trusted.prepare("unused/manifest.json")
                self.assertEqual(reading.call_count,1)

if __name__=="__main__":unittest.main()
