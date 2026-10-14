"""Real isolated FX input checks; mock application tests are counted separately."""
import copy,json,hashlib,tempfile,unittest
from pathlib import Path
import test_application_component as application_fixtures
from hecate_python_env import ROOT,WORK
from benchmark_graph import digest
from application_component import Application
from candidate_bundle import load_model_provenance,check_provenance
from trusted_pytorch_import import run,check_reference,capture_binding
from unified_graph_lowering import candidate_source

SOURCE=ROOT/"scripts/baseline/cases/trusted-pytorch-general-v2"

class GeneralImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK/"results")
        cls.imports={}
        for case in ("mlp","conv_pool","residual","batchnorm","matrix_fusion","shape_linear"):
            folder=Path(cls.temp.name)/case
            result=run(SOURCE/(case+".json"),folder,resource_profile="general-tensors-v1",trusted=True)
            model=json.loads((folder/"model.json").read_text())
            cls.imports[case]=(folder,model,load_model_provenance(folder,model),result)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_six_real_captures_and_references_not_fhe_claims(self):
        for _,model,value,result in self.imports.values():
            check_provenance(value,model)
            report=value["reference_report"]
            self.assertEqual(report["probes"],16)
            self.assertTrue(result["reference_verified"])
            self.assertFalse(result["encrypted_execution"])
            self.assertEqual(report["reference_dtype"],"float64")
            self.assertFalse(report["source_precision_execution_compared"])
            self.assertLessEqual(report["max_reference_tolerance_ratio"],1)
    def test_source_factory_policy_and_tree_are_bound(self):
        _,model,value,_=self.imports["matrix_fusion"]
        metadata=value["reference_report"]["capture_metadata"]
        self.assertEqual(metadata["output_structure"]["kind"],"dict")
        self.assertEqual(metadata["output_structure"]["items"][1]["value"]["kind"],"list")
        bad=copy.deepcopy(value);bad["trusted_pytorch"]["manifest"]["factory"]["kwargs"]["scale"]=2
        with self.assertRaises(ValueError):check_provenance(bad,model)
        bad=copy.deepcopy(value);bad["reference_report"]["capture_metadata"]["dtype_policy"]="convert-float64-v1"
        with self.assertRaises(ValueError):check_provenance(bad,model)
    def test_structure_order_and_dtype_record_tampering_rejected(self):
        _,model,value,_=self.imports["batchnorm"]
        for mutate in (
            lambda v:v["reference_report"]["capture_metadata"]["output_structure"]["items"].reverse(),
            lambda v:v["reference_report"]["capture_metadata"]["constant_dtypes"].update(unknown="float64"),
            lambda v:v["reference_report"].update(reference_dtype="float32"),
            lambda v:v["reference_report"].update(source_precision_execution_compared=True)):
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(ValueError):check_provenance(bad,model)
    def test_provenance_is_checked_without_importing_user_modules(self):
        from unittest.mock import patch
        _,model,value,_=self.imports["mlp"]
        with patch("importlib.import_module",side_effect=AssertionError("Source must stay inert")):
            check_provenance(value,model)
    def test_application_import_and_export_keeps_provenance_private(self):
        folder,model,value,_=self.imports["mlp"]
        class Backend(application_fixtures.Backend):
            def export(self,result,destination,level,*,model_provenance=None):
                self.received=model_provenance
                return super().export(result,destination,level)
        class Provider:
            kind="scripted_replay";calls=0;agent_calls=0
            def generate(self,request,history):
                self.calls+=1
                if "model_provenance" in request or "trusted_pytorch" in request:
                    raise AssertionError("Private provenance in generation request")
                return json.dumps(dict(schema=1,request_id=request["request_id"],
                    hecate_source=candidate_source(request)[0]))
        backend=Backend()
        with tempfile.TemporaryDirectory(dir=WORK/"results") as store:
            app=Application(store,backend)
            task=app.submit_import(folder,options=dict(chunk_period=4,generation_guidance="explicit-v10"))
            provider=Provider();result=app.run(task,provider)
            self.assertEqual(result["status"],"succeeded")
            self.assertEqual(backend.received,value)
            self.assertEqual(provider.calls,1)
            unused=Provider()
            task2=app.submit_import(folder,options=dict(chunk_period=4,generation_guidance="explicit-v10"))
            self.assertTrue(app.run(task2,unused)["package"]["reused"])
            self.assertEqual(unused.calls,0)
    def test_provenance_changes_reuse_identity_without_changing_generation_request(self):
        folder,model,value,_=self.imports["mlp"]
        changed=copy.deepcopy(value);package=changed["trusted_pytorch"]
        package["files"]["models.py"]+="\n# equivalent source snapshot\n"
        package["manifest"]["files"]["models.py"]=hashlib.sha256(package["files"]["models.py"].encode()).hexdigest()
        changed["reference_report"]["manifest_sha256"]=digest(package["manifest"])
        _,changed["decomposition"]=capture_binding(changed["original"],package["manifest"],
            resource_profile="general-tensors-v1",capture_metadata=changed["reference_report"]["capture_metadata"])
        check_provenance(changed,model)
        with tempfile.TemporaryDirectory(dir=WORK/"results") as store:
            app=Application(store,application_fixtures.Backend())
            a=app.submit(model,model_provenance=value,options=dict(chunk_period=4))
            b=app.submit(model,model_provenance=changed,options=dict(chunk_period=4))
            first=app._load(app._dir(a)/"task.json");second=app._load(app._dir(b)/"task.json")
            self.assertEqual(first["request"]["request_id"],second["request"]["request_id"])
            self.assertNotEqual(app._load(app._dir(a)/"state.json")["reuse_key"],
                                app._load(app._dir(b)/"state.json")["reuse_key"])
    def test_local_output_tree_is_a_hash_bound_snapshot(self):
        from benchmark_graph import canonical
        from program_execution import package_output_structure
        _,model,value,_=self.imports["matrix_fusion"]
        with tempfile.TemporaryDirectory(dir=WORK/"results") as root:
            path=Path(root)/"model-provenance.json";raw=canonical(value);path.write_bytes(raw)
            manifest=dict(model_provenance="included",files={"model-provenance.json":hashlib.sha256(raw).hexdigest()})
            tree=package_output_structure(root,manifest,model)
            path.write_text("{}")
            self.assertEqual(tree,value["reference_report"]["capture_metadata"]["output_structure"])
            with self.assertRaises(ValueError):package_output_structure(root,manifest,model)
    def test_source_capabilities_are_shared_and_do_not_claim_execution(self):
        from application_capabilities import input_sources,describe
        from component_contract import capabilities
        self.assertEqual(capabilities()["input_sources"],input_sources())
        self.assertEqual(describe()["input_sources"],input_sources())
        entry=input_sources()["trusted_pytorch"]
        self.assertIn("poseidon-trusted-pytorch-v2",entry["formats"])
        self.assertEqual(entry["output_tensor_leaves"],4)
        self.assertNotIn("encrypted_execution",entry)

    def test_crlf_sources_keep_exact_bytes_through_capture_and_package(self):
        with tempfile.TemporaryDirectory(dir=WORK/"results") as root:
            root=Path(root)
            raw=(SOURCE/"models.py").read_bytes().replace(b"\n",b"\r\n")
            (root/"models.py").write_bytes(raw)
            spec=json.loads((SOURCE/"mlp.json").read_text())
            spec["files"]["models.py"]=hashlib.sha256(raw).hexdigest()
            (root/"manifest.json").write_text(json.dumps(spec))
            folder=root/"captured"
            run(root/"manifest.json",folder,resource_profile="general-tensors-v1",trusted=True)
            model=json.loads((folder/"model.json").read_text())
            value=load_model_provenance(folder,model)
            self.assertEqual(value["trusted_pytorch"]["files"]["models.py"].encode(),raw)
            self.assertEqual((folder/"trusted-python/models.py").read_bytes(),raw)

    def test_invalid_import_is_terminal_without_provider(self):
        with tempfile.TemporaryDirectory(dir=WORK/"results") as store:
            app=Application(store,application_fixtures.Backend())
            task=app.submit_import(Path(store)/"missing")
            provider=application_fixtures.Provider()
            self.assertEqual(app.run(task,provider)["status"],"unsupported")
            self.assertEqual(provider.calls,0)

if __name__=="__main__":unittest.main()
