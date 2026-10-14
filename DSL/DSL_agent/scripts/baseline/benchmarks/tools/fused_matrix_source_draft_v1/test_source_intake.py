import base64,copy,hashlib,json,os,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
import pytorch_model_intake as intake
from benchmark_graph import digest,canonical
from pytorch_ordered_contract import FUSED_SOURCE_FORMAT as FORMAT,FUSED_PACKAGE_FORMAT as PACKAGE_FORMAT
from pytorch_ordered_contract import source_identity,encode_weight_files,check_capture_metadata
from pytorch_weight_package import export_state,read,binding
from pytorch_fused_matrix_normalization import check_transport
from candidate_bundle import check_provenance,audit_provenance,load_model_provenance
from source_models import build,dimensions
PROFILE="general-tensors-v1"
EMPTY=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[])
CACHE={};MEASUREMENTS=[];PACKAGES=[];SUBMISSIONS=[]

def captured(op,context):
    key=op+"_"+context
    if key in CACHE:return CACHE[key]
    style={"cipher":"function","public_right":"method","public_bias":"keywords"}[context]
    fixture=Path("/out/source-fixtures")/key;fixture.mkdir(parents=True)
    files={n:Path("/tests",n).read_text() for n in ("source_models.py","source_helpers.py")}
    for n,t in files.items():(fixture/n).write_text(t)
    module=build(op,context,style)
    records=export_state(module,fixture/"weights",resource_profile=PROFILE)
    for n in {r["file"] for r in records.values()}:(fixture/n).write_bytes((fixture/"weights"/n).read_bytes())
    inputs=[dict(name=n,shape=s) for n,s in zip(("bias","left","right"),dimensions(op))]
    spec=dict(format=FORMAT,entry="source_models.build",factory=dict(args=[],kwargs=dict(op=op,context=context,style=style)),
              files={n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()},
              inputs=inputs,outputs=["result","current","previous"],weights=records,
              dtype_policy="strict-float64-v1",approximations=copy.deepcopy(EMPTY))
    (fixture/"manifest.json").write_text(json.dumps(spec,indent=2,sort_keys=True))
    actual,_=intake.prepare(fixture/"manifest.json")
    if actual!=spec:raise AssertionError("Facade source manifest")
    arrays,raw=read(fixture,records,resource_profile=PROFILE);expected=binding(arrays,records)
    out=Path("/out")/("source-worker-"+key);out.mkdir()
    request=out/"request.json";request.write_text(json.dumps(dict(manifest=spec,resource_profile=PROFILE,
                    runtime=intake.legacy.runtime_identity(),source_hashes=source_identity())))
    env={k:v for k,v in os.environ.items() if k in ("LD_LIBRARY_PATH","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT",
         "HOME","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","LANG")}
    env.update(PYTHONPATH="/app:"+str(fixture),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    code="from pytorch_ordered_intake import capture_request;capture_request(*__import__('sys').argv[1:])"
    proc=subprocess.run([sys.executable,"-B","-c",code,str(request),str(fixture),str(out)],env=env,
                        text=True,capture_output=True,timeout=25)
    (out/"worker.log").write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise AssertionError("Source worker failed: "+proc.stderr[-9000:])
    high=json.loads((out/"original.json").read_text());report=json.loads((out/"capture-reference.json").read_text())
    low,bound=intake.capture_binding(high,spec,resource_profile=PROFILE,capture_metadata=report["capture_metadata"],
                                    expected_weight_binding=expected)
    intake.check_reference(report,high,low,spec,resource_profile=PROFILE,expected_weight_binding=expected)
    value=dict(original=high,decomposition=bound,reference_report=report,
               trusted_pytorch=dict(format=PACKAGE_FORMAT,manifest=spec,files=files,
                  weight_files=encode_weight_files(spec,raw,resource_profile=PROFILE)))
    check_provenance(value,low);audit_provenance(value,low)
    MEASUREMENTS.append(dict(case=key,operation=op,context=context,style=style,
        source_package_sha256=digest(spec),provenance_sha256=digest(value),reference=report["finite_reference"],
        original_sha256=digest(high),model_sha256=digest(low),capture_binding_sha256=bound["binding_sha256"],
        source_files=len(files),actual_source_worker=True,compiler=False,encrypted_execution=False))
    CACHE[key]=(high,low,bound,report,spec,expected,value,fixture,out)
    return CACHE[key]
def rebind(metadata):
    alias=metadata["ordered_alias_capture"];adapter=alias["capture_adapter"]
    fused=adapter["fused_matrix_normalization"]
    fused["binding_sha256"]=digest({k:v for k,v in fused.items() if k!="binding_sha256"})
    adapter["binding"]=digest({k:v for k,v in adapter.items() if k!="binding"})
    alias["binding"]=digest({k:v for k,v in alias.items() if k!="binding"})
    return metadata
class SourceTests(unittest.TestCase):
    def test_addmm_three_source_contexts(self):self.group("addmm")
    def test_baddbmm_three_source_contexts(self):self.group("baddbmm")
    def test_addbmm_three_source_contexts(self):self.group("addbmm")
    def group(self,op):
        for context in ("cipher","public_right","public_bias"):
            with self.subTest(context=context):
                high,low,bound,report,spec,*_=captured(op,context)
                self.assertEqual(bound["capture_policy_version"],FORMAT)
                self.assertEqual(report["format"],"poseidon-trusted-pytorch-reference-v14")
                self.assertTrue(report["finite_reference"]["input_storage_isolated"])
                self.assertFalse(report["encrypted_execution"])
    def test_three_two_file_packages_reload(self):
        for op in ("addmm","baddbmm","addbmm"):
            with self.subTest(op=op):
                high,low,bound,report,spec,expected,value,fixture,out=captured(op,"public_right")
                folder=Path("/out")/("reloaded-"+op);folder.mkdir();(folder/"trusted-python").mkdir()
                for key,name in (("original","original.json"),("decomposition","decomposition.json"),
                                 ("reference_report","reference-report.json")):
                    (folder/name).write_text(json.dumps(value[key],allow_nan=False))
                (folder/"trusted-python/manifest.json").write_text(json.dumps(spec))
                for name,text in value["trusted_pytorch"]["files"].items():(folder/"trusted-python"/name).write_text(text)
                for name,encoded in value["trusted_pytorch"]["weight_files"].items():
                    (folder/"trusted-python"/name).write_bytes(base64.b64decode(encoded,validate=True))
                actual=load_model_provenance(folder,low,max_bytes=4*1024**2)
                self.assertEqual(canonical(actual),canonical(value))
                PACKAGES.append(dict(case=op,source_files=2,weights=len(spec["weights"]),provenance_sha256=digest(value),
                                     reloaded=True,compiler=False,encrypted_execution=False))
    def test_prepare_never_imports_source_or_executes_factory(self):
        *_,fixture,out=captured("addmm","cipher")
        with patch("importlib.import_module",side_effect=AssertionError("prepare cannot execute source")):
            spec,paths=intake.prepare(fixture/"manifest.json")
        self.assertEqual(spec["format"],FORMAT);self.assertEqual(set(paths),set(spec["files"]))
    def test_record_removal_cannot_hide_as_v13(self):
        high,low,bound,report,spec,expected,*_=captured("addmm","cipher")
        changed=copy.deepcopy(spec);changed["format"]="poseidon-trusted-pytorch-v13"
        with self.assertRaisesRegex(ValueError,"fused matrix capture version"):
            check_capture_metadata(report["capture_metadata"],high,changed,resource_profile=PROFILE,expected_weight_binding=expected)
    def test_normalized_inventory_reordering_rejected(self):
        high,low,bound,report,spec,expected,*_=captured("addmm","cipher")
        metadata=copy.deepcopy(report["capture_metadata"]);record=metadata["ordered_alias_capture"]["capture_adapter"]["fused_matrix_normalization"]
        for n in record["normalized_inventory"]:
            if n["target"]=="torch.mm":
                n["args"]["items"].reverse();break
        rebind(metadata)
        with self.assertRaisesRegex(ValueError,"rewrite differs"):
            check_capture_metadata(metadata,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)
    def test_alpha_beta_rebinding_without_source_change_rejected(self):
        high,low,bound,report,spec,expected,*_=captured("baddbmm","public_right")
        for name in ("alpha","beta"):
            metadata=copy.deepcopy(report["capture_metadata"])
            metadata["ordered_alias_capture"]["capture_adapter"]["fused_matrix_normalization"]["entries"][0]["plan"][name]=.875
            rebind(metadata)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"transport plan"):
                check_capture_metadata(metadata,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)
    def test_omitted_extra_and_wrong_product_steps_rejected(self):
        high,low,bound,report,spec,expected,*_=captured("addbmm","cipher")
        for mode in ("omit","extra","product"):
            metadata=copy.deepcopy(report["capture_metadata"])
            rec=metadata["ordered_alias_capture"]["capture_adapter"]["fused_matrix_normalization"]
            if mode=="omit":rec["entries"].clear()
            elif mode=="extra":rec["entries"][0]["emitted_nodes"].append("unrelated_step")
            else:
                for n in rec["normalized_inventory"]:
                    if n["target"]=="torch.bmm":n["target"]="torch.mm";break
            rebind(metadata)
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                check_capture_metadata(metadata,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)
    def test_original_signature_and_output_metadata_rejected(self):
        high,low,bound,report,spec,expected,*_=captured("addmm","cipher")
        for mode in ("signature","shape","dangling"):
            metadata=copy.deepcopy(report["capture_metadata"])
            rec=metadata["ordered_alias_capture"]["capture_adapter"]["fused_matrix_normalization"]
            if mode=="signature":rec["input_signature"][0]["shape"]=[1]
            elif mode=="shape":
                for n in rec["source_inventory"]:
                    if n["target"]=="torch.addmm":n["tensor"]["shape"]=[1];break
            else:
                for n in rec["source_inventory"]:
                    if n["target"]=="torch.addmm":n["args"]["items"][1]["name"]="future_operand";break
            rebind(metadata)
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                check_capture_metadata(metadata,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)
    def test_original_source_and_common_chain_remain_bound(self):
        high,low,bound,report,spec,expected,*_=captured("addmm","cipher")
        metadata=copy.deepcopy(report["capture_metadata"]);adapter=metadata["ordered_alias_capture"]["capture_adapter"]
        rec=adapter["fused_matrix_normalization"];rec["source_fx_text"]="changed"
        rec["source_fx_sha256"]=hashlib.sha256(b"changed").hexdigest();rebind(metadata)
        with self.assertRaisesRegex(ValueError,"original source identity"):
            check_capture_metadata(metadata,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)
    def test_loaded_weight_content_changes_record_rejected(self):
        high,low,bound,report,spec,expected,value,*_=captured("addmm","public_right")
        changed=copy.deepcopy(value);payload=changed["trusted_pytorch"]["weight_files"]
        key=next(iter(payload));raw=bytearray(base64.b64decode(payload[key]));raw[-1]^=1
        payload[key]=base64.b64encode(raw).decode()
        with self.assertRaisesRegex(ValueError,"State weight size/hash"):check_provenance(changed,low)
        metadata=copy.deepcopy(report["capture_metadata"])
        rec=metadata["ordered_alias_capture"]["capture_adapter"]["fused_matrix_normalization"]
        rec["public_state"]["weight"]["normalized_float64_sha256"]="0"*64;rebind(metadata)
        with self.assertRaisesRegex(ValueError,"source state differs"):
            check_capture_metadata(metadata,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)
    def test_missing_file_and_reserved_implementation_name_rejected(self):
        *_,value,fixture,out=captured("addmm","cipher");spec=value["trusted_pytorch"]["manifest"];files=value["trusted_pytorch"]["files"]
        changed=copy.deepcopy(files);changed.pop("source_helpers.py")
        with self.assertRaisesRegex(ValueError,"Source inventory"):intake.validate_package(spec,changed)
        changed=copy.deepcopy(spec);text="pass\n"
        changed["files"]["pytorch_fused_matrix_normalization.py"]=hashlib.sha256(text.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError,"conflicts"):
            intake.validate_package(changed,{**files,"pytorch_fused_matrix_normalization.py":text})
    def test_capability_reports_registration_separately(self):
        from application_capabilities import input_sources
        value=input_sources()["trusted_pytorch"]
        self.assertIn(FORMAT,value["formats"])
        record=value["ordered_fused_matrix_calls"]
        self.assertEqual(record["calls"],["addmm","baddbmm","addbmm"])
        self.assertEqual(record["input_status"],"registered")
        for name in ("compiler_status","encrypted_status","agent_status"):self.assertEqual(record[name],"unverified")
    def test_legacy_format_and_request_choices_preserved(self):
        self.assertEqual(intake.ordered(dict(format="poseidon-trusted-pytorch-v12")),True)
        self.assertEqual(intake.ordered(dict(format="poseidon-trusted-pytorch-v13")),True)
        self.assertEqual(intake.package_format(captured("addmm","cipher")[4]),PACKAGE_FORMAT)
        self.assertEqual(intake.shim(dict(format=FORMAT))["format"],intake.legacy.SOURCE_V7)
class ApplicationTests(unittest.TestCase):
    def test_three_submissions_snapshot_multifile_source_without_generation(self):
        from application_component import Application
        from application_model_source import FORMAT as REQUEST_FORMAT
        from test_application_component import Backend
        for op in ("addmm","baddbmm","addbmm"):
            with self.subTest(op=op):
                *_,fixture,out=captured(op,"public_right")
                app=Application(Path("/out")/("application-"+op),Backend())
                source=dict(format=REQUEST_FORMAT,kind="trusted_pytorch",manifest=str(fixture/"manifest.json"),trust_local_code=True)
                with patch("importlib.import_module",side_effect=AssertionError("submit cannot execute factory")):
                    task=app.submit_source(source,resource_profile=PROFILE)
                status=app.status(task)
                self.assertEqual((status["status"],status["phase"]),("queued","intake_queued"))
                self.assertEqual((status["generations"],status["http_attempts"]),(0,0))
                self.assertFalse(status["intake"]["reference_verified"])
                snapshot=app._dir(task)/"source"
                for name in ("source_models.py","source_helpers.py"):
                    self.assertEqual((snapshot/name).read_bytes(),(fixture/name).read_bytes())
                self.assertEqual(json.loads((snapshot/"manifest.json").read_text())["format"],FORMAT)
                app.cancel(task);self.assertEqual(app.status(task)["status"],"cancelled")
                SUBMISSIONS.append(dict(operation=op,source_snapshot=True,source_factory_calls=0,provider_calls=0,
                                       backend="injected identity protocol only",cancelled=True,validation_level=status["validation_level"]))


LEGACY_MEASUREMENTS=[]
class RetainedSourceTests(unittest.TestCase):
    def test_actual_v12_and_v13_source_workers_preserve_interfaces(self):
        for version,fixture in (("v12",Path("/compat-v12")),("v13",Path("/compat-v13"))):
            with self.subTest(version=version):
                spec=json.loads((fixture/"manifest.json").read_text())
                self.assertEqual(spec["format"],"poseidon-trusted-pytorch-"+version)
                arrays,_=read(fixture,spec["weights"],resource_profile=PROFILE)
                expected=binding(arrays,spec["weights"])
                out=Path("/out")/("legacy-source-"+version);out.mkdir()
                req=out/"request.json";req.write_text(json.dumps(dict(manifest=spec,resource_profile=PROFILE,
                    runtime=intake.legacy.runtime_identity(),source_hashes=source_identity())))
                env={k:v for k,v in os.environ.items() if k in ("LD_LIBRARY_PATH","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT",
                    "HOME","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","LANG")}
                env.update(PYTHONPATH="/app:"+str(fixture),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
                code="from pytorch_ordered_intake import capture_request;capture_request(*__import__('sys').argv[1:])"
                proc=subprocess.run([sys.executable,"-B","-c",code,str(req),str(fixture),str(out)],env=env,
                                    text=True,capture_output=True,timeout=25)
                (out/"worker.log").write_text(proc.stdout+proc.stderr)
                self.assertEqual(proc.returncode,0,proc.stderr[-7000:])
                high=json.loads((out/"original.json").read_text());report=json.loads((out/"capture-reference.json").read_text())
                low,bound=intake.capture_binding(high,spec,resource_profile=PROFILE,
                    capture_metadata=report["capture_metadata"],expected_weight_binding=expected)
                intake.check_reference(report,high,low,spec,resource_profile=PROFILE,expected_weight_binding=expected)
                self.assertEqual(bound["capture_policy_version"],spec["format"])
                self.assertNotIn("fused_matrix_normalization",report["capture_metadata"]["ordered_alias_capture"]["capture_adapter"])
                LEGACY_MEASUREMENTS.append(dict(version=version,model_sha256=digest(low),reference=report["finite_reference"],
                                                actual_source_worker=True,compiler=False,encrypted_execution=False))
    def test_old_source_workers_do_not_silently_enable_fused_calls(self):
        *_,spec,expected,value,fixture,out=captured("addmm","cipher")
        for version in ("v12","v13"):
            with self.subTest(version=version):
                changed=copy.deepcopy(spec);changed["format"]="poseidon-trusted-pytorch-"+version
                failed=Path("/out")/("old-fused-rejection-"+version);failed.mkdir()
                req=failed/"request.json";req.write_text(json.dumps(dict(manifest=changed,resource_profile=PROFILE,
                    runtime=intake.legacy.runtime_identity(),source_hashes=source_identity())))
                env={k:v for k,v in os.environ.items() if k in ("LD_LIBRARY_PATH","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT",
                    "HOME","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","LANG")}
                env.update(PYTHONPATH="/app:"+str(fixture),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
                code="from pytorch_ordered_intake import capture_request;capture_request(*__import__('sys').argv[1:])"
                proc=subprocess.run([sys.executable,"-B","-c",code,str(req),str(fixture),str(failed)],env=env,
                                    text=True,capture_output=True,timeout=25)
                (failed/"worker.log").write_text(proc.stdout+proc.stderr)
                self.assertNotEqual(proc.returncode,0)
                self.assertIn("Unsupported PyTorch call:",proc.stderr)
                self.assertIn("method addmm",proc.stderr)
                self.assertFalse((failed/"capture-reference.json").exists())
