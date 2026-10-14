import base64,copy,hashlib,json,os,subprocess,sys,unittest
from pathlib import Path
import numpy as np
import pytorch_model_intake as intake
from benchmark_graph import digest,canonical
from pytorch_ordered_contract import (
 COMMON_SOURCE_FORMAT as FORMAT,COMMON_PACKAGE_FORMAT as PACKAGE_FORMAT,
 prepare,source_identity,capture_binding,check_reference,
 check_capture_metadata,encode_weight_files,weight_binding_from_payload)
from pytorch_weight_package import export_state,read,binding
from candidate_bundle import check_provenance,load_model_provenance,audit_provenance
from test_composition import PROFILE,EMPTY_APPROX,VALID_APPROX
from common_models import build
CACHE={}
MEASUREMENTS=[]
PACKAGE_CHECKS=[]

def captured(kind):
 if kind in CACHE:return CACHE[kind]
 fixture=Path("/out/source-fixtures")/kind;fixture.mkdir(parents=True)
 source=Path("/tests/common_models.py").read_bytes();(fixture/"common_models.py").write_bytes(source)
 module=build(kind)
 records=export_state(module,fixture/"weights",resource_profile=PROFILE)
 # Source packages use flat filenames. Original independent export stays retained.
 for name in {r["file"] for r in records.values()}:
  (fixture/name).write_bytes((fixture/"weights"/name).read_bytes())
 sdpa=kind in ("none","boolean","additive","causal")
 inputs=[dict(name=n,shape=[2,2]) for n in ("x","y","z")] if sdpa else [dict(name="x",shape=[2,3])]
 outputs=["attention","current","previous"] if sdpa else ["result","current","previous"]
 spec=dict(format=FORMAT,entry="common_models.build",factory=dict(args=[],kwargs=dict(kind=kind)),
  files={"common_models.py":hashlib.sha256(source).hexdigest()},inputs=inputs,outputs=outputs,weights=records,
  dtype_policy="strict-float64-v1",approximations=copy.deepcopy(VALID_APPROX if sdpa else EMPTY_APPROX))
 (fixture/"manifest.json").write_text(json.dumps(spec,sort_keys=True,indent=2)+"\n")
 prepared,_=intake.prepare(fixture/"manifest.json")
 if prepared!=spec:raise AssertionError("Facade manifest")
 arrays,raw=read(fixture,records,resource_profile=PROFILE);expected=binding(arrays,records)
 out=Path("/out")/("source-worker-"+kind);out.mkdir()
 request=out/"request.json"
 request.write_text(json.dumps(dict(manifest=spec,resource_profile=PROFILE,
  runtime=intake.legacy.runtime_identity(),source_hashes=source_identity())))
 env={k:v for k,v in os.environ.items() if k in (
  "LD_LIBRARY_PATH","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT","HOME","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","LANG")}
 env.update(PYTHONPATH="/app:"+str(fixture),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
 code="from pytorch_ordered_intake import capture_request;capture_request(*__import__('sys').argv[1:])"
 p=subprocess.run([sys.executable,"-B","-c",code,str(request),str(fixture),str(out)],
  env=env,text=True,capture_output=True,timeout=25)
 (out/"worker.log").write_text(p.stdout+p.stderr)
 if p.returncode:raise AssertionError("Source worker failed: "+p.stderr[-6000:])
 high=json.loads((out/"original.json").read_text());report=json.loads((out/"capture-reference.json").read_text())
 low,bound=intake.capture_binding(high,spec,resource_profile=PROFILE,
  capture_metadata=report["capture_metadata"],expected_weight_binding=expected)
 intake.check_reference(report,high,low,spec,resource_profile=PROFILE,expected_weight_binding=expected)
 provenance=dict(original=high,decomposition=bound,reference_report=report,
  trusted_pytorch=dict(format=PACKAGE_FORMAT,manifest=spec,files={"common_models.py":source.decode()},
   weight_files=encode_weight_files(spec,raw,resource_profile=PROFILE)))
 check_provenance(provenance,low);audit_provenance(provenance,low)
 MEASUREMENTS.append(dict(case=kind,source_package_sha256=digest(spec),groups=16,
  compared_values=report["finite_reference"]["compared_values"],reference=report["finite_reference"],
  source_hashes=report["source_hashes"],original_sha256=digest(high),model_sha256=digest(low),
  ordered_binding=bound["binding_sha256"],provenance_sha256=digest(provenance)))
 CACHE[kind]=(high,low,bound,report,spec,expected,provenance,fixture,out)
 return CACHE[kind]

class SourcePackageTests(unittest.TestCase):
 def test_all_common_rules_use_real_source_workers_and_independent_references(self):
  for kind in ("alpha","dropout","module_dropout","none","boolean","additive","causal"):
   with self.subTest(kind=kind):
    high,low,bound,report,spec,expected,provenance,fixture,out=captured(kind)
    self.assertEqual(bound["capture_policy_version"],FORMAT)
    self.assertEqual(report["format"],"poseidon-trusted-pytorch-reference-v13")
    self.assertEqual(report["finite_reference"]["groups"],16)
    self.assertFalse(report["encrypted_execution"])
    self.assertEqual(report["capture_metadata"]["weight_binding"],expected)
 def test_manifest_prepare_is_read_only_and_never_imports_source(self):
  from unittest.mock import patch
  *_,fixture,out=captured("alpha")
  with patch("importlib.import_module",side_effect=AssertionError("No source execution during prepare")):
   spec,paths=intake.prepare(fixture/"manifest.json")
  self.assertEqual(spec["format"],FORMAT);self.assertEqual(set(paths),set(spec["files"]))
 def test_alpha_boolean_causal_packages_reload_original_npy(self):
  for kind in ("alpha","boolean","causal"):
   with self.subTest(kind=kind):
    high,low,bound,report,spec,expected,value,fixture,out=captured(kind)
    folder=Path("/out")/("retained-package-"+kind);folder.mkdir()
    (folder/"trusted-python").mkdir()
    for name,data in (("original.json",high),("decomposition.json",bound),("reference-report.json",report)):
     (folder/name).write_text(json.dumps(data))
    (folder/"trusted-python/manifest.json").write_text(json.dumps(spec))
    (folder/"trusted-python/common_models.py").write_bytes((fixture/"common_models.py").read_bytes())
    for name,encoded in value["trusted_pytorch"]["weight_files"].items():
     (folder/"trusted-python"/name).write_bytes(base64.b64decode(encoded,validate=True))
    reloaded=load_model_provenance(folder,low,max_bytes=4*1024**2)
    self.assertEqual(canonical(reloaded),canonical(value))
    PACKAGE_CHECKS.append(dict(case=kind,provenance_sha256=digest(value),
      reloaded=True,encrypted_execution=False,agent_generated=False))
 def test_v13_cannot_be_claimed_as_v12(self):
  high,low,bound,report,spec,expected,*_=captured("alpha")
  changed=copy.deepcopy(spec);changed["format"]="poseidon-trusted-pytorch-v12"
  with self.assertRaisesRegex(ValueError,"common capture version"):
   check_capture_metadata(report["capture_metadata"],high,changed,
    resource_profile=PROFILE,expected_weight_binding=expected)
 def test_common_source_record_hash_rebinding_cannot_hide_source_mismatch(self):
  high,low,bound,report,spec,expected,*_=captured("alpha")
  metadata=copy.deepcopy(report["capture_metadata"]);alias=metadata["ordered_alias_capture"]
  adapter=alias["capture_adapter"];common=adapter["common_normalization"]
  common["source_fx_text"]="different";common["source_fx_sha256"]=hashlib.sha256(b"different").hexdigest()
  common.pop("binding_sha256");common["binding_sha256"]=digest(common)
  adapter.pop("binding");adapter["binding"]=digest(adapter)
  alias.pop("binding");alias["binding"]=digest(alias)
  with self.assertRaisesRegex(ValueError,"common source identity"):
   check_capture_metadata(metadata,high,spec,resource_profile=PROFILE,expected_weight_binding=expected)
 def test_source_weight_payload_change_rejected(self):
  high,low,bound,report,spec,expected,value,*_=captured("alpha")
  changed=copy.deepcopy(value);payload=changed["trusted_pytorch"]["weight_files"]
  key=next(iter(payload));raw=bytearray(base64.b64decode(payload[key]));raw[-1]^=1
  payload[key]=base64.b64encode(raw).decode()
  with self.assertRaisesRegex(ValueError,"State weight size/hash"):check_provenance(changed,low)
 def test_v13_package_has_explicit_weight_payload(self):
  high,low,bound,report,spec,expected,value,*_=captured("alpha")
  changed=copy.deepcopy(value);changed["trusted_pytorch"].pop("weight_files")
  with self.assertRaisesRegex(ValueError,"provenance package"):check_provenance(changed,low)
 def test_capability_registration_does_not_claim_encrypted_or_agent_pass(self):
  from application_capabilities import input_sources
  record=input_sources()["trusted_pytorch"]["ordered_common_calls"]
  self.assertEqual(record["version"],FORMAT)
  self.assertEqual(record["input_status"],"registered")
  self.assertEqual(record["compiler_status"],"unverified")
  self.assertEqual(record["encrypted_status"],"unverified")
  self.assertEqual(record["agent_status"],"unverified")


class OptionalSignatureRejectionTests(unittest.TestCase):
 def test_omitted_optional_forward_placeholders_are_explicitly_rejected(self):
  from test_composition import prepared
  with self.assertRaisesRegex(ValueError,"Input declaration count"):
   prepared(build("optional_signature"),[dict(name="x",shape=[2,3])],
    ["result","current","previous"],common_calls=True)
