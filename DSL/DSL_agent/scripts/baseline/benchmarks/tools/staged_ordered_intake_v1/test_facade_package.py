import base64,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmark_graph import canonical,digest
import pytorch_model_intake as intake
from pytorch_ordered_contract import (
 FORMAT,PACKAGE_FORMAT,encode_weight_files,weight_binding_from_payload)
from pytorch_weight_package import read
from candidate_bundle import check_provenance,load_model_provenance,audit_provenance
from test_intake import captured,PROFILE

PACKAGES=[]

def package(name):
 high,low,bound,report,spec,expected=captured(name)
 arrays,raw=read(Path("/fixtures")/name,spec["weights"],resource_profile=PROFILE)
 files={n:(Path("/fixtures")/name/n).read_text() for n in spec["files"]}
 result=dict(original=copy.deepcopy(high),decomposition=copy.deepcopy(bound),
             reference_report=copy.deepcopy(report),
             trusted_pytorch=dict(format=PACKAGE_FORMAT,manifest=copy.deepcopy(spec),files=files,
             weight_files=encode_weight_files(spec,raw,resource_profile=PROFILE)))
 return result,low,expected

class FacadePackageTests(unittest.TestCase):
 def test_new_version_uses_existing_facade_without_changing_legacy_recipe(self):
  spec,_=intake.prepare("/fixtures/linear/manifest.json")
  self.assertIn(FORMAT,intake.SOURCE_FORMATS)
  self.assertEqual(intake.package_format(spec),PACKAGE_FORMAT)
  self.assertEqual(intake.source_shape_policy(spec),"signature-static-ordered-alias-staging-v1")
  self.assertEqual(len(intake.source_identity()),19)
  for name,sha in intake.source_identity().items():
   self.assertEqual(sha,hashlib.sha256((Path("/app")/name).read_bytes()).hexdigest())

 def test_facade_prepare_does_not_import_model_or_invoke_factory(self):
  with patch("importlib.import_module",side_effect=AssertionError("No model import during prepare")):
   spec,paths=intake.prepare("/fixtures/linear/manifest.json")
  self.assertEqual(spec["format"],FORMAT)
  self.assertEqual(set(paths),set(spec["files"]))

 def test_old_versions_still_delegate_package_format_and_preparation(self):
  for v in range(1,8):
   spec=dict(format="poseidon-trusted-pytorch-v"+str(v))
   with patch.object(intake.legacy,"package_format",return_value="retained") as delegated:
    self.assertEqual(intake.package_format(spec),"retained")
    delegated.assert_called_once_with(spec)
  for version in (intake.FORMAT,intake.SHAPE_FORMAT,intake.OPERATOR_SHAPE_FORMAT,intake.COMMON_FORMAT):
   self.assertEqual(intake.shim(dict(format=version))["format"],intake.legacy.SOURCE_V7)

 def test_three_source_packages_reload_with_independent_retained_npy(self):
  for name in ("linear","norm","split_weight"):
   value,model,expected=package(name)
   checked=check_provenance(value,model)
   self.assertEqual(checked,value["decomposition"])
   self.assertEqual(weight_binding_from_payload(value["trusted_pytorch"]["manifest"],
            value["trusted_pytorch"]["weight_files"],resource_profile=PROFILE),expected)
   audit_provenance(value,model)
   folder=Path("/out")/("retained-import-"+name)
   folder.mkdir();(folder/"trusted-python").mkdir()
   for key,filename in (("original","original.json"),("decomposition","decomposition.json"),
                        ("reference_report","reference-report.json")):
    (folder/filename).write_text(json.dumps(value[key],allow_nan=False))
   spec=value["trusted_pytorch"]["manifest"]
   (folder/"trusted-python/manifest.json").write_text(json.dumps(spec))
   for filename,text in value["trusted_pytorch"]["files"].items():
    (folder/"trusted-python"/filename).write_text(text)
   for filename,encoded in value["trusted_pytorch"]["weight_files"].items():
    (folder/"trusted-python"/filename).write_bytes(base64.b64decode(encoded,validate=True))
   reloaded=load_model_provenance(folder,model,max_bytes=4*1024**2)
   self.assertEqual(canonical(reloaded),canonical(value))
   PACKAGES.append(dict(case=name,provenance_sha256=digest(value),
       provenance_bytes=len(canonical(value)),weights=len(spec["weights"]),
       reference_reloaded=True,encrypted_execution=False,agent_generated=False))

 def test_raw_weight_change_rejected_even_when_metadata_is_rebound(self):
  value,model,_=package("linear")
  key=next(iter(value["trusted_pytorch"]["weight_files"]))
  raw=bytearray(base64.b64decode(value["trusted_pytorch"]["weight_files"][key],validate=True))
  raw[-1]^=1
  value["trusted_pytorch"]["weight_files"][key]=base64.b64encode(raw).decode("ascii")
  with self.assertRaisesRegex(ValueError,"State weight size/hash"):
   check_provenance(value,model)

 def test_missing_extra_and_noncanonical_payload_rejected(self):
  value,model,_=package("linear")
  for mode in ("missing","extra","encoding"):
   changed=copy.deepcopy(value)
   payload=changed["trusted_pytorch"]["weight_files"]
   if mode=="missing":payload.pop(next(iter(payload)))
   elif mode=="extra":payload["unexpected.npy"]=""
   else:payload[next(iter(payload))]="!"*len(payload[next(iter(payload))])
   with self.assertRaises(ValueError):check_provenance(changed,model)

 def test_v12_package_cannot_hide_its_weight_files_as_v11_provenance(self):
  value,model,_=package("linear")
  value["trusted_pytorch"]["format"]="poseidon-trusted-pytorch-provenance-v11"
  with self.assertRaisesRegex(ValueError,"provenance package"):check_provenance(value,model)

 def test_capabilities_separate_registration_from_backend_acceptance(self):
  from application_capabilities import input_sources
  descriptions=input_sources()["trusted_pytorch"]
  record=descriptions["ordered_tensor_effects"]
  self.assertIn(FORMAT,descriptions["formats"])
  self.assertEqual(record["version"],FORMAT)
  self.assertEqual(record["input_status"],"registered")
  for name in ("compiler_status","encrypted_status","agent_status"):
   self.assertEqual(record[name],"unverified")

 def test_new_factory_loader_has_explicit_identity_authority(self):
  from application_identity import DYNAMIC,source_closure
  pair=("pytorch_ordered_intake.py","importlib.import_module(module_name)")
  self.assertIn(pair,DYNAMIC)
  closure=source_closure(Path("/app"),["pytorch_model_intake.py"])
  self.assertIn("pytorch_ordered_intake.py",closure["sources"])
  self.assertIn("pytorch_ordered_reference.py",closure["sources"])
  denied=dict(DYNAMIC);denied.pop(pair)
  with self.assertRaisesRegex(ValueError,"Unbounded dynamic dependency"):
   source_closure(Path("/app"),["pytorch_model_intake.py"],dynamic_authorities=denied)
