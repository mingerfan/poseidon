"""One pinned Nix session: three source contexts and authenticated compatibility checks."""
import io,json,os,pathlib,shlex,sys,time,hashlib,subprocess,shutil,unittest,copy,importlib.util
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";sys.path.insert(0,str(BASE))
os.environ.update(POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work")
OLD=HERE.parent/"public_tensor_source_v2"
OUT=pathlib.Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-boolean-source-draft-v1")
PRIOR_SECONDS=2.13408050299995+1.3285648869999704+137.50568162600007
REMAINING_SECONDS=339
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def measure(path):
 return max(int(subprocess.check_output(["du",arg,"-s","-B1",str(path)],text=True,timeout=30).split()[0])
  for arg in("--apparent-size","--count-links"))
def unchanged():
 before=json.loads((HERE/"before.json").read_text())
 assert all(sha(ROOT/"scripts/baseline"/p)==h for p,h in before["production_python"].items())
 assert all(sha(ROOT/p)==h for p,h in before["previous_source_draft"].items())
 assert all(sha(ROOT/p)==h for p,h in before["previous_v2_python"].items())
 assert sha(pathlib.Path(before["previous_v2_capture_report"]["path"]))==before["previous_v2_capture_report"]["sha256"]
def inside():
 from hecate_python_env import VENV
 from benchmark_graph import digest
 from pytorch_weight_package import export_state,read,binding
 from sandbox import run,draft_identity
 from capture_adapter import check_record
 from pytorch_ordered_contract import source_identity
 import test_boolean_arithmetic
 assert os.environ.get("IN_NIX_SHELL")=="pure"and pathlib.Path(sys.prefix)==VENV
 unchanged();assert not OUT.exists();OUT.mkdir(mode=0o700);started=time.monotonic()
 assert PRIOR_SECONDS+REMAINING_SECONDS<480
 frozen=source_identity();draft=draft_identity()
 def boundary():
  assert time.monotonic()-started<REMAINING_SECONDS
  assert measure(OUT)+measure(OUT.parent/"agent-public-tensor-source-draft-v2")<=128*1024**2
  assert measure(OUT.parent)<=32*1024**3 and shutil.disk_usage(OUT).free>=4*1024**3
  assert source_identity()==frozen and draft_identity()==draft
  unchanged()
 boundary()
 fixtures=HERE/"fixtures";assert not fixtures.exists();fixtures.mkdir()
 case=fixtures/"method_channel_mask";case.mkdir()
 definition=HERE/"fixture_model.py";spec=importlib.util.spec_from_file_location("fixed_mask_fixture",definition)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 weights=export_state(module.make(),case,resource_profile="general-tensors-v1")
 (case/"model.py").write_bytes(definition.read_bytes())
 manifest=dict(format="poseidon-trusted-pytorch-v21",entry="model.make",factory=dict(args=[],kwargs={}),
  files={"model.py":sha(case/"model.py")},inputs=[dict(name="x",shape=[2,3])],outputs=["masked","residual"],
  dtype_policy="strict-float64-v1",weights=weights,
  approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]))
 manifest_path=case/"manifest.json";manifest_path.write_text(json.dumps(manifest,allow_nan=False,sort_keys=True,indent=2))
 prior_cases=json.loads((OLD/"cases.json").read_text())
 cases=[c for c in prior_cases if c["case"]in("bool_multiply","mask_index_residual")]
 cases.append(dict(case="method_channel_mask",manifest=str(manifest_path.relative_to(ROOT)),manifest_sha256=sha(manifest_path)))
 assert len(cases)==3
 with(HERE/"cases.json").open("x")as f:json.dump(cases,f,allow_nan=False,sort_keys=True,indent=2)
 plan=dict(format="poseidon-public-boolean-source-plan-v1",source_hashes=frozen,draft_hashes=draft,
  retained_cases=cases[:2],new_context=cases[2],prior_elapsed_seconds=PRIOR_SECONDS,remaining_seconds=REMAINING_SECONDS,
  cumulative_wall_ceiling=480,total_source_artifact_bytes=128*1024**2,global_result_bytes=32*1024**3,
  reserve_bytes=4*1024**3,provider_calls=0,compiler_calls=0,encrypted_calls=0,production_activation=False)
 plan["binding"]=digest(plan);(OUT/"plan.json").write_text(json.dumps(plan,allow_nan=False,sort_keys=True,indent=2))
 rows=[]
 for c in cases:
  boundary();assert sha(ROOT/c["manifest"])==c["manifest_sha256"];row=dict(case=c["case"],status="running")
  try:
   receipt=run(ROOT/c["manifest"],OUT/c["case"],trusted=True)
   row.update(status="captured",receipt_binding=receipt["binding"],seconds=receipt["seconds"])
  except Exception as e:
   row.update(status="failed",error_type=type(e).__name__,message=str(e)[:140])
  rows.append(row);print(json.dumps(row),flush=True)
  boundary()
 def loaded(c):
  manifest_path=ROOT/c["manifest"];s=json.loads(manifest_path.read_text())
  r=json.loads((OUT/c["case"]/"capture.json").read_text());receipt=json.loads((OUT/c["case"]/"receipt.json").read_text())
  assert receipt["binding"]==digest({k:v for k,v in receipt.items()if k!="binding"})
  a,_=read(manifest_path.parent,s["weights"],resource_profile="general-tensors-v1")
  b,_=read(OUT/c["case"]/"internal-weight-snapshot",r["internal_manifest"]["weights"],resource_profile="general-tensors-v1")
  return r,s,a,b,receipt["capture_binding"]
 class ActualSourceTests(unittest.TestCase):
  def test_three_authenticated_captures(self):
   for c in cases:
    with self.subTest(case=c["case"]):
     r,s,a,b,authority=loaded(c);check_record(r,s,a,b,expected_record_binding=authority)
     self.assertEqual(r["original_reference"]["groups"],16)
     self.assertTrue(r["original_reference"]["public_state_immutable"])
     self.assertTrue(r["boolean_mask_capture"]["entries"])
  def test_raw_bool_payload_replacement_is_rejected(self):
   for c in cases:
    with self.subTest(case=c["case"]):
     r,s,a,b,authority=loaded(c);a={k:v.copy()for k,v in a.items()};a["mask"].reshape(-1)[0]=not a["mask"].reshape(-1)[0]
     with self.assertRaisesRegex(ValueError,"Original source public weights"):check_record(r,s,a,b,expected_record_binding=authority)
  def test_raw_bool_dtype_replacement_is_rejected(self):
   r,s,a,b,authority=loaded(cases[0]);a=dict(a);a["mask"]=a["mask"].astype("<f8")
   with self.assertRaises(ValueError):check_record(r,s,a,b,expected_record_binding=authority)
  def test_derived_mask_value_replacement_is_rejected(self):
   for c in cases:
    with self.subTest(case=c["case"]):
     r,s,a,b,authority=loaded(c);b={k:v.copy()for k,v in b.items()}
     name=r["boolean_mask_capture"]["entries"][0]["derived_state_name"];b[name].reshape(-1)[0]=2.
     with self.assertRaisesRegex(ValueError,"Derived public weights"):check_record(r,s,a,b,expected_record_binding=authority)
  def test_mask_implementation_binding_is_rejected(self):
   r,s,a,b,_=loaded(cases[0]);r["boolean_mask_capture"]["implementation_sha256"]="0"*64
   r["boolean_mask_capture"]["binding"]=digest({k:v for k,v in r["boolean_mask_capture"].items()if k!="binding"})
   r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
   with self.assertRaisesRegex(ValueError,"trusted identity"):check_record(r,s,a,b,expected_record_binding=r["binding"])
  def test_original_index_dtype_remains_int64(self):
   r,s,a,b,authority=loaded(cases[1]);check_record(r,s,a,b,expected_record_binding=authority)
   self.assertEqual(a["index"].dtype.str,"<i8");self.assertEqual(b["index"].dtype.str,"<i8")
  def test_sixteen_previous_captures_are_data_revalidated(self):
   p=OLD/"capture_adapter.py";spec=importlib.util.spec_from_file_location("authenticated_legacy_v2_checker",p)
   previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
   old_out=OUT.parent/"agent-public-tensor-source-draft-v2";report=json.loads((old_out/"report.json").read_text())
   selected=[c for c in prior_cases if next(r for r in report["rows"]if r["case"]==c["case"])["status"]=="captured"]
   self.assertEqual(len(selected),16)
   for c in selected:
    with self.subTest(case=c["case"]):
     manifest_path=ROOT/c["manifest"];s=json.loads(manifest_path.read_text())
     r=json.loads((old_out/c["case"]/"capture.json").read_text())
     receipt=json.loads((old_out/c["case"]/"receipt.json").read_text())
     self.assertEqual(receipt["binding"],digest({k:v for k,v in receipt.items()if k!="binding"}))
     self.assertEqual(receipt["draft_hashes"],report["draft_hashes"])
     a,_=read(manifest_path.parent,s["weights"],resource_profile="general-tensors-v1")
     b,_=read(old_out/c["case"]/"internal-weight-snapshot",r["internal_manifest"]["weights"],resource_profile="general-tensors-v1")
     previous.check_record(r,s,a,b,expected_record_binding=receipt["capture_binding"])
 boundary();stream=io.StringIO();suite=unittest.TestSuite([
  unittest.defaultTestLoader.loadTestsFromModule(test_boolean_arithmetic),
  unittest.defaultTestLoader.loadTestsFromTestCase(ActualSourceTests)])
 controls=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);(OUT/"controls.log").write_text(stream.getvalue())
 boundary()
 report=dict(format="poseidon-public-boolean-source-report-v1",plan_binding=plan["binding"],rows=rows,
  status="passed"if all(r["status"]=="captured"for r in rows)and controls.wasSuccessful()and not controls.skipped else"failed",
  source_captures=len(rows),source_capture_passed=sum(r["status"]=="captured"for r in rows),
  previous_captures_data_revalidated=16 if not controls.errors and not controls.failures else None,
  tests_run=controls.testsRun,passed=controls.testsRun-len(controls.failures)-len(controls.errors)-len(controls.skipped),
  failed=len(controls.failures),errors=len(controls.errors),skipped=len(controls.skipped),
  seconds=time.monotonic()-started,prior_elapsed_seconds=PRIOR_SECONDS,production_activation=False,
  source_hashes=frozen,draft_hashes=draft,provider_calls=0,compiler_calls=0,encrypted_calls=0,network_configuration_changes=0)
 report["binding"]=digest(report);(OUT/"report.json").write_text(json.dumps(report,allow_nan=False,sort_keys=True,indent=2))
 print(json.dumps({k:v for k,v in report.items()if k not in("source_hashes","draft_hashes","rows")}),flush=True)
 return 0 if report["status"]=="passed"else 1
def outer():
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 return enter_nix(cmd,seconds=REMAINING_SECONDS,keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
