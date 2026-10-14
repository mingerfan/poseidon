"""Post-activation Agent controls; immutable historical test records remain."""
import ast,hashlib,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";sys.path.insert(0,str(BASE))
from benchmark_graph import canonical,digest
from application_identity import source_closure
from application_planning import resources,backend_parameters
from test_backend_parameters_boundary import ParameterBoundaryTests
from test_application_identity import IdentityTests
from test_compiled_execution_precheck import (BoundArtifactControls,TypedResourceControls,NativeExecutionControls,
 CandidateAdapterControls,OutputProvenanceControls)
from test_portability import PortablePathsTests,LauncherTests
STAGES=("generation","compilation","execution","validation")
PROPOSAL=json.loads((ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1317.json").read_text())
TOOL=Path(__file__).resolve().parent
class ActivationTests(unittest.TestCase):
 def test_legacy_parameter_interfaces_are_exact_aliases(self):
  for name in ("_integer","native_address_limit","parameters"):
   self.assertIs(getattr(resources,name),getattr(backend_parameters,name))
 def test_activated_patch_is_reversible_and_all_before_bytes_retained(self):
  result=subprocess.run(["git","apply","--reverse","--check",str(BASE/"benchmarks/tools/resource_parameter_boundary_v1/activate.patch")],
    cwd=ROOT,capture_output=True,text=True,timeout=10)
  self.assertEqual(result.returncode,0,result.stderr)
  before=json.loads((TOOL/"source-transition.json").read_text())["source_before"]
  for name,wanted in before.items():
   self.assertEqual(hashlib.sha256((TOOL/"source-before"/name).read_bytes()).hexdigest(),wanted)
 def test_active_stages_bind_reader_and_exclude_generation_from_non_generation(self):
  for stage in STAGES:
   closure=source_closure(BASE,tuple(PROPOSAL["source_stage_identity"][stage]["roots"]))
   self.assertIn("application_planning/backend_parameters.py",closure["sources"])
   self.assertNotEqual(closure["sources"],PROPOSAL["source_stage_identity"][stage]["sources"])
   if stage!="generation":
    for name in ("deepseek_provider.py","agent_credentials.py","application_generation.py",
     "application_component.py","run_candidate.py","run_agent_batch.py","sandbox_policy.py"):
     self.assertNotIn(name,closure["sources"])
 def test_provider_mutation_only_invalidates_generation_in_a_source_copy(self):
  with tempfile.TemporaryDirectory(prefix="source-proof-")as folder:
   copied=Path(folder);closures={}
   for stage in STAGES:
    closures[stage]=source_closure(BASE,tuple(PROPOSAL["source_stage_identity"][stage]["roots"]))
   names=set().union(*(set(c["sources"])for c in closures.values()))
   for name in names:
    target=copied/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((BASE/name).read_bytes())
   provider=copied/"deepseek_provider.py";provider.write_bytes(provider.read_bytes()+b"\n# Mutation fixture.\n")
   for stage in STAGES:
    altered=source_closure(copied,tuple(PROPOSAL["source_stage_identity"][stage]["roots"]))
    if stage=="generation":self.assertNotEqual(altered["sources"],closures[stage]["sources"])
    else:self.assertEqual(altered,closures[stage])
   reader=copied/"application_planning/backend_parameters.py";reader.write_bytes(reader.read_bytes()+b"\n# Mutation fixture.\n")
   for stage in STAGES:
    altered=source_closure(copied,tuple(PROPOSAL["source_stage_identity"][stage]["roots"]))
    self.assertNotEqual(altered["sources"],closures[stage]["sources"])
 def test_retired_pending_plan_refuses_changed_source_without_api(self):
  result=subprocess.run([sys.executable,"-B",str(BASE/"benchmarks/tools/mha_feedback_live_v3/run.py"),"--plan"],
    cwd=ROOT,capture_output=True,text=True,timeout=30)
  self.assertNotEqual(result.returncode,0)
  self.assertIn("frozen_source_changed",result.stderr)
  self.assertFalse((ROOT/"docs/baseline/agent-mha-feedback-live-authorization-r1316.json").exists())
 def test_only_declared_protected_source_changes(self):
  protected=json.loads((ROOT/"docs/baseline/agent-v12-live-rebound-proposal-r1304.json").read_text())["protected_sources"]
  normalize=lambda n:str((Path(n)if Path(n).is_absolute()else ROOT/n).relative_to(ROOT))
  changed={normalize(name)for name,want in protected.items()if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=want}
  declared={"scripts/baseline/application_planning/resources.py",
            "scripts/baseline/application_planning/compiled_resources.py",
            "scripts/baseline/test_compiled_execution_precheck.py"}
  self.assertEqual(changed,declared&{normalize(n)for n in protected})
  before=json.loads((TOOL/"source-transition.json").read_text())["source_before"]
  current_changed={name for name,want in before.items()if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=want}
  self.assertEqual(current_changed,declared-{"scripts/baseline/test_compiled_execution_precheck.py"})
class Results(unittest.TextTestResult):
 def startTest(self,test):
  self.ids=getattr(self,"ids",[]);self.ids.append(test.id());super().startTest(test)
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument("--report",required=True);args=ap.parse_args()
 out=Path(args.report);assert not out.exists()
 names=("application_planning/resources.py","application_planning/compiled_resources.py",
        "application_planning/backend_parameters.py","test_compiled_execution_precheck.py")
 pins={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()for n in names}
 suite=unittest.TestSuite()
 omitted={"test_activation_patch_is_clean_and_current_source_is_frozen",
          "test_patch_removes_backedges_without_changing_old_interface"}
 for name in unittest.defaultTestLoader.getTestCaseNames(ParameterBoundaryTests):
  if name not in omitted:suite.addTest(ParameterBoundaryTests(name))
 for cls in (ActivationTests,IdentityTests,BoundArtifactControls,TypedResourceControls,NativeExecutionControls,
  CandidateAdapterControls,OutputProvenanceControls,PortablePathsTests,LauncherTests):
  suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
 start=time.monotonic();r=unittest.TextTestRunner(verbosity=2,resultclass=Results).run(suite)
 after={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()for n in names}
 proof={stage:source_closure(BASE,tuple(PROPOSAL["source_stage_identity"][stage]["roots"]))for stage in STAGES}
 report=dict(format="poseidon-backend-parameter-activation-controls-v1",tests=r.testsRun,
  passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),failed=len(r.failures),errors=len(r.errors),
  skipped=len(r.skipped),test_ids=r.ids,failed_ids=[t.id()for t,_ in r.failures],error_ids=[t.id()for t,_ in r.errors],
  selected_historical_tests=9,omitted_pre_activation_only_tests=sorted(omitted),
  source_sha256=pins,source_unchanged=pins==after,
  source_closures=proof,source_counts={s:len(c["sources"])for s,c in proof.items()},
  control_outputs_injected=True,actual_compiler_calls=0,actual_encrypted_calls=0,api_calls=0,
  elapsed_seconds=time.monotonic()-start)
 report["binding"]=digest(report)
 with out.open("xb")as f:f.write(canonical(report))
 print(json.dumps({k:report[k]for k in("tests","passed","failed","errors","skipped","source_counts","binding")}))
 return 0 if r.wasSuccessful()and pins==after else 1
if __name__=="__main__":raise SystemExit(main())
