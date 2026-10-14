"""Receipt/accounting/quiescence controls only; no production apply."""
import contextlib,copy,hashlib,io,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import cutover as c

class CutoverTests(unittest.TestCase):
 def setUp(self):
  self.plan={"binding":"test-runtime"}
  self.report=dict(format="poseidon-ordered-common-offline-report-v1",binding=self.plan["binding"],
   status="passed",provider_calls=0,http_attempts=0,new_agent_generation=False,production_applied=False,
   candidate_origin=c.scope.CANDIDATE_ORIGIN,pending=[],completed_phases=36,compilations=12,
   rows=[dict(case=n,phase=p,status="passed",exit_code=0) for n in c.scope.CASES for p in c.scope.PHASES])
 def test_complete_synthetic_phase_accounting_is_accepted(self):
  self.assertEqual(len(c.validate_report(self.report,self.plan)),36)
 def test_missing_phase_is_rejected(self):
  self.report["rows"].pop()
  with self.assertRaisesRegex(ValueError,"thirty-six"):c.validate_report(self.report,self.plan)
 def test_duplicate_phase_cannot_replace_missing_context(self):
  self.report["rows"][-1]=copy.deepcopy(self.report["rows"][0])
  with self.assertRaisesRegex(ValueError,"thirty-six"):c.validate_report(self.report,self.plan)
 def test_pending_failure_and_skip_are_not_success(self):
  for mutation in ("pending","failed","skipped"):
   report=copy.deepcopy(self.report)
   if mutation=="pending":report["pending"]=[{"case":"pending"}]
   else:report["rows"][0]["status"]=mutation
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):c.validate_report(report,self.plan)
 def test_rehashed_or_different_runtime_binding_is_not_accepted(self):
  self.report["binding"]="different"
  with self.assertRaisesRegex(ValueError,"completed joint"):c.validate_report(self.report,self.plan)
 def test_provider_or_agent_claims_are_rejected(self):
  for key,value in (("provider_calls",1),("http_attempts",1),("new_agent_generation",True),("production_applied",True)):
   report=copy.deepcopy(self.report);report[key]=value
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,"provenance"):c.validate_report(report,self.plan)
 def test_inherited_cache_acceptance_does_not_add_compilations(self):
  # Summary must count the twelve qualification phases, not cached provenance.
  self.report["compilations"]=24
  with self.assertRaisesRegex(ValueError,"accounting"):c.validate_report(self.report,self.plan)
 def test_boolean_count_and_exit_are_not_integer_receipts(self):
  for key in ("completed_phases","compilations"):
   r=copy.deepcopy(self.report);r[key]=True
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,"accounting"):c.validate_report(r,self.plan)
  self.report["rows"][0]["exit_code"]=False
  with self.assertRaisesRegex(ValueError,"unknown"):c.validate_report(self.report,self.plan)
 def test_candidate_origin_remains_non_agent(self):
  self.report["candidate_origin"]="new_agent"
  with self.assertRaisesRegex(ValueError,"provenance"):c.validate_report(self.report,self.plan)
 def test_active_process_defers_before_qualification(self):
  guard=types.SimpleNamespace(check=Mock())
  with patch.object(c,"applied_state",return_value=None),patch.object(c.scope,"guard",return_value=guard),\
       patch.object(c,"active_processes",return_value=[123]),patch.object(c,"qualification") as qualification:
   self.assertEqual(c.check()["status"],"waiting_for_process_exit")
   qualification.assert_not_called()
 def test_apply_without_qualification_never_acquires_lock_or_writes(self):
  with patch.object(c,"check",return_value=dict(status="waiting_for_offline_approval",production_applied=False)),\
       patch.object(c,"execution_locks") as locks,patch.object(c.transaction,"apply_transaction") as transaction:
   self.assertEqual(c.apply()["status"],"waiting_for_offline_approval")
   locks.assert_not_called();transaction.assert_not_called()
 def test_unqualified_boundary_is_read_only_with_approved_receipt(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);approval=root/"approval.json";approval.write_text("{}")
   guard=types.SimpleNamespace(check=Mock())
   with patch.object(c,"applied_state",return_value=None),patch.object(c.scope,"guard",return_value=guard),\
        patch.object(c,"active_processes",return_value=[]),patch.object(c.scope,"APPROVAL",approval),\
        patch.object(c.scope,"RUN",root/"absent-run"),patch.object(c,"qualification") as qualification:
    self.assertEqual(c.check()["status"],"waiting_for_joint_qualification")
    qualification.assert_not_called()
 def test_missing_final_record_requires_recovery_review(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);folder=root/"partial";folder.mkdir()
   (folder/"journal.json").write_text(json.dumps(dict(state="installing")))
   with patch.object(c,"JOURNALS",root),self.assertRaisesRegex(ValueError,"recovery review"):c.applied_state()

class AppliedFixture:
 """Fabricated receipts in a temp checkout; never calls production apply."""
 def __init__(self,root):
  self.root=root;self.here=root/"controls";self.run=root/"run";self.journals=self.here/"cutover-records"
  self.folder=self.journals/"synthetic-recovery";self.folder.mkdir(parents=True)
  self.tools={name:"1"*64 for name in ("cutover.py","source_transaction.py","evidence_audit.py",
    "test_source_transaction.py","test_evidence_audit.py","test_cutover.py")}
  self.extra=root/"scripts/baseline/golden_cases/synthetic.py";self.write_bytes(self.extra,b"# synthetic extra closure")
  reference=root/"scripts/baseline/reference.py";self.write_bytes(reference,b"# synthetic protected reference")
  self.protected={"scripts/baseline/reference.py":self.sha(reference)}
  protected_path=self.here/"protected.json";self.write(protected_path,self.protected)
  targets={};original={"synthetic/%03d.py"%i:"2"*64 for i in range(696)};inside=dict(original)
  self.targets={};self.original_bytes={}
  for index,name in enumerate(c.TARGET_ORDER):
   relative="scripts/baseline/"+name;target=root/relative;overlay=self.here/"overlay"/name
   self.write_bytes(overlay,("# synthetic new "+name).encode());self.write_bytes(target,overlay.read_bytes())
   os.chmod(target,0o644)
   old=None if index<2 else self.sha_bytes(("# synthetic original "+name).encode())
   if old is not None:original[relative]=old
   inside[relative]=self.sha(target)
   targets[name]=dict(source=str(overlay.relative_to(root)),target_sha256=inside[relative],original_sha256=old,
    original_mode=None if old is None else 0o644,target_mode=0o644,new_file=old is None)
   item={k:targets[name][k] for k in ("source","target_sha256","original_sha256","original_mode","target_mode")}
   item["source"]=str(overlay);self.targets[relative]=item
   if old is not None:
    self.original_bytes[relative]=("# synthetic original "+name).encode()
  self.plan=self.seal(dict(format="poseidon-common-fused-integration-v1",stage="prepared_pending_backend_qualification",
    production_applied=False,root=str(root),targets=targets,original_production_hashes=original,inside_source_hashes=inside,
    protected_sources={"protected":dict(path=str(protected_path),sha256=self.sha(protected_path))}))
  self.write(self.here/"plan.json",self.plan)
  bindings=dict(files={str(self.here/"plan.json"):self.sha(self.here/"plan.json")},combined_plan_binding=self.plan["binding"],
    inside_sources=inside,original_production=original,protected_sources=self.protected,
    requests={name:dict(request_id=c.digest(name)) for name in c.scope.CASES},backend_identity=dict(format="synthetic-sdk-only"))
  self.proposal=self.seal(dict(bindings=bindings,cases=list(c.scope.CASES),phases=list(c.scope.PHASES)))
  self.write(self.here/"proposal.json",self.proposal)
  self.runtime=self.seal(dict(bindings=bindings,cases=list(c.scope.CASES),phases=list(c.scope.PHASES),started_wall=1,deadline_wall=3601))
  self.write(self.run/"plan.json",self.runtime)
  self.write(self.run/"staged-sources.json",inside)
  self.write(self.run/"namespace-receipt.json",dict(synthetic_control_only=True))
  self.write(self.run/"report.json",dict(synthetic_control_only=True))
  cases={};evidence={str(self.run/f):self.sha(self.run/f) for f in ("plan.json","staged-sources.json","namespace-receipt.json","report.json")}
  for name in c.scope.CASES:
   files={str(self.extra):self.sha(self.extra)}
   for phase in c.scope.PHASES:
    p=self.run/name/(phase+".json");self.write(p,dict(synthetic_control_only=True,case=name,phase=phase));files[str(p)]=self.sha(p)
   cases[name]=dict(case=name,status="passed",runtime_binding=self.runtime["binding"],source_count=706,
     request_id=bindings["requests"][name]["request_id"],candidate_origin=c.scope.CANDIDATE_ORIGIN,
     reference_semantics="frozen_reviewed_runner",compilations=1,qualification_encrypted_phases=1,
     new_input_encrypted_phases=1,cache_generations=0,provider_calls=0,http_attempts=0,
     candidate_sha256="3"*64,package_binding="4"*64,application_identity_sha256="5"*64,
     backend_identity_sha256=c.digest(bindings["backend_identity"]),
     qualification=dict(compared_values=8,max_absolute_error=0.0,atol=1e-5,rtol=1e-4),
     new_inputs=dict(compared_values=6,max_absolute_error=0.0,atol=1e-5,rtol=1e-4),evidence_files=files)
   evidence.update(files)
  self.evidence=self.seal(dict(format="poseidon-joint-cutover-evidence-v1",runtime_binding=self.runtime["binding"],
    proposal_binding=self.proposal["binding"],combined_binding=self.plan["binding"],passed=12,cases=cases,evidence_files=evidence,
    provider_calls=0,compiler_calls=0,encrypted_executions=0,evidence_observed_only=True,agent_generated=False,production_applied=False))
  self.context=dict(combined_plan_binding=self.plan["binding"],qualification_binding=self.evidence["binding"],tool_hashes=self.tools,
    historical_records_unchanged=True,compiler_calls=0,encrypted_executions=0,provider_calls=0,quiescence_claim=c.QUIESCENCE_CLAIM)
  intents={};retained=[]
  for relative,item in self.targets.items():
   target=root/relative;captured=None
   if item["original_sha256"] is not None:
    path=self.folder/"install-displaced"/target.name/"current";self.write_bytes(path,self.original_bytes[relative]);os.chmod(path,0o644)
    captured=str(path);retained.append(captured)
   intents[relative]=dict(state="installed",retained=captured,prepared=str(target.with_name("."+target.name+"."+"a"*32+".tmp")))
  self.journal=dict(format="poseidon-agent-source-transaction-v1",state="applied",context=self.context,root=str(root),targets=self.targets,
    installed=list(self.targets),restored=[],rollback_conflicts=[],rollback_intents={},retained_files=retained,
    install_intents=intents,install_conflicts=[],created_wall=1.0,completed_wall=2.0)
  self.write(self.folder/"journal.json",self.journal);self.write(self.folder/"qualification.json",self.evidence)
  self.record=dict(format="poseidon-common-fused-cutover-v1",state="applied",recovery=str(self.folder),
    journal_sha256=self.sha(self.folder/"journal.json"),qualification_sha256=self.sha(self.folder/"qualification.json"),context=self.context,
    production_sources=inside,protected_sources=self.protected,production_applied=True,new_agent_generation=False,
    provider_calls=0,compiler_calls=0,encrypted_executions=0)
  self.save_record()

 @staticmethod
 def seal(value):
  value=copy.deepcopy(value);value.pop("binding",None);value["binding"]=c.digest(value);return value
 @staticmethod
 def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
 def sha(self,path):return self.sha_bytes(path.read_bytes())
 @staticmethod
 def write_bytes(path,raw):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 def write(self,path,value):self.write_bytes(path,c.canonical(value))
 def save_record(self):self.record=self.seal(self.record);self.write(self.folder/"record.json",self.record)
 def rehash_journal(self):
  self.write(self.folder/"journal.json",self.journal);self.record["journal_sha256"]=self.sha(self.folder/"journal.json");self.save_record()
 def rehash_qualification(self):
  value=json.loads((self.folder/"qualification.json").read_text());value=self.seal(value)
  self.write(self.folder/"qualification.json",value);self.record["qualification_sha256"]=self.sha(self.folder/"qualification.json")
  self.record["context"]["qualification_binding"]=value["binding"]
  self.journal["context"]=copy.deepcopy(self.record["context"]);self.rehash_journal()

class AppliedStateTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix="poseidon-cutover-controls-");self.addCleanup(self.temp.cleanup)
  self.f=AppliedFixture(Path(self.temp.name));self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
  for obj,name,value in ((c,"ROOT",self.f.root),(c,"HERE",self.f.here),(c,"JOURNALS",self.f.journals),
    (c.scope,"RUN",self.f.run),(c.scope,"PROPOSAL",self.f.here/"proposal.json")):
   self.stack.enter_context(patch.object(obj,name,value))
  self.stack.enter_context(patch.object(c.scope,"before_launch",return_value=None))
  self.stack.enter_context(patch.object(c.scope,"check_plan",return_value=None))
  self.stack.enter_context(patch.object(c,"tool_hashes",return_value=self.f.tools))
  self.stack.enter_context(patch.object(c,"current_inventory",return_value=self.f.plan["inside_source_hashes"]))
  self.stack.enter_context(patch.object(c,"qualification",side_effect=lambda mode="single_run":copy.deepcopy(self.f.evidence)))
 def test_complete_semantic_synthetic_chain_accepted(self):
  self.assertTrue(c.applied_state()["production_applied"])
 def test_rehashed_non_applied_journal_rejected(self):
  self.f.journal["state"]="rolled_back";self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"journal semantics"):c.applied_state()
 def test_wrong_checkout_rehashed_journal_rejected(self):
  self.f.journal["root"]="/other/root";self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"journal semantics"):c.applied_state()
 def test_missing_installed_target_rehashed_journal_rejected(self):
  self.f.journal["installed"].pop();self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"journal semantics"):c.applied_state()
 def test_reordered_installed_targets_rejected(self):
  self.f.journal["installed"].reverse();self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"journal semantics"):c.applied_state()
 def test_different_reviewed_target_rehashed_journal_rejected(self):
  self.f.journal["targets"]=copy.deepcopy(self.f.journal["targets"])
  self.f.journal["targets"][next(iter(self.f.targets))]["target_sha256"]="0"*64;self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"journal semantics"):c.applied_state()
 def test_rollback_intent_cannot_be_applied(self):
  self.f.journal["rollback_intents"]={"target":{}};self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"journal semantics"):c.applied_state()
 def test_incomplete_install_intent_rejected(self):
  self.f.journal["install_intents"].pop(next(iter(self.f.targets)));self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"installation intents"):c.applied_state()
 def test_retained_original_bytes_checked(self):
  Path(self.f.journal["retained_files"][0]).write_bytes(b"changed retained original")
  with self.assertRaisesRegex(ValueError,"retained original"):c.applied_state()
 def test_new_target_cannot_claim_displaced_original(self):
  first=next(iter(self.f.targets));self.f.journal["install_intents"][first]["retained"]="/invented";self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"no displaced"):c.applied_state()
 def test_missing_prepared_consumption_rejected(self):
  first=next(iter(self.f.targets));path=Path(self.f.journal["install_intents"][first]["prepared"]);path.write_bytes(b"unconsumed")
  with self.assertRaisesRegex(ValueError,"Consumed reviewed"):c.applied_state()
 def test_invented_context_even_when_journal_agrees_rejected(self):
  self.f.record["context"]["quiescence_claim"]="global lock held"
  self.f.journal["context"]=copy.deepcopy(self.f.record["context"]);self.f.rehash_journal()
  with self.assertRaisesRegex(ValueError,"context semantics"):c.applied_state()
 def test_record_inventory_cannot_diverge_from_plan(self):
  self.f.record["production_sources"]={};self.f.save_record()
  with self.assertRaisesRegex(ValueError,"immutable complete inventories"):c.applied_state()
 def test_missing_case_rehashed_qualification_rejected(self):
  path=self.f.folder/"qualification.json";value=json.loads(path.read_text());value["cases"].pop(c.scope.CASES[-1]);self.f.write(path,value);self.f.rehash_qualification()
  with self.assertRaisesRegex(ValueError,"twelve qualification"):c.applied_state()
 def test_case_wrong_request_rehashed_qualification_rejected(self):
  path=self.f.folder/"qualification.json";value=json.loads(path.read_text());value["cases"][c.scope.CASES[0]]["request_id"]="f"*64;self.f.write(path,value);self.f.rehash_qualification()
  with self.assertRaisesRegex(ValueError,"Qualified case identity"):c.applied_state()
 def test_case_wrong_sdk_rehashed_qualification_rejected(self):
  path=self.f.folder/"qualification.json";value=json.loads(path.read_text());value["cases"][c.scope.CASES[0]]["backend_identity_sha256"]="f"*64;self.f.write(path,value);self.f.rehash_qualification()
  with self.assertRaisesRegex(ValueError,"exact SDK"):c.applied_state()
 def test_bad_qualification_self_binding_rejected(self):
  path=self.f.folder/"qualification.json";value=json.loads(path.read_text());value["binding"]="0"*64;self.f.write(path,value)
  self.f.record["qualification_sha256"]=self.f.sha(path);self.f.save_record()
  with self.assertRaisesRegex(ValueError,"self binding"):c.applied_state()
 def test_missing_union_evidence_rehashed_qualification_rejected(self):
  path=self.f.folder/"qualification.json";value=json.loads(path.read_text());value["evidence_files"].pop(str(self.f.extra));self.f.write(path,value);self.f.rehash_qualification()
  with self.assertRaisesRegex(ValueError,"evidence hash chain"):c.applied_state()
 def test_extra_ast_closure_drift_rejected(self):
  self.f.extra.write_bytes(b"changed extra closure")
  with self.assertRaisesRegex(ValueError,"source boundary"):c.applied_state()
 def test_only_ten_targets_are_exempt_from_qualification_hash_protection(self):
  evidence=copy.deepcopy(self.f.evidence)
  target=str(self.f.root/next(iter(self.f.targets)));evidence["evidence_files"][target]="0"*64
  c.protect_evidence(evidence,self.f.targets)
  unrelated=str(self.f.extra);evidence["evidence_files"][unrelated]="0"*64
  with self.assertRaisesRegex(ValueError,"source boundary"):c.protect_evidence(evidence,self.f.targets)
 def test_qualification_is_independently_recreated(self):
  path=self.f.folder/"qualification.json";value=json.loads(path.read_text());value["cases"][c.scope.CASES[0]]["candidate_sha256"]="f"*64;self.f.write(path,value);self.f.rehash_qualification()
  with self.assertRaisesRegex(ValueError,"Independent qualification semantic"):c.applied_state()
 def test_changed_installed_permission_rejected(self):
  os.chmod(self.f.root/next(iter(self.f.targets)),0o600)
  with self.assertRaisesRegex(ValueError,"permissions"):c.applied_state()

 def test_persisted_composite_evidence_selects_composite_revalidation(self):
  value=dict(format="poseidon-joint-composite-cutover-evidence-v1",
   combined_binding=self.f.plan["binding"],source_runs=dict(
    original=dict(runtime_binding=self.f.runtime["binding"],proposal_binding=self.f.proposal["binding"]),
    capacity=dict()),cases=copy.deepcopy(self.f.evidence["cases"]),passed=12,phases=36,
   evidence_files=copy.deepcopy(self.f.evidence["evidence_files"]),provider_calls=0,compiler_calls=0,
   encrypted_executions=0,evidence_observed_only=True,agent_generated=False,production_applied=False)
  value=self.f.seal(value);self.f.evidence=value
  self.f.write(self.f.folder/"qualification.json",value);self.f.rehash_qualification()
  module=types.ModuleType("composite_evidence");module.validate_evidence=Mock(return_value=None)
  with patch.dict(sys.modules,{"composite_evidence":module}),\
       patch.object(c,"qualification",return_value=copy.deepcopy(value)) as qualification:
   self.assertTrue(c.applied_state()["production_applied"])
   qualification.assert_called_once_with(mode="composite")
   module.validate_evidence.assert_called_once_with(value,self.f.plan)


class CompositeModeTests(unittest.TestCase):
 """Synthetic routing controls; no transaction or backend is executed."""
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix="poseidon-composite-cutover-controls-");self.addCleanup(self.temp.cleanup)
  folder=Path(self.temp.name);self.original=folder/"original";self.original.mkdir()
  (self.original/"report.json").write_text("{}")
  self.original_approval=folder/"original-approval.json";self.original_approval.write_text("{}")
  self.capacity=types.SimpleNamespace(APPROVAL=folder/"capacity-approval.json",RUN=folder/"capacity")
  self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
  for obj,name,value in ((c,"applied_state",Mock(return_value=None)),(c,"active_processes",Mock(return_value=[])),
    (c.scope,"guard",Mock(return_value=types.SimpleNamespace(check=Mock()))),
    (c.scope,"APPROVAL",self.original_approval),(c.scope,"RUN",self.original),
    (c,"capacity_scope_for_evidence",Mock(return_value=self.capacity))):
   self.stack.enter_context(patch.object(obj,name,value))
  self.proof=self.seal(dict(format="poseidon-joint-composite-cutover-evidence-v1",combined_binding="integration",
   source_runs=dict(original=dict(runtime_binding="runtime",proposal_binding="proposal"),capacity=dict()),
   passed=12,phases=36,cases={},evidence_files={},provider_calls=0,compiler_calls=0,
   encrypted_executions=0,evidence_observed_only=True,agent_generated=False,production_applied=False))
  self.runtime={"binding":"runtime"};self.proposal={"binding":"proposal"};self.plan={"binding":"integration"}
 @staticmethod
 def seal(value):
  value=copy.deepcopy(value);value.pop("binding",None);value["binding"]=c.digest(value);return value
 def module(self):
  module=types.ModuleType("composite_evidence")
  module.qualification=Mock(return_value=copy.deepcopy(self.proof))
  module.validate_evidence=Mock(return_value=None)
  return module
 def test_default_single_run_does_not_require_capacity(self):
  with patch.object(c,"qualification",return_value={"binding":"single"}) as qualification:
   self.assertEqual(c.check()["status"],"ready")
   qualification.assert_called_once_with(mode="single_run")
   c.capacity_scope_for_evidence.assert_not_called()
 def test_unapproved_capacity_has_explicit_waiting_state_without_qualification(self):
  with patch.object(c,"qualification") as qualification:
   self.assertEqual(c.check(mode="composite")["status"],"waiting_for_capacity_offline_approval")
   qualification.assert_not_called()
 def test_approved_capacity_missing_report_has_separate_waiting_state(self):
  self.capacity.APPROVAL.write_text("{}")
  with patch.object(c,"qualification") as qualification:
   self.assertEqual(c.check(mode="composite")["status"],"waiting_for_capacity_qualification")
   qualification.assert_not_called()
 def test_completed_composite_mode_routes_qualification_without_new_execution(self):
  self.capacity.APPROVAL.write_text("{}");self.capacity.RUN.mkdir()
  (self.capacity.RUN/"report.json").write_text("{}")
  with patch.object(c,"qualification",return_value=self.proof) as qualification:
   state=c.check(mode="composite")
   self.assertEqual((state["status"],state["cases"],state["phases"]),("ready",12,36))
   self.assertFalse(state["production_applied"])
   qualification.assert_called_once_with(mode="composite")
 def test_composite_validation_delegates_only_after_original_context_match(self):
  module=self.module()
  with patch.dict(sys.modules,{"composite_evidence":module}):
   self.assertEqual(c.validate_qualification(self.proof,self.runtime,self.proposal,self.plan),self.proof)
   module.validate_evidence.assert_called_once_with(self.proof,self.plan)
 def test_wrong_original_runtime_or_proposal_cannot_be_hidden_by_rehash(self):
  for field in ("runtime_binding","proposal_binding"):
   module=self.module();value=copy.deepcopy(self.proof)
   value["source_runs"]["original"][field]="other";value=self.seal(value)
   with self.subTest(field=field),patch.dict(sys.modules,{"composite_evidence":module}):
    with self.assertRaisesRegex(ValueError,"original composite qualification context"):
     c.validate_qualification(value,self.runtime,self.proposal,self.plan)
    module.validate_evidence.assert_not_called()
 def test_missing_original_source_run_is_rejected_before_delegate(self):
  module=self.module();value=copy.deepcopy(self.proof);value["source_runs"].pop("original");value=self.seal(value)
  with patch.dict(sys.modules,{"composite_evidence":module}):
   with self.assertRaisesRegex(ValueError,"original composite source run"):
    c.validate_qualification(value,self.runtime,self.proposal,self.plan)
   module.validate_evidence.assert_not_called()
 def test_composite_qualification_calls_data_only_module_and_validator(self):
  module=self.module()
  with patch.object(c,"reviewed_context",return_value=(self.plan,self.runtime,self.proposal,{},{})),\
       patch.dict(sys.modules,{"composite_evidence":module}):
   self.assertEqual(c.qualification(mode="composite"),self.proof)
   module.qualification.assert_called_once_with()
   module.validate_evidence.assert_called_once_with(self.proof,self.plan)
 def test_apply_waits_for_capacity_approval_without_locks_or_transaction(self):
  with patch.object(c,"execution_locks") as locks,patch.object(c.transaction,"apply_transaction") as transaction:
   self.assertEqual(c.apply(mode="composite")["status"],"waiting_for_capacity_offline_approval")
   locks.assert_not_called();transaction.assert_not_called()
 def test_apply_rechecks_same_selected_mode_at_lock_boundary(self):
  with patch.object(c,"check",side_effect=[{"status":"ready"},{"status":"waiting_for_process_exit"}]) as check,\
       patch.object(c,"execution_locks",return_value=contextlib.nullcontext()),\
       patch.object(c.transaction,"apply_transaction") as transaction:
   with self.assertRaisesRegex(ValueError,"cutover boundary changed"):c.apply(mode="composite")
   self.assertEqual(check.call_args_list,[unittest.mock.call(mode="composite"),unittest.mock.call(mode="composite")])
   transaction.assert_not_called()
 def test_invalid_mode_never_reaches_qualification_or_transaction(self):
  with patch.object(c,"reviewed_context") as context,patch.object(c.transaction,"apply_transaction") as transaction:
   for method in (c.check,c.qualification,c.apply):
    with self.subTest(method=method.__name__),self.assertRaisesRegex(ValueError,"evidence mode"):method(mode="other")
   context.assert_not_called();transaction.assert_not_called()
 def test_cli_default_and_explicit_composite_selection_are_read_only(self):
  for args,mode in (([],"single_run"),(["--evidence-mode","composite"],"composite")):
   with patch.object(c,"check",return_value={"status":"waiting","production_applied":False}) as check,\
        patch.object(c,"apply") as apply,contextlib.redirect_stdout(io.StringIO()):
    c.main(args)
    check.assert_called_once_with(mode=mode);apply.assert_not_called()
 def test_cli_explicit_apply_passes_mode_to_controlled_action(self):
  with patch.object(c,"check") as check,patch.object(c,"apply",return_value={"status":"waiting"}) as apply,\
       contextlib.redirect_stdout(io.StringIO()):
   c.main(["--apply","--evidence-mode","composite"])
   apply.assert_called_once_with(mode="composite");check.assert_not_called()
 def test_tool_identity_includes_composite_validator_and_tests(self):
  with patch.object(c.scope,"sha",return_value="0"*64):
   files=c.tool_hashes()
   self.assertIn("composite_evidence.py",files);self.assertIn("test_composite_evidence.py",files)
   self.assertEqual(len(files),8)

if __name__=="__main__":unittest.main(verbosity=2)
