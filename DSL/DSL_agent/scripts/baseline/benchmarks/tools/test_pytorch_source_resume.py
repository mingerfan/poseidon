"""Pure continuation guards; no provider or encrypted execution."""
import copy,sys,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_graph import digest
from check_pytorch_sources import select_resume_rows

class ResumeBoundaryTests(unittest.TestCase):
    def fixture(self):
        cases=[dict(id=n) for n in ("done","partial","pending","failed")]
        sources={"source.py":"a"*64};identity={"backend":"poseidon_cpu"}
        plan=dict(format="poseidon-pytorch-offline-plan-v1",cases=cases,source_hashes=sources,
            backend_identity=identity,capture_only=False,paid_calls=0,source="rule baseline")
        plan["binding"]=digest(plan)
        report=dict(binding=plan["binding"],status="stopped",paid_calls=0,agent_generated=False,
            passed=1,failed=1,rows=[dict(id=n,status=s,capture=dict(reference_verified=True))
                for n,s in zip(("done","partial","pending","failed"),
                               ("passed","running","reference_passed","failed"))])
        bounds=dict(original_wall_seconds=3600,additional_wall_seconds=3600,total_wall_seconds=7200,
            aggregate_added_artifact_bytes=4*1024**3,global_artifact_bytes=32*1024**3,
            free_disk_bytes=4*1024**3,compiler_jobs=2,link_jobs=1,encrypted_jobs=1,
            sandbox_address_space_bytes=4*1024**3)
        proposal=dict(format="poseidon-offline-continuation-proposal-v1",parent_binding=plan["binding"],
            api_calls=0,downloads=0,parameters_changed=False,bounds=bounds,candidate_ids=["partial","pending"])
        return plan,report,proposal,cases,sources,identity
    def test_only_unfinished_original_cases(self):
        args=self.fixture();rows,selected=select_resume_rows(*args)
        self.assertEqual(selected,["partial","pending"]);self.assertEqual(rows,args[1]["rows"])
    def test_parent_records_are_not_mutated(self):
        args=self.fixture();before=copy.deepcopy(args);rows,_=select_resume_rows(*args)
        rows[0]["status"]="changed";self.assertEqual(args,before)
    def test_plan_hash_tampering_refused(self):
        args=list(self.fixture());args[0]["capture_only"]=True
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_report_binding_refused(self):
        args=list(self.fixture());args[1]["binding"]="b"*64
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_live_checkpoint_requires_terminal_observation(self):
        args=list(self.fixture());args[1]["status"]="running"
        with self.assertRaises(ValueError):select_resume_rows(*args)
        self.assertEqual(select_resume_rows(*args,terminal_observed=True)[1],["partial","pending"])
    def test_completed_batch_cannot_resume(self):
        args=list(self.fixture());args[1]["status"]="complete"
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_paid_record_refused(self):
        args=list(self.fixture());args[1]["paid_calls"]=1
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_agent_source_refused(self):
        args=list(self.fixture());args[1]["agent_generated"]=True
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_source_drift_refused(self):
        args=list(self.fixture());args[4]={"source.py":"b"*64}
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_backend_drift_refused(self):
        args=list(self.fixture());args[5]={"backend":"other"}
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_index_drift_refused(self):
        args=list(self.fixture());args[3]=list(reversed(args[3]))
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_paid_or_parameter_extension_refused(self):
        for field,value in (("api_calls",1),("downloads",1),("parameters_changed",True)):
            args=list(self.fixture());args[2][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):select_resume_rows(*args)
    def test_resource_extension_refused(self):
        for field,value in (("additional_wall_seconds",7200),("encrypted_jobs",2),
                            ("free_disk_bytes",1),("global_artifact_bytes",64*1024**3)):
            args=list(self.fixture());args[2]["bounds"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):select_resume_rows(*args)
    def test_approval_scope_missing_remaining_case_refused(self):
        args=list(self.fixture());args[2]["candidate_ids"]=["partial"]
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_approval_duplicates_refused(self):
        args=list(self.fixture());args[2]["candidate_ids"]=["partial","pending","pending"]
        with self.assertRaises(ValueError):select_resume_rows(*args)
    def test_false_reference_or_counters_refused(self):
        for update in ("reference","counter"):
            args=list(self.fixture())
            if update=="reference":args[1]["rows"][1]["capture"]["reference_verified"]=False
            else:args[1]["passed"]=2
            with self.subTest(update=update),self.assertRaises(ValueError):select_resume_rows(*args)
    def test_failed_and_passed_cases_are_never_selected(self):
        args=list(self.fixture());args[2]["candidate_ids"]+=["done","failed"]
        self.assertEqual(select_resume_rows(*args)[1],["partial","pending"])
if __name__=="__main__":unittest.main()
