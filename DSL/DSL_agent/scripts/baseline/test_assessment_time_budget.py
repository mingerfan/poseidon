"""Persisted budget/admission tests; numerical execution is an injected test backend."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import assessment_time_budget as clock
from application_assessment import ApplicationAssessment,AssessmentControllerError,read,seal
import test_application_assessment as fixtures
from benchmark_graph import canonical
BOOT="11111111-1111-1111-1111-111111111111"
class ClockContractTests(unittest.TestCase):
    def now(self,wall=1000.,ticks=100.,boot=BOOT):
        return dict(wall=wall,clock=ticks,boot=boot)
    def make(self,seconds=10):
        with patch.object(clock,"current_clock",return_value=self.now()):return clock.create(seconds)
    def test_duration_bound_does_not_accept_bool_or_expand_command_maximum(self):
        for v in (True,False,0,-1,14401,10.,None):
            with self.subTest(v=v),self.assertRaises(clock.TimeBudgetError):self.make(v)
        for v in (1,14400):self.assertEqual(self.make(v)["seconds"],v)
    def test_both_clock_deadlines_stop_at_the_exact_boundary(self):
        b=self.make()
        for wall,ticks,status in ((1009.,109.,None),(1010.,109.,"timed_out"),(1009.,110.,"timed_out")):
            with self.subTest(wall=wall,ticks=ticks),patch.object(clock,"current_clock",return_value=self.now(wall,ticks)):
                r=clock.stop_reason(b);self.assertEqual(None if r is None else r[0],status)
    def test_backward_clock_and_reboot_are_service_failures_without_restarting_time(self):
        b=self.make()
        for now in (self.now(999.,102.),self.now(1001.,99.),self.now(1001.,101.,"22222222-2222-2222-2222-222222222222")):
            with self.subTest(now=now),patch.object(clock,"current_clock",return_value=now):
                self.assertEqual(clock.stop_reason(b),("service_error","assessment_controller_clock_unavailable"))
    def test_deadline_precedes_a_late_cancellation_and_cancel_precedes_normal_work(self):
        b=self.make()
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)):
            self.assertEqual(clock.stop_reason(b,cancelled=True)[0],"cancelled")
        with patch.object(clock,"current_clock",return_value=self.now(1011.,111.)):
            self.assertEqual(clock.stop_reason(b,cancelled=True)[0],"timed_out")
    def test_malformed_budget_or_changed_deadline_is_rejected(self):
        b=self.make()
        bads=[dict(b,seconds=True),dict(b,deadline_wall=b["deadline_wall"]+1),
            dict(b,deadline_clock=b["deadline_clock"]+1),dict(b,created_wall=float("nan")),
            dict(b,boot_id="PRIVATE"),dict(b,extra="PRIVATE")]
        for v in bads:
            with self.subTest(keys=v.keys()),self.assertRaises(clock.TimeBudgetError):clock.checked(v)
    def test_legacy_plan_cannot_hide_a_new_budget_and_v2_requires_it(self):
        b=self.make()
        self.assertIsNone(clock.plan_budget(dict(format=clock.PLAN_V1)))
        for p in (dict(format=clock.PLAN_V1,time_budget=b),dict(format=clock.PLAN_V2),dict(format="unknown")):
            with self.assertRaises(clock.TimeBudgetError):clock.plan_budget(p)
        self.assertEqual(clock.plan_budget(dict(format=clock.PLAN_V2,time_budget=b)),b)
    def test_clock_unavailable_is_a_known_nonretryable_stop(self):
        b=self.make()
        with patch.object(clock,"current_clock",side_effect=clock.TimeBudgetError("PRIVATE")):
            self.assertEqual(clock.stop_reason(b),("service_error","assessment_controller_clock_unavailable"))
class DeadlineAssessmentTests(unittest.TestCase):
    setUp=fixtures.ApplicationAssessmentTests.setUp
    ready=fixtures.ApplicationAssessmentTests.ready
    prepared=fixtures.ApplicationAssessmentTests.prepared
    submitted=fixtures.ApplicationAssessmentTests.submitted
    def now(self,wall=1000.,ticks=100.,boot=BOOT):return dict(wall=wall,clock=ticks,boot=boot)
    def limited(self,n=17,quota=2):
        with patch.object(clock,"current_clock",return_value=self.now()):
            return self.submitted(n,quota,total_wall_seconds=10)
    def test_expired_queued_request_does_not_admit_a_call(self):
        c,h,task,p,outcome,*_=self.limited()
        with patch.object(clock,"current_clock",return_value=self.now(1010.,110.)):
            r=c.advance(h,max_new_parts=2)
        self.assertEqual((r["status"],r["budget"]["used"]),("timed_out",0))
        self.assertEqual(r["failure"]["code"],"assessment_controller_deadline_reached")
        self.assertEqual(self.backend.execute_calls,0);self.assertEqual(self.app.result(task),outcome)
    def test_total_deadline_is_shared_across_controller_reconstruction_and_commands(self):
        c,h,*_=self.limited()
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)):
            r=c.advance(h);self.assertEqual(r["status"],"ready")
        p=c.store/h["evaluation_id"]/"plan.json";before=p.read_bytes()
        new=ApplicationAssessment(self.app,c.store)
        with patch.object(clock,"current_clock",return_value=self.now(1011.,111.)):
            r=new.status(h);again=new.advance(h,max_new_parts=4096)
        self.assertEqual(r["status"],"timed_out");self.assertEqual(again["binding"],r["binding"])
        self.assertEqual(r["budget"]["used"],1);self.assertEqual(self.backend.execute_calls,1)
        self.assertEqual(p.read_bytes(),before);self.assertEqual(r["completed_parts"],1)
    def test_expiry_during_direct_call_keeps_finished_receipt_and_stops_remaining_work(self):
        c,h,*_=self.limited();now=self.now(1001.,101.)
        self.backend.after_execute=lambda:now.update(wall=1011.,clock=111.)
        with patch.object(clock,"current_clock",side_effect=lambda:dict(now)):r=c.advance(h,max_new_parts=2)
        self.backend.after_execute=None
        self.assertEqual((r["status"],r["budget"]["used"],r["completed_parts"]),("timed_out",1,1))
        self.assertEqual(self.backend.execute_calls,1);self.assertFalse(r["acceptance"]["numerical_on_declared_dataset"])
    def test_reboot_before_any_execution_is_service_error_with_zero_calls(self):
        c,h,*_=self.limited()
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.,"22222222-2222-2222-2222-222222222222")):
            r=c.advance(h)
        self.assertEqual(r["status"],"service_error");self.assertEqual(r["budget"]["used"],0)
        self.assertEqual(r["failure"]["code"],"assessment_controller_clock_unavailable")
        self.assertEqual(self.backend.execute_calls,0)
    def test_orphan_recovery_after_deadline_preserves_used_and_unknown_execution(self):
        c,h,*_=self.limited()
        class Stopped(BaseException):pass
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)),patch.object(c.numeric,"assess",side_effect=Stopped):
            with self.assertRaises(Stopped):c.advance(h)
        with patch.object(clock,"current_clock",return_value=self.now(1011.,111.)):
            r=ApplicationAssessment(self.app,c.store).status(h)
        self.assertEqual((r["status"],r["budget"]["used"]),("timed_out",1))
        self.assertTrue(r["ledger"][0]["execution_uncertain"])
        self.assertEqual(c.advance(h)["binding"],r["binding"]);self.assertEqual(self.backend.execute_calls,0)
    def test_orphan_cancel_is_durable_and_does_not_clear_uncertainty(self):
        c,h,*_=self.limited()
        class Stopped(BaseException):pass
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)),patch.object(c.numeric,"assess",side_effect=Stopped):
            with self.assertRaises(Stopped):c.advance(h)
        with patch.object(clock,"current_clock",return_value=self.now(1002.,102.)):r=c.cancel(h)
        self.assertEqual((r["status"],r["budget"]["used"]),("cancelled",1))
        self.assertTrue(r["ledger"][0]["execution_uncertain"]);self.assertEqual(self.backend.execute_calls,0)
    def test_no_new_budget_keeps_legacy_shape_and_does_not_read_the_clock(self):
        c,h,*_=self.submitted(3,quota=1)
        plan=read(c.store/h["evaluation_id"]/"plan.json")
        self.assertEqual(plan["format"],clock.PLAN_V1);self.assertNotIn("time_budget",plan)
        with patch.object(clock,"current_clock",side_effect=AssertionError("legacy clock consulted")):r=c.advance(h)
        self.assertEqual(r["status"],"succeeded");self.assertNotIn("total_wall_seconds",r["budget"])
    def test_completed_result_is_stable_after_time_limit_or_reboot(self):
        c,h,*_=self.limited(3,quota=1)
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)):r=c.advance(h)
        with patch.object(clock,"current_clock",side_effect=AssertionError("terminal clock consulted")):
            self.assertEqual(c.status(h)["binding"],r["binding"]);self.assertEqual(c.cancel(h)["binding"],r["binding"])
        self.assertEqual(self.backend.execute_calls,1)
    def test_v2_public_budget_contains_duration_and_deadline_but_not_boot_identity(self):
        c,h,*_=self.limited()
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)):r=c.status(h)
        self.assertEqual(r["budget"]["total_wall_seconds"],10);self.assertEqual(r["budget"]["deadline_wall"],1010.)
        self.assertNotIn(BOOT,json.dumps(r));self.assertEqual(self.backend.execute_calls,0)
    def test_invalid_duration_rejected_before_a_new_assessment_directory(self):
        c,task,folder,m,*_=self.prepared(3);before=set(c.store.iterdir())
        for n in (True,0,-1,14401,10.):
            with self.subTest(n=n),self.assertRaises(AssessmentControllerError):
                c.submit(task,folder/"dataset.json",expected_dataset_binding=m["binding"],total_wall_seconds=n)
        self.assertEqual(set(c.store.iterdir()),before);self.assertEqual(self.backend.execute_calls,0)
    def test_deadline_tampering_is_rejected_before_execution(self):
        c,h,*_=self.limited();p=c.store/h["evaluation_id"]/"plan.json";plan=read(p)
        plan["time_budget"]["deadline_wall"]+=600;p.write_bytes(canonical(seal(plan)))
        with self.assertRaises(AssessmentControllerError):c.advance(h)
        self.assertEqual(self.backend.execute_calls,0)
    def test_corrupted_owned_dataset_is_integrity_failure_even_after_time_is_over(self):
        c,h,*_=self.limited(3,quota=1)
        p=c.store/h["evaluation_id"]/"dataset/input_0.npy";raw=p.read_bytes();p.write_bytes(raw[:-1]+b"x")
        with patch.object(clock,"current_clock",return_value=self.now(1011.,111.)),self.assertRaises(AssessmentControllerError):c.advance(h)
        self.assertEqual(self.backend.execute_calls,0)
    def test_budget_watch_is_metadata_only_and_detects_plan_and_cancel_changes(self):
        c,h,*_=self.limited(3,quota=1);w=clock.BudgetWatch(c.store,h["evaluation_id"],h["binding"])
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)),patch("assessment_time_budget._read",wraps=clock._read) as reads:
            self.assertIsNone(w.check())
        self.assertTrue(all(Path(call.args[0]).name in ("plan.json","state.json") for call in reads.call_args_list))
        marker=c.store/h["evaluation_id"]/"cancel.json"
        marker.write_text(json.dumps(dict(format="poseidon-application-assessment-v1",evaluation_id=h["evaluation_id"],
            submission_binding=h["binding"],cancel_requested=True)))
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)):self.assertEqual(w.check()[0],"cancelled")
        marker.write_text('{"cancel_requested":true}')
        with self.assertRaises(clock.TimeBudgetError):w.check()
    def test_wrong_watch_handle_and_symlink_metadata_are_rejected(self):
        c,h,*_=self.limited(3,quota=1)
        with self.assertRaises(clock.TimeBudgetError):clock.BudgetWatch(c.store,h["evaluation_id"],"0"*64)
        w=clock.BudgetWatch(c.store,h["evaluation_id"],h["binding"]);p=w.path/"state.json"
        backup=w.path/"saved-state.json";p.rename(backup);p.symlink_to(backup)
        with self.assertRaises(clock.TimeBudgetError):w.check()
    def test_live_watch_keeps_deadline_after_worker_commits_a_terminal_state(self):
        c,h,*_=self.limited(3,quota=1)
        w=clock.BudgetWatch(c.store,h["evaluation_id"],h["binding"])
        p=w.path/"state.json";state=read(p);state["status"]="service_error"
        p.write_bytes(canonical(seal(state)))
        with patch.object(clock,"current_clock",return_value=self.now(1011.,111.)):
            self.assertEqual(w.check(),("timed_out","assessment_controller_deadline_reached"))
    def test_completed_watch_can_reuse_terminal_results_without_a_new_deadline(self):
        c,h,*_=self.limited(3,quota=1)
        with patch.object(clock,"current_clock",return_value=self.now(1001.,101.)):c.advance(h)
        w=clock.BudgetWatch(c.store,h["evaluation_id"],h["binding"])
        with patch.object(clock,"current_clock",side_effect=AssertionError("completed result renewed")):
            self.assertIsNone(w.check())
        p=w.path/"state.json";state=read(p);state["status"]="ready"
        p.write_bytes(canonical(seal(state)))
        with self.assertRaises(clock.TimeBudgetError):w.check()
if __name__=="__main__":unittest.main()
