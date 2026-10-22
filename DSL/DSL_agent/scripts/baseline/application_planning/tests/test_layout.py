"""Layout admission and request bindings; no compiler or paid provider calls."""
import copy,json,signal,threading,time,unittest
from unittest.mock import patch
from application_planning.layout import plan_model,check_plan,submit_planned,POLICY,source_authority
from application_policy import prepare as app_prepare
from benchmark_graph import digest
from application_component import Application
import test_application_component as fixtures
from hecate_python_env import WORK
from pathlib import Path
import tempfile

def square(n=32,*,profile="general-tensors-v1",shape=None):
    return dict(format="poseidon-model-graph-v2",id="layout_square",resource_profile=profile,
        inputs=[dict(name="x",shape=shape or [n])],constants={},
        nodes=[dict(id="s",op="square",inputs=["x"],attrs={},outputs=["v"])],
        outputs=[dict(name="out",value="v")])

def conv_pool():
    return dict(format="poseidon-model-graph-v2",id="layout_conv_pool",resource_profile="general-tensors-v1",
        inputs=[dict(name="x",shape=[1,1,8])],
        constants=dict(w=[[[.25,-.125,.0625]]],b=[.03125]),
        nodes=[dict(id="conv",op="conv1d",inputs=["x","w","b"],attrs=dict(stride=[1],padding=[1],dilation=[1],groups=1),outputs=["cv"]),
               dict(id="pool",op="avg_pool1d",inputs=["cv"],attrs=dict(kernel=[2],stride=[2],padding=[0],count_include_pad=True),outputs=["pv"])],
        outputs=[dict(name="out",value="pv")])

class LayoutTests(unittest.TestCase):
    def test_smaller_physical_io_for_three_sizes(self):
        for n in (17,32,65):
            model=square(n)
            plan=plan_model(model,options={"generation_guidance":"explicit-v10"})
            self.assertEqual(plan["status"],"prepared")
            request=check_plan(plan,model)
            row=plan["candidates"][plan["selection"]["index"]]
            self.assertEqual(row["resources"]["declared_io_ciphertexts"]["value"],2)
            self.assertGreaterEqual(request["layout"]["input_slot_period"],n)
            self.assertTrue(all(v is False for v in plan["acceptance"].values()))

    def test_explicit_options_and_old_request_are_unchanged(self):
        model=square(8)
        options=dict(chunk_period=4,generation_guidance="explicit-v10")
        previous=app_prepare(model,"poseidon_cpu","numerical",options)[0]
        before=copy.deepcopy(model);copy_options=copy.deepcopy(options)
        plan=plan_model(model,options=options)
        self.assertEqual(plan["candidate_specs"],[options])
        self.assertEqual(check_plan(plan,model),previous)
        self.assertEqual(model,before);self.assertEqual(options,copy_options)

    def test_explicit_none_preserves_existing_default(self):
        model=square(8);options=dict(chunk_period=None)
        plan=plan_model(model,options=options)
        self.assertEqual(plan["candidate_specs"],[options])
        self.assertEqual(check_plan(plan,model),app_prepare(model,"poseidon_cpu","numerical",options)[0])

    def test_normal_application_default_guidance_matches_selected_request(self):
        model=square(17);plan=plan_model(model)
        request=check_plan(plan,model)
        self.assertEqual(request,app_prepare(model,"poseidon_cpu","numerical",plan["selection"]["options"])[0])

    def test_costs_do_not_claim_runtime_memory_or_depth(self):
        plan=plan_model(conv_pool())
        self.assertEqual(plan["status"],"prepared")
        row=plan["candidates"][plan["selection"]["index"]]
        for name in ("peak_memory_bytes","execution_seconds","multiplicative_depth","intermediate_ciphertexts"):
            self.assertEqual(row["resources"][name]["certainty"],"unknown")
            self.assertIsNone(row["resources"][name]["value"])
        self.assertEqual(row["resources"]["logical_public_array_bytes"]["value"],
                         8*row["resources"]["public_registry_elements"]["value"])
        for name in ("compiler","encrypted","agent"):self.assertEqual(row["stages"][name]["status"],"unverified")

    def test_rank32_and_4096_tail_declarations(self):
        for model in (square(32,shape=[1]*31+[32]),square(4096),square(257)):
            plan=plan_model(model);request=check_plan(plan,model)
            self.assertEqual(request["model"],model)
            self.assertLessEqual(len(request["layout"]["inputs"]),16)
            self.assertEqual(len(request["layout"]["output_selectors"]),model["inputs"][0]["shape"][-1])

    def test_graph_and_node_names_do_not_select_layout(self):
        first=square(17);second=copy.deepcopy(first)
        second["id"]="unrelated";second["inputs"][0]["name"]="z"
        second["nodes"][0].update(id="changed",inputs=["z"],outputs=["z_value"])
        second["outputs"][0].update(name="answer",value="z_value")
        a=plan_model(first);b=plan_model(second)
        self.assertEqual(a["selection"]["options"],b["selection"]["options"])
        self.assertEqual(a["selection"]["score"],b["selection"]["score"])
        self.assertNotEqual(a["model_sha256"],b["model_sha256"])

    def test_duplicate_requests_are_scored_once(self):
        plan=plan_model(square(4))
        ids=[r["request_id"] for r in plan["candidates"] if r["status"]=="admitted"]
        self.assertEqual(len(ids),len(set(ids)))
        for row in plan["candidates"]:
            if row["status"]=="duplicate":
                self.assertEqual(row["request_id"],plan["candidates"][row["same_as"]]["request_id"])

    def test_wrong_types_profiles_backend_and_unknown_options_rejected(self):
        for model,options,backend in (
            (square(),{"unexpected":"secret"},"poseidon_cpu"),
            (square(),{"compiler_configuration":"seal-cpu-eva-w45-v1"},"poseidon_cpu"),
            (square(),None,"poseidon_cuda"),
            (dict(square(),resource_profile="not-configured"),None,"poseidon_cpu"),
            (dict(square(),inputs=[dict(name="x",shape=[True])]),None,"poseidon_cpu")):
            plan=plan_model(model,options=options,backend=backend)
            self.assertEqual(plan["status"],"rejected")
            self.assertIsNone(plan["selection"])
            self.assertNotIn("secret",json.dumps(plan["failure"]))

    def test_invalid_fixed_period_and_wide_packing_keep_rejection(self):
        for options in (dict(chunk_period=3),dict(chunk_period=True),
                        dict(chunk_period=512,capacity_packing_version="capacity-public-tensors-v7")):
            plan=plan_model(square(8),options=options)
            self.assertEqual(plan["status"],"rejected")
            self.assertEqual(plan["failure"]["code"],"no_admitted_layout")
            self.assertEqual(len(plan["candidate_specs"]),1)

    def test_candidate_budget_and_invalid_budget(self):
        plan=plan_model(square(17),limits=dict(wall_seconds=120,candidate_limit=1))
        self.assertEqual(len(plan["candidates"]),1)
        for limits in (dict(wall_seconds=0,candidate_limit=8),dict(wall_seconds=181,candidate_limit=8),
                       dict(wall_seconds=True,candidate_limit=8),dict(wall_seconds=1,candidate_limit=9)):
            self.assertEqual(plan_model(square(),limits=limits)["failure"]["code"],"planning_limits_rejected")

    def test_whole_plan_timeout_restores_alarm(self):
        handler=signal.getsignal(signal.SIGALRM)
        with patch("application_planning.layout.prepare_application",side_effect=lambda *_:time.sleep(2)):
            plan=plan_model(square(),limits=dict(wall_seconds=1,candidate_limit=1))
        self.assertEqual(plan["status"],"budget_exhausted")
        self.assertEqual(plan["failure"]["code"],"planning_budget_exhausted")
        self.assertIsNone(plan["selection"])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
        self.assertEqual(signal.getsignal(signal.SIGALRM),handler)

    def test_busy_worker_does_not_replace_existing_alarm(self):
        signal.setitimer(signal.ITIMER_REAL,30)
        try:
            plan=plan_model(square())
            self.assertEqual(plan["failure"]["code"],"planning_worker_busy")
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0],29)
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        result=[];worker=threading.Thread(target=lambda:result.append(plan_model(square())))
        worker.start();worker.join()
        self.assertEqual(result[0]["status"],"service_error")

    def test_re_signed_selected_cost_model_and_options_tampering_rejected(self):
        model=square();plan=plan_model(model)
        for mutate in (
            lambda p:p.update(model_sha256="0"*64),
            lambda p:p["selection"]["options"].update(chunk_period=256),
            lambda p:p["candidates"][p["selection"]["index"]]["resources"]["declared_io_ciphertexts"].update(value=0),
            lambda p:p["source_authority"].update(sha256="0"*64)):
            bad=copy.deepcopy(plan);mutate(bad)
            bad["binding"]=digest({k:v for k,v in bad.items() if k!="binding"})
            with self.assertRaises(ValueError):check_plan(bad,model)

    def test_existing_application_owns_submission_and_budget(self):
        model=square(17);plan=plan_model(model)
        with tempfile.TemporaryDirectory(dir=WORK/"results") as root:
            app=Application(root,fixtures.Backend())
            submitted=submit_planned(app,model,plan)
            payload=json.loads((Path(root)/submitted["task_id"]/"task.json").read_text())
            self.assertEqual(payload["request"]["request_id"],plan["selection"]["request_id"])
            self.assertEqual(payload["budget"]["generations"],4)
            self.assertEqual(app.status(submitted["task_id"])["status"],"queued")
            with self.assertRaises(ValueError):submit_planned(app,model,plan,options={})
            app.cancel(submitted["task_id"])
            self.assertEqual(app.status(submitted["task_id"])["status"],"cancelled")

if __name__=="__main__":unittest.main()
