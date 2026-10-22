"""Planner behavior at shared physical-output and logical-output boundaries."""
import copy,unittest,sys
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmark_graph import CAPACITY_FORMAT
from component_contract import prepare_task
from application_planning.layout import candidate_options,plan_model,check_plan,submit_planned

def model(distinct=False):
    nodes=[
        dict(id="sq",op="square",inputs=["x"],attrs={},outputs=["a"]),
        dict(id="neg",op="negate",inputs=["x"],attrs={},outputs=["b"]),
        dict(id="sum",op="add",inputs=["x","c"],attrs={},outputs=["c_out"]),
        dict(id="product",op="multiply",inputs=["x","c"],attrs={},outputs=["d"])]
    values=["a","b","c_out","d"] if distinct else ["a"]*4
    return dict(format=CAPACITY_FORMAT,id="output_contract",resource_profile="general-tensors-v1",
        inputs=[dict(name="x",shape=[20])],constants={"c":.25},nodes=nodes,
        outputs=[dict(name="out"+str(i),value=v)for i,v in enumerate(values)])

OPTIONS=dict(output_binding_policy="exact-values-v1",generation_guidance="explicit-v15",
             public_weight_policy="descriptors-math-roles-v2",capacity_packing_version="capacity-public-tensors-v9")

class OutputAliasPlanningTests(unittest.TestCase):
    def test_lowest_feasible_period_counts_shared_value_once(self):
        periods=[x.get("chunk_period")for x in candidate_options(model(),OPTIONS,8)]
        self.assertIn(4,periods)
        q=prepare_task(model(),{**OPTIONS,"chunk_period":4})
        self.assertEqual(q["layout"]["output_ciphertexts"],5)
        self.assertEqual(q["layout"]["output_shape"],[80])
    def test_different_values_with_same_shape_are_separate(self):
        periods=[x.get("chunk_period")for x in candidate_options(model(True),OPTIONS,8)]
        self.assertNotIn(4,periods)
        self.assertIn(8,periods)
        q=prepare_task(model(True),{**OPTIONS,"chunk_period":8})
        self.assertEqual(q["layout"]["output_ciphertexts"],12)
    def test_legacy_output_cost_retained(self):
        old={k:v for k,v in OPTIONS.items()if k!="output_binding_policy"}
        periods=[x.get("chunk_period")for x in candidate_options(model(),old,8)]
        self.assertNotIn(4,periods)
        self.assertIn(8,periods)
    def test_requested_period_is_not_changed(self):
        selected={**OPTIONS,"chunk_period":4}
        self.assertEqual(candidate_options(model(),selected,8),[selected])
    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            candidate_options(model(),{**OPTIONS,"output_binding_policy":"any-equivalent-value"},8)
    def test_model_and_options_not_mutated(self):
        m=model();o=copy.deepcopy(OPTIONS);original=copy.deepcopy((m,o))
        candidate_options(m,o,8)
        self.assertEqual((m,o),original)
    def test_logical_capacity_still_rejects_repeated_results(self):
        m=model();m["inputs"][0]["shape"]=[1025]
        result=plan_model(m,options=OPTIONS)
        self.assertEqual(result["status"],"rejected")
        self.assertEqual(result["failure"]["layer"],"input")
    def test_plan_submission_carries_same_canonical_request(self):
        m=model();result=plan_model(m,options=OPTIONS)
        self.assertEqual(result["status"],"prepared")
        q=check_plan(result,m)
        selected=result["selection"]["options"]
        self.assertEqual(q,prepare_task(m,selected))
        self.assertIn(4,[s.get("chunk_period")for s in result["candidate_specs"]])
        self.assertEqual(q["layout"]["output_ciphertexts"],1)
        self.assertEqual(q["layout"]["output_shape"],[80])
        host=mock.Mock()
        host.submit.return_value="submitted"
        submitted=submit_planned(host,m,result,validation_level="numerical")
        self.assertEqual(submitted["task_id"],"submitted")
        self.assertEqual(submitted["request_id"],q["request_id"])
        self.assertEqual(submitted["layout_plan_binding"],result["binding"])
        self.assertEqual(host.submit.call_args.kwargs["options"],selected)
    def test_plan_still_reports_execution_as_unverified(self):
        result=plan_model(model(),options=OPTIONS)
        self.assertEqual(result["acceptance"],dict(compiled=False,encrypted_execution=False,numerically_validated=False))
        row=next(r for r in result["candidates"]if r["index"]==result["selection"]["index"])
        self.assertEqual(row["resources"]["peak_memory_bytes"]["certainty"],"unknown")
        self.assertEqual(row["resources"]["execution_seconds"]["certainty"],"unknown")

if __name__=="__main__":unittest.main()
