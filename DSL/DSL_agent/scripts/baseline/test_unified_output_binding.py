"""Output bindings and feedback controls; mocks are not encrypted execution."""
import copy,json,tempfile,unittest
from pathlib import Path
from contextlib import ExitStack
from unittest import mock
from poseidon_artifact_gate import DuplicateFinalOutput,check_final_output_bindings
from application_diagnostics import make,checked,render,attach,repeat_key
from test_unified_construction_guidance import request,candidate
from component_contract import reconstruct_request
from unified_output_binding_guidance import VERSION,RULE,repair_hint
from public_feedback_contract import public_feedback
class OutputBindingControls(unittest.TestCase):
    def test_distinct_outputs_keep_declared_order(self):
        for values in (["a"],["a","b"],["b","a","c"]):
            with self.subTest(values=values):
                before=copy.deepcopy(values);check_final_output_bindings(values);self.assertEqual(values,before)
    def test_first_duplicate_witness(self):
        with self.assertRaises(DuplicateFinalOutput) as caught:
            check_final_output_bindings(["private-a","private-b","private-a","private-b"])
        e=caught.exception;self.assertEqual((e.previous_index,e.output_index,e.output_count),(0,2,4))
        d=checked(e.structured_diagnostic())
        self.assertEqual(d["location"],dict(kind="output",index=2))
        self.assertEqual(d["context"],dict(previous_output_index=0,output_count=4))
        self.assertIsNone(repeat_key(d))
    def test_nonadjacent_alias_rejected(self):
        with self.assertRaises(DuplicateFinalOutput):check_final_output_bindings(["a","b","c","b"])
    def test_controlled_feedback_does_not_leak_value_ids(self):
        try:check_final_output_bindings(["secret-path","secret-path"])
        except DuplicateFinalOutput as e:public=public_feedback(render(e.structured_diagnostic()))
        self.assertEqual(public["layer"],"artifact_gate")
        self.assertIn("distinct Value IDs",public["diagnostic"])
        self.assertNotIn("secret-path",json.dumps(public))
    def test_witness_bounds_and_types(self):
        pairs=[(0,0,2),(1,0,2),(0,2,2),(-1,1,2),(0,1,8193),(True,1,2),(0,1.0,2)]
        for first,index,count in pairs:
            with self.subTest(values=(first,index,count)),self.assertRaises(ValueError):
                make("artifact_duplicate_output",location=dict(kind="output",index=index),
                     context=dict(previous_output_index=first,output_count=count))
    def test_diagnostic_cannot_forge_repeat_flag(self):
        d=DuplicateFinalOutput(0,1,2).structured_diagnostic();d["repeat_key_available"]=True
        with self.assertRaises(ValueError):checked(d)
    def test_bad_witness_fields_rejected(self):
        for context in ({"previous_output_index":0},{"previous_output_index":0,"output_count":2,"value":"private"}):
            with self.subTest(context=context),self.assertRaises(ValueError):
                make("artifact_duplicate_output",location=dict(kind="output",index=1),context=context)
    def test_legacy_request_reconstructs_unchanged(self):
        old=request(version="explicit-v13");self.assertEqual(reconstruct_request(old),old)
        new=request(version=VERSION);self.assertEqual(reconstruct_request(new),new)
        self.assertNotEqual(old["request_id"],new["request_id"])
        a,b=copy.deepcopy(old),copy.deepcopy(new)
        for x in (a,b):x.pop("generation_guidance");x.pop("request_id")
        self.assertEqual(a,b)
    def test_guidance_binds_physical_output_count(self):
        r=request(version=VERSION)
        g=r["generation_guidance"]["poseidon_expression_rules"]["final_output_binding"]
        self.assertEqual(g["physical_outputs"],r["layout"]["output_ciphertexts"])
        self.assertEqual(g["rule"],RULE);self.assertNotIn("final_output_binding",request()["generation_guidance"]["poseidon_expression_rules"])
    def test_repair_hint_inherits_static_counts_and_output_rule(self):
        r=request(version=VERSION);source=r["generation_guidance"]["entry_header"]+"\n    return "+r["generation_guidance"]["ordered_parameters"][0]+"\n"
        self.assertIn(RULE,repair_hint(r,source,stage="artifact_gate"))
        self.assertIn("Admitted rotation",repair_hint(r,source,stage="static_check"))
        self.assertEqual(repair_hint(request(),source,stage="artifact_gate"),"")
    def test_application_feedback_uses_typed_witness(self):
        r=request(version=VERSION);c=candidate(r,r["generation_guidance"]["entry_header"]+"\n    return zero_ct\n")
        d=DuplicateFinalOutput(0,1,2).structured_diagnostic()
        result=attach(dict(status="failed",structured_diagnostic=d,
            failure=dict(layer="artifact_gate"),public_feedback=dict(status="failed",layer="artifact_gate",
            category="pipeline_unclassified",diagnostic="private path and reference")),candidate=c,request=r)
        self.assertEqual(result["failure"]["code"],"compiled_duplicate_final_output")
        self.assertEqual(result["failure"]["retry_policy"],"repair_within_budget")
        self.assertIn(RULE,result["public_feedback"]["diagnostic"])
        self.assertNotIn("private path",result["public_feedback"]["diagnostic"])
    def test_cli_version_opt_in(self):
        from run_candidate import parse_args
        self.assertEqual(parse_args(["--case","x.json","--self-test","--unified-guidance",VERSION]).unified_guidance,VERSION)

class OutputPlanControls(unittest.TestCase):
    from test_poseidon_runtime_backend import RuntimeGateTests as Fixture
    def setUp(self):
        self.Fixture.setUp(self)
        from benchmark_suite import Builder
        from component_contract import prepare_task
        b=Builder([(4,)])
        self.request=prepare_task(b.finish("input0",b.node("negate",["input0"])))
        assert self.request["layout"]["output_ciphertexts"]==2
    tearDown=Fixture.tearDown
    value=Fixture.value
    run_gate=Fixture.run_gate
    def test_real_artifact_structure_duplicate_rejected(self):
        self.plan["final_outputs"]=["1","1"]
        with self.assertRaises(DuplicateFinalOutput) as caught:self.run_gate()
        self.assertEqual(caught.exception.structured_diagnostic()["context"],
                         dict(previous_output_index=0,output_count=2))
    def test_distinct_declared_values_pass_new_gate_identity(self):
        self.plan["final_outputs"]=["0","1"]
        q=self.run_gate()
        self.assertEqual(q["format"],"poseidon-artifact-gate-v2")
        self.assertEqual(q["final_output_binding"],"distinct-value-ids-v1")
    def test_undefined_output_still_rejected_before_alias_witness(self):
        self.plan["final_outputs"]=["unknown","unknown"]
        with self.assertRaisesRegex(ValueError,"Final outputs") as caught:self.run_gate()
        self.assertNotIsInstance(caught.exception,DuplicateFinalOutput)

class OutputAdapterControls(unittest.TestCase):
    def test_duplicate_rejected_before_any_execution_or_resource_analysis(self):
        from validation_adapter import ValidationContext,CandidateValidationAdapter
        with tempfile.TemporaryDirectory() as directory:
            result=Path(directory);calls=[]
            r=dict(request_id="fixture",task="fixture",compiler_configuration=dict(backend="poseidon_cpu"))
            def native(payload,out,argv,*args):
                calls.append(argv)
                (out/"trace-evidence.json").write_text(json.dumps(dict(frontend="real_Hecate")))
                return 0
            ctx=ValidationContext(result,r,dict(attempts=[]),{},dict(atol=1e-5,rtol=1e-4),45,
                result/"keys",False,native,lambda:None)
            with ExitStack() as stack:
                for name,value in (("validation_adapter.validate_candidate",{}),
                    ("validation_adapter.trace_payload_json","{}"),
                    ("poseidon_runtime_compile.prepare_entrypoint",{}),
                    ("poseidon_runtime_compile.command",["compiler"])):
                    stack.enter_context(mock.patch(name,return_value=value))
                stack.enter_context(mock.patch("poseidon_validation.gate",side_effect=DuplicateFinalOutput(0,1,2)))
                pre=stack.enter_context(mock.patch("application_planning.compiled_resources.execution_storage_precheck"))
                prepare=stack.enter_context(mock.patch("poseidon_validation.prepare_inputs"))
                execute=stack.enter_context(mock.patch("poseidon_validation.execute"))
                feedback=CandidateValidationAdapter(ctx).evaluate(json.dumps(dict(hecate_source="x=1")),0)
            self.assertEqual(len(calls),2);pre.assert_not_called();prepare.assert_not_called();execute.assert_not_called()
            item=ctx.report["attempts"][0];self.assertTrue(item["compiled"]);self.assertFalse(item["executed"])
            self.assertEqual(feedback["layer"],"artifact_gate")
            self.assertEqual(item["structured_diagnostic"]["reason_code"],"artifact_duplicate_output")
