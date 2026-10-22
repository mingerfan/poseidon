"""Construction accounting and repair controls; no API, compilation or FHE."""
import copy,json,unittest
from benchmark_suite import Builder
from benchmark_graph import CAPACITY_FORMAT,digest
from component_contract import prepare_task,validate_program,reconstruct_request,failure
from candidate_check_diagnostic import CandidateCheckError
from application_diagnostics import attach,from_result,make,repeat_key
from unified_graph_contract import validate_candidate,validate_request,static_repair_hint
from unified_construction_guidance import VERSION,resources,specification,cost_facts,repair_hint
from public_weight_context import CONTEXT_BYTES
from deepseek_provider import public_request as generation_context

def graph():
    b=Builder([(4,)])
    w=b.const([[.013*(1+i*4+j) for j in range(4)] for i in range(4)])
    return b.finish(b.node("linear",["input0",w,b.const([.01,.02,.03,.04])]))
def request(*,capacity=True,version=VERSION,public=False):
    g=graph();options=dict(generation_guidance=version)
    if capacity:g.update(format=CAPACITY_FORMAT,resource_profile="general-tensors-v1");options["chunk_period"]=4
    if public:
        from unified_public_contract import CONTRACT
        options["construction_profile"]=CONTRACT
    return prepare_task(g,options)
def candidate(r,s):return dict(schema=1,request_id=r["request_id"],hecate_source=s)
def expensive(r,kind="single",times=None):
    n=len(r["public_constants"])
    if kind=="single":
        definitions='@hc.func("c")\ndef helper(value):\n    return value+value\n'
        per_call=n+2
    elif kind=="nested":
        definitions=('@hc.func("c")\ndef inner(value):\n    return value+value\n'
            '@hc.func("c")\ndef helper(value):\n    part=inner(value)\n    return part*part\n')
        per_call=2*n+4
    else:
        definitions=('@hc.func("c")\ndef helper(value):\n    pair=np.array([value,value],dtype=object)\n'
            '    return pair[0]+pair[1]\n')
        per_call=n+6
    times=1024//per_call+1 if times is None else times
    assert 1<=times<=128
    input_name=r["generation_guidance"]["ordered_parameters"][0]
    return definitions+r["generation_guidance"]["entry_header"]+(
        "\n    value="+input_name+"\n    for index in range("+str(times)+"):\n        value=helper(value)\n    return value\n"),times,per_call
def witness(r,s):
    try:validate_candidate(candidate(r,s),r)
    except CandidateCheckError as error:
        return from_result(dict(status="failed",failure=failure("static_check",error)))
    raise AssertionError("Expected authoritative rejection")

class ConstructionGuidanceTests(unittest.TestCase):
    def test_version_opt_in_and_request_reconstruction(self):
        a,b=request(version="explicit-v12"),request()
        self.assertNotEqual(a["request_id"],b["request_id"])
        self.assertEqual(reconstruct_request(a),a);self.assertEqual(reconstruct_request(b),b)
        left=copy.deepcopy(a);right=copy.deepcopy(b)
        for v in (left,right):v.pop("request_id");v.pop("generation_guidance")
        self.assertEqual(left,right)

    def test_limits_read_authoritative_constants_without_expansion(self):
        r=request();v=resources(r)
        self.assertEqual(v["golden_work_limit"],1024)
        self.assertEqual(v["source_bytes"],65536)
        self.assertEqual(v["native"]["module_work_limit"],4096)
        self.assertEqual(v["native"]["function_initial_work"],len(r["public_constants"]))
        self.assertEqual(v["native"]["helper_call_surcharge"],1)
        self.assertIn("including unused bindings",v["native"]["rule"])

    def test_three_cost_contexts_match_actual_rejection(self):
        r=request();n=len(r["public_constants"])
        for kind in ("single","nested","array"):
            with self.subTest(kind=kind):
                source,count,per_call=expensive(r,kind)
                d=witness(r,source);self.assertEqual(d["reason_code"],"expanded_work_budget")
                v=cost_facts(r,source,d)
                self.assertEqual(v["observed"],n+count*per_call)
                self.assertEqual(v["direct_native_helper_calls"],count)
                self.assertEqual(v["direct_native_call_work"],count*per_call)
                self.assertEqual(v["direct_registry_work"],count*n)
                self.assertFalse(v["acceptance_granted"]);self.assertFalse(v["candidate_executed"])
                self.assertIsNone(repeat_key(d))

    def test_reused_value_avoids_repeating_native_call(self):
        r=request();source,_,_=expensive(r,times=1)
        source=source.replace("    value="+r["generation_guidance"]["ordered_parameters"][0]+"\n    for index in range(1):\n        value=helper(value)",
            "    saved=helper("+r["generation_guidance"]["ordered_parameters"][0]+")\n    value=saved+saved")
        checked=validate_candidate(candidate(r,source),r)
        n=len(r["public_constants"])
        self.assertEqual(checked["functions"]["golden"]["expanded_cost"],2*n+3)
        self.assertIsNone(cost_facts(r,source))

    def test_work_budget_remains_a_rejection(self):
        r=request();source,_,_=expensive(r)
        d=witness(r,source);advice=repair_hint(r,source,witness=d)
        self.assertIn("Static work accounting:",advice)
        with self.assertRaises(CandidateCheckError):validate_candidate(candidate(r,source),r)

    def test_actual_helper_locations_without_source_or_identifier_echo(self):
        r=request();source,_,_=expensive(r,"nested")
        value=cost_facts(r,source,witness(r,source))
        self.assertEqual(value["largest_calls"][0]["location"]["line"],5)
        text=repair_hint(r,source)
        self.assertIn("Helper at line 5",text)
        self.assertNotIn("def helper",text);self.assertNotIn("def inner",text)

    def test_wrong_count_or_location_cannot_generate_cost_claim(self):
        r=request();source,_,_=expensive(r);d=witness(r,source)
        for field in ("count","location"):
            bad=copy.deepcopy(d)
            if field=="count":bad["context"]["observed"]+=1
            else:bad["location"]["line"]+=1
            self.assertIsNone(cost_facts(r,source,bad))

    def test_other_witness_cannot_generate_work_facts(self):
        r=request();source,_,_=expensive(r)
        other=make("native_reserved_binding",location=dict(kind="source",line=1,column=0))
        self.assertIsNone(cost_facts(r,source,other))

    def test_rotation_steps_come_from_frozen_layout(self):
        for capacity in (False,True):
            r=request(capacity=capacity);v=resources(r)
            self.assertEqual(v["rotation"]["admitted_steps"],[1,2])
            bad=r["generation_guidance"]["entry_header"]+"\n    return "+r["generation_guidance"]["ordered_parameters"][0]+".rotate(-1)\n"
            d=witness(r,bad)
            self.assertEqual(d["reason_code"],"rotation_step_not_allowed")
            v=attach(dict(status="failed",failure=d),candidate=candidate(r,bad),request=r)
            self.assertIn("Admitted rotation steps: [1, 2]",v["public_feedback"]["diagnostic"])

    def test_zero_displacement_is_reuse_not_new_key(self):
        r=request();text=json.dumps(resources(r))
        self.assertIn("zero displacement reuses the value",text)
        self.assertNotIn(0,resources(r)["rotation"]["admitted_steps"])

    def test_public_contract_has_distinct_accounting(self):
        r=request(public=True);v=resources(r)
        self.assertEqual(v["contract"],"public");self.assertNotIn("native",v)
        self.assertEqual(v["public"]["interpretation_step_limit"],4096)
        self.assertEqual(v["public"]["emitted_cipher_operation_limit"],256)
        self.assertIsNone(cost_facts(r,"invalid native program"))
        self.assertEqual(reconstruct_request(r),r)

    def test_legacy_nonchunk_layout_has_correct_element_count(self):
        for version in ("explicit-v12",VERSION):
            r=request(capacity=False,version=version)
            rows=r["generation_guidance"]["poseidon_expression_rules"]["tensor_axis_annotations"]["input_chunks"]
            self.assertEqual(rows[0]["valid_elements"],4)
            self.assertEqual(rows[0]["logical_offset"],0)
            self.assertNotIn("elements",r["layout"]["inputs"][0])

    def test_context_retains_required_constraints_under_budget(self):
        r=request()
        context=r["generation_guidance"]["poseidon_expression_rules"]
        self.assertLessEqual(len(json.dumps(context,sort_keys=True).encode()),32768)
        self.assertLessEqual(len(json.dumps(generation_context(r),sort_keys=True,separators=(",",":")).encode()),CONTEXT_BYTES)
        self.assertIn("construction_resources",context)
        self.assertEqual(context["version"],VERSION)

    def test_rehashed_constraint_tampering_is_rejected(self):
        r=request()
        r["generation_guidance"]["poseidon_expression_rules"]["construction_resources"]["golden_work_limit"]=2048
        r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
        with self.assertRaises(ValueError):reconstruct_request(r)

    def test_host_and_integrity_failure_take_priority(self):
        r=request();source,_,_=expensive(r)
        for code in ("environment_failure","integrity_rejected"):
            value=attach(dict(status="failed",failure=make(code)),candidate=candidate(r,source),request=r)
            self.assertNotIn("Static work accounting:",value["public_feedback"]["diagnostic"])
            self.assertNotIn("Binding and shape guidance",value["public_feedback"]["diagnostic"])

    def test_private_result_fields_and_tool_text_do_not_leak(self):
        r=request();source,_,_=expensive(r);d=witness(r,source)
        value=attach(dict(status="failed",failure=d,private_inputs="PRIVATE_ARRAY",
            reference="PRIVATE_REFERENCE",diagnostic="/home/private PRIVATE_KEY"),candidate=candidate(r,source),request=r)
        text=json.dumps(value["public_feedback"])
        for secret in ("PRIVATE_ARRAY","PRIVATE_REFERENCE","PRIVATE_KEY","/home/private"):self.assertNotIn(secret,text)
        self.assertIn("Static work accounting:",text)
        self.assertLessEqual(len(value["public_feedback"]["diagnostic"].encode()),8000)

    def test_malformed_oversized_invalid_and_recursive_sources_are_data_only(self):
        r=request()
        for source in ("not python (","x"*65537,'@hc.func("c")\ndef helper(v):\n    return helper(v)\n'+
            r["generation_guidance"]["entry_header"]+"\n    return helper(x)\n"):
            self.assertIsNone(cost_facts(r,source))
        self.assertEqual(repair_hint(r,"x"*65537),"")
        self.assertEqual(repair_hint(r,"not python ("),"")

    def test_model_id_does_not_dispatch_guidance(self):
        r=request();original=resources(r)
        r["model"]["id"]="unseen_model_family"
        self.assertEqual(resources(r),original)

    def test_old_guidance_does_not_receive_new_hints(self):
        r=request(version="explicit-v12");source,_,_=expensive(r)
        self.assertEqual(repair_hint(r,source),"")
        self.assertNotIn("construction_resources",r["generation_guidance"]["poseidon_expression_rules"])

    def test_cli_accepts_explicit_v13(self):
        from run_candidate import parse_args,forward_options
        import shlex
        args=parse_args(["--case","scripts/baseline/cases/unified-two-input-two-output.json","--prepare","--unified-guidance",VERSION])
        self.assertIn(VERSION,shlex.split(forward_options(args)))

    def test_both_sandbox_policies_bind_the_new_module(self):
        from sandbox_policy import MODULES as legacy
        from poseidon_sandbox_policy import MODULES as current
        for v in (legacy,current):self.assertEqual(v.count("unified_construction_guidance.py"),1)

if __name__=="__main__":unittest.main(verbosity=2)
