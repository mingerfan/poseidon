"""Versioned origin admission/transport controls; fixtures are not backend evidence."""
import copy,json,unittest
from unittest.mock import patch
from benchmark_graph import digest,canonical
from test_generation_model_origin import fixture
from generation_model_origin import build,verify,checked_public
from component_contract import prepare_task,reconstruct_request,validate_program
from unified_graph_contract import validate_request
from public_weight_context import project,verify_context,request_digest
from deepseek_provider import public_request,DeepSeekProvider,Config
from test_deepseek_provider import FixtureTransport,completion
from operator_arithmetic_graph import decompose
from unified_graph_lowering import candidate_source

def origin_request(capacity=False):
    p,r=fixture()
    if capacity:
        model,binding=decompose(p["original"],resource_profile="general-tensors-v1")
        p["decomposition"]=binding;p["reference_report"]["model_sha256"]=digest(model)
        opts=dict(generation_guidance="explicit-v20",chunk_period=4,
                  public_weight_policy="descriptors-math-roles-v2",
                  capacity_packing_version="capacity-public-tensors-v10")
        r=prepare_task(model,opts)
    else:
        opts=dict(generation_guidance="explicit-v20")
    context=build(p,r)
    opts["generation_guidance"]="explicit-v21"
    return p,prepare_task(r["model"],opts,model_origin=context)

def rebind(value):
    value["binding"]=digest({k:v for k,v in value.items()if k!="binding"})
    return value

class OriginWiringTests(unittest.TestCase):
    def test_new_request_roundtrip_and_host_verification(self):
        for capacity in (False,True):
            p,r=origin_request(capacity)
            self.assertIsInstance(validate_request(r),int)
            self.assertEqual(reconstruct_request(r),r)
            self.assertEqual(verify(r["generation_model_origin"],p,r),r["generation_model_origin"])
            self.assertEqual(r["request_id"],request_digest(r))
    def test_historical_versions_keep_original_request_bytes(self):
        p,r=fixture()
        self.assertNotIn("generation_model_origin",r)
        self.assertEqual(reconstruct_request(r),r)
        self.assertEqual(prepare_task(r["model"],dict(generation_guidance="explicit-v20")),r)
    def test_optional_new_version_without_origin(self):
        p,r=fixture()
        newer=prepare_task(r["model"],dict(generation_guidance="explicit-v21"))
        self.assertNotIn("generation_model_origin",newer)
        self.assertEqual(reconstruct_request(newer),newer)
        self.assertNotEqual(newer["request_id"],r["request_id"])
    def test_origin_is_not_an_unchecked_component_option(self):
        p,r=origin_request()
        with self.assertRaisesRegex(ValueError,"option fields"):
            prepare_task(r["model"],dict(model_origin=r["generation_model_origin"]))
    def test_origin_cannot_be_attached_to_old_guidance(self):
        p,r=origin_request()
        with self.assertRaisesRegex(ValueError,"explicit-v21"):
            prepare_task(r["model"],dict(generation_guidance="explicit-v20"),
                         model_origin=r["generation_model_origin"])
    def test_payload_and_origin_hash_tampering_rejected(self):
        p,r=origin_request()
        for path in ("model_hash","source_hash","reference_claim","extra"):
            value=copy.deepcopy(r["generation_model_origin"])
            if path=="model_hash":value["target_model_sha256"]="f"*64
            elif path=="source_hash":value["original_model_sha256"]="f"*64
            elif path=="reference_claim":value["reference_answers_included"]=True
            else:value["private_reference"]="DO NOT SEND"
            with self.subTest(path=path),self.assertRaises(ValueError):
                checked_public(value,r)
    def test_rehashed_wrong_target_or_source_map_rejected(self):
        p,r=origin_request()
        for variant in ("target","map"):
            value=copy.deepcopy(r["generation_model_origin"])
            if variant=="target":value["target_model_sha256"]="f"*64
            else:value["source_map"].pop(next(iter(value["source_map"])))
            rebind(value)
            with self.subTest(variant=variant),self.assertRaises(ValueError):checked_public(value,r)
    def test_structure_check_does_not_replace_host_source_truth(self):
        p,r=origin_request();value=copy.deepcopy(r["generation_model_origin"])
        value["nodes"][0]["op"]="tanh";rebind(value)
        self.assertEqual(checked_public(value,r),value)
        with self.assertRaisesRegex(ValueError,"changed"):verify(value,p,r)
    def test_duplicate_nodes_invalid_descriptors_and_forward_refs_rejected(self):
        p,r=origin_request()
        for variant in ("node","input","descriptor"):
            value=copy.deepcopy(r["generation_model_origin"])
            if variant=="node":value["nodes"].append(copy.deepcopy(value["nodes"][0]))
            elif variant=="input":value["nodes"][0]["inputs"]=["missing"]
            else:value["constants"]["c"]={"kind":"tensor","shape":[2],"elements":3,"content_sha256":"f"*64}
            rebind(value)
            with self.subTest(variant=variant),self.assertRaises(ValueError):checked_public(value,r)
    def test_provider_descriptors_keep_verified_origin_and_new_hash(self):
        p,r=origin_request(True);value=public_request(r)
        self.assertEqual(verify_context(value),value)
        self.assertEqual(value["generation_model_origin"],r["generation_model_origin"])
        self.assertTrue(all(type(x)is dict for x in value["public_constants"].values()))
        self.assertEqual(request_digest(value),r["request_id"])
        text=canonical(value).decode()
        for private in ("reference_report","test_inputs"):
            self.assertNotIn(private,text)
        self.assertNotIn("hecate_source",value)
        self.assertNotIn("hecate_source",value["generation_model_origin"])
    def test_fixture_provider_receives_origin_without_additional_call(self):
        p,r=origin_request(True);source,_=candidate_source(r)
        raw=json.dumps(dict(schema=1,request_id=r["request_id"],hecate_source=source))
        transport=FixtureTransport([completion(raw)])
        provider=DeepSeekProvider(Config(),transport=transport)
        provider.generate(r,[])
        self.assertEqual(len(transport.sent),1)
        sent=json.loads(transport.sent[0][0]["messages"][1]["content"])
        self.assertEqual(sent,public_request(r));self.assertEqual(provider.agent_calls,0)
    def test_candidate_static_contract_uses_fixed_lowered_graph(self):
        p,r=origin_request(True);source,_=candidate_source(r)
        result=validate_program(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        self.assertEqual(result["status"],"static_accepted_not_executed")
    def test_origin_feedback_inherits_audited_suffix_rule_without_mutation(self):
        p,r=origin_request();saved=copy.deepcopy(r)
        from unified_contract_scope_guidance import repair_hint
        with patch("unified_suffix_stage_guidance.repair_hint",return_value="audited hint")as previous:
            self.assertEqual(repair_hint(r,"def f(): pass"),"audited hint")
        transient=previous.call_args.args[0]
        self.assertEqual(transient["generation_guidance"]["version"],"explicit-v20")
        self.assertNotIn("generation_model_origin",transient)
        validate_request(transient);self.assertEqual(saved,r)
    def test_registered_readonly_modules_preserve_existing_sandboxes(self):
        import sandbox_policy,poseidon_sandbox_policy
        for name in ("generation_model_origin.py","unified_origin_guidance.py"):
            self.assertIn(name,sandbox_policy.MODULES)
            self.assertIn(name,poseidon_sandbox_policy.MODULES)
if __name__=="__main__":unittest.main()
