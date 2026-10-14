"""Descriptor/admission and isolated transport fixtures; no live API or FHE."""
import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmark_graph import canonical,digest
from component_contract import prepare_task,reconstruct_request,runner_options,validate_program
from candidate_contract import request_file_bytes,trace_payload_json,strict_json
from public_weight_context import (POLICY,FORMAT,binding,project,verify_binding,verify_context,
                                   request_digest,read_payload,read_request,TRUSTED_BYTES)
from public_weight_cases import models,period
from deepseek_provider import public_request,DeepSeekProvider,Config,HTTPSTransport,ProviderError
from unified_graph_lowering import candidate_source
from test_deepseek_provider import FixtureTransport,completion

def request(model):
    return prepare_task(model,dict(chunk_period=period(model),generation_guidance="explicit-v10",
                                   public_weight_policy=POLICY))
class WeightContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models=models()
        cls.requests=[request(model) for model in cls.models]
    def test_10k_request_has_compact_context_and_complete_trusted_values(self):
        q=self.requests[-1];context=public_request(q)
        self.assertGreater(len(canonical(q["model"])),131072)
        self.assertEqual(q["public_weight_binding"]["model_constants"]["c0"]["elements"],10000)
        self.assertLessEqual(len(canonical(context)),131072)
        self.assertGreater(len(request_file_bytes(q)),131072)
        self.assertLess(len(request_file_bytes(q)),TRUSTED_BYTES)
        self.assertEqual(verify_context(context),context)
        flat=[x for row in q["model"]["constants"]["c0"] for x in row]
        self.assertEqual(len(set(flat)),10000)
        self.assertNotIn(repr(flat[0]),canonical(context).decode())
    def test_weight_names_are_not_confused_with_prebound_dsl_registry(self):
        context=public_request(self.requests[0])
        self.assertIn("c0",context["model"]["constants"])
        self.assertNotIn("c0",context["public_constants"])
        self.assertIn("Only names listed in public_constants",context["rules"])
        self.assertIn("constant_origins",context["rules"])
        self.assertNotIn("Model constants and public_constants are prebound",context["rules"])
    def test_round_trip_reconstructs_full_values_not_descriptors(self):
        for q in self.requests:
            self.assertEqual(reconstruct_request(read_request(request_file_bytes(q).decode())),q)
            self.assertIn("--public-weights",runner_options(q))
            self.assertEqual(verify_binding(q),project(q))
    def test_original_weight_change_changes_hash_and_stale_binding_is_rejected(self):
        q=copy.deepcopy(self.requests[0]);changed=copy.deepcopy(q["model"])
        changed["constants"]["c0"][0][0]+=0.001
        self.assertNotEqual(request(changed)["request_id"],q["request_id"])
        q["model"]=changed
        with self.assertRaisesRegex(ValueError,"binding changed"):verify_binding(q)
    def test_changed_prebound_registry_is_rejected(self):
        q=copy.deepcopy(self.requests[0])
        name=next(iter(q["public_constants"]))
        q["public_constants"][name]=0.0
        with self.assertRaisesRegex(ValueError,"binding changed"):verify_binding(q)
    def test_descriptors_commit_to_shape_and_exact_content(self):
        q=self.requests[0]
        d=q["public_weight_binding"]["model_constants"]["c0"]
        self.assertEqual(d["shape"],[3,4])
        self.assertEqual(d["content_sha256"],digest(q["model"]["constants"]["c0"]))
        bad=public_request(q);bad["public_weight_binding"]["model_constants"]["c0"]["elements"]=7
        bad["model"]["constants"]=copy.deepcopy(bad["public_weight_binding"]["model_constants"])
        bad["request_id"]=request_digest(bad)
        with self.assertRaisesRegex(ValueError,"descriptor shape/hash"):verify_context(bad)
    def test_provider_context_never_contains_full_weight_values(self):
        q=self.requests[-1];source,_=candidate_source(q)
        raw=json.dumps(dict(schema=1,request_id=q["request_id"],hecate_source=source))
        transport=FixtureTransport([completion(raw)])
        provider=DeepSeekProvider(Config(),transport=transport)
        provider.generate(q,[])
        sent=json.loads(transport.sent[0][0]["messages"][1]["content"])
        self.assertEqual(sent,public_request(q))
        self.assertTrue(all(type(v) is dict for v in sent["model"]["constants"].values()))
        self.assertTrue(all(type(v) is dict for v in sent["public_constants"].values()))
        self.assertEqual(provider.agent_calls,0)
    def test_approval_matches_descriptor_hash_and_rejects_changed_weight(self):
        q=self.requests[0];context=public_request(q);c=Config()
        transport=HTTPSTransport(api_key="synthetic-only",approved_request_id=q["request_id"],enabled=True,approved_config=c)
        body=canonical(dict(c.settings(),messages=[dict(role="system",content=__import__("deepseek_provider").SYSTEM),
                             dict(role="user",content=canonical(context).decode())]))
        transport.validate(body,c.timeout_seconds)
        changed=copy.deepcopy(context);changed["public_weight_binding"]["model_constants"]["c0"]["content_sha256"]="f"*64
        changed["model"]["constants"]=copy.deepcopy(changed["public_weight_binding"]["model_constants"])
        changed["request_id"]=request_digest(changed)
        payload=json.loads(body);payload["messages"][1]["content"]=canonical(changed).decode()
        with self.assertRaisesRegex(ProviderError,"approval"):transport.validate(canonical(payload),c.timeout_seconds)
    def test_large_trace_payload_retains_128k_candidate_limit(self):
        q=self.requests[-1];source,_=candidate_source(q)
        candidate=dict(schema=1,request_id=q["request_id"],hecate_source=source)
        raw=trace_payload_json(q,candidate)
        self.assertEqual(read_payload(raw)["request"]["request_id"],q["request_id"])
        bad=dict(candidate,hecate_source="x"*131073)
        with self.assertRaisesRegex(ValueError,"candidate response budget"):
            read_payload(json.dumps(dict(request=q,candidate=bad)))
    def test_old_requests_keep_exact_hash_and_small_payload_policy(self):
        q=prepare_task(self.models[0],dict(chunk_period=4,generation_guidance="explicit-v10"))
        self.assertNotIn("public_weight_binding",q)
        self.assertEqual(q["request_id"],digest({k:v for k,v in q.items() if k!="request_id"}))
        self.assertEqual(reconstruct_request(q),q)
        with self.assertRaisesRegex(ValueError,"policy budget"):
            read_payload(json.dumps(dict(request=q,candidate=dict(x="x"*131073))))
    def test_unknown_policy_profile_and_json_duplicates_are_rejected(self):
        from benchmark_suite import Builder
        b=Builder([(2,)]);small=b.finish(b.node("square",["input0"]))
        for opts in (dict(public_weight_policy=POLICY),dict(public_weight_policy="arbitrary")):
            with self.assertRaises(ValueError):prepare_task(small,opts)
        with self.assertRaisesRegex(ValueError,"Duplicate"):read_payload('{"request":{},"request":{},"candidate":{}}')
    def test_rule_sources_still_use_fixed_names_and_pass_static_contract(self):
        for q in self.requests:
            source,_=candidate_source(q)
            result=validate_program(dict(schema=1,request_id=q["request_id"],hecate_source=source),q)
            self.assertEqual(result["status"],"static_accepted_not_executed")


    def test_imported_npy_weight_records_reconstruct_decomposition(self):
        import io,tempfile,numpy as np
        from model_weight_package import load,FORMAT as WEIGHT_FORMAT,check_recorded_binding
        from model_decomposition import decompose,FORMAT as OPERATOR
        from candidate_bundle import check_provenance
        model=self.models[0]
        source=copy.deepcopy(model);source.pop("resource_profile");source["format"]=OPERATOR
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);weights={}
            unfilled=copy.deepcopy(source);unfilled["constants"]={}
            for name,value in source["constants"].items():
                raw=io.BytesIO();np.save(raw,np.asarray(value,dtype="<f8"),allow_pickle=False)
                payload=raw.getvalue();filename=name+".npy";(root/filename).write_bytes(payload)
                weights[name]=dict(file=filename,sha256=hashlib.sha256(payload).hexdigest(),
                                   bytes=len(payload),shape=list(np.shape(value)),dtype="<f8")
            path=root/"manifest.json";path.write_bytes(canonical(dict(format=WEIGHT_FORMAT,
                resource_profile="general-tensors-v1",model=unfilled,weights=weights)))
            original,profile,record=load(path)
            lowered,bound=decompose(original,resource_profile=profile)
            bound["weight_package"]=record;bound.pop("binding_sha256",None);bound["binding_sha256"]=digest(bound)
            provenance=dict(original=original,decomposition=bound,
                reference_report=dict(original_sha256=digest(original),model_sha256=digest(lowered),
                    reference_status="passed",approximation_accuracy_certified=False))
            self.assertEqual(check_provenance(provenance,lowered),bound)
            mutated=copy.deepcopy(record);mutated["weights"]["c0"]["shape"]=[1,12]
            mutated["binding_sha256"]=digest({k:v for k,v in mutated.items() if k!="binding_sha256"})
            with self.assertRaisesRegex(ValueError,"shape/dtype"):
                check_recorded_binding(mutated,original,profile)
            (root/"c0.npy").write_bytes(b"changed")
            with self.assertRaises(ValueError):load(path)

    def test_existing_capacity_wire_codec_keeps_old_request_hash(self):
        from general_tensor_cases import Builder
        b=Builder("legacy_vector_wire",[[256]])
        model=b.finish(b.node("multiply",["input0",b.constant(0.125)]))
        q=prepare_task(model,dict(chunk_period=256,generation_guidance="explicit-v10"))
        original=copy.deepcopy(q)
        context=public_request(q)
        self.assertEqual(q,original)
        self.assertEqual(context["request_id"],q["request_id"])
        self.assertLessEqual(len(canonical(context)),131072)
        self.assertEqual(reconstruct_request(copy.deepcopy(context)),q)
        transport=HTTPSTransport(api_key="synthetic-only",approved_request_id=q["request_id"],enabled=True)
        from deepseek_provider import SYSTEM
        config=Config()
        payload=canonical(dict(config.settings(),messages=[dict(role="system",content=SYSTEM),
                dict(role="user",content=canonical(context).decode())]))
        transport.validate(payload,config.timeout_seconds)

    def test_weight_manifest_rejects_duplicate_and_nonfinite_json_before_loading(self):
        import tempfile
        from model_weight_package import load
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"manifest.json"
            for raw in ('{"format":"x","format":"y"}','{"unexpected":NaN}','{"unexpected":1e400}'):
                p.write_text(raw)
                with self.assertRaisesRegex(ValueError,"Duplicate|Nonfinite"):load(p)

    def test_recorded_weight_shape_rejects_bool_and_float_dimensions(self):
        from model_weight_package import FORMAT as WEIGHT_FORMAT,check_recorded_binding
        from model_resources import identity
        original=self.models[0]
        original=copy.deepcopy(original);original.pop("resource_profile")
        for dims in ([3.0,4],[True,4]):
            value=dict(format=WEIGHT_FORMAT,resource_profile=identity("general-tensors-v1"),
                model_sha256=digest(original),manifest_sha256="0"*64,
                weights=dict(c0=dict(file="c0.npy",sha256="1"*64,bytes=128,shape=dims,dtype="<f8")))
            value["binding_sha256"]=digest(value)
            with self.assertRaisesRegex(ValueError,"shape/dtype"):
                check_recorded_binding(value,original,"general-tensors-v1")
