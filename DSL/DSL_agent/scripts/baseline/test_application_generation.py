"""Synthetic transport and backend fixtures only; zero network or FHE calls."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application_component import Application
from application_generation import capture, verify, FORMAT, METRIC_KEYS, CALL_KEYS
from benchmark_graph import digest, canonical
from component_contract import prepare_task
from deepseek_provider import Config, DeepSeekProvider
from test_application_component import Backend as FakeBackend, Provider as ReplayProvider
import test_application_identity as identity_fixture
from test_deepseek_provider import completion
from benchmark_suite import Builder

class SyntheticTransport:
    is_live = True  # Exercises the live control branch; post never opens a socket.
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    def post(self, body, timeout):
        self.requests.append(json.loads(body))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return completion(item)

def receipt_fixture(request, source, stage):
    raw = json.dumps(dict(schema=1, request_id=request["request_id"], hecate_source=source))
    provider = DeepSeekProvider(Config(), transport=SyntheticTransport([raw]))
    provider.generate(request, [])
    metrics = {k:v for k,v in provider.metrics().items() if k in METRIC_KEYS}
    metrics["calls"] = [{k:v for k,v in call.items() if k in CALL_KEYS} for call in metrics["calls"]]
    value = dict(format=FORMAT, request_id=request["request_id"],
                 candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
                 configuration=dict(provider="deepseek", model="deepseek-flash", reasoning_effort="high",
                    max_tokens=8192, max_repairs=3, api_timeout_seconds=120,
                    provider_retries=0, loopback_proxy_port=0, stream=False),
                 metrics=metrics, answers=[dict(index=0,raw=raw,feedback=dict(status="passed",layer="complete"))],
                 generation_identity_sha256=stage)
    value["binding"] = digest(value)
    return value

class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        fixture = identity_fixture.IdentityTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.identities = fixture.manifest()
        identities = self.identities
        class Backend(FakeBackend):
            receipt = None
            def identity_manifest(self):
                return copy.deepcopy(identities)
            def export(self, result, destination, level, *, generation_provenance=None):
                self.receipt = copy.deepcopy(generation_provenance)
                super().export(result, destination, level)
        self.backend = Backend()
        self.app = Application(self.tmp.name, self.backend)
        b = Builder([(2,)])
        self.model = b.finish(b.node("square", ["input0"]))
        self.request = prepare_task(self.model)
        self.source = '@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [x*x]\n'
    def provider(self, sources, **config):
        raw = [json.dumps(dict(schema=1, request_id=self.request["request_id"], hecate_source=s))
               if type(s) is str else s for s in sources]
        return DeepSeekProvider(Config(**config), transport=SyntheticTransport(raw))
    def test_component_attaches_exact_generation_receipt_and_reuses_it(self):
        provider = self.provider([self.source])
        task = self.app.submit(self.model)
        result = self.app.run(task, provider)
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["generation"]["agent_generated"])
        self.assertEqual(result["generation"]["http_attempts"], 1)
        value = self.backend.receipt
        self.assertEqual(verify(value, self.request, self.source)["generations"], 1)
        self.assertNotIn("transport_diagnostics", json.dumps(value))
        self.assertEqual(result["generation"]["usage"]["total_tokens"],130)
        self.assertEqual(result["generation"]["usage"]["reasoning_tokens"],10)
        self.assertIsNone(result["generation"]["usage"]["cost_usd"])
        folder=Path(self.tmp.name)/task
        metrics=self.app._load(folder/"provider-metrics.json")
        retained=self.app._load(folder/"loop.json")
        self.assertEqual(metrics["calls"][0]["usage"]["prompt_cache_hit_tokens"],20)
        self.assertEqual(retained["provider_metrics"],metrics)
        self.assertNotIn("transport_diagnostics",json.dumps(retained))
        reused = self.app.run(self.app.submit(self.model), ReplayProvider())
        self.assertTrue(reused["package"]["reused"])
        self.assertEqual(reused["generation"], result["generation"])
    def test_repairs_and_http_retries_are_separate(self):
        self.backend.failures = ["static_check"]
        changed = self.source.replace("x*x", "x*x+zero_ct")
        provider = self.provider([TimeoutError(),self.source,changed],provider_retries=1)
        with patch("deepseek_provider.time.sleep"):
            result = self.app.run(self.app.submit(self.model), provider)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["generation"]["generations"], 2)
        self.assertEqual(result["generation"]["http_attempts"], 3)
        self.assertEqual(result["generation"]["repairs_used"], 1)
        self.assertFalse(result["generation"]["first_attempt_passed"])
        summary=verify(self.backend.receipt, self.request, changed)
        self.assertEqual(summary["usage"]["responses_with_usage"],2)
        self.assertEqual(summary["usage"]["unresolved_http_attempts"],1)
        self.assertEqual(summary["usage"]["total_tokens"],260)
    def test_scripted_candidate_is_not_live_generation(self):
        result = self.app.run(self.app.submit(self.model), ReplayProvider())
        self.assertEqual(result["status"], "succeeded")
        self.assertFalse(result["generation"]["agent_generated"])
        self.assertIsNone(self.backend.receipt)
    def test_altered_executed_source_is_rejected_even_after_resigning(self):
        receipt = receipt_fixture(self.request,self.source,self.identities["generation"]["sha256"])
        changed = self.source.replace("x*x","x*x+zero_ct")
        receipt["candidate_sha256"] = hashlib.sha256(changed.encode()).hexdigest()
        receipt["binding"] = digest({k:v for k,v in receipt.items() if k!="binding"})
        with self.assertRaisesRegex(ValueError,"returned Agent answer"):
            verify(receipt,self.request,changed)
    def test_changed_request_and_stage_are_rejected(self):
        value = receipt_fixture(self.request,self.source,self.identities["generation"]["sha256"])
        with self.assertRaisesRegex(ValueError,"request identity"):
            verify(value,dict(self.request,request_id="f"*64),self.source)
        with self.assertRaisesRegex(ValueError,"stage mismatch"):
            verify(value,self.request,self.source,generation_identity_sha256="f"*64)
    def test_missing_or_reordered_ledger_is_rejected(self):
        value = receipt_fixture(self.request,self.source,self.identities["generation"]["sha256"])
        value["metrics"]["calls"][0]["index"] = 1
        value["binding"] = digest({k:v for k,v in value.items() if k!="binding"})
        with self.assertRaisesRegex(ValueError,"HTTP ledger"):
            verify(value,self.request,self.source)
    def test_private_feedback_and_transport_fields_are_rejected(self):
        value = receipt_fixture(self.request,self.source,self.identities["generation"]["sha256"])
        value["metrics"]["calls"][0]["private_reference"] = [1.23]
        value["binding"] = digest({k:v for k,v in value.items() if k!="binding"})
        with self.assertRaisesRegex(ValueError,"HTTP fields"):
            verify(value,self.request,self.source)
    def test_changed_retained_answer_stops_packaging(self):
        task = self.app.submit(self.model)
        qualify = self.backend.qualify
        def replace(q,c,level):
            result = qualify(q,c,level)
            self.app._write(Path(self.tmp.name)/task/"generation-0.json",dict(raw="{}"))
            return result
        self.backend.qualify = replace
        result = self.app.run(task,self.provider([self.source]))
        self.assertEqual(result["status"],"service_error")
        self.assertIsNone(result["package"])
        self.assertIsNone(self.backend.receipt)

    def test_invalid_or_private_usage_is_rejected(self):
        value=receipt_fixture(self.request,self.source,self.identities["generation"]["sha256"])
        for key,bad in (("total_tokens",129),("prompt_tokens",True),("reasoning_tokens",31),("private_key","sensitive")):
            changed=copy.deepcopy(value)
            changed["metrics"]["calls"][0]["usage"][key]=bad
            changed["binding"]=digest({k:v for k,v in changed.items() if k!="binding"})
            with self.assertRaises(ValueError):
                verify(changed,self.request,self.source)
    def test_legacy_receipt_without_usage_is_not_given_zero_usage(self):
        value=receipt_fixture(self.request,self.source,self.identities["generation"]["sha256"])
        value["metrics"]["calls"][0].pop("usage")
        value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
        summary=verify(value,self.request,self.source)["usage"]
        self.assertEqual(summary["responses_without_usage"],1)
        self.assertNotIn("total_tokens",summary)
        self.assertIsNone(summary["cost_usd"])
