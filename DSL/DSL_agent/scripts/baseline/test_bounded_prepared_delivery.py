"""Twelve new controls; offline fixtures do not certify Agent, compiler or HE."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from delivery_component import bound, canonical, sha
from deepseek_provider import DeepSeekProvider, HTTPSTransport, SYSTEM
from generation_protocol import Config, ProviderError
import lossless_host_bound_delivery as host
import completed_delivery_startup as startup
import bounded_generation_context as context
import bounded_generation_transport as transport
import prepared_first_delivery as prepared
import test_host_bound_delivery as fixtures


class OfflineWire:
    is_live = False
    def __init__(self, selection, wire, config, answer, refused=False):
        self.selection, self.wire, self.config, self.answer = selection, wire, config, answer
        self.calls = []; self.refused = refused; self.on_post = None
    def post(self, body, timeout):
        base, actual = context.bodies(self.selection, self.wire, self.config)
        if body != base: raise AssertionError('Fixture received a different original body')
        self.calls.append(actual)
        if self.on_post is not None: self.on_post()
        return 200, canonical(dict(object='chat.completion', model=self.config.model,
            choices=[dict(finish_reason='stop', message=dict(role='assistant', content=self.answer,
                refusal='fixture refusal' if self.refused else None))],
            usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2)))


class BoundedPreparedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only the retained fixture builder is used; none of its old test methods
        # or numerical/backend campaigns is collected or executed.
        fixtures.HostBoundDeliveryTests.setUpClass()

    def setUp(self):
        self.f = fixtures.HostBoundDeliveryTests('test_unready_plan_needs_no_provider')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root = self.f.root; self.full = self.f.full; self.wire = host.public_request(self.full)
        self.config = self.f.config; self.output = self.f.output
        self.binding = host.freeze_host_request(self.f.host_path)
        self.selection = context.freeze(self.wire)
        self.mechanism = self.freeze(self.selection)
        self.source = 'not valid python {{{\r\n\r\n'

    def freeze(self, selection):
        base = host.generator_manifest('bounded-prepared-offline-controls-v1', list(prepared.source_manifest()),
            host_bindings=[self.binding], plans=[self.f.planning],
            assessment_paths={self.full['request_id']: str(self.f.assessment)}, backend_identity=self.f.identity,
            baseline_binding='0'*64, evaluation_policy_sha256=sha(canonical(self.f.plan)), provider_config=self.config)
        self.original_base = base
        base = startup.external.bind_assessment(base)
        base = startup.bind_startup(base)
        return prepared.bind(base, {self.full['request_id']: selection})

    def provider(self, *, answer=None, config=None, refused=False):
        config = self.config if config is None else config
        answer = json.dumps(dict(schema=1, request_id=self.full['request_id'], hecate_source=self.source)) if answer is None else answer
        self.offline = OfflineWire(self.selection, self.wire, config, answer, refused)
        self.p = DeepSeekProvider(config, transport=self.offline)
        return self.p

    def deliver(self, provider=None, **updates):
        args = dict(mechanism=self.mechanism, plan=self.f.plan, backend=self.f.backend,
                    backend_identity=self.f.identity, test_only=True)
        args.update(updates)
        return prepared.generate_once(self.full, self.provider() if provider is None else provider, self.output, **args)

    def rewrite(self, path, value):
        path.chmod(0o600); path.write_bytes(canonical(value))

    def test_01_public_actual_loop_limits_are_source_bound_without_model_routes(self):
        public = self.selection['public']
        self.assertEqual(public['native_loop_limits'], dict(MAX_RANGE=128, MAX_ITERATIONS=4096, MAX_NODES=4096, MAX_INTEGER=1048576))
        self.assertFalse(public['compiler_or_CKKS_feasibility_certified'])
        self.assertIsNone(public['mathematical_model_origin'])
        self.assertEqual(context.verify(self.selection, self.wire), self.wire)
        changed = copy.deepcopy(self.selection); changed['public']['native_loop_limits']['MAX_RANGE'] = 129
        changed = bound({k: v for k, v in changed.items() if k != 'binding'})
        with self.assertRaises(ValueError): context.verify(changed, self.wire)
        with patch.object(context.loops.native_public_loops, 'MAX_RANGE', 129), self.assertRaises(ValueError):
            context.verify(self.selection, self.wire)

    def test_02_optional_origin_requires_host_verified_original_provenance(self):
        import test_generation_model_origin as origin_fixtures
        provenance, request = origin_fixtures.fixture()
        path = self.root/'origin.json'; path.write_bytes(canonical(provenance))
        descriptor = dict(path=str(path), sha256=sha(path.read_bytes()))
        selected = context.freeze(request, host=request, provenance=descriptor)
        value = selected['public']['mathematical_model_origin']
        self.assertFalse(value['reference_answers_included']); self.assertFalse(value['rule_dsl_included'])
        self.assertEqual(context.verify(selected, request, host=request), host.public_request(request))
        with self.assertRaises(ValueError): context.freeze(request, provenance=descriptor)
        path.write_bytes(b'{}')
        with self.assertRaises(ValueError): context.verify(selected, request, host=request)

    def test_03_entire_system_addendum_counts_against_original_public_budget(self):
        wire = dict(request_id='a'*64, rules='x')
        selected = context._selection(wire, None, None, None)
        self.assertEqual(selected['combined_public_context_bytes'], len(canonical(wire)) + len(context.addendum(selected['public']).encode()))
        wire['rules'] = 'x'*(context.CONTEXT_BYTES - 200)
        self.assertLess(len(canonical(wire)), context.CONTEXT_BYTES)
        with self.assertRaisesRegex(ValueError, 'Combined original public context'): context._selection(wire, None, None, None)

    def test_04_exact_augmented_body_preserves_original_request_and_config(self):
        before = canonical(self.full), canonical(self.wire)
        base, actual = context.bodies(self.selection, self.wire, self.config)
        b, a = json.loads(base), json.loads(actual)
        self.assertEqual(b['messages'][0]['content'], SYSTEM)
        self.assertEqual(a['messages'][1], b['messages'][1])
        self.assertEqual(a['messages'][0]['content'], SYSTEM + context.addendum(self.selection['public']))
        self.assertEqual({k: v for k, v in a.items() if k != 'messages'}, self.config.settings())
        self.assertEqual(before, (canonical(self.full), canonical(self.wire)))
        self.assertEqual(len(a['messages']), 2)
        self.assertNotIn('source_sha256', a['messages'][0]['content'])

    def test_05_transport_retains_original_post_worker_and_validates_exact_baseline(self):
        frozen = transport.freeze(self.selection, self.wire, self.config)
        t = transport.ContextHTTPSTransport(selection=self.selection, request=self.wire, approved_body=frozen,
            api_key='offline-fixture-not-a-credential', enabled=True, approved_config=self.config)
        self.assertIs(transport.ContextHTTPSTransport.post, HTTPSTransport.post)
        base, actual = context.bodies(self.selection, self.wire, self.config)
        self.assertEqual(canonical(t.validate(base, self.config.timeout_seconds)), actual)
        with self.assertRaises(ProviderError): t.validate(actual, self.config.timeout_seconds)
        altered = json.loads(base); altered['messages'].append(dict(role='user', content='repair'))
        with self.assertRaises(ProviderError): t.validate(canonical(altered), self.config.timeout_seconds)
        with self.assertRaises(ProviderError): t.validate(base, self.config.timeout_seconds+1)

    def test_06_context_or_body_change_rejects_before_original_attempt_counters(self):
        frozen = transport.freeze(self.selection, self.wire, self.config)
        t = transport.ContextHTTPSTransport(selection=self.selection, request=self.wire, approved_body=frozen,
            api_key='offline-fixture-not-a-credential', enabled=True, approved_config=self.config)
        p = DeepSeekProvider(self.config, transport=t)
        t.approved_body = dict(frozen, actual_body_sha256='e'*64)
        with patch('deepseek_provider.subprocess.run', side_effect=AssertionError('HTTP forbidden')) as worker:
            with self.assertRaises(ProviderError): p.generate(self.wire, [])
            self.assertFalse(worker.called)
        self.assertEqual((p.generation_attempts, p.request_attempts), (0, 0))

    def test_07_prepared_delivery_admits_full_host_once_and_freezes_invalid_original_bytes(self):
        p = self.provider(); phase = dict(after=False); resolve = host.resolve_host
        def checked_resolve(*args, **kwargs):
            if phase['after']: raise AssertionError('Full-host reconstruction after output')
            return resolve(*args, **kwargs)
        self.offline.on_post = lambda: phase.update(after=True)
        with patch.object(host, 'resolve_host', checked_resolve):
            receipt = self.deliver(p)
            rows = prepared.verify_preserved(self.output, receipt['binding'])
        self.assertEqual(len(self.f.backend.seen), 1)
        self.assertEqual(self.f.backend.seen[0], self.full)
        self.assertEqual(rows[2], self.source.encode('utf-8'))
        self.assertEqual((self.output/'candidate.py').read_bytes(), self.source.encode('utf-8'))
        self.assertEqual(receipt['status'], 'dsl_delivered'); self.assertFalse(receipt['first_output_eligible'])
        self.assertEqual((p.generation_attempts, p.request_attempts, len(self.offline.calls)), (1, 1, 1))

    def test_08_unselected_mode_and_changed_frozen_body_settings_stop_before_call(self):
        p = self.provider()
        with self.assertRaises(ValueError): self.deliver(p, mechanism=self.original_base)
        changed = copy.deepcopy(self.mechanism)
        changed['frozen_generation_bodies'][self.full['request_id']]['configuration']['max_tokens'] += 1
        changed = bound({k: v for k, v in changed.items() if k != 'binding'})
        with self.assertRaises(ValueError): self.deliver(p, mechanism=changed)
        self.assertEqual(p.request_attempts, 0); self.assertFalse(self.f.backend.seen)

    def test_09_changed_saved_DSL_or_actual_wire_bytes_never_verify(self):
        receipt = self.deliver()
        source = self.output/'candidate.py'; source.chmod(0o600); source.write_bytes(b'replacement')
        with self.assertRaises(ValueError): prepared.verify_preserved(self.output, receipt['binding'])
        source.write_bytes(self.source.encode())
        body = host._admission_path(self.output)/'body.json'; body.chmod(0o600); body.write_bytes(b'{}')
        with self.assertRaises(ValueError): prepared.verify_preserved(self.output, receipt['binding'])

    def test_10_contract_failure_retains_one_answer_without_repair(self):
        p = self.provider(answer='{}'); receipt = self.deliver(p)
        self.assertEqual(receipt['status'], 'contract_rejected')
        self.assertEqual((self.output/'response.txt').read_bytes(), b'{}')
        self.assertFalse((self.output/'candidate.py').exists())
        self.assertEqual(p.request_attempts, 1); self.assertEqual(receipt['post_output_repairs'], 0)
        self.assertIsNone(prepared.verify_preserved(self.output, receipt['binding'])[2])

    def test_11_existing_slot_and_multi_call_or_retry_configuration_cannot_regenerate(self):
        self.deliver(); p = self.provider()
        with self.assertRaises(FileExistsError): self.deliver(p)
        self.assertEqual(p.request_attempts, 0)
        for config in (Config(max_calls=2), Config(max_calls=1, provider_retries=1)):
            with self.subTest(config=config), self.assertRaises(ValueError): transport.freeze(self.selection, self.wire, config)
        with self.assertRaises(TypeError): self.deliver(self.provider(), evaluator=lambda *_: True)

    def test_12_backend_block_and_host_drift_preserve_zero_call_failure(self):
        self.f.backend = fixtures.Backend(reject=True); p = self.provider(); receipt = self.deliver(p)
        self.assertEqual(receipt['status'], 'backend_blocked')
        self.assertFalse(receipt['first_output_eligible']); self.assertEqual(p.request_attempts, 0)
        self.assertEqual(receipt['origin'], 'pre_generation_admission')
        altered = copy.deepcopy(self.full); altered['model']['constants']['w'][0][0] += .01
        self.f.host_path.write_bytes(canonical(altered)); p = self.provider()
        with self.assertRaises(ValueError): self.deliver(p)
        self.assertEqual(p.request_attempts, 0)


if __name__ == '__main__': unittest.main()
