"""Opt-in request/provider plumbing tests; all provider replies are offline fixtures."""
import unittest

from candidate_contract import make_request, validate_candidate, request_input_names, request_rotations
from native_function_core_fixtures import PROGRAMS
from native_function_rules import TASK
from deepseek_provider import Config, DeepSeekProvider, public_request
from test_candidate_pipeline import request_fixture
from test_deepseek_provider import FixtureTransport, completion
import json


def native_request(**kwargs):
    original = request_fixture()
    translation = {k: original[k] for k in ('fx_graph', 'public_constants', 'constant_origins', 'layout')}
    translation.update(kwargs.pop('translation_changes', {}))
    return make_request(translation, original['model'], original['compiler_profile_sha256'],
                        native_functions=True, **kwargs)


class NativeAgentContractTests(unittest.TestCase):
    def test_distinct_hash_and_legacy_grammar_preserved(self):
        old = request_fixture()
        request = native_request()
        self.assertNotEqual(request['request_id'], old['request_id'])
        self.assertEqual(request['task'], TASK)
        source = PROGRAMS['scalar']
        candidate = dict(schema=1, request_id=request['request_id'], hecate_source=source)
        check = validate_candidate(candidate, request)
        self.assertIn('native_functions', check)
        self.assertFalse(check['safe_to_execute_untrusted'])
        self.assertFalse(check['encrypted_correctness_checked'])
        with self.assertRaises(ValueError):
            validate_candidate(dict(candidate, request_id=old['request_id']), old)
        self.assertEqual(request_fixture(), old)

    def test_no_silent_union_with_frozen_old_construction_contracts(self):
        for flag in ('extended_arithmetic', 'public_construction', 'function_composition', 'closures',
                     'call_binding', 'public_iteration', 'function_literals', 'public_sequences',
                     'public_numbers', 'public_control', 'public_strings', 'public_polynomial',
                     'object_arrays', 'public_mappings', 'object_arithmetic', 'scalar_conversion', 'object_unary'):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                native_request(**{flag: True})
        with self.assertRaises(ValueError):
            native_request(construction_exercise='ou-neg')

    def test_provider_whitelist_and_generation_protocol_offline(self):
        request = native_request()
        self.assertEqual(public_request(request), request)
        response = dict(schema=1, request_id=request['request_id'], hecate_source=PROGRAMS['scalar'])
        transport = FixtureTransport([completion(json.dumps(response))])
        provider = DeepSeekProvider(Config(), transport=transport)
        self.assertEqual(json.loads(provider.generate(request, [])), response)
        exported = json.loads(transport.sent[0][0]['messages'][1]['content'])
        self.assertEqual(exported, request)
        self.assertFalse({'reference', 'inputs', 'keys', 'hecate_source'} & set(exported))

    def test_cli_and_input_abi(self):
        from run_candidate import parse_args, forward_options
        args = parse_args(['--case', 'cases/arithmetic-alias-chain.json', '--prepare', '--native-functions'])
        self.assertIn('--native-functions', forward_options(args))
        request = native_request()
        self.assertEqual(request_input_names(request), ('x',))
        self.assertEqual(tuple(request_rotations(request)), (-3, -2, -1, 1, 2, 3))
        layout = dict(request['layout'], inputs=[dict(name='left', dsl_name='x', shape=[4]),
                                                dict(name='right', dsl_name='y', shape=[4])])
        layout.pop('input_shape')
        self.assertEqual(request_input_names(native_request(translation_changes={'layout': layout})), ('x', 'y'))


if __name__ == '__main__':
    unittest.main()
