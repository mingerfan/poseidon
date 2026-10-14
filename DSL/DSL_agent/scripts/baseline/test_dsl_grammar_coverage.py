"""Coverage is typed AST evidence, not substring counts or an FHE oracle."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from candidate_contract import validate_candidate
from dsl_grammar_coverage import FEATURES, analyze_source
from audit_dsl_coverage import source_evidence, summarize_cases, audit_coverage
from test_multi_input_protocol import fixture


def program(body, count=1):
    names = ('x', 'y', 'z', 't')[:count]
    return ('@hc.func("' + ','.join(['c']*count) + '")\ndef golden(' + ', '.join(names) + '):\n' +
            '\n'.join('    ' + line for line in body.splitlines()) + '\n')


def analyze(body, constants=None, count=1, outputs=1):
    return analyze_source(program(body, count), constants or {}, outputs,
                          contract='hecate-function-v3' if count > 1 else 'hecate-function-v2',
                          input_names=('x', 'y', 'z', 't')[:count])


class GrammarCoverageTests(unittest.TestCase):
    def test_every_feature_partition_has_a_positive_static_fixture(self):
        seen = set()
        for count in range(1, 5):
            seen.update(analyze('return x', count=count)['output_dependency_counts'])
            seen.update(analyze('return [' + ', '.join(['x']*count) + ']', outputs=count)['output_dependency_counts'])
        for op, symbol in (('add', '+'), ('subtract', '-'), ('multiply', '*')):
            for suffix, operand, constants in (('cipher', 'x', {}), ('scalar', 'w', {'w': .5}),
                                               ('length1', 'w', {'w': [.5]}),
                                               ('length4', 'w', {'w': [.5]*4})):
                counts = analyze(f'return x {symbol} {operand}', constants)['output_dependency_counts']
                self.assertEqual(counts[f'{op}.{suffix}'], 1)
                seen.update(counts)
        for step in (-3, -2, -1, 1, 2, 3):
            seen.update(analyze(f'return x.rotate({step})')['output_dependency_counts'])
        seen.update(analyze('a = x\nb = -a\nreturn b + (x * x)')['output_dependency_counts'])
        self.assertEqual(seen, set(FEATURES))

    def test_dead_operators_and_unused_constants_do_not_count_as_output_coverage(self):
        result = analyze('dead = -x\nunused = dead.rotate(-3)\nreturn x + w', {'w': .25, 'unusedpublic': [1]*4})
        self.assertIn('negate.cipher', result['present_counts'])
        self.assertNotIn('negate.cipher', result['output_dependency_counts'])
        self.assertNotIn('rotate.-3', result['output_dependency_counts'])
        self.assertNotIn('add.length4', result['present_counts'])
        self.assertEqual(result['dead_assignments'], ['dead', 'unused'])

    def test_ssa_reused_dependencies_are_counted_once(self):
        result = analyze('a = x * w\nb = a + a\nreturn b + a', {'w': [.5]})
        counts = result['output_dependency_counts']
        self.assertEqual(counts['multiply.length1'], 1)
        self.assertEqual(counts['add.cipher'], 2)
        self.assertEqual(counts['statement.assignment'], 2)
        self.assertEqual(counts, result['present_counts'])

    def test_rotations_literals_are_not_negation_and_comments_not_features(self):
        counts = analyze('# rotate(3), -x, x * x\nreturn x.rotate(-2)')['output_dependency_counts']
        self.assertEqual(counts['rotate.-2'], 1)
        for feature in ('negate.cipher', 'rotate.+3', 'multiply.cipher'):
            self.assertNotIn(feature, counts)

    def test_invalid_programs_never_supply_coverage(self):
        for body in ('return w + x', 'return x / w', 'return x.rotate(0)',
                     'return x.rotate(True)', 'return x.relu()', 'return x[0]',
                     'return [w]', 'x = -x\nreturn x', 'if x:\n    return x\nreturn x'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                analyze(body, {'w': .5})
        with self.assertRaises(ValueError):
            analyze('return x + w', {'w': [1, 2]})
        with self.assertRaises(ValueError):
            analyze_source(program('return -x'), {}, contract='hecate-function-v0')

    def test_model_name_does_not_establish_native_subtraction_coverage(self):
        row = dict(case='subtraction-model', grammar=analyze('return x + x * w', {'w': -1}))
        result = summarize_cases([row])
        self.assertIn('subtract.cipher', result['missing_feature_partitions'])
        self.assertNotIn('multiply.scalar', result['missing_feature_partitions'])
        self.assertFalse(result['all_semantics_verified'])

    def test_complete_partition_observation_is_still_not_semantic_proof(self):
        result = summarize_cases([dict(case='synthetic-unit-fixture', grammar=dict(
            present_counts=dict.fromkeys(FEATURES, 1), output_dependency_counts=dict.fromkeys(FEATURES, 1)))])
        self.assertTrue(result['all_feature_partitions_observed'])
        self.assertFalse(result['all_semantics_verified'])


class TracedSourceBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = fixture()
        self.source = program('return x + y', 2)
        self.candidate = dict(schema=1, request_id=self.request['request_id'], hecate_source=self.source)
        (self.root/'attempt-00').mkdir()
        self.payload = dict(request=self.request, candidate=self.candidate)
        hashes = {}
        for name, data in (('request.json', self.request), ('attempt-00/trace-payload.json', self.payload)):
            raw = json.dumps(data).encode()
            (self.root/name).write_bytes(raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        (self.root/'attempt-00/candidate.py').write_bytes(self.source.encode('utf-8'))
        self.report = dict(frozen_hashes=hashes, attempts=[dict(index=0, numerically_correct=True,
            static_check=validate_candidate(self.candidate, self.request), artifact_gate=dict(opcode_counts={'6': 1}),
            trace=dict(frontend='real_Hecate', candidate_python_executed=False, request_id=self.request['request_id']))])

    def test_actual_traced_source_and_frozen_request_are_bound(self):
        result = source_evidence(self.root, self.report)
        self.assertEqual(result['grammar']['output_dependency_counts']['add.cipher'], 1)

    def test_changed_loose_python_is_not_credited(self):
        (self.root/'attempt-00/candidate.py').write_text(program('return x - y', 2))
        with self.assertRaisesRegex(ValueError, 'sidecar'):
            source_evidence(self.root, self.report)

    def test_changed_payload_hash_or_missing_frozen_entry_rejected(self):
        (self.root/'attempt-00/trace-payload.json').write_text(json.dumps(self.payload)+' ')
        with self.assertRaisesRegex(ValueError, 'hash'):
            source_evidence(self.root, self.report)
        del self.report['frozen_hashes']['attempt-00/trace-payload.json']
        with self.assertRaisesRegex(ValueError, 'Missing'):
            source_evidence(self.root, self.report)

    def test_mismatched_static_or_execution_markers_rejected(self):
        for field, value in (('static_check', {}), ('trace', dict(frontend='simulated',
                candidate_python_executed=True, request_id=self.request['request_id']))):
            report = copy.deepcopy(self.report)
            report['attempts'][0][field] = value
            with self.assertRaises(ValueError):
                source_evidence(self.root, report)


@unittest.skipUnless(os.environ.get('POSEIDON_DSL_COVERAGE_LINEAGE'), 'requires saved actual Agent CPU lineage')
class ActualCoverageTests(unittest.TestCase):
    def test_real_96_outputs_and_traced_payloads_without_api_calls(self):
        report = audit_coverage(Path(os.environ['POSEIDON_DSL_COVERAGE_LINEAGE']))
        self.assertEqual(report['model_cases'], 96)
        self.assertEqual(report['input_executions'], 384)
        self.assertEqual(report['new_api_calls'], 0)
        self.assertIn('inputs.3', report['missing_feature_partitions'])
        self.assertIn('inputs.4', report['missing_feature_partitions'])
        self.assertFalse(report['all_semantics_verified'])


if __name__ == '__main__':
    unittest.main()
