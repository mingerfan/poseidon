"""Broadcast/alias goldens, independent references, and opt-in real FHE audit."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import unittest

from broadcast_fixtures import ROOT, GOLDENS, FIXTURES, PLANS
from candidate_contract import validate_candidate, make_request
from dsl_grammar_coverage import analyze_source
from hecate_contract import validate_function
from model_graph import validate_graph, evaluate_reference


def descriptor(name):
    return json.loads((ROOT / 'scripts/baseline/cases' / (FIXTURES[name]['case'] + '.json')).read_text())


def hand_reference(name, values):
    # No graph traversal, DSL evaluation, compiler or decryption used here.
    if name == 'add_one':
        return [v + .375 for v in values]
    if name == 'mul_one':
        return [-.5 * v for v in values]
    if name == 'sub_one':
        return [v - .375 for v in values]
    if name == 'sub_four':
        return [v - c for v, c in zip(values, [.125, -.25, .5, .75])]
    if name == 'sub_scalar':
        return [math.fsum(w * v for w, v in zip([.5, -.25, .125, .75], values)) - .375]
    raise ValueError('Unknown hand fixture')


class BroadcastContractTests(unittest.TestCase):
    def test_independent_reference_zero_signed_boundary_and_random(self):
        rng = random.Random(621)
        inputs = [[0]*4, [.5, -1, .25, -.75], [-1, 1, -1, 1], [1]*4,
                  *[[rng.uniform(-1, 1) for _ in range(4)] for _ in range(8)]]
        for name in FIXTURES:
            data = descriptor(name)
            validate_graph(data)
            for values in inputs:
                for actual, expected in zip(evaluate_reference(data, values), hand_reference(name, values)):
                    self.assertAlmostEqual(actual, expected, places=12)
            renamed = copy.deepcopy(data)
            renamed['id'] = 'free-user-name'
            self.assertEqual(evaluate_reference(data, inputs[1]), evaluate_reference(renamed, inputs[1]))

    def test_all_six_feature_partitions_are_live_in_manual_programs(self):
        seen = set()
        for name, row in FIXTURES.items():
            source = (GOLDENS / (name + '.py')).read_text()
            grammar = analyze_source(source, row['constants'], contract='hecate-function-v1')
            self.assertTrue(set(row['features']) <= set(grammar['output_dependency_counts']))
            self.assertEqual(grammar['dead_assignments'], [])
            seen.update(row['features'])
            # Counterexamples are type-valid so numerical verification must catch them.
            validate_function((GOLDENS / ('wrong_' + name + '.py')).read_text(), row['constants'],
                              contract='hecate-function-v1')
        self.assertEqual(seen, {'statement.alias', 'add.length1', 'multiply.length1',
                               'subtract.scalar', 'subtract.length1', 'subtract.length4'})

    def test_bad_broadcast_and_mutating_aliases_rejected(self):
        template = '@hc.func("c")\ndef golden(x):\n    return x + c0\n'
        for value in ([1, 2], [1, 2, 3], [[.5]], [], [float('nan')], [float('inf')], [True]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_function(template, {'c0': value}, contract='hecate-function-v1')
        for body in ('alias = x\n    alias = x + c0\n    return alias',
                     'alias = x\n    alias += c0\n    return alias',
                     'alias = c0\n    return x + alias', 'return c0 - x'):
            with self.assertRaises(ValueError):
                validate_function('@hc.func("c")\ndef golden(x):\n    ' + body + '\n',
                                  {'c0': [.5]}, contract='hecate-function-v1')

    def test_graph_rank_mismatch_is_not_silently_broadcast(self):
        for value in ([[.375]], [.375, .375], [.375]*3):
            data = descriptor('add_one')
            data['constants']['offset'] = value
            with self.assertRaises(ValueError):
                validate_graph(data)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'requires pinned numerical dependencies')
class BroadcastTorchTests(unittest.TestCase):
    def test_torch_translator_registry_and_manual_ast_have_independent_agreement(self):
        import numpy as np
        import torch
        from candidate_trace import evaluate_tree
        from fx_to_hecate import translate
        from model_graph import build_graph_model
        from model_catalog import test_inputs

        class Packed:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float64)
            def __add__(self, rhs):
                return Packed(self.value + (rhs.value if isinstance(rhs, Packed) else rhs))
            def __sub__(self, rhs):
                return Packed(self.value - (rhs.value if isinstance(rhs, Packed) else rhs))
            def __mul__(self, rhs):
                return Packed(self.value * (rhs.value if isinstance(rhs, Packed) else rhs))
            def __neg__(self):
                return Packed(-self.value)
            def rotate(self, step):
                return Packed(np.roll(self.value, -step))

        for name, row in FIXTURES.items():
            data = descriptor(name)
            model, shape = build_graph_model(data)
            payload = translate(model, shape)
            self.assertEqual(payload['public_constants'], row['constants'])
            request = make_request(payload, data, 'a'*64)
            constants = {k: np.asarray(v) for k, v in row['constants'].items()}
            wrong_detected = False
            for wrong in (False, True):
                source = (GOLDENS / (('wrong_' if wrong else '') + name + '.py')).read_text()
                validate_candidate(dict(schema=1, request_id=request['request_id'], hecate_source=source), request)
                for values in test_inputs([4]):
                    expected = np.asarray(hand_reference(name, values))
                    np.testing.assert_allclose(model(torch.from_numpy(values)).detach().numpy(), expected,
                                               rtol=1e-12, atol=1e-12)
                    evaluated = evaluate_tree(source, constants, Packed(values))
                    actual = evaluated.value[:1] if name == 'sub_scalar' else evaluated.value
                    if wrong:
                        wrong_detected |= not np.allclose(actual, expected, rtol=1e-4, atol=1e-5)
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            self.assertTrue(wrong_detected)


@unittest.skipUnless(os.environ.get('POSEIDON_BROADCAST_GOLDEN_REPORT'), 'requires actual broadcast FHE batch')
class BroadcastEvidenceTests(unittest.TestCase):
    def test_five_goldens_five_counterexamples_real_fhe_immutable_reference_and_cleanup(self):
        import numpy as np
        from audit_agent_lineage import metadata, read
        from model_catalog import test_inputs
        from seal_cpu_golden import compare
        from hecate_python_env import WORK

        results = WORK / 'results'
        batch, _ = metadata(Path(os.environ['POSEIDON_BROADCAST_GOLDEN_REPORT']), results)
        self.assertEqual(batch['status'], 'passed')
        self.assertEqual(batch['agent_calls'], 0)
        self.assertEqual(len(batch['cases']), 10)
        for saved, (case, golden, wrong) in zip(batch['cases'], PLANS):
            self.assertEqual((saved['case'], saved['counterexample']), (case, wrong))
            self.assertTrue(saved['matched_expected'])
            name = golden.removeprefix('wrong_')
            run = Path(saved['run']).resolve()
            self.assertEqual(run.parent, results)
            report, _ = metadata(run/'report.json', results)
            self.assertEqual(report['agent_calls'], 0)
            self.assertFalse(report['llm_generation_validated'])
            self.assertEqual(report['backend'], 'upstream_SEAL_HEVM_CPU')
            self.assertFalse(report['poseidon_gpu_validated'])
            self.assertEqual(report['tolerance'], dict(atol=1e-5, rtol=1e-4))
            self.assertEqual(report['parameters']['security_check'], 'tc128')
            self.assertEqual(report['parameters']['modulus_bits'], [60]*14)
            self.assertEqual(report['parameters']['polynomial_degree'], 32768)
            self.assertTrue(report['parameters']['parameters_set'])
            self.assertEqual(report['parameters']['seal_version'], '4.0.0')
            for path, digest in report['frozen_hashes'].items():
                self.assertEqual(read(run/path, run)[1], digest)
            model, _ = metadata(run/'model.json', run)
            self.assertEqual(model, descriptor(name))
            self.assertEqual(len(report['attempts']), 1)
            attempt = report['attempts'][0]
            output = run/'attempt-00/output'
            self.assertTrue({'lowered._hecate_golden.hevm', '_hecate_golden.cst', 'lowered.ckks.mlir'} <=
                            set(attempt['artifact_hashes']))
            for path, digest in attempt['artifact_hashes'].items():
                self.assertEqual(read(output/path, output)[1], digest)
            payload, _ = metadata(run/'attempt-00/trace-payload.json', run)
            source = (GOLDENS/(golden+'.py')).read_text()
            self.assertEqual(payload['candidate']['hecate_source'], source)
            request, _ = metadata(run/'request.json', run)
            self.assertEqual(payload['request'], request)
            self.assertEqual(request['public_constants'], FIXTURES[name]['constants'])
            self.assertEqual(validate_candidate(payload['candidate'], request), attempt['static_check'])
            self.assertEqual(attempt['trace']['frontend'], 'real_Hecate')
            self.assertFalse(attempt['trace']['candidate_python_executed'])
            execution = attempt['execution']
            self.assertTrue(execution['encrypted_execution'])
            self.assertFalse(execution['bootstrap_executed'])
            self.assertEqual(execution['input_batches'], 4)
            self.assertEqual(execution['encrypted_input_count'], 1)
            self.assertTrue(execution['rotation_key_check']['actual_key_file_verified'])
            with np.load(run/'arrays.npz', allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['inputs'], test_inputs([4]))
                expected = np.asarray([hand_reference(name, x) for x in arrays['inputs']])
                np.testing.assert_allclose(arrays['reference'], expected, rtol=1e-12, atol=1e-12)
                compared = compare(np.load(output/'decrypted.npy', allow_pickle=False), arrays['reference'], 1e-5, 1e-4)
            self.assertEqual(compared, attempt['comparison'])
            self.assertEqual(compared['passed'], not wrong)
            if wrong:
                self.assertEqual(attempt['failure_layer'], 'numerical_comparison')
            else:
                self.assertEqual(report['status'], 'passed')
                grammar = analyze_source(source, request['public_constants'], contract='hecate-function-v1')
                self.assertTrue(set(FIXTURES[name]['features']) <= set(grammar['output_dependency_counts']))
            self.assertFalse((run/'private-keys').exists())


if __name__ == '__main__':
    unittest.main()
