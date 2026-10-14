"""Frozen recent-model cohort; these checks make no paid API calls."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

from custom_batch_manifest import load_manifest

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / 'cases/recent-semantics-agent-20-manifest.json'
EXTRAS = ('arithmetic-alias-chain', 'construction-linear',
          'construction-sumslots3', 'construction-sumslots4',
          'construction-genpoly3', 'construction-genpoly5',
          'construction-genpoly7', 'construction-genpoly-even')


class RecentAgentManifestTests(unittest.TestCase):
    def test_exact_existing_graphs_and_weights_not_golden_answers(self):
        rows, data, _ = load_manifest(MANIFEST)
        expected = json.loads((BASE / 'cases/advanced-agent-12-manifest.json').read_text())['cases']
        expected += [json.loads((BASE / 'cases' / (name + '.json')).read_text()) for name in EXTRAS]
        self.assertEqual(data['cases'], expected)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r['descriptor']['id'] for r in rows}), 20)
        self.assertTrue(all(r['status'] == 'pending' for r in rows))

    def test_even_polynomial_keeps_its_independent_model(self):
        _, data, _ = load_manifest(MANIFEST)
        cases = {c['id']: c for c in data['cases']}
        self.assertNotEqual(cases['construction-genpoly3']['constants'],
                            cases['construction-genpoly-even']['constants'])
        self.assertEqual(len(cases['construction-linear']['constants']['weight']), 2)
        for width in range(5, 9):
            self.assertEqual(len(cases[f'custom-wide-mlp-{width}']['constants']['weight_in']), width)

    def test_offline_plan_exposes_latest_contract_and_fixed_concurrency(self):
        command = [sys.executable, '-B', str(BASE / 'run_agent_batch.py'), '--plan',
                   '--case-manifest', str(MANIFEST), '--object-arrays',
                   '--provider', 'deepseek', '--model', 'deepseek-flash', '--jobs', '10']
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual((plan['mode'], plan['agent_calls']), ('plan_only', 0))
        self.assertTrue(plan['object_arrays'])
        self.assertEqual((plan['api_concurrency'], plan['native_execution_concurrency']), (10, 2))
        self.assertEqual(plan['summary']['planned'], 20)
        self.assertEqual(plan['summary']['completed'], 0)


if __name__ == '__main__':
    unittest.main()
