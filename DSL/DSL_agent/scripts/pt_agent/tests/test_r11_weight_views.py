"""R11 system wrapper and consistent result/registry round labels."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.paths import INTAKE_PYTHON
from pt_agent.tests._artifacts import require
from pt_agent.tests._r11_weight_view_cases import INNER_CASE_COUNT


INTAKE_ARTIFACT = INTAKE_PYTHON.parent if INTAKE_PYTHON.is_symlink() else INTAKE_PYTHON


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))


@require(INTAKE_ARTIFACT)
class WeightViewEnvironmentChecks(unittest.TestCase):
    def test_real_weight_view_cases(self):
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '180',
                   '--python', str(INTAKE_PYTHON), '--', '-m', 'unittest',
                   'pt_agent.tests._r11_weight_view_cases', '-v']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=300, preexec_fn=limited)
        print(completed.stdout, end='', flush=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
        self.assertIsNotNone(count, completed.stdout)
        self.assertEqual(int(count.group(1)), INNER_CASE_COUNT, completed.stdout)
        self.assertNotRegex(completed.stdout, r'skipped=\d+')
        print('R11_WEIGHT_VIEW_INNER_TESTS='+str(count.group(1)), flush=True)


class RoundLabelChecks(unittest.TestCase):
    def test_finalize_default_and_explicit_label(self):
        from pt_agent.experiments.r10_paper_preflight import finalize_result
        values = dict(phase='capabilities', worker=None, processes=[], timing={}, status='ok',
                      error=None, git_state={}, out=Path('/fixture/run'))
        self.assertEqual(finalize_result(**values)['round'], 'R10')
        self.assertEqual(finalize_result(**values, round_label='R11')['round'], 'R11')

    def run_capabilities(self, *, label=None):
        from pt_agent.experiments import r10_paper_preflight as preflight
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(phase='capabilities', out=root/'run', model=None)
            if label is not None:
                args.round_label = label
            with patch.object(preflight.paths, 'RESULTS_DIR', root), \
                 patch.object(preflight, '_git_state', return_value=dict(branch='fixture', working_tree_changed=False)), \
                 patch.object(preflight, 'capabilities', return_value=dict(measurements=dict(model_runs=0))), \
                 patch.object(preflight.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*1024**3)):
                result = preflight.execute(args)
            saved = json.loads((args.out/'result.json').read_text())
            records = [json.loads(line) for line in (root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(records), 1)
        self.assertEqual(set(records[0]), {'run_id', 'round', 'model', 'mode', 'git_rev',
                                           'metrics', 'artifacts_dir', 'ts'})
        self.assertEqual(result['round'], saved['round'])
        self.assertEqual(saved['round'], records[0]['round'])
        return result['round']

    def test_execute_default_label_matches_result_and_registry(self):
        self.assertEqual(self.run_capabilities(), 'R10')

    def test_execute_r11_label_matches_result_and_registry(self):
        self.assertEqual(self.run_capabilities(label='R11'), 'R11')

    def test_cli_default_explicit_and_rejected_round_labels(self):
        from pt_agent.experiments import r10_paper_preflight as preflight
        for flags, expected in (([], 'R10'), (['--round-label', 'R11'], 'R11')):
            with self.subTest(flags=flags), \
                 patch.object(sys, 'argv', ['preflight', '--phase', 'capabilities', '--out', '/fixture/run', *flags]), \
                 patch.object(preflight, 'execute', return_value=dict(status='ok')) as execute, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(preflight.main(), 0)
                self.assertEqual(execute.call_args.args[0].round_label, expected)
        with patch.object(sys, 'argv', ['preflight', '--phase', 'capabilities', '--out', '/fixture/run',
                                       '--round-label', 'R12']), \
             patch.object(preflight, 'execute') as execute, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                preflight.main()
            self.assertEqual(caught.exception.code, 2)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
