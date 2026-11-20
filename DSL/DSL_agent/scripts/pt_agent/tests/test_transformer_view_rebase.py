"""Run the new consumed-view regressions in the existing intake environment."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from pt_agent import paths
from pt_agent.tests._artifacts import require


BOOTSTRAP = r'''
import json,os,sys,tempfile,unittest
from pathlib import Path
work=Path(sys.argv[1]);work.mkdir(parents=True,exist_ok=True)
os.environ['TMPDIR']=str(work);tempfile.tempdir=str(work)
suite=unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests._transformer_view_rebase_cases.MaterializedViewCases')
run=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(testsRun=run.testsRun,failures=len(run.failures),errors=len(run.errors),skips=len(run.skipped),success=run.wasSuccessful())
(work/'inner-result.json').write_text(json.dumps(record)+'\n')
raise SystemExit(0 if record['success'] and record['testsRun']==3 and record['skips']==0 else 1)
'''


@require(paths.INTAKE_PYTHON.parent if paths.INTAKE_PYTHON.is_symlink() else paths.INTAKE_PYTHON)
class MaterializedViewWrapper(unittest.TestCase):
    def test_only_new_materialized_view_cases(self):
        from pt_agent.experiments.r10_paper_metrics import run_process
        if os.environ.get('QWEN_REBASE_TEST_WORK'):
            work = Path(os.environ['QWEN_REBASE_TEST_WORK'])
            work.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory(prefix='qwen-rebase-')
            self.addCleanup(temporary.cleanup)
            work = Path(temporary.name)
        argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '90',
                '--python', str(paths.INTAKE_PYTHON), '--', '-B', '-c', BOOTSTRAP, str(work)]
        obs = run_process(argv, cwd=Path(__file__).resolve().parents[2], log=work/'inner.log',
                          resources=work/'resources.jsonl', timeout=120, mem_gib=2, fsize_gib=1,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        print((work/'inner.log').read_text(), end='', flush=True)
        self.assertEqual(obs['exit_code'], 0)
        self.assertFalse(obs['timed_out'])
        self.assertIsNone(obs['signal'])
        self.assertEqual(obs['remaining_pids'], [])
        self.assertEqual(json.loads((work/'inner-result.json').read_text()),
                         dict(testsRun=3, failures=0, errors=0, skips=0, success=True))
        (work/'process.json').write_text(json.dumps(obs, indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
