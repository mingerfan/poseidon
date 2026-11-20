"""Only the new tiny CNN/generic-tuner regression inside the intake environment."""

import ast
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from pt_agent import paths
from pt_agent.tests._artifacts import require


PYTHON=paths.INTAKE_PYTHON
BOOTSTRAP=r'''import json, os, signal, sys, tempfile, unittest
from pathlib import Path
work=Path(sys.argv[1])
os.environ['TMPDIR']=str(work)
tempfile.tempdir=str(work)
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
signal.alarm(60)
suite=unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests._r28_cnn_tuner_cases.CNNTunerCases')
result=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(testsRun=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),success=result.wasSuccessful())
(work/'inner-unit-result.json').write_text(json.dumps(record)+'\n')
raise SystemExit(0 if record['success'] and record['testsRun']>0 and record['skips']==0 else 1)
'''


@require(PYTHON.parent if PYTHON.is_symlink() else PYTHON)
class TinyGenericCNNTunerChecks(unittest.TestCase):
    def test_actual_generic_cnn_tuning_with_original_math_and_calibration_only_domains(self):
        from pt_agent.experiments.r10_paper_metrics import run_process
        tree=ast.parse(Path(__file__).with_name('_r28_cnn_tuner_cases.py').read_text())
        expected=sum(isinstance(child,ast.FunctionDef) and child.name.startswith('test_')
                     for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='CNNTunerCases' for child in node.body)
        if os.environ.get('R28_CNN_TUNER_TEST_WORK'):
            work=Path(os.environ['R28_CNN_TUNER_TEST_WORK']);work.mkdir(parents=True,exist_ok=True)
        else:
            temporary=TemporaryDirectory(prefix='r28-cnn-generic-')
            self.addCleanup(temporary.cleanup);work=Path(temporary.name)
        observation=run_process([sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','90','--python',str(PYTHON),
            '--','-B','-c',BOOTSTRAP,str(work)],cwd=Path(__file__).resolve().parents[2],log=work/'inner.log',
            resources=work/'resources.jsonl',timeout=120,mem_gib=2,fsize_gib=1,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        output=(work/'inner.log').read_text();print(output,end='',flush=True)
        self.assertEqual(observation['exit_code'],0,output)
        self.assertFalse(observation['timed_out'])
        self.assertIsNone(observation['signal'])
        self.assertEqual(observation['remaining_pids'],[])
        self.assertEqual(json.loads((work/'inner-unit-result.json').read_text()),
                         dict(testsRun=expected,failures=0,errors=0,skips=0,success=True))
        self.assertGreater(expected,0)


if __name__=='__main__':
    unittest.main()
