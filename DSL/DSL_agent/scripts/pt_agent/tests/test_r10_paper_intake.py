"""Run R10 real-tensor checks in the pre-existing intake environment."""

from pathlib import Path
import re
import resource
import subprocess
import sys
import unittest

from pt_agent.paths import INTAKE_PYTHON
from pt_agent.tests._artifacts import require
from pt_agent.tests._r10_intake_cases import INNER_CASE_COUNT


INTAKE_ARTIFACT = INTAKE_PYTHON.parent if INTAKE_PYTHON.is_symlink() else INTAKE_PYTHON


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))


@require(INTAKE_ARTIFACT)
class IntakeEnvironmentChecks(unittest.TestCase):
    def test_real_intake_cases(self):
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '180',
                   '--python', str(INTAKE_PYTHON), '--', '-m', 'unittest',
                   'pt_agent.tests._r10_intake_cases', '-v']
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=300, preexec_fn=limited)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        count = re.search(r'Ran (\d+) tests? in ', completed.stdout)
        self.assertIsNotNone(count, completed.stdout)
        self.assertEqual(int(count.group(1)), INNER_CASE_COUNT, completed.stdout)
        self.assertNotRegex(completed.stdout, r'skipped=\d+')
        print('R10_INTAKE_INNER_TESTS='+str(count.group(1)), flush=True)


if __name__ == '__main__':
    unittest.main()
