"""Pinned-environment public dict/shape checks; no API or encrypted execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import unittest

from hecate_python_env import ROOT, VENV, WORK, enter_nix


FILES = ('scripts/baseline/function_construction.py',
         'scripts/baseline/test_public_mappings.py',
         'scripts/baseline/run_public_mapping_checks.py',
         'third_party/dacapo/python/poly/poly/MPCB.py')


def hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError('Requires actual Poseidon source root')
    if not args.inside:
        command = ('LD_LIBRARY_PATH=$HECATE_PYTHON_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 '
                   'PYTHONPATH=scripts/baseline '+shlex.join([str(VENV/'bin/python'),
                   'scripts/baseline/run_public_mapping_checks.py','--inside']))
        return enter_nix(command,seconds=120)
    import numpy as np
    if np.__version__ != '1.25.2':
        raise ValueError('Requires fixed NumPy 1.25.2')
    before = hashes()
    suite = unittest.defaultTestLoader.loadTestsFromName('test_public_mappings')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged = before == hashes()
    report = dict(schema=1,layer='public-construction-semantics-only',numpy_version=np.__version__,
                  tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                  skipped=len(result.skipped),source_sha256=before,source_unchanged=unchanged,
                  passed=result.wasSuccessful() and not result.skipped and unchanged,
                  planned_shape_comparisons=108,api_calls=0,new_fhe_executions=0,
                  agent_contract_enabled=False,experiment_option='normalize(..., public_mappings=True)',
                  failure_details=[dict(test=str(test),traceback=traceback)
                                   for test,traceback in result.failures+result.errors])
    directory = Path(tempfile.mkdtemp(prefix='public-mapping-checks-',dir=WORK/'results'))
    output = directory/'report.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Public construction evidence:',output,flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
