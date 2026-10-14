"""Reproduce object-array semantic checks in existing pinned Nix; no API/FHE."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import unittest

from hecate_python_env import ROOT, VENV, WORK, enter_nix


FILES = ('scripts/baseline/object_arrays.py',
         'scripts/baseline/function_construction.py',
         'scripts/baseline/candidate_contract.py',
         'scripts/baseline/candidate_trace.py',
         'scripts/baseline/hecate_contract.py',
         'scripts/baseline/dsl_grammar_coverage.py',
         'scripts/baseline/candidate_sandbox.py',
         'scripts/baseline/run_candidate.py',
         'scripts/baseline/run_agent_batch.py',
         'scripts/baseline/test_object_arrays.py',
         'scripts/baseline/run_object_array_checks.py',
         'third_party/dacapo/python/hecate/hecate/expr.py',
         'third_party/dacapo/python/poly/poly/MPCB.py')


def hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError('Run from the actual Poseidon source root')
    if not args.inside:
        command = ('LD_LIBRARY_PATH=$HECATE_PYTHON_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 '
                   'PYTHONPATH=scripts/baseline '+shlex.join([str(VENV/'bin/python'),
                   'scripts/baseline/run_object_array_checks.py','--inside']))
        return enter_nix(command,seconds=120)
    import numpy as np
    if np.__version__ != '1.25.2':
        raise ValueError('Authoritative checks require fixed NumPy 1.25.2')
    before = hashes()
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'test_object_arrays.'+name for name in ('ObjectStorageTests','EmptyOracleTests',
            'ObjectConstructionTests','ObjectContractTests','ObjectGoldenPlainTests')])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged = before == hashes()
    report = dict(schema=1,layer='construction-semantics-only',numpy_version=np.__version__,
                  tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                  skipped=len(result.skipped),source_sha256=before,source_unchanged=unchanged,
                  passed=result.wasSuccessful() and not result.skipped and unchanged,
                  api_calls=0,new_fhe_executions=0,agent_contract_enabled=True,
                  llm_generation_validated=False,request_task='hecate-function-synthesis-v18',
                  experiment_option='--object-arrays / normalize(..., object_arrays=True)',
                  failure_details=[dict(test=str(test),traceback=traceback)
                                   for test,traceback in result.failures+result.errors])
    directory = Path(tempfile.mkdtemp(prefix='object-array-checks-',dir=WORK/'results'))
    output = directory/'report.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Construction-only evidence:',output,flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
