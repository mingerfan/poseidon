"""Run each actual native ABI boundary test in a fresh process; no API or FHE."""
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest
from python_compiler_smoke import BUILD, logged
from probe_native_calls import SOURCE
from seal_artifact_gate import require
from test_native_function_calls import NativeFunctionTests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Wrong workspace')
    if not args.inside:
        cmd = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'), str(Path(__file__).resolve()), '--inside'])
        return enter_nix(cmd, seconds=900)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    compatibility = WORK/'build-dacapo/hecate-python-root'
    require((compatibility/'build').resolve() == BUILD, 'Missing frontend library link')
    env = dict(os.environ, HECATE=str(compatibility), POSEIDON_NATIVE_CALLS_LIVE='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    out = Path(tempfile.mkdtemp(prefix='native-call-boundaries-', dir=WORK/'results'))
    tests = sorted(n for n in dir(NativeFunctionTests) if n.startswith('test_'))
    report = dict(schema=1, status='running', cases=[], agent_calls=0, encrypted_execution=False,
        source_sha256=digest(SOURCE), library_sha256=digest(BUILD/'lib/libHecateFrontend.so'),
        tests_sha256=digest(Path(__file__).with_name('test_native_function_calls.py')))
    print('Native boundary evidence:', out, flush=True)
    for test in tests:
        command = [str(VENV/'bin/python'), str(Path(__file__).with_name('test_native_function_calls.py')),
                   'NativeFunctionTests.'+test, '-v']
        code = logged(command, out/(test+'.log'), seconds=40, env=env)
        report['cases'].append(dict(test=test, exit_code=code, passed=code == 0))
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(test, 'PASS' if code == 0 else 'FAIL', flush=True)
    require(digest(SOURCE) == report['source_sha256'], 'Source changed during probe')
    report['status'] = 'passed' if all(r['passed'] for r in report['cases']) else 'failed'
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
