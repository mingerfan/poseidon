"""Check trusted AST registration against previously executed native-call artifacts.

Real tracing and compiler calls; no candidate Python exec, paid API or new FHE run.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest
from python_compiler_smoke import BUILD, logged
from seal_cpu_golden import PROFILE, WATERLINE
from seal_artifact_gate import require, inspect_artifacts
from probe_native_calls import SOURCE
from native_function_core_fixtures import PROGRAMS, options
from decorated_functions import register

SCRIPT = Path(__file__).resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('executed', type=Path)
    p.add_argument('--inside', action='store_true')
    p.add_argument('--trace', choices=tuple(PROGRAMS))
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Wrong workspace')
    require(a.executed.resolve().parent == WORK/'results', 'Requires native evidence directory')
    if a.trace:
        require(a.output and a.output.resolve().is_relative_to(WORK/'results') and a.output.is_dir(), 'Invalid output')
        require(Path(sys.prefix) == VENV and os.environ.get('IN_NIX_SHELL'), 'Pinned environment required')
        spec = importlib.util.spec_from_file_location('checked_native_expr', SOURCE)
        hc = importlib.util.module_from_spec(spec); sys.modules[spec.name] = hc
        spec.loader.exec_module(hc)
        functions, plan = register(PROGRAMS[a.trace], {}, hc, **options(a.trace))
        # Dispatch remains through actual hc.Func -> native createCall, not AST flattening.
        hc.save(str(a.output), str(a.output))
        (a.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        (a.output/'source.py').write_text(PROGRAMS[a.trace])
        require(all(f.evaluation_state == 'done' for f in functions.values()), 'Untraced native declaration')
        return 0
    require(not a.output, 'Output is child-only')
    if not a.inside:
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'), str(SCRIPT), str(a.executed.resolve()), '--inside'])
        return enter_nix(command, seconds=900)
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    executed = a.executed.resolve()
    cpu = json.loads((executed/'report.json').read_text())
    require(cpu['status'] == 'passed' and cpu['backend'] == 'upstream_SEAL_HEVM_CPU' and cpu['agent_calls'] == 0,
            'Requires completed actual manual CPU experiment')
    original = Path(cpu['probe']); require(original.parent == WORK/'results', 'Unexpected original probe')
    require(digest(original/'report.json') == cpu['probe_sha256'], 'Original probe changed')
    compile_report = json.loads((original/'report.json').read_text())
    require(compile_report['frontend_library_sha256'] == digest(BUILD/'lib/libHecateFrontend.so') and
            compile_report['profile_sha256'] == digest(PROFILE), 'Native library/profile changed')
    env = dict(os.environ, HECATE=str(WORK/'build-dacapo/hecate-python-root'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    result = Path(tempfile.mkdtemp(prefix='checked-native-functions-', dir=WORK/'results'))
    print('Checked native AST evidence:', result, flush=True)
    report = dict(schema=1, status='running', cases=[], new_api_calls=0, new_fhe_executions=0,
        agent_contract_enabled=True, agent_generation_validated=False, executed_report=str(executed/'report.json'),
        executed_report_sha256=digest(executed/'report.json'), frontend_source_sha256=digest(SOURCE),
        checker_sha256=digest(SCRIPT.with_name('decorated_functions.py')),
        fixtures_sha256=digest(SCRIPT.with_name('native_function_core_fixtures.py')),
        compiler_sha256=digest(BUILD/'bin/hecate-opt'), profile_sha256=digest(PROFILE))
    for name in PROGRAMS:
        folder = result/name; folder.mkdir()
        cmd = [str(VENV/'bin/python'),str(SCRIPT),str(executed),'--trace',name,'--output',str(folder)]
        row = dict(case=name,trace_exit=logged(cmd, folder/'trace.log', seconds=40, env=env), matched=False)
        report['cases'].append(row)
        if row['trace_exit'] == 0:
            earth = folder/'probe_checked_native_functions.mlir'
            require(earth.exists(), 'Expected Earth artifact missing')
            row['earth_sha256'] = digest(earth)
            cmd = [str(BUILD/'bin/hecate-opt'),str(earth),'--eva','--ckks-config='+str(PROFILE),
                   '--waterline='+str(WATERLINE),'--mlir-disable-threading','--verify-each','-o',str(folder/'lowered.mlir')]
            row['compile_exit'] = logged(cmd,folder/'compile.log',seconds=60,env=env)
            if row['compile_exit'] == 0:
                baseline = next(r for r in cpu['cases'] if r['case'] == name and r['style'] == 'call')
                rawdir = executed/(name+'-call-'+str(baseline['group']))
                require(baseline['execution']['encrypted_execution'] and baseline['comparison']['passed'], 'No real correct execution')
                same = True
                for filename in ('lowered._hecate_golden.hevm','_hecate_golden.cst'):
                    require(digest(rawdir/filename) == baseline['frozen_hashes'][filename], 'Executed artifact changed')
                    row[filename] = digest(folder/filename)
                    same = same and row[filename] == baseline['frozen_hashes'][filename]
                row['artifact'] = inspect_artifacts((folder/'lowered._hecate_golden.hevm').read_bytes(),
                     (folder/'_hecate_golden.cst').read_bytes(),expected_inputs=len(options(name)['input_names']))
                row['matched'] = same
        (result/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name, 'MATCHED_EXECUTED_ARTIFACT' if row['matched'] else 'NOT_MATCHED', flush=True)
    require(report['checker_sha256'] == digest(SCRIPT.with_name('decorated_functions.py')), 'Checker changed during probe')
    report['status'] = 'passed' if all(c['matched'] for c in report['cases']) else 'failed'
    (result/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status'] == 'passed' else 1


if __name__=='__main__': raise SystemExit(main())
