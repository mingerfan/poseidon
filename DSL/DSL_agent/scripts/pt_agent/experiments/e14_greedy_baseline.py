"""E14 (API 0): the greedy bootstrap placement of our DaCapo build on the Table II programs, under the qwen48 setting.

Table II compares DaCapo's search, the depth-based placement and Arranger's planner on the same programs. Our
DaCapo build also has a greedy placement (`--boot-placement=greedy`). This round compiles every Table II program once
more with that placement, changing exactly one compiler argument in a copy of the program's package, as M7 did for
the search: the four CIFAR programs of R8e7 (exported again with the depth-based placement first, which also gives
their depth-based compile time), the BERT-tiny and Qwen2 programs of A1 (their exported packages), and DaCapo's own
ResNet-20 (the E10a trace). No library or compiler change; no API call; no report data.

Outer:  python3 -B -m pt_agent.experiments.e14_greedy_baseline --root <dir> [--programs ...]
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from ..paths import RESULTS_DIR

ROUND = 'E14'
OLD, NEW = '--boot-placement=depth-dp', '--boot-placement=greedy'
SCRIPTS = Path(__file__).resolve().parents[2]
NIX_SECONDS = 4200
CNN_SOURCES = {name: RESULTS_DIR/'r8e7-reconfirm-20261022T122558'/'models'/name
               for name in ('mlp', 'cnn_plain', 'resnet20_silu', 'resnet20')}
A1 = RESULTS_DIR/'A1-a624-20261102'
TRANSFORMER_ROOTS = {name: A1/name/'acceptance-replay-session-01' for name in ('bert', 'qwen')}
DACAPO_TRACE = RESULTS_DIR/'E10-baselines-20261111'/'a'/'trace'/'traced'/'ResNet.mlir'
PROGRAMS = ('dacapo-resnet20', *CNN_SOURCES, *TRANSFORMER_ROOTS)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + '\n')


def swap(argv):
    """The argument list with the depth-based placement replaced by the greedy one."""
    if argv.count(OLD) != 1 or NEW in argv:
        raise RuntimeError('Expected exactly one depth-dp placement argument')
    argv = list(argv)
    argv[argv.index(OLD)] = NEW
    return argv


def swap_package(source, out):
    """Copy an exported package into `out` and switch its compile arguments to the greedy placement."""
    package = out/Path(source).name
    shutil.copytree(source, package)
    path = package/'config/compile_argv.json'
    write(path, swap(read(path)))
    return package


# ---------------------------------------------------------------- inner workers

def cnn_worker(name, out):
    from .. import toolchain
    from ..run import _compile_export, _inside
    toolchain.setup_env()
    depth = out/'depth-dp'
    depth.mkdir(parents=True)
    for file in ('program.ptdsl', 'calibration.json'):
        shutil.copyfile(CNN_SOURCES[name]/file, depth/file)
    result = dict(complete=False, program=name, source=str(CNN_SOURCES[name]), limits=toolchain.TRUSTED_LIMITS)
    started = time.monotonic()
    _inside(SimpleNamespace(out=depth, model=name, models_dir=RESULTS_DIR/'models', planner='rule', deg=None,
                            margin=None, refresh='compiler', act_method='cheb', stages=['check', 'export', 'compile'],
                            final=False))
    reference = read(depth/'REPORT.json')
    result['depth_dp'] = dict(complete=reference.get('complete'), boots=(reference.get('plan') or {}).get('boots'),
                              compile_s=(reference.get('compile') or {}).get('seconds'),
                              stage_s=time.monotonic()-started, error=reference.get('error'))
    write(out/'REPORT.json', result)
    if not reference.get('complete'):
        result['error'] = 'depth-dp export or compile failed'
        write(out/'REPORT.json', result)
        return 1
    greedy = out/'greedy'
    greedy.mkdir()
    package = swap_package(reference['export']['package'], greedy)
    compiled = _compile_export(dict(package=str(package), weights=reference['export']['weights']))
    plan = compiled.get('plan') or {}
    stage = compiled.get('compile') or {}
    result['greedy'] = dict(complete=stage.get('exit') == 0 and bool(plan), boots=plan.get('boots'),
                            compile_s=stage.get('seconds'), timeout=stage.get('timeout'), exit=stage.get('exit'),
                            log_tail=(stage.get('log_tail') or '')[-400:])
    result['complete'] = result['greedy']['complete']
    result['pruned'] = {key: toolchain.prune(greedy/key) for key in ('trace', 'compiled') if (greedy/key).exists()}
    write(out/'REPORT.json', result)
    return 0 if result['complete'] else 1


def transformer_worker(name, out):
    from .. import toolchain
    from .u4_relative_acceptance import trace_environment
    toolchain.setup_env()
    from hecate_python_env import VENV
    reference = read(TRANSFORMER_ROOTS[name]/'compiler/REPORT.json')
    package = swap_package(reference['package'], out)
    function = read(package/'config/model.json')['name']
    result = dict(complete=False, program=reference['program'], package=str(package), weights=reference['weights'],
                  depth_dp=dict(boots=(reference.get('stats') or {}).get('boots'),
                                compile_s=(reference.get('compile') or {}).get('seconds')))

    def entry(kind, directory, extra=()):
        command = [str(VENV/'bin/python'), '-B', '-m', 'pt_agent.experiments.u4_native_continuation',
                   '--package', str(package), '--weights', reference['weights'], '--mode', 'compiler', '--out', str(out),
                   '--hecate-opt', str(toolchain.host_paths()['hecate_opt']), '--entry', kind,
                   '--directory', str(directory), *extra]
        return toolchain.trusted_run(out, command, out/(kind+'.log'), env_extra=trace_environment())

    result['trace'] = entry('trace', out/'trace')
    bytecode = out/'trace/trace_model.mlirbc'
    if result['trace']['exit'] or not bytecode.is_file():
        result['error'] = 'trace failed'
        write(out/'REPORT.json', result)
        return 1
    stage = entry('compile', out/'compiled', ('--input', str(bytecode)))
    result['greedy'] = dict(complete=stage['exit'] == 0, compile_s=stage.get('seconds'), timeout=stage.get('timeout'),
                            exit=stage.get('exit'), log_tail=(stage.get('log_tail') or '')[-400:])
    if stage['exit'] == 0:
        result['greedy']['boots'] = toolchain.plan_stats(out/'compiled', function)['boots']
    result['complete'] = result['greedy']['complete']
    result['pruned'] = {key: toolchain.prune(out/key) for key in ('trace', 'compiled') if (out/key).exists()}
    write(out/'REPORT.json', result)
    return 0 if result['complete'] else 1


def dacapo_worker(out):
    from .. import toolchain
    from .e10a_dacapo_settings import compile_arguments
    toolchain.setup_env()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(toolchain.BOOT_CONFIG_DIR/'compiler-config.json', out/'boot-compiler-config.json')
    argv = compile_arguments('depth-dp', out, DACAPO_TRACE)
    argv = swap(argv)
    result = dict(complete=False, program='dacapo-resnet20', argv=argv)
    result['greedy'] = stage = toolchain.trusted_run(out, argv, out/'compile.log')
    plans = sorted(out.glob('lowered._hecate_*.runtime-plan.json'))
    if stage.get('exit') == 0 and len(plans) == 1:
        function = plans[0].name[len('lowered._hecate_'):-len('.runtime-plan.json')]
        stage.update(boots=toolchain.plan_stats(out, function)['boots'], complete=True, compile_s=stage.get('seconds'))
        result['complete'] = True
    else:
        stage.update(complete=False, compile_s=stage.get('seconds'))
    result['pruned'] = toolchain.prune(out)
    write(out/'REPORT.json', result)
    return 0 if result['complete'] else 1


# ---------------------------------------------------------------- outer steps

def leftovers(out):
    keep, pid = set(), os.getpid()
    while pid > 1:
        keep.add(pid)
        pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in keep:
            continue
        try:
            command = (entry/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if str(out) in command:
            found.append(dict(pid=int(entry.name), command=command[:200]))
    return found


def register(program, out, metrics):
    from ..registry import append, git_rev, new_run_id
    append(dict(run_id=new_run_id(ROUND, program, 'greedy-placement'), round=ROUND, model=program,
                mode='greedy-placement', git_rev=git_rev(), metrics=metrics, artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()))


def child(root, program):
    out = Path(root)/program
    if (out/'REPORT.json').exists():
        raise RuntimeError(f'Refusing to run {program} twice')
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(NIX_SECONDS), '--',
               '-m', 'pt_agent.experiments.e14_greedy_baseline', '--worker', program, '--out', str(out)]
    started = time.monotonic()
    print('E14_STEP='+program, flush=True)
    with (out/'run.log').open('w') as log:
        code = subprocess.run(command, cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT).returncode
    killed = leftovers(out)
    for item in killed:
        try:
            os.kill(item['pid'], signal.SIGKILL)
        except ProcessLookupError:
            item['gone'] = True
    result = read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False, error='No child REPORT')
    result.update(nix_exit=code, outer_seconds=time.monotonic()-started, leftover_processes_killed=killed)
    write(out/'REPORT.json', result)
    register(program, out, {key: result.get(key) for key in ('complete', 'error', 'depth_dp', 'nix_exit', 'outer_seconds')}
             | dict(greedy={key: (result.get('greedy') or {}).get(key) for key in ('complete', 'boots', 'compile_s', 'timeout')}))
    return result


def summarize(root, data):
    show = lambda value: '—' if value is None else (f'{value:.2f}' if isinstance(value, float) else str(value))
    lines = ['# E14：greedy 放置（qwen48，API 0）', '',
             f'账本 {data.get("api_before")} → {data.get("api_after")}。异常：{data.get("error") or "无"}。', '',
             '| 程序 | depth-based 自举 / 编译秒 | greedy 自举 / 编译秒 | greedy 结果 |', '|---|---|---|---|']
    for program, entry in (data.get('programs') or {}).items():
        depth, greedy = entry.get('depth_dp') or {}, entry.get('greedy') or {}
        outcome = '完成' if greedy.get('complete') else ('超时' if greedy.get('timeout') else str(entry.get('error') or greedy.get('exit')))
        lines.append(f'| {program} | {show(depth.get("boots"))} / {show(depth.get("compile_s"))} | '
                     f'{show(greedy.get("boots"))} / {show(greedy.get("compile_s"))} | {outcome} |')
    (Path(root)/'E14.md').write_text('\n'.join(lines)+'\n')


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    rows = lambda: sum(1 for line in ledger.read_text().splitlines() if line.strip())
    data = dict(api_before=rows(), programs={})
    error = None
    try:
        for program in PROGRAMS:
            if args.programs and program not in args.programs:
                continue
            data['programs'][program] = child(root, program)
            write(root/'E14.json', data)
            summarize(root, data)
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    data.update(api_after=rows(), error=error)
    if data['api_after'] != data['api_before']:
        data['error'] = (error or '') + ' The API ledger changed.'
    write(root/'E14.json', data)
    summarize(root, data)
    print('E14_DONE='+str(root/'E14.json')+' ERROR='+str(data['error']), flush=True)
    return 0 if data['error'] is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--programs', nargs='+', choices=PROGRAMS)
    parser.add_argument('--worker', choices=PROGRAMS)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.worker == 'dacapo-resnet20':
            return dacapo_worker(args.out)
        if args.worker in CNN_SOURCES:
            return cnn_worker(args.worker, args.out)
        return transformer_worker(args.worker, args.out)
    if args.root is None:
        parser.error('--root is required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
