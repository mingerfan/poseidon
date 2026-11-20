"""E10a (API 0): DaCapo's own ResNet-20 compiled with the CKKS settings our programs use.

E4c compiled DaCapo's hand-written ResNet-20 (SiLU, DaCapo's trained weights) with DaCapo's HEAAN profile: N=2^17 and
bootstraps that restore up to 16 levels. ArrangerDSL programs compile through toolchain.compile_argv(mode, "qwen48"):
compiler_profiles/qwen48/compiler-config.json, N=2^16, at most 8 levels, bootstraps that restore 1-5 levels and
waterline 43. This round traces the unchanged DaCapo benchmark once, then compiles the trace with the qwen48 settings
under DaCapo's search (dacapo-optimized) and under depth-based placement (dacapo-depth-dp), so both placements can be
compared with our programs under one setting. No API call is made and the ledger must stay unchanged.

Outer:  python3 -B -m pt_agent.experiments.e10a_dacapo_settings --root <dir>
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..paths import RESULTS_DIR

ROUND = 'E10'
MODEL = 'ResNet'
REGISTRY_MODEL = 'dacapo-resnet20-silu'
PLACEMENTS = {'optimized': 'dacapo-optimized', 'depth-dp': 'dacapo-depth-dp'}
INPUT = '/out/candidate_trace.mlir'
NIX_SECONDS = 4200
SCRIPTS = Path(__file__).resolve().parents[2]
KEPT = ('complete', 'boots', 'exit', 'seconds', 'peak_rss_mib', 'trace_bytes', 'error', 'timeout')
# Earlier results on the same network, for the summary only.
HEAAN = {'optimized': 19, 'depth-dp': 85}                 # E4c, DaCapo's HEAAN profile
OURS = dict(depth_dp=114, e5b=104, rule=210, agent=91, tuner_only=86)   # our ResNet-20 SiLU programs, qwen48


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + '\n')


def ledger_rows(path):
    path = Path(path)
    return [line for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def trace_path(root):
    return Path(root)/'trace'/'traced'/f'{MODEL}.mlir'


def compile_arguments(placement, work, source):
    """The qwen48 compiler command of our programs, with DaCapo's trace as input, mapped to host paths."""
    from .. import toolchain
    argv = toolchain.compile_argv(PLACEMENTS[placement], 'qwen48', Path(work))
    if argv.count(INPUT) != 1:
        raise ValueError('Expected exactly one compiler input argument')
    argv = [str(source) if arg == INPUT else arg for arg in argv]
    return toolchain.to_host_argv(argv, Path(work))


# ---------------------------------------------------------------- inner workers

def launch_worker(script):
    """Run the unchanged upstream benchmark with the torchvision stub E4c used."""
    import runpy
    from ..models.dacapo_builders import _torchvision_imports
    sys.argv = [str(script)]
    with _torchvision_imports():
        runpy.run_path(str(script), run_name='__main__')
    return 0


def trace_worker(root):
    from .. import toolchain
    from ..paths import DACAPO_ROOT, DACAPO_UPSTREAM, PLATFORM_DIR
    toolchain.setup_env()
    from hecate_python_env import VENV
    work = Path(root)/'trace'
    (work/'traced').mkdir(parents=True, exist_ok=True)
    env = toolchain.package_trace_env()
    env['PYTHONPATH'] = os.pathsep.join(map(str, (SCRIPTS, DACAPO_ROOT/'python/hecate', DACAPO_UPSTREAM/'python/poly',
                                                   PLATFORM_DIR/'deps/poly-python/einops-0.6.1')))
    script = DACAPO_UPSTREAM/f'examples/benchmarks/{MODEL}.py'
    argv = [str(VENV/'bin/python'), '-B', '-m', 'pt_agent.experiments.e10a_dacapo_settings', '--worker', 'launch',
            '--script', str(script)]
    result = toolchain.trusted_run(work, argv, work/'trace.log', env_extra=env)
    path = trace_path(root)
    result.update(script=str(script), trace=str(path), trace_bytes=path.stat().st_size if path.exists() else None)
    result['complete'] = result.get('exit') == 0 and path.is_file()
    write(work/'REPORT.json', result)
    return 0 if result['complete'] else 1


def compile_worker(root, placement):
    from .. import toolchain
    toolchain.setup_env()
    work = Path(root)/placement
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(toolchain.BOOT_CONFIG_DIR/'compiler-config.json', work/'boot-compiler-config.json')
    argv = compile_arguments(placement, work, trace_path(root))
    result = dict(placement=placement, mode=PLACEMENTS[placement], boot_profile='qwen48', argv=argv,
                  config=read(toolchain.BOOT_CONFIG_DIR/'compiler-config.json'), boots=None, complete=False)
    result['config'] = {key: value for key, value in result['config'].items() if key not in ('latencyTable', 'noiseTable')}
    try:
        result.update(toolchain.trusted_run(work, argv, work/'compile.log'))
        plans = sorted(work.glob('lowered._hecate_*.runtime-plan.json'))
        if result.get('exit') == 0 and len(plans) == 1:
            function = plans[0].name[len('lowered._hecate_'):-len('.runtime-plan.json')]
            stats = toolchain.plan_stats(work, function)
            result.update(function=function, plan=stats, boots=stats['boots'], complete=True)
        elif result.get('exit') == 0:
            result['error'] = f'Expected one RuntimePlan, found {len(plans)}'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['pruned'] = toolchain.prune(work)
        write(work/'REPORT.json', result)
    return 0 if result['complete'] else 1


# ---------------------------------------------------------------- outer steps

def register(mode, out, metrics):
    from ..registry import append, git_rev, new_run_id
    append(dict(run_id=new_run_id(ROUND, REGISTRY_MODEL, mode), round=ROUND, model=REGISTRY_MODEL, mode=mode,
                git_rev=git_rev(), metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))


def child(root, step, mode, extra=()):
    out = Path(root)/step
    if (out/'REPORT.json').exists():
        raise RuntimeError(f'Refusing to run {step} twice: {out}/REPORT.json exists')
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(NIX_SECONDS), '--',
               '-m', 'pt_agent.experiments.e10a_dacapo_settings', '--root', str(root), *extra]
    started = time.monotonic()
    print('E10A_STEP='+step+' '+str(out), flush=True)
    with (out/'run.log').open('w') as log:
        code = subprocess.run(command, cwd=SCRIPTS, stdout=log, stderr=subprocess.STDOUT).returncode
    result = read(out/'REPORT.json') if (out/'REPORT.json').exists() else dict(complete=False, error='No child REPORT')
    result.update(nix_exit=code, outer_seconds=time.monotonic()-started)
    write(out/'REPORT.json', result)
    register(mode, out, {key: result.get(key) for key in KEPT if key in result})
    return result


def summarize(root, data):
    show = lambda value: '—' if value is None else str(value)
    lines = ['# E10a：DaCapo 自带的 ResNet-20 在我们的 CKKS 设置下编译（API 0）', '',
             '同一网络（DaCapo 训练好的 ResNet-20 SiLU 权重）。E4c 用 DaCapo 的 HEAAN 配置（N=2^17，自举最高恢复到 16 层）；'
             '本轮用我们程序的 qwen48 配置（N=2^16，至多 8 层，自举恢复 1–5 层，waterline 43）。'
             f'账本 {data.get("api_before")} → {data.get("api_after")}。异常：{data.get("error") or "无"}。', '',
             '| 放置 | HEAAN 配置（E4c） | qwen48 配置（本轮） | 编译秒 | 峰值内存 MiB | 结果 |', '|---|---|---|---|---|---|']
    for placement in PLACEMENTS:
        entry = (data.get('compile') or {}).get(placement) or {}
        lines.append(f'| {placement} | {HEAAN[placement]} | {show(entry.get("boots"))} | {show(entry.get("seconds"))} | '
                     f'{show(entry.get("peak_rss_mib"))} | {"完成" if entry.get("complete") else show(entry.get("error"))} |')
    lines += ['', '我们在同一网络上的程序（qwen48，program-plan 编译）：规则程序 depth-dp 230 / program-plan '
              f'{OURS["rule"]}；E5b 冻结程序 depth-dp {OURS["depth_dp"]} / 规划器 {OURS["e5b"]}；'
              f'E9b Agent {OURS["agent"]}；E9d 只跑精调器 {OURS["tuner_only"]}。']
    (Path(root)/'E10a.md').write_text('\n'.join(lines) + '\n')


def run(args):
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    ledger = RESULTS_DIR/'llm_calls.jsonl'
    data = dict(api_before=len(ledger_rows(ledger)), compile={})
    try:
        data['trace'] = child(root, 'trace', 'e10a-trace', ['--worker', 'trace'])
        if not data['trace'].get('complete'):
            raise RuntimeError('trace failed: ' + str(data['trace'].get('error') or data['trace'].get('log_tail')))
        for placement in PLACEMENTS:
            data['compile'][placement] = child(root, placement, 'e10a-compile-' + placement,
                                               ['--worker', 'compile', '--placement', placement])
    except Exception as exc:
        data['error'] = f'{type(exc).__name__}: {exc}'
    data['api_after'] = len(ledger_rows(ledger))
    if data['api_after'] != data['api_before']:
        data['error'] = (data.get('error') or '') + ' The API ledger changed.'
    data['complete'] = data.get('error') is None and all(e.get('complete') for e in data['compile'].values()) \
        and len(data['compile']) == len(PLACEMENTS)
    write(root/'E10a.json', data)
    summarize(root, data)
    register('e10a-summary', root, {key: data.get(key) for key in ('complete', 'error', 'api_before', 'api_after')})
    print('E10A_DONE=' + str(root/'E10a.json') + ' ERROR=' + str(data.get('error')), flush=True)
    return 0 if data['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--worker', choices=('launch', 'trace', 'compile'))
    parser.add_argument('--placement', choices=tuple(PLACEMENTS))
    parser.add_argument('--script', type=Path)
    args = parser.parse_args()
    if args.worker == 'launch':
        return launch_worker(args.script)
    if args.root is None:
        parser.error('--root is required')
    if args.worker == 'trace':
        return trace_worker(args.root)
    if args.worker == 'compile':
        if args.placement is None:
            parser.error('--worker compile requires --placement')
        return compile_worker(args.root, args.placement)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
