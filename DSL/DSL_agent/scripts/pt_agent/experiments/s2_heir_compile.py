"""Single native HEIR attempt on an unchanged saved Rule63 arithmetic program."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import threading
import time

from .. import paths
from .a_group_runs import append_record, empty_record
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_cnn_preflight import native_artifact_metadata
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'S2-HEIR-QWEN48'
MODE = 'unchanged-rule63-native-heir-five-level-compile'
GIB = 1024**3
MODEL_NAMES = dict(resnet20='resnet20_silu', vgg16='vgg16_silu_avgpool', alexnet='alexnet_silu_avgpool')


def read(path):
    return json.loads(Path(path).read_text())


def parameter_literal(probe):
    """Reuse the unchanged compiler's actual N16/52+8*43/3*60 construction."""
    values = probe['parameters']
    if len(values) != 1:
        raise ValueError('One native residual parameter literal required')
    p = values[0]
    bits = lambda name: [round(math.log2(v)) for v in p[name]]
    if (p['logN'] != 16 or p['log_default_scale'] != 43 or
            bits('Q') != [52] + [43] * 8 or bits('P') != [60] * 3):
        raise ValueError('Native residual modulus bit policy differs from qwen48 prefix')
    return dict(logN=16, Q=p['Q'], P=p['P'], logDefaultScale=43,
                encryptionType='pk', encryptionTechnique='standard')


def parameter_pipeline(original, parameter_file):
    text = original.strip().removeprefix('--pass-pipeline=')
    expected = {'level-budget=10': 2, 'bootstrap-waterline=0': 2,
                'first-mod-bits=55': 1, 'scaling-mod-bits=45': 1}
    for old, count in expected.items():
        if text.count(old) != count:
            raise ValueError('Original native pipeline parameter occurrence differs: ' + old)
    text = text.replace('level-budget=10', 'level-budget=5')
    text = text.replace('bootstrap-waterline=0', 'bootstrap-waterline=1')
    text = text.replace('first-mod-bits=55', 'first-mod-bits=52')
    text = text.replace('scaling-mod-bits=45', 'scaling-mod-bits=43')
    text, count = re.subn(r'parameter-file=[^\s}]+', 'parameter-file=' + str(parameter_file), text)
    if count != 1:
        raise ValueError('One native parameter-file reference required')
    return text


def outcome(observation, guard_error=None, driver_error=None):
    if guard_error:
        return 'resource_limit'
    if observation and observation.get('timed_out'):
        return 'timeout'
    if observation and observation.get('exit_code') != 0:
        return 'compiler_crash'
    if driver_error or not observation:
        return 'driver_error'
    return 'ok'


def inventory(mlir, function):
    """Certify a single straight-line evaluation, otherwise retain static count."""
    text = Path(mlir).read_text()
    functions, current = [], None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'  func\.func (?:private )?@([^\s(]+)', line)
        if match:
            current = dict(name=match[1], line=number, boot_lines=[], control_flow=[], calls=[])
            functions.append(current)
        if current is None:
            continue
        if '= lattigo.ckks.bootstrap ' in line:
            current['boot_lines'].append(number)
        if re.search(r'\b(?:affine|scf)\.(?:for|while|parallel|if)\b', line):
            current['control_flow'].append(number)
        call = re.search(r'\b(?:call|func.call) @([^\s(]+)', line)
        if call:
            current['calls'].append(call[1])
    all_boots = sum(len(x['boot_lines']) for x in functions)
    mapping = {x['name']: x for x in functions}
    body, entry = mapping.get(function + '__preprocessed'), mapping.get(function)
    proved = bool(body and entry and not body['control_flow'] and not body['calls'] and
                  not entry['control_flow'] and entry['calls'].count(body['name']) == 1 and
                  all_boots == len(body['boot_lines']))
    return dict(boot_sites_static=all_boots, boots_compiled=all_boots if proved else None,
                boot_kind='execution_count_from_straight_line_IR' if proved else 'static',
                execution_count_scope='One exported entry invocation; no CKKS execution',
                control_flow_proof=functions,
                loop_count_explanation='Full per-function loop/condition locations and call counts retained; execution multiplicity is claimed only for a straight-line body called once',
                ops_by_native_name=dict(Counter(re.findall(r'\blattigo\.[a-zA-Z_][\w.]*', text))))


def latest_sample(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        rows = stream.read().splitlines()
    for line in reversed(rows):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def peak_rss(resources):
    peak = 0
    if resources.is_file():
        for line in resources.read_text().splitlines():
            sample = json.loads(line)
            rss = sum(x.get('vmrss_kib', 0) for x in sample.get('processes', []))
            rss += (sample.get('outer_process') or {}).get('vmrss_kib', 0)
            peak = max(peak, rss)
    return peak / 1024**2


def append_registry(case, result):
    row = dict(run_id=case.name, round=ROUND, model=result['model'], mode=MODE,
               git_rev=result['git_rev'], metrics=dict(status=result['status'], compile_s=result.get('compile_s'),
               compiler_attempts=result['compiler_attempts'], API=0, CKKS=0),
               artifacts_dir=str(case), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def execute(args):
    from ..registry import git_rev
    started = time.perf_counter_ns()
    out = args.out.resolve()
    if out.parent != paths.RESULTS_DIR.resolve() or out.exists():
        raise ValueError('Use a new direct RESULTS_DIR child')
    out.mkdir()
    (out / 'resources').mkdir()
    (out / 'tmp').mkdir()
    rec = StageRecorder(out / 'events.jsonl')
    result = dict(round=ROUND, run_id=out.name, model=args.model, mode=MODE, status='running', error=None,
                  git_rev=git_rev(), compiler_attempts=0, compile_s=None, guard_error=None,
                  source_check=str(args.source_check), model_forwards=0, sample_reads=0, API=0, CKKS=0,
                  downloads=0, installations=0, integrity_checks=0,
                  limits=dict(native_seconds=7200, AS_gib=24, RSS_gib=14, available_reserve_gib=1.25,
                              case_gib=32, free_disk_gib=10), program_representation_changed=False,
                  scalar_approximation_changed=False, observation=None)
    watcher = None
    stop = threading.Event()
    active = threading.Event()
    old_term = signal.getsignal(signal.SIGTERM)
    resources = out / 'compiler-resources.jsonl'
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(InterruptedError('Native supervisor interrupted')))

    def monitor():
        while not stop.wait(.25):
            try:
                sample = latest_sample(resources)
                rss = sum(x.get('vmrss_kib', 0) for x in (sample or {}).get('processes', [])) * 1024
                rss += ((sample or {}).get('outer_process') or {}).get('vmrss_kib', 0) * 1024
                available = next(int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines()
                                 if x.startswith('MemAvailable:'))
                free, size = shutil.disk_usage(out).free, artifact_bytes(out)
                if rss > 14 * GIB or available < 1.25 * GIB or free < 10 * GIB or size > 32 * GIB:
                    result['guard_error'] = dict(rss_bytes=rss, available_bytes=available, free_bytes=free, case_bytes=size)
                    if active.is_set():
                        os.kill(os.getpid(), signal.SIGTERM)
                    return
            except BaseException as exc:
                result['monitor_error'] = type(exc).__name__ + ': ' + str(exc)
                if active.is_set():
                    os.kill(os.getpid(), signal.SIGTERM)
                return

    try:
        proof = read(args.source_check)
        source = next(x for x in proof['rows'] if x['model'] == args.model)
        if not source['exact_current_approximation_fields_match'] or source['changed_contracts']:
            raise ValueError('Saved HEIR arithmetic must have exactly current A2b Rule63 approximation fields')
        result['source'] = source
        export = read(source['source_export'])
        if not export['weights_preserved']:
            raise ValueError('Original saved weights must be preserved')
        params = parameter_literal(read(args.parameter_probe))
        write_json(out / 'native-parameters.json', params)
        pipeline = parameter_pipeline(args.base_pipeline.read_text(), out / 'native-parameters.json')
        (out / 'pass-pipeline.txt').write_text('--pass-pipeline=' + pipeline + '\n')
        result['parameter_contract'] = dict(source_probe=str(args.parameter_probe), native_parameters=params,
             configured_level_upper_bound=8, greedy_level_budget=5, bootstrap_waterline=1, scale=43,
             original_packing_slots=16384, same_declared_residual_modulus_bits=True,
             residual_modulus_bits=[52] + [43] * 8, special_modulus_bits=[60] * 3,
             exact_qwen48_QP_equality_verified=None,
             full_bootstrap_algorithm_moduli_verified=False,
             equality_scope='Compiler JSON has no concrete Q/P primes. Reuse native HEIR-generated primes with the requested residual bit policy; whole bootstrap and CKKS runtime equivalence remain unmeasured')
        argv = [str(args.heir_opt), '--mlir-disable-threading', '--pass-pipeline=' + pipeline,
                source['source_mlir'], '-o', str(out / 'model-lattigo.mlir'),
                '--ext-const-output-dir=' + str(out / 'resources'), '--ext-const-runtime-load-dir=' + str(out / 'resources')]
        write_json(out / 'command.json', dict(argv=argv, source_program_unchanged=True, parameter_changes_only=True))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'NIX_', '__NIX_'))
               and k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
        env.update(TMPDIR=str(out / 'tmp'), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('r') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            result['compiler_attempts'] = 1
            result['native_launch_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(out / 'progress.json', result)
            with rec.span('native_HEIR_placement_lowering_emission', argv=argv):
                active.set()
                try:
                    obs = run_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=out / 'compiler.log',
                                      resources=resources, timeout=7200, mem_gib=24, fsize_gib=16, env=env)
                finally:
                    active.clear()
                result['observation'] = obs
                result['compile_s'] = (obs['ended_ns'] - obs['started_ns']) / 1e9
            if outcome(obs, result['guard_error']) != 'ok':
                raise RuntimeError('Native HEIR compiler terminated: exit=' + str(obs['exit_code']))
            if obs['remaining_pids'] or obs['signal'] is not None or result.get('monitor_error'):
                raise RuntimeError('Native process cleanup or supervision incomplete')
            actual = native_artifact_metadata(out / 'model-lattigo.mlir')
            if len(actual['parameters']) != 1:
                raise ValueError('One emitted native parameter literal expected')
            emitted = actual['parameters'][0]
            if any(emitted[k] != params[k] for k in ('logN', 'Q', 'P')) or emitted['log_default_scale'] != 43:
                raise ValueError('Emitted native parameters differ from the requested static parameter file')
            details = inventory(out / 'model-lattigo.mlir', MODEL_NAMES[args.model])
            write_json(out / 'compiled-inventory.json', details)
            result.update(compiled_inventory=details, native_artifact=actual)
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        stop.set()
        if watcher:
            watcher.join(5)
        result['monitor_stopped'] = watcher is None or not watcher.is_alive()
        signal.signal(signal.SIGTERM, old_term)
        result['status'] = outcome(result['observation'], result['guard_error'], result['error'])
        result['peak_rss_gib'] = peak_rss(resources)
        if (out / 'compiler.log').is_file():
            log = (out / 'compiler.log').read_text(errors='replace')
            result['first_compiler_error_lines'] = [x[:1500] for x in log.splitlines() if 'error:' in x][:5]
        result['timing'] = summarize_events(read_events(out / 'events.jsonl')) if (out / 'events.jsonl').is_file() else None
        write_json(out / 'result.json', result)
        append_registry(out, result)
        completion = persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                    timing=result['timing'] or dict(status='ok', attribution_ready=False, spans=[], diagnostics=[]),
                    required_artifact_bytes=artifact_bytes(out))
        inv = result.get('compiled_inventory', {})
        row = empty_record(run_id=out.name, task='S2-HEIR', model=args.model, system='heir', param_group='ours-N16-boot5',
              status=result['status'], status_detail=result['error'], budget_s=7200, compile_s=result['compile_s'],
              peak_rss_gib=result['peak_rss_gib'], boots_compiled=inv.get('boots_compiled'),
              boot_sites_static=inv.get('boot_sites_static'), boot_kind=inv.get('boot_kind'),
              pred_latency_s=None, pred_table=str(paths.PROFILES_DIR / 'qwen48/compiler-config.json'),
              prediction_unavailable_reason='Native Lattigo preprocessing/linear-transforms and control flow require a lossless qwen48 operator-level mapping; no such complete mapping is available',
              source=str(out / 'result.json'), finished_utc=datetime.now(timezone.utc).isoformat(), api_calls=0,
              stages_s=dict(compile=result['compile_s']), clock_kind='actual_compile_only', cold=False,
              generation_e2e_s=None, complete_compile_case_s=completion['complete_case_elapsed_seconds'],
              parameter_contract=result.get('parameter_contract'), compiler_attempts=result['compiler_attempts'],
              program_representation_changed=False, loop_count_explanation=inv.get('loop_count_explanation'))
        write_json(out / 'paper-record.json', row)
        if args.paper_runs:
            append_record(args.paper_runs, row)
        print(json.dumps(dict(case=str(out), status=result['status'], compile_s=result['compile_s'],
                             boots_compiled=inv.get('boots_compiled'), boot_sites_static=inv.get('boot_sites_static'),
                             error=result['error']), ensure_ascii=False), flush=True)
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    for name in ('out', 'source-check', 'parameter-probe', 'base-pipeline', 'paper-runs'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--heir-opt', type=Path, default=paths.WORKSPACE / 'baselines/heir-2026.11.1-venv/bin/heir-opt')
    return parser.parse_args(argv)


if __name__ == '__main__':
    execute(arguments())
