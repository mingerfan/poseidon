"""Build the original Orbit MLIR interpreter without running it.

Orbit's official Go backend reads optimized Earth MLIR at runtime.  This entry
binds the existing optimizer artifact to a new package and builds that original
interpreter.  Its binary build is recorded separately from model optimization;
it does not claim a model-specific native compiler or ciphertext execution.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'ORBIT-BACKEND-BUILD-ONLY'
MODE = 'original-mlir-interpreter-build-and-optimized-model-package'
MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
GIB = 1024 ** 3
ZERO_FIELDS = ('model_forward_runs', 'sample_reads', 'constant_payload_reads', 'optimizer_runs',
               'solver_runs', 'llm_calls', 'keygen_runs', 'encryptions', 'decryptions',
               'real_ckks_attempts', 'real_ckks_completed', 'report_samples',
               'confirm_samples', 'downloads', 'installations', 'extra_integrity_checks')
FORMAL_FIELDS = ('compiled_bootstraps', 'model_backend_compile_seconds', 'CKKS_latency',
                 'predicted_latency', 'accuracy_acceptance', 'security_acceptance')


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('A saved object record is required')
    return value


def _file(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('A required existing file is absent: ' + str(path))
    return dict(path=str(path), bytes=path.stat().st_size)


def _same(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def source_gate(source_case):
    """Bind the successful original ResNet optimizer and original constant path."""
    case = Path(source_case).resolve()
    result = read(case / 'result.json')
    if (result.get('round') != 'R23' or result.get('model') != MODEL
            or result.get('status') != 'ok' or result.get('native_optimizer_completed') is not True
            or result.get('placement_artifact_checked') is not True):
        raise ValueError('Require the original successful full ResNet Orbit optimizer case')
    for phase in ('optimize', 'validate'):
        if (result.get('attempted', {}).get(phase) != 1
                or result.get('completed', {}).get(phase) != 1):
            raise ValueError('Retain the one completed optimization and validation')
    phases = result['phase_results']; optimized = phases['optimize']; validated = phases['validate']
    for row in (optimized, validated):
        if (row.get('status') != 'ok' or row.get('model') != MODEL
                or row.get('source_checked') is not True or row.get('environment_restored') is not True):
            raise ValueError('Original worker source and terminal gates differ')
    if (validated.get('placement_artifact_checked') is not True
            or validated.get('check_tdag_calls') != 1 or validated.get('check_tdag_completed') != 1):
        raise ValueError('The actual optimized model must pass the original TDAG checker')
    mlir = case / 'work/optimized.mlir'
    for row in (optimized, validated):
        stored = row.get('output_mlir') or {}
        if not _same(stored.get('path'), mlir) or stored.get('size_bytes') != mlir.stat().st_size:
            raise ValueError('Retain the exact recorded optimized MLIR path and size metadata')
    params = validated['params']
    expected = dict(poly_deg=131072, max_slot=65536, Sf=51, Sw=40, Csw=40,
                    lvl_ub=16, bts_lb=3, bts_ub=16, dacapo_mlir_in=True, dacapo_mlir_out=True)
    if any(params.get(key) != value for key, value in expected.items()):
        raise ValueError('The original 64k/native16/scale51 profile is required')
    process_rows = [row for row in result['processes'] if row.get('stage') in ('model-optimize', 'model-validate')]
    if {row['stage'] for row in process_rows} != {'model-optimize', 'model-validate'} or len(process_rows) != 2:
        raise ValueError('Both original optimizer supervision records are required')
    for row in process_rows:
        observed = row['observation']
        if (not {'exit_code', 'timed_out', 'signal', 'signals_sent', 'remaining_pids'}.issubset(observed)
                or type(observed['exit_code']) is not int or observed['exit_code'] != 0
                or observed.get('timed_out') is not False
                or not isinstance(observed['signals_sent'], list) or not isinstance(observed['remaining_pids'], list)
                or observed.get('signal') is not None or observed.get('signals_sent')
                or observed.get('remaining_pids')):
            raise ValueError('Preserve only cleanly completed original optimizer processes')
    frontend_path = Path(result['source_records']['r22_result']).resolve()
    frontend = read(frontend_path)
    if frontend.get('round') != 'R22' or frontend.get('status') != 'ok' or frontend.get('model') != MODEL:
        raise ValueError('The optimizer must retain the successful original full-model frontend')
    constants = frontend['phase_results']['trace']['constant_files']
    if len(constants) != 1:
        raise ValueError('One original full-model constant file is required')
    constant = _file(constants[0]['path'])
    if constant['bytes'] != constants[0]['size_bytes']:
        raise ValueError('Original constant-file metadata differs')
    return dict(source_result=str(case / 'result.json'), frontend_result=str(frontend_path),
                model=MODEL, optimized_mlir=_file(mlir), constants=constant,
                optimizer_params=params, placement_bootstrap_nodes=result['placement_bootstrap_nodes'],
                placement_bootstrap_weight_sum=result['placement_bootstrap_weight_sum'],
                prior_optimizer_times_included=False, model_payload_or_inputs_read=False,
                scope='Original optimized Earth MLIR plus a separately built generic Orbit interpreter')


def build_command(go_sdk, output):
    """The official backend package is compiled, and its main is never run."""
    return [str(Path(go_sdk) / 'bin/go'), 'build', '-p=1', '-buildvcs=false', '-mod=readonly',
            '-o', str(output), './lowering/fhe']


def native_environment(work, go_sdk, original=None):
    original = os.environ if original is None else original
    env = {key: value for key, value in original.items()
           if not key.startswith(('GO', 'LD_', 'NIX_', '__NIX_', 'HECATE_'))
           and key not in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    work, go_sdk = Path(work), Path(go_sdk)
    env.update(GOROOT=str(go_sdk), GOOS='linux', GOARCH='arm64', GO111MODULE='on',
               GOPATH=str(work/'gopath'), GOMODCACHE=str(work/'module-cache'), GOCACHE=str(work/'build-cache'),
               GOTMPDIR=str(work/'tmp'), TMPDIR=str(work/'tmp'), TEST_TELEMETRY_DIR=str(work/'telemetry'),
               GOENV='off', GOWORK='off', GOTOOLCHAIN='local', GOPROXY='off', GOSUMDB='off',
               GOAUTH='off', GOVCS='*:off', GOPRIVATE='', GONOPROXY='', GONOSUMDB='',
               CGO_ENABLED='0', GOMAXPROCS='1', GOMEMLIMIT='3GiB')
    return env


def receive_signal(result, signum):
    """Keep our guard stop separate from an external cancellation."""
    if signum == signal.SIGUSR1 and (result.get('resource_guard') or result.get('resource_monitor_error')):
        result['guard_stop_signal'] = signum
    else:
        result['incoming_signal'] = signum
    raise InterruptedError('Orbit interpreter build interrupted')


def prepare_sources(args, out, source):
    """Copy source/cache evidence and a small MLIR; never open constant payloads."""
    work = out/'work'; work.mkdir()
    for name in ('gopath', 'tmp', 'telemetry'):
        (work/name).mkdir()
    (work/'telemetry/mode').write_text('off\n')
    snapshot = out/'source'; snapshot.mkdir()
    shutil.copyfile(__file__, snapshot/Path(__file__).name)
    package = args.orbit_root/'backend/lattigo/lowering/fhe'
    copied = []
    for path in sorted(package.glob('*.go')):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Require original regular backend Go sources')
        shutil.copyfile(path, snapshot/path.name); copied.append(_file(path))
    if not {'main.go', 'parser.go', 'scheme.go', 'evaluator.go'}.issubset({Path(row['path']).name for row in copied}):
        raise ValueError('Complete original interpreter source package is required')
    for name in ('go.mod', 'go.sum'):
        shutil.copyfile(args.orbit_root/'backend/lattigo'/name, snapshot/name)
    shutil.copyfile(args.go_sdk/'VERSION', snapshot/'go-version.txt')
    shutil.copytree(args.module_cache, work/'module-cache', symlinks=False)
    (out/'model').mkdir(); shutil.copyfile(source['optimized_mlir']['path'], out/'model/optimized.mlir')
    manifest = dict(source, packaged_mlir=str(out/'model/optimized.mlir'),
                    constant_payload_copied=False, original_constant_file_required=True,
                    interpreter_package=str(package), interpreter_sources=copied,
                    model_specific_native_compiler_available=False, runtime_execution_scheduled=False,
                    no_execution_reason='Run and --plain generate secret/public/relinearization keys; this entry only builds')
    write_json(out/'package.json', manifest)
    return work, manifest


def _resource_observation(args, out, active, started):
    from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
    from .r31_lattigo_runtime_plan import physical_memory_observation
    observation = physical_memory_observation(_latest_rss_sample(active), _available_ram_bytes(),
                                              rss_gib=4, minimum_available_gib=1.25)
    observation.update(case_bytes=artifact_bytes(out), free_bytes=shutil.disk_usage(out).free,
                       elapsed_seconds=(time.perf_counter_ns()-started)/1e9)
    if observation['case_bytes'] > args.case_gib*GIB-GIB//1024:
        observation['reasons'].append('case artifact allowance exhausted')
    if observation['free_bytes'] < 10*GIB:
        observation['reasons'].append('10GiB disk reserve exhausted')
    if observation['elapsed_seconds'] >= args.seconds:
        observation['reasons'].append('complete build-only wall allowance exhausted')
    observation['stop'] = bool(observation['reasons'])
    return observation


def _persist(args, out, result, recorder, started, revision):
    result['recorded_pre_persistence_seconds'] = (time.perf_counter_ns()-started)/1e9
    result['timing'] = summarize_events(read_events(out/'events.jsonl'))
    result['metrics'] = {key: result.get(key) for key in ('status', 'runtime_interpreter_compiled',
                         'model_specific_native_compiler_available', 'go_build_attempts', 'go_build_completed')}
    result['metrics'].update(dict.fromkeys(FORMAL_FIELDS)); result['metrics'].update(dict.fromkeys(ZERO_FIELDS, 0))
    result['artifact_bytes'] = 0
    for _ in range(20):
        write_json(out/'result.json', result); size = artifact_bytes(out)
        if size+GIB//1024 > args.case_gib*GIB and not result.get('terminal_budget_error'):
            result.update(status='error', terminal_budget_error='Completion reserve exhausted')
            result['metrics']['status'] = 'error'
            result['artifact_bytes'] = size
            continue
        if size == result['artifact_bytes']:
            break
        result['artifact_bytes'] = size
    else:
        raise RuntimeError('Build-only result size did not converge')
    with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('Preserve the existing registry identity')
        row = dict(run_id=out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=revision,
                   metrics=result['metrics'], artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                            timing=result['timing'], required_artifact_bytes=artifact_bytes(out))


def execute(args):
    """One original interpreter build, with no launch of its output binary."""
    out = args.out.resolve()
    if (out.exists() or out.is_symlink() or not out.is_relative_to(paths.RESULTS_DIR.resolve())
            or out == paths.RESULTS_DIR.resolve()):
        raise ValueError('Use a fresh isolated configured results case')
    out.mkdir(); started = time.perf_counter_ns(); recorder = StageRecorder(out/'events.jsonl')
    from ..registry import git_rev
    revision = git_rev()
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
                  runtime_interpreter_compiled=False, model_specific_native_compiler_available=False,
                  model_backend_compiled=None, go_build_attempts=0, go_build_completed=0,
                  process=None, cleanup_unknown=False, native_go_build_mechanism=True,
                  incoming_signal=None, guard_stop_signal=None,
                  native_verification_scope='Existing offline Go compiler/module mechanisms only; no added integrity checks',
                  limits=dict(address_space_gib=12, subtree_RSS_gib=4, minimum_host_available_gib=1.25,
                              file_gib=8, minimum_free_disk_gib=10, case_gib=args.case_gib,
                              build_seconds=900, complete_seconds=args.seconds, concurrency=1),
                  **dict.fromkeys(ZERO_FIELDS, 0), **dict.fromkeys(FORMAL_FIELDS))
    stop = threading.Event(); failures = []; thread = None; active = out/'build-resources.jsonl'
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    def interrupted(sig, frame):
        receive_signal(result, sig)
    for sig in old:
        signal.signal(sig, interrupted)
    def watch():
        while not stop.wait(.5):
            try:
                observation = _resource_observation(args, out, active, started)
                if not observation['stop']:
                    continue
                failures.append(observation); result['resource_guard'] = observation
                try:
                    write_json(out/'resource-guard.json', observation)
                finally:
                    os.kill(os.getpid(), signal.SIGUSR1)
                return
            except BaseException as exc:
                if not failures:
                    failures.append(str(exc)); result['resource_monitor_error'] = str(exc)
                    os.kill(os.getpid(), signal.SIGUSR1)
                return
    try:
        thread = threading.Thread(target=watch, daemon=True); thread.start()
        with recorder.span('source_admission'):
            source = source_gate(args.source_case); result['source'] = source
            for path in (args.go_sdk/'bin/go', args.go_sdk/'VERSION', args.orbit_root/'backend/lattigo/go.mod', args.orbit_root/'backend/lattigo/go.sum'):
                _file(path)
            if not args.module_cache.is_dir():
                raise ValueError('The existing offline module cache is required')
        with recorder.span('isolated_backend_source_and_model_package'):
            work, manifest = prepare_sources(args, out, source)
            result['package'] = manifest
        observation = _resource_observation(args, out, active, started)
        if observation['stop']:
            raise RuntimeError(str(observation['reasons']))
        command = build_command(args.go_sdk, out/'orbit-fhe'); write_json(out/'build-argv.json', command)
        remaining = args.seconds-(time.perf_counter_ns()-started)/1e9
        if remaining <= 15:
            raise TimeoutError('No bounded build allowance remains')
        result['go_build_attempts'] = 1; result['cleanup_unknown'] = True; write_json(out/'result.json', result)
        with recorder.span('official_runtime_interpreter_build'):
            process = run_process(command, cwd=args.orbit_root/'backend/lattigo', log=out/'build.log',
                                  resources=active, timeout=min(900, remaining-10), mem_gib=12, fsize_gib=8,
                                  env=native_environment(work, args.go_sdk))
        result.update(process=process, cleanup_unknown=False)
        if (failures or process['exit_code'] != 0 or process['timed_out'] or process['signal'] is not None
                or process['remaining_pids']):
            raise RuntimeError('The one original interpreter build did not finish cleanly')
        result.update(status='ok', go_build_completed=1, runtime_interpreter_compiled=True,
                      binary=_file(out/'orbit-fhe'))
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        try:
            _persist(args, out, result, recorder, started, revision)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-case', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--orbit-root', type=Path, default=paths.ORBIT_ROOT)
    p.add_argument('--go-sdk', type=Path, default=paths.ORBIT_ROOT.parent/'toolchains/go125/go')
    p.add_argument('--module-cache', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=1200); p.add_argument('--case-gib', type=int, default=8)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if not 1 <= args.seconds <= 1200 or not 4 <= args.case_gib <= 16:
        p.error('Use finite wall allowance<=1200s and case4..16GiB; original build is capped900s')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    value = execute(args)
    print(json.dumps({key: value.get(key) for key in ('status', 'error', 'runtime_interpreter_compiled', 'model_backend_compiled')}))
    return int(value['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
