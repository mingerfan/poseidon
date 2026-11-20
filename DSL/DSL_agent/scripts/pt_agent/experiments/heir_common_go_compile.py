"""Continue successful HEIR VGG/Alex lowering through split Go library build.

The two original translator modes and offline library command are the ones
already used for ResNet. Generated initializers, configure/preprocessing and
model functions are compiled, never executed by this entry. No source MLIR,
parameter literal, model, polynomial or original library is rewritten.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_initializer_probe import (durable_json, physical_observation, serial_lock,
    interruption_status, receive_signal, observe_termination)
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample

ROUND = 'HEIR-COMMON-GO-COMPILE'
MODE = 'source-bound-original-split-lattigo-offline-library-build'
MODULE = 'pt_agent.experiments.heir_common_go_compile'
MODELS = {'vgg16': 'vgg16_silu_avgpool', 'alexnet': 'alexnet_silu_avgpool'}
GIB = 1024**3
TOTAL_SECONDS = 1800
PHASE_SECONDS = dict(utils_emission=300, main_emission=300, library_build=600)
CASE_BYTES = 8*GIB
RESERVE_BYTES = 1024**2
ZERO_FIELDS = ('llm_calls', 'downloads', 'installations', 'sample_reads', 'model_forwards',
    'heldout_samples', 'confirm_samples', 'report_samples', 'heir_lowering_runs',
    'generated_binary_runs', 'go_package_initializations', 'configure_runs',
    'preprocessing_runs', 'context_constructions', 'keygen_runs', 'encryptions',
    'decryptions', 'real_ckks_attempts', 'real_ckks_completed', 'extra_integrity_checks')
FORMAL_FIELDS = ('accuracy_acceptance', 'predicted_latency', 'CKKS_latency', 'security_acceptance')


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):raise ValueError('Object source metadata required')
    return value


def _same(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _file(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:raise ValueError('Required source file missing: '+str(path))
    return dict(path=str(path), size_bytes=path.stat().st_size)


def _pipeline(model):
    if model == 'vgg16':
        from . import heir_vgg_fused_pipeline as pipeline
    elif model == 'alexnet':
        from . import heir_alexnet_wide_pool as pipeline
    else:raise ValueError('Explicit original common model required')
    return pipeline


def clean(observation):
    if (type(observation.get('exit_code')) is not int or observation['exit_code'] != 0
            or observation.get('timed_out') is not False or observation.get('signal') is not None
            or observation.get('signals_sent') != [] or observation.get('remaining_pids') != []):
        raise ValueError('Every source/tool phase must have a clean saved process observation')


def model_source_gate(args):
    """Recheck saved original math/parameters; stat, do not read compiled MLIR."""
    from . import heir_vgg_fused_pipeline as common_contract
    case = args.source_case.resolve();record = read(case/'result.json');pipeline = _pipeline(args.model)
    zero_fields = getattr(pipeline, 'ZERO_FIELDS', common_contract.ZERO_FIELDS)
    formal_fields = getattr(pipeline, 'FORMAL', common_contract.FORMAL)
    if (record.get('status') != 'ok' or record.get('round') != pipeline.ROUND
            or record.get('mode') != pipeline.MODE or record.get('variant') != pipeline.VARIANT
            or record.get('model') != MODELS[args.model] or record.get('model_compiled') is not True
            or record.get('strict_math_passed') is not True or record.get('quality_selection') is not False
            or record.get('original_model_changed') is not False
            or record.get('guard_trigger') is not None or record.get('incoming_signal') is not None
            or record.get('timeout_signal') is not None or record.get('wall_timeout_observed') is not False
            or any(record.get(key) != 0 for key in zero_fields)
            or any(record.get(key) is not None for key in formal_fields)):
        raise ValueError('Successful same-model strict-math/full-lowering record required')
    completion = read(case/'completion.json')
    if (completion.get('status') != 'ok' or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True):
        raise ValueError('Original successful durable completion required')
    phases = record.get('phases') or {}
    for phase in ('export_math', 'native_compile'):
        for field in ('attempted', 'completed'):
            value = (record.get(field) or {}).get(phase)
            if type(value) is not int or value != 1:raise ValueError('Exactly one successful original phase required')
        if phases.get(phase, {}).get('status') != 'ok':raise ValueError('Original phase is incomplete')
        clean(phases[phase].get('observation') or {})
    saved = read(case/'source.json')
    if record.get('source') != saved:raise ValueError('Outer and saved original source differ')
    original_paths = saved.get('paths') or {}
    if set(original_paths) != set(pipeline.SOURCE_FIELDS if hasattr(pipeline, 'SOURCE_FIELDS') else
            ('intake_result', 'graph_dir', 'program', 'frontend_result', 'probes', 'probe_source', 'pipeline', 'native_parameters')):
        raise ValueError('Complete original model/program/probe/parameter sources required')
    checked = pipeline.source_gate({key: Path(value) for key, value in original_paths.items()})
    if checked != saved:raise ValueError('Current original source metadata differs from the successful source')
    worker = read(case/'worker/result.json')
    if worker != record.get('worker'):raise ValueError('Outer and original strict worker differ')
    audit = pipeline.check_worker(worker, checked)
    if phases['export_math'].get('audit') != audit:raise ValueError('Original saved strict-math audit differs')
    artifact = read(case/'native/model-native-artifact.json')
    if artifact != phases['native_compile'].get('artifact'):
        raise ValueError('Original native artifact metadata differs')
    actual = artifact.get('parameters') or [];wanted = checked['native_parameters']
    if (len(actual) != 1 or any(actual[0].get(key) != wanted[field] for key, field in
            (('logN', 'logN'), ('Q', 'Q'), ('P', 'P'), ('log_default_scale', 'logDefaultScale')))
            or artifact.get('backend_executed') is not False or artifact.get('security_validated') is not False
            or type(artifact.get('static_bootstrap_operations')) is not int
            or artifact['static_bootstrap_operations'] < 0
            or record.get('static_native_bootstraps') != artifact['static_bootstrap_operations']):
        raise ValueError('Complete original public native parameters and static site inventory required')
    source_mlir = case/'worker/model.mlir';native = case/'native/model-lattigo.mlir'
    request = record.get('native_request') or {}
    if (request.get('parameters') != wanted or not _same(request.get('source_mlir'), source_mlir)
            or not _same(request.get('pipeline'), case/'native/explicit.pipeline')
            or request.get('original_model_contract') != checked['contract']
            or request.get('lowered_adapter_contract') != worker.get('lowered_adapter_contract')):
        raise ValueError('Original lowering request lineage differs')
    command = phases['native_compile'].get('argv') or []
    if not command or Path(command[0]).name != 'heir-opt' or Path(command[0]).resolve().parent != args.translate.resolve().parent:
        raise ValueError('Same installed original compiler/translator and native output path required')
    from .heir_cnn_preflight import compiler_command
    expected_command = compiler_command(command[0], source_mlir, native, case/'native/resources')
    expected_command = [value for value in expected_command if not value.startswith(
        ('--annotate-module=', '--torch-linalg-to-ckks=', '--scheme-to-lattigo'))]
    expected_command.insert(2, '--pass-pipeline='+(case/'native/explicit.pipeline').read_text().strip())
    if command != expected_command:
        raise ValueError('Original compiler source, full explicit pipeline and native output argv differ')
    return dict(source_result=str(case/'result.json'), source_case=str(case), model=MODELS[args.model],
        variant=pipeline.VARIANT, input_mlir=_file(native), native_artifact=artifact,
        original_source=saved, strict_math=audit, original_model_changed=False,
        previous_lowering_clock_included=False, compiled_mlir_or_external_payload_read_by_gate=False)


def backend_gate(args):
    """Admit the already prepared Go1.26/Lattigo6.1 cache, without tools/network."""
    backend = read(args.backend_case/'result.json');cache = args.cache_source.resolve()
    if (backend.get('round') != 'R20' or backend.get('mode') != 'heir-go-backend-build-preflight'
            or backend.get('status') != 'ok' or backend.get('backend_library_build_passed') is not True
            or backend.get('approved_go_version') != '1.26.0' or backend.get('lattigo_version') != '6.1.0'
            or backend.get('go_sdk_version_observed') != 'go version go1.26.0 linux/arm64'
            or not _same(backend.get('work_directory'), cache)):
        raise ValueError('Completed original R20 Go1.26/Lattigo6.1 preparation and its cache required')
    go = cache/'toolchain/go/bin/go';version = cache/'toolchain/go/VERSION'
    if not _same(args.go, go) or not os.access(args.go, os.X_OK) or version.read_text().splitlines()[0] != 'go1.26.0':
        raise ValueError('Original installed Go SDK differs')
    if not args.translate.is_file() or not os.access(args.translate, os.X_OK):
        raise ValueError('Existing original heir-translate executable required')
    mod = (cache/'module/go.mod').read_text()
    if (not mod.startswith('module ') or not re.search(r'^go 1\.26\.0\s*$', mod, re.M)
            or not re.search(r'^require github\.com/tuneinsight/lattigo/v6 v6\.1\.0\s*$', mod, re.M)
            or re.search(r'^\s*(?:replace|toolchain)\b', mod, re.M)):
        raise ValueError('Existing unmodified pinned offline module contract required')
    for name in ('module-cache', 'build-cache', 'module-cache/github.com/tuneinsight/lattigo/v6@v6.1.0'):
        if not (cache/name).is_dir():raise ValueError('Original offline module/build cache missing')
    return dict(backend_result=str(args.backend_case/'result.json'), cache_source=str(cache),
        go=_file(go), translate=_file(args.translate), go_version='1.26.0', lattigo_version='6.1.0',
        source_go_mod=_file(cache/'module/go.mod'), source_go_sum=_file(cache/'module/go.sum'),
        additional_integrity_commands=0, intrinsic_offline_go_build_behavior_retained=True)


def source_gate(args):
    return dict(model=model_source_gate(args), backend=backend_gate(args))


def commands(args, out, checked):
    source = checked['model']['input_mlir']['path'];module = out/'module'
    return dict(utils_emission=[str(args.translate), source, '--mlir-disable-threading',
            '--emit-lattigo-preprocessing', '--package-name=heircnn_utils', '-o', str(module/'heircnn_utils/heircnn_utils.go')],
        main_emission=[str(args.translate), source, '--mlir-disable-threading',
            '--emit-lattigo-preprocessed', '--extra-imports=heircnn/heircnn_utils', '--package-name=heircnn',
            '-o', str(module/'heircnn.go')],
        library_build=[str(args.go), 'build', '-p=1', '-mod=readonly', '-buildvcs=false', './...'])


def environment(out):
    env = {key: value for key, value in os.environ.items()
           if key not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV') and not key.startswith(('LD_', 'NIX_', '__NIX_', 'GO'))}
    env.update(GOTOOLCHAIN='local', GOPROXY='off', GOSUMDB='off', GOMAXPROCS='1', GOMEMLIMIT='3GiB',
        CGO_ENABLED='0', GOPATH=str(out/'gopath'), GOMODCACHE=str(out/'module-cache'),
        GOCACHE=str(out/'build-cache'), GOTMPDIR=str(out/'tmp'), TMPDIR=str(out/'tmp'),
        TEST_TELEMETRY_DIR=str(out/'telemetry'), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):env[name] = '1'
    return env


def prepare(out, checked):
    module = out/'module';(module/'heircnn_utils').mkdir(parents=True)
    (out/'telemetry').mkdir();(out/'telemetry/mode').write_text('off\n')
    cache = Path(checked['backend']['cache_source'])
    original = (cache/'module/go.mod').read_text()
    (module/'go.mod').write_text('module heircnn\n'+original[original.index('\n')+1:])
    shutil.copyfile(cache/'module/go.sum', module/'go.sum')
    source = out/'source';source.mkdir()
    from . import heir_cnn_codegen
    snapshots = []
    for path in (Path(__file__), Path(heir_cnn_codegen.__file__), Path(_pipeline(checked['model']['model'].split('_')[0]).__file__)):
        target = source/path.name;shutil.copyfile(path, target)
        snapshots.append(dict(source=str(path.resolve()), snapshot=str(target), size_bytes=target.stat().st_size))
    durable_json(source/'snapshots.json', dict(files=snapshots, hashes_computed=0))


def check_package(path, expected):
    """Only a bounded generated header is inspected; numeric tables are untouched."""
    _file(path)
    match = None;remaining = 8192
    with path.open('rb') as stream:
        while remaining:
            line = stream.readline(remaining)
            if not line:break
            remaining -= len(line)
            match = re.fullmatch(rb'package\s+(\w+)\s*', line)
            if match:break
    if not match or match[1].decode('ascii') != expected:
        raise ValueError('Expected generated library package '+expected)
    return dict(path=str(path), size_bytes=path.stat().st_size, package=expected, header_only=True)


def append_registry(out, result, registry):
    entry = dict(run_id=out.name, round=ROUND, model=result['model'], mode=MODE, git_rev=result['git_rev'],
        metrics={key: result[key] for key in ('status', 'attempted', 'completed', 'go_sources_generated',
            'library_build_passed', 'static_native_bootstraps', *FORMAL_FIELDS)},
        artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('A case may have one eight-field registry row')
        stream.seek(0, os.SEEK_END);stream.write(json.dumps(entry, allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def terminal(out, result):
    if artifact_bytes(out) > CASE_BYTES-RESERVE_BYTES:
        result.update(status='resource_stopped', error='Case allowance including completion reserve exhausted')
    # The successful build observation stays visible if a later resource check fails.
    result['go_library_compile_accepted'] = result['status'] == 'ok' and result['library_build_passed'] is True
    for _ in range(20):
        durable_json(out/'result.json', result);size = artifact_bytes(out)
        if result.get('artifact_bytes') == size:return
        result['artifact_bytes'] = size
    raise RuntimeError('Terminal metadata size did not stabilize')


def execute(args):
    started = time.perf_counter_ns();root = paths.RESULTS_DIR.resolve();out = args.out.resolve()
    if out.exists() or out == root or not out.is_relative_to(root):raise ValueError('A fresh result case is required')
    with serial_lock(root):
        out.mkdir(parents=True);(out/'tmp').mkdir();recorder = StageRecorder(out/'events.jsonl')
        deadline = time.monotonic()+TOTAL_SECONDS
        result = dict(round=ROUND, mode=MODE, model=MODELS[args.model], status='running', error=None,
            git_rev=os.environ.get('PT_AGENT_GIT_REV'), attempted={key: 0 for key in PHASE_SECONDS},
            completed={key: 0 for key in PHASE_SECONDS}, phases={}, go_sources_generated=False,
            library_build_passed=False, go_library_compile_accepted=False, static_native_bootstraps=None,
            original_libraries_modified=False, original_model_modified=False, parameters_modified=False,
            model_backend_executed=False, full_ckks_validated=False, generated_initializer_executed=False,
            incoming_signal=None, timeout_signal=None, wall_timeout_observed=False, guard_trigger=None,
            limits=dict(total_seconds=TOTAL_SECONDS, phase_seconds=PHASE_SECONDS, AS_gib=4,
                subtree_RSS_gib=14, minimum_available_gib=1.25, case_gib=8, file_gib=8,
                minimum_free_disk_gib=10, concurrency=1),
            scope='Continuation from a saved successful lowering, through split Go source emission and offline library compilation; generated code is never run',
            **{key: 0 for key in ZERO_FIELDS}, **{key: None for key in FORMAL_FIELDS})
        stop = threading.Event();failures = [];active = [None]
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        def check():
            if failures:raise RuntimeError(failures[0])
            if time.monotonic() >= deadline:
                result['wall_timeout_observed'] = True;raise TimeoutError('Finite Go compile case expired')
            observed = physical_observation(_latest_rss_sample(active[0]) if active[0] else None,
                _available_ram_bytes(), shutil.disk_usage(out).free)
            if observed['stop']:
                result['guard_trigger'] = observed;durable_json(out/'resource-guard.json', observed)
                raise RuntimeError('; '.join(observed['reasons']))
            if artifact_bytes(out) > CASE_BYTES-RESERVE_BYTES:raise RuntimeError('8GiB case allowance exhausted')
        def monitor():
            while not stop.wait(.25):
                try:check()
                except BaseException as exc:
                    failures.append(type(exc).__name__+': '+str(exc));result['monitor_stop_reason'] = failures[-1]
                    try:durable_json(out/'monitor-stop.json', dict(reason=failures[-1], guard=result.get('guard_trigger')))
                    finally:
                        try:os.kill(os.getpid(), signal.SIGTERM)
                        except OSError as error:result['monitor_signal_error'] = str(error)
                    return
        watcher = threading.Thread(target=monitor, daemon=True)
        for sig in old:signal.signal(sig, lambda signum, frame: receive_signal(result, signum))
        signal.setitimer(signal.ITIMER_REAL, TOTAL_SECONDS)
        try:
            with recorder.span('complete-common-go-library-continuation'):
                check();watcher.start()
                with recorder.span('source-admission-and-private-module'):
                    checked = source_gate(args);result['source'] = checked
                    result['static_native_bootstraps'] = checked['model']['native_artifact']['static_bootstrap_operations']
                    prepare(out, checked);planned = commands(args, out, checked);result['commands'] = planned
                    durable_json(out/'source.json', checked)
                result['status'] = 'prepared';durable_json(out/'result.json', result)
                if args.run:
                    with recorder.span('private-offline-cache-copy'):
                        cache = Path(checked['backend']['cache_source'])
                        for name in ('module-cache', 'build-cache'):
                            check();shutil.copytree(cache/name, out/name);check()
                    for phase, command in planned.items():
                        check();result['attempted'][phase] = 1
                        row = dict(argv=command, status='running', observation=None);result['phases'][phase] = row
                        durable_json(out/'result.json', result)
                        with recorder.span(phase, argv=command):
                            active[0] = out/(phase+'-resources.jsonl')
                            try:
                                observed = run_process(command, cwd=out/'module', log=out/(phase+'.log'), resources=active[0],
                                    timeout=min(PHASE_SECONDS[phase], max(.01, deadline-time.monotonic())),
                                    mem_gib=4, fsize_gib=8, env=environment(out))
                                row['observation'] = observed;observe_termination(result, observed)
                            finally:active[0] = None
                            durable_json(out/'result.json', result);clean(observed)
                            # Completion counts reflect actual clean child exits, even if a following guard stops the case.
                            result['completed'][phase] = 1
                            if phase == 'utils_emission':row['artifact'] = check_package(out/'module/heircnn_utils/heircnn_utils.go', 'heircnn_utils')
                            elif phase == 'main_emission':
                                row['artifact'] = check_package(out/'module/heircnn.go', 'heircnn');result['go_sources_generated'] = True
                            else:result['library_build_passed'] = True
                            check()
                        row['status'] = 'ok';durable_json(out/'result.json', result)
                    result['status'] = 'ok'
        except BaseException as exc:
            result.update(status=interruption_status(result), error=type(exc).__name__+': '+str(exc)[:4000])
            for row in result['phases'].values():
                if row['status'] == 'running':row.update(status=result['status'], error=result['error'])
        finally:
            stop.set()
            if watcher.is_alive():watcher.join(timeout=3)
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old.items():signal.signal(sig, handler)
            if failures:result.update(status=interruption_status(result), error=failures[0])
            result['timing'] = summarize_events(read_events(out/'events.jsonl'), time.perf_counter_ns())
            terminal(out, result);append_registry(out, result, args.registry or root/'registry.jsonl')
            marker = persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=result['timing'], required_artifact_bytes=artifact_bytes(out))
            if marker['case_artifact_bytes'] > CASE_BYTES or (out/'completion.json').stat().st_size > RESERVE_BYTES:
                raise RuntimeError('Completion exceeded reserved metadata allowance')
        return result


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    command.add_argument('--model', choices=tuple(MODELS), required=True)
    for name in ('source-case', 'out', 'backend-case', 'cache-source', 'translate', 'go'):
        command.add_argument('--'+name, type=Path, required=True)
    command.add_argument('--registry', type=Path);command.add_argument('--run', action='store_true')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):setattr(args, name, value.expanduser().resolve())
    result = execute(args)
    print(json.dumps({key: result[key] for key in ('model', 'status', 'error', 'go_sources_generated', 'library_build_passed')}))
    return int(result['status'] not in ('prepared', 'ok'))


if __name__ == '__main__':raise SystemExit(main())
