"""Approved R22: one original ResNet20 Orbit frontend and TDAG preflight."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import re
import shutil
import signal
import sys
import threading
import time
import urllib.parse
import urllib.request

from .. import paths
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r13_common_models import FACTORIES

MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-resnet20-trace-tdag-preflight'
MODULE = 'pt_agent.experiments.r22_orbit_model_worker'
PROFILE = Path('cost_models/profiled_LATTIGONEW_CPU64k_3_16.json')
METADATA_URL = 'https://pypi.org/pypi/einops/0.8.1/json'
WHEEL_NAME = 'einops-0.8.1-py3-none-any.whl'
MEMORY_GIB, FILE_GIB, OVERALL_SECONDS = 4, 8, 3600
CASE_BYTES, FREE_BYTES = 4 * 1024**3, 10 * 1024**3
POLY_FILES = ('MPCB.py', 'Poly.py', 'Func.py')
DATA_FILES = ('treeStr.txt', 'coeffStr.txt', 'sgn151527.txt', 'tree15.txt', 'tree27.txt')
CATEGORIES = ('metadata_download', 'wheel_download', 'dependency_install', 'tests', 'trace', 'parse')
FORMAL = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')


def _locations():
    root = paths.ORBIT_ROOT.resolve()
    return dict(results=paths.RESULTS_DIR.resolve(), repo=paths.REPO,
                scripts=paths.REPO / 'DSL/DSL_agent/scripts', orbit=root,
                frontend=root / 'frontend/dacapo', intake_python=paths.INTAKE_PYTHON.absolute(),
                orbit_python=root.parent / 'orbit-venv/bin/python',
                profile=root / PROFILE,
                benchmark=root / 'frontend/dacapo_patch/benchmarks/ResNetSiLU64k.py',
                model_source=root / 'frontend/dacapo/python/poly/poly/models/ResNetSiLU.py',
                checkpoint=Path(FACTORIES[MODEL]['checkpoint']).resolve(),
                poly_data=paths.DACAPO_UPSTREAM / 'python/poly/poly/data',
                r13_result=paths.RESULTS_DIR / 'r13-common-resnet20-silu-20261115/result.json',
                r16_result=paths.RESULTS_DIR / 'r16-baseline-compatibility-20261115/orbit-resnet20_silu/result.json')


def _source_preflight(locations):
    required = [locations[k] for k in ('benchmark', 'model_source', 'checkpoint', 'profile', 'r13_result', 'r16_result')]
    required += [locations['frontend'] / 'python/hecate/hecate/expr.py',
                 locations['frontend'] / 'build/lib/libHecateFrontend.so']
    required += [locations['frontend'] / 'python/poly/poly' / n for n in POLY_FILES]
    required += [locations['poly_data'] / n for n in DATA_FILES]
    missing = [str(p) for p in required if not p.is_file() or not p.stat().st_size]
    missing += [str(locations[k]) for k in ('intake_python', 'orbit_python')
                if not (locations[k].is_file() or locations[k].is_symlink())]
    if missing:
        raise RuntimeError('Missing approved R22 inputs: ' + ', '.join(missing))
    r13 = json.loads(locations['r13_result'].read_text())
    r16 = json.loads(locations['r16_result'].read_text())
    if any(d.get('status') != 'ok' or d.get('model') != MODEL for d in (r13, r16)):
        raise ValueError('R22 requires saved original ResNet success records')
    if r13.get('round') != 'R13' or r16.get('round') != 'R16' or r16.get('baseline') != 'orbit':
        raise ValueError('Saved source record identities differ')
    w13, w16 = r13.get('worker', {}), r16.get('worker', {})
    if (w13.get('round') != 'R13' or w16.get('round') != 'R16' or w16.get('baseline') != 'orbit'
            or any(w.get('status') != 'ok' or w.get('model') != MODEL for w in (w13, w16))):
        raise ValueError('Saved source worker identities differ')
    if not all(w13.get(k) is True for k in ('import_completed', 'real_probes_passed', 'graph_math_passed', 'roundtrip_passed', 'static_coverage')):
        raise ValueError('R13 original intake evidence incomplete')
    if not all(w16.get(k) is True for k in ('source_checked', 'checkpoint_checked', 'model_built', 'strict_load_passed', 'structure_passed', 'probes_checked', 'math_probes_passed', 'compatibility_passed', 'all_finite')):
        raise ValueError('R16 original Orbit compatibility evidence incomplete')
    summary = w16.get('source_summary', {})
    expected = dict(r13_result=locations['r13_result'], model_source=locations['model_source'], checkpoint=locations['checkpoint'])
    if any(Path(summary.get(k, '')).resolve() != Path(v).resolve() for k, v in expected.items()):
        raise ValueError('Saved Orbit source locations differ')
    if Path(r16.get('source_r13', {}).get('result', '')).resolve() != locations['r13_result'].resolve():
        raise ValueError('Saved R16 original R13 reference differs')
    if w16.get('forward_attempts') != 3 or w16.get('forward_completed') != 3:
        raise ValueError('R16 original three references incomplete')
    probes = w16.get('math_probes', [])
    if (len(probes) != 3 or [p.get('probe_index') for p in probes] != [0, 1, 2]
            or any(not all(p.get(k) is True for k in ('passed', 'finite', 'shape_ok', 'dtype_ok'))
                   or p.get('criterion') != '1e-12+1e-12*abs(reference)' for p in probes)):
        raise ValueError('R16 strict reference policy differs')
    state = w16.get('state_report', {})
    if (not state.get('passed') or not state.get('strict_load') or not state.get('structure', {}).get('passed')
            or not state.get('original_state')
            or any(item.get('finite') is not True or item.get('numerically_equal') is not True
                   for item in state['original_state'])):
        raise ValueError('Original state and structure evidence incomplete')
    return dict(r13_result=str(locations['r13_result']), r16_result=str(locations['r16_result']),
                model=MODEL, original_reference_probes=3, references_repeated=0,
                checkpoint=str(locations['checkpoint']), model_source=str(locations['model_source']),
                training_preprocessing_verified=False, r13_recorded_mux_slots=16384,
                proposed_native_template_slots=65536, coefficient_source_equivalence_verified=False,
                inputs=[dict(path=str(p), bytes=p.stat().st_size) for p in required])


def select_wheel(metadata):
    if metadata.get('info', {}).get('name') != 'einops' or metadata.get('info', {}).get('version') != '0.8.1':
        raise ValueError('R22 requires the fixed einops 0.8.1 release')
    selected = [x for x in metadata.get('urls', []) if x.get('filename') == WHEEL_NAME and x.get('packagetype') == 'bdist_wheel']
    if len(selected) != 1:
        raise ValueError('Expected one fixed platform-independent wheel')
    item = selected[0]
    url = urllib.parse.urlsplit(item['url'])
    if (url.scheme != 'https' or url.hostname != 'files.pythonhosted.org' or url.port not in (None, 443)
            or url.username or url.password or url.fragment or url.query
            or not url.path.endswith('/' + WHEEL_NAME)
            or type(item.get('size')) is not int or not 0 < item['size'] <= 1024**2):
        raise ValueError('Unexpected wheel location or download size')
    return dict(filename=WHEEL_NAME, url=item['url'], bytes=item['size'])


def dependency_install_command(locations, wheel, target):
    return [str(locations['orbit_python']), '-B', '-m', 'pip', '--isolated',
            '--disable-pip-version-check', '--no-cache-dir', 'install', '--no-index',
            '--no-deps', '--no-compile', '--target', str(target), str(wheel)]


class _RestrictedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, host):
        self.host = host
        self.redirects = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.redirects.append(dict(source=req.full_url, target=newurl, status=code))
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != 'https' or parsed.hostname != self.host or parsed.username or parsed.password:
            raise ValueError('R22 redirect outside approved host')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url, destination, seconds, record_path, checkpoint):
    end = time.monotonic() + seconds
    record = dict(url=url, final_url=None, bytes=0, complete=False, error=None, redirects=[])
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    timer_started = time.monotonic()
    def individual_expired(signum, frame):
        raise TimeoutError('R22 individual download absolute allowance exhausted')
    signal.signal(signal.SIGALRM, individual_expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, previous_timer[0]) if previous_timer[0] else seconds)
    try:
        host = urllib.parse.urlsplit(url).hostname
        redirect = _RestrictedRedirect(host)
        record['redirects'] = redirect.redirects
        opener = urllib.request.build_opener(redirect)
        with opener.open(urllib.request.Request(url, headers={'User-Agent': 'Poseidon-R22'}), timeout=seconds) as response:
            record['final_url'] = response.geturl()
            if urllib.parse.urlsplit(record['final_url']).hostname != host:
                raise ValueError('R22 unexpected final download host')
            with destination.open('xb') as stream:
                while True:
                    checkpoint()
                    if time.monotonic() >= end:
                        raise TimeoutError('R22 individual download allowance exhausted')
                    data = response.read(65536)
                    if not data:
                        break
                    record['bytes'] += len(data)
                    if record['bytes'] > 1024**2:
                        raise RuntimeError('R22 download exceeds 1 MiB')
                    stream.write(data)
        if not record['bytes']:
            raise RuntimeError('R22 empty download')
        record['complete'] = True
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            write_json(record_path, record)
        finally:
            signal.signal(signal.SIGALRM, previous_handler)
            remaining = max(.001, previous_timer[0] - (time.monotonic() - timer_started)) if previous_timer[0] else 0
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])
    return record


def prepare_dependencies(locations, work, result, recorder, launch, checkpoint):
    wheels = work / 'wheels'
    wheels.mkdir()
    metadata_path = wheels / 'release.json'
    for name, url, dest, seconds in [('metadata_download', METADATA_URL, metadata_path, 60)]:
        result['failed_stage'] = name
        result['attempted'][name] += 1
        with recorder.span(name):
            _download(url, dest, seconds, work / (name + '.json'), checkpoint)
            selected = select_wheel(json.loads(dest.read_text()))
        result['completed'][name] += 1
    wheel = wheels / selected['filename']
    result['selected_dependency'] = selected
    name = 'wheel_download'
    result['failed_stage'] = name
    result['attempted'][name] += 1
    with recorder.span(name):
        observation = _download(selected['url'], wheel, 120, work / (name + '.json'), checkpoint)
        if observation['bytes'] != selected['bytes']:
            raise ValueError('R22 wheel size differs from release metadata')
    result['completed'][name] += 1
    target = work / 'python-deps'
    launch('dependency-install', dependency_install_command(locations, wheel, target), 180, 'dependency_install')
    if not (target / 'einops/__init__.py').is_file():
        raise RuntimeError('R22 isolated dependency package missing')
    return target


def prepare_local_poly(locations, work, result):
    target = work / 'local-poly/poly'
    target.mkdir(parents=True)
    (target / 'data').mkdir()
    result['local_poly_files'] = []
    for names, source, destination in [(POLY_FILES, locations['frontend'] / 'python/poly/poly', target),
                                       (DATA_FILES, locations['poly_data'], target / 'data')]:
        for name in names:
            src, dst = source / name, destination / name
            shutil.copyfile(src, dst)
            result['local_poly_files'].append(dict(source=str(src), destination=str(dst), bytes=dst.stat().st_size))
    return target


def phase_commands(locations, work, phase, mlir=None):
    phase_work = work / (phase + '-worker')
    if phase == 'trace':
        return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '1860',
                '--python', str(locations['intake_python']), '--', '-m', MODULE,
                '--phase', 'trace', '--work-dir', str(phase_work),
                '--frontend', str(locations['frontend']), '--model-source', str(locations['model_source']),
                '--benchmark', str(locations['benchmark']), '--checkpoint', str(locations['checkpoint']),
                '--poly-dir', str(work / 'local-poly/poly'), '--deps-dir', str(work / 'python-deps'),
                '--r13-result', str(locations['r13_result']), '--r16-result', str(locations['r16_result'])]
    if phase != 'parse' or mlir is None:
        raise ValueError('R22 requires trace or parse with a saved MLIR path')
    return [str(locations['orbit_python']), '-B', '-m', MODULE, '--phase', 'parse',
            '--work-dir', str(phase_work), '--orbit-root', str(locations['orbit']),
            '--profile', str(locations['profile']), '--mlir', str(mlir)]


def case_size(out):
    total = 0
    for parent, _, names in os.walk(out):
        for name in names:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def check_budget(out):
    amount = case_size(out)
    if amount > CASE_BYTES or shutil.disk_usage(out).free < FREE_BYTES:
        raise RuntimeError('R22 artifact or disk reserve allowance exceeded')
    return amount


def write_terminal_result(out, result):
    path = out / 'result.json'
    for _ in range(20):
        result['metrics'].update(status=result['status'], case_artifact_bytes=result.get('case_artifact_bytes', 0))
        write_json(path, result)
        amount = case_size(out)
        if amount == result.get('case_artifact_bytes'):
            return
        result['case_artifact_bytes'] = result['artifact_bytes'] = amount
    raise RuntimeError('R22 terminal artifact-size fixed point did not converge')


def append_registry(out, result, registry):
    row = dict(run_id=out.name, round='R22', model=MODEL, mode=MODE, git_rev=result['git_rev'],
               metrics=result['metrics'], artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def _child_environment(work):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('LD_', 'NIX_', '__NIX_', 'HECATE_'))
           and k not in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TMPDIR=str(work / 'tmp'),
               R22_DEPS_DIR=str(work / 'python-deps'), R22_TEST_WORK=str(work / 'test-worker'),
               PIP_CACHE_DIR=str(work / 'pip-cache'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def _base_environment_snapshot():
    observed = []
    for relative in ('deps/nix-portable-v012', 'deps/nix-portable-v012/launcher.lock',
                     'deps/nix-portable-v012/xdg-config', 'deps/nix-portable-v012/xdg-state',
                     'cache/nix-portable-v012', 'cache/nix-portable-tmp'):
        path = paths.PLATFORM_DIR / relative
        record = dict(path=str(path), exists=path.exists())
        if path.exists():
            stat = path.stat()
            record.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, directory=path.is_dir())
        observed.append(record)
    return observed


def execute(args):
    started = time.perf_counter_ns()
    deadline = time.monotonic() + OVERALL_SECONDS
    locations = _locations()
    out = Path(args.out).absolute()
    if out.exists() or out.is_symlink() or out.resolve().parent != locations['results'].resolve():
        raise ValueError('R22 requires a new direct child of RESULTS_DIR')
    out.mkdir()
    work = out / 'work'
    work.mkdir()
    for name in ('tmp', 'pip-cache', 'test-worker'):
        (work / name).mkdir()
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))
    saved_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    saved_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame):
        raise RuntimeError('R22 outer signal ' + str(signum))
    for sig in saved_handlers:
        signal.signal(sig, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.monotonic()))
    recorder = StageRecorder(out / 'events.jsonl')
    result = dict(run_id=out.name, round='R22', model=MODEL, mode=MODE, status='running', error=None,
                  failed_stage=None, git_rev=None, git_branch=None, attempted=dict.fromkeys(CATEGORIES, 0),
                  completed=dict.fromkeys(CATEGORIES, 0), processes=[], phase_results={}, retry_count=0,
                  source_records=None, local_poly_files=[], selected_dependency=None, worker_record_errors=[],
                  git_metadata_attempted=0, git_metadata_completed=0,
                  limited_original_launcher_check_approved=True, native_go_verification_exception_used=False,
                  extra_integrity_checks=0, resource_monitor_error=None, artifact_bytes=0, case_artifact_bytes=0,
                  orbit_model_trace_passed=False, tdag_parse_passed=False, optimizer_ready=False,
                  backend_ready=False, formal_matrix_complete=False, new_llm_calls=0,
                  model_forward_runs=0, sample_reads=0, solver_runs=0, backend_compile_runs=0,
                  keygens=0, ckks_runs=0, confirm_samples=0, report_samples=0, candidates_frozen=0)
    env = _child_environment(work)
    stop = threading.Event()
    child_events = []
    def checkpoint():
        if result['resource_monitor_error']:
            raise RuntimeError(result['resource_monitor_error'])
        check_budget(out)
        if time.monotonic() >= deadline:
            raise TimeoutError('R22 entire-batch allowance exhausted')
    def monitor():
        while not stop.wait(1):
            try:
                amount = case_size(out)
                free = shutil.disk_usage(out).free
                with (out / 'artifact-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(), artifact_bytes=amount, disk_free_bytes=free)) + '\n')
                if amount > CASE_BYTES or free < FREE_BYTES:
                    raise RuntimeError('R22 periodic artifact/disk allowance exceeded')
            except BaseException as exc:
                result['resource_monitor_error'] = f'{type(exc).__name__}: {exc}'
                os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    def launch(name, argv, seconds, category=None):
        checkpoint()
        result['failed_stage'] = name
        if category:
            result['attempted'][category] += 1
        else:
            result['git_metadata_attempted'] += 1
        command = dict(stage=name, argv=[str(x) for x in argv], cwd=str(locations['scripts']), observation=None, native_termination_signal=None)
        result['processes'].append(command)
        print(json.dumps(dict(stage=name, state='started')), flush=True)
        with recorder.span(name, command=command['argv']) as parent:
            try:
                obs = run_process(command['argv'], cwd=locations['scripts'], log=out / (name + '.log'),
                                  resources=out / (name + '-resources.jsonl'),
                                  timeout=min(seconds, max(.01, deadline - time.monotonic())),
                                  mem_gib=4, fsize_gib=8, env=env)
                command['observation'] = obs
                code = obs['exit_code']
                command['native_termination_signal'] = -code if isinstance(code, int) and code < 0 else None
            finally:
                if category in ('trace', 'parse'):
                    worker_dir = work / (category + '-worker')
                    try:
                        for event in read_events(worker_dir / 'events.jsonl'):
                            if event.get('type') in ('begin', 'end') and event.get('parent_id') is None:
                                event['parent_id'] = parent['event_id']
                            child_events.append(event)
                        worker_result = worker_dir / 'phase_result.json'
                        if worker_result.is_file():
                            result['phase_results'][category] = json.loads(worker_result.read_text())
                    except BaseException as record_error:
                        result['worker_record_errors'].append(dict(phase=category, error=f'{type(record_error).__name__}: {record_error}'))
            if code != 0 or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']:
                raise RuntimeError('R22 supervised phase failed: ' + name)
            if category in ('trace', 'parse'):
                if any(e['phase'] == category for e in result['worker_record_errors']):
                    raise RuntimeError('R22 worker record invalid: ' + name)
                worker = result['phase_results'].get(category)
                if not worker or worker.get('status') != 'ok':
                    raise RuntimeError('R22 inner phase failed: ' + name)
            if category:
                result['completed'][category] += 1
            else:
                result['git_metadata_completed'] += 1
        checkpoint()
        print(json.dumps(dict(stage=name, state='completed')), flush=True)
        return obs
    try:
        watcher.start()
        with recorder.span('r22-preflight'):
            result['failed_stage'] = 'source-preflight'
            with recorder.span('source-preflight'):
                checkpoint()
                result['source_records'] = _source_preflight(locations)
                write_json(out / 'source-metadata.json', result['source_records'])
                prepare_local_poly(locations, work, result)
                write_json(out / 'local-poly-metadata.json', result['local_poly_files'])
                result['base_environment_work_paths'] = [str(paths.PLATFORM_DIR / p) for p in
                    ('deps/nix-portable-v012', 'cache/nix-portable-v012', 'cache/nix-portable-tmp')]
                result['base_environment_work_scope'] = 'Existing launcher locks/config/state/temp/cache paths; directory-entry metadata only, not claimed confined to out, no full-filesystem audit'
                result['base_environment_before'] = _base_environment_snapshot()
            launch('git-revision', ['git', 'rev-parse', '--short', 'HEAD'], 30)
            result['git_rev'] = (out / 'git-revision.log').read_text().strip()
            launch('git-branch', ['git', 'branch', '--show-current'], 30)
            result['git_branch'] = (out / 'git-branch.log').read_text().strip()
            prepare_dependencies(locations, work, result, recorder, launch, checkpoint)
            launch('r22-tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'pt_agent/tests', '-p', 'test_r22_*.py', '-v'], 600, 'tests')
            # A successful unittest invocation may still report skips; they are unexpected here.
            test_log = (out / 'r22-tests.log').read_text()
            tests_ran = re.findall(r'Ran (\d+) tests? in ', test_log)
            if not tests_ran or int(tests_ran[-1]) <= 0:
                raise RuntimeError('R22 test discovery completed without test cases')
            if 'skipped=' in test_log or '... skipped ' in test_log:
                raise RuntimeError('R22 tests unexpectedly skipped')
            inner_summary = json.loads((work / 'test-worker/inner-wrapper-result.json').read_text())
            if (inner_summary.get('status') != 'ok' or inner_summary.get('wrapper_runs') != 1
                    or not isinstance(inner_summary.get('inner_tests'), int) or inner_summary['inner_tests'] <= 0
                    or inner_summary.get('inner_skips') != 0):
                raise RuntimeError('R22 unique inner test run did not pass')
            result['test_summary'] = dict(system_tests=int(tests_ran[-1]), inner=inner_summary)
            for phase, seconds in [('trace', 2040), ('parse', 300)]:
                mlir = result['phase_results']['trace']['mlir']['path'] if phase == 'parse' else None
                launch('model-' + phase, phase_commands(locations, work, phase, mlir), seconds, phase)
                if phase == 'parse' and result['phase_results'][phase].get('graph_check', {}).get('passed') is not True:
                    raise RuntimeError('R22 TDAG graph acceptance missing')
                result['orbit_model_trace_passed' if phase == 'trace' else 'tdag_parse_passed'] = True
            result.update(status='ok', error=None, failed_stage=None)
    except BaseException as exc:
        result.update(status='stopped_on_error', error=f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join()
        ended = time.perf_counter_ns()
        events = read_events(out / 'events.jsonl') + child_events
        with (out / 'combined-events.jsonl').open('w') as stream:
            for event in events:
                stream.write(json.dumps(event) + '\n')
        timing = summarize_events(events, termination_observed_ns=ended)
        timing['outer_interval'] = dict(start_ns=started, end_ns=ended, elapsed_ns=ended-started,
                                       boundary='Before terminal timing/result fixed-point writes and registry append')
        write_json(out / 'timing.json', timing)
        result['timing'] = dict(attribution_ready=timing['attribution_ready'], diagnostics=timing['diagnostics'],
                                span_count=len(timing['spans']), recorded_outer_seconds=(ended-started)/1e9,
                                complete_batch_elapsed_seconds=None,
                                boundary=timing['outer_interval']['boundary'])
        if result['resource_monitor_error']:
            result.update(status='stopped_on_error', error=result['resource_monitor_error'])
        if not timing['attribution_ready'] or timing['diagnostics']:
            result.update(status='stopped_on_error', timing_record_error=True)
        result['base_environment_after'] = _base_environment_snapshot()
        result['root_raw_getrusage_hwm_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        trace, parse = (result['phase_results'].get(p, {}) for p in ('trace', 'parse'))
        result['metrics'] = dict(status=result['status'], preflight_only=True,
                                 attempted=dict(result['attempted']), completed=dict(result['completed']),
                                 orbit_model_trace_passed=result['orbit_model_trace_passed'], tdag_parse_passed=result['tdag_parse_passed'],
                                 model_forward_runs=0, sample_reads=0, new_llm_calls=0, optimizer_runs=0,
                                 keygens=0, ckks_runs=0, formal_matrix_complete=False,
                                 trace_runs=trace.get('trace_runs', None if result['attempted']['trace'] else 0),
                                 parse_runs=parse.get('parse_runs', None if result['attempted']['parse'] else 0),
                                 **dict.fromkeys(FORMAL, None))
        write_terminal_result(out, result)
        if result['case_artifact_bytes'] > CASE_BYTES or shutil.disk_usage(out).free < FREE_BYTES:
            result.update(status='stopped_on_error', terminal_budget_error=True)
            write_terminal_result(out, result)
        append_registry(out, result, locations['results'] / 'registry.jsonl')
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in saved_handlers.items():
            signal.signal(sig, handler)
        if saved_timer[0] or saved_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, *saved_timer)
        print(json.dumps(dict(status=result['status'], error=result['error'], failed_stage=result['failed_stage'], out=str(out), attempted=result['attempted'], completed=result['completed'], artifact_bytes=result['artifact_bytes'])), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if execute(args)['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
