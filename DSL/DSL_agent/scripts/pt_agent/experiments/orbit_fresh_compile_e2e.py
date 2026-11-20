"""One continuous original Orbit frontend/optimizer/parser compilation case.

The three existing stages run once, serially, with their original algorithms
and parameters. The master completion is the new continuous clock. Child clocks
are nested observations; historical runs are never added. No CKKS is executed.
"""
import argparse
from contextlib import contextmanager
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
from types import FunctionType

from .. import paths
from . import orbit_common_model_frontend as frontend
from . import orbit_common_model_optimizer as optimizer
from . import orbit_common_backend_parse as parser_entry
from . import orbit_common_resource_retry as retry
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_initializer_probe import durable_json
from .r10_paper_metrics import StageRecorder, _scan, read_events, run_process, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes
from .r31_lattigo_runtime_plan import physical_memory_observation

MODULE = 'pt_agent.experiments.orbit_fresh_compile_e2e'
ROUND = 'ORBIT-FRESH-COMPILE-E2E'
MODE = 'original-three-stage-continuous-fresh-compile'
MODELS = tuple(frontend.MODEL_RECORDS)
STAGES = ('frontend', 'optimizer', 'parser')
CHILD_SECONDS = dict(frontend=3060, optimizer=8460, parser=1560)
TOTAL_SECONDS = 13200  # Original outer budgets3000+8400+1500, supervision180, closure120.
GIB, CASE_BYTES, RESERVE = 1024**3, 48*1024**3, 1024**2
REGISTRY_FIELDS = {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'}
ZERO = ('api_calls', 'confirm_samples', 'report_samples', 'keygen_runs', 'CKKS_runs',
        'downloads', 'installations', 'additional_integrity_checks')


def read(path):
    return json.loads(Path(path).read_text())


def clean(observation):
    if (type(observation.get('exit_code')) is not int or observation['exit_code'] != 0
            or observation.get('timed_out') is not False or 'signal' not in observation
            or observation['signal'] is not None or observation.get('signals_sent') != []
            or observation.get('remaining_pids') != []):
        raise ValueError('Require a clean closed process with explicit cleanup observations')


def parser_namespace():
    """Extend only the common parser's model inventory to its existing Res record."""
    original = vars(parser_entry)
    namespace = dict(original, MODEL_RECORDS={key: dict(optimizer.MODEL_RECORDS[key]) for key in MODELS},
                     __file__=__file__)
    for name, function in original.items():
        if isinstance(function, FunctionType) and function.__globals__ is original:
            if hasattr(function, '__wrapped__'):
                raise ValueError('Decorated functions must keep their original namespace')
            cloned = FunctionType(function.__code__, namespace, function.__name__,
                                  function.__defaults__, function.__closure__)
            cloned.__kwdefaults__ = function.__kwdefaults__
            namespace[name] = cloned
    return namespace


def child_cases(args):
    # The unique basename is also the original child registry run_id.
    return {stage: args.out/'cases'/(args.out.name+'--'+stage) for stage in STAGES}


def commands(args, cases):
    common = ['--model', args.model, '--registry', str(args.registry), '--orbit-root', str(args.orbit_root)]
    first = [sys.executable, '-B', '-m', retry.MODULE, *common, '--out', str(cases['frontend']),
        '--memory-gib', '24', '--trace-seconds', '1800', '--parse-seconds', '300',
        '--overall-seconds', '3000', '--case-gib', '32']
    for name in ('checkpoint', 'r13_result', 'r16_result', 'poly_dir', 'deps_dir', 'intake_python', 'orbit_python'):
        value = getattr(args, name)
        if value is not None: first += ['--'+name.replace('_', '-'), str(value)]
    second = [sys.executable, '-B', '-m', optimizer.__spec__.name, *common,
        '--source-result', str(cases['frontend']/'result.json'), '--out', str(cases['optimizer']),
        '--optimize-seconds', '7200', '--validate-seconds', '300', '--overall-seconds', '8400', '--case-gib', '32']
    if args.orbit_python is not None: second += ['--orbit-python', str(args.orbit_python)]
    third = [sys.executable, '-B', '-m', parser_entry.__spec__.name]
    if args.model == 'resnet20': third = [sys.executable, '-B', '-m', MODULE, '--delegate-parser']
    third += [*common, '--out', str(cases['parser']), '--source-case', str(cases['optimizer']),
        '--shared-build-case', str(args.shared_build_case), '--go-sdk', str(args.go_sdk),
        '--initialization-audit', str(args.initialization_audit), '--run']
    return dict(zip(STAGES, (first, second, third)))


def registry_row(registry, case, result):
    matches = []
    with registry.open() as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == case.name: matches.append(row)
    if (len(matches) != 1 or set(matches[0]) != REGISTRY_FIELDS
            or any(matches[0].get(key) != result.get(key) for key in ('round', 'model', 'mode'))
            or Path(matches[0].get('artifacts_dir', '')).resolve() != case.resolve()):
        raise ValueError('Require one exact eight-field child registry identity')
    expected = result.get('metrics')
    if expected is None:
        expected = {key: result[key] for key in ('status', 'attempted', 'completed',
            'backend_parser_consumption_verified', 'compiler_graph_completed',
            'model_specific_native_codegen', 'formal_latency', 'accuracy_acceptance', 'security_acceptance')}
        expected.update(dict.fromkeys(parser_entry.ZERO_FIELDS, 0))
    if matches[0]['metrics'] != expected:
        raise ValueError('Child registry metrics differ from its terminal result')
    return matches[0]


def accept_child(args, stage, cases):
    case = cases[stage]; result = read(case/'result.json'); completion = read(case/'completion.json')
    factory = frontend.MODEL_RECORDS[args.model]['factory']
    if (result.get('status') != 'ok' or result.get('model') != factory
            or completion.get('status') != 'ok' or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True or completion.get('timing_status') != 'ok'
            or completion.get('exclusive_attribution_ready') is not True
            or completion.get('overlap_and_timing_diagnostics') != []):
        raise ValueError('Same-model child result and durable successful completion required')
    if stage in ('frontend', 'optimizer'):
        module = frontend if stage == 'frontend' else optimizer
        phases = ('trace', 'parse') if stage == 'frontend' else ('optimize', 'validate')
        if (result.get('round') != module.ROUND or result.get('mode') != module.MODE
                or result.get('model_key') != args.model
                or any(type(result.get(key, {}).get(p)) is not int or result[key][p] != 1
                       for key in ('attempted', 'completed') for p in phases)):
            raise ValueError('Original same-model stage counts or identity differ')
        process_rows = result.get('processes', [])
        if [row.get('phase') for row in process_rows] != list(phases):
            raise ValueError('Every original child phase must have its ordered observation')
        for row in process_rows: clean(row.get('observation') or {})
        if any(result.get(key) != 0 for key in module.worker.ZERO_FIELDS):
            raise ValueError('Original child invoked an excluded action')
        if stage == 'frontend':
            records = result.get('source_records') or {}
            if (result.get('orbit_model_trace_passed') is not True or result.get('tdag_parse_passed') is not True
                    or records.get('model') != factory
                    or any(records.get(key) is not False for key in
                           ('original_model_modified', 'weights_modified', 'polynomial_modified'))):
                raise ValueError('Original model trace/source must remain unchanged')
        elif (result.get('native_optimizer_completed') is not True
                or result.get('placement_artifact_checked') is not True
                or Path((result.get('source_records') or {}).get('source_record', '')).resolve()
                   != (cases['frontend']/'result.json').resolve()):
            raise ValueError('Original optimizer must consume this fresh frontend')
    else:
        if (result.get('round') != parser_entry.ROUND or result.get('mode') != parser_entry.MODE
                or result.get('compiler_graph_completed') is not True
                or result.get('backend_parser_consumption_verified') is not True
                or result.get('model_specific_native_codegen') is not None
                or any(type(result.get(key, {}).get(p)) is not int or result[key][p] != 1
                       for key in ('attempted', 'completed') for p in ('build', 'selftest', 'parse'))
                or any(result.get(key) != 0 for key in parser_entry.ZERO_FIELDS)):
            raise ValueError('Original target parser must complete every phase without CKKS')
        for phase in ('build', 'selftest', 'parse'):
            clean((result.get('phases', {}).get(phase) or {}).get('observation') or {})
        bound = (result.get('source') or {}).get('optimized_source') or {}
        if (bound.get('model_key') != args.model or bound.get('model') != factory
                or type(bound.get('placement_bootstrap_nodes')) is not int or bound['placement_bootstrap_nodes'] < 0
                or Path(bound.get('source_result', '')).resolve() != (cases['optimizer']/'result.json').resolve()
                or Path(bound.get('frontend_result', '')).resolve() != (cases['frontend']/'result.json').resolve()
                or Path(result['source'].get('shared_build_case', '')).resolve() != args.shared_build_case.resolve()):
            raise ValueError('Parser must consume this fresh optimized model and shared backend')
    row = registry_row(args.registry, case, result)
    return dict(result=str(case/'result.json'), completion=str(case/'completion.json'),
        registry=row, complete_case_seconds=completion.get('complete_case_elapsed_seconds'),
        source_status=result['status'], model=factory,
        placement_bootstrap_nodes=((result.get('source') or {}).get('optimized_source') or {}).get('placement_bootstrap_nodes') if stage=='parser' else None,
        scope='Nested child case; its clock is not added to the master continuous clock')


def resource_observation(args):
    table = _scan(); owned = {os.getpid()}
    while True:
        expanded = owned | {pid for pid, row in table.items() if row.get('ppid') in owned}
        if expanded == owned: break
        owned = expanded
    sample = dict(subtree_rss_bytes=sum((table.get(pid, {}).get('vmrss_kib') or 0)*1024 for pid in owned))
    observation = physical_memory_observation(sample, _available_ram_bytes(), rss_gib=16, minimum_available_gib=1.25)
    observation.update(case_bytes=artifact_bytes(args.out), disk_free_bytes=shutil.disk_usage(args.out).free)
    if observation['case_bytes']+RESERVE > CASE_BYTES: observation['reasons'].append('48GiB aggregate artifact allowance')
    if observation['disk_free_bytes'] < 10*GIB: observation['reasons'].append('10GiB free disk reserve')
    observation['stop'] = bool(observation['reasons'])
    return observation


@contextmanager
def master_lease(root):
    with (root/'orbit-fresh-compile-e2e.lock').open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def execute(args):
    if args.out.exists() or args.out.is_symlink() or args.out.parent != paths.RESULTS_DIR.resolve():
        raise ValueError('A fresh direct master result case is required')
    started = time.perf_counter_ns(); deadline = time.monotonic()+args.seconds
    args.out.mkdir(); (args.out/'cases').mkdir(); cases = child_cases(args)
    recorder = StageRecorder(args.out/'events.jsonl'); stop = threading.Event(); watcher = None
    result = dict(round=ROUND, model=frontend.MODEL_RECORDS[args.model]['factory'], model_key=args.model,
        mode=MODE, status='running', error=None, attempted=dict.fromkeys(STAGES, 0), completed=dict.fromkeys(STAGES, 0),
        accepted=False, processes=[], child_cases={key:str(value) for key,value in cases.items()}, children={},
        guard_trigger=None, guard_stop_signal=None, incoming_signal=None, formal_latency=None, accuracy_acceptance=None,
        security_acceptance=None, model_specific_native_codegen=None, compiled_bootstraps=None,
        placement_bootstrap_nodes=None,
        bootstrap_count_unit='Optimizer/parser static graph nodes; not ciphertext executions',
        historical_times_added=False, original_trace_reused=False, complete_e2e_clock_file=str(args.out/'completion.json'),
        clock_scope='Continuous fresh frontend through optimizer, target parser and required master/registry persistence; child clocks nested',
        limits=dict(AS_gib=24, aggregate_RSS_gib=16, minimum_available_gib=1.25, file_gib=16,
            aggregate_case_gib=48, minimum_free_disk_gib=10, total_seconds=args.seconds,
            original_outer_seconds=dict(frontend=3000,optimizer=8400,parser=1500),
            child_supervisor_seconds=CHILD_SECONDS, original_child_physical_guards_retained=True),
        resnet_parser_model_record_extension_only=args.model=='resnet20', **dict.fromkeys(ZERO,0))
    handlers = {}
    def interrupt(sig, frame):
        result['guard_stop_signal' if result['guard_trigger'] else 'incoming_signal'] = sig
        raise InterruptedError('Orbit E2E interrupted '+str(sig))
    def inspect():
        observation = resource_observation(args)
        if time.monotonic() >= deadline-30:
            observation['stop'] = True; observation['reasons'].append('Finite master wall/closure reserve')
        if observation['stop']: result['guard_trigger'] = observation
        return observation
    def monitor():
        while not stop.wait(.25):
            try:
                observed = inspect()
                if not observed['stop']: continue
                durable_json(args.out/'guard.json', observed)
            except BaseException as exc:
                result['guard_trigger'] = dict(error=type(exc).__name__+': '+str(exc))
            finally:
                if result['guard_trigger'] is not None:
                    result['guard_stop_signal']=signal.SIGTERM
                    os.kill(os.getpid(),signal.SIGTERM)
            if result['guard_trigger'] is not None:return
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupt)
        with master_lease(paths.RESULTS_DIR),recorder.span('complete-fresh-orbit-compile'):
            for path in (args.shared_build_case/'result.json',args.shared_build_case/'completion.json',args.initialization_audit):
                if not path.is_file():raise ValueError('Existing shared backend/initialization evidence required')
            shutil.copyfile(__file__,args.out/'driver-source.py')
            declared=commands(args,cases);durable_json(args.out/'commands.json',declared)
            if inspect()['stop']:raise RuntimeError('Orbit E2E resource admission failed')
            watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
            for stage in STAGES:
                if inspect()['stop']:raise RuntimeError('Orbit E2E resource allowance exhausted')
                remaining=deadline-time.monotonic()-30
                if remaining<=0:raise TimeoutError('No bounded child wall allowance remains')
                row=dict(stage=stage,argv=declared[stage],observation=None,cleanup_unknown=True)
                result['processes'].append(row)
                result['attempted'][stage]=1;durable_json(args.out/'result.json',result)
                with recorder.span(stage):
                    row['observation']=run_process(declared[stage],cwd=paths.REPO/'DSL/DSL_agent/scripts',
                        log=args.out/(stage+'.log'),resources=args.out/(stage+'-resources.jsonl'),
                        timeout=min(CHILD_SECONDS[stage],remaining),mem_gib=24,fsize_gib=16)
                    row['cleanup_unknown']=False
                    clean(row['observation'])
                    result['children'][stage]=accept_child(args,stage,cases)
                    result['completed'][stage]=1
            result['placement_bootstrap_nodes']=result['children']['parser']['placement_bootstrap_nodes']
            result['status']='ok'
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if watcher is not None:watcher.join(timeout=3)
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:
            try:result['resource_completion']=inspect()
            except BaseException as exc:
                result['resource_completion']=dict(stop=True,reasons=[type(exc).__name__+': '+str(exc)])
            if result['guard_trigger'] or result['resource_completion']['stop']:
                result.update(status='error',error=result['error']or'Terminal resource policy rejected')
            result['timing']=summarize_events(read_events(recorder.path))
            for _ in range(20):
                size=artifact_bytes(args.out)
                if size+RESERVE>CASE_BYTES:result.update(status='error',error='Master completion reserve exhausted')
                result['accepted']=result['status']=='ok'and result['completed']==dict.fromkeys(STAGES,1)
                result['metrics']={key:result[key]for key in ('status','attempted','completed','accepted',*ZERO)}
                durable_json(args.out/'result.json',result)
                if artifact_bytes(args.out)==size:break
            else:raise RuntimeError('Master terminal artifact accounting did not converge')
            from ..registry import git_rev
            row=dict(run_id=args.out.name,round=ROUND,model=result['model'],mode=MODE,git_rev=git_rev(),
                metrics=result['metrics'],artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat())
            with args.registry.open('a+')as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.seek(0)
                if any(json.loads(line).get('run_id')==args.out.name for line in stream if line.strip()):
                    raise ValueError('Preserve duplicate master registry identity')
                stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],
                timing=result['timing'],required_artifact_bytes=artifact_bytes(args.out))
            if artifact_bytes(args.out)>CASE_BYTES and result['status']!='error':
                raise RuntimeError('Completion exceeded its reserved artifact allowance')
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return result


def parser():
    cli=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    cli.add_argument('--model',choices=MODELS,required=True);cli.add_argument('--out',type=Path,required=True)
    for name in ('shared-build-case','go-sdk','initialization-audit'):cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--orbit-root',type=Path,default=paths.ORBIT_ROOT)
    cli.add_argument('--registry',type=Path,default=paths.RESULTS_DIR/'registry.jsonl')
    for name in ('checkpoint','r13-result','r16-result','poly-dir','deps-dir','intake-python','orbit-python'):
        cli.add_argument('--'+name,type=Path)
    cli.add_argument('--seconds',type=int,default=TOTAL_SECONDS);cli.add_argument('--run',action='store_true')
    return cli


def main(argv=None):
    values=sys.argv[1:]if argv is None else list(argv)
    if '--delegate-parser'in values:
        if values.count('--delegate-parser')!=1:raise ValueError('One explicit private parser delegation required')
        values.remove('--delegate-parser');return parser_namespace()['main'](values)
    cli=parser();args=cli.parse_args(values)
    if not args.run or not 1<=args.seconds<=TOTAL_SECONDS:cli.error('Use --run and a finite1..13200second allowance')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.expanduser().absolute()if name.endswith('python')else value.expanduser().resolve())
    result=execute(args)
    print(json.dumps({key:result.get(key)for key in ('status','accepted','completed','error')}),flush=True)
    return int(result['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
