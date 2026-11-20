"""A5/A6: original fixed63 programs, optimized DaCapo, current qwen48.

Private function globals reuse R29's source gate, trusted Emit/trace, compiler
command and observation code. Only this entry admits 24 GiB address space and
adds a 14 GiB physical guard. It writes one paper row after each model closes.
"""
import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from types import FunctionType
from contextlib import contextmanager

from .. import paths
from . import r29_local_compile_matrix as original
from .a_group_runs import empty_record, append_record
from .dacapo_compile_continuation import latest_resource, available_ram
from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json
from .completion_metrics import persist_case_completion
from .r30_common_model_e2e import diagnostic_compile_argv

MODULE = 'pt_agent.experiments.a5_a6_rule63_search'
GIB = 1024**3
FACTORIES = dict(vgg16='pt_agent.experiments.r13_common_models:vgg16_silu_avgpool',
                 alexnet='pt_agent.experiments.r13_common_models:alexnet_silu_avgpool')


def worker_command(args, system, phase, program, work):
    argv, seconds = original._launch_worker(args, system, phase, program, work)
    index = argv.index('pt_agent.experiments.r29_local_compile_matrix')
    argv[index] = MODULE
    if args.diagnostic_elide_final_ir:
        argv.append('--diagnostic-elide-final-ir')
    return argv, seconds


def compiler_command(*args, diagnostic_elide_final_ir=False, **kwargs):
    command = original.compiler_argv(*args, **kwargs)
    return diagnostic_compile_argv(command, elide_final_ir=diagnostic_elide_final_ir)


def physical_guard_observation(resources, *, rss_gib=14, reserve_gib=1.25):
    sample = latest_resource(resources)
    available = available_ram()
    rss = sample['subtree_plus_supervisor_rss_bytes'] if sample else None
    reasons = []
    if rss is not None and rss > rss_gib * GIB:
        reasons.append('subtree including supervisors exceeds14GiB RSS')
    if available < reserve_gib * GIB:
        reasons.append('host available memory below1.25GiB')
    return dict(sample=sample, available_bytes=available, rss_bytes=rss,
                rss_limit_bytes=rss_gib * GIB, available_reserve_bytes=reserve_gib * GIB,
                reasons=reasons, stop=bool(reasons))


def guarded_process(*args, **kwargs):
    """Monitor exactly one outer R10 subtree; disarm before handler restoration."""
    resources = Path(kwargs['resources'])
    stop, armed, lock = threading.Event(), threading.Event(), threading.Lock()
    armed.set()
    def inspect():
        while not stop.wait(.25):
            try:
                row = physical_guard_observation(resources)
                if not row['stop']:
                    continue
            except BaseException as exc:
                row = dict(stop=True, reasons=['own physical monitor error'], error=type(exc).__name__ + ': ' + str(exc))
            with lock:
                if not armed.is_set():
                    return
                try:
                    write_json(resources.with_suffix('.physical-guard.json'), row)
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=inspect, daemon=True)
    watcher.start()
    try:
        return original.run_process(*args, **kwargs)
    finally:
        with lock:
            armed.clear()
        stop.set()
        watcher.join(timeout=2)
        if watcher.is_alive():
            raise RuntimeError('Own physical guard did not stop')


def namespace(args, *, outer=False):
    result = dict(vars(original))
    for name, value in tuple(result.items()):
        if isinstance(value, FunctionType) and value.__module__ == original.__name__:
            implementation = getattr(value, '__wrapped__', value)
            clone = FunctionType(implementation.__code__, result, implementation.__name__,
                                 implementation.__defaults__, implementation.__closure__)
            clone.__kwdefaults__ = implementation.__kwdefaults__
            result[name] = contextmanager(clone) if value is original.native_metadata_commands else clone
    result['__file__'] = __file__
    result['_launch_worker'] = worker_command
    result['compiler_argv'] = lambda *a, **kw: compiler_command(*a,
        diagnostic_elide_final_ir=args.diagnostic_elide_final_ir, **kw)
    def rule_source(*a, **kw):
        accepted = original.validate_rule_source(*a, **kw)
        return check63_program(accepted)
    result['validate_rule_source'] = rule_source
    if outer:
        result['run_process'] = guarded_process
    return result


def check63_program(accepted):
    from ..dsl.parser import parse
    program = parse(Path(accepted['program']).read_text())
    acts = [op for op in program.ops if op.kind == 'act']
    if (not acts or any(op.attrs.get('method') != 'cheb' or op.attrs.get('deg') != 63 for op in acts)
            or program.header.refresh != 'compiler'):
        raise ValueError('Retain the original fixed Chebyshev63 compiler-refresh program')
    return dict(accepted, activation_count=len(acts), degree=63, new_quality_evaluations=0,
                source_program_modified=False)


def fixed63_source(args):
    record = original._read(args.rule_result)
    info = original._read(args.model_info)
    accepted = original.validate_rule_source(record, result_path=args.rule_result,
        model_info=args.model_info, model_info_record=info, model_factory=args.model_factory)
    return check63_program(accepted)


def terminal_category(record, out):
    guards = list(Path(out).glob('*.physical-guard.json'))
    if any(original._read(path).get('error') for path in guards):
        return 'driver_error', 'Own physical monitor failed; see preserved guard observation'
    if guards or record.get('resource_monitor_error'):
        return 'resource_limit', '14 GiB RSS / 1.25 GiB host reserve / 64 GiB artifacts / 10 GiB free; see saved guard'
    if record.get('status') == 'ok':
        return 'ok', 'Original fixed Chebyshev63; optimized placement; current qwen48; no new samples or LLM'
    phase = (record.get('phases') or {}).get('compile') or {}
    process = phase.get('process') or {}
    if process.get('timed_out'):
        return 'timeout', 'Native compiler reached7200s wall allowance'
    if process.get('signal') is not None or process.get('exit_code') not in (None, 0):
        return 'compiler_crash', 'native compiler exit=' + str(process.get('exit_code')) + ' signal=' + str(process.get('signal'))
    messages = [record.get('error')] + [p.get('error') for p in (record.get('phases') or {}).values()]
    if 'MemoryError' in str(messages) or 'bad_alloc' in str(messages):
        return 'resource_limit', 'Allocation failed under24GiB address-space allowance; see original worker log'
    return 'driver_error', record.get('error') or 'Own preparation/trace/export driver did not complete'


def paper_record(args, record, out):
    status, detail = terminal_category(record, out)
    phase = (record.get('phases') or {}).get('compile') or {}
    process = phase.get('process') or {}
    spans = (phase.get('timing') or {}).get('spans') or []
    clocks = [row['exclusive_ns'] / 1e9 for row in spans
              if row.get('stage_id') == 'native_compiler_7200s' and row.get('exclusive_ns') is not None]
    peak = sampled_subtree_peak(out)
    complete = record.get('completion') or {}
    return empty_record(run_id=Path(args.out).name, task='A5' if args.model == 'vgg16' else 'A6',
        model=args.model, system='dacapo_search_ours', param_group='ours-N16-boot5', status=status,
        status_detail=detail, budget_s=7200, compile_s=clocks[0] if len(clocks) == 1 else None,
        peak_rss_gib=peak, boots_compiled=record.get('compiled_bootstraps'),
        exec_ops=phase.get('execution_steps'), api_calls=0,
        generation_e2e_s=None,
        stages_s={'compile':clocks[0]} if len(clocks) == 1 else {},
        source=str(Path(out) / 'result.json'), finished_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic_elide_final_ir=args.diagnostic_elide_final_ir,
        peak_rss_scope='maximum sampled active R10 child subtree plus its R10 supervisor across all model phases',
        preparation_compile_e2e_s=complete.get('complete_case_elapsed_seconds'),
        timing_scope='compile_s is exclusive native compiler span after trace; generation_e2e_s unmeasured for this compile-only control')


def sampled_subtree_peak(out):
    peak = None
    for path in Path(out).glob('*-resources.jsonl'):
        with path.open() as stream:
            for line in stream:
                try:
                    sample = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(sample.get('processes'), list):
                    continue
                amount = sum(p.get('vmrss_kib', 0) for p in sample['processes'])
                amount += (sample.get('outer_process') or {}).get('vmrss_kib', 0)
                peak = max(peak or 0, amount / 1024**2)
    return peak


def close_parent(args, child, record, row, *, started_ns, timing, registry=None):
    """Persist the requested outer case without adding child and parent clocks."""
    out=Path(args.out)
    child_completion=Path(child)/'completion.json'
    closed=original._read(child_completion) if child_completion.is_file() else {}
    status='ok' if (row['status']=='ok' and closed.get('status')=='ok'
        and closed.get('registry_persisted')is True and closed.get('required_artifacts_persisted')is True) else 'error'
    summary=dict(status=status,error=record.get('error') or (None if status=='ok' else row['status_detail']),
        model=args.model,system='dacapo-optimized',child_case=str(child),
        child_result=str(Path(child)/'result.json'),child_completion=str(child_completion),
        child_status=record.get('status'),paper_status=row['status'],
        compiled_bootstraps=record.get('compiled_bootstraps'),compile_s=row['compile_s'],
        generation_e2e_s=None,parent_clock_scope='Own launcher entry through parent artifact/registry persistence; includes original child once; no child/parent clock summation and no fresh Agent generation',
        timing=timing,api_calls=0)
    write_json(out/'result.json',summary)
    registry=Path(registry) if registry is not None else paths.RESULTS_DIR/'registry.jsonl'
    entry=dict(run_id=out.name,round='A5-A6-FIXED63',model=args.model,mode='optimized-parent',
        git_rev=args.git_rev,metrics=dict(status=status,paper_status=row['status'],
        compile_s=row['compile_s'],api_calls=0),artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):
            raise ValueError('Preserve previous parent registry identity')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(entry,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    persist_case_completion(out,round_label=entry['round'],started_ns=started_ns,status=status,
        timing=timing,required_artifact_bytes=original._size(out))
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=FACTORIES)
    p.add_argument('--model-factory', choices=tuple(FACTORIES.values()))
    p.add_argument('--worker-phase', choices=('emit','trace','compile'))
    p.add_argument('--system', choices=('dacapo-optimized',), default='dacapo-optimized')
    for name in ('out','work-dir','model-info','program','rule-result','metadata-template','profile','frontend','paper-runs'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--compile-seconds', type=int, default=7200)
    p.add_argument('--trace-seconds', type=int, default=1800)
    p.add_argument('--memory-gib', type=int, default=24)
    p.add_argument('--case-gib', type=int, default=64)
    p.add_argument('--diagnostic-elide-final-ir', action='store_true')
    return p


def main(argv=None):
    p = parser();args = p.parse_args(argv)
    if args.model_factory:
        selected = [name for name,factory in FACTORIES.items() if factory == args.model_factory][0]
        if args.model is not None and args.model != selected:
            p.error('Model/factory identities differ')
        args.model = selected
    if args.model is None:
        p.error('Choose one original VGG16 or AlexNet')
    args.model_factory = FACTORIES[args.model]
    if (args.compile_seconds != 7200 or args.trace_seconds != 1800
            or args.memory_gib != 24 or args.case_gib != 64):
        p.error('Use fixed compile7200/trace1800/AS24GiB/case64GiB protocol')
    args.debug_snapshots = False;args.archive_emit_constants_after_trace = False
    if args.worker_phase:
        if any(getattr(args,n) is None for n in ('work_dir','program','model_info','metadata_template','profile','frontend')):
            p.error('Worker paths must all arrive by argv')
        return namespace(args)['worker'](args)
    if args.out is None or args.paper_runs is None:
        p.error('Fresh result directory and paper-runs destination required')
    short = args.model_factory.split(':')[1]
    original_name = dict(vgg16='vgg16-silu-avgpool',alexnet='alexnet-silu-avgpool')[args.model]
    args.model_info = args.model_info or paths.RESULTS_DIR / ('r13-common-' + original_name + '-20261115/worker/model_info/model_info.json')
    args.rule_result = args.rule_result or paths.RESULTS_DIR / 'r17a-margin-heldout-20261115' / short / 'result.json'
    args.profile = args.profile or paths.PROFILES_DIR / 'qwen48/compiler-config.json'
    args.frontend = args.frontend or paths.DACAPO_ROOT / 'python/hecate/hecate/expr.py'
    args.metadata_template = args.metadata_template or paths.RESULTS_DIR / 'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json'
    args.agent_result = args.agent_audit = None
    for name in ('out','model_info','rule_result','metadata_template','profile','frontend','paper_runs'):
        setattr(args,name,getattr(args,name).absolute())
    if args.out.exists():
        raise ValueError('Preserve every prior model attempt')
    args.out.mkdir(parents=True)
    started=time.perf_counter_ns();rec=StageRecorder(args.out/'parent-events.jsonl')
    shutil.copyfile(original.__file__, args.out / 'original-r29-driver.py')
    observation = original.run_process(['git','rev-parse','HEAD'],cwd=paths.REPO,
        log=args.out/'git-revision.log',resources=args.out/'git-revision-resources.jsonl',
        timeout=15,mem_gib=.5)
    args.git_rev = ((args.out/'git-revision.log').read_text().strip()
                    if observation['exit_code']==0 else None)
    # Result identity uses the original R29 per-system registry and completion.
    out = args.out / 'dacapo-optimized'
    try:
        with rec.span('original_child_case'):
            record = namespace(args, outer=True)['run_case'](args,'dacapo-optimized',out)
    except BaseException as exc:
        record=original._read(out/'result.json') if (out/'result.json').is_file() else {}
        record.update(status='error',error=type(exc).__name__+': '+str(exc))
    write_json(args.out / 'source-readiness.json', record.get('source') or
               dict(status='source_gate_failed',error=record.get('error')))
    row = paper_record(args,record,out)
    append_record(args.paper_runs,row)
    write_json(args.out / 'paper-record.json',row)
    parent=close_parent(args,out,record,row,started_ns=started,
        timing=summarize_events(read_events(args.out/'parent-events.jsonl')))
    print(json.dumps(dict(out=str(args.out),status=row['status'],source=row['source'])))
    return 0 if parent['status']=='ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
