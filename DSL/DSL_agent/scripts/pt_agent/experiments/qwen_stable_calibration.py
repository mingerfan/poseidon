"""Source-bound full-Qwen stable-softmax calibration and unfrozen Count entry.

The calibration phase checks all six original HF states and the exact
float64 stable-softmax decomposition. The Count phase separately fits the
existing trusted rule and inventories it; it never freezes or evaluates a
candidate on a holdout, emits code, compiles, or executes CKKS.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
from types import ModuleType

from .r10_paper_metrics import StageRecorder, write_json


@contextmanager
def _metadata_intake_namespace():
    """Read original pure graph modules without initializing Torch frontends.

    This isolated fit-only path loads the real graph/core-op files through
    their regular Python module names. Its package directory is unchanged;
    the public intake initializer, which eagerly imports Torch, is unused.
    """
    name='pt_agent.intake'
    if name in sys.modules:
        yield dict(original_package_already_loaded=True,package_initializer_skipped=False)
        return
    before=dict(sys.modules)
    parent=sys.modules['pt_agent']
    previous_attribute=getattr(parent,'intake',None)
    had_attribute=hasattr(parent,'intake')
    package=ModuleType(name)
    package.__package__=name
    package.__path__=[str(Path(__file__).resolve().parents[1]/'intake')]
    sys.modules[name]=package
    try:
        metadata=dict(original_package_already_loaded=False,package_initializer_skipped=True,
            original_record_module_paths=[str(Path(package.__path__[0])/filename)
                                          for filename in ('graph.py','core_ops.py')])
        yield metadata
    finally:
        for key in list(sys.modules):
            if key==name or key.startswith(name+'.'):
                if key in before:
                    sys.modules[key]=before[key]
                else:
                    sys.modules.pop(key,None)
        if had_attribute:
            parent.intake=previous_attribute
        elif hasattr(parent,'intake'):
            delattr(parent,'intake')


def _fit_resident_snapshot():
    from .r10_paper_metrics import _scan
    table=_scan()
    # The direct worker's parent is its dedicated R10 supervisor. Including
    # that supervisor and its descendants accounts for the whole fitting job.
    family={os.getpid(),os.getppid()}
    while True:
        children={pid for pid,row in table.items() if row['ppid'] in family}
        if children.issubset(family):
            break
        family.update(children)
    rss=sum(table[pid].get('vmrss_kib',0) for pid in family if pid in table)
    available=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))
    return rss,available


@contextmanager
def _fit_resident_guard(limit_mib=600,available_floor_gib=2,observation=None):
    """Bound the separate NumPy fitting process without reading weight data."""
    if signal.getitimer(signal.ITIMER_REAL) != (0.,0.):
        raise RuntimeError('The isolated fitting worker requires an unused real-time timer')
    previous = signal.getsignal(signal.SIGALRM)
    observation={} if observation is None else observation
    observation.update(limit_mib=limit_mib,host_available_floor_gib=available_floor_gib,
        interval_seconds=.1,samples=0,peak_job_rss_kib=0,minimum_host_available_kib=None,
        scope='dedicated supervisor and all observed descendants',stopped=False)

    def check(signum=None,frame=None):
        rss_kib,available_kib=_fit_resident_snapshot()
        observation['samples']+=1
        observation['peak_job_rss_kib']=max(observation['peak_job_rss_kib'],rss_kib)
        previous_available=observation['minimum_host_available_kib']
        observation['minimum_host_available_kib']=available_kib if previous_available is None else min(previous_available,available_kib)
        if rss_kib > limit_mib*1024:
            observation.update(stopped=True,reason='fitting job resident limit')
            raise MemoryError('Separate fitting job RSS exceeded '+str(limit_mib)+'MiB')
        if available_kib < available_floor_gib*1024**2:
            observation.update(stopped=True,reason='host available memory reserve')
            raise MemoryError('Host available memory dropped below the fitting reserve')

    signal.signal(signal.SIGALRM,check)
    try:
        check()
        signal.setitimer(signal.ITIMER_REAL,.1,.1)
        yield observation
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        signal.signal(signal.SIGALRM,previous)


def _source(args):
    framework = json.loads(args.framework_result.read_text())
    worker = framework.get('worker') or {}
    if (framework.get('status') != 'ok' or worker.get('framework_precision_reference_passed') is not True
            or worker.get('mapped_forward_completed') != 6
            or Path(framework['model_info']).resolve() != args.model_info.resolve()
            or Path(framework['original_hf_reference']).resolve() != args.reference.resolve()
            or len(worker.get('math_checks', [])) != 6
            or any(row.get('passed') is not True for row in worker['math_checks'])):
        raise ValueError('The preserved original six-state HF/framework reference is required')
    original = json.loads(args.reference.read_text())
    if (original.get('status') != 'ok' or original.get('hf_forward_completed') != 6
            or len(original.get('trajectories', [])) != 3
            or any(not item['sample_id'].startswith('train:') for item in original['trajectories'])):
        raise ValueError('Only original real training-prefix trajectories are accepted')
    calibration_path = Path(worker['calibration'])
    return framework, worker, original, json.loads(calibration_path.read_text())


def calibrate(args, recorder, result, np, torch):
    from types import SimpleNamespace
    from ..models.mapped_info import load_mapped_info
    from ..planners.stable_softmax_bridge import build_bridge, observe_softmax
    decomposition = getattr(args,'decomposition_variant','minimum-domain-public-shift-square-v1')
    if decomposition not in ('minimum-domain-public-shift-square-v1','unit-width-public-square-v2'):
        raise ValueError('Unknown explicitly selected stable decomposition')
    reference = None
    if decomposition == 'unit-width-public-square-v2':
        from ..planners.precision_variants import build_unit_width_bridge as build_bridge
        from ..kernels.unit_width_softmax_reference import fixed_shift_squaring as reference
    from . import qwen_reference_worker as reference_worker
    with recorder.span('stable_bridge_source'):
        source, previous, original, calibration = _source(args)
        info = load_mapped_info(args.model_info)
        bridge = build_bridge(info.graph, calibration, raw_issues=previous.get('calibration_issues', []))
        result.update(bridge=bridge,decomposition_variant=decomposition,source=dict(model_info=str(args.model_info.resolve()),
            framework_result=str(args.framework_result.resolve()), original_hf_reference=str(args.reference.resolve()),
            calibration=str(previous['calibration'])), stable_softmax_checks=[])
        # The second mapped reader below shares file-backed pages, while these
        # metadata-only mappings are released before the real six forwards.
        del info
    write_json(args.work_dir/'worker-result.json', result)

    def observe(graph, values, trajectory, row, original_check):
        with recorder.span('stable_softmax_real_scores', sample_id=trajectory['sample_id'], step=row['step']):
            for selected in bridge['rows']:
                result['stable_softmax_attempts'] += 1
                result['active_stable_softmax'] = dict(node_id=selected['node_id'],
                    sample_id=trajectory['sample_id'], step=row['step'], completed=False)
                mask = values[selected['mask']] if selected['mask'] is not None else None
                if reference is not None:
                    directory = args.work_dir/'original-score-cache'
                    directory.mkdir(exist_ok=True)
                    stem = str(result['stable_softmax_attempts']).zfill(3)+'-'+selected['node_id']
                    score_path = directory/(stem+'-scores.npy')
                    np.save(score_path,values[selected['score_input']],allow_pickle=False)
                    mask_path = directory/(stem+'-mask.npy') if mask is not None else None
                    if mask_path is not None:
                        np.save(mask_path,mask,allow_pickle=False)
                    result.setdefault('score_cache',[]).append(dict(node_id=selected['node_id'],
                        sample_id=trajectory['sample_id'],step=row['step'],scores=str(score_path),
                        mask=str(mask_path) if mask_path else None,original_framework_values=True))
                arguments = dict(reference=reference) if reference is not None else {}
                _, checked = observe_softmax(values[selected['score_input']], mask, selected, **arguments)
                checked.update(sample_id=trajectory['sample_id'], step=row['step'])
                result['stable_softmax_checks'].append(checked)
                result['stable_softmax_completed'] += 1
                result['active_stable_softmax']['completed'] = True

    selected = SimpleNamespace(**dict(vars(args), phase='framework-reference',
        local_model=Path(original['original_hf_source']), _observation_callback=observe))
    reference_worker.mapped_reference(selected, recorder, result, np, torch)
    if result['mapped_forward_completed'] != 6 or result['stable_softmax_completed'] != 6*bridge['softmax_nodes']:
        raise ValueError('Every original forward and selected softmax must complete once')
    bridge['actual_score_coverage_measured'] = True
    bridge['real_forward_states'] = 6
    bridge['real_training_trajectories'] = 3
    bridge['stable_float64_math_passed'] = True
    bridge['original_hf_precision_math_passed'] = True
    write_json(args.work_dir/'stable-bridge.json', bridge)
    result.update(status='ok', stable_bridge_validated=True, bridge_path=str(args.work_dir/'stable-bridge.json'),
                  selected_decomposition_calibration_complete=True, raw_calibration_complete=result['calibration_complete'])


def count(args, recorder, result):
    from ..models.mapped_info import load_mapped_info
    from ..planners.stable_softmax_bridge import plan_with_bridge
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    variant = getattr(args, 'linear_variant', 'original-streaming')
    if variant == 'compact-rectangular':
        from ..kernels.compact_linear import compact_transformer_projections as projections
    elif variant == 'original-streaming':
        from ..kernels.streaming_linear import streaming_transformer_projections as projections
    else:
        raise ValueError('Unknown explicitly selected full-model linear schedule')
    from ..backends.streaming_count import StreamingCountBackend
    from ..lower import lower
    _source(args)
    result['linear_variant'] = variant
    result['decomposition_variant'] = getattr(args,'decomposition_variant','minimum-domain-public-shift-square-v1')
    result['newton_budget'] = getattr(args,'newton_budget',5)
    measured = json.loads(args.bridge_result.read_text())
    if (measured.get('status') != 'ok' or measured.get('stable_bridge_validated') is not True
            or measured.get('selected_decomposition_calibration_complete') is not True
            or Path(measured['source']['model_info']).resolve() != args.model_info.resolve()
            or Path(measured['source']['original_hf_reference']).resolve() != args.reference.resolve()):
        raise ValueError('Count requires the source-matched successful stable calibration')
    bridge = json.loads(Path(measured['bridge_path']).read_text())
    if bridge['variant'] != result['decomposition_variant']:
        raise ValueError('The selected decomposition differs from its actual measured bridge')
    calibration = json.loads(Path(measured['calibration']).read_text())
    with recorder.span('map_original_count_weights'):
        info = load_mapped_info(args.model_info)
        result['weight_mapping'] = info.mapped_weights
    result['fit_records'] = []
    builder,selector = None,None
    if result['decomposition_variant'] == 'unit-width-public-square-v2':
        from ..planners.precision_variants import build_unit_width_bridge
        builder = build_unit_width_bridge
    if getattr(args,'fit_source',None) is not None:
        with recorder.span('read_source_matched_fit_evidence'):
            from ..planners.precision_variants import recorded_selector
            from ..planners.transformer_rule import select_approximation
            original = json.loads(args.fit_source.read_text())
            source = original['source']
            for field,path in (('model_info',args.model_info),('original_hf_reference',args.reference),
                               ('framework_result',args.framework_result)):
                if Path(source[field]).resolve() != path.resolve():
                    raise ValueError('Recorded fitting evidence belongs to a different original source: '+field)
            prior = original['worker']
            if (original.get('model')!='Qwen2-0.5B' or prior.get('candidate_frozen') is not False
                    or len([row for row in prior.get('fit_records',[]) if row.get('kind')=='end'])!=121):
                raise ValueError('The complete unfrozen original121-role fitting evidence is required')
            selector,statistics = recorded_selector(prior['fit_records'],select_approximation,
                newton_budget=result['newton_budget'],observe=lambda:write_json(args.work_dir/'worker-result.json',result))
            result.update(recorded_fit_source=str(args.fit_source.resolve()),fitting_evidence_counts=statistics)
    elif result['newton_budget'] != 5:
        raise ValueError('The explicit Newton7 variant requires its source-matched recorded evidence')

    def observe_fit(kind, record):
        result['fit_records'].append(dict(kind=kind,observed_ns=time.perf_counter_ns(),**record))
        write_json(args.work_dir/'worker-result.json', result)

    with recorder.span('fit_unfrozen_stable_rule'):
        result['rule_plan_attempts'] = 1
        try:
            text = plan_with_bridge(info.graph, calibration, bridge, fit_observer=observe_fit,
                                    policy_builder=builder,approximation_selector=selector)
        except Exception as exc:
            from ..planners.transformer_rule import ApproximationLimitError
            if isinstance(exc, ApproximationLimitError):
                result['approximation_failure'] = dict(key=exc.key, report=exc.report,
                    issues=getattr(exc,'issues',[]), fixed_registered_search_preserved=True)
                write_json(args.work_dir/'worker-result.json', result)
            raise
        result['rule_plan_completed'] = 1
        path = args.work_dir/'diagnostic-program.ptdsl'
        path.write_text(text)
        result['diagnostic_program'] = str(path)
        program = parse(text)
        checked = check(program, {name: value.shape for name, value in info.graph.weights.items()}, graph=info.graph)
        if not checked.ok:
            raise ValueError('The automatically derived unfrozen program failed source/type checks: '+str(checked.diags))
        result['diagnostic_typecheck_passed'] = True
        write_json(args.work_dir/'worker-result.json', result)
    if getattr(args,'stop_after','count') == 'fit':
        result.update(status='ok',fit_only=True,count_only=False,
            numeric_fitting_diagnostics_recorded=True,continuous_polynomial_error_certificate=False,
            original_model_binding_verified=False,weight_payloads_read=0,
            fit_resident_limit_mib=600)
        return
    with recorder.span('streaming_full_model_count'):
        result['count_attempts'] = 1
        write_json(args.work_dir/'worker-result.json', result)
        backend = StreamingCountBackend(program.header.slots)
        with projections():
            values, layouts = lower(program, info.graph.weights, backend)
        result['count_stats'] = backend.stats()
        result['retained_constant_arrays'] = sum(item.data is not None for item in backend._constants.values())
        result['constant_identity_entries'] = len(backend._constant_objects)
        if result['retained_constant_arrays'] or result['constant_identity_entries']:
            raise ValueError('Streaming Count unexpectedly retained plaintext arrays')
        result['output_blocks'] = len(values)
        result['count_completed'] = 1
    result.update(status='ok', count_only=True, original_model_binding_verified=False,
                  numeric_fitting_diagnostics_recorded=True, continuous_polynomial_error_certificate=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('calibration', 'count'), required=True)
    for name in ('work-dir', 'model-info', 'reference', 'framework-result'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--bridge-result', type=Path)
    parser.add_argument('--linear-variant', choices=('original-streaming','compact-rectangular'),
                        default='original-streaming')
    parser.add_argument('--decomposition-variant',choices=('minimum-domain-public-shift-square-v1','unit-width-public-square-v2'),
                        default='minimum-domain-public-shift-square-v1')
    parser.add_argument('--newton-budget',type=int,choices=(5,7),default=5)
    parser.add_argument('--fit-source',type=Path)
    parser.add_argument('--stop-after',choices=('fit','count'),default='count')
    args = parser.parse_args(argv)
    if args.phase == 'count' and args.bridge_result is None:
        parser.error('--bridge-result is required for Count')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir/'worker-result.json').exists():
        raise FileExistsError('Preserve the previous worker result')
    recorder = StageRecorder(args.work_dir/'events.jsonl')
    result = dict(status='running', phase=args.phase, candidate_frozen=False, stable_bridge_validated=False,
        hf_forward_attempts=0, hf_forward_completed=0, mapped_forward_attempts=0, mapped_forward_completed=0,
        stable_softmax_attempts=0, stable_softmax_completed=0, rule_plan_attempts=0, rule_plan_completed=0,
        count_attempts=0, count_completed=0, api_calls=0, compiler_runs=0, emit_runs=0, keygen_runs=0,
        ckks_executions=0, heldout_samples=0, confirm_samples=0, report_samples=0,
        polynomial_quality_measured=False, compiled_bootstraps=None, predicted_latency=None,
        CKKS_latency=None, accuracy_acceptance=None)
    try:
        with recorder.span('qwen_stable_'+args.phase):
            if args.phase == 'calibration':
                if args.stop_after != 'count':
                    raise ValueError('The fit-only boundary belongs to the separate count planning phase')
                from .qwen_reference_worker import _numeric
                calibrate(args, recorder, result, *_numeric())
            elif args.stop_after == 'fit':
                for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
                    os.environ[name]='1'
                result['fit_resident_limit_mib']=600
                result['resident_guard']={}
                with _fit_resident_guard(observation=result['resident_guard']):
                    with _metadata_intake_namespace() as metadata:
                        result['fit_metadata_import']=metadata
                        count(args,recorder,result)
            else:
                from .qwen_reference_worker import _numeric
                _numeric()
                count(args, recorder, result)
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:3000])
    result['worker_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.work_dir/'worker-result.json', result)
    print(json.dumps({name: result[name] for name in ('status', 'phase', 'stable_softmax_completed',
                     'mapped_forward_completed', 'count_completed', 'worker_maxrss_kib')}))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
