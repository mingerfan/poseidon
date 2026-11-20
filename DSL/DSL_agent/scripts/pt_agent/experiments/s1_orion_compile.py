"""One native Orion fit/compile attempt using the original trained CNN.

The saved A2b training selection and R16a original-model compatibility evidence
are metadata prerequisites. Orion constructs its own native program and domain.
Its initialization, fit/compile and complete case clocks remain separate.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import FunctionType

from .. import paths
from . import cold_rule63_whole_model as cold
from . import a_group_future_closed_archive as closed
from . import a5_a6_shared_rule63_compile as shared
from .r10_paper_metrics import write_json

MODULE = 'pt_agent.experiments.s1_orion_compile'
WORKER = 'pt_agent.experiments.s1_orion_worker'
ROUND = 'S1-ORION-NATIVE-CNN'
MODE = 'native-orion-degree63-fit-compile-five-usable-levels'
MODELS = dict(resnet20='resnet20_silu', vgg16='vgg16_silu_avgpool', alexnet='alexnet_silu_avgpool')
DEADLINE = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
GIB = 1024**3
require, raw_stat = shared.require, shared.raw_stat


def read(path):
    value = shared.read(path)
    # Preserve the worker's native count and give the reused supervisor its
    # ordinary terminal field. This adds no RuntimePlan or latency estimate.
    if Path(path).name == 'phase_result.json' and value.get('mode') == 'orion-native-rule63-fit-compile':
        value = dict(value, compiled_bootstraps=value.get('compiled_execution_bootstraps'),
                     runtime_plan=None, predicted_latency_us=None)
        if isinstance(value.get('error'), dict):
            value['native_error'] = value['error']
            value['error'] = value['error'].get('type', '')+': '+value['error'].get('message', '')
    return value


def registry_row(case, result):
    selected = []
    with (paths.RESULTS_DIR/'registry.jsonl').open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('run_id') == result['run_id']:
                    selected.append(row)
    require(len(selected) == 1 and set(selected[0]) == closed.REGISTRY_FIELDS,
            'One exact eight-field source registry row required')
    row = selected[0]
    require(all(row.get(k) == result.get(k) for k in ('round', 'model', 'mode', 'git_rev'))
            and row.get('metrics', {}).get('status') == 'ok'
            and Path(row.get('artifacts_dir', '')).resolve() == case,
            'Source registry identity differs')
    return row


def configuration(config):
    """Read the prepared JSON; worker checks its complete native YAML origin."""
    original = read(config)
    native = original.get('orion') or {}
    require(native.get('backend') == 'lattigo' and native.get('io_mode') == 'none'
            and native.get('margin') == 1.5
            and original.get('ckks_params', {}).get('LogN') == 16
            and original['ckks_params'].get('LogQ') == [55,40,40,40,40,40]
            and original['ckks_params'].get('LogScale') == 40,
            'Prepared native N16/scale40/five-level/margin1.5 configuration required')
    return original


def source_gate(args, *, created=False):
    """Read bounded metadata and ordinary file stat; no tensors or samples."""
    from .r13_common_models import FACTORIES
    require(args.host_pid_namespace == os.readlink('/proc/self/ns/pid'),
            'Actual source closure must use the specified host PID namespace')
    root = paths.RESULTS_DIR.resolve()
    require(args.out.parent == root and not args.out.is_symlink()
            and (created or not args.out.exists()), 'A fresh results child required')
    require(datetime.now(timezone.utc) < DEADLINE, 'Orion experiment deadline exhausted')
    expected = root/('a2b-'+args.short_model+'-rule63-fresh-cold-depth-20261117')
    require(args.source_case == expected, 'Exact matching original A2b source required')
    result = read(expected/'result.json'); completion = read(expected/'completion.json')
    phases = dict.fromkeys(cold.PHASES, 1)
    require(result.get('status') == 'ok' and result.get('round') == cold.ROUND
            and result.get('mode') == cold.MODE and result.get('model') == args.model
            and result.get('run_id') == expected.name and result.get('placement') == 'depth'
            and result.get('attempted') == phases and result.get('completed') == phases
            and result.get('monitor_stopped') is True and not result.get('guard_error')
            and not result.get('monitor_error')
            and all(result.get('phases', {}).get(p, {}).get('status') == 'ok' for p in cold.PHASES),
            'Complete original A2b source required')
    require(completion.get('status') == 'ok' and completion.get('round') == cold.ROUND
            and completion.get('registry_persisted') is True
            and completion.get('required_artifacts_persisted') is True,
            'A2b completion must be durable')
    registry_row(expected, result); shared.metadata_cleanup(result)
    require([p.get('phase') for p in result.get('processes', [])] == list(cold.PHASES)
            and all(cold.clean(p['observation']) for p in result['processes']),
            'All original A2b workers must be cleanly closed')
    calibration = result['phases']['calibration']
    source = read(expected/'work/calibration/calibration-source.json')
    require(calibration.get('calibration_completed') is True
            and calibration.get('calibration_samples') == 1000
            and calibration.get('calibration_batch') == 1 and calibration.get('source') == source
            and source.get('split') == 'train' and source.get('count') == 1000
            and source.get('batch') == 1 and len(source.get('indices', [])) == 1000
            and len(set(source['indices'])) == 1000 and len(source.get('sample_ids', [])) == 1000,
            'Original fixed 1000 training calibration selection required')
    info = expected/'work/intake/model_info/model_info.json'
    require(Path(result['phases']['intake']['artifacts']['model_info']).resolve() == info.parent,
            'Original intake metadata directory differs')
    original = FACTORIES['pt_agent.experiments.r13_common_models:'+args.model]
    model_source, checkpoint = (Path(original[k]).resolve() for k in ('source', 'checkpoint'))
    r16 = root/'r16-baseline-compatibility-20261115'/('orion-'+args.model)
    require(args.compatibility_case == r16, 'Exact original R16a Orion compatibility case required')
    prior = read(r16/'result.json'); worker = prior.get('worker') or {}
    require(prior.get('round') == 'R16' and prior.get('mode') == 'baseline-plaintext-compatibility'
            and prior.get('baseline') == 'orion' and prior.get('status') == 'ok'
            and prior.get('model') == 'pt_agent.experiments.r13_common_models:'+args.model
            and worker.get('status') == 'ok'
            and all(worker.get(k) is True for k in ('compatibility_passed', 'all_finite',
                'source_checked', 'checkpoint_checked', 'model_built', 'strict_load_passed',
                'structure_passed', 'probes_checked', 'math_probes_passed')),
            'Original R16a strict original-model compatibility proof required')
    registry_row(r16, prior); shared.metadata_cleanup(prior)
    require(len(prior.get('processes', [])) == 1 and cold.clean(prior['processes'][0]),
            'Original R16a worker must be cleanly closed')
    artifacts = prior['source_r13']['artifacts']
    require(Path(artifacts['checkpoint']).resolve() == checkpoint
            and Path(artifacts['model_source']).resolve() == model_source
            and Path(artifacts['baseline_root']).resolve() == args.orion_root,
            'Canonical original weight/model/Orion paths differ')
    selected = dict(checkpoint=checkpoint, model_source=model_source, model_info=info,
                    records=Path(source['records']).resolve(), selections=Path(source['selections']).resolve())
    for key, path in selected.items():
        require(path.is_dir() if key == 'records' else path.is_file(), 'Missing original '+key)
    require(args.orion_root == paths.ORION_ROOT.resolve() and args.orion_root.is_dir()
            and args.orion_python == (args.orion_root.parent/'orion-venv/bin/python').absolute()
            and args.orion_python.is_file() and os.access(args.orion_python, os.X_OK),
            'Existing original Orion environment required')
    require(shutil.disk_usage(args.out.parent).free >= 10*GIB, 'resource_limit: free10GiB required')
    births = dict(a2b=closed.producer_births(expected), r16a=closed.producer_births(r16))
    require(all(v > 0 for v in births.values()), 'Saved source producer births required')
    return dict(a2b_case=str(expected), compatibility_case=str(r16),
        compatibility_persistence_scope='Historical R16a result, registry and actual clean process; original format has no completion.json',
        selected={k:str(v) for k,v in selected.items()}, calibration_source=str(expected/'work/calibration/calibration-source.json'),
        calibration=dict(split='train', count=1000, batch=1, indices=source['indices'],
                         sample_ids=source['sample_ids'], normalization=source['normalization']),
        closed_producer_births=births, host_pid_namespace=args.host_pid_namespace,
        config=configuration(args.config), checkpoint_stat=raw_stat(checkpoint),
        source_parameter_group='ours-N16-boot5', source_payload_reads=0)


def snapshot(args, proof):
    destination = args.out/'source'; destination.mkdir()
    for name in ('s1_orion_compile', 's1_orion_worker', 's1_orion_layer_translation',
                 'cold_rule63_whole_model', 'r16_baseline_validation_preflight', 'a_group_runs'):
        file = Path(__file__).with_name(name+'.py')
        shutil.copyfile(file, destination/file.name)
    shutil.copyfile(args.orion_root/'configs/resnet.yml', destination/'original-resnet.yml')
    write_json(destination/'orion-config.json', proof['config'])
    write_json(destination/'source-proof.json', proof)


def phase_command(args, phase, remaining):
    require(phase == 'compile', 'Only one original Orion worker allowed')
    seconds = min(args.native_seconds, int(remaining)-180)
    if seconds <= 0:
        raise TimeoutError('No Orion worker/closure allowance remains')
    proof = read(args.out/'source/source-proof.json')
    selected = proof['selected']
    command = [str(args.orion_python), '-B', '-m', WORKER, '--model', args.short_model,
        '--work-dir', str(args.out/'work/compile'), '--checkpoint', selected['checkpoint'],
        '--model-source', selected['model_source'], '--model-info', selected['model_info'],
        '--records', selected['records'], '--selections', selected['selections'],
        '--orion-root', str(args.orion_root), '--config', str(args.out/'source/orion-config.json')]
    return command, seconds


def classification(result):
    category = cold.classify(result)
    if category in ('resource_limit', 'timeout'):
        return category
    for row in result.get('processes', []):
        code = row.get('observation', {}).get('exit_code')
        if type(code) is int and code < 0:
            return 'compiler_crash'
    worker = result.get('phases', {}).get('compile') or {}
    proof = worker.get('no_placement_proof') or {}
    deep = proof.get('first_depth_exceeding_l_eff') or {}
    if (worker.get('status') == 'no_placement' and worker.get('compile_calls') == 1
            and proof.get('compile_calls') == 1 and proof.get('official_exception_origin') is True
            and proof.get('exception_type') == 'ValueError'
            and proof.get('function') == 'finally_solve_full_level_dag'
            and str(proof.get('message', '')).startswith('Automatic bootstrap placement failed.')
            and deep.get('degree') == 63 and isinstance(deep.get('depth'), int) and deep['depth'] > 5
            and proof.get('predicted_before_compile') is False):
        return 'no_placement'
    return 'ok' if result.get('status') == 'ok' else 'driver_error'


def paper_record(args, result, completion, timing):
    from .a_group_runs import empty_record
    worker = result.get('phases', {}).get('compile') or {}
    clocks = worker.get('stage_clocks_ns') or {}
    duration = worker.get('fit_compile_seconds')
    if duration is not None:
        require(isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0,
                'Finite observed Orion fit/compile clock required')
    row = empty_record(run_id=args.out.name, task='S1-Orion', model=args.short_model, system='orion',
        param_group='ours-N16-boot5', status=classification(result),
        status_detail=worker.get('error') or result.get('error'), budget_s=7200,
        compile_s=duration, peak_rss_gib=cold.peak_rss(args),
        boots_compiled=worker.get('compiled_execution_bootstraps'), exec_ops=None,
        pred_latency_s=None, pred_table=None, generation_e2e_s=None,
        stages_s={k:(v.get('elapsed_ns')/1e9 if isinstance(v, dict) and type(v.get('elapsed_ns')) is int else None)
                  for k,v in clocks.items()}, api_calls=0, source=str(args.out/'result.json'),
        finished_utc=datetime.now(timezone.utc).isoformat())
    row.update(cold=False, source_case=str(args.source_case),
        original_model_source=str(args.compatibility_case/'result.json'),
        compile_scope='Observed native Orion fit plus compile; initialization and key preparation excluded',
        init_scheme_seconds=worker.get('init_scheme_seconds'), stage_clocks_ns=clocks,
        actual_case_seconds=completion['complete_case_elapsed_seconds'],
        actual_case_scope='Current source admission, native initialization, fit, compile and persistence; no original A2b prefix reused',
        generation_e2e_scope='Unmeasured cold model-entry comparison; native Orion model construction is a separate program',
        same_full_parameters=False, domain_equivalent=False, same_program_as_rule63=False,
        native_parameter_mapping=worker.get('native_parameter_mapping') or dict(official_LogN=16, official_LogQ=[55,40,40,40,40,40],
            official_LogScale=40, official_input_level=5, official_usable_boot_output=5,
            comparison_input_level=8, comparison_boot_output=5, comparison_LogScale=43),
        native_domain_scope='Orion centered margin1.5 fit; Rule63 endpoint-scaled margin1.5 domains differ',
        native_reported_weighted_boots=worker.get('native_reported_weighted_boots'),
        bootstrap_count_scope='Official compilation plan count; encrypted execution unmeasured',
        CKKS_runs=0, actual_ckks_executed_bootstraps=None, formal_quality=None, runtime_plan=None,
        quality_retested=False, heldout_samples=0, confirm_samples=0, report_samples=0,
        fit_calls=worker.get('fit_calls', 0), compile_calls=worker.get('compile_calls', 0),
        no_placement_evidence=worker.get('no_placement_proof'),
        configuration=str(args.out/'source/orion-config.json'))
    return row


def execute(args):
    state = {}
    def admit(current, *, created=False):
        state['proof'] = source_gate(current, created=created)
        return state['proof']
    def snap(current): snapshot(current, state['proof'])
    def launch(*positional, **kwargs):
        from .r16_baseline_validation_preflight import _native_orion_env
        kwargs['env'] = _native_orion_env(kwargs.get('env'))
        return cold.run_process(*positional, **kwargs)
    def durable(path, value):
        if Path(path).parent == args.out and Path(path).name in ('result.json', 'progress.json'):
            value.update(cold=False, original_intake_reused=True, training_calibration_reused=True,
                source_case=str(args.source_case), same_full_parameters=False, domain_equivalent=False,
                same_program_as_rule63=False, original_A2b_elapsed_reused=False,
                limits=dict(memory_gib=24, physical_rss_gib=14, available_ram_gib=1.25,
                    file_gib=16, case_gib=args.case_gib, free_gib=10, fit_compile_worker_seconds=7200,
                    total_seconds=args.seconds, deadline_utc=DEADLINE.isoformat()))
            if Path(path).name == 'result.json':
                try:
                    if value['status'] == 'ok':
                        worker = value.get('phases', {}).get('compile') or {}
                        require(worker.get('native_compile_accepted') is True
                            and all(worker.get(k) == 1 for k in ('fit_calls', 'fit_completed',
                                'compile_calls', 'compile_completed', 'init_scheme_calls', 'init_scheme_completed'))
                            and worker.get('calibration_samples') == 1000
                            and worker.get('strict_translation_math_passed') is True
                            and worker.get('actual_CKKS_executions') == 0,
                            'Actual original Orion init/fit/compile and training math proof required')
                        original_calibration = state['proof']['calibration']
                        observed_calibration = worker.get('calibration_source') or {}
                        require(all(observed_calibration.get(k) == v for k,v in original_calibration.items()),
                                'Actual Orion calibration must match the saved original training selection')
                    from .dacapo_compile_continuation import available_ram
                    from .completion_metrics import artifact_bytes
                    require(available_ram() >= 1.25*GIB and shutil.disk_usage(args.out).free >= 10*GIB
                            and artifact_bytes(args.out) <= args.case_gib*GIB,
                            'resource_limit: final available RAM/disk/case guard')
                except BaseException as exc:
                    value.update(status='error', error=type(exc).__name__+': '+str(exc))
        write_json(path, value)
    namespace = dict(cold.__dict__)
    namespace.update(ROUND=ROUND, MODE=MODE, PHASES=('compile',), admission=admit, snapshot=snap,
        phase_command=phase_command, paper_record=paper_record, read=read, write_json=durable,
        run_process=launch)
    run = FunctionType(cold.execute.__code__, namespace, cold.execute.__name__,
                       cold.execute.__defaults__, cold.execute.__closure__)
    return run(args)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    for name in ('source-case', 'compatibility-case', 'out', 'paper-runs', 'orion-root', 'orion-python', 'config'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--host-pid-namespace', required=True)
    parser.add_argument('--seconds', type=int, default=7500)
    parser.add_argument('--native-seconds', type=int, default=7200)
    parser.add_argument('--case-gib', type=int, default=32)
    args = parser.parse_args(argv)
    require(all(getattr(args, k) is not None for k in ('out', 'paper_runs', 'config')),
            'Fresh output, prepared config and paper ledger required')
    args.short_model, args.model, args.placement = args.model, MODELS[args.model], 'orion'
    args.orion_root = args.orion_root or paths.ORION_ROOT
    args.orion_python = args.orion_python or args.orion_root.parent/'orion-venv/bin/python'
    args.source_case = args.source_case or paths.RESULTS_DIR/('a2b-'+args.short_model+'-rule63-fresh-cold-depth-20261117')
    args.compatibility_case = args.compatibility_case or paths.RESULTS_DIR/'r16-baseline-compatibility-20261115'/('orion-'+args.model)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().absolute() if key == 'orion_python' else value.expanduser().resolve())
    require(180 < args.seconds <= 7500 and 0 < args.native_seconds <= 7200 and args.case_gib == 32,
            'Finite total7500/native7200/case32GiB required')
    remaining = int((DEADLINE-datetime.now(timezone.utc)).total_seconds())-30
    args.seconds = min(args.seconds, remaining)
    require(args.seconds > 180, 'No Orion deadline/terminal persistence allowance remains')
    return args


def main(argv=None):
    args = arguments(argv)
    return int(execute(args)['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
