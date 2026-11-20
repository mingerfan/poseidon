"""Independent bounded original-embedding and unfrozen full-Count jobs.

These entries reuse existing source artifacts and dependencies.  A successful
job covers its named diagnostic, while complete encrypted generation remains
unbound and no candidate is frozen.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .. import paths
from .r10_paper_metrics import read_events, run_process, summarize_events, write_json


VARIANTS = ('original-streaming', 'compact-rectangular')
CASE_BYTES = 4*1024**3
RESERVE_BYTES = 10*1024**3


def job_definition(args):
    """Build explicit commands without reading weights or starting a child."""
    work = args.out/'worker'
    variant = args.linear_variant
    if variant not in VARIANTS:
        raise ValueError('Unknown explicitly selected linear schedule')
    if args.job == 'quality-own-mock-step1':
        command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','5400',
            '--python',str(paths.INTAKE_PYTHON),'--','-B','-m','pt_agent.experiments.qwen_own_output_packed',
            '--work-dir',str(work),'--model-info',str(args.model_info),
            '--reference',str(args.original_hf_reference),'--count-result',str(args.count_result),
            '--own-output-result',str(args.own_output_result),'--prefill-result',str(args.prefill_result),
            '--prefill-logits',str(args.prefill_logits),'--supervisor-pid',str(os.getpid())]
        return dict(command=command,memory_gib=12,resident_limit_gib=5,timeout=5460,file_gib=1,
            mode='fixed-precision-cached-packed-prefill-own-output-step1-math')
    if args.job == 'quality-own-tensor':
        command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','900',
            '--python',str(paths.INTAKE_PYTHON),'--','-B','-m','pt_agent.experiments.qwen_autoregressive_quality',
            '--work-dir',str(work),'--model-info',str(args.model_info),
            '--reference',str(args.original_hf_reference),'--count-result',str(args.count_result)]
        return dict(command=command,memory_gib=6,timeout=960,file_gib=1,
            mode='fixed-precision-approximate-own-output-training-generation')
    if args.job in ('quality-tensor','quality-mock'):
        packed=args.job=='quality-mock'
        command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','5400' if packed else '900',
            '--python',str(paths.INTAKE_PYTHON),'--','-B','-m','pt_agent.experiments.qwen_precision_quality',
            '--phase','mock' if packed else 'tensor','--work-dir',str(work),'--model-info',str(args.model_info),
            '--reference',str(args.original_hf_reference),'--count-result',str(args.count_result)]
        if packed:
            command.extend(['--tensor-result',str(args.tensor_result)])
        return dict(command=command,memory_gib=12 if packed else 6,timeout=5460 if packed else 960,file_gib=1,
            mode='fixed-precision-original-training-state-packed-math' if packed else
                 'fixed-precision-original-training-prefix-quality')
    if args.job == 'embedding':
        command = [str(paths.ORBIT_ROOT.parent/'orbit-venv/bin/python'), '-B', '-m',
            'pt_agent.experiments.qwen_embedding_validation', '--phase', 'validate',
            '--work-dir', str(work), '--model-info', str(args.model_info/'model_info.json'),
            '--config', str(args.local_model/'config.json'), '--weights-archive',
            str(args.model_info/'graph/weights.npz'), '--member', 'original.model.embed_tokens.weight.npy',
            '--token-record', str(args.token_record), '--expected-token', '8559',
            '--linear-variant', variant]
        return dict(command=command, memory_gib=4, timeout=900, file_gib=1,
                    mode=variant+'-original-embedding-count-math')
    if args.job not in ('stable-calibration','stable-fit','stable-count'):
        raise ValueError('Unknown diagnostic job')
    calibration = args.job == 'stable-calibration'
    decomposition = getattr(args,'decomposition_variant','minimum-domain-public-shift-square-v1')
    newton_budget = getattr(args,'newton_budget',5)
    fitting = args.job == 'stable-fit'
    command = ([str(paths.ORBIT_ROOT.parent/'orbit-venv/bin/python'),'-B','-m'] if fitting else
        [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '900' if calibration else '1800',
         '--python', str(paths.INTAKE_PYTHON), '--', '-B', '-m'])+[
        'pt_agent.experiments.qwen_stable_calibration', '--phase', 'calibration' if calibration else 'count', '--work-dir', str(work),
        '--model-info', str(args.model_info), '--reference', str(args.original_hf_reference),
        '--framework-result', str(args.framework_result), '--bridge-result', str(args.stable_calibration_result),
        '--decomposition-variant',decomposition,'--newton-budget',str(newton_budget)]
    if not calibration and getattr(args,'fit_source',None) is not None:
        command.extend(['--fit-source',str(args.fit_source)])
    if fitting:
        command.extend(['--stop-after','fit'])
    command.extend(['--linear-variant',variant])
    suffix = ('-stable-softmax-calibration' if calibration else
              '-stable-softmax-unfrozen-fit-only' if fitting else '-stable-softmax-unfrozen-count')
    if decomposition != 'minimum-domain-public-shift-square-v1' or newton_budget != 5:
        suffix += '-'+decomposition+'-newton'+str(newton_budget)
    return dict(command=command,memory_gib=6 if calibration or fitting else 12,
                resident_limit_mib=600 if fitting else None,timeout=960 if calibration else 900 if fitting else 1860,
                file_gib=1,mode=variant+suffix)


def check_completed(worker, job, variant):
    if worker.get('status') != 'ok' or (job not in ('stable-calibration','quality-tensor','quality-mock','quality-own-tensor','quality-own-mock-step1') and worker.get('linear_variant') != variant):
        raise ValueError('The selected diagnostic did not complete with its recorded schedule')
    if job=='quality-own-mock-step1':
        expected=dict(mock_forward_attempts=1,mock_forward_completed=1,tensor_forward_attempts=0,
            tensor_forward_completed=0,retained_constant_arrays=0,constant_identity_entries=0,
            fresh_hf_forwards=0,approximation_selection_runs=0,api_calls=0,emit_runs=0,
            compiler_runs=0,keygen_runs=0,ckks_executions=0,heldout_samples=0,confirm_samples=0,report_samples=0)
        if (worker.get('source_checked') is not True
                or worker.get('independent_packed_math_passed') is not True
                or worker.get('cached_first_step_continuation_verified') is not True
                or worker.get('strict_tensor_mock_math',{}).get('passed') is not True
                or worker.get('count_mock_schedule_matched') is not True
                or worker.get('handoff',{}).get('mock_argmax_matches_tensor') is not True
                or worker.get('handoff',{}).get('actual_second_input_matches_tensor') is not True
                or worker.get('resident_guard',{}).get('stopped') is not False):
            raise ValueError('The actual cached packed output must drive the verified second packed state')
    elif job=='quality-own-tensor':
        expected=dict(tensor_forward_attempts=6,tensor_forward_completed=6,
            state_update_attempts=3,state_update_completed=3,generated_token_count=6,
            mock_forward_attempts=0,mock_forward_completed=0,fresh_hf_forwards=0,
            heldout_samples=0,confirm_samples=0,report_samples=0,api_calls=0,
            approximation_selection_runs=0,emit_runs=0,compiler_runs=0,keygen_runs=0,
            ckks_executions=0,intermediate_decryptions=0)
        if (worker.get('source_checked') is not True
                or worker.get('own_output_generation_completed') is not True
                or worker.get('all_finite') is not True
                or worker.get('approximate_tokens_fed_back') is not True
                or worker.get('full_ciphertext_model_binding_verified') is not False
                or worker.get('candidate_frozen') is not False
                or type(worker.get('training_quality_passed')) is not bool
                or len(worker.get('sample_rows',[])) != 6
                or len(worker.get('trajectory_rows',[])) != 3):
            raise ValueError('The original three training prompts must complete actual approximate Tensor feedback')
    elif job=='quality-tensor':
        expected=dict(tensor_forward_attempts=6,tensor_forward_completed=6,
                      mock_forward_attempts=0,mock_forward_completed=0)
        if (worker.get('source_checked') is not True
                or worker.get('training_prefix_measurement_completed') is not True
                or worker.get('all_finite') is not True
                or worker.get('approximate_tokens_fed_back') is not False):
            raise ValueError('The fixed original training-prefix measurement must remain complete and finite')
    elif job=='quality-mock':
        expected=dict(tensor_forward_attempts=0,tensor_forward_completed=0,
                      mock_forward_attempts=1,mock_forward_completed=1,
                      retained_constant_arrays=0,constant_identity_entries=0)
        if (worker.get('source_checked') is not True
                or worker.get('independent_packed_math_passed') is not True
                or worker.get('strict_tensor_mock_math',{}).get('passed') is not True
                or worker.get('count_mock_schedule_matched') is not True):
            raise ValueError('The independent original-state packed comparison must pass its original strict math')
    elif job == 'embedding':
        expected = dict(count_attempts=1,count_completed=1,math_attempts=1,math_completed=1,
            mapping_completed=1,original_weight_members_mapped=1,count_constant_identity_entries=0,
            count_retained_array_constants=0)
        if (worker.get('token_probe', {}).get('token_id') != 8559
                or worker.get('math', {}).get('passed') is not True):
            raise ValueError('The original training token or strict embedding mathematics differs')
    elif job in ('stable-fit','stable-count'):
        expected = dict(rule_plan_attempts=1,rule_plan_completed=1,count_attempts=0 if job=='stable-fit' else 1,
                        count_completed=0 if job=='stable-fit' else 1)
        if job=='stable-count':
            expected.update(retained_constant_arrays=0,constant_identity_entries=0)
        elif worker.get('fit_only') is not True or worker.get('weight_payloads_read') != 0:
            raise ValueError('The separate fit-only diagnostic must stop before Count and weight arithmetic')
        if worker.get('diagnostic_typecheck_passed') is not True:
            raise ValueError('Full Count requires the original-model diagnostic typecheck')
    else:
        expected = dict(mapped_forward_attempts=6,mapped_forward_completed=6,
                        stable_softmax_attempts=144,stable_softmax_completed=144)
        if (worker.get('stable_bridge_validated') is not True
                or worker.get('selected_decomposition_calibration_complete') is not True):
            raise ValueError('Every original state and selected softmax domain must pass')
    for key, value in expected.items():
        if type(worker.get(key)) is not int or worker[key] != value:
            raise ValueError('Unexpected actual diagnostic count: '+key)
    for key in ('compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance'):
        if worker.get(key) is not None:
            raise ValueError('An unmeasured formal result was filled: '+key)


def _write_terminal(out, result):
    target = out/'result.json'
    for _ in range(20):
        payload = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
        size = sum(path.stat().st_size for path in out.rglob('*')
                   if path.is_file() and path != target)+len(payload.encode())
        if result.get('case_artifact_bytes') == size:
            target.write_text(payload)
            return size
        result['case_artifact_bytes'] = size
    raise RuntimeError('Terminal diagnostic metadata size did not stabilize')


def execute(args):
    args.out = args.out.resolve()
    if args.out.parent != paths.RESULTS_DIR.resolve() or args.out.exists():
        raise ValueError('Use a new independent run directly under the configured results root')
    if shutil.disk_usage(paths.RESULTS_DIR).free < RESERVE_BYTES:
        raise RuntimeError('Preserve 10GiB free disk before starting this diagnostic')
    definition = job_definition(args)
    args.out.mkdir()
    work = args.out/'worker'
    work.mkdir()
    write_json(args.out/'command.json', definition)
    snapshots = ('experiments/qwen_diagnostic_jobs.py','experiments/qwen_embedding_validation.py',
        'experiments/qwen_precision_quality.py','experiments/qwen_mock_progress.py',
        'experiments/qwen_autoregressive_quality.py',
        'experiments/qwen_own_output_packed.py',
        'experiments/qwen_stable_calibration.py','kernels/compact_linear.py','kernels/streaming_linear.py',
        'planners/stable_softmax_bridge.py','planners/precision_variants.py','kernels/unit_width_softmax_reference.py')
    source = Path(__file__).resolve().parents[2]
    for relative in snapshots:
        destination = args.out/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/'pt_agent'/relative, destination)
    environment = {key:value for key,value in os.environ.items()
        if not key.startswith(('LD_','NIX_','__NIX_'))
        and key not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','IN_NIX_SHELL')}
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[key] = '1'
    result = dict(run_id=args.out.name, round='autonomous-qwen-generation', model='Qwen2-0.5B',
        mode=definition['mode'], status='running', job=args.job, linear_variant=args.linear_variant,
        linear_variant_applied=args.job not in ('quality-tensor','quality-own-tensor'),
        decomposition_variant=getattr(args,'decomposition_variant','minimum-domain-public-shift-square-v1'),
        newton_budget=getattr(args,'newton_budget',5),
        full_ciphertext_model_binding_verified=False, candidate_frozen=False,
        count_scope=('one original-token embedding projection' if args.job == 'embedding' else
            'three training prompts with two approximate own-output Tensor forwards and one token update each' if args.job=='quality-own-tensor' else
            'six fixed original HF training-prefix states; approximate model quality only' if args.job=='quality-tensor' else
            'cached first packed prefill output drives one newly executed second packed forward' if args.job=='quality-own-mock-step1' else
            'one original train:0 prefill state; independent packed arithmetic versus the same fixed tensor program' if args.job=='quality-mock' else
            'six original-framework mapped states and144 stable softmax checks' if args.job=='stable-calibration' else
            'source-matched unfrozen121-role fitting and typecheck; zero weight arithmetic or Count' if args.job=='stable-fit' else
            'one complete original 32-position forward graph; two forwards, selector and encrypted prefix state are not composed'),
        source=dict(model_info=str(args.model_info),token_record=str(args.token_record),
                    original_hf_reference=str(args.original_hf_reference),framework_result=str(args.framework_result),
                    stable_calibration_result=str(args.stable_calibration_result)))
    observation, worker = None, None
    try:
        observation = run_process(definition['command'], cwd=source, log=args.out/'worker.log',
            resources=args.out/'resources.jsonl', timeout=definition['timeout'],
            mem_gib=definition['memory_gib'], fsize_gib=definition['file_gib'], env=environment)
        if (work/'worker-result.json').is_file():
            worker = json.loads((work/'worker-result.json').read_text())
        if observation['exit_code'] != 0 or not worker:
            raise RuntimeError('The bounded diagnostic stopped; no following job was started')
        check_completed(worker, args.job, args.linear_variant)
        if args.job in ('quality-tensor','quality-mock','quality-own-tensor','quality-own-mock-step1'):
            result.update(decomposition_variant=worker['decomposition_variant'],newton_budget=worker['newton_budget'])
        if args.job not in ('embedding','quality-tensor','quality-mock','quality-own-tensor','quality-own-mock-step1') and worker.get('decomposition_variant')!=result['decomposition_variant']:
            raise ValueError('The actual selected decomposition differs from the requested variant')
        if args.job in ('stable-fit','stable-count') and worker.get('newton_budget') != result['newton_budget']:
            raise ValueError('The actual Newton budget differs from the explicitly requested variant')
        if args.job == 'stable-calibration' and result['decomposition_variant']=='unit-width-public-square-v2':
            if len(worker.get('score_cache',[])) != 144:
                raise ValueError('All144 original score observations must be preserved')
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc))
    result.update(process=observation, worker=worker,
        timing=summarize_events(read_events(work/'events.jsonl')),
        disk_free_after=shutil.disk_usage(paths.RESULTS_DIR).free)
    metrics = {key:(worker or {}).get(key) for key in
        ('count_attempts','count_completed','math_attempts','math_completed','rule_plan_attempts',
         'rule_plan_completed','count_stats','linear_variant','mapped_forward_attempts','mapped_forward_completed',
         'stable_softmax_attempts','stable_softmax_completed','stable_bridge_validated','decomposition_variant',
         'newton_budget','fitting_evidence_counts','tensor_forward_attempts','tensor_forward_completed',
         'training_prefix_measurement_completed','training_quality_passed','mock_forward_attempts',
         'mock_forward_completed','independent_packed_math_passed','strict_tensor_mock_math','mock_stats',
         'state_update_attempts','state_update_completed','generated_token_count','own_output_generation_completed',
         'approximate_tokens_fed_back','original_hf_trajectory_matched','cached_first_step_continuation_verified','handoff')}
    metrics.update(strict_embedding_math_passed=(worker or {}).get('math',{}).get('passed'),
        full_ciphertext_model_binding_verified=False,candidate_frozen=False,api_calls=0,
        full_model_reference_forwards=(worker or {}).get('mapped_forward_completed') if args.job=='stable-calibration' else 0,
        full_generation_count_runs=0,emit_runs=0,compiler_runs=0,keygen_runs=0,real_ckks_executions=0,
        compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None)
    metrics['status'] = result['status']
    result['metrics'] = metrics
    if args.job in ('quality-tensor','quality-mock','quality-own-tensor','quality-own-mock-step1'):
        result['source']['count_result']=str(args.count_result)
    if args.job=='quality-mock':
        result['source']['tensor_result']=str(args.tensor_result)
    if args.job=='quality-own-mock-step1':
        result['source'].update(own_output_result=str(args.own_output_result),
            prefill_result=str(args.prefill_result),prefill_logits=str(args.prefill_logits))
    git = subprocess.run(['git','rev-parse','HEAD'], cwd=source, capture_output=True, text=True, check=True)
    result['git_rev'] = git.stdout.strip()
    result['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    size = _write_terminal(args.out, result)
    if size > CASE_BYTES or result['disk_free_after'] < RESERVE_BYTES:
        result.update(status='error',error_type='ResourceBudgetError',error='Diagnostic artifact or free-disk budget exceeded')
        metrics['status'] = result['status']
        _write_terminal(args.out, result)
    row = dict(run_id=result['run_id'],round=result['round'],model=result['model'],mode=result['mode'],
        git_rev=result['git_rev'],metrics=metrics,artifacts_dir=str(args.out),ts=result['ended_utc'])
    fd = os.open(paths.RESULTS_DIR/'registry.jsonl',os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o644)
    try:
        os.write(fd,(json.dumps(row)+'\n').encode())
    finally:
        os.close(fd)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('embedding','stable-calibration','stable-fit','stable-count','quality-tensor','quality-mock','quality-own-tensor','quality-own-mock-step1'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--linear-variant', choices=VARIANTS, default='compact-rectangular')
    parser.add_argument('--decomposition-variant',choices=('minimum-domain-public-shift-square-v1','unit-width-public-square-v2'),
                        default='minimum-domain-public-shift-square-v1')
    parser.add_argument('--newton-budget',type=int,choices=(5,7),default=5)
    parser.add_argument('--fit-source',type=Path)
    parser.add_argument('--count-result',type=Path,
        default=paths.RESULTS_DIR/'qwen-unit-width-newton7-count-corrected-20261115/result.json')
    parser.add_argument('--tensor-result',type=Path,
        default=paths.RESULTS_DIR/'qwen-fixed-precision-training-quality-20261115/worker/worker-result.json')
    parser.add_argument('--own-output-result',type=Path,
        default=paths.RESULTS_DIR/'qwen-own-output-training-generation-20261115/result.json')
    parser.add_argument('--prefill-result',type=Path,
        default=paths.RESULTS_DIR/'qwen-fixed-precision-packed-math-20261115/result.json')
    parser.add_argument('--prefill-logits',type=Path,
        default=paths.RESULTS_DIR/'qwen-fixed-precision-packed-math-20261115/worker/mock-logits.npy')
    parser.add_argument('--model-info',type=Path,
        default=paths.RESULTS_DIR/'qwen-normalization-rebased-20261115/work/import/model_info')
    parser.add_argument('--local-model',type=Path,
        default=paths.RESULTS_DIR/'preparation-20261115/models/Qwen2-0.5B')
    parser.add_argument('--token-record',type=Path,
        default=paths.RESULTS_DIR/'qwen-real-embedding-streaming-20261115/work/tokenize/worker-result.json')
    parser.add_argument('--original-hf-reference',type=Path,
        default=paths.RESULTS_DIR/'qwen-full-reference-20261115/work/hf-reference/worker-result.json')
    parser.add_argument('--framework-result',type=Path,
        default=paths.RESULTS_DIR/'qwen-framework-reference-complete-20261115/result.json')
    parser.add_argument('--stable-calibration-result',type=Path,
        default=paths.RESULTS_DIR/'qwen-stable-softmax-calibration-20261115/worker/worker-result.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare_only:
        print(json.dumps(job_definition(args),indent=2))
        return 0
    result = execute(args)
    print(json.dumps(dict(out=str(args.out),status=result['status'],metrics=result['metrics']),indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
