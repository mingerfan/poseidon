"""Portable, bounded original-Qwen preparation for the full-generation job.

The complete encrypted model binding remains an explicit missing stage.  This
runner reproduces the existing original-model reference path and stops with a
machine-readable incomplete status before any unsupported full native trace.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from .. import paths
from .r10_paper_metrics import run_process, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('reference','embedding','stable-calibration','stable-fit','stable-count','quality-tensor','quality-mock','quality-own-tensor','full'), default='full')
    parser.add_argument('--local-model', type=Path,
                        default=paths.RESULTS_DIR/'preparation-20261115/models/Qwen2-0.5B')
    parser.add_argument('--samples', type=Path,
                        default=paths.RESULTS_DIR/'r10-paper-qwen-import-20261115/worker/text-input/samples.jsonl')
    parser.add_argument('--source-rows', type=Path,
                        default=paths.RESULTS_DIR/'r10-paper-qwen-import-20261115/worker/text-input/source_rows.json')
    parser.add_argument('--saved-model-info', type=Path)
    parser.add_argument('--linear-variant', choices=('original-streaming','compact-rectangular'),
                        default='compact-rectangular')
    parser.add_argument('--decomposition-variant',
        choices=('minimum-domain-public-shift-square-v1','unit-width-public-square-v2'),
        default='minimum-domain-public-shift-square-v1')
    parser.add_argument('--newton-budget',type=int,choices=(5,7),default=5)
    parser.add_argument('--fit-source',type=Path)
    parser.add_argument('--count-result',type=Path,
        default=paths.RESULTS_DIR/'qwen-unit-width-newton7-count-corrected-20261115/result.json')
    parser.add_argument('--tensor-result',type=Path,
        default=paths.RESULTS_DIR/'qwen-fixed-precision-training-quality-20261115/worker/worker-result.json')
    parser.add_argument('--token-record', type=Path,
        default=paths.RESULTS_DIR/'qwen-real-embedding-streaming-20261115/work/tokenize/worker-result.json')
    parser.add_argument('--original-hf-reference', type=Path,
        default=paths.RESULTS_DIR/'qwen-full-reference-20261115/work/hf-reference/worker-result.json')
    parser.add_argument('--framework-result', type=Path,
        default=paths.RESULTS_DIR/'qwen-framework-reference-complete-20261115/result.json')
    parser.add_argument('--stable-calibration-result', type=Path,
        default=paths.RESULTS_DIR/'qwen-stable-softmax-calibration-20261115/worker/worker-result.json')
    parser.add_argument('--memory-gib', type=float, default=64.,
        help='Reference/full stages only; independent diagnostic jobs retain their fixed memory budgets')
    parser.add_argument('--phase-seconds', type=int, default=21600,
        help='Reference/full stages only; independent diagnostic jobs retain their fixed time budgets')
    args = parser.parse_args(argv)
    if args.memory_gib <= 0 or args.phase_seconds <= 0:
        parser.error('Resource limits must be positive')
    if args.stage in ('embedding','stable-calibration','stable-fit','stable-count','quality-tensor','quality-mock','quality-own-tensor'):
        from .qwen_diagnostic_jobs import main as diagnostic_main
        forwarded = ['--job',args.stage,'--out',str(args.out),'--linear-variant',args.linear_variant,
            '--local-model',str(args.local_model),'--token-record',str(args.token_record),
            '--original-hf-reference',str(args.original_hf_reference),'--framework-result',str(args.framework_result),
            '--stable-calibration-result',str(args.stable_calibration_result),
            '--count-result',str(args.count_result),'--tensor-result',str(args.tensor_result),
            '--decomposition-variant',args.decomposition_variant,'--newton-budget',str(args.newton_budget)]
        if args.fit_source is not None:
            forwarded.extend(['--fit-source',str(args.fit_source)])
        if args.saved_model_info is not None:
            forwarded.extend(['--model-info',str(args.saved_model_info)])
        return diagnostic_main(forwarded)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    work = args.out/'work'
    work.mkdir()
    model_info = args.saved_model_info or work/'import/model_info'
    hf = work/'hf-reference'
    mapped = work/'framework-reference'
    common = ['--model-info', str(model_info), '--local-model', str(args.local_model)]
    phases = []
    if args.saved_model_info is None:
        bootstrap = "import os,runpy,sys; [os.environ.__setitem__(n,'1') for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')]; import torch; torch.set_num_threads(1); torch.set_num_interop_threads(1); sys.argv=['pt_agent.experiments.r10_paper_worker']+sys.argv[1:]; runpy.run_module('pt_agent.experiments.r10_paper_worker',run_name='__main__')"
        phases.append(('import', ['-B', '-c', bootstrap, '--phase', 'qwen-import', '--work-dir',
            str(work/'import'), '--inputs', str(args.samples), '--model', str(args.local_model)],
            work/'import/worker_result.json'))
    phases.extend([
        ('hf-reference', ['-B', '-m', 'pt_agent.experiments.qwen_reference_worker', '--phase',
            'hf-reference', '--work-dir', str(hf), *common, '--samples', str(args.samples),
            '--source-rows', str(args.source_rows)], hf/'worker-result.json'),
        ('framework-reference', ['-B', '-m', 'pt_agent.experiments.qwen_reference_worker', '--phase',
            'framework-reference', '--work-dir', str(mapped), *common, '--reference',
            str(hf/'worker-result.json')], mapped/'worker-result.json')])
    environment = {key: value for key, value in os.environ.items()
        if not key.startswith(('LD_', 'NIX_', '__NIX_'))
        and key not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'IN_NIX_SHELL')}
    result = dict(status='running', stage=args.stage, model='Qwen2-0.5B',
        original_layers=24, phases={}, generated_tokens=0, ciphertext_generation_executed=False,
        api_calls=0, compiler_runs=0, keygen_runs=0, ckks_executions=0, candidate_frozen=False,
        compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None,
        requested_linear_variant=args.linear_variant,linear_variant_applied=False)
    try:
        for name, arguments, worker_path in phases:
            command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds',
                       str(args.phase_seconds), '--python', str(paths.INTAKE_PYTHON), '--', *arguments]
            write_json(args.out/(name+'-command.json'), command)
            observation = run_process(command, cwd=source, log=args.out/(name+'.log'),
                resources=args.out/(name+'-resources.jsonl'), timeout=args.phase_seconds+60,
                mem_gib=args.memory_gib, fsize_gib=8, env=environment)
            worker = json.loads(worker_path.read_text()) if worker_path.is_file() else None
            result['phases'][name] = dict(observation=observation, worker=worker)
            write_json(args.out/'result.json', result)
            if observation['exit_code'] != 0 or not worker or worker['status'] != 'ok':
                raise RuntimeError(name+' failed; no later stage was started')
        result['status'] = 'ok'
        if args.stage == 'full':
            result.update(status='incomplete', remaining_stages=[
                'Accept a feasible polynomial candidate against original-model heldout quality',
                'Bind two original full24-layer forwards to one backend and encrypted prefix state',
                'Fit and validate selection on real logit gaps and approximate embedding inputs',
                'Measure complete Count inventory and dense native constants before Emit',
                'Trace, compile, validate parameters and execute the encrypted full program'],
                full_native_trace_ready=False,
                reason='Current verified preparation ends before complete encrypted model binding')
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc))
    result['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True, text=True)
    git_rev = git.stdout.strip() if git.returncode == 0 else None
    result['git_rev'] = git_rev
    result['metrics'] = {name: result[name] for name in (
        'generated_tokens', 'ciphertext_generation_executed', 'api_calls', 'compiler_runs',
        'keygen_runs', 'ckks_executions', 'candidate_frozen', 'compiled_bootstraps',
        'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')}
    result['metrics']['status'] = result['status']
    write_json(args.out/'result.json', result)
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(run_id=args.out.name, round='autonomous-qwen-generation', model=result['model'],
               mode='server-' + args.stage, git_rev=git_rev, metrics=result['metrics'],
               artifacts_dir=str(args.out), ts=result['ended_utc'])
    descriptor = os.open(paths.RESULTS_DIR/'registry.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, (json.dumps(row) + '\n').encode())
    finally:
        os.close(descriptor)
    print(json.dumps(dict(out=str(args.out), status=result['status'],
                         ciphertext_generation_executed=False)))
    return 0 if result['status'] == 'ok' else 3 if result['status'] == 'incomplete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
