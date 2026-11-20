"""Bounded real-training-logit diagnostics for the existing polynomial selector.

Fit and evaluation are separate commands so the actual coefficient artifact
can be reviewed before any full-vocabulary selector executes.  This module
does not load a model, run packed arithmetic, or perform cryptography.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time


BOUND = 32.0
GAP = 2.0 ** -27
DEGREES = (15, 15, 27)
VOCABULARY = 151936
TOTAL_SECONDS = 900
RSS_BYTES = 512 * 1024 ** 2
FALSE_FIELDS = ('candidate_frozen', 'encrypted_generation_verified',
                'full_model_binding_verified', 'selector_embedding_quality_verified')
ZERO_FIELDS = ('hf_forwards', 'tensor_forwards', 'mock_forwards', 'count_runs',
               'api_calls', 'keygen_runs', 'ckks_executions', 'heldout_samples',
               'confirm_samples', 'report_samples', 'new_installations', 'new_downloads')


def _numeric():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    return np


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return dict(nonfinite=repr(value))
    return value


def save(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(_safe(record), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fixed_config():
    return dict(difference_bound=BOUND, normalized_gap=GAP, degrees=list(DEGREES),
        logit_absolute_envelope=16.0, gap_source='Prior exact-max training tournament minimum; approximate-parent coverage is measured separately',
        arithmetic_criterion='1e-12+1e-12*abs(reference)',
        one_hot_error_acceptance=None, greedy_agreement_pct=99.0)


def check_out(out, own_result, coefficients=None):
    out = Path(out).resolve()
    protected = [Path(own_result).resolve().parent]
    if coefficients is not None:
        protected.append(Path(coefficients).resolve().parent)
    if any(out == path or path in out.parents for path in protected):
        raise ValueError('New diagnostics must preserve the original source and fit directories')
    return out


def source_rows(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    worker = record.get('worker') or {}
    if (record.get('status') != 'ok' or record.get('job') != 'quality-own-tensor'
            or worker.get('status') != 'ok' or worker.get('own_output_generation_completed') is not True
            or worker.get('tensor_forward_completed') != 6 or worker.get('state_update_completed') != 3
            or worker.get('candidate_frozen') is not False):
        raise ValueError('The original accepted, unfrozen own-output training source is required')
    rows = worker.get('sample_rows', [])
    expected = [('train:' + str(index // 2), index % 2) for index in range(6)]
    if [(row.get('sample_id'), row.get('step')) for row in rows] != expected:
        raise ValueError('Exactly the original six training-prefix rows are required')
    selected = []
    for index, row in enumerate(rows):
        filename = 'own-output-' + str(index) + '-logits.npy'
        if row.get('logits_file') != filename:
            raise ValueError('The original local training-logit filename differs')
        data = path.parent / 'worker' / filename
        if not data.is_file() or not data.stat().st_size:
            raise ValueError('A saved original training-logit array is absent')
        selected.append(dict(sample_id=row['sample_id'], step=row['step'], path=str(data),
            size_bytes=data.stat().st_size, original_hf_token=row['original_greedy_token'],
            own_output_token=row['approximate_greedy_token']))
    return selected


def read_logits(row, np):
    value = np.load(row['path'], allow_pickle=False, mmap_mode='r')
    if (value.dtype != np.float64 or value.shape != (1, 1, VOCABULARY)
            or not np.isfinite(value).all()):
        raise ValueError('Original finite float64 full-vocabulary logits are required')
    if int(value.argmax()) != row['own_output_token']:
        raise ValueError('The actual saved greedy token differs from its source record')
    return value.reshape(-1)


def _fit_modules():
    from ..kernels import approx, encrypted_generation
    return approx, encrypted_generation


def fit_once(np, result, checkpoint):
    approx, selector = _fit_modules()
    original = approx._exchange
    stages = result.setdefault('fit_stages', [])

    def observed(*args, **kwargs):
        index = len(stages)
        if index >= len(DEGREES):
            raise RuntimeError('The fixed three-stage fit budget was exceeded')
        row = dict(index=index, degree=args[1], domain=list(kwargs['interval']),
                   attempted=1, completed=0, started_ns=time.perf_counter_ns())
        stages.append(row)
        checkpoint()
        try:
            coefficients, report = original(*args, **kwargs)
            row.update(completed=1, report=report)
            return coefficients, report
        except Exception as exc:
            row['error'] = dict(type=type(exc).__name__, message=str(exc),
                iterations=getattr(exc, 'iterations', None), max_error=getattr(exc, 'max_error', None),
                error_amplitude=getattr(exc, 'error_amplitude', None), history=getattr(exc, 'history', None))
            raise
        finally:
            row['ended_ns'] = time.perf_counter_ns()
            checkpoint()

    approx._exchange = observed
    try:
        result['composite_fit_calls'] = 1
        coefficients = selector.composite_half_sign(DEGREES, GAP)
        reports = approx.composite_report(DEGREES, GAP)
    finally:
        approx._exchange = original
    if len(stages) != 3 or any(stage['completed'] != 1 for stage in stages):
        raise ValueError('Exactly one completed original fit per stage is required')
    if len(coefficients) != 3 or len(reports) != 3:
        raise ValueError('The original three fitted coefficient stages are required')
    for degree, stage in zip(DEGREES, coefficients):
        if (stage.dtype != np.float64 or stage.shape != (degree + 1,) or not np.isfinite(stage).all()
                or np.any(stage[::2] != 0) or not np.any(stage[1::2] != 0)):
            raise ValueError('The original finite odd Chebyshev stage contract differs')
    return dict(format=1, config=fixed_config(), stages=[stage.tolist() for stage in coefficients],
        reports=reports, fit_stage_attempts=3, fit_stage_completed=3,
        final_stage_half_sign_factor_already_applied=True,
        coefficient_source='Original composite_half_sign/composite_coeffs, original deterministic odd Remez')


def load_coefficients(path, own_result, np):
    path = Path(path).resolve()
    result = json.loads((path.parent / 'result.json').read_text())
    artifact = json.loads(path.read_text())
    if (result.get('status') != 'ok' or result.get('phase') != 'fit'
            or Path(result['own_output_result']).resolve() != Path(own_result).resolve()
            or result.get('config') != fixed_config() or artifact.get('config') != fixed_config()
            or artifact.get('format') != 1 or artifact.get('fit_stage_attempts') != 3
            or artifact.get('fit_stage_completed') != 3
            or artifact.get('final_stage_half_sign_factor_already_applied') is not True):
        raise ValueError('The reviewed fixed real-fit source and policy must remain unchanged')
    reports = artifact.get('reports', [])
    stages = artifact.get('stages', [])
    if len(stages) != 3 or len(reports) != 3:
        raise ValueError('The saved original three-stage fit is incomplete')
    expected = (GAP, 1.0)
    coefficients = []
    for degree, stage, report in zip(DEGREES, stages, reports):
        value = np.asarray(stage, dtype=np.float64)
        if (value.shape != (degree + 1,) or not np.isfinite(value).all()
                or np.any(value[::2] != 0) or not np.any(value[1::2] != 0)
                or report.get('degree') != degree or tuple(report.get('domain', [])) != expected
                or type(report.get('iterations')) is not int or not 1 <= report['iterations'] <= 50
                or not math.isfinite(report.get('max_error', float('nan')))):
            raise ValueError('A coefficient or original fit-domain report is invalid')
        value.flags.writeable = False
        coefficients.append(value)
        expected = (1.0 - report['max_error'], 1.0 + report['max_error'])
    return tuple(coefficients), artifact, result


@contextmanager
def observe_rounds(np, vocabulary, reports, rows, checkpoint):
    """Observe original chebval inputs/outputs while leaving its arithmetic intact."""
    original = np.polynomial.chebyshev.chebval
    width, calls = vocabulary, 0

    def observed(value, coefficients, *args, **kwargs):
        nonlocal width, calls
        stage = calls % len(DEGREES)
        data = np.asarray(value)
        parents = (width + 1) // 2
        if data.dtype != np.float64 or data.shape != (parents,):
            raise ValueError('The existing selector stage shape or dtype differs')
        valid = data[:width // 2]
        if not np.isfinite(data).all():
            raise ValueError('An approximate parent comparison became nonfinite')
        if stage == 0:
            rows.append(dict(round=len(rows), width=width, parents=parents, actual_pairs=width // 2,
                odd_carry_excluded=bool(width % 2), min_difference=float(valid.min()) * BOUND,
                max_difference=float(valid.max()) * BOUND,
                min_normalized_absolute_difference=float(np.abs(valid).min()),
                max_normalized_absolute_difference=float(np.abs(valid).max()),
                comparisons_below_gap=int((np.abs(valid) < GAP).sum()),
                comparisons_above_bound=int((np.abs(valid) > 1.0).sum()), stages=[]))
        lo, hi = reports[stage]['domain']
        observation = dict(stage=stage, input_min=float(valid.min()), input_max=float(valid.max()),
            expected_absolute_domain=[lo, hi],
            actual_pairs_outside_stage_domain=int(((np.abs(valid) < lo) | (np.abs(valid) > hi)).sum()))
        rows[-1]['stages'].append(observation)
        checkpoint()
        output = original(value, coefficients, *args, **kwargs)
        if not np.isfinite(output).all():
            raise ValueError('A selector polynomial output became nonfinite')
        observation.update(output_finite=True, output_min=float(output.min()), output_max=float(output.max()))
        calls += 1
        if calls % len(DEGREES) == 0:
            width = parents
        checkpoint()
        return output

    np.polynomial.chebyshev.chebval = observed
    try:
        yield
        if width != 1 or calls != 3 * len(rows):
            raise ValueError('The complete original tournament schedule was not observed')
    finally:
        np.polynomial.chebyshev.chebval = original


def one_hot_quality(logits, selected, np):
    if selected.dtype != np.float64 or selected.shape != logits.shape or not np.isfinite(selected).all():
        raise ValueError('The actual approximate selection must be a finite full-vocabulary vector')
    token = int(logits.argmax())
    discrete = np.zeros_like(selected)
    discrete[token] = 1.0
    error = np.abs(selected - discrete)
    return dict(actual_selected_token=int(selected.argmax()), original_greedy_token=token,
        token_agreement=bool(selected.argmax() == token), max_absolute_one_hot_error=float(error.max()),
        l1_one_hot_error=float(error.sum()), sum=float(selected.sum()), minimum=float(selected.min()),
        maximum=float(selected.max()), greedy_weight=float(selected[token]),
        exact_discrete_one_hot_claimed=False, one_hot_quality_acceptance=None)


@contextmanager
def resident_guard(supervisor_pid, record):
    from .qwen_own_output_packed import _resident_snapshot
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise ValueError('The dedicated selector diagnostic requires a free interval timer')
    record.update(limit_bytes=RSS_BYTES, peak_observed_bytes=0, stopped=False)

    def check(*_):
        observed = _resident_snapshot(supervisor_pid) * 1024
        record['peak_observed_bytes'] = max(record['peak_observed_bytes'], observed)
        if observed > RSS_BYTES:
            record['stopped'] = True
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise MemoryError('The selector supervisor subtree exceeded the fixed 0.5GiB RSS cap')

    signal.signal(signal.SIGALRM, check)
    try:
        check()
        signal.setitimer(signal.ITIMER_REAL, .1, .1)
        yield
        check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def worker(args):
    np = _numeric()
    out = args.out.resolve()
    result = dict(status='running', phase=args.phase, own_output_result=str(args.own_result.resolve()),
        config=fixed_config(), started_ns=time.perf_counter_ns(), budget_started_ns=args.budget_started_ns,
        fit_stages=[], selection_attempts=0,
        selection_completed=0, rows=[], resource_guard={},
        timing_scope='Training diagnostic overlapping the separately scheduled R31 native workload; not an isolated latency benchmark',
        **{key: False for key in FALSE_FIELDS}, **{key: 0 for key in ZERO_FIELDS},
        compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None)
    checkpoint = lambda: save(out / 'result.json', result)
    try:
        with resident_guard(args.supervisor_pid, result['resource_guard']):
            sources = source_rows(args.own_result)
            result['source_rows'] = sources
            result['source_input_observations'] = []
            for row in sources:
                values = read_logits(row, np)
                result['source_input_observations'].append(dict(sample_id=row['sample_id'], step=row['step'],
                    minimum=float(values.min()), maximum=float(values.max()), all_finite=True,
                    argmax=int(values.argmax()), original_logit_envelope_contained=bool(np.abs(values).max() <= 16.0)))
                if np.abs(values).max() > 16.0:
                    raise ValueError('The fixed source-derived original logit envelope is exceeded')
            checkpoint()
            if args.phase == 'fit':
                artifact = fit_once(np, result, checkpoint)
                artifact['source_rows'] = sources
                save(out / 'coefficients.json', artifact)
                result.update(status='ok', coefficients=str(out / 'coefficients.json'), fit_completed=True,
                    approximation_quality_measured=False, independent_review_required_before_evaluation=True)
            else:
                coefficients, artifact, fit = load_coefficients(args.coefficients, args.own_result, np)
                if fit['source_rows'] != sources:
                    raise ValueError('The fit and evaluation source identities differ')
                result.update(fit_source=str(args.coefficients.resolve()), fresh_fit_calls=0,
                    original_fit_started_ns=fit['started_ns'])
                from ..kernels.encrypted_generation import numpy_selection
                for index, row in enumerate(sources):
                    values = read_logits(row, np)
                    measured = dict(sample_id=row['sample_id'], step=row['step'], rounds=[], completed=False)
                    result['rows'].append(measured)
                    result['selection_attempts'] += 1
                    checkpoint()
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        with observe_rounds(np, len(values), artifact['reports'], measured['rounds'], checkpoint):
                            selected, contract = numpy_selection(values, bound=BOUND, normalized_gap=GAP,
                                                                 coefficients=coefficients)
                    if selected.dtype != np.float64 or selected.shape != values.shape or not np.isfinite(selected).all():
                        raise ValueError('The actual selector output is nonfinite or changed shape/dtype')
                    measured['one_hot_file'] = 'selection-' + str(index) + '.npy'
                    np.save(out / measured['one_hot_file'], selected, allow_pickle=False)
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        measured.update(contract=contract, quality=one_hot_quality(values, selected, np))
                    measured['actual_parent_domain_passed'] = bool(all(
                        not item['comparisons_below_gap'] and not item['comparisons_above_bound']
                        for item in measured['rounds']))
                    measured['completed'] = True
                    result['selection_completed'] += 1
                    checkpoint()
                result.update(status='ok', evaluation_completed=True,
                    greedy_agreement_pct=100.0 * sum(row['quality']['token_agreement'] for row in result['rows']) / 6,
                    all_comparison_contracts_passed=all(row['contract']['comparison_gap_contract_passed'] for row in result['rows']),
                    quality_status='measured_training_only', selector_one_hot_quality_accepted=None)
                result['greedy_policy_passed'] = result['greedy_agreement_pct'] >= 99.0
    except BaseException as exc:
        result.update(status='error', error=dict(type=type(exc).__name__, message=str(exc)))
    finally:
        result.update(ended_ns=time.perf_counter_ns(), worker_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result['worker_seconds'] = (result['ended_ns'] - result['started_ns']) / 1e9
        checkpoint()
    return result


def record_run(out, phase, registry, *, variant='remez151527', revision=None, registration='direct', extra_metrics=None, run_id=None):
    """Register one observed phase, including retained failures, exactly once."""
    out, registry = Path(out).resolve(), Path(registry)
    result = json.loads((out / 'result.json').read_text()) if (out / 'result.json').is_file() else {}
    supervision = json.loads((out / 'supervision.json').read_text())
    observation = supervision.get('observation') or {}
    status = 'ok' if (result.get('status') == 'ok' and observation.get('exit_code') == 0
        and observation.get('timed_out') is False and observation.get('remaining_pids') == []) else 'error'
    metrics = {key: result.get(key) for key in ZERO_FIELDS}
    metrics.update(status=status, phase=phase, variant=variant, registration=registration,
        fit_stage_attempts=len(result['fit_stages']) if 'fit_stages' in result else None,
        fit_stage_completed=sum(row['completed'] for row in result['fit_stages']) if 'fit_stages' in result else None,
        selection_attempts=result.get('selection_attempts'), selection_completed=result.get('selection_completed'),
        greedy_agreement_pct=result.get('greedy_agreement_pct'), greedy_policy_passed=result.get('greedy_policy_passed'),
        all_comparison_contracts_passed=result.get('all_comparison_contracts_passed'), error=result.get('error'),
        compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None)
    metrics.update(extra_metrics or {})
    if revision is None:
        from ..registry import git_rev
        revision = git_rev()
    row = dict(run_id=run_id or out.parent.name + '-' + phase, round='autonomous-qwen-selector', model='Qwen2-0.5B',
        mode='real-logit-selector-' + variant + '-' + phase, git_rev=revision, metrics=metrics,
        artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    with registry.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        existing = [json.loads(line) for line in stream if line.strip()]
        matches = [item for item in existing if item.get('run_id') == row['run_id'] and item.get('mode') == row['mode']]
        if matches:
            if len(matches) != 1 or matches[0]['metrics'] != metrics or matches[0]['artifacts_dir'] != str(out):
                raise ValueError('A different selector phase registry record already exists')
            return matches[0]
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('fit', 'evaluate'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--own-result', type=Path, required=True)
    parser.add_argument('--coefficients', type=Path)
    parser.add_argument('--inner', action='store_true')
    parser.add_argument('--supervisor-pid', type=int)
    parser.add_argument('--budget-started-ns', type=int)
    args = parser.parse_args(argv)
    if args.phase == 'evaluate' and args.coefficients is None:
        parser.error('Evaluation requires the independently reviewed coefficient artifact')
    if args.inner:
        if args.supervisor_pid is None:
            parser.error('The explicit supervisor PID is required')
        if args.budget_started_ns is None:
            parser.error('The explicit shared diagnostic clock is required')
        result = worker(args)
        print(json.dumps(dict(status=result['status'], phase=args.phase,
                              fit_stage_attempts=len(result['fit_stages']), selection_completed=result['selection_completed'])))
        return 0 if result['status'] == 'ok' else 1
    from .r10_paper_metrics import run_process
    from .. import paths
    args.out = check_out(args.out, args.own_result, args.coefficients)
    args.out.mkdir(parents=True, exist_ok=False)
    budget_start = time.perf_counter_ns()
    deadline = budget_start + TOTAL_SECONDS * 10 ** 9
    if args.phase == 'evaluate':
        fit = json.loads((args.coefficients.parent / 'result.json').read_text())
        budget_start = fit['budget_started_ns']
        deadline = min(deadline, budget_start + TOTAL_SECONDS * 10 ** 9)
    remaining = int((deadline - time.perf_counter_ns()) / 10 ** 9)
    if remaining < 1:
        save(args.out / 'supervision.json', dict(status='budget_exhausted', total_seconds=TOTAL_SECONDS))
        return 1
    command = [str(paths.ORBIT_ROOT.parent / 'orbit-venv/bin/python'), '-B', '-m', __spec__.name,
        '--phase', args.phase, '--out', str(args.out), '--own-result', str(args.own_result),
        '--inner', '--supervisor-pid', str(os.getpid()), '--budget-started-ns', str(budget_start)]
    if args.coefficients is not None:
        command.extend(['--coefficients', str(args.coefficients)])
    save(args.out / 'command.json', command)
    observation = run_process(command, cwd=Path(__file__).resolve().parents[2], log=args.out / 'worker.log',
        resources=args.out / 'resources.jsonl', timeout=remaining, mem_gib=1, fsize_gib=.1,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                 MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'))
    save(args.out / 'supervision.json', dict(observation=observation, total_seconds=TOTAL_SECONDS,
        shared_fit_evaluation_deadline_ns=deadline, remaining_at_start_seconds=remaining))
    record_run(args.out, args.phase, paths.RESULTS_DIR / 'registry.jsonl')
    return 0 if observation['exit_code'] == 0 and not observation['timed_out'] and not observation['remaining_pids'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
