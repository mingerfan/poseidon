"""An explicit bounded cubic comparison candidate on saved training logits.

The original Remez selector and every library default remain unchanged.
Preparation writes the exact analytic coefficients for independent review;
evaluation reuses the original numpy_selection tournament once per saved row.
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from . import qwen_selector_validation as base


VARIANT = 'bounded-cubic-composite52-v1'
STAGES = 52


def config():
    return dict(variant=VARIANT, difference_bound=base.BOUND, normalized_gap=base.GAP,
        stage_count=STAGES, degrees=[3] * STAGES,
        polynomial='p(x)=(3*x-x**3)/2', chebyshev=[0., 1.125, 0., -.125],
        final_stage_half_sign_factor_already_applied=True,
        comparison_gap_scope='Observe all approximate-parent pair distances; the exact-max training minimum alone is not a coverage proof',
        one_hot_error_acceptance=None, greedy_agreement_pct=99.0)


def analytic_coefficients(np):
    stages = []
    for index in range(STAGES):
        value = np.asarray([0., 1.125, 0., -.125], dtype=np.float64)
        if index == STAGES - 1:
            value *= .5
        value.flags.writeable = False
        stages.append(value)
    return tuple(stages)


def prepare_artifact(sources, np):
    coefficients = analytic_coefficients(np)
    x = base.GAP
    scalar = []
    for index in range(STAGES):
        x = (3. * x - x * x * x) / 2.
        scalar.append(dict(compositions=index + 1, value=x, one_minus_value=1. - x))
    return dict(format=1, config=config(), source_rows=sources,
        stages=[value.tolist() for value in coefficients], coefficient_fits=0,
        coefficient_source='Exact analytic binary64 Chebyshev constants, with the final half factor applied once',
        exact_arithmetic_argument=dict(p_minus_x='x*(1-x*x)/2', derivative='3*(1-x*x)/2',
            endpoints={'p(-1)': -1, 'p(0)': 0, 'p(1)': 1},
            conclusion='On [-1,1], p is monotone, odd, sign preserving and maps the interval to itself; compositions preserve these properties. The final factor1/2 gives the half-sign range.',
            scope='Exact real arithmetic. Actual binary64 stage and parent ranges are observed separately.'),
        scalar_gap_observations=scalar, scalar_scope='Binary64 scalar observations, not a uniform error certificate')


def load_artifact(path, own_result, np):
    path = Path(path).resolve()
    artifact = json.loads(path.read_text())
    prepared = json.loads((path.parent / 'result.json').read_text())
    sources = base.source_rows(own_result)
    if (prepared.get('status') != 'ok' or prepared.get('phase') != 'prepare'
            or prepared.get('config') != config() or artifact.get('config') != config()
            or prepared.get('source_rows') != sources or artifact.get('source_rows') != sources
            or Path(prepared['own_output_result']).resolve() != Path(own_result).resolve()
            or artifact.get('format') != 1 or artifact.get('coefficient_fits') != 0):
        raise ValueError('The independently reviewed analytic coefficient/source contract differs')
    expected = analytic_coefficients(np)
    values = artifact.get('stages', [])
    if len(values) != STAGES:
        raise ValueError('Exactly52 unchanged analytic stages are required')
    for value, target in zip(values, expected):
        actual = np.asarray(value, dtype=np.float64)
        if actual.shape != (4,) or not np.array_equal(actual, target):
            raise ValueError('An analytic stage or the final half factor differs')
    return expected, artifact, prepared


@contextmanager
def observe_rounds(np, vocabulary, rows, checkpoint):
    original = np.polynomial.chebyshev.chebval
    width, calls = vocabulary, 0
    last_write = time.monotonic()

    def observed(value, coefficients, *args, **kwargs):
        nonlocal width, calls, last_write
        stage = calls % STAGES
        data = np.asarray(value)
        parents = (width + 1) // 2
        if data.dtype != np.float64 or data.shape != (parents,) or not np.isfinite(data).all():
            raise ValueError('A complete finite binary64 original selector stage is required')
        valid = data[:width // 2]
        if stage == 0:
            rows.append(dict(round=len(rows), width=width, parents=parents, actual_pairs=width // 2,
                odd_carry_excluded=bool(width % 2), min_difference=float(valid.min()) * base.BOUND,
                max_difference=float(valid.max()) * base.BOUND,
                min_normalized_absolute_difference=float(np.abs(valid).min()),
                max_normalized_absolute_difference=float(np.abs(valid).max()),
                comparisons_below_gap=int((np.abs(valid) < base.GAP).sum()),
                comparisons_above_bound=int((np.abs(valid) > 1.).sum()), stages=[]))
            checkpoint()
        row = dict(stage=stage, actual_pairs=width // 2,
            input_min=float(valid.min()), input_max=float(valid.max()),
            inputs_outside_exact_interval=int((np.abs(valid) > 1.).sum()))
        rows[-1]['stages'].append(row)
        output = original(value, coefficients, *args, **kwargs)
        if not np.isfinite(output).all():
            raise ValueError('A cubic stage produced nonfinite output')
        interval = .5 if stage == STAGES - 1 else 1.
        row.update(output_min=float(output.min()), output_max=float(output.max()), all_finite=True,
            exact_output_absolute_bound=interval,
            outputs_outside_exact_interval=int((np.abs(output[:width // 2]) > interval).sum()))
        calls += 1
        if calls % STAGES == 0:
            width = parents
            checkpoint()
        elif time.monotonic() - last_write >= 10:
            checkpoint()
            last_write = time.monotonic()
        return output

    np.polynomial.chebyshev.chebval = observed
    try:
        yield
        if width != 1 or calls != STAGES * len(rows):
            raise ValueError('The complete original tournament was not observed')
    finally:
        np.polynomial.chebyshev.chebval = original


def worker(args):
    np = base._numeric()
    out = args.out.resolve()
    result = dict(status='running', phase=args.phase, variant=VARIANT, config=config(),
        own_output_result=str(args.own_result.resolve()), started_ns=time.perf_counter_ns(),
        budget_started_ns=args.budget_started_ns, selection_attempts=0, selection_completed=0,
        fit_stages=[], analytic_coefficient_constructions=0, resource_guard={}, rows=[],
        numerical_quality_scope='Three raw unlabelled SST-2 training sentences, two saved causal prefixes each; standard LM and SST-2 classification performance unmeasured',
        timing_scope='Bounded training diagnostic concurrent with R31 native work; not isolated inference latency',
        **{key: False for key in base.FALSE_FIELDS}, **{key: 0 for key in base.ZERO_FIELDS},
        coefficient_fits=0, compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None)
    checkpoint = lambda: base.save(out / 'result.json', result)
    try:
        with base.resident_guard(args.supervisor_pid, result['resource_guard']):
            sources = base.source_rows(args.own_result)
            result['source_rows'] = sources
            result['source_input_observations'] = []
            for source in sources:
                values = base.read_logits(source, np)
                if np.abs(values).max() > 16.:
                    raise ValueError('The fixed source-derived logit envelope16 is exceeded')
                result['source_input_observations'].append(dict(sample_id=source['sample_id'], step=source['step'],
                    minimum=float(values.min()), maximum=float(values.max()), argmax=int(values.argmax()), all_finite=True))
            checkpoint()
            if args.phase == 'prepare':
                artifact = prepare_artifact(sources, np)
                base.save(out / 'coefficients.json', artifact)
                result.update(status='ok', analytic_coefficient_constructions=1,
                    coefficients=str(out / 'coefficients.json'), independent_review_required_before_evaluation=True)
            else:
                coefficients, artifact, prepared = load_artifact(args.coefficients, args.own_result, np)
                result.update(coefficient_source=str(args.coefficients.resolve()), prepared_source=str(args.coefficients.parent / 'result.json'))
                from ..kernels.encrypted_generation import numpy_selection
                for index, source in enumerate(sources):
                    values = base.read_logits(source, np)
                    row = dict(sample_id=source['sample_id'], step=source['step'], rounds=[], completed=False)
                    result['rows'].append(row)
                    result['selection_attempts'] += 1
                    checkpoint()
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        with observe_rounds(np, len(values), row['rounds'], checkpoint):
                            selected, contract = numpy_selection(values, bound=base.BOUND, normalized_gap=base.GAP,
                                                                 coefficients=coefficients)
                        if selected.dtype != np.float64 or selected.shape != values.shape or not np.isfinite(selected).all():
                            raise ValueError('The actual selected vector is nonfinite or changed shape/dtype')
                        row['one_hot_file'] = 'selection-' + str(index) + '.npy'
                        np.save(out / row['one_hot_file'], selected, allow_pickle=False)
                        row.update(contract=contract, quality=base.one_hot_quality(values, selected, np))
                    row.update(completed=True, real_comparisons_observed=sum(item['actual_pairs'] for item in row['rounds']),
                        actual_parent_domain_passed=all(not item['comparisons_below_gap'] and not item['comparisons_above_bound'] for item in row['rounds']),
                        all_stages_finite=all(item.get('all_finite') is True for parent in row['rounds'] for item in parent['stages']))
                    result['selection_completed'] += 1
                    checkpoint()
                agreement = 100. * sum(row['quality']['token_agreement'] for row in result['rows']) / 6
                result.update(status='ok', evaluation_completed=True, greedy_agreement_pct=agreement,
                    greedy_policy_passed=agreement >= 99.,
                    all_comparison_contracts_passed=all(row['contract']['comparison_gap_contract_passed'] for row in result['rows']),
                    selector_one_hot_quality_accepted=None, quality_status='training_observation_completed')
    except BaseException as exc:
        result.update(status='error', error=dict(type=type(exc).__name__, message=str(exc)))
    finally:
        result.update(ended_ns=time.perf_counter_ns(), worker_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result['worker_seconds'] = (result['ended_ns'] - result['started_ns']) / 1e9
        checkpoint()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'evaluate'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--own-result', type=Path, required=True)
    parser.add_argument('--coefficients', type=Path)
    parser.add_argument('--inner', action='store_true')
    parser.add_argument('--supervisor-pid', type=int)
    parser.add_argument('--budget-started-ns', type=int)
    args = parser.parse_args(argv)
    if args.phase == 'evaluate' and args.coefficients is None:
        parser.error('Evaluation requires the independently reviewed analytic artifact')
    if args.inner:
        if args.supervisor_pid is None or args.budget_started_ns is None:
            parser.error('The explicit supervision context is required')
        result = worker(args)
        print(json.dumps(dict(status=result['status'], phase=args.phase, selection_completed=result['selection_completed'])))
        return 0 if result['status'] == 'ok' else 1
    from .r10_paper_metrics import run_process
    from .. import paths
    args.out = base.check_out(args.out, args.own_result, args.coefficients)
    args.out.mkdir(parents=True, exist_ok=False)
    budget_start = time.perf_counter_ns()
    if args.phase == 'evaluate':
        budget_start = json.loads((args.coefficients.parent / 'result.json').read_text())['budget_started_ns']
    deadline = budget_start + base.TOTAL_SECONDS * 10 ** 9
    remaining = int((deadline - time.perf_counter_ns()) / 10 ** 9)
    if remaining < 1:
        base.save(args.out / 'supervision.json', dict(status='budget_exhausted', total_seconds=base.TOTAL_SECONDS))
        return 1
    command = [str(paths.ORBIT_ROOT.parent / 'orbit-venv/bin/python'), '-B', '-m', __spec__.name,
        '--phase', args.phase, '--out', str(args.out), '--own-result', str(args.own_result), '--inner',
        '--supervisor-pid', str(os.getpid()), '--budget-started-ns', str(budget_start)]
    if args.coefficients is not None:
        command.extend(['--coefficients', str(args.coefficients)])
    base.save(args.out / 'command.json', command)
    observation = run_process(command, cwd=Path(__file__).resolve().parents[2], log=args.out / 'worker.log',
        resources=args.out / 'resources.jsonl', timeout=remaining, mem_gib=1, fsize_gib=.1,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                 MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'))
    base.save(args.out / 'supervision.json', dict(observation=observation, total_seconds=base.TOTAL_SECONDS,
        shared_prepare_evaluation_deadline_ns=deadline, remaining_at_start_seconds=remaining))
    base.record_run(args.out, args.phase, paths.RESULTS_DIR / 'registry.jsonl', variant=VARIANT)
    return 0 if observation['exit_code'] == 0 and not observation['timed_out'] and not observation['remaining_pids'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
