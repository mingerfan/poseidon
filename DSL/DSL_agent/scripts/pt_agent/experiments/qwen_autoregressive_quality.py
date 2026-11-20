"""Training-only Tensor generation using each approximate output as feedback.

The original HF own-token trajectories remain independent references.  Only
their first input initializes generation; later inputs come from this run.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import resource

from . import qwen_precision_quality as fixed
from .r10_paper_metrics import StageRecorder, write_json


def advance_state(state, next_token, *, vocabulary_size):
    """Append one actual output token to a copied right-padded prefix state."""
    ids = state['prefix_token_ids']
    length = state['prefix_length']
    capacity = len(ids)
    mask, selection = state['attention_mask'], state['sel']
    if (type(length) is not int or not 1 <= length < capacity
            or type(state['step']) is not int or state['step'] < 0
            or type(vocabulary_size) is not int or vocabulary_size < 1
            or type(next_token) is not int or not 0 <= next_token < vocabulary_size
            or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids)
            or mask != [[1.] * length + [0.] * (capacity-length)]
            or selection != [[0.] * (length-1) + [1.] + [0.] * (capacity-length)]):
        raise ValueError('Generation requires a finite-capacity right-padded original token state')
    updated = copy.deepcopy(state)
    updated['prefix_token_ids'][length] = next_token
    updated['prefix_length'] = length+1
    updated['attention_mask'][0][length] = 1.
    updated['sel'][0][length-1] = 0.
    updated['sel'][0][length] = 1.
    updated['step'] = state['step']+1
    return updated


def _run(*args, **kwargs):
    from ..transformer_tensor import run
    return run(*args, **kwargs)


def _polynomial(value, attrs):
    from ..transformer_tensor import polynomial
    return polynomial(value, attrs, False)


def _measure(args, info, program, current, identity, recorder, result, np):
    arrays = fixed.arrays_for(info, current, np)
    point_start = len(result['point_rows'])
    softmax_start = len(result['softmax_rows'])
    expected_points = sum(1 if op.kind in ('act', 'rmsnorm', 'layernorm') else
        2+op.attrs.get('steps', 0) if op.kind == 'softmax' else 0 for op in program.ops)
    softmax_count = sum(op.kind == 'softmax' for op in program.ops)
    local_softmax = {}

    def point(op, role, value, attrs, mask):
        row = fixed.domain_record(value, attrs, np, mask)
        row.update(identity, op_id=op.id, line=op.lineno, role=role)
        result['point_rows'].append(row)
        result['active_point'] = dict(row, completed=False)
        observed = _polynomial(value, attrs)
        row['polynomial_output_finite'] = bool(np.isfinite(observed).all())
        result['active_point'].update(completed=True,
            polynomial_output_finite=row['polynomial_output_finite'])
        if not row['polynomial_output_finite']:
            raise ValueError('A fixed approximation produced nonfinite values')
        if op.kind == 'softmax' and role == 'exp':
            transformed = np.asarray(value)*(2.**op.attrs.get('steps', 0))
            selected = np.ones_like(transformed, dtype=bool) if mask is None else mask
            if not selected.any(axis=-1).all():
                raise ValueError('Every attention row requires an effective key')
            hidden = np.where(selected, transformed, -np.inf)
            exponentials = np.exp(hidden-hidden.max(axis=-1, keepdims=True))
            local_softmax[op.id] = exponentials/exponentials.sum(axis=-1, keepdims=True)
        return observed

    def step(op, value):
        data = np.asarray(value)
        finite = bool(np.isfinite(data).all())
        row = dict(identity, op_id=op.id, line=op.lineno, kind=op.kind,
            shape=list(data.shape), finite=finite,
            minimum=float(data.min()) if finite else None,
            maximum=float(data.max()) if finite else None)
        result['op_rows'].append(row)
        result['active_op'] = row
        if op.kind == 'softmax':
            reference = local_softmax.pop(op.id)
            result['softmax_rows'].append(dict(identity, op_id=op.id, line=op.lineno,
                finite=finite, max_absolute_difference=float(np.abs(data-reference).max()) if finite else None,
                maximum_sum_error=float(np.abs(data.sum(-1)-1.).max()) if finite else None,
                minimum_probability=float(data.min()) if finite else None,
                reference='Independent stable exact softmax on the same public affine-transformed polynomial input'))
        if len(result['op_rows']) % 64 == 0 or not finite:
            write_json(args.work_dir/'worker-result.json', result)
        if not finite:
            raise ValueError('The fixed approximate trajectory produced nonfinite values')

    with recorder.span('own_output_tensor_forward', **identity):
        result['tensor_forward_attempts'] += 1
        result['active_input'] = dict(identity, prefix_length=current['prefix_length'],
            prefix_token_ids=list(current['prefix_token_ids']),
            attention_mask=copy.deepcopy(current['attention_mask']), sel=copy.deepcopy(current['sel']),
            input_shapes={key:list(value.shape) for key, value in arrays.items()})
        write_json(args.work_dir/'worker-result.json', result)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            observed = _run(program, info.graph.weights, arrays,
                point_evaluator=point, step_observer=step)
        if (len(result['point_rows'])-point_start != expected_points
                or len(result['softmax_rows'])-softmax_start != softmax_count
                or local_softmax):
            raise ValueError('Every selected nonlinear role must be observed once per trajectory state')
    return observed, result['point_rows'][point_start:], result['softmax_rows'][softmax_start:]


def run_validation(args, recorder, result, np):
    from ..metrics import compare, passes
    info, program, hf = fixed._load(args, result)
    result.update(sample_rows=[], trajectory_rows=[], point_rows=[], softmax_rows=[], op_rows=[])
    for trajectory in hf['trajectories']:
        initial = trajectory['steps'][0]
        if (initial['step'] != 0 or len(initial['prefix_token_ids']) != 32
                or not 1 <= initial['prefix_length'] <= 30
                or [row['step'] for row in trajectory['steps']] != [0, 1]):
            raise ValueError('Preserved training prompts must leave two positions in the original capacity32')
        current = copy.deepcopy(initial)
        generated, trajectory_matches = [], []
        for index in range(2):
            reference_row = trajectory['steps'][index]
            identity = dict(sample_id=trajectory['sample_id'], step=index)
            same_prefix = all(current[key] == reference_row[key] for key in
                ('prefix_token_ids', 'prefix_length', 'attention_mask', 'sel'))
            observed, points, softmax = _measure(args, info, program, current, identity, recorder, result, np)
            reference = np.load(args.reference.parent/reference_row['logits_file'],
                allow_pickle=False, mmap_mode='r')
            if (observed.dtype != np.float64 or observed.shape != reference.shape
                    or observed.shape != (1, 1, 151936) or not np.isfinite(observed).all()):
                raise ValueError('Own-output logits must retain the original finite float64 vocabulary shape')
            measured = compare(dict(values=reference, task=info.task,
                lengths=[current['prefix_length']], selected_last=True), observed, None)
            token = int(observed.argmax())
            generated.append(token)
            trajectory_matches.append(same_prefix)
            filename = 'own-output-'+str(len(result['sample_rows']))+'-logits.npy'
            np.save(args.work_dir/filename, observed, allow_pickle=False)
            result['sample_rows'].append(dict(identity, logits_file=filename, metrics=measured,
                policy_passed=passes(measured, info.threshold), input_state=copy.deepcopy(result['active_input']),
                original_hf_prefix_matched=same_prefix, original_greedy_token=int(reference.argmax()),
                approximate_greedy_token=token, max_absolute_difference=float(np.abs(observed-reference).max()),
                domain_points=len(points), all_domains_contained=all(row['contained'] for row in points),
                softmax_checks=len(softmax), all_finite=True,
                input_source='original first training prefix' if index == 0 else 'this run approximate output feedback'))
            result['tensor_forward_completed'] += 1
            result['generated_token_count'] += 1
            write_json(args.work_dir/'worker-result.json', result)
            if index == 0:
                with recorder.span('approximate_token_state_update', **identity):
                    result['state_update_attempts'] += 1
                    current = advance_state(current, token, vocabulary_size=observed.shape[-1])
                    result['state_update_completed'] += 1
                    result['approximate_tokens_fed_back'] = True
                    write_json(args.work_dir/'worker-result.json', result)
        result['trajectory_rows'].append(dict(sample_id=trajectory['sample_id'],
            generated_tokens=generated, original_hf_prefixes_matched=all(trajectory_matches),
            forward_completed=2, approximate_output_updates=1))
    if (result['tensor_forward_attempts'] != 6 or result['tensor_forward_completed'] != 6
            or result['state_update_attempts'] != 3 or result['state_update_completed'] != 3):
        raise ValueError('Exactly three original training trajectories require two forwards and one own-output update each')
    result.update(status='ok', own_output_generation_completed=True,
        training_quality_passed=all(row['policy_passed'] and row['all_domains_contained']
            for row in result['sample_rows']), all_finite=True, approximate_tokens_fed_back=True,
        generated_token_count=6, original_hf_trajectory_matched=all(
            row['original_hf_prefixes_matched'] for row in result['trajectory_rows']))


def initial_result():
    return dict(status='running', phase='own-output-tensor', model='Qwen2-0.5B',
        source_checked=False, candidate_frozen=False, tensor_forward_attempts=0, tensor_forward_completed=0,
        state_update_attempts=0, state_update_completed=0, generated_token_count=0,
        own_output_generation_completed=False, approximate_tokens_fed_back=False,
        training_quality_passed=None, original_hf_trajectory_matched=None,
        heldout_samples=0, confirm_samples=0, report_samples=0, api_calls=0,
        fresh_hf_forwards=0, approximation_selection_runs=0, coefficient_reconstruction_calls=None,
        coefficient_policy='Original deterministic evaluators reconstruct fixed selected coefficients; call count is unmeasured',
        mock_forward_attempts=0, mock_forward_completed=0, emit_runs=0, compiler_runs=0,
        keygen_runs=0, ckks_executions=0, intermediate_decryptions=0,
        full_ciphertext_model_binding_verified=False, compiled_bootstraps=None, predicted_latency=None,
        own_output_packed_equivalence_verified=False,
        CKKS_latency=None, accuracy_acceptance=None,
        scope='Three original training prompts; approximate Tensor greedy feedback for two outputs; independent original HF own-token references')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work-dir', 'count-result', 'model-info', 'reference'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir/'worker-result.json').exists():
        raise FileExistsError('Preserve the previous own-output measurement')
    result = initial_result()
    recorder = StageRecorder(args.work_dir/'events.jsonl')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    try:
        import numpy as np
        with recorder.span('qwen_own_output_tensor_generation'):
            run_validation(args, recorder, result, np)
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:3000],
            training_quality_passed=None)
    result['worker_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.work_dir/'worker-result.json', result)
    print(json.dumps({key:result[key] for key in ('status', 'tensor_forward_completed', 'state_update_completed')}))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
