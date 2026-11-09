"""Confirm frozen candidates after the internal generation loop has ended."""

from dataclasses import dataclass, replace
import time

import numpy as np

from ..analyzer import reference_forward
from ..data.cifar import confirm_set
from ..dsl.parser import parse
from ..dsl.printer import print_program
from ..metrics import passes
from .agent_tools import AgentContext, MAX_ACC_DROP_PP, VAL_BATCH, validate


MAX_CONFIRM_EVALS = 6


def confirmation_rule(ctx, rule):
    if rule not in ('c1', 'c2'):
        raise ValueError('confirm_rule must be c1 or c2')
    info = getattr(ctx, 'model_info', None)
    applicable = info is not None and getattr(info, 'split', {}).get('kind') != 'legacy_cifar'
    return rule if applicable else 'c1'


def build_non_report_context(ctx, *, splits=('calibration', 'heldout', 'confirm')):
    if tuple(splits) != ('calibration', 'heldout', 'confirm'):
        raise ValueError('Tail context requires calibration, heldout, confirm in that order; report is forbidden')
    if confirmation_rule(ctx, 'c2') != 'c2':
        raise ValueError('Non-report tail context is not applicable to the CIFAR path')
    parts = [ctx.samples.get(name) for name in splits]
    inputs = {name: np.concatenate([arrays[name] for arrays, _ in parts]) for name in ctx.graph.inputs}
    label_parts = [labels for _, labels in parts]
    if all(labels is None for labels in label_parts):
        labels = None
    elif any(labels is None for labels in label_parts):
        raise ValueError('Non-report partitions have inconsistent labels')
    else:
        labels = np.concatenate(label_parts)
    positions = [dict(partition=name, index=i) for name, (arrays, _) in zip(splits, parts)
                 for i in range(len(arrays[ctx.graph.inputs[0]]))]
    logits = []
    count = len(positions)
    for start in range(0, count, VAL_BATCH):
        values = reference_forward(ctx.graph, {name: value[start:start+VAL_BATCH] for name, value in inputs.items()})
        logits.append(values[ctx.graph.outputs[0]].copy())
    exact = np.concatenate(logits)
    result = replace(ctx, val_images=inputs[ctx.graph.inputs[0]], val_inputs=inputs,
        val_labels=labels, val_exact_logits=exact, val_split='non_report',
        val_exact_top1=float(100*np.mean(exact.argmax(axis=1) == labels)) if labels is not None else None)
    result.tail_sample_positions = positions
    return result


def _metric_outputs(ctx, values):
    if ctx.model_info.task != 'causal_lm':
        return values
    count = len(values)
    selected = ctx.model_info.input_handling.get('single_lm_head', False)
    indices = (np.zeros(count, dtype=int) if selected else
               ctx.val_inputs['attention_mask'][:count].sum(1).astype(int)-1)
    return values[np.arange(count), indices]


def tail_check(source, ctx, *, engine_profile='mock'):
    if engine_profile not in ('mock', 'tensor'):
        raise ValueError('engine_profile must be mock or tensor')
    started = time.monotonic()
    outputs = []
    result = validate(parse(source), ctx, stop_on_divergence=False, output_callback=outputs.append,
                      **(dict(engine='tensor') if engine_profile == 'tensor' else {}))
    exact = _metric_outputs(ctx, ctx.val_exact_logits)
    actual = _metric_outputs(ctx, outputs[0])
    denominator = float(np.max(np.abs(exact)))
    with np.errstate(invalid='ignore', over='ignore'):
        errors = np.max(np.abs(exact-actual).reshape(len(exact), -1), axis=1)
    errors = np.where(np.isfinite(errors), errors, np.inf)
    worst = []
    for index in sorted(range(len(errors)), key=lambda i: (-errors[i], i))[:5]:
        ratio = float(errors[index])/denominator if denominator else (0. if errors[index] == 0 else None)
        worst.append(dict(ctx.tail_sample_positions[index], relative_error=ratio))
    return dict(total=result['total'], agree=result.get('agree'),
        agreement_pct=result.get('agreement_pct'), logit_rel_err=result.get('logit_rel_err'),
        n_diverged=result['n_diverged'], diverged=result['diverged'],
        passed=bool(passes(result, ctx.model_info.threshold) and not result['diverged']),
        seconds=time.monotonic()-started, worst_samples=worst,
        metrics={key: result.get(key) for key in ctx.model_info.threshold})


@dataclass
class Candidate:
    source: str
    origin: str
    iteration: int
    latency_pred_us: float
    boots_pred: int


def build_confirm_context(ctx) -> AgentContext:
    if getattr(ctx, 'model_info', None) is not None:
        inputs, labels = ctx.samples.get('confirm')
        images = inputs[ctx.graph.inputs[0]]
        split_name = 'confirm'
    else:
        images, labels = confirm_set(normalization=ctx.normalization)
        inputs = None
        split_name = 'test[8000:9000]'
    graph = ctx.graph
    images = images.reshape((len(images), *graph.tensors[graph.inputs[0]].shape[1:]))
    logits = []
    for start in range(0, len(images), VAL_BATCH):
        values = reference_forward(graph, {name: value[start:start+VAL_BATCH] for name, value in inputs.items()}
                                   if inputs is not None else {graph.inputs[0]: images[start:start+VAL_BATCH]})
        logits.append(values[graph.outputs[0]].copy())
    exact = np.concatenate(logits)
    return replace(ctx, val_images=images, val_labels=labels, val_exact_logits=exact,
                   val_exact_top1=(float(100*np.mean(exact.argmax(axis=1) == labels))
                                   if labels is not None else None),
                   val_split=split_name, val_inputs=inputs)


def confirm_select(candidates: list[Candidate], ctx, *, max_evals=MAX_CONFIRM_EVALS,
                   engine_profile: str = "mock", confirm_rule: str = 'c1') -> dict:
    if engine_profile not in ("mock", "tensor"):
        raise ValueError('engine_profile must be "mock" or "tensor"')
    requested_rule = confirm_rule
    confirm_rule = confirmation_rule(ctx, confirm_rule)
    from .. import domain_criteria as criteria
    v8=criteria.enabled(ctx)
    if v8:
        from ..dsl.interface import require_interface
        for candidate in candidates:
            require_interface(parse(candidate.source),ctx.model_info)
            from ..margin import require_margin
            require_margin(parse(candidate.source),ctx)
    normalized = [print_program(parse(candidate.source)) for candidate in candidates]
    representatives = {}
    for index, (candidate, key) in enumerate(zip(candidates, normalized)):
        if key not in representatives or candidate.latency_pred_us < candidates[representatives[key]].latency_pred_us:
            representatives[key] = index
    kept = sorted(representatives.values())
    duplicates = [dict(iteration=candidate.iteration, origin=candidate.origin,
                       kept_iteration=candidates[representatives[key]].iteration,
                       kept_origin=candidates[representatives[key]].origin)
                  for index, (candidate, key) in enumerate(zip(candidates, normalized))
                  if index != representatives[key]]
    ordered = sorted((candidates[index] for index in kept),
                     key=lambda c: (c.latency_pred_us, c.iteration, c.origin))
    selected, evals = None, []
    confirm_ctx = None
    tail_ctx = None
    for candidate in ordered:
        if len(evals) >= max_evals:
            break
        if confirm_ctx is None:
            confirm_ctx = ctx if ctx.val_split in ("test[8000:9000]", 'confirm') else build_confirm_context(ctx)
        result = validate(parse(candidate.source), confirm_ctx, stop_on_divergence=False,
                          **(dict(engine="tensor") if engine_profile == "tensor" else {}))
        boot_ok = (result["boot_input_bound"] is None
                   or result["boot_input_max_abs"] <= result["boot_input_bound"])
        info = getattr(ctx, 'model_info', None)
        passed = passes(result, info.threshold if info is not None else {'acc_drop_pp': MAX_ACC_DROP_PP}) and boot_ok
        token_agent=getattr(ctx,'prompt_version','r8-v4') in ('r8-v7','r8-v8','r8-v9','r8-v10','r8-v11')
        if token_agent:
            if v8:criteria.require_analytic_coverage(result)
            passed=passed and result['feasible'] and not result['diverged'] and (
                criteria.domain_ok(result) if v8 else result['max_u']<=0.90)
        metric_keys = ('top1', 'exact_top1', 'acc_drop_pp') if info is None or (
            info.task == 'classification' and ctx.val_labels is not None) else tuple(info.threshold)
        evals.append(dict(iteration=candidate.iteration, origin=candidate.origin,
                          latency_pred_us=candidate.latency_pred_us, boots_pred=candidate.boots_pred,
                          **{key: result[key] for key in (*metric_keys, "max_u", "diverged", "n_diverged")}, passed=passed))
        if token_agent:
            evals[-1].update({key:result[key] for key in ('agreement_pct','logit_rel_err','relative_error','agree','total')
                             if key in result})
        if v8:
            evals[-1].update({key:result[key] for key in ('max_u_calibrated','max_u_analytic','worst_acts')})
        if confirm_rule == 'c2' and passed:
            if tail_ctx is None:
                tail_ctx = build_non_report_context(ctx)
            checked = tail_check(candidate.source, tail_ctx, engine_profile=engine_profile)
            evals[-1]['tail'] = checked
            passed = passed and checked['passed']
            evals[-1]['passed'] = passed
        print(f"CONFIRM iteration={candidate.iteration} origin={candidate.origin} "
              f"passed={passed} metrics={ {key: result[key] for key in metric_keys} } "
              f"n_diverged={result['n_diverged']} max_u={result['max_u']}", flush=True)
        if passed:
            selected = candidate
            break
    record = dict(selected=selected, evals=evals,
                  deduplicated=dict(before=len(candidates), after=len(kept)), duplicates=duplicates)
    if requested_rule == 'c2':
        record.update(confirm_rule=confirm_rule, applicable=confirm_rule == 'c2',
                      max_evals_reached=selected is None and len(evals) >= max_evals,
                      tail_evals=sum('tail' in row for row in evals),
                      tail_seconds=sum(row.get('tail', {}).get('seconds', 0.) for row in evals))
    return record
