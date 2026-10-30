"""Confirm frozen candidates after the internal generation loop has ended."""

from dataclasses import dataclass, replace

import numpy as np

from ..analyzer import reference_forward
from ..data.cifar import confirm_set
from ..dsl.parser import parse
from ..dsl.printer import print_program
from ..metrics import passes
from .agent_tools import AgentContext, MAX_ACC_DROP_PP, VAL_BATCH, validate


MAX_CONFIRM_EVALS = 6


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
                   engine_profile: str = "mock") -> dict:
    if engine_profile not in ("mock", "tensor"):
        raise ValueError('engine_profile must be "mock" or "tensor"')
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
        token_agent=getattr(ctx,'prompt_version','r8-v4') in ('r8-v7','r8-v8','r8-v9','r8-v10')
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
        print(f"CONFIRM iteration={candidate.iteration} origin={candidate.origin} "
              f"passed={passed} metrics={ {key: result[key] for key in metric_keys} } "
              f"n_diverged={result['n_diverged']} max_u={result['max_u']}", flush=True)
        if passed:
            selected = candidate
            break
    return dict(selected=selected, evals=evals,
                deduplicated=dict(before=len(candidates), after=len(kept)), duplicates=duplicates)
