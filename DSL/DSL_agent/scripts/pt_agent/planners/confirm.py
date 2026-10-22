"""Confirm frozen candidates after the internal generation loop has ended."""

from dataclasses import dataclass, replace

import numpy as np

from ..analyzer import reference_forward
from ..data.cifar import confirm_set
from ..dsl.parser import parse
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
    images, labels = confirm_set(normalization=ctx.normalization)
    graph = ctx.graph
    images = images.reshape((len(images), *graph.tensors[graph.inputs[0]].shape[1:]))
    logits = []
    for start in range(0, len(images), VAL_BATCH):
        values = reference_forward(graph, {graph.inputs[0]: images[start:start+VAL_BATCH]})
        logits.append(values[graph.outputs[0]].copy())
    exact = np.concatenate(logits)
    return replace(ctx, val_images=images, val_labels=labels, val_exact_logits=exact,
                   val_exact_top1=float(100*np.mean(exact.argmax(axis=1) == labels)),
                   val_split="test[8000:9000]")


def confirm_select(candidates: list[Candidate], ctx, *, max_evals=MAX_CONFIRM_EVALS) -> dict:
    ordered = sorted(candidates, key=lambda c: (c.latency_pred_us, c.iteration, c.origin))
    selected, evals, seen = None, [], set()
    confirm_ctx = None
    for candidate in ordered:
        if len(evals) >= max_evals:
            break
        if candidate.source in seen:
            continue
        seen.add(candidate.source)
        if confirm_ctx is None:
            confirm_ctx = ctx if ctx.val_split == "test[8000:9000]" else build_confirm_context(ctx)
        result = validate(parse(candidate.source), confirm_ctx, stop_on_divergence=False)
        boot_ok = (result["boot_input_bound"] is None
                   or result["boot_input_max_abs"] <= result["boot_input_bound"])
        passed = (result["acc_drop_pp"] is not None
                  and result["acc_drop_pp"] <= MAX_ACC_DROP_PP and boot_ok)
        evals.append(dict(iteration=candidate.iteration, origin=candidate.origin,
                          latency_pred_us=candidate.latency_pred_us, boots_pred=candidate.boots_pred,
                          **{key: result[key] for key in ("top1", "exact_top1", "acc_drop_pp",
                                                        "max_u", "diverged", "n_diverged")}, passed=passed))
        print(f"CONFIRM iteration={candidate.iteration} origin={candidate.origin} "
              f"passed={passed} top1={result['top1']} acc_drop_pp={result['acc_drop_pp']} "
              f"n_diverged={result['n_diverged']} max_u={result['max_u']}", flush=True)
        if passed:
            selected = candidate
            break
    return dict(selected=selected, evals=evals)
