"""Deterministic activation tuning using held-out validation and predicted cost."""

import copy
from dataclasses import dataclass
import time

from ..cost import estimate
from ..dsl.ir import Program
from ..dsl.printer import print_program
from ..tensor_sim import cached_estimate
from .agent_tools import _exact_check, validate


LADDER = (255, 127, 63, 31, 15, 7)
TARGET_U = 0.80
N_COARSE = 200
TUNER_MAX_ACC_DROP_PP = 0.3


@dataclass
class TuneResult:
    program: Program
    source: str
    changes: list[dict]
    cost_before: dict
    cost_after: dict
    coarse_evals: int
    full_evals: int
    seconds: float
    history: list[dict]
    feasible: bool


def _strict(result):
    return result['feasible'] and result['acc_drop_pp'] <= TUNER_MAX_ACC_DROP_PP


def _changes(original, current):
    changes = []
    for old, new in zip(original.ops, current.ops):
        if old.kind in ('act', 'maxpool'):
            for kind in ('domain', 'deg'):
                if old.attrs[kind] != new.attrs[kind]:
                    changes.append(dict(line=old.lineno, call_lines=list(old.call_lines), kind=kind,
                                        old=old.attrs[kind], new=new.attrs[kind]))
    return changes


def tune(program, ctx, *, budget_s: float = 2400, max_sweeps: int = 1,
         engine: str = "tensor", cache: bool = True) -> TuneResult:
    started = time.monotonic()
    def cost_of(candidate):
        return cached_estimate(candidate, ctx.weights, ctx.profile, estimate, cache=cache)

    def validation(candidate, n=None, cost=None):
        if engine == "mock":
            cost = cost_of(candidate) if cost is None else cost
            return validate(candidate, ctx, n, boot_nodes=cost['boot_nodes'], engine=engine, cache=cache)
        return validate(candidate, ctx, n, engine=engine, cache=cache)

    cur = copy.deepcopy(program)
    before = cost_of(cur)
    history = []
    coarse_evals = full_evals = 0

    def full(candidate):
        nonlocal full_evals
        full_evals += 1
        return validation(candidate)

    def record(phase, candidate, cost, result, accepted, **details):
        entry = dict(phase=phase, latency_pred_us=cost['latency_pred_us'],
                     agree=result.get('agree'), total=result.get('total'), max_u=result.get('max_u'),
                     top1=result.get('top1'), exact_top1=result.get('exact_top1'),
                     acc_drop_pp=result.get('acc_drop_pp'),
                     accepted=accepted, **details)
        history.append(entry)
        print(f"TUNER phase={phase} accepted={accepted} agree={entry['agree']}/{entry['total']} "
              f"max_u={entry['max_u']} latency_pred_us={entry['latency_pred_us']}", flush=True)

    def finish(candidate, result, feasible):
        after = cost_of(candidate)
        if initial['feasible'] and after['latency_pred_us'] > before['latency_pred_us']:
            candidate, result, after, feasible = copy.deepcopy(program), initial, before, True
            record('rollback_cost', candidate, after, result, False)
        if engine == "tensor":
            verified = validate(candidate, ctx, boot_nodes=after['boot_nodes'],
                                engine="mock", cache=cache)
            for field in ('feasible', 'agree'):
                if verified[field] != result[field]:
                    raise RuntimeError(
                        f"Final Mock verification differs from tensor: {field} "
                        f"tensor={result[field]!r}, mock={verified[field]!r}")
            result = verified
        record('result', candidate, after, result, feasible, feasible=feasible,
               boot_input_max_abs=result.get('boot_input_max_abs'), worst_acts=result.get('worst_acts', []))
        return TuneResult(candidate, print_program(candidate), _changes(program, candidate), before, after,
                          coarse_evals, full_evals, time.monotonic()-started, history, feasible)

    initial = full(cur)
    if initial['diverged'] or not _exact_check(cur, ctx)['ok']:
        return finish(cur, initial, False)
    aligned = copy.deepcopy(cur)
    acts = [op for op in aligned.ops if op.kind in ('act', 'maxpool')]
    for op, row in zip(acts, initial['rows']):
        lo, hi = row['obs_min'], row['obs_max']
        if lo is None or hi is None:
            continue
        if op.attrs['method'] == 'minimax_comp':
            bound = max(abs(lo), abs(hi)) / TARGET_U
            domain = (-bound, bound)
        else:
            center, half = (lo+hi)/2, (hi-lo)/2/TARGET_U
            domain = (center-half, center+half)
        op.attrs['domain'] = tuple(round(value, 3) for value in domain)
    # Degenerate rounded domains cannot be executed; reject this whole alignment.
    if all(op.attrs['domain'][0] < op.attrs['domain'][1] for op in acts):
        aligned_result = full(aligned)
        aligned_cost = cost_of(aligned)
    else:
        aligned_result = dict(initial, feasible=False)
        aligned_cost = before
    accepted_alignment = _strict(aligned_result)
    record('domain', aligned, aligned_cost, aligned_result, accepted_alignment,
           changes=_changes(program, aligned))
    if accepted_alignment:
        cur, current_result, current_cost = aligned, aligned_result, aligned_cost
    elif initial['feasible']:
        current_result, current_cost = initial, before
        record('rollback_domain', cur, before, initial, False)
    else:
        return finish(copy.deepcopy(program), initial, False)

    for sweep in range(max_sweeps):
        if time.monotonic()-started >= budget_s:
            break
        base = copy.deepcopy(cur)
        base_result = current_result
        accepted = []
        hotspots = current_cost.get('hotspots', [])
        ranked = {(row['line'], tuple(row.get('call_lines', ()))): rank
                  for rank, row in reversed(list(enumerate(sorted(hotspots,
                                      key=lambda row: row.get('latency_share', 0), reverse=True))))}
        order = sorted((index for index, op in enumerate(cur.ops) if op.kind in ('act', 'maxpool')),
                       key=lambda index: (ranked.get((cur.ops[index].lineno, cur.ops[index].call_lines), len(ranked)), index))
        for index in order:
            if time.monotonic()-started >= budget_s:
                break
            op = cur.ops[index]
            if op.attrs['method'] != 'cheb':
                continue
            degree = next((degree for degree in LADDER if degree < op.attrs['deg']), None)
            if degree is None:
                continue
            trial = copy.deepcopy(cur)
            trial.ops[index].attrs['deg'] = degree
            cost = cost_of(trial)
            detail = dict(act=index, line=op.lineno, call_lines=list(op.call_lines), kind='deg',
                          old=op.attrs['deg'], new=degree, sweep=sweep)
            if cost['latency_pred_us'] >= current_cost['latency_pred_us']:
                record('skip_cost', trial, cost, {}, False, **detail)
                continue
            result = validation(trial, N_COARSE, cost)
            coarse_evals += 1
            record('coarse', trial, cost, result, _strict(result), **detail)
            if _strict(result):
                accepted.append((index, degree))
                cur, current_cost = trial, cost
        current_result = full(cur)
        record('confirm', cur, current_cost, current_result, _strict(current_result), sweep=sweep)
        if not _strict(current_result):
            # Prefix 0 is the already confirmed sweep base. Search all longer prefixes.
            low, high = 0, len(accepted)-1
            kept, kept_result = base, base_result
            while low < high:
                middle = (low+high+1)//2
                trial = copy.deepcopy(base)
                for index, degree in accepted[:middle]:
                    trial.ops[index].attrs['deg'] = degree
                result = full(trial)
                cost = cost_of(trial)
                record('prefix', trial, cost, result, _strict(result), prefix=middle, sweep=sweep)
                if _strict(result):
                    low, kept, kept_result = middle, trial, result
                else:
                    high = middle-1
            cur, current_result = kept, kept_result
            current_cost = cost_of(cur)
            record('rollback', cur, current_cost, current_result, True,
                   kept=low, removed=len(accepted)-low, sweep=sweep)
        if not accepted or time.monotonic()-started >= budget_s:
            break
    return finish(cur, current_result, current_result['feasible'])
