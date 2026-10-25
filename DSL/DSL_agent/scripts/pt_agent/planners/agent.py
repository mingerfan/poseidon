"""Choose the best feasible program using only pre-freeze internal feedback."""

import json
from dataclasses import asdict
from pathlib import Path
import time

from ..llm import LLMError
from ..dsl.parser import parse
from ..dsl.printer import print_program
from .agent_tools import AgentContext, evaluate
from .confirm import Candidate, confirm_select
from .prompts import build_system_prompt, feedback_message, initial_message
from .tuner import tune


def run_agent(ctx: AgentContext, client, *, max_iters: int = 6, trace_path: Path,
              tuner: bool = True, tuner_budget_s: float = 2400,
              tune_policy: str = "new_best", engine_profile: str = "mock") -> dict:
    if tune_policy not in ("new_best", "all"):
        raise ValueError('tune_policy must be "new_best" or "all"')
    if engine_profile not in ("mock", "tensor"):
        raise ValueError('engine_profile must be "mock" or "tensor"')
    if engine_profile == "tensor" and getattr(ctx, 'prompt_version', 'r8-v4') != 'r8-v5':
        raise ValueError('engine_profile="tensor" requires r8-v5')
    profile_args = dict(engine_profile="tensor") if engine_profile == "tensor" else {}
    started = time.monotonic()
    messages = [dict(role="system", content=build_system_prompt(ctx)), dict(role="user", content=initial_message(ctx))]
    best_program = best_iteration = first_feasible = best = None
    best_llm_lines = None
    llm_min = None
    has_tuned = False
    tune_cache = {}
    history = []
    pool, records = [], {}
    calls = prompt_tokens = completion_tokens = stale = 0
    failed = None
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, max_iters+1):
        previous = None if best is None else best["latency_pred_us"]
        calls += 1
        try:
            reply = client.chat(messages, purpose=f"agent-iter-{iteration}")
        except LLMError as exc:
            failed = exc.code
            fb = dict(stage="llm", feasible=False, error=exc.code, diagnostics=exc.diagnostics)
            history.append(dict(iteration=iteration, stage="llm", feasible=False, latency_pred_us=None,
                                boots_pred=None, agree=None, max_u=None))
            with trace_path.open("a") as stream:
                stream.write(json.dumps(dict(iteration=iteration, messages=[], fb=fb, usage={}, seconds=0,
                                             tuned=None, tune_policy=tune_policy))+"\n")
            break
        usage = reply.get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)
        source = None
        try:
            response = json.loads(reply["content"])
            source = response["program"]
            if not isinstance(source, str):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            fb = dict(stage="response", feasible=False, error="response_not_json_with_program")
        else:
            fb = evaluate(source, ctx, **(dict(engine="tensor", **profile_args) if profile_args else {}))
        cost, validation = fb.get("cost") or {}, fb.get("validation") or {}
        row = dict(iteration=iteration, stage=fb["stage"], feasible=fb.get("feasible", False),
                            latency_pred_us=cost.get("latency_pred_us"), boots_pred=cost.get("boots_pred"),
                            agree=validation.get("agree"), max_u=validation.get("max_u"), tuned=None)
        candidates = [(source, cost, validation, 'llm')] if fb.get('feasible') else []
        tuned = None
        new_llm_best = bool(cost and (llm_min is None or cost['latency_pred_us'] < llm_min))
        if cost:
            llm_min = cost['latency_pred_us'] if llm_min is None else min(llm_min, cost['latency_pred_us'])
        eligible = (fb.get('exact', {}) and fb['exact']['ok'] and validation
                    and not validation['diverged'] and fb['stage'] == 'complete'
                    and (validation.get('boot_input_bound') is None
                         or validation['boot_input_max_abs'] <= validation['boot_input_bound']))
        tuned_reused_from = None
        if tuner and eligible and (tune_policy == "all" or new_llm_best or not has_tuned):
            has_tuned = True
            parsed = parse(source)
            normalized = print_program(parsed)
            if normalized in tune_cache:
                result, tuned_reused_from, tune_source = tune_cache[normalized]
            else:
                result = tune(parsed, ctx, budget_s=tuner_budget_s, max_sweeps=1, **profile_args)
                tune_source = source
                tune_cache[normalized] = (result, iteration, tune_source)
            tuned = {key: value for key, value in asdict(result).items() if key != 'program'}
            # A reused result retains its original source locations and snippets.
            lines = tune_source.splitlines()
            for change in tuned['changes']:
                change['line_text'] = lines[change['line']-1].strip()
                change['call_texts'] = [lines[line-1].strip() for line in change['call_lines']]
            confirmation = result.history[-1]
            row['tuned'] = dict(feasible=result.feasible, latency_pred_us=result.cost_after['latency_pred_us'],
                                boots_pred=result.cost_after['boots_pred'], agree=confirmation['agree'],
                                max_u=confirmation['max_u'], coarse_evals=result.coarse_evals,
                                full_evals=result.full_evals, seconds=result.seconds)
            if result.feasible:
                candidates.append((result.source, result.cost_after, confirmation, 'tuned'))
        row.update(has_tuned=has_tuned, new_llm_best=new_llm_best, llm_min=llm_min,
                   tuned_reused_from=tuned_reused_from)
        for candidate, candidate_cost, confirmed, origin in candidates:
            pool.append(Candidate(candidate, origin, iteration, candidate_cost['latency_pred_us'],
                                  candidate_cost['boots_pred']))
            records[(iteration, origin, candidate)] = (
                dict(candidate_cost, iteration=iteration, source=origin,
                     agree=confirmed['agree'], max_u=confirmed['max_u'],
                     boot_input_max_abs=confirmed.get('boot_input_max_abs'),
                     worst_acts=confirmed.get('worst_acts', [])),
                sum(bool(line.strip()) for line in source.splitlines()))
            if first_feasible is None:
                first_feasible = iteration
            if best is None or candidate_cost['latency_pred_us'] < best['latency_pred_us']:
                best_program, best_iteration = candidate, iteration
                best_llm_lines = sum(bool(line.strip()) for line in source.splitlines())
                best = dict(candidate_cost, iteration=iteration, source=origin,
                            agree=confirmed['agree'], max_u=confirmed['max_u'],
                            boot_input_max_abs=confirmed.get('boot_input_max_abs'),
                            worst_acts=confirmed.get('worst_acts', []))
        history.append(row)
        added = [dict(role="assistant", content=reply["content"]),
                 dict(role="user", content=feedback_message(fb, iteration, best, tuned=tuned))]
        messages.extend(added)
        with trace_path.open("a") as stream:
            stream.write(json.dumps(dict(iteration=iteration, messages=added, fb=fb, usage=usage,
                                         seconds=reply["seconds"], tuned=tuned,
                                         tuned_reused_from=tuned_reused_from,
                                         tune_policy=tune_policy), allow_nan=False)+"\n")
        print(f"AGENT iteration={iteration} stage={fb['stage']} feasible={fb.get('feasible', False)}", flush=True)
        if best is not None:
            stale = 0 if previous is None or previous-best["latency_pred_us"] >= .05*previous else stale+1
            if stale >= 2:
                break
    confirmation = confirm_select(pool, ctx, **profile_args)
    selected = confirmation['selected']
    if selected is None:
        best_program = best_iteration = best = best_llm_lines = None
    else:
        best_program, best_iteration = selected.source, selected.iteration
        best, best_llm_lines = records[(selected.iteration, selected.origin, selected.source)]
    return dict(status="error" if failed else "feasible" if selected is not None else "confirm_failed",
                best_program=best_program, best_iteration=best_iteration, first_feasible_iteration=first_feasible,
                best=best, best_source=best['source'] if best else None, llm_source_lines=best_llm_lines,
                frozen_source_lines=sum(bool(line.strip()) for line in best_program.splitlines()) if best_program else None,
                iterations=len(history), llm_calls=calls, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                seconds=time.monotonic()-started, history=history, error=failed,
                confirm=confirmation, pool_size=len(pool), tune_policy=tune_policy)
