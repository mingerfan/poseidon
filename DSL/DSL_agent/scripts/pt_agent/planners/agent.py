"""Choose the best feasible program using only pre-freeze internal feedback."""

import json
from contextlib import nullcontext
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
from .transformer_tuner import (DEFAULT_TUNE_RULE, PRE_TUNE_RULE, CONVERGENT_TUNE_RULE, convergent_rule,
                               validate_tune_rule, tune_rule_scope)


def _run_agent_once(ctx: AgentContext, client, *, max_iters: int = 6, trace_path: Path,
              tuner: bool = True, tuner_budget_s: float = 2400,
              tune_policy: str = "new_best", engine_profile: str = "mock",
              tune_rule: str | None = None) -> dict:
    tune_rule = validate_tune_rule(getattr(ctx, 'prompt_version', 'r8-v4'), tune_rule)
    if tune_policy not in ("new_best", "all"):
        raise ValueError('tune_policy must be "new_best" or "all"')
    if engine_profile not in ("mock", "tensor"):
        raise ValueError('engine_profile must be "mock" or "tensor"')
    if engine_profile == "tensor" and getattr(ctx, 'prompt_version', 'r8-v4') not in ('r8-v5','r8-v6','r8-v7','r8-v8','r8-v9','r8-v10'):
        raise ValueError('engine_profile="tensor" requires r8-v5, r8-v6, r8-v7 or r8-v8')
    v8 = getattr(ctx, 'prompt_version', 'r8-v4') in ('r8-v8','r8-v9','r8-v10')
    rule_record = dict(tune_rule=tune_rule) if v8 else {}
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
                                             tuned=None, tune_policy=tune_policy, **rule_record))+"\n")
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
            if v8 and reply.get('finish_reason') == 'length':
                fb = dict(stage='length', feasible=False, output_limit=65536,
                          error='The previous response reached the output limit (65536 tokens) before writing a program. Shorten the reasoning and output the complete program directly.')
        else:
            if getattr(ctx,'prompt_version','r8-v4') in ('r8-v7','r8-v8','r8-v9','r8-v10'):
                # Retain parsed response source before any internal tool runs.
                response_path=trace_path.parent/'sources'/f'iteration-{iteration}.ptdsl'
                response_path.parent.mkdir(parents=True,exist_ok=True)
                with response_path.open('x') as stream:stream.write(source)
            fb = evaluate(source, ctx, **(dict(engine="tensor", **profile_args) if profile_args else {}))
            if v8 and not source.strip() and reply.get('finish_reason') == 'length':
                fb = dict(stage='length', feasible=False, output_limit=65536,
                          error='The previous response reached the output limit (65536 tokens) before writing a program. Shorten the reasoning and output the complete program directly.')
        cost, validation = fb.get("cost") or {}, fb.get("validation") or {}
        row = dict(iteration=iteration, stage=fb["stage"], feasible=fb.get("feasible", False),
                            latency_pred_us=cost.get("latency_pred_us"), boots_pred=cost.get("boots_pred"),
                            agree=validation.get("agree"), max_u=validation.get("max_u"), tuned=None)
        candidates = [(source, cost, validation, 'llm')] if fb.get('feasible') else []
        tuned = None
        comparable = cost and (not v8 or tune_rule == PRE_TUNE_RULE or fb.get('feasible', False))
        new_llm_best = bool(comparable and (llm_min is None or cost['latency_pred_us'] < llm_min))
        if comparable:
            llm_min = cost['latency_pred_us'] if llm_min is None else min(llm_min, cost['latency_pred_us'])
        eligible = (fb.get('exact', {}) and fb['exact']['ok'] and validation
                    and not validation['diverged'] and fb['stage'] == 'complete'
                    and (validation.get('boot_input_bound') is None
                         or validation['boot_input_max_abs'] <= validation['boot_input_bound']))
        tuned_reused_from = None
        if tuner and eligible and not fb.get('stop') and (tune_policy == "all" or new_llm_best or not has_tuned):
            has_tuned = True
            parsed = parse(source)
            normalized = print_program(parsed)
            if normalized in tune_cache:
                result, tuned_reused_from, tune_source = tune_cache[normalized]
            else:
                with tune_rule_scope(tune_rule) if v8 else nullcontext():
                    result = tune(parsed, ctx, budget_s=tuner_budget_s,
                                  max_sweeps=8 if convergent_rule(tune_rule) else 1, **profile_args)
                tune_source = source
                tune_cache[normalized] = (result, iteration, tune_source)
                if convergent_rule(tune_rule) and getattr(result,'converged',False):
                    tune_cache[print_program(result.program)] = (result, iteration, tune_source)
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
            if v8:
                row['tuned'].update({key: confirmation[key] for key in ('max_u_calibrated','max_u_analytic')})
            if convergent_rule(tune_rule):
                row['tuned'].update(sweeps=result.sweeps,converged=result.converged)
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
                     **({key:confirmed[key] for key in ('max_u_calibrated','max_u_analytic')} if v8 else {}),
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
                            **({key:confirmed[key] for key in ('max_u_calibrated','max_u_analytic')} if v8 else {}),
                            boot_input_max_abs=confirmed.get('boot_input_max_abs'),
                            worst_acts=confirmed.get('worst_acts', []))
        history.append(row)
        token_agent=getattr(ctx,'prompt_version','r8-v4') in ('r8-v7','r8-v8','r8-v9','r8-v10')
        from .transformer_session import json_safe
        shown_fb,shown_tuned=(json_safe(fb),json_safe(tuned)) if token_agent else (fb,tuned)
        added = [dict(role="assistant", content=reply["content"]),
                 dict(role="user", content=feedback_message(shown_fb, iteration, best, tuned=shown_tuned))]
        messages.extend(added)
        with trace_path.open("a") as stream:
            stream.write(json.dumps(dict(iteration=iteration, messages=added, fb=shown_fb, usage=usage,
                                         seconds=reply["seconds"], tuned=shown_tuned,
                                         tuned_reused_from=tuned_reused_from,
                                         tune_policy=tune_policy, **rule_record), allow_nan=False)+"\n")
        print(f"AGENT iteration={iteration} stage={fb['stage']} feasible={fb.get('feasible', False)}", flush=True)
        if v8 and fb.get('stop'):
            failed=fb['stop']
            break
        if best is not None:
            stale = 0 if previous is None or previous-best["latency_pred_us"] >= .05*previous else stale+1
            if stale >= 2:
                break
    confirmation = (dict(selected=None,evals=[],deduplicated=dict(before=len(pool),after=0),duplicates=[])
                    if v8 and failed else confirm_select(pool, ctx, **profile_args))
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
                confirm=confirmation, pool_size=len(pool), tune_policy=tune_policy, **rule_record)


def new_agent_state(ctx, *, max_iters=6, tuner=True, tuner_budget_s=2400,
                    tune_policy='new_best', engine_profile='mock', tune_rule=None):
    """The frozen loop's locals, with an explicit process boundary at each tool."""
    if tune_policy not in ('new_best', 'all'):
        raise ValueError('tune_policy must be "new_best" or "all"')
    if engine_profile not in ('mock', 'tensor'):
        raise ValueError('engine_profile must be "mock" or "tensor"')
    version = getattr(ctx, 'prompt_version', 'r8-v4')
    tune_rule = validate_tune_rule(version, tune_rule)
    if version not in ('r8-v8','r8-v9','r8-v10'):
        raise ValueError('Resumable steps require r8-v8')
    split=getattr(getattr(ctx,'model_info',None),'split',{})
    return dict(phase='evaluate', iteration=1, max_iters=max_iters, tuner=tuner,
        tuner_budget_s=tuner_budget_s, tune_policy=tune_policy, engine_profile=engine_profile,
        prompt_version=version, tune_rule=tune_rule,
        messages=[dict(role='system', content=build_system_prompt(ctx)),
                  dict(role='user', content=initial_message(ctx))],
        best_program=None, best_iteration=None, first_feasible=None, best=None,
        best_llm_lines=None, llm_min=None, has_tuned=False, tune_cache={},
        history=[], pool=[], records={}, calls=0, prompt_tokens=0,
        completion_tokens=0, stale=0, failed=None, elapsed_s=0., pending=None,
        **(dict(model_call_offset=getattr(ctx, 'a1_model_call_offset', 0)) if version in ('r8-v9','r8-v10') else {}),
        **(dict(split_version=split['version']) if 'version' in split else {}))


def _finish_iteration(state, trace_path):
    from .transformer_session import json_safe
    pending = state['pending']
    iteration, source, fb = pending['iteration'], pending['source'], pending['fb']
    row, tuned = pending['row'], pending['tuned']
    row.update(has_tuned=state['has_tuned'], new_llm_best=pending['new_llm_best'],
               llm_min=state['llm_min'], tuned_reused_from=pending['tuned_reused_from'])
    for candidate, candidate_cost, confirmed, origin in pending['candidates']:
        state['pool'].append(asdict(Candidate(candidate, origin, iteration,
            candidate_cost['latency_pred_us'], candidate_cost['boots_pred'])))
        record = dict(candidate_cost, iteration=iteration, source=origin,
            agree=confirmed['agree'], max_u=confirmed['max_u'],
            **{key: confirmed[key] for key in ('max_u_calibrated', 'max_u_analytic')},
            boot_input_max_abs=confirmed.get('boot_input_max_abs'),
            worst_acts=confirmed.get('worst_acts', []))
        lines = sum(bool(line.strip()) for line in source.splitlines())
        state['records'][(iteration, origin, candidate)] = (record, lines)
        if state['first_feasible'] is None:
            state['first_feasible'] = iteration
        if state['best'] is None or candidate_cost['latency_pred_us'] < state['best']['latency_pred_us']:
            state.update(best_program=candidate, best_iteration=iteration,
                         best_llm_lines=lines, best=record)
    state['history'].append(row)
    shown_fb, shown_tuned = json_safe(fb), json_safe(tuned)
    reply = pending['reply']
    added = [dict(role='assistant', content=reply['content']),
             dict(role='user', content=feedback_message(shown_fb, iteration, state['best'], tuned=shown_tuned))]
    state['messages'].extend(added)
    with Path(trace_path).open('a') as stream:
        stream.write(json.dumps(dict(iteration=iteration, messages=added, fb=shown_fb,
            usage=pending['usage'], seconds=reply['seconds'], tuned=shown_tuned,
            tuned_reused_from=pending['tuned_reused_from'], tune_policy=state['tune_policy'],
            tune_rule=state.get('tune_rule', PRE_TUNE_RULE),
            **(dict(prompt_version=state['prompt_version'],
                    model_call=state.get('model_call_offset', 0)+state['calls'])
               if state['prompt_version'] in ('r8-v9','r8-v10') else {}),
            **(dict(split_version=state['split_version']) if 'split_version' in state else {})),
            allow_nan=False)+'\n')
    print(f"AGENT iteration={iteration} stage={fb['stage']} feasible={fb.get('feasible', False)}", flush=True)
    if fb.get('stop'):
        state['failed'] = fb['stop']
        state['phase'] = 'confirm'
    else:
        if state['best'] is not None:
            previous = pending['previous']
            state['stale'] = (0 if previous is None or previous-state['best']['latency_pred_us'] >= .05*previous
                              else state['stale']+1)
        state['phase'] = 'confirm' if state['stale'] >= 2 or iteration >= state['max_iters'] else 'evaluate'
    state['iteration'] = iteration+1
    state['pending'] = None


def _finish_tune(state, result, trace_path, *, reused_from=None, tune_source=None):
    pending = state['pending']
    tune_source = pending['source'] if tune_source is None else tune_source
    tuned = {key: value for key, value in asdict(result).items() if key != 'program'}
    lines = tune_source.splitlines()
    for change in tuned['changes']:
        change['line_text'] = lines[change['line']-1].strip()
        change['call_texts'] = [lines[line-1].strip() for line in change['call_lines']]
    confirmation = result.history[-1]
    pending['row']['tuned'] = dict(feasible=result.feasible,
        latency_pred_us=result.cost_after['latency_pred_us'], boots_pred=result.cost_after['boots_pred'],
        agree=confirmation['agree'], max_u=confirmation['max_u'], coarse_evals=result.coarse_evals,
        full_evals=result.full_evals, seconds=result.seconds,
        **{key: confirmation[key] for key in ('max_u_calibrated', 'max_u_analytic')})
    if convergent_rule(state.get('tune_rule')):
        pending['row']['tuned'].update(sweeps=result.sweeps,converged=result.converged)
    if result.feasible:
        pending['candidates'].append((result.source, result.cost_after, confirmation, 'tuned'))
    pending.update(tuned=tuned, tuned_reused_from=reused_from)
    _finish_iteration(state, trace_path)


def _evaluate_step(ctx, state, client, trace_path, *, reply=None, feedback=None):
    iteration = state['iteration']
    previous = None if state['best'] is None else state['best']['latency_pred_us']
    state['calls'] += 1
    try:
        if reply is None:
            reply = client.chat(state['messages'], purpose=f'agent-iter-{iteration}')
    except LLMError as exc:
        state['failed'] = exc.code
        fb = dict(stage='llm', feasible=False, error=exc.code, diagnostics=exc.diagnostics)
        state['history'].append(dict(iteration=iteration, stage='llm', feasible=False,
            latency_pred_us=None, boots_pred=None, agree=None, max_u=None))
        with Path(trace_path).open('a') as stream:
            stream.write(json.dumps(dict(iteration=iteration, messages=[], fb=fb, usage={}, seconds=0,
                tuned=None, tune_policy=state['tune_policy'],
                tune_rule=state.get('tune_rule', PRE_TUNE_RULE),
                **(dict(split_version=state['split_version']) if 'split_version' in state else {})))+'\n')
        state['phase'] = 'confirm'
        return
    usage = reply.get('usage') or {}
    state['prompt_tokens'] += usage.get('prompt_tokens', 0)
    state['completion_tokens'] += usage.get('completion_tokens', 0)
    source = None
    try:
        response = json.loads(reply['content'])
        source = response['program']
        if not isinstance(source, str):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        fb = dict(stage='response', feasible=False, error='response_not_json_with_program')
        if reply.get('finish_reason') == 'length':
            fb = dict(stage='length', feasible=False, output_limit=65536,
                error='The previous response reached the output limit (65536 tokens) before writing a program. Shorten the reasoning and output the complete program directly.')
    else:
        response_path = Path(trace_path).parent/'sources'/f'iteration-{iteration}.ptdsl'
        response_path.parent.mkdir(parents=True, exist_ok=True)
        with response_path.open('x') as stream:
            stream.write(source)
        profile_args = dict(engine_profile='tensor') if state['engine_profile'] == 'tensor' else {}
        fb = feedback if feedback is not None else evaluate(source, ctx,
            **(dict(engine='tensor', **profile_args) if profile_args else {}))
        if not source.strip() and reply.get('finish_reason') == 'length':
            fb = dict(stage='length', feasible=False, output_limit=65536,
                error='The previous response reached the output limit (65536 tokens) before writing a program. Shorten the reasoning and output the complete program directly.')
    cost, validation = fb.get('cost') or {}, fb.get('validation') or {}
    row = dict(iteration=iteration, stage=fb['stage'], feasible=fb.get('feasible', False),
        latency_pred_us=cost.get('latency_pred_us'), boots_pred=cost.get('boots_pred'),
        agree=validation.get('agree'), max_u=validation.get('max_u'), tuned=None)
    comparable = cost and (state.get('tune_rule', PRE_TUNE_RULE) == PRE_TUNE_RULE or fb.get('feasible', False))
    new_llm_best = bool(comparable and (state['llm_min'] is None or cost['latency_pred_us'] < state['llm_min']))
    if comparable:
        state['llm_min'] = (cost['latency_pred_us'] if state['llm_min'] is None
                            else min(state['llm_min'], cost['latency_pred_us']))
    eligible = (fb.get('exact', {}) and fb['exact']['ok'] and validation
                and not validation['diverged'] and fb['stage'] == 'complete'
                and (validation.get('boot_input_bound') is None
                     or validation['boot_input_max_abs'] <= validation['boot_input_bound']))
    state['pending'] = dict(iteration=iteration, previous=previous, reply=reply, source=source,
        fb=fb, usage=usage, row=row, new_llm_best=new_llm_best, tuned=None, tuned_reused_from=None,
        candidates=[(source, cost, validation, 'llm')] if fb.get('feasible') else [])
    if state['tuner'] and eligible and not fb.get('stop') and (
            state['tune_policy'] == 'all' or new_llm_best or not state['has_tuned']):
        state['has_tuned'] = True
        normalized = print_program(parse(source))
        if normalized in state['tune_cache']:
            cached, reused_from, tune_source = state['tune_cache'][normalized]
            _finish_tune(state, _read_tune_result(cached), trace_path,
                         reused_from=reused_from, tune_source=tune_source)
        else:
            state['phase'] = 'tune'
    else:
        _finish_iteration(state, trace_path)


def _read_tune_result(saved):
    from .tuner import TuneResult
    from .transformer_tuner import ConvergentTuneResult
    record = dict(saved)
    program = record.pop('program_text')
    result_type = ConvergentTuneResult if 'converged' in record else TuneResult
    return result_type(program=parse(program), **record)


def saved_tune_result(result):
    record = {key: value for key, value in asdict(result).items() if key != 'program'}
    return dict(record, program_text=print_program(result.program))


def agent_step(ctx, state, *, trace_path, client=None, reply=None, feedback=None, tune_result=None):
    """Run exactly one LLM/evaluate, tune, or confirm step; no decision in the caller."""
    started = time.monotonic()
    validate_tune_rule(getattr(ctx, 'prompt_version', 'r8-v4'), state.get('tune_rule', PRE_TUNE_RULE))
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profile_args = dict(engine_profile='tensor') if state['engine_profile'] == 'tensor' else {}
    if state['phase'] == 'evaluate':
        _evaluate_step(ctx, state, client, trace_path, reply=reply, feedback=feedback)
    elif state['phase'] == 'tune':
        pending = state['pending']
        parsed = parse(pending['source'])
        with tune_rule_scope(state.get('tune_rule', PRE_TUNE_RULE)):
            result = tune_result if tune_result is not None else tune(parsed, ctx,
                budget_s=state['tuner_budget_s'],
                max_sweeps=8 if convergent_rule(state.get('tune_rule')) else 1, **profile_args)
        state['tune_cache'][print_program(parsed)] = (
            saved_tune_result(result), pending['iteration'], pending['source'])
        if convergent_rule(state.get('tune_rule')) and getattr(result,'converged',False):
            state['tune_cache'][print_program(result.program)] = (
                saved_tune_result(result), pending['iteration'], pending['source'])
        _finish_tune(state, result, trace_path)
    elif state['phase'] == 'confirm':
        confirmation = (dict(selected=None, evals=[], deduplicated=dict(before=len(state['pool']), after=0),
                             duplicates=[]) if state['failed'] else
                        confirm_select([Candidate(**item) for item in state['pool']], ctx, **profile_args))
        selected = confirmation['selected']
        if selected is None:
            state.update(best_program=None, best_iteration=None, best=None, best_llm_lines=None)
        else:
            state.update(best_program=selected.source, best_iteration=selected.iteration)
            state['best'], state['best_llm_lines'] = state['records'][
                (selected.iteration, selected.origin, selected.source)]
        state['confirmation'] = dict(confirmation, selected=asdict(selected) if selected is not None else None)
        state['phase'] = 'done'
    else:
        raise ValueError('No pending Agent step: '+state['phase'])
    state['elapsed_s'] += time.monotonic()-started
    return state


def agent_result(state):
    if state['phase'] != 'done':
        raise ValueError('Agent confirmation has not finished')
    confirmation = dict(state['confirmation'])
    if confirmation['selected'] is not None:
        confirmation['selected'] = Candidate(**confirmation['selected'])
    best, best_program = state['best'], state['best_program']
    return dict(status='error' if state['failed'] else 'feasible' if confirmation['selected'] is not None else 'confirm_failed',
        best_program=best_program, best_iteration=state['best_iteration'],
        first_feasible_iteration=state['first_feasible'], best=best, best_source=best['source'] if best else None,
        llm_source_lines=state['best_llm_lines'],
        frozen_source_lines=sum(bool(line.strip()) for line in best_program.splitlines()) if best_program else None,
        iterations=len(state['history']), llm_calls=state['calls'], prompt_tokens=state['prompt_tokens'],
        completion_tokens=state['completion_tokens'], seconds=state['elapsed_s'], history=state['history'],
        error=state['failed'], confirm=confirmation, pool_size=len(state['pool']), tune_policy=state['tune_policy'],
        tune_rule=state.get('tune_rule', PRE_TUNE_RULE),
        **(dict(split_version=state['split_version']) if 'split_version' in state else {}))


def run_agent(ctx: AgentContext, client, *, max_iters: int = 6, trace_path: Path,
              tuner: bool = True, tuner_budget_s: float = 2400,
              tune_policy: str = 'new_best', engine_profile: str = 'mock',
              tune_rule: str | None = None) -> dict:
    tune_rule = validate_tune_rule(getattr(ctx, 'prompt_version', 'r8-v4'), tune_rule)
    if getattr(ctx, 'prompt_version', 'r8-v4') not in ('r8-v8','r8-v9','r8-v10'):
        return _run_agent_once(ctx, client, max_iters=max_iters, trace_path=trace_path, tuner=tuner,
            tuner_budget_s=tuner_budget_s, tune_policy=tune_policy, engine_profile=engine_profile)
    state = new_agent_state(ctx, max_iters=max_iters, tuner=tuner, tuner_budget_s=tuner_budget_s,
                            tune_policy=tune_policy, engine_profile=engine_profile, tune_rule=tune_rule)
    while state['phase'] != 'done':
        agent_step(ctx, state, trace_path=trace_path, client=client)
    return agent_result(state)
