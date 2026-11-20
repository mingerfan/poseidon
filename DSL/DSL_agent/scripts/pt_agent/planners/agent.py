"""Choose the best feasible program using only pre-freeze internal feedback."""

import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import time

from ..llm import LLMError
from ..quality import quality_fields
from ..dsl.parser import parse
from ..dsl.printer import print_program
from .agent_tools import AgentContext, evaluate, apply_placement
from .confirm import Candidate, confirm_select, confirmation_rule
from .prompts import build_system_prompt, feedback_message, initial_message
from .tuner import tune
from .transformer_tuner import (DEFAULT_TUNE_RULE, PRE_TUNE_RULE, CONVERGENT_TUNE_RULE, convergent_rule,
                               validate_tune_rule, tune_rule_scope)


NUMERIC_FIELDS = ('deg', 'deg0', 'iters', 'domain', 'degrees', 'inv_budget')
STRUCTURE_NOTE = ('The tuner sets degrees, iteration counts and domains. On every operation whose structure you keep, '
                  'the system replaces these settings in your program with the tuned values. Reduce latency by changing '
                  'structure: approximation methods, the decomposition of softmax and normalization, kernel algorithms '
                  'and layouts.')


def _numeric(key):
    """Tuner-owned settings: degrees, iteration counts, domains and the reciprocal budget, with any role prefix."""
    return key in NUMERIC_FIELDS or key.endswith(tuple('_'+field for field in NUMERIC_FIELDS))


def _structure(op):
    return op.kind, json.dumps({key: value for key, value in op.attrs.items() if not _numeric(key)},
                               sort_keys=True, default=str)


def _aligned_pairs(base, new):
    """Pairs of operations with the same structure, aligned in program order; value names play no part."""
    import difflib
    matcher = difflib.SequenceMatcher(None, [_structure(op) for op in base.ops], [_structure(op) for op in new.ops],
                                      autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            yield base.ops[block.a+offset], new.ops[block.b+offset]


def changed_lines(source, base_source):
    """Lines of `source` whose operation has no identical counterpart in `base_source`.

    Operations are aligned by kind and structural settings in program order, so renamed values do not count as
    changes. A focused retune moves only the approximation points on these lines or called from them; a program in
    which every operation differs yields every line, so the retune equals an unrestricted tune."""
    new, base = parse(source), parse(base_source)
    unchanged = {id(op) for old, op in _aligned_pairs(base, new) if old.attrs == op.attrs}
    lines = set()
    for op in new.ops:
        if id(op) not in unchanged:
            lines.add(op.lineno)
            lines.update(op.call_lines)
    return lines


def lock_numeric(source, base_source):
    """Copy the tuner-owned settings of `base_source` onto every operation of `source` whose structure is unchanged.

    Returns the resulting source and the number of operations whose settings were replaced. Operations whose
    method, decomposition, algorithm or layout changed keep the LLM's settings and are tuned afterwards."""
    new, base = parse(source), parse(base_source)
    locked = 0
    for old, op in _aligned_pairs(base, new):
        fields = {key for key in set(old.attrs) | set(op.attrs) if _numeric(key)}
        if any(old.attrs.get(key) != op.attrs.get(key) for key in fields):
            locked += 1
            for key in fields:
                if key in old.attrs:
                    op.attrs[key] = old.attrs[key]
                else:
                    op.attrs.pop(key, None)
    return (print_program(new) if locked else source), locked


def _add_intermediate_states(pool, records, result, iteration, source, *, serialized=False):
    lines = sum(bool(line.strip()) for line in source.splitlines())
    for item in getattr(result, 'accepted_states', []):
        origin = 'tuned-r'+str(item['round'])
        cost = item['cost']
        candidate = Candidate(item['source'], origin, iteration, cost['latency_pred_us'], cost['boots_pred'])
        pool.append(asdict(candidate) if serialized else candidate)
        records[(iteration, origin, candidate.source)] = (dict(cost, iteration=iteration, source=origin), lines)


def _selected_record(records, selected, confirmation):
    record, lines = records[(selected.iteration, selected.origin, selected.source)]
    if selected.origin.startswith('tuned-r'):
        record = dict(record)
        measured = next(row for row in confirmation['evals'] if row['iteration'] == selected.iteration
                        and row['origin'] == selected.origin)
        record.update({key: measured[key] for key in ('agree', 'relative_error', 'max_u', 'max_u_calibrated',
                       'max_u_analytic', 'worst_acts') if key in measured})
    return record, lines


def _run_agent_once(ctx: AgentContext, client, *, max_iters: int = 6, trace_path: Path,
              tuner: bool = True, tuner_budget_s: float = 2400,
              tune_policy: str = "new_best", engine_profile: str = "mock",
              tune_rule: str | None = None, confirm_rule: str = 'c1', stale_limit: int = 2,
              tune_sweeps: int | None = None, focus_retune: bool = False, pool_states: bool = False,
              numeric_lock: bool = False) -> dict:
    """`pool_states` adds every sweep's accepted tuner state to the confirmation pool, as rule c2 does where it
    applies, so a multi-sweep tune keeps its earlier states when the last one fails confirmation. `numeric_lock`
    gives every later reply the best program's tuned settings on the operations whose structure it keeps, and the
    feedback says so."""
    if stale_limit < 1 or (tune_sweeps is not None and tune_sweeps < 1):
        raise ValueError('stale_limit and tune_sweeps must be positive')
    requested_confirm_rule = confirm_rule
    confirm_rule = confirmation_rule(ctx, confirm_rule)
    tune_rule = validate_tune_rule(getattr(ctx, 'prompt_version', 'r8-v4'), tune_rule)
    if tune_policy not in ("new_best", "all"):
        raise ValueError('tune_policy must be "new_best" or "all"')
    if engine_profile not in ("mock", "tensor"):
        raise ValueError('engine_profile must be "mock" or "tensor"')
    if engine_profile == "tensor" and getattr(ctx, 'prompt_version', 'r8-v4') not in ('r8-v5','r8-v6','r8-v7','r8-v8','r8-v9','r8-v10','r8-v11'):
        raise ValueError('engine_profile="tensor" requires r8-v5, r8-v6, r8-v7 or r8-v8')
    v8 = getattr(ctx, 'prompt_version', 'r8-v4') in ('r8-v8','r8-v9','r8-v10','r8-v11')
    rule_record = dict(tune_rule=tune_rule) if v8 else {}
    rule_record['confirm_rule'] = confirm_rule
    if requested_confirm_rule != confirm_rule:
        rule_record['confirm_rule_applicability'] = 'not_applicable'
    if getattr(ctx, 'placement', None):
        rule_record['placement'] = ctx.placement
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
        locked_ops = None
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
            if getattr(ctx, 'placement', None) or getattr(ctx,'prompt_version','r8-v4') in ('r8-v7','r8-v8','r8-v9','r8-v10','r8-v11'):
                # Retain parsed response source before any internal tool runs.
                response_path=trace_path.parent/'sources'/f'iteration-{iteration}.ptdsl'
                response_path.parent.mkdir(parents=True,exist_ok=True)
                with response_path.open('x') as stream:stream.write(source)
            source = apply_placement(source, ctx)
            if numeric_lock and best_program is not None:
                source, locked_ops = lock_numeric(source, best_program)
            fb = evaluate(source, ctx, **(dict(engine="tensor", **profile_args) if profile_args else {}))
            if v8 and not source.strip() and reply.get('finish_reason') == 'length':
                fb = dict(stage='length', feasible=False, output_limit=65536,
                          error='The previous response reached the output limit (65536 tokens) before writing a program. Shorten the reasoning and output the complete program directly.')
        cost, validation = fb.get("cost") or {}, fb.get("validation") or {}
        row = dict(iteration=iteration, stage=fb["stage"], feasible=fb.get("feasible", False),
                            latency_pred_us=cost.get("latency_pred_us"), boots_pred=cost.get("boots_pred"),
                            **quality_fields(validation), max_u=validation.get("max_u"), tuned=None)
        if locked_ops is not None:
            row['locked_ops'] = locked_ops
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
        focus = None
        if tuner and eligible and not fb.get('stop') and (tune_policy == "all" or new_llm_best or not has_tuned):
            has_tuned = True
            parsed = parse(source)
            normalized = print_program(parsed)
            if normalized in tune_cache:
                result, tuned_reused_from, tune_source = tune_cache[normalized]
            else:
                if focus_retune and best_program is not None:
                    focus = changed_lines(source, best_program)
                with tune_rule_scope(tune_rule) if v8 else nullcontext():
                    result = tune(parsed, ctx, budget_s=tuner_budget_s,
                                  max_sweeps=tune_sweeps or (8 if convergent_rule(tune_rule) else 1), **profile_args,
                                  **({} if focus is None else dict(focus=focus)))
                tune_source = source
                tune_cache[normalized] = (result, iteration, tune_source)
                if convergent_rule(tune_rule) and getattr(result,'converged',False):
                    tune_cache[print_program(result.program)] = (result, iteration, tune_source)
            tuned = {key: value for key, value in asdict(result).items() if key != 'program'}
            if focus is not None:
                tuned['focus_lines'] = sorted(focus)
            # A reused result retains its original source locations and snippets.
            lines = tune_source.splitlines()
            for change in tuned['changes']:
                change['line_text'] = lines[change['line']-1].strip()
                change['call_texts'] = [lines[line-1].strip() for line in change['call_lines']]
            confirmation = result.history[-1]
            row['tuned'] = dict(feasible=result.feasible, latency_pred_us=result.cost_after['latency_pred_us'],
                                boots_pred=result.cost_after['boots_pred'], **quality_fields(confirmation),
                                max_u=confirmation['max_u'], coarse_evals=result.coarse_evals,
                                full_evals=result.full_evals, seconds=result.seconds)
            if v8:
                row['tuned'].update({key: confirmation[key] for key in ('max_u_calibrated','max_u_analytic')})
            if convergent_rule(tune_rule):
                row['tuned'].update(sweeps=result.sweeps,converged=result.converged)
            if focus is not None:
                row['tuned']['focus_lines'] = sorted(focus)
            if result.feasible:
                candidates.append((result.source, result.cost_after, confirmation, 'tuned'))
            if confirm_rule == 'c2' or pool_states:
                _add_intermediate_states(pool, records, result, iteration, source)
        row.update(has_tuned=has_tuned, new_llm_best=new_llm_best, llm_min=llm_min,
                   tuned_reused_from=tuned_reused_from)
        for candidate, candidate_cost, confirmed, origin in candidates:
            pool.append(Candidate(candidate, origin, iteration, candidate_cost['latency_pred_us'],
                                  candidate_cost['boots_pred']))
            records[(iteration, origin, candidate)] = (
                dict(candidate_cost, iteration=iteration, source=origin,
                     **quality_fields(confirmed), max_u=confirmed['max_u'],
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
                            **quality_fields(confirmed), max_u=confirmed['max_u'],
                            **({key:confirmed[key] for key in ('max_u_calibrated','max_u_analytic')} if v8 else {}),
                            boot_input_max_abs=confirmed.get('boot_input_max_abs'),
                            worst_acts=confirmed.get('worst_acts', []))
        history.append(row)
        token_agent=getattr(ctx,'prompt_version','r8-v4') in ('r8-v7','r8-v8','r8-v9','r8-v10','r8-v11')
        from .transformer_session import json_safe
        shown_fb,shown_tuned=(json_safe(fb),json_safe(tuned)) if token_agent else (fb,tuned)
        added = [dict(role="assistant", content=reply["content"]),
                 dict(role="user", content=feedback_message(shown_fb, iteration, best, tuned=shown_tuned)
                      + ('\n\n'+STRUCTURE_NOTE if numeric_lock else ''))]
        messages.extend(added)
        with trace_path.open("a") as stream:
            stream.write(json.dumps(dict(iteration=iteration, messages=added, fb=shown_fb, usage=usage,
                                         seconds=reply["seconds"], tuned=shown_tuned,
                                         tuned_reused_from=tuned_reused_from,
                                         **({} if locked_ops is None else dict(locked_ops=locked_ops)),
                                         tune_policy=tune_policy, **rule_record), allow_nan=False)+"\n")
        print(f"AGENT iteration={iteration} stage={fb['stage']} feasible={fb.get('feasible', False)}", flush=True)
        if v8 and fb.get('stop'):
            failed=fb['stop']
            break
        if best is not None:
            stale = 0 if previous is None or previous-best["latency_pred_us"] >= .05*previous else stale+1
            if stale >= stale_limit:
                break
    confirmation = (dict(selected=None,evals=[],deduplicated=dict(before=len(pool),after=0),duplicates=[])
                    if v8 and failed else confirm_select(pool, ctx, **profile_args,
                        **(dict(confirm_rule=confirm_rule) if confirm_rule != 'c1' else {})))
    selected = confirmation['selected']
    if selected is None:
        best_program = best_iteration = best = best_llm_lines = None
    else:
        best_program, best_iteration = selected.source, selected.iteration
        best, best_llm_lines = _selected_record(records, selected, confirmation)
    return dict(status="error" if failed else "feasible" if selected is not None else "confirm_failed",
                best_program=best_program, best_iteration=best_iteration, first_feasible_iteration=first_feasible,
                best=best, best_source=best['source'] if best else None, llm_source_lines=best_llm_lines,
                frozen_source_lines=sum(bool(line.strip()) for line in best_program.splitlines()) if best_program else None,
                iterations=len(history), llm_calls=calls, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                seconds=time.monotonic()-started, history=history, error=failed,
                confirm=confirmation, pool_size=len(pool), tune_policy=tune_policy, **rule_record)


def new_agent_state(ctx, *, max_iters=6, tuner=True, tuner_budget_s=2400,
                    tune_policy='new_best', engine_profile='mock', tune_rule=None, confirm_rule='c1',
                    stale_limit=2, focus_retune=False, numeric_lock=False):
    """The frozen loop's locals, with an explicit process boundary at each tool.

    `stale_limit` and `focus_retune` are stored only when they differ from the defaults, so states written by
    earlier rounds read the same."""
    if tune_policy not in ('new_best', 'all'):
        raise ValueError('tune_policy must be "new_best" or "all"')
    if stale_limit < 1:
        raise ValueError('stale_limit must be positive')
    if engine_profile not in ('mock', 'tensor'):
        raise ValueError('engine_profile must be "mock" or "tensor"')
    version = getattr(ctx, 'prompt_version', 'r8-v4')
    from .transformer_tuner import validate_tune_search
    search=validate_tune_search(ctx)
    tune_rule = validate_tune_rule(version, tune_rule)
    if version not in ('r8-v8','r8-v9','r8-v10','r8-v11'):
        raise ValueError('Resumable steps require r8-v8')
    split=getattr(getattr(ctx,'model_info',None),'split',{})
    effective_confirm_rule = confirmation_rule(ctx, confirm_rule)
    return dict(phase='evaluate', iteration=1, max_iters=max_iters, tuner=tuner,
        tuner_budget_s=tuner_budget_s, tune_policy=tune_policy, engine_profile=engine_profile,
        prompt_version=version, tune_rule=tune_rule, tune_search=search,
        confirm_rule=effective_confirm_rule,
        **(dict(confirm_rule_applicability='not_applicable') if effective_confirm_rule != confirm_rule else {}),
        **(dict(placement=ctx.placement) if getattr(ctx, 'placement', None) else {}),
        messages=[dict(role='system', content=build_system_prompt(ctx)),
                  dict(role='user', content=initial_message(ctx))],
        best_program=None, best_iteration=None, first_feasible=None, best=None,
        best_llm_lines=None, llm_min=None, has_tuned=False, tune_cache={},
        history=[], pool=[], records={}, calls=0, prompt_tokens=0,
        completion_tokens=0, stale=0, failed=None, elapsed_s=0., pending=None,
        **(dict(stale_limit=stale_limit) if stale_limit != 2 else {}),
        **(dict(focus_retune=True) if focus_retune else {}),
        **(dict(numeric_lock=True) if numeric_lock else {}),
        **(dict(model_call_offset=getattr(ctx, 'a1_model_call_offset', 0)) if version in ('r8-v9','r8-v10','r8-v11') else {}),
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
            **quality_fields(confirmed), max_u=confirmed['max_u'],
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
             dict(role='user', content=feedback_message(shown_fb, iteration, state['best'], tuned=shown_tuned)
                  + ('\n\n'+STRUCTURE_NOTE if state.get('numeric_lock') else ''))]
    state['messages'].extend(added)
    with Path(trace_path).open('a') as stream:
        stream.write(json.dumps(dict(iteration=iteration, messages=added, fb=shown_fb,
            usage=pending['usage'], seconds=reply['seconds'], tuned=shown_tuned,
            tuned_reused_from=pending['tuned_reused_from'], tune_policy=state['tune_policy'],
            **({} if 'locked_ops' not in row else dict(locked_ops=row['locked_ops'])),
            confirm_rule=state.get('confirm_rule', 'c1'),
            tune_rule=state.get('tune_rule', PRE_TUNE_RULE),
            **(dict(tune_search=state.get('tune_search','sweep')) if state['prompt_version']=='r8-v11' else {}),
            **(dict(placement=state['placement']) if state.get('placement') else {}),
            **(dict(prompt_version=state['prompt_version'],
                    model_call=state.get('model_call_offset', 0)+state['calls'])
               if state['prompt_version'] in ('r8-v9','r8-v10','r8-v11') else {}),
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
        state['phase'] = ('confirm' if state['stale'] >= state.get('stale_limit', 2) or iteration >= state['max_iters']
                          else 'evaluate')
    state['iteration'] = iteration+1
    state['pending'] = None


def _finish_tune(state, result, trace_path, *, reused_from=None, tune_source=None):
    pending = state['pending']
    tune_source = pending['source'] if tune_source is None else tune_source
    tuned = {key: value for key, value in asdict(result).items() if key != 'program'}
    if pending.get('focus') is not None and reused_from is None:
        tuned['focus_lines'] = list(pending['focus'])
    lines = tune_source.splitlines()
    for change in tuned['changes']:
        change['line_text'] = lines[change['line']-1].strip()
        change['call_texts'] = [lines[line-1].strip() for line in change['call_lines']]
    confirmation = result.history[-1]
    pending['row']['tuned'] = dict(feasible=result.feasible,
        latency_pred_us=result.cost_after['latency_pred_us'], boots_pred=result.cost_after['boots_pred'],
        **quality_fields(confirmation), max_u=confirmation['max_u'], coarse_evals=result.coarse_evals,
        full_evals=result.full_evals, seconds=result.seconds,
        **{key: confirmation[key] for key in ('max_u_calibrated', 'max_u_analytic')})
    if convergent_rule(state.get('tune_rule')):
        pending['row']['tuned'].update(sweeps=result.sweeps,converged=result.converged)
    if pending.get('focus') is not None and reused_from is None:
        pending['row']['tuned']['focus_lines'] = list(pending['focus'])
    if result.feasible:
        pending['candidates'].append((result.source, result.cost_after, confirmation, 'tuned'))
    if state.get('confirm_rule', 'c1') == 'c2':
        _add_intermediate_states(state['pool'], state['records'], result, pending['iteration'],
                                 pending['source'], serialized=True)
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
                confirm_rule=state.get('confirm_rule', 'c1'),
                tune_rule=state.get('tune_rule', PRE_TUNE_RULE),
                **(dict(tune_search=state.get('tune_search','sweep')) if state['prompt_version']=='r8-v11' else {}),
                **(dict(placement=state['placement']) if state.get('placement') else {}),
                **(dict(split_version=state['split_version']) if 'split_version' in state else {})))+'\n')
        state['phase'] = 'confirm'
        return
    usage = reply.get('usage') or {}
    state['prompt_tokens'] += usage.get('prompt_tokens', 0)
    state['completion_tokens'] += usage.get('completion_tokens', 0)
    source = None
    locked_ops = None
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
        source = apply_placement(source, ctx)
        if state.get('numeric_lock') and state['best_program'] is not None:
            source, locked_ops = lock_numeric(source, state['best_program'])
        profile_args = dict(engine_profile='tensor') if state['engine_profile'] == 'tensor' else {}
        fb = feedback if feedback is not None else evaluate(source, ctx,
            **(dict(engine='tensor', **profile_args) if profile_args else {}))
        if not source.strip() and reply.get('finish_reason') == 'length':
            fb = dict(stage='length', feasible=False, output_limit=65536,
                error='The previous response reached the output limit (65536 tokens) before writing a program. Shorten the reasoning and output the complete program directly.')
    cost, validation = fb.get('cost') or {}, fb.get('validation') or {}
    row = dict(iteration=iteration, stage=fb['stage'], feasible=fb.get('feasible', False),
        latency_pred_us=cost.get('latency_pred_us'), boots_pred=cost.get('boots_pred'),
        **quality_fields(validation), max_u=validation.get('max_u'), tuned=None)
    comparable = cost and (state.get('tune_rule', PRE_TUNE_RULE) == PRE_TUNE_RULE or fb.get('feasible', False))
    new_llm_best = bool(comparable and (state['llm_min'] is None or cost['latency_pred_us'] < state['llm_min']))
    if comparable:
        state['llm_min'] = (cost['latency_pred_us'] if state['llm_min'] is None
                            else min(state['llm_min'], cost['latency_pred_us']))
    eligible = (fb.get('exact', {}) and fb['exact']['ok'] and validation
                and not validation['diverged'] and fb['stage'] == 'complete'
                and (validation.get('boot_input_bound') is None
                     or validation['boot_input_max_abs'] <= validation['boot_input_bound']))
    if locked_ops is not None:
        row['locked_ops'] = locked_ops
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
            if state.get('focus_retune') and state['best_program'] is not None:
                state['pending']['focus'] = sorted(changed_lines(source, state['best_program']))
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


def agent_step(ctx, state, *, trace_path, client=None, reply=None, feedback=None, tune_result=None,tune_client=None):
    """Run exactly one LLM/evaluate, tune, or confirm step; no decision in the caller."""
    started = time.monotonic()
    from .transformer_tuner import validate_tune_search
    if validate_tune_search(ctx)!=state.get('tune_search','sweep'):
        raise ValueError('Session tune_search differs from context')
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
                max_sweeps=8 if convergent_rule(state.get('tune_rule')) else 1, **profile_args,
                **(dict(client=tune_client,iteration=pending['iteration'])
                   if state.get('tune_search')=='llm-select' else {}),
                **({} if pending.get('focus') is None else dict(focus=set(pending['focus']))))
        state['tune_cache'][print_program(parsed)] = (
            saved_tune_result(result), pending['iteration'], pending['source'])
        if convergent_rule(state.get('tune_rule')) and getattr(result,'converged',False):
            state['tune_cache'][print_program(result.program)] = (
                saved_tune_result(result), pending['iteration'], pending['source'])
        _finish_tune(state, result, trace_path)
    elif state['phase'] == 'confirm':
        confirmation = (dict(selected=None, evals=[], deduplicated=dict(before=len(state['pool']), after=0),
                             duplicates=[]) if state['failed'] else
                        confirm_select([Candidate(**item) for item in state['pool']], ctx, **profile_args,
                            **(dict(confirm_rule=state['confirm_rule']) if state.get('confirm_rule', 'c1') != 'c1' else {})))
        selected = confirmation['selected']
        if selected is None:
            state.update(best_program=None, best_iteration=None, best=None, best_llm_lines=None)
        else:
            state.update(best_program=selected.source, best_iteration=selected.iteration)
            state['best'], state['best_llm_lines'] = _selected_record(state['records'], selected, confirmation)
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
        confirm_rule=state.get('confirm_rule', 'c1'),
        **(dict(confirm_rule_applicability=state['confirm_rule_applicability']) if 'confirm_rule_applicability' in state else {}),
        tune_rule=state.get('tune_rule', PRE_TUNE_RULE),
        **(dict(tune_search=state.get('tune_search','sweep')) if state['prompt_version']=='r8-v11' else {}),
        **(dict(split_version=state['split_version']) if 'split_version' in state else {}))


def run_agent(ctx: AgentContext, client, *, max_iters: int = 6, trace_path: Path,
              tuner: bool = True, tuner_budget_s: float = 2400,
              tune_policy: str = 'new_best', engine_profile: str = 'mock',
              tune_rule: str | None = None, confirm_rule: str = 'c1', stale_limit: int = 2,
              tune_sweeps: int | None = None, focus_retune: bool = False, pool_states: bool = False,
              numeric_lock: bool = False) -> dict:
    from .transformer_session import validate_tune_search
    validate_tune_search(ctx)
    tune_rule = validate_tune_rule(getattr(ctx, 'prompt_version', 'r8-v4'), tune_rule)
    if getattr(ctx, 'prompt_version', 'r8-v4') not in ('r8-v8','r8-v9','r8-v10','r8-v11'):
        return _run_agent_once(ctx, client, max_iters=max_iters, trace_path=trace_path, tuner=tuner,
            tuner_budget_s=tuner_budget_s, tune_policy=tune_policy, engine_profile=engine_profile,
            confirm_rule=confirm_rule, stale_limit=stale_limit, tune_sweeps=tune_sweeps, focus_retune=focus_retune,
            pool_states=pool_states, numeric_lock=numeric_lock)
    if tune_sweeps is not None or pool_states:
        raise ValueError('tune_sweeps and pool_states apply to the r8-v4 loop; the resumable loop uses its rules')
    state = new_agent_state(ctx, max_iters=max_iters, tuner=tuner, tuner_budget_s=tuner_budget_s,
                            tune_policy=tune_policy, engine_profile=engine_profile, tune_rule=tune_rule,
                            confirm_rule=confirm_rule, stale_limit=stale_limit, focus_retune=focus_retune,
                            numeric_lock=numeric_lock)
    while state['phase'] != 'done':
        agent_step(ctx, state, trace_path=trace_path, client=client)
    return agent_result(state)
