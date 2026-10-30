"""A1 §6.18 offline retrospective evaluation; this module has no provider client."""

import copy
import json
from pathlib import Path

from .. import domain_criteria as criteria
from ..dsl.parser import parse
from ..metrics import passes
from ..planners.agent_tools import build_context
from ..planners.prompts import (build_system_prompt, initial_message,
                                LINEAR_U_EXPLANATION, LOG_U_EXPLANATION)
from ..planners.transformer_session import json_safe
from ..run import _write
from ..transformer_validation import points
from .a1_transformer_agent import read, samples_for


LABEL = '事后重评：r8-v10 判据，Agent 当时收到的是 r8-v9 反馈'


def context(model_dir):
    from ..models.info import ModelInfo
    info = ModelInfo.load(model_dir)
    ctx = build_context(info, read(model_dir/'calibration.json'), samples=samples_for(model_dir, info))
    ctx.prompt_version = 'r8-v10'
    return ctx


def requests(args):
    from ..llm import Config
    ctx = context(args.model_dir)
    settings = Config(model='deepseek-flash', reasoning_effort='high', max_tokens=65536,
                      timeout_seconds=600).settings()
    after = dict(settings, messages=[dict(role='system', content=build_system_prompt(ctx)),
                                   dict(role='user', content=initial_message(ctx))])
    ctx.prompt_version = 'r8-v9'
    before = dict(settings, messages=[dict(role='system', content=build_system_prompt(ctx)),
                                    dict(role='user', content=initial_message(ctx))])
    restored = copy.deepcopy(after)
    restored['messages'][0]['content'] = restored['messages'][0]['content'].replace(
        LOG_U_EXPLANATION, LINEAR_U_EXPLANATION)
    checks = dict(v9_request_unchanged=before == read(args.response_file),
                  only_u_explanation=restored == before,
                  new_explanation_count=after['messages'][0]['content'].count(LOG_U_EXPLANATION) == 1,
                  graph_unchanged=after['messages'][1] == before['messages'][1])
    _write(args.out/'first-request.pretty.json', after)
    _write(args.out/'REQUEST_DIFF.json', dict(checks=checks, old=LINEAR_U_EXPLANATION,
        new=LOG_U_EXPLANATION, original_request=str(args.response_file)))
    if not all(checks.values()):
        raise RuntimeError('r8-v10 first request differs outside the u explanation')
    return dict(complete=True, model=args.model, prompt_version='r8-v10', llm_calls=0,
                report_images=0, checks=checks, request=str(args.out/'first-request.pretty.json'))


def regrade_validation(saved, program, ctx):
    """Recompute coverage from saved extrema, preserving all other measurements."""
    result = copy.deepcopy(saved)
    descriptors = points(program)
    if len(result.get('rows', [])) != len(descriptors):
        raise ValueError('Saved validation lacks complete approximation rows')
    for row, point, origin in zip(result['rows'], descriptors, criteria.provenance(program, ctx, descriptors)):
        if row.get('obs_min') is None or row.get('obs_max') is None:
            raise ValueError('Saved validation lacks approximation extrema')
        if tuple(row['domain']) != tuple(point.attrs['domain']) or row['program_line'] != point.op.lineno:
            raise ValueError('Saved validation belongs to a different program')
        row.update(origin)
        row['max_u'] = criteria.maximum_u([row['obs_min'], row['obs_max']], row['domain'],
            fn=point.attrs.get('fn'), domain_source=row['domain_source'], version=ctx.prompt_version)
    result.update(criteria.summarize(result['rows']))
    # Per-sample first-violation attribution is unchanged: log and linear use
    # the same positive interval boundaries; retain the original budget chain.
    if 'analytic_domain_violation' in saved:
        result['analytic_domain_violation'] = saved['analytic_domain_violation']
    result['max_u'] = max((r['max_u'] for r in result['rows']), default=0.)
    result['worst_acts'] = sorted(result['rows'], key=lambda r:(r['max_u'] <= r['u_limit'],-r['max_u']))[:5]
    result['feasible'] = (not result['diverged'] and criteria.domain_ok(result)
                          and passes(result, ctx.model_info.threshold))
    return result


def _saved_validation(manifest, source):
    for entry in manifest.get('saved_validations', []):
        if Path(entry['source']).read_text() != source:
            continue
        value = read(Path(entry['report']))
        for key in entry['keys']:
            value = value[key]
        if value is not None and value.get('rows'):
            return value, dict(report=entry['report'], keys=entry['keys'])
    return None, None


def _coordinate_summary(validation):
    if not validation:
        return None
    rows = validation.get('rows', validation.get('worst_acts', []))
    deciding = max(rows, key=lambda r:r['max_u']/r.get('u_limit', .9), default=None)
    return dict(max_u=validation.get('max_u'), max_u_calibrated=validation.get('max_u_calibrated'),
                max_u_analytic=validation.get('max_u_analytic'), deciding_point=deciding)


def regrade(args):
    from ..transformer_validation import validate
    manifest = read(args.response_file)
    ctx = context(args.model_dir)
    result = dict(complete=False, label=LABEL, model=args.model, llm_calls=0,
                  report_images=0, cases=[], baselines=[], original_manifest=str(args.response_file))
    for session in manifest['sessions']:
        directory = Path(session['path'])
        trace = [json.loads(line) for line in (directory/'agent_trace.jsonl').read_text().splitlines()]
        for row in trace:
            call = session['offset']+row['iteration']
            old_fb = row['fb']
            fb = copy.deepcopy(old_fb)
            source_path = directory/'sources'/f"iteration-{row['iteration']}.ptdsl"
            source = source_path.read_text() if source_path.exists() else None
            old_validation, proof = None, None
            mode = 'retained_other_rejection'
            if old_fb['stage'] == 'complete' and old_fb.get('validation') is not None:
                program = parse(source)
                old_validation, proof = _saved_validation(manifest, source)
                if old_validation is not None:
                    measured = regrade_validation(old_validation, program, ctx)
                    mode = 'saved_extrema'
                else:
                    captured = []
                    measured = validate(program, ctx, n=20, engine='tensor', output_callback=captured.append)
                    import numpy as np
                    if ctx.model_info.task == 'classification':
                        measured['logit_rel_err'] = float(np.max(np.abs(captured[0]-ctx.val_exact_logits))
                            /np.max(np.abs(ctx.val_exact_logits)))
                    mode = 'tensor_heldout20'
                criteria.require_analytic_coverage(measured)
                fb['validation'] = {k:v for k,v in measured.items() if k not in ('rows','feasible','error')}
                fb['feasible'] = bool(measured['feasible'] and old_fb.get('exact',{}).get('ok'))
                if not passes(old_fb['validation'],ctx.model_info.threshold) or old_fb['validation']['diverged']:
                    fb['feasible'] = False
                _write(args.out/f'call-{call:03d}-validation.json', json_safe(measured))
            else:
                measured = old_fb.get('validation')
            feedback_path = args.out/f'call-{call:03d}-feedback.json'
            _write(feedback_path, json_safe(fb))
            item = dict(call=call, session=str(directory), iteration=row['iteration'], source=str(source_path)
                if source is not None else None, mode=mode, saved_validation=proof,
                old_stage=old_fb['stage'], new_stage=fb['stage'], old_feasible=old_fb.get('feasible',False),
                new_feasible=fb.get('feasible',False), old=_coordinate_summary(old_validation or old_fb.get('validation')),
                new=_coordinate_summary(measured), feedback=str(feedback_path))
            result['cases'].append(item)
            print('REGRADE',args.model,call,mode,item['old_feasible'],item['new_feasible'],flush=True)
            _write(args.out/'REEVALUATION.json',json_safe(result))
    for entry in manifest['baselines']:
        source = Path(entry['source']).read_text()
        saved, proof = _saved_validation(manifest, source)
        if saved is None:
            raise ValueError('Expected retained baseline heldout rows')
        measured = regrade_validation(saved,parse(source),ctx)
        criteria.require_analytic_coverage(measured)
        _write(args.out/(entry['label']+'-validation.json'),json_safe(measured))
        result['baselines'].append(dict(label=entry['label'],source=entry['source'],saved_validation=proof,
            old_feasible=saved['feasible'],new_feasible=measured['feasible'],
            old=_coordinate_summary(saved),new=_coordinate_summary(measured)))
    result.update(complete=True,first_feasible_call=next((r['call'] for r in result['cases'] if r['new_feasible']),None))
    _write(args.out/'REEVALUATION.json',json_safe(result))
    return json_safe(result)


def replay_step(args):
    from ..planners.agent import new_agent_state, agent_step, agent_result
    from ..planners.transformer_session import load_state, save_state, write_session_result
    from ..cost import release_estimate_caches
    ctx = context(args.model_dir)
    original = Path(args.resume_session)
    review = read(args.response_file)
    cases = [row for row in review['cases'] if row['session'] == str(original)]
    if not cases:
        raise ValueError('Original session has no reevaluation records')
    directory = args.root
    directory.mkdir(parents=True,exist_ok=True)
    checkpoint = directory/'SESSION.json'
    state = load_state(checkpoint) if checkpoint.exists() else new_agent_state(ctx,max_iters=len(cases),
        engine_profile='tensor',tune_rule='v8-6.17')
    state.update(retrospective=True,label=LABEL,original_session=str(original))
    state['model_call_offset'] = cases[0]['call']-1
    phase = state['phase']
    if phase == 'done':
        raise ValueError('Retrospective session already complete')
    kwargs = {}
    if phase == 'evaluate':
        old = [json.loads(line) for line in (original/'agent_trace.jsonl').read_text().splitlines()]
        old = old[state['iteration']-1]
        response = dict(content=old['messages'][0]['content'],seconds=old['seconds'],usage=old['usage'],
                        finish_reason='length' if old['fb']['stage']=='length' else 'stop')
        kwargs = dict(reply=response,feedback=read(Path(cases[state['iteration']-1]['feedback'])))
    agent_step(ctx,state,trace_path=directory/'agent_trace.jsonl',**kwargs)
    save_state(checkpoint,state)
    release_estimate_caches()
    if phase == 'tune':
        import resource
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 3.8*1024**3:
            raise RuntimeError('A1 §6.18 tuning peak exceeds 3.8 GiB')
    record = None
    if state['phase']=='done':
        result = dict(agent_result(state),label=LABEL,retrospective=True,api_calls=0,
                      original_session=str(original))
        record = write_session_result(result,directory,prompt_version='r8-v10')
        frozen = directory/'FROZEN.json'
        if frozen.exists():
            _write(frozen,dict(read(frozen),label=LABEL,retrospective=True,api_calls=0,
                              original_session=str(original)))
    return dict(complete=True,label=LABEL,llm_calls=0,report_images=0,step=phase,
                next_step=state['phase'],iteration=state['iteration'],replayed_calls=state['calls'],
                original_session=str(original),state=str(checkpoint),agent=record,
                tune_rule='v8-6.17',retrospective=True)


def baseline_reuse(args):
    """Retain completed native proofs; coverage alone changes retrospectively."""
    ctx = context(args.model_dir)
    program = parse(args.program.read_text())
    original = args.resume_session
    result = dict(complete=False,label=LABEL,llm_calls=0,report_images=100,
                  program=str(args.program),reused_from=str(original),parts={})
    for part in ('report','cost','compiler','preflight','managed'):
        path = original/part/'REPORT.json'
        saved = read(path)
        if not saved['complete'] or Path(saved['program']).read_text() != args.program.read_text():
            raise ValueError('Completed native proof must use the same unchanged baseline')
        current = copy.deepcopy(saved)
        if part == 'report':
            for engine, measured in saved['engines'].items():
                current['engines'][engine] = regrade_validation(measured, program, ctx)
        current.update(label=LABEL,prompt_version='r8-v10',reused_from=str(path),
                       no_new_execution=True,program=str(args.program))
        (args.root/part).mkdir(parents=True,exist_ok=True)
        _write(args.root/part/'REPORT.json',json_safe(current))
        result['parts'][part] = str(args.root/part/'REPORT.json')
    result['complete'] = True
    return result


def depths(args):
    """The §4 local inverse-depth report, using the unchanged Count kernels."""
    from ..backends import CountBackend
    from ..kernels import transformer as kernel
    program = parse(args.program.read_text())
    rows = []
    for point in points(program):
        if point.op.kind != 'softmax' or point.prefix not in ('inv','step_inv','final_inv'):
            continue
        backend = CountBackend(program.header.slots)
        value = kernel.apply(backend,backend.input(0),point.attrs)
        rows.append(dict(line=point.op.lineno,role=point.role,method=point.attrs['method'],
            fit=point.attrs.get('fit','absolute'),domain=point.attrs['domain'],depth=value.depth,
            counts=backend.stats()['counts']))
    return dict(complete=True,label=LABEL,llm_calls=0,report_images=0,
                program=str(args.program),softmax_depths=rows)


def observe_rejection(args):
    """Fill absent extrema for a retained numerical rejection, without selection."""
    from ..transformer_validation import validate
    measured = validate(parse(args.program.read_text()),context(args.model_dir),n=20,
                        engine='tensor',stop_on_divergence=False)
    iteration = int(args.program.stem.rsplit('-',1)[1])
    retained = next(json.loads(line)['fb'] for line in
                    (args.resume_session/'agent_trace.jsonl').read_text().splitlines()
                    if json.loads(line)['iteration'] == iteration)
    _write(args.out/'OBSERVATION.json',json_safe(dict(validation=measured,original=retained)))
    if measured['feasible'] or measured['n_diverged'] != retained['validation']['n_diverged']:
        raise RuntimeError('Retained numerical rejection changed unexpectedly')
    criteria.require_analytic_coverage(measured)
    return json_safe(dict(complete=True,label=LABEL,llm_calls=0,report_images=0,affects_selection=False,
                program=str(args.program),validation=measured,
                original_rejection_retained=True))
