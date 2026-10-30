"""A1 §6.15 finite offline checks, with no report or provider access."""

import copy
import json

from ..run import _write
from ..dsl.parser import parse
from ..margin import check_margin
from ..planners.agent_tools import build_context
from ..planners.prompts import build_system_prompt, initial_message
from ..planners.transformer_session import json_safe, model_call_offset
from .a1_transformer_agent import read, samples_for


def context(args):
    from ..models.info import ModelInfo
    info = ModelInfo.load(args.model_dir)
    ctx = build_context(info, read(args.model_dir/'calibration.json'), samples=samples_for(args.model_dir, info))
    ctx.prompt_version = 'r8-v9'
    return ctx


def score_calibration_proof():
    import numpy as np
    from types import SimpleNamespace
    from ..analyzer import approximation_values
    raw = np.array([[[-2., 1., 4.]]])
    node = SimpleNamespace(op='softmax', inputs=['scores'], outputs=['_softmax'], attrs={})
    actual = approximation_values(SimpleNamespace(nodes=[node]), {'scores': raw})['_softmax::exp']
    equal = np.array_equal(actual, raw.reshape(1, -1))
    if not equal:
        raise RuntimeError('A1 §6.15 exp calibration is not raw scores')
    return dict(raw_scores=True, calibration_key='_softmax::exp', min=float(actual.min()),
                max=float(actual.max()), source='analyzer.approximation_values')


def audit(args):
    ctx = context(args)
    result = dict(complete=False, model=args.model, score_calibration=score_calibration_proof(),
                  baselines={}, replay=[], llm_calls=0, report_images=0)
    for name, path in [('R0', args.r0_program), ('R1', args.program)]:
        result['baselines'][name] = dict(program=str(path), **check_margin(parse(path.read_text()), ctx))
    number = 21
    for directory in args.legacy_sessions:
        rows = [json.loads(line) for line in (directory/'agent_trace.jsonl').read_text().splitlines()]
        for row in rows:
            number += 1
            path = directory/'sources'/f"iteration-{row['iteration']}.ptdsl"
            if path.exists():
                check = check_margin(parse(path.read_text()), ctx)
                result['replay'].append(dict(call=number, origin='llm', source=str(path), **check))
            else:
                result['replay'].append(dict(call=number, origin='llm', source=None,
                    old_stage=row['fb']['stage'], note='No DSL in the retained response'))
            tuned = row.get('tuned')
            if tuned is not None:
                check = check_margin(parse(tuned['source']), ctx)
                result['replay'].append(dict(call=number, origin='tuned',
                    source=str(directory/'agent_trace.jsonl'), **check))
        _write(args.out/'MARGIN_REPLAY.json', json_safe(result))
    if args.legacy_sessions and (number != 34 or args.model != 'bert'):
        raise ValueError('A1 §6.15 replay requires exactly BERT calls 22–34')
    result['complete'] = True
    result['development_only'] = True
    _write(args.out/'MARGIN_REPLAY.json', json_safe(result))
    return json_safe(result)


def requests(args):
    from ..llm import Config
    ctx = context(args)
    text = build_system_prompt(ctx)
    overlap = [name for name in ctx.weights if name in text]
    if overlap:
        raise ValueError('Target weight names found outside graph summary: '+repr(overlap))
    settings = Config(model='deepseek-flash', reasoning_effort='high', max_tokens=65536,
                      timeout_seconds=600).settings()
    body = dict(settings, messages=[dict(role='system', content=text),
                                  dict(role='user', content=initial_message(ctx))])
    ctx.prompt_version = 'r8-v8'
    before = dict(settings, messages=[dict(role='system', content=build_system_prompt(ctx)),
                                    dict(role='user', content=initial_message(ctx))])
    original = read(args.model_dir.parent/'first-request.pretty.json')
    summary = json.loads(body['messages'][1]['content'])
    domain_fields = []
    for node in summary['nodes']:
        for point in node.get('approximation_points', []):
            if 'min_domain' in point:
                domain_fields.append(dict(point=point['name'], min_domain=point.pop('min_domain')))
    start = text.index('calib = [lo, hi]')
    end = text.index(' Heldout samples judge feasibility', start)
    old_start = before['messages'][0]['content'].index('calib = [min, max]')
    old_end = before['messages'][0]['content'].index(' Heldout samples judge feasibility', old_start)
    restored = copy.deepcopy(body)
    restored['messages'][0]['content'] = text[:start]+before['messages'][0]['content'][old_start:old_end]+text[end:]
    restored['messages'][1]['content'] = json.dumps(summary, separators=(',', ':'))
    checks = dict(v8_unchanged=before == original, only_sentence_and_min_domain=restored == before,
                  min_domain_present=bool(domain_fields))
    _write(args.out/'first-request.pretty.json', body)
    _write(args.out/'REQUEST_DIFF.json', dict(checks=checks, old_sentence=before['messages'][0]['content'][old_start:old_end],
        new_sentence=text[start:end], min_domains=domain_fields))
    if not all(checks.values()):
        raise RuntimeError('A1 §6.15 request differs outside the allowed fields')
    return dict(complete=True, model=args.model, prompt_version='r8-v9', checks=checks,
        request=str(args.out/'first-request.pretty.json'), split_version='split-v2',
        endpoint='https://api.deepseek.com/chat/completions', max_tokens=65536,
        model_calls_used=model_call_offset(args.model, 'r8-v9'),
        remaining_model_calls=40-model_call_offset(args.model, 'r8-v9'),
        max_total_calls=450, global_start=200, llm_calls=0, report_images=0)
