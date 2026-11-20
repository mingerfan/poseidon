"""Resume the saved thirteenth Qwen reply after the recorded RSS guard stop.

The original failed case stays intact. A separate live session starts at its
last committed checkpoint; a local transaction supplies the retained Reply to
the original Agent step. Only later iterations14/15 may request the provider.
"""

import ast
import copy
import inspect
import json
import math
from pathlib import Path
import shutil
import textwrap

from . import qwen_bounded_feedback as bounded


original = bounded.original
bridge = bounded.bridge
MODULE = 'pt_agent.experiments.qwen_guarded_replay'
ROUND = 'autonomous-qwen-guarded-replay'
MODE = 'original-full-decoder-reply13-rss16-replay'
session_outcome = bounded.session_outcome
PENDING = 13
MAX_ATTEMPTS = 15
CLOCK_SCOPE = ('Local replay of retained reply13, copied23-prefix calibration, fresh same-split20framework/'
    '4canonical references and remaining original Agent steps; prior597.679s and original network clock excluded')


def read(path):
    return original.read_json(path)


def _same(value):
    return json.loads(json.dumps(value, allow_nan=False))


def limits(args):
    values = (args.memory_gib, args.rss_gib, args.minimum_available_gib)
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
            or values != (24, 16, 1.25) or type(args.case_gib) is not int or args.case_gib != 32
            or type(args.seconds) is not int or not 1 <= args.seconds <= 10800
            or args.server_mode or args.make_server_job is not None):
        raise ValueError('Use explicit local AS24/RSS16/host reserve1.25/case32 and finite wall<=10800')
    return dict(address_space_gib=24, subtree_rss_gib=16, minimum_available_gib=1.25,
                case_gib=32, file_gib=8, complete_wall_seconds=args.seconds,
                prior_rss_guard_gib=14, PSS_measured=False, success_guaranteed=False)


def _invalid_limits(args):
    try:
        limits(args)
        return False
    except ValueError:
        return True


def check_write_paths(args):
    bounded.check_write_paths(args)
    source = Path(args.partial_from).resolve()
    for name in ('out', 'work_dir', 'session', 'codegen', 'calibration'):
        value = getattr(args, name, None)
        if value is None:
            continue
        target = Path(value).resolve()
        if source == target or source.is_relative_to(target) or target.is_relative_to(source):
            raise ValueError('Guarded replay writes require a fresh case outside both inherited sources')


def inspect_partial(directory, *, source=None):
    """Read records and candidate syntax only; model/data/provider stay unopened."""
    from ..planners.transformer_session import load_state
    directory = Path(directory).resolve()
    record, complete = read(directory/'result.json'), read(directory/'completion.json')
    state = load_state(directory/'session/SESSION.json')
    if (record.get('round') != bounded.ROUND or record.get('mode') != bounded.MODE
            or record.get('model') != original.MODEL or record.get('status') != 'error'
            or record.get('candidate_frozen') is not False
            or record.get('termination', {}).get('cause') != 'own_budget_guard'
            or complete.get('status') != 'error' or complete.get('registry_persisted') is not True
            or complete.get('required_artifacts_persisted') is not True):
        raise ValueError('Require the closed original bounded Qwen resource-guard case')
    if source is not None and _same(record.get('source')) != _same(source):
        raise ValueError('Partial case must retain every original model/split/bridge/presentation source field')
    if (state.get('phase') != 'evaluate' or state.get('iteration') != 13 or state.get('calls') != 12
            or state.get('pending') is not None or state.get('failed') is not None
            or state.get('max_iters') != 15 or state.get('tuner') is not False
            or state.get('prompt_version') != 'r8-v11' or state.get('placement') != 'program-plan'
            or state.get('confirm_rule') != 'c2' or state.get('pool') or state.get('best') is not None
            or state.get('presentation_variant') != bounded.PRESENTATION_ID
            or state.get('method_guidance_version') != bounded.METHOD_GUIDANCE_VERSION
            or state.get('prompt_variant_id') != bridge.PROMPT_ID
            or state.get('algorithm_variant') != bridge.VARIANT
            or state.get('recovery', {}).get('inherited_attempts') != 9
            or [row.get('iteration') for row in state.get('history', [])] != list(range(1, 13))):
        raise ValueError('Require the exact uncommitted evaluate13/calls12/failedNone checkpoint')
    if [(row.get('iteration'), row.get('stage')) for row in state['history'][-3:]] != [(10, 'parse'), (11, 'parse'), (12, 'parse')]:
        raise ValueError('Retain the three completed new syntax rejections')
    session = directory/'session'
    if any((session/name).exists() for name in ('AGENT.json', 'FROZEN.json', 'program.ptdsl')):
        raise ValueError('The partial source must remain unfrozen and unconfirmed')
    requests = {p.name for p in (session/'calls').glob('call-*.request.json')}
    responses = {p.name for p in (session/'calls').glob('call-*.response.json')}
    if (requests != {f'call-{i:04d}.request.json' for i in range(1, 14)}
            or responses != {f'call-{i:04d}.response.json' for i in (*range(1, 9), *range(10, 14))}):
        raise ValueError('Preserve thirteen requests, twelve replies and the ninth local failure')
    request, reply = read(session/'calls/call-0013.request.json'), read(session/'calls/call-0013.response.json')
    if request.get('purpose') != 'agent-iter-13' or not isinstance(reply.get('content'), str):
        raise ValueError('Require the retained thirteenth provider Reply')
    pinned = {int(key): value for key, value in state['recovery']['pinned_error_counts'].items()}
    projected, _ = bounded.project_messages(state['messages'], pinned_errors=pinned)
    if request.get('messages') != projected:
        raise ValueError('Retained request13 must be the projection of the exact saved checkpoint')
    program = json.loads(reply['content'])['program']
    saved = (session/'sources/iteration-13.ptdsl').read_text()
    if (not isinstance(program, str) or not program.strip()
            or ast.dump(ast.parse(program), include_attributes=False) != ast.dump(ast.parse(saved), include_attributes=False)):
        raise ValueError('Retained source13 and original Reply must describe the same candidate')
    trace = [json.loads(line) for line in (session/'agent_trace.jsonl').read_text().splitlines() if line.strip()]
    if any(row.get('iteration') == 13 for row in trace) or (session/'full-feedback/iteration-0013.full.json').exists():
        raise ValueError('Iteration13 must have no committed feedback/trace yet')
    for name in ('calibration.json', 'unit-width-bridge.json', 'generation-input.json'):
        if not (directory/name).is_file():
            raise ValueError('Require the original23-prefix calibration artifacts')
    return dict(partial_case=str(directory), source_checkpoint=str(session/'SESSION.json'),
        pending_reply=str(session/'calls/call-0013.response.json'), pending_source=str(session/'sources/iteration-13.ptdsl'),
        inherited_registered_global_calls=record.get('api_registration', {}).get('registered'),
        committed_calls=12, pending_provider_attempt=13, pending_reply_already_paid=True,
        pending_replay_new_HTTP=0, maximum_cumulative_attempts=15, remaining_possible_HTTP=[14, 15],
        calibration_samples_reused=23, framework_reference_reconstruction=20, canonical_reference_reconstruction=4,
        cached_numeric_references_claimed=False, original_case_modified=False), state, reply


def seed_partial(args):
    from ..planners.transformer_session import save_state
    check_write_paths(args)
    session = Path(args.session)
    if (session/'guarded-replay.json').exists():
        return
    if session.exists() and any(session.iterdir()):
        raise ValueError('New live session must be empty before explicit migration')
    metadata, state, _ = inspect_partial(args.partial_from)
    case = Path(args.calibration).parent
    inherited = case/'inherited'
    inherited.mkdir(exist_ok=False)
    shutil.copytree(Path(args.partial_from)/'session', inherited/'session')
    for name in ('result.json', 'completion.json', 'guard-failure.json'):
        shutil.copy2(Path(args.partial_from)/name, inherited/name)
    session.mkdir(exist_ok=True)
    old = inherited/'session'
    for path in old.iterdir():
        if path.name in ('SESSION.json', 'sources'):
            continue
        if path.is_dir():
            shutil.copytree(path, session/path.name)
        else:
            shutil.copy2(path, session/path.name)
    (session/'sources').mkdir()
    for path in (old/'sources').iterdir():
        if path.suffix != '.ptdsl' or not path.stem.startswith('iteration-'):
            raise ValueError('Require explicit original source artifact names')
        number = int(path.stem.removeprefix('iteration-'))
        if not 1 <= number <= 13:
            raise ValueError('Uncommitted future sources must not enter replay')
        if number <= 12:
            shutil.copy2(path, session/'sources'/path.name)
    save_state(session/'SESSION.json', state)
    original.write_json(session/'guarded-replay.json', dict(metadata, status='pending_local_replay',
        inherited_session=str(inherited/'session'), authoritative_checkpoint='before-evaluate13',
        initial_prompt_tokens=state['prompt_tokens'], initial_completion_tokens=state['completion_tokens']))


def source_gate(args):
    check_write_paths(args)
    source = bounded.source_gate(args)
    metadata, _, _ = inspect_partial(args.partial_from, source=source)
    source.update(guarded_replay=metadata, resource_policy=limits(args),
                  guarded_replay_source=str(Path(__file__).resolve()))
    out = getattr(args, 'out', None)
    if out is not None and out.is_dir():
        shutil.copy2(__file__, out/'variant-source/qwen_guarded_replay.py')
    return source


def _calibrate(args, recorder, result):
    seed_partial(args)
    for name, target in (('calibration.json', args.calibration),
            ('unit-width-bridge.json', bridge._bridge_path(args)),
            ('generation-input.json', args.calibration.parent/'generation-input.json')):
        shutil.copy2(Path(args.partial_from)/name, target)
    result.update(calibration_completed=True, bridge_calibration_completed=True, calibration_samples=23,
        calibration_cache=dict(source=str(args.partial_from), fresh_calibration_forwards=0),
        fresh_calibration_forwards=0, calibration_forward_attempts=0, calibration_forward_completed=0,
        stable_softmax_attempts=0, stable_softmax_completed=0)


def _committed(state):
    return (state.get('calls', 0) >= 13 and state.get('iteration', 0) >= 14
        and sum(row.get('iteration') == 13 for row in state['history']) == 1 and state.get('pending') is None)


def replay_once(args, recorder, result):
    """Commit local13 atomically; a failed working state never replaces SESSION."""
    from ..planners.transformer_session import load_state, save_state
    marker_path = args.session/'guarded-replay.json'
    marker = read(marker_path)
    state_path = args.session/'SESSION.json'
    authoritative = load_state(state_path)
    if _committed(authoritative):
        # Handles the narrow crash window after atomic SESSION replacement.
        trace = [json.loads(line) for line in (args.session/'agent_trace.jsonl').read_text().splitlines() if line.strip()]
        if sum(row.get('iteration') == 13 for row in trace) != 1:
            raise ValueError('Committed replay requires one durable original iteration13trace')
        original.write_json(marker_path, dict(marker, status='committed', HTTP_added=0))
        return False
    if (marker.get('status') != 'pending_local_replay' or authoritative.get('phase') != 'evaluate'
            or authoritative.get('calls') != 12 or authoritative.get('iteration') != 13):
        raise ValueError('Unexpected local replay checkpoint; preserve the source for diagnosis')
    if (args.session/'sources/iteration-13.ptdsl').exists():
        raise ValueError('An earlier partial replay source exists; retain it and use an isolated new case')
    reply = read(Path(marker['inherited_session'])/'calls/call-0013.response.json')
    working = copy.deepcopy(authoritative)
    try:
        ctx = bridge._context(args, result)
        functions = bounded._agent_functions(args.session)
        result['reference_reconstruction'] = dict(framework=20, canonical=4,
            same_fixed_heldout_indices=True, old_numeric_cache_reused=False)
        with recorder.span('local_retained_reply13_evaluate'):
            functions['agent_step'](ctx, working, trace_path=args.session/'agent_trace.jsonl', client=None, reply=reply)
        usage = reply.get('usage') or {}
        if (not _committed(working) or working['calls'] != 13
                or working['prompt_tokens'] != authoritative['prompt_tokens'] + usage.get('prompt_tokens', 0)
                or working['completion_tokens'] != authoritative['completion_tokens'] + usage.get('completion_tokens', 0)):
            raise ValueError('Original reply13 must advance accounting exactly once')
        save_state(state_path, working)
        original.write_json(marker_path, dict(marker, status='committed', HTTP_added=0,
            committed_calls=13, next_iteration=working['iteration'], next_phase=working['phase']))
    except BaseException as exc:
        save_state(args.work_dir/'partial-local-evaluate-state.json', working)
        original.write_json(args.work_dir/'local-replay-error.json', dict(error=type(exc).__name__+': '+str(exc),
            authoritative_checkpoint_unchanged=load_state(state_path)['calls'] == 12,
            original_reply13_retained=True, new_HTTP=0, needs_new_isolated_attempt=True))
        raise
    outcome = bounded.session_outcome(args.session)
    result.update(api_calls=0, retained_reply_replays=1, retained_provider_attempt=13,
        new_provider_requests_for_13=0, session=outcome, candidate_frozen=False,
        step=dict(complete=True, step='evaluate', next_step=working['phase'], iteration=working['iteration'], session_calls=13))
    return True


def _agent_phase(args, recorder, result):
    seed_partial(args)
    marker = read(args.session/'guarded-replay.json')
    if args.worker_phase == 'prompt':
        bounded._agent_phase(args, recorder, result)
        result.update(api_required=False, retained_reply13_required=True)
        return
    if marker.get('status') != 'committed' and replay_once(args, recorder, result):
        return
    state = read(args.session/'SESSION.json')
    if state.get('calls', 0) < 13:
        raise ValueError('No new provider request before local13commit')
    bounded._agent_phase(args, recorder, result)


def phase_command(args, phase, work):
    functions = bridge._namespace()
    functions['MODULE'] = MODULE
    return functions['phase_command'](args, phase, work) + [
        '--resume-from', str(args.resume_from), '--partial-from', str(args.partial_from),
        '--rss-gib', str(args.rss_gib), '--minimum-available-gib', str(args.minimum_available_gib),
        '--seconds', str(args.seconds)]


def parser():
    value = bounded.parser()
    value.add_argument('--partial-from', type=Path, required=True)
    value.add_argument('--minimum-available-gib', type=float, default=1.25)
    value.set_defaults(memory_gib=24., rss_gib=16., case_gib=32, seconds=10800)
    for action in value._actions:
        if action.dest == 'through':
            action.choices = ('metadata', 'prompt', 'agent')
    return value


def _delegate(function, namespace, *, entry=False):
    """Edit only trusted resource admission/display predicates in a private AST."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    found, clock = 0, 0
    for node in ast.walk(tree):
        if entry and isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp):
            fields = {item.attr for item in ast.walk(node.test) if isinstance(item, ast.Attribute)
                      and isinstance(item.value, ast.Name) and item.value.id == 'args'}
            if {'memory_gib', 'rss_gib', 'case_gib', 'seconds', 'server_mode'} <= fields:
                node.test = ast.parse('_invalid_limits(args)', mode='eval').body
                node.body[0].value.args[0] = ast.Constant('Use guarded replay AS24/RSS16/reserve1.25/case32/wall<=10800')
                found += 1
        elif not entry and isinstance(node, ast.Compare):
            if ast.dump(node, include_attributes=False) == ast.dump(ast.parse('available < GIB', mode='eval').body, include_attributes=False):
                node.comparators[0] = ast.parse('args.minimum_available_gib * GIB', mode='eval').body
                found += 1
        if not entry and isinstance(node, ast.Constant) and node.value == bounded.OLD_CLOCK_SCOPE:
            node.value = CLOCK_SCOPE
            clock += 1
    if found != 1 or (not entry and clock != 1):
        raise ValueError('Original trusted resource entry/monitor source changed')
    ast.fix_missing_locations(tree)
    exec(compile(tree, inspect.getsourcefile(function) or __file__, 'exec'), namespace)
    return namespace[function.__name__]


def _namespace():
    functions = bridge._namespace()
    functions.update(__file__=__file__, __doc__=__doc__, MODULE=MODULE, ROUND=ROUND, MODE=MODE,
        source_gate=source_gate, _calibrate=_calibrate, _agent_phase=_agent_phase, _program=bounded._program,
        session_outcome=bounded.session_outcome, phase_command=phase_command, parser=parser,
        _invalid_limits=_invalid_limits)
    delegated_execute = _delegate(functions['execute'], functions)
    delegated_worker = functions['worker']
    def execute(args):
        limits(args)
        check_write_paths(args)
        return delegated_execute(args)
    def worker(args):
        limits(args)
        check_write_paths(args)
        return delegated_worker(args)
    functions.update(execute=execute, worker=worker)
    functions['main'] = _delegate(functions['main'], functions, entry=True)
    return functions


def main(argv=None):
    return _namespace()['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
