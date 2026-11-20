"""Finite Qwen Agent variant using the measured public square-normalize family.

The original continuation remains the orchestration implementation. Its code
objects run in a private namespace with the explicitly named hooks below;
neither that module nor the planner/confirmation modules are modified. The
engine protocol remains r8-v11, while this prompt and algorithm have their own
identity. Original HF rounding supplies quality references. Canonical float64
references separately supply the independent arithmetic checks.
"""

import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import shutil
import types

from . import qwen_agent_continuation as original

MODULE = 'pt_agent.experiments.qwen_bridge_agent'
ROUND = 'autonomous-qwen-unit-width-agent'
MODE = 'original-full-decoder-unit-width-newton7-agent-no-tuner'
VARIANT = 'qwen-unit-width-newton7-no-tuner-v1'
PROMPT_ID = 'r8-v11-qwen-unit-width-newton7-no-tuner-v1'
MAX_STEPS, MAX_NEWTON = 11, 7
read_json, write_json = original.read_json, original.write_json


def _clone_namespace(module, replacements):
    """Delegate trusted functions without changing their module globals."""
    namespace = dict(vars(module))
    for name, function in vars(module).items():
        if inspect.isfunction(function) and function.__module__ == module.__name__:
            cloned = types.FunctionType(function.__code__, namespace, name,
                function.__defaults__, function.__closure__)
            cloned.__kwdefaults__ = copy.copy(function.__kwdefaults__)
            namespace[name] = cloned
    namespace.update(replacements)
    return namespace


def source_gate(args):
    source = original.source_gate(args)
    source.update(algorithm_variant=VARIANT, prompt_variant_id=PROMPT_ID,
        planner_engine_protocol='r8-v11', tuner_enabled=False,
        stable_decomposition='unit-width-public-square-v2', maximum_square_steps=MAX_STEPS,
        maximum_newton_iterations=MAX_NEWTON, raw_inverse_statistics_used=False,
        quality_reference='Original HF float32 RMSNorm/softmax rounding, with original float64 outputs',
        math_reference='Independent original canonical graph in float64',
        delegated_continuation_source=str(Path(original.__file__).resolve()),
        delegation='Original code objects in a private globals namespace; original defaults unchanged')
    # These snapshots are created within the existing outer timing/budget and
    # before its completion marker. The inner has no --out or results lookup.
    out = getattr(args, 'out', None)
    if out is not None and out.is_dir():
        directory = out/'variant-source'; directory.mkdir(exist_ok=True)
        root = Path(__file__).resolve().parents[1]
        for relative in ('experiments/qwen_bridge_agent.py', 'experiments/qwen_agent_continuation.py',
                         'intake/mapped_reference.py', 'intake/framework_precision.py',
                         'planners/precision_variants.py', 'planners/stable_softmax_bridge.py',
                         'kernels/unit_width_softmax_reference.py', 'transformer_tensor.py'):
            shutil.copyfile(root/relative, directory/relative.replace('/', '_'))
    return source


def _numpy():
    import numpy as np
    return np


def _plain(value):
    if isinstance(value, dict): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(item) for item in value]
    return value


class References:
    """Read-only original weights; framework and canonical arithmetic separate."""
    def __init__(self, info):
        from ..intake.framework_precision import expand_reference_precision
        from ..intake.mapped_reference import MappedReference
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2Attention
        if len(info.graph.weights) != 559:
            raise ValueError('Retain all559 original saved weight tensors')
        policies = {node.id: 'torch.float32' for node in info.graph.nodes
                    if node.op in ('rmsnorm', 'softmax')}
        evidence = []
        for function in (Qwen2RMSNorm.forward, Qwen2Attention.forward):
            lines, line = inspect.getsourcelines(function)
            if 'torch.float32' not in ''.join(lines):
                raise ValueError('Original framework intermediate rounding source differs')
            evidence.append(dict(function=function.__qualname__, path=inspect.getsourcefile(function),
                                 first_line=line, intermediate_dtype='torch.float32'))
        self.framework = MappedReference(expand_reference_precision(info.graph, policies))
        self.canonical = MappedReference(info.graph)
        self.evidence = evidence
        self.counts = dict(framework_attempts=0, framework_completed=0,
                           canonical_attempts=0, canonical_completed=0)
        if any(item.observation['private_weight_copies'] != 0 for item in (self.framework, self.canonical)):
            raise ValueError('The full original model must retain its read-only mapped weights')

    def forward(self, arrays, *, framework=True, keep=False):
        np = _numpy()
        name = 'framework' if framework else 'canonical'
        self.counts[name+'_attempts'] += 1
        values = (self.framework if framework else self.canonical).forward(arrays, keep=keep)
        converted = {key: value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
                     for key, value in values.items()}
        self.counts[name+'_completed'] += 1
        return converted


def _stat(stats, name, value):
    """The existing finite original observation collector, unchanged."""
    from .qwen_reference_worker import _stats_update
    _stats_update(stats, name, value, _numpy())


def collect_points(graph, values, stats, raw_issues, *, sample_id):
    """Collect original points; raw exp-sums remain diagnostics, never domains."""
    np = _numpy()
    for name, tensor in graph.tensors.items():
        if tensor.kind == 'act': _stat(stats, name, values[name])
    for node in graph.nodes:
        x = np.asarray(values[node.inputs[0]])
        if node.op in ('rmsnorm', 'layernorm'):
            centered = x if node.op == 'rmsnorm' else x-x.mean(-1, keepdims=True)
            _stat(stats, node.outputs[0]+'::rsqrt',
                  (centered*centered).mean(-1, keepdims=True)+node.attrs['eps'])
        elif node.op == 'softmax':
            keep = np.ones(x.shape, dtype=bool)
            if node.attrs['keep'] == 'causal':
                keep &= np.arange(x.shape[-1])[None, :] <= np.arange(x.shape[-2])[:, None]
            if node.attrs.get('mask') is not None:
                mask = np.asarray(values[node.attrs['mask']])
                if not np.isin(mask, (0., 1.)).all(): raise ValueError('Original binary mask differs')
                keep &= mask.reshape((len(x),)+(1,)*(x.ndim-2)+(x.shape[-1],)).astype(bool)
            if not keep.any(-1).all(): raise ValueError('Every attention query requires an effective key')
            _stat(stats, node.outputs[0]+'::exp', x[keep])
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                raw = (np.exp(x)*keep).sum(-1, keepdims=True)
            raw_issues.append(dict(sample_id=sample_id, point=node.outputs[0]+'::inv',
                status='unused_raw_observation', all_finite=bool(np.isfinite(raw).all()),
                elements=int(raw.size), finite_elements=int(np.isfinite(raw).sum()),
                used_for_fit_or_domain=False,
                reason='The selected kernel consumes the shifted/scaled exponential sum; raw sum is diagnostic'))


def _bridge_path(args):
    return args.calibration.parent/'unit-width-bridge.json'


def _calibrate(args, recorder, result):
    np = _numpy()
    from ..planners.precision_variants import build_unit_width_bridge
    from ..planners.stable_softmax_bridge import observe_softmax
    from ..kernels.unit_width_softmax_reference import fixed_shift_squaring
    info, samples = original._load_model(args, result)
    arrays, _ = samples.get('calibration')
    if len(arrays[info.graph.inputs[0]]) != 23: raise ValueError('Exactly original23 calibration prefixes required')
    references = References(info); stats = {}; issues = []; cache = []
    result['reference_counts'] = references.counts
    directory = args.work_dir/'original-score-cache'; directory.mkdir()
    result.update(algorithm_variant=VARIANT, prompt_variant_id=PROMPT_ID, tuner_enabled=False,
        calibration_forward_attempts=0, calibration_forward_completed=0,
        stable_softmax_attempts=0, stable_softmax_completed=0,
        reference_precision_source=references.evidence, raw_inverse_statistics_used=False,
        bridge_calibration_completed=False)
    softmax = [node for node in info.graph.nodes if node.op == 'softmax']
    for index in range(23):
        result['calibration_forward_attempts'] += 1
        write_json(args.work_dir/'worker-result.json', result)
        sample_id = info.split['sample_ids'][info.split['calibration'][index]]
        one = {name: value[index:index+1] for name, value in arrays.items()}
        with recorder.span('original_mapped_training_prefix', sample_id=sample_id):
            values = references.forward(one, framework=True, keep=True)
            collect_points(info.graph, values, stats, issues, sample_id=sample_id)
            for node in softmax:
                prefix = str(index).zfill(2)+'-'+node.id
                score = directory/(prefix+'-scores.npy'); mask = directory/(prefix+'-mask.npy')
                np.save(score, values[node.inputs[0]], allow_pickle=False)
                if node.attrs.get('mask') is not None:
                    np.save(mask, values[node.attrs['mask']], allow_pickle=False)
                cache.append(dict(sample_id=sample_id, node_id=node.id, scores=str(score),
                                  mask=str(mask) if node.attrs.get('mask') is not None else None))
        del values
        result['calibration_forward_completed'] += 1
        write_json(args.work_dir/'worker-result.json', result)
    bridge = build_unit_width_bridge(info.graph, stats, raw_issues=issues)
    rows = {row['node_id']: row for row in bridge['rows']}
    result['stable_softmax_checks'] = []
    for record in cache:
        result['stable_softmax_attempts'] += 1
        write_json(args.work_dir/'worker-result.json', result)
        with recorder.span('original_score_stable_math', sample_id=record['sample_id'], node=record['node_id']):
            _, checked = observe_softmax(np.load(record['scores'], mmap_mode='r', allow_pickle=False),
                np.load(record['mask'], mmap_mode='r', allow_pickle=False) if record['mask'] else None,
                rows[record['node_id']], reference=fixed_shift_squaring)
        result['stable_softmax_checks'].append(dict(checked, sample_id=record['sample_id']))
        result['stable_softmax_completed'] += 1
    if result['stable_softmax_completed'] != 23*24:
        raise ValueError('All original23 prefixes and24 attention nodes require independent stable math')
    bridge.update(actual_score_coverage_measured=True, real_forward_states=23,
        real_training_trajectories=23, stable_float64_math_passed=True,
        source=dict(model_info=str(args.model_info), inputs=str(args.inputs),
                    calibration_indices=list(info.split['calibration'])),
        raw_unshifted_inverse_used=False, agent_variant=VARIANT)
    write_json(args.calibration, stats); write_json(_bridge_path(args), bridge)
    lengths = arrays['attention_mask'].sum(1).astype(int)
    eligible = [i for i, length in enumerate(lengths) if 1 <= length <= 30]
    if not eligible: raise ValueError('No original calibration prefix has room for two generated tokens')
    index = info.split['calibration'][eligible[0]]
    generation = dict(index=index, sample_id=info.split['sample_ids'][index], partition='calibration',
        prompt_length=int(lengths[eligible[0]]), sequence_length=32,
        selection_rule='First stored calibration prompt with two free token positions')
    write_json(args.calibration.parent/'generation-input.json', generation)
    result.update(calibration_completed=True, bridge_calibration_completed=True, calibration_samples=23,
        calibration_points=len(stats), raw_inverse_diagnostics=issues, bridge_path=str(_bridge_path(args)),
        score_cache_metadata=dict(arrays=len(cache), bytes=sum(Path(row['scores']).stat().st_size for row in cache)),
        generation_input=generation, reference_ownership=dict(framework=references.framework.observation,
                                                           canonical=references.canonical.observation))


def checked_bridge(info, calibration, record, args):
    from ..planners.precision_variants import build_unit_width_bridge
    expected = build_unit_width_bridge(info.graph, calibration, raw_issues=record.get('raw_issues', ()))
    if any(node.outputs[0]+'::inv' in calibration for node in info.graph.nodes if node.op == 'softmax'):
        raise ValueError('Raw inverse statistics do not belong to this calibration namespace')
    source = record.get('source') or {}
    if (record.get('agent_variant') != VARIANT or record.get('actual_score_coverage_measured') is not True
            or record.get('stable_float64_math_passed') is not True or record.get('real_forward_states') != 23
            or not original._same(source.get('model_info', ''), args.model_info)
            or not original._same(source.get('inputs', ''), args.inputs)
            or source.get('calibration_indices') != info.split['calibration']
            or _plain(record.get('rows')) != _plain(expected['rows'])):
        raise ValueError('Use the same newly measured23-prefix bridge and original model/input sources')
    if not any(row['policy']['steps'] > 0 for row in record['rows']):
        raise ValueError('This variant requires the existing staged exact observer path')
    return record


def _reference_logits(references, graph, arrays, *, framework):
    np = _numpy(); values = []
    for index in range(len(arrays[graph.inputs[0]])):
        one = {key: value[index:index+1] for key, value in arrays.items()}
        output = references.forward(one, framework=framework)[graph.outputs[0]]
        if output.dtype != np.float64 or not np.isfinite(output).all():
            raise ValueError('Original reference logits require finite float64 values')
        values.append(output.copy())
    return np.concatenate(values)


def _attach(context, source):
    for name in ('stable_bridge', 'reference_owner', 'prompt_variant_id', 'algorithm_variant'):
        setattr(context, name, getattr(source, name))
    return context


def partition_context(ctx, *, splits):
    """Confirmation and C2 tail retain original HF references, without report."""
    np = _numpy()
    if tuple(splits) not in (('confirm',), ('calibration', 'heldout', 'confirm')):
        raise ValueError('Only original confirm or C2 non-report partitions are allowed')
    parts = [ctx.samples.get(name) for name in splits]
    if any(labels is not None for _, labels in parts): raise ValueError('Original Qwen inputs are unlabelled')
    arrays = {key: np.concatenate([value[key] for value, _ in parts]) for key in ctx.graph.inputs}
    exact = _reference_logits(ctx.reference_owner, ctx.graph, arrays, framework=True)
    new = _attach(replace(ctx, val_images=arrays[ctx.graph.inputs[0]], val_inputs=arrays,
        val_labels=None, val_exact_top1=None, val_exact_logits=exact,
        val_split='confirm' if tuple(splits) == ('confirm',) else 'non_report'), ctx)
    new.math_exact_logits = None  # Confirmation evaluates quality; no mathematical oracle is fabricated.
    if len(splits) > 1:
        new.tail_sample_positions = [dict(partition=name, index=index)
            for name, (values, _) in zip(splits, parts) for index in range(len(values[ctx.graph.inputs[0]]))]
    return new


def _context(args, result):
    from ..analyzer import approximation_points
    from ..planners.agent_tools import AgentContext
    info, samples = original._load_model(args, result)
    calibration = read_json(args.calibration)
    bridge = checked_bridge(info, calibration, read_json(_bridge_path(args)), args)
    references = References(info); result['reference_counts'] = references.counts
    arrays, labels = samples.get('heldout', 20)
    if labels is not None: raise ValueError('Original Qwen inputs are unlabelled')
    exact = _reference_logits(references, info.graph, arrays, framework=True)
    probe = {key: value[:4] for key, value in arrays.items()}
    mathematical = _reference_logits(references, info.graph, probe, framework=False)
    names = [key for _, key in approximation_points(info.graph)]
    ctx = AgentContext(graph=info.graph, weights=info.graph.weights, calibration=calibration,
        weight_shapes={key: value.shape for key, value in info.graph.weights.items()},
        val_images=arrays[info.graph.inputs[0]], val_exact_logits=exact, val_labels=None, val_exact_top1=None,
        probe_images=probe[info.graph.inputs[0]], probe_act_inputs=[None]*len(names),
        needed_margin=dict.fromkeys(names, 0.), act_tensors=names, model_name=info.graph.name,
        normalization=None, model_info=info, samples=samples, val_inputs=arrays, val_split='heldout',
        prompt_version='r8-v11', placement='program-plan', tune_search='sweep')
    ctx.stable_bridge, ctx.reference_owner, ctx.math_exact_logits = bridge, references, mathematical
    ctx.prompt_variant_id, ctx.algorithm_variant = PROMPT_ID, VARIANT
    result.update(algorithm_variant=VARIANT, prompt_variant_id=PROMPT_ID, tuner_enabled=False,
                  reference_precision_source=references.evidence,
                  quality_reference='Original framework rounding', math_reference='Canonical float64 original graph',
                  heldout_samples=20, report_samples=0)
    return ctx


def prompt_initial(ctx):
    from ..planners.prompts import initial_message
    class PointMetadata(dict):
        def __missing__(self, key):
            if key.endswith('::inv'):
                return dict(status='raw_statistic_unused', domain_source='analytical selected-kernel sum',
                            raw_unshifted_exp_sum_used=False)
            raise KeyError(key)
    proxy = copy.copy(ctx); proxy.calibration = PointMetadata(ctx.calibration)
    message = json.loads(initial_message(proxy))
    message.update(prompt_variant_id=PROMPT_ID, algorithm_variant=VARIANT, tuner_enabled=False,
                   softmax_policy=ctx.stable_bridge['rows'], maximum_newton_iterations=MAX_NEWTON,
                   original_partition_counts=original.PARTITIONS)
    message['reference_policy'] = dict(quality='Original HF intermediate rounding',
                                      arithmetic='Independent canonical float64 graph; strict1e-12+1e-12*abs(ref)')
    return json.dumps(message, separators=(',', ':'), allow_nan=False)


def prompt_system(ctx):
    from ..planners.prompts import build_system_prompt
    from ..capabilities import V7_CAPABILITIES
    text = build_system_prompt(ctx)
    old = next(row.constraints[0] for row in V7_CAPABILITIES if row.op == 'softmax')
    if text.count(old) != 1: raise ValueError('Original softmax capability source changed')
    text = text.replace(old,
        'This explicit unit-width-public-square-v2 variant uses each supplied softmax_policy row. '
        'Preserve its shift, inv_scale and steps exactly; steps is0..11. Apply '
        'exp to (scores-shift)*2**(-steps), and inv to sum(exp*mask)/inv_scale. '
        'Normalize by inv_scale, then perform the declared square-normalize rounds. '
        'Declare each approximation, including the shared step_inv. Initial inverse '
        'and step inverse domains must contain the registered analytical bounds and '
        'the supplied fit-error envelope. Raw unshifted exp-sums are unavailable diagnostics. '
        'Approximation coefficients and methods remain explicit. Exp target is1e-3/2**steps; '
        'inverse relative target is1e-3. The existing degree ladder <=63 and explicit Newton '
        'iterations0..7 are available. Domains derive only from original23 training prefixes. '
        'Heldout20 judges quality and never rebuilds calibration domains.')
    old_tuner = ('After each feasible program, a deterministic tuner lowers activation degrees and tightens domains one at a time and reports the settings it kept ("tuned"). Concentrate on structural decisions (approximation method, layout and algorithms, macro structure); you may adopt the tuned settings.')
    if text.count(old_tuner) != 1: raise ValueError('Original tuner prompt source changed')
    text = text.replace(old_tuner,
        'This named variant disables the tuner. Supply every numerical approximation setting. '
        'Internal parse/typecheck, independent mathematical checks, Count and heldout validation '
        'provide feedback. Confirmation occurs after generation; report samples are unavailable.')
    return 'Prompt variant: '+PROMPT_ID+'; engine protocol r8-v11.\n'+text


def candidate_gate(program, ctx):
    nodes = [op for op in program.ops if op.kind == 'softmax']
    rows = ctx.stable_bridge['rows']
    if len(nodes) != len(rows): raise ValueError('Keep every original attention softmax')
    for op, row in zip(nodes, rows):
        for key in ('shift', 'inv_scale', 'steps'):
            if op.attrs.get(key) != row['policy'][key]:
                raise ValueError('Preserve the measured unit-width public '+key+' at '+str(op.id))
    from ..transformer_validation import points
    for point in points(program):
        if point.attrs.get('method') == 'newton':
            iterations = point.attrs.get('iters')
            if type(iterations) is not int or not 0 <= iterations <= MAX_NEWTON:
                raise ValueError('The named variant permits integer Newton iterations0..7')


def strict_math(reference, observed):
    np = _numpy(); a, b = np.asarray(reference), np.asarray(observed)
    if a.dtype != np.float64 or b.dtype != np.float64 or a.shape != b.shape or not a.size:
        raise ValueError('Independent math requires matching original float64 shapes')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Independent math requires finite values')
    error = np.abs(a-b); tolerance = 1e-12+1e-12*np.abs(a)
    return dict(ok=bool(np.all(error <= tolerance)), finite=True, elements=int(a.size),
        criterion='1e-12+1e-12*abs(reference)', max_absolute_difference=float(error.max()),
        max_tolerance_ratio=float((error/tolerance).max()))


def exact_check(program, ctx, *, engine='tensor'):
    from ..transformer_validation import exact_check as legacy_check, execute
    candidate_gate(program, ctx)
    proxy = copy.copy(ctx); proxy.val_exact_logits = ctx.math_exact_logits
    checked = legacy_check(program, proxy, engine=engine)
    arrays = {key: value[:4] for key, value in ctx.val_inputs.items()}
    observed = execute(program, ctx.weights, arrays, engine=engine, exact=True)
    strict = strict_math(ctx.math_exact_logits, observed)
    return dict(checked, ok=bool(checked['ok'] and strict['ok']), independent_float64_math=strict,
                quality_reference_separate=True)


def _agent_functions():
    from ..planners import agent, agent_tools, confirm
    from ..dsl.parser import parse
    from ..dsl.errors import PTDSLError
    tools = _clone_namespace(agent_tools, {'_exact_check': exact_check})
    original_evaluate = tools['evaluate']
    def evaluate(source, ctx, **kwargs):
        try:
            candidate_gate(parse(source), ctx)
        except PTDSLError:
            return original_evaluate(source, ctx, **kwargs)
        except ValueError as exc:
            return dict(stage='stable_bridge', error=str(exc), feasible=False)
        return original_evaluate(source, ctx, **kwargs)
    confirmations = _clone_namespace(confirm, {
        'build_confirm_context': lambda ctx: partition_context(ctx, splits=('confirm',)),
        'build_non_report_context': lambda ctx, **kwargs: partition_context(ctx, splits=('calibration','heldout','confirm'))})
    # Recheck the same public attributes at every quality execution, including C2.
    old_validate = confirm.validate
    def validate(program, ctx, **kwargs):
        candidate_gate(program, ctx)
        return old_validate(program, ctx, **kwargs)
    confirmations['validate'] = validate
    return _clone_namespace(agent, dict(evaluate=evaluate, initial_message=prompt_initial,
        build_system_prompt=prompt_system, confirm_select=confirmations['confirm_select']))


def session_outcome(directory):
    outcome = original.session_outcome(directory)
    path = Path(directory)/'SESSION.json'
    if path.exists():
        record = read_json(path)
        if record.get('prompt_variant_id') != PROMPT_ID or record.get('tuner') is not False:
            raise ValueError('A named no-tuner bridge session must not resume an original raw/default session')
    return dict(outcome, prompt_variant_id=PROMPT_ID, tuner_enabled=False)


def _agent_phase(args, recorder, result):
    from ..planners.transformer_session import load_state, save_state, write_session_result, output_tokens
    from ..autonomous_budget import audit_public_messages
    ctx = _context(args, result); functions = _agent_functions()
    state_path = args.session/'SESSION.json'; args.session.mkdir(exist_ok=True)
    if state_path.exists():
        session_outcome(args.session); state = load_state(state_path)
    else:
        state = functions['new_agent_state'](ctx, max_iters=10, tuner=False, engine_profile='tensor',
            tune_rule='v8-6.25', confirm_rule='c2', tune_policy='all', stale_limit=2,
            focus_retune=False, numeric_lock=False)
        state.update(prompt_variant_id=PROMPT_ID, algorithm_variant=VARIANT)
        save_state(state_path, state)
    if args.worker_phase == 'prompt':
        messages = state['messages']
        write_json(args.work_dir/'public-api-request.json', dict(
            endpoint='https://api.deepseek.com/chat/completions', prompt_variant_id=PROMPT_ID,
            messages=messages, message_audit=audit_public_messages(messages),
            raw_sentences_included=False, raw_weights_included=False, report_samples=0,
            provenance='Authorized DSL instructions, original public architecture and finite training statistics; analytical inverse policies explicitly separate'))
        result.update(prompt_completed=True, api_required=state['phase']=='evaluate', session=session_outcome(args.session))
        return
    if state['phase'] == 'done':
        outcome = session_outcome(args.session)
        result.update(session=outcome, status=outcome['status'], candidate_frozen=outcome['frozen']); return
    client = None; before = state['phase']
    if before == 'evaluate':
        from ..autonomous_budget import RenewingLLMClient
        from .bert_sst2_worker import RecordedRenewingClient
        client = RecordedRenewingClient(RenewingLLMClient(budget_path=args.ledger,
            allowance_path=args.allowance, recorder=recorder, artifact_dir=args.session/'api',
            model='deepseek-flash', reasoning_effort='high', max_tokens=output_tokens('r8-v11'), timeout=600),
            args.session/'calls', recorder=recorder)
    with recorder.span('bridge_agent_'+before):
        functions['agent_step'](ctx, state, trace_path=args.session/'agent_trace.jsonl', client=client)
        save_state(state_path, state)
        if state['phase'] == 'done':
            write_session_result(functions['agent_result'](state), args.session,
                                 prompt_version='r8-v11', tune_policy=state['tune_policy'])
    outcome = session_outcome(args.session)
    result.update(step=dict(complete=True, step=before, next_step=state['phase'], iteration=state['iteration'],
                           session_calls=state['calls']), api_calls=client.calls if client else 0,
                  session=outcome, candidate_frozen=outcome['frozen'])
    if outcome['done']: result['status'] = outcome['status']


def _program(args, result):
    delegated = _clone_namespace(original, {'session_outcome': session_outcome})
    info, program = delegated['_program'](args, result)
    calibration = read_json(args.calibration)
    bridge = checked_bridge(info, calibration, read_json(_bridge_path(args)), args)
    candidate_gate(program, types.SimpleNamespace(stable_bridge=bridge))
    return info, program


def _namespace():
    return _clone_namespace(original, dict(__file__=__file__, __doc__=__doc__, MODULE=MODULE,
        ROUND=ROUND, MODE=MODE, source_gate=source_gate, _calibrate=_calibrate,
        _context=_context, _agent_phase=_agent_phase, _program=_program, session_outcome=session_outcome))


def main(argv=None):
    return _namespace()['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
