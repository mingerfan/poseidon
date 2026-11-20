"""Typed partition and symbolic composition of the original24-layer Qwen source.

No runner, model loader, planner, native compiler or freeze decision lives here.
A future runner must first use qwen_prefill_compile.frozen_source/_load_program,
create each plan through the real planner, and independently verify/Count replay
it. Semantic object binding below records source identity without proving that
an arbitrary supplied plan came from that planner. Native artifact linkage and
native level/scale/context acceptance remain separate work.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import re

from ..dsl.ir import Program, ValueRef, WeightRef, canonical
from ..dsl.typecheck import check, referenced_weights
from . import qwen_staged_generation as staged


VARIANT = 'full24-two-token-explicit-layer-plan-reuse-v1'
LAYERS = 24
SLOTS = 16384
HIDDEN = (1, 32, 896)
VOCABULARY = 151936
VOCABULARY_BLOCKS = 19
EXPLICIT_BOUNDARY_BOOTSTRAPS = 2*LAYERS*32 + 2*VOCABULARY_BLOCKS + 3


@dataclass(frozen=True, eq=False)
class Segment:
    role: str
    index: int | None
    program: Program
    parent: Program
    original_op_ids: tuple
    local_to_original_ids: tuple
    aliases: tuple
    weight_shapes: tuple
    _source: dict


@dataclass(frozen=True, eq=False)
class LayeredProgram:
    original: Program
    layers: tuple
    head: Segment
    weight_shapes: tuple
    diagnostics: tuple


@dataclass(frozen=True, eq=False)
class BoundPlan:
    segment: Segment
    plan: dict
    source: dict
    scope: str = 'semantic-object-binding-only'
    planner_provenance_verified: bool = False
    independent_replay_verified: bool = False


def _rewrite(value, aliases):
    if isinstance(value, ValueRef):
        return ValueRef(aliases.get(value.name, value.name))
    if isinstance(value, dict):
        return {key: _rewrite(item, aliases) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_rewrite(item, aliases) for item in value)
    if isinstance(value, list):
        return [_rewrite(item, aliases) for item in value]
    return value


def _source_record(segment):
    """Keep original statement IDs/names, all arithmetic attrs and weight shapes."""
    def semantic(value):
        if isinstance(value, WeightRef):
            return dict(weight=value.name)
        if isinstance(value, ValueRef):
            return dict(value=value.name)
        if isinstance(value, dict):
            return {key: semantic(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [semantic(item) for item in value]
        return value
    return dict(format='qwen-layer-source-binding-v1', role=segment.role, index=segment.index,
        parent=canonical(segment.parent), program=canonical(segment.program),
        original_ops=[dict(id=op.id, kind=op.kind, inputs=list(op.inputs), outputs=list(op.outputs),
            attrs=semantic(op.attrs), lineno=op.lineno, region=op.region,
            macro_stack=list(op.macro_stack), call_lines=list(op.call_lines))
            for op in segment.parent.ops if op.id in segment.original_op_ids],
        local_to_original_ids=semantic(segment.local_to_original_ids), aliases=semantic(segment.aliases),
        weight_shapes=semantic(segment.weight_shapes))


def source_binding(segment):
    """Return a metadata snapshot; source mutation after partition is rejected."""
    if not isinstance(segment, Segment) or _source_record(segment) != segment._source:
        raise ValueError('The typed original segment source changed after partition')
    return deepcopy(segment._source)


def _typed(program, weight_shapes):
    result = check(program, weight_shapes)
    if not result.ok:
        raise ValueError('Partition source failed original typecheck: '+
                         '; '.join(d.msg for d in result.diags if d.severity == 'error'))
    return result


def split_program(program, weight_shapes):
    """Partition AST records only; preserve every original nonpack statement once.

    This is a typed structural check. It does not replace the original graph,
    stable-bridge, full-weight and independent C2 gates in the future runner.
    Each internal layer declares hidden, attention-mask and selector outputs;
    the two masks are aliases of its original inputs, with no new arithmetic.
    """
    if (not isinstance(program, Program) or program.header.slots != SLOTS
            or program.header.manage != 'compiler' or program.header.refresh != 'agent'
            or program.header.placement != 'program-plan'
            or program.header.l_max != 5 or program.header.boot_out_level != 5
            or [p.name for p in program.params] != ['inputs_embeds', 'attention_mask', 'sel']
            or [p.kind for p in program.params] != ['cipher', 'mask', 'mask']
            or [p.shape for p in program.params] != [HIDDEN, (1, 32), (1, 32)]
            or [op.id for op in program.ops] != list(range(len(program.ops)))):
        raise ValueError('Retain the original qwen48 full24 typed source signature and IDs')
    shapes = {name: tuple(shape) for name, shape in weight_shapes.items()}
    original_check = _typed(program, shapes)
    packs = program.ops[:3]
    if (len(packs) != 3 or any(op.kind != 'pack' or op.inputs != (param.name,)
            or len(op.outputs) != 1 for op, param in zip(packs, program.params))
            or [op.attrs.get('layout') for op in packs] != ['token', 'flat', 'flat']
            or any(op.kind == 'pack' for op in program.ops[3:])):
        raise ValueError('Retain the original three input packs and physical34-input layout')
    hidden, attention, selector = (op.outputs[0] for op in packs)
    groups, position = [], 3
    while position < len(program.ops):
        first = program.ops[position]
        if not first.macro_stack or first.macro_stack[0] != 'qwen_layer':
            break
        if not first.call_lines:
            raise ValueError('Retain each original qwen_layer source call position')
        key = first.call_lines[0]
        end = position
        while end < len(program.ops):
            op = program.ops[end]
            if not op.macro_stack or op.macro_stack[0] != 'qwen_layer' or op.call_lines[0] != key:
                break
            end += 1
        groups.append(program.ops[position:end])
        position = end
    tail = program.ops[position:]
    if (len(groups) != LAYERS or len({group[0].call_lines[0] for group in groups}) != LAYERS
            or [op.kind for op in tail] != ['rmsnorm', 'select', 'linear']
            or any(op.macro_stack for op in tail) or len(program.outputs) != 1
            or program.outputs != tail[-1].outputs):
        raise ValueError('Preserve24 contiguous original layer calls and the complete three-op head')

    def segment(role, index, body, input_hidden, outputs):
        aliases = {} if input_hidden == hidden else {input_hidden: hidden}
        originals = (*packs, *body)
        ops = tuple(replace(op, id=local, inputs=tuple(aliases.get(v, v) for v in op.inputs),
                            attrs=_rewrite(op.attrs, aliases)) for local, op in enumerate(originals))
        sliced = replace(program, name=program.name+'_'+role+('' if index is None else str(index)),
                         ops=ops, outputs=outputs, regions=dict(program.regions))
        checked = _typed(sliced, shapes)
        expected = HIDDEN if role == 'layer' else (1, 1, VOCABULARY)
        if checked.shapes[outputs[0]] != expected:
            raise ValueError('Original layer/head output shape differs from its physical interface')
        refs = referenced_weights(sliced)
        weights = tuple((name, shapes[name]) for name in sorted(refs))
        result = Segment(role, index, sliced, program, tuple(op.id for op in originals),
            tuple((local, op.id) for local, op in enumerate(originals)),
            tuple(aliases.items()), weights, {})
        object.__setattr__(result, '_source', _source_record(result))
        return result, tuple(checked.diags)

    layers, diagnostics, current = [], list(original_check.diags), hidden
    for index, body in enumerate(groups):
        produced = {name for op in body for name in op.outputs}
        external = {name for op in body for name in op.inputs} - produced
        names = referenced_weights(replace(program, ops=body))
        layer_ids = {int(match.group(1)) for name in names
                     if (match := re.search(r'^original\.model\.layers\.(\d+)\.', name))}
        if current not in external or not external <= {current, attention} or layer_ids != {index}:
            raise ValueError('Layer source must retain its own original weights and preceding hidden/mask inputs')
        last = body[-1].outputs
        if len(last) != 1:
            raise ValueError('Retain one original hidden tensor at each layer boundary')
        item, diags = segment('layer', index, body, current, (last[0], attention, selector))
        layers.append(item); diagnostics.extend(diags); current = last[0]
    if (tail[0].inputs != (current,) or tail[1].inputs != (tail[0].outputs[0], selector)
            or tail[2].inputs != tail[1].outputs
            or tail[0].attrs.get('w').name != 'original.model.norm.weight'
            or tail[2].attrs.get('w').name != 'original.model.embed_tokens.weight'):
        raise ValueError('Retain the original final norm, token selection and tied vocabulary projection')
    head, diags = segment('head', None, tail, current, program.outputs)
    diagnostics.extend(diags)
    covered = tuple(op_id for item in (*layers, head) for op_id in item.original_op_ids[3:])
    if covered != tuple(op.id for op in program.ops[3:]):
        raise ValueError('Every original arithmetic statement must be covered once in source order')
    used = set().union(*(referenced_weights(item.program) for item in (*layers, head)))
    if used != referenced_weights(program):
        raise ValueError('Partition changed the original referenced-weight inventory')
    return LayeredProgram(program, tuple(layers), head,
        tuple((name, shapes[name]) for name in sorted(used)), tuple(diagnostics))


def bind_plan(segment, refresh_plan, *, planned_source):
    """Bind exact source metadata, without claiming planner or replay provenance."""
    source = source_binding(segment)
    if planned_source != source:
        raise ValueError('Plan metadata must bind original op IDs, source arithmetic and weight shapes')
    if (not isinstance(refresh_plan, dict) or refresh_plan.get('format') != 'pt-agent-refresh-plan-v1'
            or refresh_plan.get('levels') != staged.LEVELS
            or any(type(refresh_plan.get(key)) is not int or refresh_plan[key] < 0 for key in ('calls', 'boots'))
            or any(not isinstance(refresh_plan.get(key), list) for key in ('substitutions', 'outputs', 'dead'))):
        raise ValueError('Require the unchanged explicit qwen48 segment refresh-plan schema')
    return BoundPlan(segment, deepcopy(refresh_plan), source)


@dataclass(frozen=True, eq=False)
class SegmentLowering:
    segment: Segment
    weights: object

    def __call__(self, backend):
        from .qwen_prefill_compile import _same_lowering
        source_binding(self.segment)
        for name, shape in self.segment.weight_shapes:
            if tuple(self.weights[name].shape) != shape:
                raise ValueError('The original weight shape changed before segment lowering')
        return _same_lowering(self.segment.program, self.weights, backend)


def callbacks(partition, weights):
    """Use the exact original weight mapping by reference, without array copies."""
    if not isinstance(partition, LayeredProgram):
        raise ValueError('Require the original typed layered partition')
    return tuple(SegmentLowering(item, weights) for item in partition.layers), SegmentLowering(partition.head, weights)


def _bound(callback, plan, *, role, index):
    if (not isinstance(callback, SegmentLowering) or not isinstance(plan, BoundPlan)
            or plan.segment is not callback.segment or plan.segment.role != role
            or plan.segment.index != index or plan.source != source_binding(plan.segment)):
        raise ValueError('Reuse the exact segment callback and its source-bound plan')
    return plan.plan


def export_layer(backend, *, callback, bound_plan):
    """Prepare one physical32-output layer artifact through original interfaces.

    Original Refresh still observes all34 logical hidden/mask outputs. The two
    exact original input mask handles travel through explicit manifest aliases;
    Base.output receives only32 computed, actually bootstrapped hidden values.
    This prepares trusted emission, with native ABI/levels/scales/context and
    independent freezing/planner provenance still unverified.
    """
    if (not isinstance(callback, SegmentLowering) or callback.segment.role != 'layer'
            or type(callback.segment.index) is not int or not 0 <= callback.segment.index < LAYERS
            or backend.slots != SLOTS):
        raise ValueError('Export one original typed Qwen layer with its16384-slot interface')
    plan = _bound(callback, bound_plan, role='layer', index=callback.segment.index)
    state = staged.Segments(backend)
    state.metadata.update(variant=VARIANT, scope='single-layer-trusted-export-preparation-only',
        source_binding=source_binding(callback.segment), layer_index=callback.segment.index,
        source_binding_scope='semantic-object-binding-only', planner_provenance_verified=False,
        independent_replay_verified=False, native_compiled_bootstraps=None,
        native_input_metadata=None, native_output_metadata=None, native_metadata_verified=False,
        native_levels_verified=False, native_scales_verified=False, native_context_verified=False,
        compile_verified=False, C2_verified=False, whole_model_verified=False,
        real_ckks_executions=0, intermediate_decryptions=0)
    try:
        output = state.run('layer'+str(callback.segment.index), state.inputs, plan, callback)
        if len(output) != staged.INPUTS:
            raise ValueError('Layer export retains all34 original logical hidden/mask outputs')
        aliases = []
        for position in (32, 33):
            captured, original = output[position], state.inputs[position]
            state._require(captured, level=5)
            state._require(original, level=5)
            if captured.value is not original.value:
                raise ValueError('Carry the same original level5 input mask handle through the manifest')
            aliases.append(dict(logical_output=position, input=position, level=5,
                original_handle=original.serial, captured_handle=captured.serial,
                provenance='exact-original-input-alias'))
        hidden = state.normalize(output[:32], 'layer-hidden-to-next-layer')
        state.finish(hidden)
        if state.metadata['boundary_bootstraps'] != 32:
            raise ValueError('Layer export requires32 actual hidden boundary bootstrap calls')
        state.metadata.update(status='ok', physical_inputs=staged.INPUTS, logical_output_blocks=staged.INPUTS,
            physical_output_blocks=32, mask_alias_outputs=aliases,
            logical_to_physical_outputs=[dict(logical_output=i, physical_output=i) for i in range(32)],
            symbolic_output_levels=[handle.level for handle in hidden],
            explicit_boundary_bootstrap_calls=state.metadata['boundary_bootstraps'],
            internal_plan_bootstrap_calls=state.metadata['segments'][0]['refresh']['boots'])
        return state
    except BaseException:
        state.metadata['status'] = 'error'
        raise


def run_layered(backend, *, layers, head, plans24, head_plan, first_kernel_plan,
                terminal_kernel_plan, original_embedding, prompt_length, coefficients=None):
    """Symbolic two-forward composition; actual native compilation is unverified.

    Internal mask outputs are captured by original Segments.run and retain the
    same raw input handles. Only32 hidden outputs per layer receive new real
    bootstrap calls. Each plan and lowering object is reused on the second
    forward. Final physical I/O remains34 inputs and38 onehot outputs.
    """
    if (len(layers) != LAYERS or len(plans24) != LAYERS
            or type(prompt_length) is not int or not 1 <= prompt_length <= 30
            or backend.slots != SLOTS or tuple(original_embedding.shape) != (VOCABULARY, HIDDEN[-1])):
        raise ValueError('Retain24 layers, the original vocabulary/packing and two reserved prefix positions')
    plans = tuple(_bound(cb, plan, role='layer', index=i) for i, (cb, plan) in enumerate(zip(layers, plans24)))
    head_raw = _bound(head, head_plan, role='head', index=None)
    parent = layers[0].segment.parent
    weights = layers[0].weights
    if (any(item.segment.parent is not parent or item.weights is not weights for item in (*layers, head))
            or original_embedding is not weights['original.model.embed_tokens.weight']):
        raise ValueError('All segments and the tied embedding retain the same original source and weight mapping')
    first, terminal = staged.kernel_callbacks(original_embedding, coefficients)
    state = staged.Segments(backend)
    state.metadata.update(variant=VARIANT, source_binding_scope='semantic-object-binding-only',
        planner_provenance_verified=False, independent_replay_verified=False,
        native_compiled_bootstraps=None, compile_verified=False, C2_verified=False,
        whole_model_verified=False, real_ckks_executions=0, intermediate_decryptions=0)

    def forward(step, inputs):
        current = list(inputs)
        for index, (callback, plan) in enumerate(zip(layers, plans)):
            output = state.run(f'forward{step}-layer{index}', current, plan, callback)
            if (len(output) != staged.INPUTS or any(output[32+i].value is not current[32+i].value
                    or output[32+i].level != 5 for i in range(2))):
                raise ValueError('Layer boundary must retain32 hidden blocks and the same two level5 mask handles')
            current = [*state.normalize(output[:32], f'forward{step}-layer{index}-hidden'), *current[32:]]
        logits = state.run(f'forward{step}-head', current, head_raw, head)
        if len(logits) != VOCABULARY_BLOCKS:
            raise ValueError('Retain all19 original vocabulary head blocks')
        return state.normalize(logits, f'forward{step}-head-to-selector')

    try:
        first_inputs = forward(0, state.inputs)
        selected0 = state.run('selector0-and-embedding', first_inputs, first_kernel_plan, first)
        if len(selected0) != VOCABULARY_BLOCKS+1:
            raise ValueError('First selection must retain its actual single embedding handle')
        handoff = state.normalize([selected0[-1], *state.masks(prompt_length+1)], 'generated-state-to-forward1')
        second_inputs = list(state.inputs)
        second_inputs[prompt_length], second_inputs[32], second_inputs[33] = handoff
        state.metadata['handoff'] = dict(source_embedding=selected0[-1].serial,
            normalized_embedding=handoff[0].serial, position=prompt_length,
            consumed_embedding=second_inputs[prompt_length] is handoff[0])
        final_inputs = forward(1, second_inputs)
        selected1 = state.run('selector1-terminal', final_inputs, terminal_kernel_plan, terminal)
        if len(selected1) != VOCABULARY_BLOCKS:
            raise ValueError('Terminal selection retains exactly19 vocabulary blocks')
        state.finish([*selected0[:VOCABULARY_BLOCKS], *selected1])
        if state.metadata['boundary_bootstraps'] != EXPLICIT_BOUNDARY_BOOTSTRAPS:
            raise ValueError('Explicit physical boundary bootstrap inventory differs')
        state.metadata.update(status='ok', scope='symbolic-layer-composition-only', layers_per_forward=LAYERS,
            forward_calls=2, physical_inputs=staged.INPUTS, vocabulary_blocks=VOCABULARY_BLOCKS,
            final_output_blocks=2*VOCABULARY_BLOCKS, consumed_embeddings=1,
            terminal_unused_embedding_eliminated=True, reused_layer_plans=LAYERS, reused_head_plans=1,
            explicit_boundary_bootstrap_calls=state.metadata['boundary_bootstraps'],
            internal_plan_bootstrap_calls=sum(row['refresh']['boots'] for row in state.metadata['segments']))
        return state
    except BaseException:
        state.metadata['status'] = 'error'
        raise
