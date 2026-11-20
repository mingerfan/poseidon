"""Private component program-plan cost policy; no runner or frozen admission.

Each original layer/head owns a separate original planner lifetime. Explicit
boundary boots are measured by the existing complete export Count callback.
The normalised planner inventory and raw Count replay inventory stay separate.
"""
from collections import Counter
from contextlib import contextmanager
import gc
import math
from pathlib import Path
import weakref

POLICY = 'qwen-component-program-plan-retained13-v1'


def _write_count_error(directory, name, exc):
    """A failed replay diagnostic, distinct from the three success artifacts."""
    from .r10_paper_metrics import write_json
    try:
        write_json(Path(directory)/(name+'-count-error.json'), dict(
            status='error', planned_verified=True, complete_count=False,
            error=type(exc).__name__, component_call=getattr(exc, 'component_call', None)))
    except InterruptedError:
        raise
    except OSError:
        pass


@contextmanager
def _dead_count_scope(active):
    """Keep raw Count calls, with original Refresh dead-call level semantics.

    Deadness is the original finalized DAG's plan.dead. Only the raw Count
    level metadata is disabled for such arithmetic; live calls and all boots
    retain physical level checks. No operand, depth, count or result is edited.
    """
    from ..backends.count import CountBackend
    from ..backends.refresh import RefreshBackend
    if not isinstance(active, RefreshBackend) or not isinstance(active.inner, CountBackend):
        yield
        return
    inner = active.inner
    if active.policy != 'plan' or active._plan is None:
        raise ValueError('Complete component Count requires the original plan policy')
    plan = active._plan
    if (type(plan.calls) is not int or plan.calls < 0 or type(plan.dead) is not set
            or any(type(seq) is not int or not 0 <= seq < plan.calls for seq in plan.dead)):
        raise ValueError('Original plan dead calls must be valid call indices')
    if getattr(inner, '_component_dead_count_scope', False):
        raise RuntimeError('Nested component Count adaptation is unsupported')
    missing = object()
    prior = inner.__dict__.get('_apply', missing)
    original = type(inner)._apply if prior is missing else prior
    owner, refresh = weakref.ref(inner), weakref.ref(active)

    def apply(kind, left, right=None, *, step=None):
        backend, current = owner(), refresh()
        seq = current._seq
        dead = seq in current._plan.dead
        saved_level = backend.input_level
        try:
            if dead and kind != 'boot':
                backend.input_level = None
            if prior is missing:
                return original(backend, kind, left, right, step=step)
            return original(kind, left, right, step=step)
        except BaseException as exc:
            scalar = lambda value: value if type(value) is int else None
            context = dict(seq=seq, plan_dead=dead, kind=kind,
                raw_counter_total=backend._counter.total,
                op_context=scalar(getattr(backend, 'op_context', None)),
                cipher_operand_levels=[scalar(value.level) for value in (left, right)
                                       if hasattr(value, 'level')])
            try:
                exc.component_call = context
            except (AttributeError, TypeError):
                pass
            raise
        finally:
            backend.input_level = saved_level

    inner._component_dead_count_scope = True
    inner._apply = apply
    try:
        yield
    finally:
        if prior is missing:
            inner.__dict__.pop('_apply', None)
        else:
            inner._apply = prior
        del inner._component_dead_count_scope


def already_scoped_lowering(callback):
    """Adapt an original callback when its caller owns the Segments scope.

    This callback retains original source/weight checks and the original active
    Refresh backend. It must run inside Segments.run's projection scope.
    """
    from . import qwen_layered_structure as layered
    if not isinstance(callback, layered.SegmentLowering):
        raise TypeError('Require the original SegmentLowering callback')
    class AlreadyScopedLowering(layered.SegmentLowering):
        def __call__(self, active):
            layered.source_binding(self.segment)
            for name, shape in self.segment.weight_shapes:
                if tuple(self.weights[name].shape) != shape:
                    raise ValueError('The original weight shape changed before segment lowering')
            from ..lower import lower
            with _dead_count_scope(active):
                return lower(self.segment.program,self.weights,active,act_mode='program')
    return AlreadyScopedLowering(callback.segment,callback.weights)


class OriginalComponents:
    """Imported only inside the worker's existing metadata-command context."""
    def __init__(self, profile, directory, diagnostics):
        from .. import cost
        from . import qwen_layered_structure as layered
        from . import qwen_layer_native_probe as probe
        from . import qwen_remaining_prefill_compile as remaining
        from . import qwen_agent_cost_lifetime as lifetime
        self.cost, self.layered, self.probe = cost, layered, probe
        self.remaining, self.lifetime = remaining, lifetime
        self.profile, self.directory, self.diagnostics = profile, directory, diagnostics
        self.boot_us = cost.load_latency(profile)['bootstrap_single'][cost.PROFILES[profile]['boot_in_level']]

    def split(self, program, weights):
        partition = self.layered.split_program(program, {k:v.shape for k,v in weights.items()})
        layers, head = self.layered.callbacks(partition, weights)
        return (*layers, head)

    def measure(self, callback):
        from ..backends.cost_dag import CostDag
        from ..backends.streaming_count import StreamingCountBackend
        from ..kernels.compact_linear import compact_transformer_projections
        from ..kernels.layout import TokenLayout, multi_flat_layout, layout_blocks
        from .qwen_prefill_compile import _projection_backend_scope
        from .r10_paper_metrics import write_json
        segment = callback.segment
        source = self.layered.source_binding(segment)
        checked = self.layered._typed(segment.program, dict(segment.weight_shapes))
        shape = checked.shapes[segment.program.outputs[0]]
        if len(shape) != 3 or shape[0] != 1:
            raise ValueError('Retain the original typed token output shape')
        layout = (TokenLayout(segment.program.header.slots, *shape[1:]) if segment.role == 'layer'
                  else multi_flat_layout(segment.program.header.slots, shape[-1]))
        boundary = len(layout_blocks(layout))
        params = segment.program.params; slots = segment.program.header.slots
        hidden_inputs = len(layout_blocks(TokenLayout(slots, *params[0].shape[1:])))
        mask_inputs = sum(len(layout_blocks(multi_flat_layout(slots, math.prod(p.shape[1:])))) for p in params[1:])
        connectors = len(layout_blocks(multi_flat_layout(slots, params[0].shape[-1]))) + mask_inputs
        life = self.lifetime.Lifetime(self.diagnostics)
        dag_class = self.probe.logical_mask_backend(CostDag) if segment.role == 'layer' else CostDag
        count_class = self.probe.logical_mask_backend(StreamingCountBackend) if segment.role == 'layer' else StreamingCountBackend
        planner = self.lifetime.private_planner(life, dag_class=dag_class, count_class=count_class)
        original_plan = planner['plan_program']; captured = {}
        def capture(*args, **kwargs):
            value = original_plan(*args, **kwargs)
            captured['plan'] = value[0]
            return value
        planner['plan_program'] = capture
        costs = self.lifetime.private_cost_namespace(life, planner=planner)
        backend = state = bound = None
        name = segment.role + ('' if segment.index is None else str(segment.index))
        verified = False
        try:
            with _projection_backend_scope(), compact_transformer_projections():
                internal = costs['estimate_normalized'](segment.program, callback.weights,
                    profile=self.profile, verify_plan=True)
            plan = captured.pop('plan')
            verified = True
            bound = self.layered.bind_plan(segment, plan, planned_source=source)
            del plan
            backend = StreamingCountBackend(segment.program.header.slots, input_level=5, boot_out_level=5)
            counted_callback = already_scoped_lowering(callback)
            state = self.remaining.export_component(backend, counted_callback, bound)
            stats = backend.stats(); interface = state.metadata
            if (interface['explicit_boundary_bootstrap_calls'] != boundary
                    or interface['internal_plan_bootstrap_calls'] != bound.plan['boots']
                    or stats['counts']['boot'] != bound.plan['boots'] + boundary
                    or interface['symbolic_output_levels'] != [5]*boundary
                    or internal['boots_pred'] != bound.plan['boots']):
                raise ValueError('Original plan, real Count boundaries and output levels must agree')
            target = self.directory/name; target.mkdir(exist_ok=False)
            write_json(target/'source-binding.json', source)
            write_json(target/'refresh-plan.json', bound.plan)
            write_json(target/'complete-count.json', dict(stats=stats, interface=interface))
            # Original estimator source lines are retained; local IDs map back.
            original_ids = dict(segment.local_to_original_ids)
            statements = [dict(item, id=original_ids[item['id']]) for item in internal.get('statement_costs', [])]
            return dict(name=name, role=segment.role, index=segment.index, source=source,
                slots=slots, output_shape=list(shape), boundary_boots=boundary,
                caller_connector_blocks=connectors, input_levels=[5]*(hidden_inputs+mask_inputs),
                output_levels=interface['symbolic_output_levels'],
                internal=internal, complete_count=stats, statements=statements,
                constants=tuple(backend._constants), plan_artifact=str(target/'refresh-plan.json'),
                explicit_boundary_scope='actual complete Count callback; native inventory unmeasured')
        except BaseException as exc:
            if verified and backend is not None:
                _write_count_error(self.directory, name, exc)
            raise
        finally:
            captured.clear()
            if backend is not None:self.lifetime.release_backend(backend)
            state = bound = backend = None
            life.finish_records()
            gc.collect()


def aggregate(rows, *, boot_us):
    """Aggregate only completed source-bound component measurements."""
    if not rows or not math.isfinite(boot_us) or boot_us < 0:
        raise ValueError('Completed components and original finite profile boot cost required')
    normalized, raw, rotations, constants = Counter(), Counter(), set(), set()
    boundary = internal_boots = 0; latency = 0.; sites = Counter(); site_boots = Counter()
    slots = rows[0]['slots']; connectors = rows[0]['caller_connector_blocks']
    for row in rows:
        core = row['internal']; n = row['boundary_boots']
        if (type(n) is not int or n < 1 or row['output_levels'] != [5]*n
                or row['slots'] != slots
                or row['role']=='layer' and row['caller_connector_blocks'] != connectors
                or row['complete_count']['counts']['boot'] != core['boots_pred']+n):
            raise ValueError('Every boundary must be a counted real bootstrap to actual level5')
        normalized.update(core['counts']); normalized['boot'] += n
        raw.update(row['complete_count']['counts'])
        rotations.update(row['complete_count']['rot_steps']); constants.update(row['constants'])
        boundary += n; internal_boots += core['boots_pred']
        latency += core['latency_pred_us'] + n*boot_us
        original_ops = {item['id']:item for item in row['source']['original_ops']}
        for item in row['statements']:
            op = original_ops[item['id']]; key = op['lineno'],tuple(op['call_lines'])
            sites[key] += item['latency_us']; site_boots[key] += item['boots']
    sites[(None,())] += boundary*boot_us; site_boots[(None,())] += boundary
    names = [row['name'] for row in rows]
    if names != [*('layer'+str(i) for i in range(len(rows)-1)), 'head']:
        raise ValueError('Retain original consecutive layers and complete head once')
    boots = internal_boots+boundary
    if normalized['boot'] != boots:raise ValueError('Normalized internal boot inventory differs')
    # No complete-model graph IDs exist under this policy. Transformer validate
    # does not consume boot_nodes (agent_tools.validate:217-220).
    return dict(placement_model='component-program-plan', placement_exact=True,
        global_optimality=False, lowering_policy=POLICY, counts=dict(normalized),
        ops_total=sum(normalized.values()), complete_count_replay_counts=dict(raw),
        complete_count_replay_ops_total=sum(raw.values()), boots_pred=boots,
        component_internal_boots=internal_boots, explicit_boundary_boots=boundary,
        max_depth=max(row['internal']['max_depth'] for row in rows),
        max_depth_scope='Maximum original component live depth; actual boundary reset after every component',
        rot_keys=len(rotations), distinct_consts=len(constants), const_bytes=len(constants)*slots*8,
        constant_inventory_scope='Union of existing original backend constant-interning identities; no added payload or integrity operation and no public payload retention',
        latency_pred_us=latency, latency_scope='Sum original component CSE program-plan profile estimates plus measured Count boundary requests at the same profile cost',
        level_hist={}, barriers=None, boot_nodes=[], global_boot_node_ids_available=False,
        hotspots=[dict(line=line,call_lines=list(calls),latency_share=amount/latency if latency else 0.,
            boots=site_boots[(line,calls)],scope='component-boundary' if line is None else'original-source')
            for (line,calls),amount in sites.most_common(5)],
        boot_share=boots*boot_us/latency if latency else 0.,
        refresh_plan=dict(variant='component-program-plan', levels_left_at_refresh=[],
            components=[dict(name=r['name'], plan=r['plan_artifact'], internal_boots=r['internal']['boots_pred'],
                boundary_boots=r['boundary_boots'], output_shape=r['output_shape'], output_levels=r['output_levels'])for r in rows]),
        full_generation_projection=dict(unique_components=len(rows)+2, forward_reuse=2,
            component_calls=2*len(rows)+2, prefill_boundary_boot_requests=boundary,
            two_forward_boundary_boot_requests=2*boundary, external_connector_boot_requests=connectors,
            total_boundary_boot_requests=2*boundary+connectors, first_internal_boots=None,
            terminal_internal_boots=None, complete_boot_inventory=None, latency_pred_us=None,
            scope='Derived from this source-bound prefill Count plus existing first22-output handoff ABI; kernels/three caller connectors are not executed here'),
        compiled_bootstraps=None, formal_latency=None, CKKS_latency=None)


def estimate(program, weights, *, profile='qwen48', directory, diagnostics=None, api=None, **kwargs):
    """One whole-model candidate, serial component DAGs, unchanged quality elsewhere."""
    if profile != 'qwen48':raise ValueError('Retain the original qwen48 profile')
    directory = Path(directory); directory.mkdir(exist_ok=False)
    diagnostics = diagnostics or __import__(__package__+'.qwen_agent_cost_lifetime',fromlist=['Diagnostics']).Diagnostics()
    api = OriginalComponents(profile,directory,diagnostics) if api is None else api
    callbacks = api.split(program,weights); rows=[]
    from .r10_paper_metrics import write_json
    try:
        for callback in callbacks:
            name = callback.segment.role+('' if callback.segment.index is None else str(callback.segment.index))
            with diagnostics.span('component_cost_'+name):row=api.measure(callback)
            rows.append(row)
            write_json(directory/'partial.json',dict(status='running',completed=len(rows),components=[r['name']for r in rows]))
        result=aggregate(rows,boot_us=api.boot_us)
        write_json(directory/'result.json',dict(status='ok',**result))
        return result
    except BaseException as exc:
        write_json(directory/'partial.json',dict(status='error',error=type(exc).__name__,completed=len(rows),components=[r['name']for r in rows]))
        raise
    finally:
        rows.clear()
