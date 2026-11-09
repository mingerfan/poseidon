"""Compiler placement estimates, lazy refresh estimates and RuntimePlan reports."""

from collections import Counter
import json
from pathlib import Path
import gc
from weakref import WeakSet

from .backends import CountBackend, DagBackend, RefreshBackend
from .lower import lower
from .placement import depth_dp, PlacementError
from .toolchain import BOOT_CONFIG_DIR


PROFILES = {"qwen48": dict(input_level=5, boot_in_level=1, boot_out_level=5,
                           latency=BOOT_CONFIG_DIR / "compiler-config.json",
                           limits=dict(const_bytes=4*2**30, rot_keys=None),
                           boot_input_bound=None, boot_abs_error=None)}


def _boot_normalization(boot: str) -> int:
    """Post-bootstrap normalization levels of the configured compiler for this boot profile."""
    from .placement import boot_normalization_levels
    from .toolchain import management_profile
    config = management_profile('dacapo-depth-dp', boot)
    return boot_normalization_levels(config['boot_window'], config['rescalingFactor'], config['waterline'])


PROFILES['qwen48']['boot_normalization_levels'] = _boot_normalization('qwen48')

_estimate_backends = WeakSet()


def release_estimate_caches():
    """Release lowering payloads before Mock, retaining cached cost results."""
    from .kernels.slots import _current
    current = _current.get()
    for backend in list(_estimate_backends):
        backend._constant_objects.clear()
        backend._constants.clear()
        if current is not None and current[0] is backend:
            current[1].clear()
            current[2].clear()
    backend = None
    # Cost DAG values and polynomial recurrences can form unreachable cycles.
    gc.collect()


def load_latency(profile: str) -> dict:
    table = json.loads(PROFILES[profile]["latency"].read_text())["latencyTable"]
    return {key.removeprefix("earth."): values for key, values in table.items()}


def depth_dp_estimate(backend, profile: str = "qwen48"):
    """The depth-dp placement the estimate uses, on a finished (and, for CostDag, finalized) DAG."""
    config = PROFILES[profile]
    return depth_dp(backend, last_boot_level=config['boot_out_level']-config['boot_in_level'],
                    boot_in_level=config['boot_in_level'], boot_out_level=config['boot_out_level'],
                    latency_table=load_latency(profile),
                    boot_normalization_levels=config.get('boot_normalization_levels', 1))


def estimate(program, weights, *, profile: str = "qwen48", placement_model=None,
             frontend_normalize=None, verify_plan=True) -> dict:
    if frontend_normalize is None:
        from .transformer_lower import uses_transformer
        frontend_normalize = uses_transformer(program)
    config = PROFILES[profile]
    if placement_model is None:
        if program.header.refresh == 'agent' and program.header.placement in ('depth-dp-plan', 'program-plan'):
            placement_model = program.header.placement
        else:
            placement_model = ('depth-dp' if program.header.refresh == 'compiler'
                               and program.header.placement == 'depth-dp' else 'lazy')
    if placement_model not in ('depth-dp', 'lazy', 'depth-dp-plan', 'program-plan'):
        raise ValueError('placement_model must be depth-dp, depth-dp-plan, program-plan or lazy')
    table = load_latency(profile)
    exact = placement_model in ('depth-dp', 'depth-dp-plan', 'program-plan') or program.header.refresh == 'agent'
    if placement_model in ('depth-dp-plan', 'program-plan'):
        # E5a: the agent places bootstraps with the same depth-dp, recorded by call number.
        from .plan import plan_program, verify_program
        plan, backend, placement = plan_program(program, weights, profile=profile,
                                                frontend_normalize=frontend_normalize)
        _estimate_backends.add(backend)
    elif placement_model == 'depth-dp':
        if frontend_normalize:
            from .backends.cost_dag import CostDag
            backend = CostDag(program.header.slots)
        else:
            backend = DagBackend(program.header.slots)
        _estimate_backends.add(backend)
        lower(program, weights, backend)
        if frontend_normalize:
            backend.finalize()
        placement = depth_dp_estimate(backend, profile)
    if placement_model in ('depth-dp', 'depth-dp-plan', 'program-plan'):
        stats = backend.stats()
        counts = stats['counts']
        counts['boot'] = placement.boots
        op_by_id = {op.id: op for op in program.ops}

        def location(nid):
            op = op_by_id.get(backend.prog_op[nid])
            return (op.lineno, tuple(op.call_lines)) if op is not None else (None, ())

        by_location, boots = Counter(), Counter()
        for nid, latency in placement.node_latency.items():
            by_location[location(nid)] += latency
        for nid in placement.boot_nodes:
            boots[location(nid)] += 1
        total = placement.latency_us
        hotspots = [dict(line=line, call_lines=list(call_lines),
                         latency_share=round(latency/total, 3) if total else 0.0,
                         boots=boots[(line, call_lines)])
                    for (line, call_lines), latency in by_location.most_common(5)]
        result = dict(counts=counts, ops_total=sum(counts.values()), max_depth=placement.max_depth,
                    rot_keys=len(stats['rot_steps']), distinct_consts=stats['distinct_consts'],
                    const_bytes=stats['distinct_consts']*program.header.slots*8,
                    boots_pred=placement.boots, latency_pred_us=total, level_hist={},
                    placement_model=placement_model, placement_exact=exact, barriers=placement.barriers,
                    boot_nodes=placement.boot_nodes, hotspots=hotspots,
                    boot_share=placement.boots*table['bootstrap_single'][config['boot_in_level']]/total if total else 0.0)
        if frontend_normalize:
            result['normalization'] = {key:value for key,value in backend.normalization.items() if key!='by_op'}
            result['normalization']['by_line'] = [dict(row,
                line=op_by_id[row['id']].lineno if row['id'] in op_by_id else None,
                op=op_by_id[row['id']].kind if row['id'] in op_by_id else 'input')
                for row in backend.normalization['by_op']]
        from .transformer_lower import uses_transformer
        if uses_transformer(program):
            from .transformer_cost import details
            result.update(details(program,weights,backend,placement))
        if placement_model in ('depth-dp-plan', 'program-plan'):
            result['refresh_plan'] = dict(variant=plan['placement'], candidates=plan.get('candidates'),
                                          calls=plan['calls'], substitutions=len(plan['substitutions']),
                                          outputs=len(plan['outputs']),
                                          dead_calls=sum(end-start for start, end in plan['dead']),
                                          timing=plan['timing'])
            if verify_plan:
                result['refresh_plan']['replay'] = verify_program(program, weights, plan, profile=profile)
            if placement_model == 'program-plan':
                from .stage_plan import simulate, model_for
                model = model_for(profile)
                simulated = simulate(backend, placement.refreshed_uses, model)
                if simulated is None:
                    raise PlacementError('Selected program-plan failed level simulation')
                level = simulated[0]
                grouped = {}
                for nid in placement.boot_nodes:
                    grouped.setdefault(location(nid), []).append(level[nid] - model.boot_in_level)
                result['refresh_plan']['levels_left_at_refresh'] = [
                    dict(line=line, call_lines=list(call_lines), mean=sum(values)/len(values))
                    for (line, call_lines), values in grouped.items()]
        return result
    inner = CountBackend(program.header.slots, boot_out_level=config["boot_out_level"])
    _estimate_backends.add(inner)
    backend = RefreshBackend(inner, **{key: config[key] for key in ("input_level", "boot_in_level", "boot_out_level")},
                             latency_table=table)
    lower(program, weights, backend)
    counts, refresh = inner.stats(), backend.stats()
    return dict(counts=counts["counts"], ops_total=counts["ops_total"], max_depth=counts["max_total_depth"],
                rot_keys=len(counts["rot_steps"]), distinct_consts=counts["distinct_consts"],
                const_bytes=counts["distinct_consts"]*program.header.slots*8,
                boots_pred=refresh["boots"], latency_pred_us=refresh["latency_us"], level_hist=refresh["level_hist"],
                placement_model=placement_model, placement_exact=exact, barriers=None,
                boot_nodes=[], hotspots=[],
                boot_share=refresh['boots']*table['bootstrap_single'][config['boot_in_level']]/refresh['latency_us']
                if refresh['latency_us'] else 0.0)


def estimate_normalized(program, weights, *, profile="qwen48", placement_model=None, verify_plan=True):
    """r8-v7 selects the normalized graph even for an existing CNN program."""
    return estimate(program,weights,profile=profile,placement_model=placement_model,frontend_normalize=True,
                    verify_plan=verify_plan)


def estimate_fast(program, weights, *, profile="qwen48", **kwargs):
    return estimate(program, weights, profile=profile, verify_plan=False, **kwargs)


def estimate_normalized_fast(program, weights, *, profile="qwen48", **kwargs):
    return estimate_normalized(program, weights, profile=profile, verify_plan=False, **kwargs)


def plan_latency(plan_path, profile: str = "qwen48") -> dict:
    plan = json.loads(Path(plan_path).read_text())
    table = load_latency(profile)
    levels = {value["id"]: value["level"] for value in plan["values"]}
    keys = dict(add_cc="add_double", sub_cc="add_double", add_cp="add_single", sub_cp="add_single",
                negate="negate_single", rotate="rotate_single", mul_cc="mul_double", mul_cp="mul_single",
                rescale="rescale_single", mod_switch="modswitch_single", boot="bootstrap_single")
    by_op, untimed = Counter(), Counter()
    for step in plan["execution"]:
        op = step["op"]
        if op in keys:
            by_op[op] += table[keys[op]][levels[step["inputs"][0]]]
        else:
            untimed[op] += 1
    return dict(latency_us=sum(by_op.values()), by_op=dict(by_op), untimed=dict(untimed),
                boots=sum(step["op"] == "boot" for step in plan["execution"]), steps=len(plan["execution"]))
