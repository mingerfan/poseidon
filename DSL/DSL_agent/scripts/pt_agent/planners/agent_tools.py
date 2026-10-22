"""Held-out structural, arithmetic and cost feedback for the LLM planner."""

from dataclasses import dataclass
import gc

import numpy as np

from ..analyzer import reference_forward
from ..backends import BackendError, MockBackend
from ..cost import PROFILES, estimate
from ..data.cifar import heldout_set, MEAN, STD
from ..dsl.errors import PTDSLError
from ..dsl.parser import parse
from ..dsl.typecheck import check
from ..intake import Graph
from ..kernels.layout import MuxLayout, unpack_flat, unpack_mux
from ..lower import LoweringError, decode_outputs, lower, pack_inputs
from ..placement import PlacementError


FEASIBLE_MAX_U = 0.90


@dataclass
class AgentContext:
    graph: Graph
    weights: dict
    calibration: dict
    weight_shapes: dict
    val_images: np.ndarray
    val_exact_logits: np.ndarray
    probe_images: np.ndarray
    probe_act_inputs: list
    needed_margin: dict
    act_tensors: list
    model_name: str
    profile: str = "qwen48"
    val_split: str = "test[9000:10000]"


def build_context(graph, calibration, n_val=1000, *, normalization=(MEAN, STD)) -> AgentContext:
    images, _ = heldout_set(n_val, normalization=normalization)
    images = images.reshape((len(images), *graph.tensors[graph.inputs[0]].shape[1:]))
    act_tensors = [node.inputs[0] for node in graph.nodes if node.op == "act"]
    needed = dict.fromkeys(act_tensors, 0.0)
    logits, probes = [], []
    for start in range(0, len(images), 50):
        batch = images[start:start+50]
        values = reference_forward(graph, {graph.inputs[0]: batch}, keep=True)
        logits.append(values[graph.outputs[0]].copy())
        if start == 0:
            probes = [values[name][:4].copy() for name in act_tensors]
        for name in act_tensors:
            value = values[name].reshape(len(batch), -1)
            limits = calibration[name]
            margin = np.maximum(value.max(axis=1)/limits["max"], value.min(axis=1)/limits["min"])
            needed[name] = max(needed[name], float(margin.max()))
        del values
    return AgentContext(graph, graph.weights, calibration, {name: value.shape for name, value in graph.weights.items()},
                        images, np.concatenate(logits), images[:4], probes, needed, act_tensors, graph.name)


def _unpack(value, layout):
    if isinstance(layout, MuxLayout):
        return np.stack([unpack_mux(sample, layout) for sample in value.value])
    return unpack_flat(value.value, layout).copy()


def _exact_check(program, ctx):
    ops = [op for op in program.ops if op.kind == "act"]
    collected = []

    def exact_observer(name, value, layout, cfg):
        collected.append(_unpack(value, layout))

    backend = MockBackend(16384, batch=4)
    backend.bind(pack_inputs(program, [ctx.probe_images]))
    values, layouts = lower(program, ctx.weights, backend, act_mode="exact", observer=exact_observer)
    logits = decode_outputs([value.value for value in values], layouts)
    reference = ctx.val_exact_logits[:4]
    logit_diff = (float(np.max(np.abs(logits[0]-reference)))
                  if len(logits) == 1 and logits[0].shape == reference.shape else None)
    first = None
    for index, (actual, expected) in enumerate(zip(collected, ctx.probe_act_inputs)):
        difference = float(np.max(np.abs(actual-expected))) if actual.shape == expected.shape else None
        if difference is None or difference > 1e-6:
            first = dict(act_index=index, graph_tensor=ctx.act_tensors[index],
                         program_line=ops[index].lineno, max_diff=difference)
            break
    if first is None and len(collected) != len(ctx.act_tensors):
        index = min(len(collected), len(ctx.act_tensors))
        first = dict(act_index=index, graph_tensor=ctx.act_tensors[index] if index < len(ctx.act_tensors) else None,
                     program_line=ops[index].lineno if index < len(ops) else None, max_diff=None)
    result = dict(ok=first is None and logit_diff is not None and logit_diff <= 1e-9,
                       logit_max_diff=logit_diff, first_mismatch=first,
                       act_count=[len(collected), len(ctx.act_tensors)])
    del backend, values, logits, collected
    gc.collect()
    return result


def validate(program, ctx, n: int | None = None, *, boot_nodes=None) -> dict:
    """Apply validation checks to the first n held-out images (all by default)."""
    images = ctx.val_images if n is None else ctx.val_images[:n]
    profile = PROFILES[ctx.profile]
    ops = [op for op in program.ops if op.kind == "act"]
    if boot_nodes is None:
        boot_nodes = estimate(program, ctx.weights, profile=ctx.profile)['boot_nodes']
    rows = [dict(program_line=op.lineno, graph_tensor=ctx.act_tensors[index], max_u=0.0,
                 obs_min=None, obs_max=None,
                 deg=op.attrs["deg"], method=op.attrs["method"], degrees=op.attrs.get("degrees"),
                 domain=op.attrs["domain"]) for index, op in enumerate(ops)]
    validation = dict(agree=0, total=len(images), diverged=False, images_done=0, worst_acts=[], max_u=0.0,
                      boot_input_max_abs=0.0, boot_input_bound=profile['boot_input_bound'])
    watch = set(boot_nodes)
    noise = (profile['boot_abs_error'], 20261022) if profile['boot_abs_error'] else None
    for start in range(0, len(images), 50):
        batch = images[start:start+50]
        observed = 0

        def observer(name, value, layout, cfg):
            nonlocal observed
            row = rows[observed]
            observed += 1
            data = value.value[..., layout.index_grid()] if isinstance(layout, MuxLayout) else value.value[..., :layout.size]
            row["obs_min"] = float(data.min()) if row["obs_min"] is None else min(row["obs_min"], float(data.min()))
            row["obs_max"] = float(data.max()) if row["obs_max"] is None else max(row["obs_max"], float(data.max()))
            _, _, lo, hi = cfg
            row["max_u"] = max(row["max_u"], float(np.max(np.abs(data*(2/(hi-lo))-(hi+lo)/(hi-lo)))))

        try:
            backend = MockBackend(16384, batch=len(batch), watch=watch, noise=noise)
            backend.bind(pack_inputs(program, [batch]))
            values, layouts = lower(program, ctx.weights, backend, observer=observer)
            logits = decode_outputs([value.value for value in values], layouts)[0]
            validation["agree"] += int(np.sum(logits.argmax(axis=1) == ctx.val_exact_logits[start:start+len(batch)].argmax(axis=1)))
            validation["images_done"] = start+len(batch)
            del values, logits
        except BackendError as exc:
            validation["diverged"] = True
            validation["error"] = str(exc)
        validation['boot_input_max_abs'] = max(validation['boot_input_max_abs'],
                                               max(backend.watch_max.values(), default=0.0))
        del backend
        gc.collect()
        if validation["diverged"]:
            break
    # Put the earliest domain violations before their downstream amplification.
    ordered = sorted(enumerate(rows), key=lambda item: (item[1]["max_u"] <= FEASIBLE_MAX_U,
                     item[0] if item[1]["max_u"] > FEASIBLE_MAX_U else -item[1]["max_u"]))
    validation["worst_acts"] = [row for _, row in ordered[:5]]
    validation["max_u"] = max((row["max_u"] for row in rows), default=0.0)
    validation["rows"] = rows
    validation["feasible"] = (not validation["diverged"] and validation["max_u"] <= FEASIBLE_MAX_U
                              and validation["agree"] >= .99*validation["total"]
                              and (validation['boot_input_bound'] is None
                                   or validation['boot_input_max_abs'] <= validation['boot_input_bound']))
    return validation


def evaluate(source: str, ctx: AgentContext) -> dict:
    fb = dict(stage="parse", feasible=False, exact=None, cost=None, validation=None)
    lines = source.splitlines()

    def line_text(lineno):
        return lines[lineno-1].strip() if lineno and 1 <= lineno <= len(lines) else ""

    def error_record(code, lineno, call_lines, message):
        return dict(code=code, line=lineno, line_text=line_text(lineno), call_lines=list(call_lines),
                    call_texts=[line_text(line) for line in call_lines], message=message)

    def lower_failure(exc):
        op = program.ops[exc.op_id] if exc.op_id is not None else None
        error = error_record("E_LOWER", op.lineno if op else exc.lineno,
                             op.call_lines if op else (), exc.msg)
        return dict(fb, stage="lower", error=str(exc), errors=[error])

    try:
        program = parse(source)
    except PTDSLError as exc:
        return dict(fb, error=str(exc))
    fb["stage"] = "typecheck"
    checked = check(program, ctx.weight_shapes)
    if not checked.ok:
        return dict(fb, errors=[error_record(diag.code, diag.lineno, diag.call_lines, diag.msg)
                               for diag in checked.diags if diag.severity == "error"][:20])
    fb["stage"] = "exact"
    try:
        fb["exact"] = _exact_check(program, ctx)
    except LoweringError as exc:
        return lower_failure(exc)
    except BackendError as exc:
        return dict(fb, error=str(exc))
    if not fb["exact"]["ok"]:
        return fb
    fb["stage"] = "cost"
    try:
        cost = estimate(program, ctx.weights, profile=ctx.profile)
        fb["cost"] = {key: cost[key] for key in ("latency_pred_us", "boots_pred", "max_depth", "ops_total", "rot_keys",
                                               "placement_model", "placement_exact", "barriers", "boot_share")}
        fb["cost"]["hotspots"] = [dict(row, line_text=line_text(row['line']),
                                        call_texts=[line_text(line) for line in row['call_lines']])
                                    for row in cost['hotspots']]
    except LoweringError as exc:
        return lower_failure(exc)
    except (BackendError, PlacementError) as exc:
        return dict(fb, error=str(exc))
    fb['stage'] = 'resources'
    profile = PROFILES[ctx.profile]
    errors = [dict(code='E_RESOURCE', limit=name, value=cost[name], max=limit)
              for name, limit in profile['limits'].items() if limit is not None and cost[name] > limit]
    if errors:
        return dict(fb, errors=errors)
    fb["stage"] = "validation"
    validation = validate(program, ctx, boot_nodes=cost['boot_nodes'])
    fb["validation"] = {key: value for key, value in validation.items() if key not in ("rows", "feasible", "error")}
    fb["feasible"] = fb["exact"]["ok"] and validation["feasible"]
    if "error" in validation:
        fb["error"] = validation["error"]
    if not validation["diverged"]:
        fb["stage"] = "complete"
    return fb
