"""Held-out structural, arithmetic and cost feedback for the LLM planner."""

from dataclasses import dataclass
import gc

import numpy as np

from ..analyzer import reference_forward, approximation_points, approximation_values
from ..backends import BackendError, MockBackend
from ..cost import PROFILES, estimate
from ..data.cifar import heldout_set, MEAN, STD
from ..dsl.errors import PTDSLError
from ..dsl.parser import parse
from ..dsl.printer import print_program
from ..dsl.typecheck import check
from ..intake import Graph
from ..metrics import compare, passes
from ..models.info import ModelInfo
from ..kernels.layout import MuxLayout, unpack_flat, unpack_mux
from ..kernels.cnn import DifferenceLayout
from ..kernels.layout import MultiMuxLayout, MultiFlatLayout, unpack_multi
from ..lower import LoweringError, decode_outputs, lower, pack_inputs
from ..placement import PlacementError
from ..tensor_sim import (cached_estimate, identity_token, result_cache_get,
                          result_cache_put, run_tensor)


FEASIBLE_MAX_U = 0.90
VAL_BATCH = 20
MAX_ACC_DROP_PP = 0.5


@dataclass
class AgentContext:
    graph: Graph
    weights: dict
    calibration: dict
    weight_shapes: dict
    val_images: np.ndarray
    val_exact_logits: np.ndarray
    val_labels: np.ndarray
    val_exact_top1: float
    probe_images: np.ndarray
    probe_act_inputs: list
    needed_margin: dict
    act_tensors: list
    model_name: str
    profile: str = "qwen48"
    val_split: str = "test[9000:10000]"
    normalization: tuple = (MEAN, STD)
    model_info: object = None
    samples: object = None
    val_inputs: object = None
    prompt_version: str = "r8-v4"


def build_context(graph, calibration, n_val=1000, *, samples=None, normalization=(MEAN, STD)) -> AgentContext:
    info = graph if isinstance(graph, ModelInfo) else None
    if info is not None:
        graph = info.graph
        if samples is None:
            raise ValueError('ModelInfo requires its original sample splits')
        inputs, labels = samples.get('heldout', n_val)
        images = inputs[graph.inputs[0]]
    else:
        # Compatibility with historical callers and their frozen reader tests.
        images, labels = heldout_set(n_val, normalization=normalization)
        images = images.reshape((len(images), *graph.tensors[graph.inputs[0]].shape[1:]))
        inputs = {graph.inputs[0]: images}
    act_tensors = [name for node, name in approximation_points(graph)]
    needed = dict.fromkeys(act_tensors, 0.0)
    logits, probes = [], []
    for start in range(0, len(images), VAL_BATCH):
        batch = images[start:start+VAL_BATCH]
        values = reference_forward(graph, {name: value[start:start+VAL_BATCH] for name, value in inputs.items()}, keep=True)
        logits.append(values[graph.outputs[0]].copy())
        point_values = approximation_values(graph, values)
        if start == 0:
            probe_values = approximation_values(graph, {name: value[:4] for name, value in values.items()})
            probes = [probe_values[name].copy() for name in act_tensors]
        for name in act_tensors:
            value = point_values[name]
            value = value.reshape(len(batch), -1) if value.ndim > 1 else value.reshape(1, -1)
            limits = calibration[name]
            margin = np.maximum(value.max(axis=1)/limits["max"], value.min(axis=1)/limits["min"])
            needed[name] = max(needed[name], float(margin.max()))
        del values
    exact_logits = np.concatenate(logits)
    return AgentContext(graph=graph, weights=graph.weights, calibration=calibration,
                        weight_shapes={name: value.shape for name, value in graph.weights.items()},
                        val_images=images, val_exact_logits=exact_logits, val_labels=labels,
                        val_exact_top1=(float(100*np.mean(exact_logits.argmax(axis=1) == labels))
                                        if labels is not None and (info is None or info.task == 'classification') else None),
                        probe_images=images[:4], probe_act_inputs=probes, needed_margin=needed,
                        act_tensors=act_tensors, model_name=graph.name, normalization=normalization,
                        model_info=info, samples=samples, val_inputs=inputs,
                        val_split=('test[9000:10000]' if info is None or info.split.get('kind') == 'legacy_cifar'
                                   else 'heldout'))


def _unpack(value, layout):
    if isinstance(layout, DifferenceLayout):
        return value.value
    if isinstance(layout, (MultiMuxLayout, MultiFlatLayout)):
        return unpack_multi([item.value for item in value], layout)
    if isinstance(layout, MuxLayout):
        return np.stack([unpack_mux(sample, layout) for sample in value.value])
    return unpack_flat(value.value, layout).copy()


def _exact_check(program, ctx, *, engine="mock"):
    ops = [op for op in program.ops if op.kind in ("act", "maxpool")]
    collected = []

    def exact_observer(name, value, layout, cfg):
        collected.append(_unpack(value, layout))

    backend = values = None
    if engine == "tensor":
        def tensor_exact_observer(name, data, cfg):
            collected.append(np.array(data, copy=True))
        logits = [run_tensor(program, ctx.weights, ctx.probe_images,
            observer=tensor_exact_observer, act_mode="exact", cache=False)]
    else:
        backend = MockBackend(program.header.slots, batch=4)
        probe_inputs = getattr(ctx, 'val_inputs', None)
        backend.bind(pack_inputs(program, [probe_inputs[name][:4] for name in ctx.graph.inputs]
                                 if probe_inputs is not None else [ctx.probe_images]))
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


def validate(program, ctx, n: int | None = None, *, boot_nodes=None,
             stop_on_divergence=True, engine: str = "mock", cache: bool = True,
             output_callback=None) -> dict:
    """Apply validation checks to the first n held-out images (all by default)."""
    images = ctx.val_images if n is None else ctx.val_images[:n]
    profile = PROFILES[ctx.profile]
    if engine not in ("mock", "tensor"):
        raise ValueError("engine must be mock or tensor")
    if engine == "tensor" and (profile['boot_input_bound'] is not None or profile['boot_abs_error'] is not None):
        raise ValueError("bootstrap input bounds and noise require engine='mock'")
    ops = [op for op in program.ops if op.kind in ("act", "maxpool")]
    image_id = None
    result_key = None
    if engine == "tensor" and cache and output_callback is None:
        # References distinguish live datasets and weights without reading or hashing their bytes.
        image_id = (identity_token(ctx.val_images), identity_token(ctx.weights))
        result_identity = (image_id, identity_token(ctx.val_exact_logits), identity_token(ctx.val_labels),
                           ctx.profile, tuple(ctx.act_tensors), tuple(op.lineno for op in ops))
        result_key = (print_program(program), result_identity, len(images), stop_on_divergence, engine)
        hit, saved = result_cache_get(result_key)
        if hit:
            return saved
    if engine == "mock" and boot_nodes is None:
        boot_nodes = cached_estimate(program, ctx.weights, ctx.profile, estimate, cache=cache)['boot_nodes']
    rows = [dict(program_line=op.lineno, graph_tensor=ctx.act_tensors[index], max_u=0.0,
                 obs_min=None, obs_max=None,
                 deg=op.attrs["deg"], method=op.attrs["method"], degrees=op.attrs.get("degrees"),
                 domain=op.attrs["domain"]) for index, op in enumerate(ops)]
    for row,op in zip(rows,ops):
        if op.attrs['method']=='composite':
            row['eps']=op.attrs['eps']
        elif op.attrs['method']=='newton':
            row.update(deg0=op.attrs['deg0'],iters=op.attrs['iters'])
    labels = None if ctx.val_labels is None else ctx.val_labels[:len(images)]
    info = getattr(ctx, 'model_info', None)
    labelled_classification = labels is not None and (info is None or info.task == 'classification')
    validation = dict(agree=0, total=len(images), diverged=False, n_diverged=0, diverged_indices=[],
                      images_done=0, worst_acts=[], max_u=0.0,
                      boot_input_max_abs=None if engine == "tensor" else 0.0,
                      boot_input_bound=profile['boot_input_bound'],
                      **(dict(top1=0.0, exact_top1=float(100*np.mean(ctx.val_exact_logits[:len(images)].argmax(axis=1)
                                                          == labels))) if labelled_classification else {}))
    outputs = np.full_like(ctx.val_exact_logits[:len(images)], np.nan)
    correct = 0
    watch = set(boot_nodes or ())
    noise = (profile['boot_abs_error'], 20261022) if profile['boot_abs_error'] else None
    def run_batch(batch, batch_start):
        # Keep failed vector-batch observations separate from the per-image retries.
        observations = [dict(obs_min=None, obs_max=None, max_u=0.0) for _ in rows]
        observed = 0

        def observer(name, value, layout, cfg):
            if isinstance(layout, (MultiMuxLayout, MultiFlatLayout, DifferenceLayout)):
                data = _unpack(value, layout)
            else:
                data = value.value[..., layout.index_grid()] if isinstance(layout, MuxLayout) else value.value[..., :layout.size]
            tensor_observer(name, data, cfg)

        def tensor_observer(name, data, cfg):
            nonlocal observed
            row = observations[observed]
            observed += 1
            row["obs_min"] = float(data.min())
            row["obs_max"] = float(data.max())
            _, _, lo, hi = cfg
            row["max_u"] = float(np.max(np.abs(data*(2/(hi-lo))-(hi+lo)/(hi-lo))))
            if ops[observed-1].kind == 'maxpool' and row['max_u'] > 1:
                raise BackendError('maxpool comparison input leaves its domain')

        def restore(records):
            for name, minimum, maximum, cfg in records:
                tensor_observer(name, np.asarray([minimum, maximum], dtype=np.float64), cfg)

        tensor_observer.restore = restore
        backend = MockBackend(program.header.slots, batch=len(batch), watch=watch, noise=noise) if engine == "mock" else None
        logits, error = None, None
        try:
            if engine == "tensor":
                logits = run_tensor(program, ctx.weights, batch, observer=tensor_observer,
                                    cache=cache, image_id=image_id, batch_start=batch_start)
            else:
                inputs = getattr(ctx, 'val_inputs', None)
                backend.bind(pack_inputs(program,
                    [inputs[name][batch_start:batch_start+len(batch)] for name in ctx.graph.inputs]
                    if inputs is not None else [batch]))
                values, layouts = lower(program, ctx.weights, backend, observer=observer)
                logits = decode_outputs([value.value for value in values], layouts)[0]
            if not np.isfinite(logits).all():
                raise BackendError("Mock arithmetic produced a nonfinite value")
        except BackendError as exc:
            logits, error = None, str(exc)
        boot_max = max(backend.watch_max.values(), default=0.0) if backend is not None else None
        del backend
        gc.collect()
        return logits, observations, boot_max, error

    def record(logits, observations, boot_max, start, count):
        nonlocal correct
        for row, observation in zip(rows, observations):
            if observation["obs_min"] is not None:
                row["obs_min"] = (observation["obs_min"] if row["obs_min"] is None
                                  else min(row["obs_min"], observation["obs_min"]))
                row["obs_max"] = (observation["obs_max"] if row["obs_max"] is None
                                  else max(row["obs_max"], observation["obs_max"]))
                row["max_u"] = max(row["max_u"], observation["max_u"])
        if engine == "mock":
            validation["boot_input_max_abs"] = max(validation["boot_input_max_abs"], boot_max)
        if logits is not None:
            outputs[start:start+count] = logits
        validation["images_done"] = start+count

    for start in range(0, len(images), VAL_BATCH):
        batch = images[start:start+VAL_BATCH]
        logits, observations, boot_max, error = run_batch(batch, start)
        if error is None:
            record(logits, observations, boot_max, start, len(batch))
            continue
        for offset in range(len(batch)):
            logits, observations, boot_max, error = run_batch(batch[offset:offset+1], start+offset)
            record(logits, observations, boot_max, start+offset, 1)
            if error is not None:
                validation["diverged_indices"].append(start+offset)
                validation["error"] = error
                if stop_on_divergence:
                    break
        if validation["diverged_indices"] and stop_on_divergence:
            break
    validation["n_diverged"] = len(validation["diverged_indices"])
    validation["diverged"] = validation["n_diverged"] > 0
    # Put the earliest domain violations before their downstream amplification.
    ordered = sorted(enumerate(rows), key=lambda item: (item[1]["max_u"] <= FEASIBLE_MAX_U,
                     item[0] if item[1]["max_u"] > FEASIBLE_MAX_U else -item[1]["max_u"]))
    validation["worst_acts"] = [row for _, row in ordered[:5]]
    validation["max_u"] = max((row["max_u"] for row in rows), default=0.0)
    validation["rows"] = rows
    incomplete = validation["diverged"] and stop_on_divergence
    reference = ctx.val_exact_logits[:len(images)]
    if output_callback is not None:
        output_callback(outputs.copy())
    metric_input = (dict(values=reference, task=info.task) if info is not None else reference)
    if info is not None and info.task == 'causal_lm':
        metric_input['lengths'] = getattr(ctx, 'val_inputs')['attention_mask'][:len(images)].sum(axis=1).astype(int)
    measured = compare(metric_input, outputs, labels)
    validation.update(measured)
    if incomplete and labelled_classification:
        validation.update(top1=None, acc_drop_pp=None)
    threshold = info.threshold if info is not None else {'acc_drop_pp': MAX_ACC_DROP_PP}
    validation["feasible"] = (not validation["diverged"] and validation["max_u"] <= FEASIBLE_MAX_U
                              and passes(validation, threshold)
                              and (validation['boot_input_bound'] is None
                                   or validation['boot_input_max_abs'] <= validation['boot_input_bound']))
    if result_key is not None:
        result_cache_put(result_key, validation)
    return validation


def evaluate(source: str, ctx: AgentContext, *, engine: str = "mock", cache: bool = True,
             engine_profile: str = "mock") -> dict:
    if engine_profile == "tensor":
        engine = "tensor"
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
    weight_usage = engine_profile == "tensor" or getattr(ctx, "prompt_version", "r8-v4") in ("r8-v5","r8-v6")
    checked = check(program, ctx.weight_shapes, **(dict(graph=ctx.graph) if weight_usage else {}))
    if not checked.ok:
        errors = [diag for diag in checked.diags if diag.severity == "error"]
        shown = errors[:20]
        if weight_usage:
            shown += [diag for diag in errors[20:] if diag.code == "E_WEIGHT_UNUSED"]
        return dict(fb, errors=[error_record(diag.code, diag.lineno, diag.call_lines, diag.msg)
                               for diag in shown])
    fb["stage"] = "exact"
    try:
        fb["exact"] = _exact_check(program, ctx, **(dict(engine="tensor") if engine_profile == "tensor" else {}))
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
    validation = validate(program, ctx, boot_nodes=cost['boot_nodes'], engine=engine, cache=cache)
    fb["validation"] = {key: value for key, value in validation.items() if key not in ("rows", "feasible", "error")}
    fb["feasible"] = fb["exact"]["ok"] and validation["feasible"]
    if "error" in validation:
        fb["error"] = validation["error"]
    if not validation["diverged"]:
        fb["stage"] = "complete"
    return fb
