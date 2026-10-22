"""Build generic prompts from lowering capabilities and structural examples."""

import json
from pathlib import Path

from ..capabilities import render_capabilities
from ..cost import PROFILES
from ..intake.patterns import annotate
from .agent_tools import FEASIBLE_MAX_U

PROMPT_VERSION = "r8-v3"
TEMPLATE = """You are an FHE compiler engineer. You write PT-DSL programs that implement a given neural network on CKKS
homomorphic encryption. A trusted lowering turns PT-DSL into Hecate code; an unmodified compiler then inserts
rescale/modswitch/bootstrap. Your program must compute exactly the given network (except for the polynomial
activation approximations you choose) and should minimize the predicted encrypted latency.

PT-DSL syntax (a restricted Python subset; it is parsed, never executed):
- One entry function decorated with @fhe.program(slots=16384, refresh="compiler", placement="depth-dp").
  Parameters are annotated as Cipher[d0, d1, ...] with the network input shape.
- Statements: `name = op(inputs..., attr=value, ...)`, `return name`, and `with region("name", level_budget=k):`.
- Optional macros: top-level `def name(params): ... return t` without decorator; call them like ops.
  Macro parameters may receive tensors, numbers, strings and tuples of numbers. Weight names can be built as W[p + ".suffix"].
- No for/if/while, no arithmetic expressions, no nested calls. Names must not contain "__".
- Weights are referenced as W["<name>"] using the weight names given in the input.

{capabilities}

The input lists every network node in order with its shapes, weights and attributes. Every act node also lists
calib = [min, max] observed on calibration images and needed_margin = the factor by which the exact network
exceeds calib on validation images; a domain must cover calib widened by needed_margin, plus room for the error
of the approximations before it.

Cost drivers: activations dominate the multiplicative depth and therefore the number of bootstraps;
rotations come from convolutions, pooling and linear layers. The feedback reports predicted latency and
bootstraps, computed with the same bootstrap placement as the compiler, and "hotspots": the program lines
(with their macro call sites) that carry the largest share of predicted latency. Spend changes where the
share is large.
After each feasible program, a deterministic tuner lowers activation degrees and tightens domains one at a time and reports the settings it kept ("tuned"). Concentrate on structural decisions (approximation method, layout and algorithms, macro structure); you may adopt the tuned settings.

Feasibility (all required): the program parses and type-checks; the exact-mode check matches the reference
network; on the {n_val} validation images no activation input leaves its domain (max |u| <= {max_u}, where u is
the input mapped to [-1, 1]) and the predicted class agrees with the exact network for at least 99% of the images. The program must also stay within the target limits given in the input ("target"); a violated limit is reported at stage "resources". When the target gives boot_input_bound, every value the compiler bootstraps must stay within it on the validation images.
Among feasible programs, lower predicted latency is better.

When feedback reports an error, it gives the line, the source text, the macro call site and the shapes
involved. Change only what the feedback points at and keep the rest of the program unchanged; do not
restructure the network or drop layers to make an error disappear.

Answer with one JSON object: {{"program": "<complete PT-DSL source>", "rationale": "<at most 80 words>"}}.
Always return the complete program, never a diff.

Examples (other, small networks; they show the syntax and the patterns):
{examples}
"""

EXAMPLES_DIR = Path(__file__).with_name("examples")


def target_features(graph, patterns) -> set:
    features = {node.op for node in graph.nodes} | {group["pattern"] for group in patterns}
    if any(len(graph.tensors[name].shape) == 2 for name in graph.inputs):
        features.add("flat_input")
    return features


def select_examples(features: set, model_name: str, k: int = 2) -> list[str]:
    ranked = []
    for example in json.loads((EXAMPLES_DIR / "index.json").read_text()):
        if example["source_model"] == model_name:
            continue
        other = set(example["features"])
        union = features | other
        score = len(features & other) / len(union) if union else 0.0
        if score > 0:
            ranked.append((-score, example["name"]))
    return [name for _, name in sorted(ranked)[:k]]


def build_system_prompt(ctx) -> str:
    patterns = annotate(ctx.graph)
    chosen = select_examples(target_features(ctx.graph, patterns), ctx.model_name)
    index = {entry["name"]: entry for entry in json.loads((EXAMPLES_DIR / "index.json").read_text())}
    examples = "\n\n".join("# example: " + name + "\n" +
                              (EXAMPLES_DIR / index[name]["file"]).read_text().rstrip("\n")
                              for name in chosen)
    return TEMPLATE.format(capabilities=render_capabilities(), max_u=FEASIBLE_MAX_U,
                           n_val=len(ctx.val_images), examples=examples)


def initial_message(ctx) -> str:
    graph = ctx.graph
    patterns = annotate(graph)
    membership = {node: group for group in patterns for node in group["nodes"]}
    profile = PROFILES[ctx.profile]
    target = dict(profile=ctx.profile, placement="depth-dp",
                  limits={key: value for key, value in profile["limits"].items() if value is not None},
                  boot_input_bound=profile["boot_input_bound"])
    nodes = []
    for node in graph.nodes:
        a = node.attrs
        row = dict(id=node.id, op=node.op, **{"in": node.inputs}, out=node.outputs[0],
                   shape=graph.tensors[node.outputs[0]].shape)
        if node.op in ("conv2d", "linear"):
            row.update(weight=a["weight"], bias=a.get("bias"))
            if node.op == "conv2d":
                row.update(stride=a["stride"], padding=a["padding"])
        elif node.op == "batchnorm":
            row.update(prefix=a["weight"].removesuffix(".weight"), eps=a["eps"])
        elif node.op == "act":
            limits = ctx.calibration[node.inputs[0]]
            row.update(fn=a["fn"], calib=[limits["min"], limits["max"]],
                       needed_margin=ctx.needed_margin[node.inputs[0]])
        elif node.op == "slice":
            row.update({key: a[key] for key in ("axis", "start", "end", "step")})
        elif node.op == "pad":
            row["pads"] = a["pads"]
        elif node.op == "mean":
            row["axes"] = a["axes"]
        if node.id in membership:
            group = membership[node.id]
            row.update(pattern=group["pattern"], group=group["group"])
        nodes.append(row)
    return json.dumps(dict(task="Write the PT-DSL program for this network.", network=graph.name,
                           patterns=patterns, model=ctx.model_name, target=target,
                           input=dict(name=graph.inputs[0], shape=graph.tensors[graph.inputs[0]].shape),
                           nodes=nodes, output=graph.outputs[0]), separators=(",", ":"))


def feedback_message(fb, iteration, best, *, tuned=None) -> str:
    if fb.get("error") or fb.get("errors"):
        instruction = "Fix the errors and return the complete program."
    elif fb.get("feasible"):
        instruction = ("The program is feasible. The tuner's result is in \"tuned\". Reduce predicted latency further "
                       "by structural changes (for example a different approximation method on the hotspot lines) "
                       "while keeping all feasibility conditions. Return the complete program.")
    else:
        instruction = "The program is not feasible. Fix the violated conditions and return the complete program."
    result = dict(iteration=iteration, feedback=fb, best_feasible_so_far=best, next=instruction)
    if tuned is not None:
        result['tuned'] = dict(latency_pred_us=tuned['cost_after']['latency_pred_us'],
                               boots_pred=tuned['cost_after']['boots_pred'],
                               changes=sorted(tuned['changes'], key=lambda change: change['line'])[:30])
    return json.dumps(result)
