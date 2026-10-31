"""Compare existing 18/27-Boot plans by their arithmetic DAG, on 188Server."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path("/home/xuming/poseidon-manual-boot-20261026")
AUDIT = ROOT / "resnet20-optimized-20261027T055506Z/placement-audit"
EXPERIMENT = AUDIT / "target-scale-experiment"
OUT = EXPERIMENT / "scale-overhead-20261028"
OUT.mkdir(exist_ok=True)
PLANS = {
    "manual_scales_18": EXPERIMENT / "budget25/automatic/manual34",
    "threshold_27": EXPERIMENT
    / "chain32-fixed25-20261028/candidates/manual/full-optimized-q34",
}
SCALARS = {
    value: "sha256:" + hashlib.sha256(struct.pack("<d", value) * 32768).hexdigest()
    for value in (1.0, 0.5)
}
registry = {}
nodes = []


def intern(key):
    if key not in registry:
        registry[key] = len(nodes)
        nodes.append(key)
    return registry[key]


def analyze(name, directory):
    filename = next(directory.glob("*.runtime-plan.json"))
    plan = json.loads(filename.read_text())
    values = {v["id"]: v for v in plan["values"]}
    expression = {
        v: intern(("input", i)) for i, v in enumerate(plan["external_inputs"])
    }
    producer = {}
    users = defaultdict(list)
    unit_steps = set()
    execution = plan["execution"]
    for step in plan["initialization"] + execution + plan["finalization"]:
        kind = step["kind"]
        if kind == "encode":
            payload = step["payload"]
            assert payload["kind"] == "bundle"
            expression[step["output"]] = intern(("constant", payload["content"]))
        elif kind == "transfer":
            for before, after in zip(step["inputs"], step["outputs"]):
                expression[after] = expression[before]
        elif kind == "compute":
            op = step["op"]
            inputs = [expression[v] for v in step["inputs"]]
            if op in ("boot", "rescale", "relinearize", "mod_switch"):
                result = inputs[0]
            elif op in ("mul_cc", "mul_cp"):
                if op == "mul_cp" and nodes[inputs[1]] == ("constant", SCALARS[1.0]):
                    unit_steps.add(step["ordinal"])
                    result = inputs[0]
                else:
                    result = intern(("mul", *sorted(inputs)))
            elif op in ("add_cc", "add_cp"):
                result = intern(("add", *sorted(inputs)))
            elif op == "negate":
                result = intern(("negate", inputs[0]))
            elif op == "rotate":
                result = intern(
                    ("rotate", inputs[0], json.dumps(step["attrs"], sort_keys=True))
                )
            else:
                raise ValueError(op)
            expression[step["output"]] = result
            producer[step["output"]] = step
            for v in step["inputs"]:
                users[v].append(step)
        else:
            raise ValueError(kind)

    cc = [s for s in execution if s.get("op") == "mul_cc"]
    assert len(cc) == 513
    half = intern(("constant", SCALARS[0.5]))
    relus = []
    claimed = set()
    for i in range(19):
        group = cc[i * 27 : (i + 1) * 27]
        final = group[-1]
        assert (
            nodes[expression[final["inputs"][0]]][0] == "add"
            or nodes[expression[final["inputs"][1]]][0] == "add"
        )
        half_operands = [
            v
            for v in final["inputs"]
            if nodes[expression[v]][0] == "add" and half in nodes[expression[v]][1:]
        ]
        assert len(half_operands) == 1
        source = next(v for v in final["inputs"] if v != half_operands[0])
        source_expr = expression[source]
        end = final["output"]
        while True:
            continuation = [
                s for s in users[end] if s["op"] in ("rescale", "relinearize")
            ]
            if not continuation:
                break
            assert len(continuation) == 1
            end = continuation[0]["output"]
        scope = {}
        stack = [end]
        while stack:
            v = stack.pop()
            step = producer.get(v)
            if step is None:
                continue
            if expression[v] == source_expr and step["ordinal"] < group[0]["ordinal"]:
                continue
            if step["ordinal"] in scope:
                continue
            scope[step["ordinal"]] = step
            stack.extend(step["inputs"])
        assert not claimed.intersection(scope)
        claimed.update(scope)
        stages = []
        bounds = [
            group[8]["ordinal"],
            group[16]["ordinal"],
            group[26]["ordinal"],
            float("inf"),
        ]
        for stage, upper in zip(("P1", "P2", "P3", "tail"), bounds):
            lower = -1 if not stages else bounds[len(stages) - 1]
            steps = [s for n, s in scope.items() if lower <= n < upper]
            stages.append(
                dict(
                    stage=stage,
                    counts=dict(Counter(s["op"] for s in steps)),
                    unit_muls=sum(s["ordinal"] in unit_steps for s in steps),
                )
            )
        entry = max(
            (
                v
                for v in values
                if v in expression
                and expression[v] == source_expr
                and (v not in producer or producer[v]["ordinal"] < group[0]["ordinal"])
            ),
            key=lambda v: producer[v]["ordinal"] if v in producer else -1,
        )
        relus.append(
            dict(
                number=i + 1,
                input=values[entry],
                saved_input_at_tail=values[source],
                output=values[end],
                output_expression=expression[end],
                counts=dict(Counter(s["op"] for s in scope.values())),
                unit_muls=sum(n in unit_steps for n in scope),
                stages=stages,
            )
        )
    outside = [
        s
        for s in execution
        if s.get("kind") == "compute" and s["ordinal"] not in claimed
    ]
    conv_end = relus[1]["input"]["id"]
    while conv_end in producer and producer[conv_end]["op"] in ("boot", "mod_switch"):
        conv_end = producer[conv_end]["inputs"][0]
    conv_scope = {}
    stack = [conv_end]
    while stack:
        value = stack.pop()
        if expression[value] == relus[0]["output_expression"]:
            continue
        step = producer.get(value)
        if step is None or step["ordinal"] in conv_scope:
            continue
        conv_scope[step["ordinal"]] = step
        stack.extend(step["inputs"])
    assert not any(s["op"] in ("boot", "mul_cc") for s in conv_scope.values())
    first_conv = dict(
        input=relus[0]["output"],
        output=values[conv_end],
        counts=dict(Counter(s["op"] for s in conv_scope.values())),
        unit_muls=sum(n in unit_steps for n in conv_scope),
    )
    rescale_owners = defaultdict(Counter)
    examples = []
    for step in execution:
        if step.get("op") != "rescale":
            continue
        ancestor = producer[step["inputs"][0]]
        while ancestor["op"] in ("rescale", "relinearize", "mod_switch"):
            ancestor = producer[ancestor["inputs"][0]]
        owner = "mul_cp_unit" if ancestor["ordinal"] in unit_steps else ancestor["op"]
        rescale_owners["relu" if step["ordinal"] in claimed else "outside_relu"][
            owner
        ] += 1
    for step in outside:
        if step["op"] != "mul_cp" or step["ordinal"] in unit_steps:
            continue
        end = step["output"]
        immediate = 0
        while True:
            nxt = [s for s in users[end] if s["op"] == "rescale"]
            if not nxt:
                break
            assert len(nxt) == 1
            immediate += 1
            end = nxt[0]["output"]
        examples.append(
            dict(
                expression=expression[step["output"]],
                ordinal=step["ordinal"],
                inputs=[values[v] for v in step["inputs"]],
                product=values[step["output"]],
                normalized=values[end],
                immediate_rescales=immediate,
            )
        )
    boot_entry = []
    for s in execution:
        if s.get("op") != "boot":
            continue
        aligned = s["inputs"][0]
        drops = 0
        while aligned in producer and producer[aligned]["op"] in (
            "rescale",
            "mod_switch",
        ):
            previous = producer[aligned]
            drops += previous["op"] == "rescale"
            aligned = previous["inputs"][0]
        alignment_plain_scale = None
        if aligned in producer and producer[aligned]["ordinal"] in unit_steps:
            alignment_plain_scale = values[producer[aligned]["inputs"][1]]["scale_log2"]
            if drops:
                assert alignment_plain_scale >= 30
        else:
            drops = 0
        v = s["inputs"][0]
        chain = []
        while v in producer and producer[v]["op"] in (
            "rescale",
            "mod_switch",
            "mul_cp",
        ):
            op = producer[v]
            if op["op"] == "mul_cp" and op["ordinal"] not in unit_steps:
                break
            chain.append(op)
            v = op["inputs"][0]
        boot_entry.append(
            dict(
                root=values[v],
                counts=dict(Counter(o["op"] for o in chain)),
                input=values[s["inputs"][0]],
                output=values[s["output"]],
                alignment_drops=drops,
                alignment_plain_scale=alignment_plain_scale,
            )
        )
    result = dict(
        name=name,
        source=str(filename),
        plan_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
        final_expression=expression[plan["final_outputs"][0]],
        total_counts=dict(
            Counter(s["op"] for s in execution if s["kind"] == "compute")
        ),
        unit_muls=len(unit_steps),
        relu_counts=dict(sum((Counter(r["counts"]) for r in relus), Counter())),
        outside_relu_counts=dict(Counter(s["op"] for s in outside)),
        outside_relu_unit_muls=sum(s["ordinal"] in unit_steps for s in outside),
        rescale_owners={k: dict(v) for k, v in rescale_owners.items()},
        outside_core_cp_immediate_rescales=dict(
            Counter(e["immediate_rescales"] for e in examples)
        ),
        outside_core_cp_examples=examples,
        first_body_conv=first_conv,
        relus=relus,
        boot_entries=boot_entry,
    )
    (OUT / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(
        name,
        {
            k: result[k]
            for k in (
                "total_counts",
                "unit_muls",
                "relu_counts",
                "outside_relu_counts",
                "outside_relu_unit_muls",
            )
        },
        flush=True,
    )
    return result


results = {name: analyze(name, path) for name, path in PLANS.items()}
a, b = results.values()
assert a["final_expression"] == b["final_expression"], "Full arithmetic DAG mismatch"
assert [r["output_expression"] for r in a["relus"]] == [
    r["output_expression"] for r in b["relus"]
]
summary = dict(
    same_full_arithmetic_dag=True,
    same_19_relu_arithmetic_dags=True,
    input_q=34,
    boot_input_q=6,
    boot_output_q=31,
    comparison={
        name: {
            k: r[k]
            for k in (
                "total_counts",
                "unit_muls",
                "relu_counts",
                "outside_relu_counts",
                "outside_relu_unit_muls",
                "rescale_owners",
                "outside_core_cp_immediate_rescales",
            )
        }
        for name, r in results.items()
    },
)
(OUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Full and 19 ReLU arithmetic DAGs agree", flush=True)
