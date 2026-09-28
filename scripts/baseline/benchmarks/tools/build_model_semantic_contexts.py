"""Prepare additional mathematical context tasks without changing a frozen corpus.

These are planning artifacts, not accepted execution results or a second semantic
ledger. Run this only against a hash-verified existing suite. The canonical
coverage ledger must explicitly adopt a bundle in a later release.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from benchmark_graph import canonical, digest, signature, validate
from benchmark_semantics import features

RECIPES = ("input_affine", "output_residual", "input_negate", "output_negate",
           "input_affine_output_residual", "input_negate_output_residual")


def augmented(model, recipe, name):
    """Compose real operations; never rename/rew eight to manufacture diversity."""
    if recipe not in RECIPES:
        raise ValueError("Unknown context recipe")
    g = copy.deepcopy(model)
    g["id"] = name
    occupied = {x["name"] for x in g["inputs"]} | set(g["constants"])
    occupied |= {n for node in g["nodes"] for n in node["outputs"]}
    occupied |= {node["id"] for node in g["nodes"]}
    def fresh(stem):
        value = "context_" + stem
        while value in occupied:
            value += "x"
        occupied.add(value)
        return value
    def node(op, refs, output):
        return dict(id=fresh("node"), op=op, inputs=refs, attrs={}, outputs=[output])
    before = []
    replacements = {}
    if recipe.startswith("input_"):
        affine = recipe.startswith("input_affine")
        if affine:
            scale, shift = fresh("scale"), fresh("shift")
            g["constants"].update({scale: -0.5, shift: 0.125})
        for inp in g["inputs"]:
            ref = inp["name"]
            value = fresh("input")
            if affine:
                tmp = fresh("scaled")
                before.extend([node("multiply", [ref, scale], tmp),
                               node("add", [tmp, shift], value)])
            else:
                before.append(node("negate", [ref], value))
            replacements[ref] = value
        for n in g["nodes"]:
            n["inputs"] = [replacements.get(ref, ref) for ref in n["inputs"]]
        g["nodes"] = before + g["nodes"]
    if "output_residual" in recipe:
        # A shared value feeds both branches, which are merged into a named output.
        ref = g["outputs"][0]["value"]
        squared, value = fresh("square"), fresh("residual")
        g["nodes"].extend([node("square", [ref], squared),
                           node("add", [squared, ref], value)])
        g["outputs"][0]["value"] = value
    elif recipe == "output_negate":
        ref = g["outputs"][0]["value"]
        value = fresh("output")
        g["nodes"].append(node("negate", [ref], value))
        g["outputs"][0]["value"] = value
    validate(g)
    return g


def read_suite(folder):
    index = json.loads((folder / "index.json").read_text())
    rows = []
    for shard in index["shards"]:
        name = shard["file"]
        if Path(name).name != name:
            raise ValueError("Unsafe shard")
        path = folder / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError("Changed shard")
        rows.extend(json.loads(path.read_text())["rows"])
    ledger_path = folder / "coverage.json"
    if hashlib.sha256(ledger_path.read_bytes()).hexdigest() != index["coverage_sha256"]:
        raise ValueError("Changed coverage ledger")
    if len(rows) != 1200:
        raise ValueError("Unexpected corpus size")
    for row in rows:
        g = row["model"]
        validate(g)
        if (digest(g), signature(g), signature(g, True)) != (
                row["model_sha256"], row["signature"], row["topology"]):
            raise ValueError("Changed model identity")
    return rows, index, json.loads(ledger_path.read_text())


def build(rows, ledger, source_binding):
    by_id = {r["model"]["id"]: r for r in rows}
    group_splits = {}
    signatures = {r["signature"] for r in rows}
    for row in rows:
        group = row["topology"]
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError("Existing split leakage")
        group_splits[group] = row["split"]
    tasks, extras = [], {}
    for requirement in ledger["requirements"]:
        if requirement["layer"] != "model":
            continue
        feature = requirement["id"]
        candidates = sorted((by_id[n] for n in requirement["models"]),
                            key=lambda r: (len(r["model"]["nodes"]), r["model"]["id"]))
        chosen, groups = [], set()
        for row in candidates:
            if feature not in features(row["model"]):
                raise ValueError("Incorrect existing feature mapping")
            if row["topology"] in groups:
                continue
            chosen.append((row, None))
            groups.add(row["topology"])
            if len(chosen) == 3:
                break
        if not candidates:
            raise ValueError("Missing positive model for " + feature)
        if len(chosen) < 3:
            for parent in candidates:
                for recipe in RECIPES:
                    name = "semantic_context_" + digest(
                        dict(parent=parent["model_sha256"], recipe=recipe))[:24]
                    try:
                        model = augmented(parent["model"], recipe, name)
                    except ValueError:
                        # Only declared shape/type/resource constraints guide selection;
                        # no DSL lowering or observed execution outcome is consulted.
                        continue
                    topology = signature(model, True)
                    sig = signature(model)
                    if topology in groups or sig in signatures:
                        continue
                    if topology in group_splits and group_splits[topology] != parent["split"]:
                        continue
                    if feature not in features(model):
                        raise ValueError("Augmentation erased target feature")
                    row = dict(model=model, model_sha256=digest(model),
                               signature=sig, topology=topology, split=parent["split"])
                    provenance = dict(parent_id=parent["model"]["id"],
                                      parent_sha256=parent["model_sha256"], recipe=recipe)
                    extras[name] = dict(**row, provenance=provenance)
                    group_splits[topology] = parent["split"]
                    signatures.add(sig)
                    chosen.append((row, provenance))
                    groups.add(topology)
                    if len(chosen) == 3:
                        break
                if len(chosen) == 3:
                    break
        if len(chosen) != 3:
            raise ValueError("Could not construct three legal contexts for " + feature)
        for position, (row, provenance) in enumerate(chosen):
            task = dict(id="math_" + digest(feature)[:16] + "_" + str(position),
                        requirement=feature, model_id=row["model"]["id"],
                        model_sha256=row["model_sha256"], topology=row["topology"],
                        split=row["split"], supplemental=provenance is not None,
                        kind="mathematical_semantic_context", generation_track="not_assigned",
                        construction_requirement=None, state="planned_not_run",
                        acceptance=["independent_dual_reference", "actual_frontend_and_compiler",
                                    "real_SEAL_decrypt_all_frozen_inputs"],
                        claim_limit="Numerical model semantics; does not prove use of a particular DSL spelling")
            task["task_sha256"] = digest(task)
            tasks.append(task)
    result = dict(schema=1, authoritative_ledger=False,
                  intended_ledger_adoption="requires_new_explicit_frozen_release",
                  source_binding=source_binding,
                  original_models=1200, supplemental_models_counted_in_original=False,
                  selection_uses_agent_or_FHE_success=False,
                  requirements=len(tasks)//3, tasks=tasks,
                  supplemental_models=sorted(extras.values(), key=lambda r: r["model"]["id"]),
                  all_executed=False, paid_calls=0)
    result["bundle_sha256"] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing bundle; choose a new output")
    rows, index, ledger = read_suite(args.suite)
    sources = [Path(__file__).resolve(), BASE/"benchmark_graph.py",
               BASE/"benchmark_semantics.py", BASE/"benchmark_suite.py"]
    binding = dict(index_sha256=hashlib.sha256((args.suite/"index.json").read_bytes()).hexdigest(),
                   coverage_sha256=index["coverage_sha256"],
                   generator_sources={str(p.relative_to(BASE.parents[1])):
                                      hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    result = build(rows, ledger, binding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(requirements=result["requirements"], tasks=len(result["tasks"]),
                          supplementary_models=len(result["supplemental_models"]),
                          bundle_sha256=result["bundle_sha256"], state="planned_not_run")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
