"""Fixed Q50/P25, 25 working primes, threshold-only chain experiments on 188Server."""

import argparse
from collections import Counter
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import resource
import shlex
import subprocess
import time

ROOT = Path("/home/xuming/poseidon-manual-boot-20261026")
AUDIT = ROOT / "resnet20-optimized-20261027T055506Z/placement-audit"
EXPERIMENT = AUDIT / "target-scale-experiment"
OUT = EXPERIMENT / "chain32-fixed25-20261028"
RUNTIME = ROOT / "src/third_party/ckks-runtime"
SPEC = json.loads((ROOT / "alignment-tests/operator-spec.json").read_text())
FIXTURE = json.loads((RUNTIME / "tests/fixtures/resnet20-manual-q50.json").read_text())
Q = SPEC["context"]["rns_moduli"]
STRIDE = 2 * SPEC["context"]["poly_degree"]
COMPILER = EXPERIMENT / "hecate-opt-wrapper"
GENERATOR = RUNTIME / "integrations/dacapo/generate_model_artifacts.py"


@lru_cache(maxsize=None)
def prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def chain(logs):
    """logs are ordered from the first working prime removed to the last."""
    assert len(logs) == 25 and max(logs) <= 32
    used = set(Q[:6] + Q[31:] + FIXTURE["p_moduli"])
    working = []
    for size in logs:
        q = (int(2**size) - 2) // STRIDE * STRIDE + 1
        while q > STRIDE and (q in used or not prime(q)):
            q -= STRIDE
        assert q > STRIDE and q.bit_length() <= 32
        working.append(q)
        used.add(q)
    return Q[:6] + list(reversed(working)) + Q[31:]


def prepare(name, values, note):
    path = OUT / "candidates" / name
    path.mkdir(parents=True, exist_ok=True)
    assert len(values) == 50 and values[:6] == Q[:6] and values[31:] == Q[31:]
    assert len(set(values + FIXTURE["p_moduli"])) == 75
    assert all(
        q.bit_length() <= 32 and q % STRIDE == 1 and prime(q)
        for q in values + FIXTURE["p_moduli"]
    )
    fixture = copy.deepcopy(FIXTURE)
    fixture["rns_moduli"] = values
    fixture_path = path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n")
    spec = copy.deepcopy(SPEC)
    spec["spec_id"] = "resnet20-chain32-" + name
    spec["context"].update(
        context_id=spec["spec_id"],
        rns_moduli=values,
        rns_moduli_log2=[q.bit_length() for q in values],
        max_modulus_log2=max(q.bit_length() for q in values),
    )
    spec["provenance"].update(
        path=str(fixture_path),
        source_sha256="sha256:" + hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
    )
    (path / "operator-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    diagnostic = copy.deepcopy(spec)
    diagnostic["levels"]["lower_bound"] = 0
    profile = diagnostic["boot_profiles"][0]
    profile.pop("input_scale_log2")
    profile.update(
        input_level_min=0, input_level_max=49, latency_us_by_input_level=[85000] * 50
    )
    (path / "diagnostic-spec.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    metadata = dict(
        name=name,
        note=note,
        working_count=25,
        output_q=31,
        boot_input_q=6,
        moduli=values,
        consumed_log2=[math.log2(q) for q in reversed(values[6:31])],
        working_log2=sum(math.log2(q) for q in values[6:31]),
        baseline_working_log2=sum(math.log2(q) for q in Q[6:31]),
        max_modulus_bits=max(q.bit_length() for q in values + FIXTURE["p_moduli"]),
    )
    (path / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def boot_alignment(level, scale, values):
    if abs(scale - 40) <= 1e-8:
        return dict(alignment_drops=0, alignment_fits=level >= 5)
    plain = 40 - scale
    for drops in range(1, 20):
        plain += math.log2(values[5 + drops])
        if plain >= 30:
            return dict(
                alignment_drops=drops,
                alignment_fits=level >= 5 + drops,
                alignment_plain_scale=plain,
            )
    raise ValueError("no alignment within the diagnostic prefix")


def run(name, phase, strategy="greedy", initial=31):
    candidate = OUT / "candidates" / name
    meta = json.loads((candidate / "candidate.json").read_text())
    diagnostic = phase != "full"
    suffix = phase if diagnostic else f"full-{strategy}-q{initial}"
    destination = candidate / suffix
    destination.mkdir(exist_ok=False)
    spec_path = candidate / (
        "diagnostic-spec.json" if diagnostic else "operator-spec.json"
    )
    source = (
        AUDIT / "one-relu-tagged.mlir" if phase == "relu" else OUT / "relu-conv.mlir"
    )
    if phase == "stages":
        source = OUT / "relu-stages.mlir"
    if phase in ("prefix1", "prefix2"):
        source = OUT / (phase + ".mlir")
    if phase == "full":
        source = Path("/home/xuming/poseidon-mixed-q-20261026/data/ResNet20.mlirbc")
    command = [
        "python3",
        str(GENERATOR),
        "resnet20" if phase == "full" else "mlp",
        "--operator-spec",
        str(spec_path),
        "--compiler-profile",
        str(ROOT / "alignment-tests/compiler.json"),
        "--hecate-opt",
        str(COMPILER),
        "--traced-mlir",
        str(source),
        "--output-dir",
        str(destination),
        "--device-counts",
        "1",
        "--plan-format",
        "json",
        "--boot-placement",
        strategy,
        "--waterline",
        "40",
        "--input-level",
        str(initial - 1),
        "--inline-payload-max-bytes",
        "0",
        "--dry-run",
    ]
    generated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert generated.returncode == 0, generated.stderr
    actual = shlex.split(generated.stdout.strip())
    actual.remove("--enable-debug-printer")
    actual[actual.index("-o") + 1] = str(destination / "model.optimized.mlirbc")
    actual.append("--emit-bytecode")
    assert not any(
        x.startswith("--native-target-") or x.startswith("--scale-strategy")
        for x in actual
    )
    report = dict(
        name=name,
        phase=phase,
        strategy=strategy,
        initial_q=initial,
        diagnostic=diagnostic,
        command=actual,
        compiler_sha256=(EXPERIMENT / "binary.sha256").read_text().strip(),
    )
    started = time.monotonic()
    with (destination / "compile.log").open("w") as log:
        child = subprocess.Popen(actual, stdout=log, stderr=subprocess.STDOUT)
        report["pid"] = child.pid
        (destination / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            code = child.wait(timeout=600 if strategy == "optimized" else 120)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            code = -9
            report["timeout"] = True
    report.update(
        returncode=code,
        seconds=time.monotonic() - started,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    if not code:
        plan_path = next(destination.glob("*.runtime-plan.json"))
        plan = json.loads(plan_path.read_text())
        values = {v["id"]: v for v in plan["values"]}
        boots = [s for s in plan["execution"] if s.get("op") == "boot"]
        output = values[plan["final_outputs"][-1]]
        report.update(
            boots=len(boots),
            counts=dict(Counter(s.get("op", s["kind"]) for s in plan["execution"])),
            input=values[plan["external_inputs"][0]],
            output=output,
            stage_outputs=[
                dict(
                    level=values[x]["level"],
                    scale_log2=values[x]["scale_log2"],
                    **(
                        {"consumed_q": initial - 1 - values[x]["level"]}
                        if not boots
                        else {}
                    ),
                )
                for x in plan["final_outputs"]
            ],
            plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        )
        if not boots:
            report.update(
                consumed_q=initial - 1 - output["level"],
                **boot_alignment(output["level"], output["scale_log2"], meta["moduli"]),
            )
            report["q_including_alignment"] = (
                report["consumed_q"] + report["alignment_drops"]
            )
        if not diagnostic:
            for boot in boots:
                before, after = values[boot["inputs"][0]], values[boot["output"]]
                assert before["level"] == 5 and abs(before["scale_log2"] - 40) < 1e-8
                assert after["level"] == 30 and after["scale_log2"] == 40
            validation = subprocess.run(
                [str(ROOT / "validate-plan"), str(plan_path), str(spec_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            report.update(
                validation=validation.stdout + validation.stderr,
                validation_code=validation.returncode,
            )
            assert validation.returncode == 0, report["validation"]
        max_error = 0.0
        for step in plan["execution"]:
            if step.get("op") != "rescale":
                continue
            before, after = values[step["inputs"][0]], values[step["output"]]
            expected = before["scale_log2"] - sum(
                math.log2(q)
                for q in meta["moduli"][after["level"] + 1 : before["level"] + 1]
            )
            max_error = max(max_error, abs(expected - after["scale_log2"]))
        assert max_error < 1e-8
        report["max_rescale_formula_error"] = max_error
    (destination / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("command", "input", "output")
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument(
        "phase", choices=["relu", "pair", "stages", "prefix1", "prefix2", "full"]
    )
    parser.add_argument("--strategy", choices=["greedy", "optimized"], default="greedy")
    parser.add_argument("--initial", type=int, choices=[31, 34], default=31)
    arguments = parser.parse_args()
    run(arguments.name, arguments.phase, arguments.strategy, arguments.initial)
