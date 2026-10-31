"""Test small actual primes at threshold-sensitive locations, still GPU <=32 bit."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import random
import chain32_probe as p

seen = {
    hashlib.sha256(json.dumps(json.loads(f.read_text())["moduli"]).encode()).hexdigest()
    for f in (p.OUT / "candidates").glob("*/candidate.json")
}
names = []
target = sum(math.log2(q) for q in p.Q[6:31])


def add(name, logs, note):
    values = p.chain(logs)
    key = hashlib.sha256(json.dumps(values).encode()).hexdigest()
    if key in seen:
        return
    seen.add(key)
    p.prepare(name, values, note)
    names.append(name)


for low in [20, 21, 22, 23, 24, 25, 26]:
    for position in range(25):
        for background in [30, 32, (target - low) / 24]:
            logs = [background] * 25
            logs[position] = low
            add(
                f"small-{low}-{position}-{round(background * 100)}",
                logs,
                "One small NTT prime at a selected working level; fixed 25Q and Boot interface",
            )
for i in range(240):
    rng = random.Random(40000 + i)
    positions = rng.sample(range(25), 2)
    lows = [rng.choice([21, 22, 23, 24, 25, 26]), rng.choice([22, 23, 24, 25, 26])]
    background = 32 if i % 2 else (target - sum(lows)) / 23
    logs = [background] * 25
    for j, low in zip(positions, lows):
        logs[j] = low
    add(f"small-pair-{i}", logs, "Two smaller NTT primes; 25Q, other primes <=32 bit")
rows = json.loads((p.OUT / "screen-results.json").read_text())
seeds = sorted(
    [
        r
        for r in rows
        if r["phase"] == "relu" and not r["returncode"] and r.get("boots") == 0
    ],
    key=lambda r: (r["q_including_alignment"], r["consumed_q"], r["counts"]["rescale"]),
)[:8]
for k, seed in enumerate(seeds):
    original = json.loads(
        (p.OUT / "candidates" / seed["name"] / "candidate.json").read_text()
    )["consumed_log2"]
    for i in range(50):
        rng = random.Random(50000 + k * 1000 + i)
        logs = original.copy()
        for j in rng.sample(range(25), 1 + i % 2):
            logs[j] = rng.uniform(21, 27)
        add(
            f"small-mutate-{k}-{i}", logs, f"Small-prime perturbation of {seed['name']}"
        )
for first in [28, 28.5, 29, 29.5, 30, 30.5, 31, 31.5, 32]:
    for shift in range(3):
        period = [first, 60 - first, 60 - first]
        logs = [period[(j + shift) % 3] for j in range(25)]
        add(
            f"period-{round(first * 10)}-{shift}",
            logs,
            "Alternating prime sizes to change successive-square normalization",
        )


results = []


def evaluate(chosen, phase):
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(p.run, name, phase): name for name in chosen}
        for future in as_completed(jobs):
            r = future.result()
            results.append(
                {k: v for k, v in r.items() if k not in ("command", "input")}
            )
            if len(results) % 32 == 0:
                (p.OUT / "small-results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                good = [
                    r for r in results if not r["returncode"] and r.get("boots") == 0
                ]
                print(
                    "SMALL PROGRESS",
                    len(results),
                    [
                        (x["name"], x.get("consumed_q"), x.get("q_including_alignment"))
                        for x in sorted(
                            good,
                            key=lambda x: (x["q_including_alignment"], x["consumed_q"]),
                        )[:3]
                    ],
                    flush=True,
                )
    (p.OUT / "small-results.json").write_text(json.dumps(results, indent=2) + "\n")


evaluate(names, "relu")
valid = sorted(
    [r for r in results if not r["returncode"] and r.get("boots") == 0],
    key=lambda r: (r["q_including_alignment"], r["consumed_q"], r["counts"]["rescale"]),
)
evaluate([r["name"] for r in valid[:64]], "pair")
print("SMALL DONE", len(results), flush=True)
