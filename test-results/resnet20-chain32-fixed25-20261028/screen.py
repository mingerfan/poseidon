"""Screen real-prime chains with the actual threshold compiler; no scale goals."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import random
import time
import chain32_probe as p

started = time.monotonic()
seen = {hashlib.sha256(json.dumps(p.Q).encode()).hexdigest()}
queue = []


def add(name, values, note):
    key = hashlib.sha256(json.dumps(values).encode()).hexdigest()
    if key in seen:
        return
    seen.add(key)
    p.prepare(name, values, note)
    queue.append(name)


for i in range(64):
    a = p.Q[6:31].copy()
    random.Random(i).shuffle(a)
    add(
        f"perm-{i}",
        p.Q[:6] + a + p.Q[31:],
        "Exact same primes and total modulus; reordered working prefix",
    )
for i, value in enumerate(
    [
        27,
        28,
        29,
        29.5,
        30,
        30.2,
        30.5,
        30.8,
        31,
        31.01,
        31.05,
        31.1,
        31.25,
        31.5,
        31.65,
        31.75,
        31.85,
        31.9,
        31.95,
        31.99,
        32,
    ]
):
    add(
        f"uniform-{i}",
        p.chain([value] * 25),
        f"Uniform target actual log2(q)={value}; fixed 25Q",
    )
for low in (28, 30, 31):
    for number in range(1, 25):
        for order in range(3):
            logs = [32] * number + [low] * (25 - number)
            if order == 1:
                logs.reverse()
            elif order == 2:
                random.Random(low * 1000 + number).shuffle(logs)
            add(
                f"mix-{low}-{number}-{order}",
                p.chain(logs),
                "Mixed actual prime sizes; fixed 25Q",
            )
for i in range(180):
    rng = random.Random(10000 + i)
    logs = [sum(math.log2(q) for q in p.Q[6:31]) / 25] * 25
    for _ in range(50 + i % 5 * 50):
        a, b = rng.sample(range(25), 2)
        delta = min(rng.random() * (0.5 + i % 5), 32 - logs[a], logs[b] - 27)
        if delta > 0:
            logs[a] += delta
            logs[b] -= delta
    add(
        f"matched-{i}",
        p.chain(logs),
        "Working actual log budget no greater than the original, before prime rounding",
    )
for i in range(180):
    rng = random.Random(20000 + i)
    width = [0.02, 0.1, 0.5, 1.0, 2.0, 4.0, 5.0][i % 7]
    logs = [32 - rng.random() * width for _ in range(25)]
    add(
        f"nearmax-{i}",
        p.chain(logs),
        "Actual logs sampled below GPU 32-bit ceiling; fixed 25Q",
    )


def key(row):
    return (
        row.get("returncode", 1) != 0 or row.get("boots", 1) != 0,
        row.get("q_including_alignment", 100),
        row.get("consumed_q", 100),
        row.get("counts", {}).get("rescale", 10000),
    )


rows = []


def evaluate(names, phase):
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(p.run, name, phase): name for name in names}
        for future in as_completed(futures):
            row = future.result()
            rows.append({k: v for k, v in row.items() if k not in ("command", "input")})
            if len(rows) % 32 == 0:
                (p.OUT / "screen-results.json").write_text(
                    json.dumps(rows, indent=2) + "\n"
                )
                best = sorted([r for r in rows if r["phase"] == phase], key=key)[:3]
                print(
                    "PROGRESS",
                    len(rows),
                    phase,
                    round(time.monotonic() - started, 1),
                    [
                        (r["name"], r.get("consumed_q"), r.get("q_including_alignment"))
                        for r in best
                    ],
                    flush=True,
                )
    (p.OUT / "screen-results.json").write_text(json.dumps(rows, indent=2) + "\n")


evaluate(queue, "relu")
seeds = sorted(
    [r for r in rows if not r["returncode"] and r.get("boots") == 0], key=key
)[:10]
mutants = []
for seed_index, seed in enumerate(seeds):
    values = json.loads(
        (p.OUT / "candidates" / seed["name"] / "candidate.json").read_text()
    )["moduli"]
    original = [math.log2(q) for q in reversed(values[6:31])]
    for i in range(32):
        rng = random.Random(30000 + seed_index * 1000 + i)
        logs = original.copy()
        for j in rng.sample(range(25), 1 + i % 8):
            logs[j] = max(
                27.0,
                min(32.0, logs[j] + rng.uniform(-1, 1) * [0.02, 0.1, 0.5, 2][i % 4]),
            )
        name = f"mutate-{seed_index}-{i}"
        count = len(queue)
        add(
            name,
            p.chain(logs),
            f"Local mutation around {seed['name']}; unchanged count and interface",
        )
        if len(queue) > count:
            mutants.append(name)
evaluate(mutants, "relu")
ranked = sorted(
    [r for r in rows if not r["returncode"] and r.get("boots") == 0], key=key
)
selected = list(dict.fromkeys(r["name"] for r in ranked))[:36]
evaluate(selected, "pair")
print("SCREEN DONE", len(rows), round(time.monotonic() - started, 1), flush=True)
