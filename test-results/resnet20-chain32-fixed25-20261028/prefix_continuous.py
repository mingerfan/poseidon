from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import chain32_probe as p

seeds = json.loads((p.OUT / "prefix-low-grid-results.json").read_text())
seeds = sorted(
    [r for r in seeds if not r["returncode"] and r.get("boots") == 0],
    key=lambda r: r["output"]["scale_log2"],
)[:10]
names = []
for i in range(1800):
    rng = random.Random(60000 + i)
    if i < 900:
        logs = [20, 20.2] + [rng.uniform(24, 32) for _ in range(6)] + [32] * 17
    else:
        seed = seeds[i % len(seeds)]
        logs = [
            min(32.0, x + 1e-6)
            for x in json.loads(
                (p.OUT / "candidates" / seed["name"] / "candidate.json").read_text()
            )["consumed_log2"]
        ]
        logs[:2] = [20, 20.2]
        for j in rng.sample(range(2, 8), 1 + i % 4):
            logs[j] = max(
                24.0,
                min(32.0, logs[j] + rng.uniform(-1, 1) * [0.05, 0.2, 1.0, 3.0][i % 4]),
            )
    n = f"p1continuous-{i}"
    p.prepare(
        n,
        p.chain(logs),
        "Continuous prime-size search after the two small initial primes; fixed threshold policy",
    )
    names.append(n)
rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(p.run, n, "prefix1") for n in names]):
        r = f.result()
        rows.append({k: v for k, v in r.items() if k not in ("command", "input")})
        if len(rows) % 64 == 0:
            (p.OUT / "prefix-continuous-results.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            good = sorted(
                [r for r in rows if not r["returncode"] and r.get("boots") == 0],
                key=lambda r: r["output"]["scale_log2"],
            )[:3]
            print(
                "CONTINUOUS",
                len(rows),
                [(r["name"], r["consumed_q"], r["output"]["scale_log2"]) for r in good],
                flush=True,
            )
(p.OUT / "prefix-continuous-results.json").write_text(json.dumps(rows, indent=2) + "\n")
good = sorted(
    [r for r in rows if not r["returncode"] and r.get("boots") == 0],
    key=lambda r: r["output"]["scale_log2"],
)
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(p.run, r["name"], "stages") for r in good[:24]]):
        r = f.result()
        rows.append({k: v for k, v in r.items() if k not in ("command", "input")})
(p.OUT / "prefix-continuous-results.json").write_text(json.dumps(rows, indent=2) + "\n")
print("CONTINUOUS DONE", len(rows), flush=True)
