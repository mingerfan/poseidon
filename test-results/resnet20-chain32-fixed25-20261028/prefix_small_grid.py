from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import chain32_probe as p

# The direct linear coefficient starts at s=80. Two available small NTT
# primes remove ~39.755 bits, rather than the >=48 residual after one 32-bit Q.
names = []
for i, values in enumerate(itertools.product([22, 24, 28, 30, 32], repeat=5)):
    if values.count(22) > 1:
        continue  # Only one additional unused NTT prime exists below 2^22.
    logs = [20, 20.2] + list(values) + [32] * 18
    name = f"p1lowgrid-{i}"
    p.prepare(
        name,
        p.chain(logs),
        "Two initial small Q and a finite grid of the next five primes; fixed 25Q",
    )
    names.append(name)
results = []
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs = {pool.submit(p.run, n, "prefix1"): n for n in names}
    for f in as_completed(jobs):
        r = f.result()
        results.append({k: v for k, v in r.items() if k not in ("command", "input")})
        if len(results) % 64 == 0:
            (p.OUT / "prefix-low-grid-results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            good = [r for r in results if not r["returncode"] and r.get("boots") == 0]
            best = sorted(
                good, key=lambda r: (r["output"]["scale_log2"], r["consumed_q"])
            )[:3]
            print(
                "LOW GRID",
                len(results),
                [(r["name"], r["consumed_q"], r["output"]["scale_log2"]) for r in best],
                flush=True,
            )
(p.OUT / "prefix-low-grid-results.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
good = sorted(
    [r for r in results if not r["returncode"] and r.get("boots") == 0],
    key=lambda r: (r["output"]["scale_log2"], r["consumed_q"]),
)
chosen = [r["name"] for r in good[:24]]
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(p.run, n, "stages") for n in chosen]):
        r = f.result()
        results.append({k: v for k, v in r.items() if k not in ("command", "input")})
(p.OUT / "prefix-low-grid-results.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print("LOW GRID DONE", len(results), flush=True)
