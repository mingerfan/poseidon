from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import chain32_probe as p

names = []
for i, (middle, leaf, small) in enumerate(
    itertools.product(
        [29, 29.5, 30, 30.1, 30.2, 30.3, 30.5, 31, 32],
        [24, 24.5, 24.8, 24.9, 25, 25.1, 25.5, 26],
        [20, 20.2],
    )
):
    logs = [30, 30, 30, middle, leaf, leaf, small] + [32] * 18
    name = f"p1shape-{i}"
    p.prepare(
        name,
        p.chain(logs),
        "Prime-size design for the two final P1 branches; unchanged compiler scale rule",
    )
    names.append(name)
rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(p.run, n, "prefix1") for n in names]):
        r = f.result()
        rows.append({k: v for k, v in r.items() if k not in ("command", "input")})
(p.OUT / "prefix-design-results.json").write_text(json.dumps(rows, indent=2) + "\n")
valid = sorted(
    [r for r in rows if not r["returncode"] and r.get("boots") == 0],
    key=lambda r: (
        max(0.0, r["output"]["scale_log2"] - 42),
        r["consumed_q"],
        r["output"]["scale_log2"],
    ),
)
chosen = [r["name"] for r in valid[:12]]
print(
    "PREFIX BEST",
    [(r["name"], r["consumed_q"], r["output"]["scale_log2"]) for r in valid[:12]],
    flush=True,
)
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(p.run, n, "stages") for n in chosen]):
        r = f.result()
        rows.append({k: v for k, v in r.items() if k not in ("command", "input")})
(p.OUT / "prefix-design-results.json").write_text(json.dumps(rows, indent=2) + "\n")
print("PREFIX DONE", len(rows), flush=True)
