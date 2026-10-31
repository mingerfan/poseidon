from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import chain32_probe as p

rows = json.loads((p.OUT / "screen-results.json").read_text())
valid = [r for r in rows if not r["returncode"] and r.get("boots") == 0]
def key(row):
    return (
        row.get("q_including_alignment", 100),
        row.get("consumed_q", 100),
        row["counts"]["rescale"],
    )
pairs = sorted([r for r in valid if r["phase"] == "pair"], key=key)
selected = [r["name"] for r in pairs[:6]]
for prefix in ["perm-", "matched-", "nearmax-", "uniform-"]:
    group = sorted(
        [r for r in valid if r["phase"] == "relu" and r["name"].startswith(prefix)],
        key=key,
    )
    if group:
        selected.append(group[0]["name"])
for r in sorted(
    [r for r in valid if r["phase"] == "relu"], key=lambda r: r["counts"]["rescale"]
)[:2]:
    selected.append(r["name"])
selected = list(dict.fromkeys(selected))
(p.OUT / "full-batch-selected.json").write_text(json.dumps(selected, indent=2) + "\n")


def evaluate(name):
    return [p.run(name, "full", strategy) for strategy in ("greedy", "optimized")]


results = []
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs = {pool.submit(evaluate, name): name for name in selected}
    for future in as_completed(jobs):
        results += future.result()
        (p.OUT / "full-batch-results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(
            "FULL CASE DONE",
            jobs[future],
            [(r["strategy"], r.get("boots"), r["returncode"]) for r in results[-2:]],
            flush=True,
        )
