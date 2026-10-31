"""Run both production Boot placers for a saved candidate list on 188Server."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys

import chain32_probe as p

label = sys.argv[1]
names = json.loads((p.OUT / (label + "-selected.json")).read_text())
seen = set()
selected = []
for name in names:
    values = tuple(
        json.loads((p.OUT / "candidates" / name / "candidate.json").read_text())[
            "moduli"
        ]
    )
    if values not in seen:
        seen.add(values)
        selected.append(name)


def evaluate(name):
    return [p.run(name, "full", strategy) for strategy in ("greedy", "optimized")]


rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs = {pool.submit(evaluate, name): name for name in selected}
    for future in as_completed(jobs):
        rows.extend(future.result())
        (p.OUT / (label + "-results.json")).write_text(
            json.dumps(rows, indent=2) + "\n"
        )
        print("FULL SELECTED DONE", jobs[future], flush=True)
