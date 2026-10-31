"""Pure Python precision sweep on the saved production plan's 27 cut states."""

from collections import Counter
import json
import math
from pathlib import Path

from simulate import State, alignment

HERE = Path(__file__).resolve().parent
controls = json.loads((HERE / "completed-before-simulation.json").read_text())
baseline = next(row for row in controls if row["encoding_floor"] == 30)
states = [
    State(
        b["source"]["level"] + 1,
        b["source"]["scale_log2"],
        math.ceil(b["source"]["scale_log2"]),
    )
    for b in baseline["boot_entries"]
]
assert len(states) == 27
rows = []
for floor in [30, 24, 20, 15, 8, 0]:
    planned = [alignment(s, floor) for s in states]
    assert all(p is not None for p in planned)
    drops = [p[0] for p in planned]
    row = dict(
        floor=floor,
        boots_fixed=27,
        rescale_levels=sum(drops),
        modswitch_levels=sum(s.q - 6 - d for s, d in zip(states, drops)),
        total_entry_levels=sum(s.q - 6 for s in states),
        drops=dict(Counter(drops)),
        smallest_plain_scale=min(p[1] for p in planned),
    )
    assert row["rescale_levels"] + row["modswitch_levels"] == row["total_entry_levels"]
    if floor == 30:
        assert drops == [b["alignment_drops"] for b in baseline["boot_entries"]]
    rows.append(row)
    print(row)

slack = Counter(
    s.q - 6 - b["alignment_drops"] for s, b in zip(states, baseline["boot_entries"])
)
same_sources = all(
    [(b["source"]["level"], b["source"]["scale_log2"]) for b in row["boot_entries"]]
    == [(s.q - 1, s.s) for s in states]
    for row in controls
)
(HERE / "fixed-cut-precision-simulation.json").write_text(
    json.dumps(
        dict(
            rows=rows,
            baseline_slack=dict(slack),
            completed_controls_same_sources=same_sources,
        ),
        indent=2,
    )
    + "\n"
)
assert same_sources
