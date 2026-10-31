"""Classify emitted Boot operands against ReLU boundaries in elided MLIR."""

from collections import Counter
from functools import lru_cache
import json
import re
import sys

import chain32_probe as p

for name in sys.argv[1:]:
    directory = p.OUT / "candidates" / name / "full-optimized-q31"
    lines = (directory / "elided.mlir").read_text().splitlines()
    ops = {}
    cc_index = {}
    for line_number, line in enumerate(lines, 1):
        match = re.match(r'\s*(%\d+) = "(ckks|dist)\.(\w+)"\(([^)]*)\)', line)
        if not match:
            continue
        result, dialect, op, operands = match.groups()
        scalar = re.search(r"payload = dense<([^>]+)>", line)
        ops[result] = dict(
            op=op,
            args=re.findall(r"%[\w]+", operands),
            scalar=float(scalar[1]) if scalar else None,
            line=line_number,
        )
        if op == "mulcc":
            cc_index[result] = len(cc_index)
    assert len(cc_index) == 513

    @lru_cache(None)
    def strip(value):
        op = ops.get(value)
        if not op:
            return value
        if op["op"] in {
            "transfer",
            "bootstrapc",
            "rescalec",
            "modswitchc",
            "relinearize",
        }:
            return strip(op["args"][0])
        if op["op"] == "mulcp" and ops[strip(op["args"][1])]["scalar"] == 1:
            return strip(op["args"][0])
        return value

    halves = [
        v
        for v, op in ops.items()
        if op["op"] == "addcp" and ops[strip(op["args"][1])]["scalar"] == 0.5
    ]
    relus = []
    for half in halves:
        final = [
            v
            for v, op in ops.items()
            if op["op"] == "mulcc" and half in [strip(a) for a in op["args"]]
        ]
        assert len(final) == 1
        final = final[0]
        source = next(strip(a) for a in ops[final]["args"] if strip(a) != half)
        relus.append(
            dict(
                number=len(relus) + 1,
                source=source,
                final=final,
                start=ops[source]["line"],
                end=ops[final]["line"],
            )
        )
    assert len(relus) == 19

    rows = []
    for value, op in ops.items():
        if op["op"] != "bootstrapc":
            continue
        root = strip(op["args"][0])
        inputs = [r for r in relus if r["source"] == root]
        inside = [r for r in relus if r["start"] < op["line"] < r["end"]]
        if inputs:
            relu = inputs[0]["number"]
            where = "relu_input"
        elif inside:
            relu = inside[0]["number"]
            completed = sum(1 for v in cc_index if ops[v]["line"] < op["line"])
            within = completed - 27 * (relu - 1)
            where = (
                "P1_boundary"
                if within == 8
                else "P2_boundary"
                if within == 16
                else "P3_boundary"
                if within == 26
                else "polynomial_interior"
            )
        else:
            before = [r for r in relus if r["end"] < op["line"]]
            relu = before[-1]["number"] if before else 0
            where = "between_relus"
        rows.append(
            dict(
                relu=relu,
                where=where,
                ssa=value,
                input_root=root,
                mlir_line=op["line"],
                input_root_op=ops[root]["op"],
            )
        )
    assert len(rows) == 27
    result = dict(
        name=name,
        boots=rows,
        counts=dict(Counter(r["where"] for r in rows)),
        method="Strip scale-management and transfer nodes. Identify 19 ReLU final x*(P3+0.5) operands; classify interior boundaries by the unchanged 8/8/10/1 MulCC groups. This is structural placement evidence, not encrypted accuracy.",
    )
    (directory / "boot-boundaries.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, result["counts"], [(r["relu"], r["where"]) for r in rows], flush=True)
