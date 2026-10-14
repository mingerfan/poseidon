#!/usr/bin/env python3
"""Extract complete execution records from an existing indented RuntimePlan.

This reads the baseline exporter layout (indent=2), without loading the plan or
regenerating the model. Samples are JSON objects for the --dom I/O benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[100000, 200000, 400000, 800000])
    args = parser.parse_args()
    counts = sorted(set(args.counts))
    if not counts or counts[0] <= 0:
        parser.error("counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {n: (args.output_dir / f"execution-{n}.json").open("wb") for n in counts}
    digests = {n: hashlib.sha256() for n in counts}

    def write(n, data):
        outputs[n].write(data)
        digests[n].update(data)

    try:
        for n in counts:
            write(n, b'{"execution":[\n')
        records = 0
        with args.source.open("rb") as source:
            for line in source:
                if line == b'  "execution": [\n':
                    break
            else:
                raise ValueError("expected an indent=2 RuntimePlan execution array")
            block = []
            for line in source:
                block.append(line)
                if line not in (b"    },\n", b"    }\n"):
                    continue
                record = b"".join(block).rstrip(b"\n").removesuffix(b",")
                block.clear()
                if not isinstance(json.loads(record), dict):
                    raise ValueError("execution record must be an object")
                records += 1
                for n in list(outputs):
                    write(n, (b",\n" if records > 1 else b"") + record)
                    if n == records:
                        write(n, b"\n]}\n")
                        outputs.pop(n).close()
                        path = args.output_dir / f"execution-{n}.json"
                        print(json.dumps({"records": n, "bytes": path.stat().st_size,
                                          "source_sha256": "sha256:" + digests[n].hexdigest(),
                                          "file": str(path)}), flush=True)
                if records == counts[-1]:
                    break
                if line == b"    }\n":
                    raise ValueError("execution array has fewer records than requested")
            else:
                raise ValueError("truncated execution array")
    finally:
        for output in outputs.values():
            output.close()


if __name__ == "__main__":
    main()
