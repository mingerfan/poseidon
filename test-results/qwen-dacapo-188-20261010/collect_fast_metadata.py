"""Collect final trace provenance without transferring the large model IR."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


root = Path(sys.argv[1])
artifacts = root / "artifacts"
source = artifacts / "block-prefill-fixed2/trace_qwen_block.mlir"
digest = hashlib.sha256()
counts = Counter()
lines = 0
with source.open("rb") as stream:
    for line in stream:
        digest.update(line)
        lines += 1
        match = re.search(rb"earth\.(?:constant|mul|add|rotate|negate)\b", line[:256])
        if match:
            counts[match[0].decode()] += 1
record = {
    "source": str(source.relative_to(root)),
    "bytes": source.stat().st_size,
    "sha256": "sha256:" + digest.hexdigest(),
    "lines": lines,
    "earth_op_counts": dict(counts),
    "total_earth_ops": sum(counts.values()),
    "frontend_slot_capacity_guard": True,
    "diagnostic_mlir_constants_elided": True,
    "runtime_plan_and_bundle_payloads_complete": True,
    "files": [],
}
for suffix in ("depth-dp-fixed", "lazy-fixed"):
    for path in sorted(artifacts.glob(f"*.{suffix}.mlir")):
        record["files"].append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    for path in sorted(artifacts.glob(f"*.{suffix}.*.runtime-plan.json")):
        bundle = path.with_name(path.name.replace(".runtime-plan.json", ".bundle"))
        manifest = json.loads((bundle / "manifest.json").read_text())
        record["files"].append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "bundle": str(bundle.relative_to(root)),
            "bundle_blobs": len(manifest["blobs"]),
            "bundle_bytes": sum(blob["byte_length"] for blob in manifest["blobs"]),
        })
report = artifacts / "fast-source-metadata.json"
report.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
