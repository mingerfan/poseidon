#!/usr/bin/env python3
"""Confirm all benchmark variants used unchanged shared backend libraries."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
expected = json.loads(args.manifest.read_text())
actual = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in expected}
assert actual == expected, "Backend libraries changed after building the variants"
args.output.write_text(json.dumps(dict(unchanged=True, sha256=actual), indent=2) + "\n")
