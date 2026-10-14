#!/usr/bin/env python3
"""Nsight CUDA profiler range excludes allocation, encoding and validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from nvidia import nvcomp

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--algorithm", choices=["LZ4", "Zstd"], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
case = next(c for c in manifest["cases"] if c["pattern"] == "dense")
entry = next(f for f in case["files"] if f["representation"] == "dense-ntt-gpu-uint32")
data = Path(entry["file"]).read_bytes()
stream = torch.cuda.Stream()
torch.cuda.set_stream(stream)
source = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy()).cuda()
codec = nvcomp.Codec(algorithm=args.algorithm, cuda_stream=stream.cuda_stream, uncomp_chunk_size=65536)
encoded = codec.encode(nvcomp.as_array(source, cuda_stream=stream.cuda_stream))
config = codec.decompression_config(encoded)
target = torch.empty_like(source)
output = nvcomp.as_array(target, cuda_stream=stream.cuda_stream)
for _ in range(10):
    codec.decode(encoded, out=output, decompression_config=config)
stream.synchronize()
begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
torch.cuda.profiler.start()
begin.record(stream)
for _ in range(20):
    codec.decode(encoded, out=output, decompression_config=config)
end.record(stream)
end.synchronize()
torch.cuda.profiler.stop()
if not torch.equal(source, target):
    raise RuntimeError("profiled decode differs")
args.output.write_text(json.dumps({"algorithm": args.algorithm, "plaintext_bytes": len(data),
    "calls": 20, "api_timeline_ms_per_call": begin.elapsed_time(end) / 20,
    "exact_round_trip": True, "scope": "profiling diagnostic, not the 3-run formal timing"}, indent=2) + "\n")
