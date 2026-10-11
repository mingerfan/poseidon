#!/usr/bin/env python3
"""Optional nvCOMP GPU LZ4/Zstd decode benchmark; not Tensor Core decode.

nvCOMP encodes the corpus during untimed setup. Native chunk headers count
towards compressed size. Output and decode configuration are reused. GPU
event times include Python/API submission gaps, with wall time also saved.
This does not claim interchange with a CPU-generated whole-buffer stream.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time


def stats(values):
    return {"median_ms": statistics.median(values), "samples_ms": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--inner", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch, args.repeat, args.inner) < 1:
        parser.error("counts must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import torch
    from nvidia import nvcomp
    import importlib.metadata
    stream = torch.cuda.current_stream()
    manifest = json.loads(args.manifest.read_text())
    if manifest["batch"] != 1:
        raise ValueError("use a one-plaintext corpus; --batch tiles its payload")
    result = {"scope": "nvCOMP CUDA SM decode, not Tensor Cores; no concurrent foreground",
              "input_scope": "batch tiles the same plaintext payload; not independent model weights",
              "timing_scope": "cached decode config and output; event includes Python/API dispatch gaps",
              "format_scope": "nvCOMP native 64KiB chunks, encoded on GPU during setup; not CPU stream interoperability",
              "gpu": torch.cuda.get_device_name(), "q_limbs": manifest["q_limbs"],
              "batch": args.batch, "repeat": args.repeat, "inner": args.inner,
              "torch_version": torch.__version__,
              "nvcomp_version": importlib.metadata.version("nvidia-nvcomp-cu13"),
              "chunk_bytes": 65536, "minimum_warmup_ms": 100, "rows": []}
    cases = []
    for case in manifest["cases"]:
        for entry in case["files"]:
            if any(entry["representation"].endswith(x) for x in
                   ("ntt-gpu-uint32", "fft-fp64", "integer-int64", "integer-bitpacked")):
                for algorithm in ("LZ4", "Zstd"):
                    cases.append((case["pattern"], entry, algorithm))
    random.Random(20261012 + args.batch).shuffle(cases)
    for pattern, entry, algorithm in cases:
        data = Path(entry["file"]).read_bytes() * args.batch
        original = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())
        source = original.to("cuda")
        codec = nvcomp.Codec(algorithm=algorithm, cuda_stream=stream.cuda_stream,
                             uncomp_chunk_size=65536)
        wrapped = nvcomp.as_array(source, cuda_stream=stream.cuda_stream)
        compressed = codec.encode(wrapped)
        config = codec.decompression_config(compressed)
        target = torch.empty_like(source)
        codec.decode(compressed, out=target, decompression_config=config)
        stream.synchronize()
        if not torch.equal(target.cpu(), original):
            raise RuntimeError("GPU codec exact round trip failed")
        compressed_cpu = np.asarray(compressed.cpu()).copy()
        host = torch.from_numpy(compressed_cpu).pin_memory()
        uploaded = torch.empty_like(host, device="cuda")
        upload_array = nvcomp.as_array(uploaded, cuda_stream=stream.cuda_stream)
        row = {"pattern": pattern, "representation": entry["representation"],
               "algorithm": algorithm, "raw_bytes": len(data),
               "source_sha256": hashlib.sha256(data).hexdigest(),
               "compressed_bytes": compressed_cpu.nbytes,
               "ratio": len(data)/compressed_cpu.nbytes, "exact_round_trip": True}
        for with_h2d in (False, True):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_ms, wall_ms = [], []
            warmed = 0
            while len(gpu_ms) < args.repeat:
                sample = warmed >= 100
                wall_start = time.perf_counter_ns()
                start.record(stream)
                for _ in range(args.inner):
                    if with_h2d:
                        uploaded.copy_(host, non_blocking=True)
                    codec.decode(upload_array if with_h2d else compressed, out=target,
                                 decompression_config=config)
                end.record(stream)
                end.synchronize()
                event_ms = start.elapsed_time(end)
                wall = (time.perf_counter_ns() - wall_start)/1e6
                if sample:
                    gpu_ms.append(event_ms/args.inner)
                    wall_ms.append(wall/args.inner)
                else:
                    warmed += event_ms
            if not torch.equal(target.cpu(), original):
                raise RuntimeError("timed GPU codec result differs")
            median = statistics.median(gpu_ms)
            row["with_h2d" if with_h2d else "device_only"] = {
                "gpu": stats(gpu_ms), "wall": stats(wall_ms),
                "decompressed_output_gb_per_second": len(data)/(median*1e6),
                "ms_per_plaintext": median/args.batch}
        result["rows"].append(row)
        print(pattern, entry["representation"], algorithm, round(row["ratio"], 3), flush=True)
        del codec, source, target, wrapped, compressed, config, uploaded, upload_array
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
