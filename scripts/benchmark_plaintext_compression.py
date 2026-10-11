#!/usr/bin/env python3
"""Lossless CPU Zstd/LZ4 comparison on the CUDA benchmark's actual corpus.

Uses system shared libraries through ctypes; no Python codec dependencies.
The input manifest is a poseidon_gpu_plaintext_compression_bench report.
CPU codec speeds are not GPU or Tensor Core speeds.
"""
import argparse
import ctypes as ct
import ctypes.util
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time


def bind(lib, name, args, result):
    method = getattr(lib, name)
    method.argtypes, method.restype = args, result
    return method


class Codecs:
    def __init__(self):
        self.zstd = ct.CDLL(ctypes.util.find_library("zstd"))
        self.lz4 = ct.CDLL(ctypes.util.find_library("lz4"))
        ptr, size, integer = ct.c_void_p, ct.c_size_t, ct.c_int
        self.zbound = bind(self.zstd, "ZSTD_compressBound", [size], size)
        self.zcompress = bind(self.zstd, "ZSTD_compress", [ptr, size, ptr, size, integer], size)
        self.zdecode = bind(self.zstd, "ZSTD_decompress", [ptr, size, ptr, size], size)
        self.zerror = bind(self.zstd, "ZSTD_isError", [size], ct.c_uint)
        self.zmessage = bind(self.zstd, "ZSTD_getErrorName", [size], ct.c_char_p)
        self.lbound = bind(self.lz4, "LZ4_compressBound", [integer], integer)
        self.lcompress = bind(self.lz4, "LZ4_compress_default", [ptr, ptr, integer, integer], integer)
        self.ldecode = bind(self.lz4, "LZ4_decompress_safe", [ptr, ptr, integer, integer], integer)
        self.versions = {
            "zstd": bind(self.zstd, "ZSTD_versionString", [], ct.c_char_p)().decode(),
            "lz4": bind(self.lz4, "LZ4_versionString", [], ct.c_char_p)().decode(),
        }

    def benchmark(self, data, algorithm, level, repeat):
        source = ct.create_string_buffer(data, len(data))
        bound = self.zbound(len(data)) if algorithm == "zstd" else self.lbound(len(data))
        compressed, decoded = ct.create_string_buffer(bound), ct.create_string_buffer(len(data))
        def encode():
            if algorithm == "zstd":
                result = self.zcompress(compressed, bound, source, len(data), level)
                if self.zerror(result):
                    raise RuntimeError(self.zmessage(result).decode())
            else:
                result = self.lcompress(source, compressed, len(data), bound)
                if result <= 0:
                    raise RuntimeError("LZ4 compression failed")
            return result
        size = encode()
        def decode():
            if algorithm == "zstd":
                result = self.zdecode(decoded, len(data), compressed, size)
                if self.zerror(result):
                    raise RuntimeError(self.zmessage(result).decode())
            else:
                result = self.ldecode(compressed, decoded, size, len(data))
            if result != len(data):
                raise RuntimeError("decompressed size mismatch")
        decode()
        if decoded.raw != data:
            raise RuntimeError("codec round trip differs")
        timings = {}
        for name, operation in (("compress", encode), ("decompress", decode)):
            for _ in range(3):
                operation()
            samples = []
            for _ in range(repeat):
                start = time.perf_counter_ns()
                operation()
                samples.append((time.perf_counter_ns() - start) / 1e6)
            ms = statistics.median(samples)
            timings[name] = {"median_ms": ms, "gb_per_second": len(data)/(ms*1e6),
                             "samples_ms": samples}
        return {"algorithm": algorithm, "level": level, "compressed_bytes": size,
                "ratio": len(data)/size, "remaining_fraction": size/len(data),
                "exact_round_trip": True, **timings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    codecs = Codecs()
    result = {"scope": "CPU whole-buffer lossless codecs; GPU/Tensor decode not measured here",
              "q_limbs": manifest["q_limbs"], "batch": manifest["batch"],
              "seed": manifest["seed"], "degree": manifest["degree"],
              "scale_log2": manifest["scale_log2"], "versions": codecs.versions,
              "cpu": platform.processor(), "repeat": args.repeat, "rows": []}
    for case in manifest["cases"]:
        for file in case["files"]:
            data = Path(file["file"]).read_bytes()
            if len(data) != file["bytes"]:
                raise ValueError("corpus size differs from manifest")
            row = {"pattern": case["pattern"], "representation": file["representation"],
                   "dtype": file["dtype"], "raw_bytes": len(data),
                   "sha256": hashlib.sha256(data).hexdigest(), "codecs": []}
            for algorithm, level in (("zstd", 1), ("zstd", 3), ("zstd", 9), ("lz4", None)):
                row["codecs"].append(codecs.benchmark(data, algorithm, level, args.repeat))
            result["rows"].append(row)
        print(case["pattern"], "Zstd/LZ4 round trips exact", flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
