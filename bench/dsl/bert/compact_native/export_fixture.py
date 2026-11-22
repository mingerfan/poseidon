#!/usr/bin/env python3
"""Export ORIGINAL float32 arrays for the standalone C++ probe, without encoding.

Writes raw little-endian float32 files. These contain every matrix entry, no
sparsification, repetition, diagonal packing, or other weight substitution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="original .npz fixture")
    parser.add_argument("--matrix", choices=("q", "k", "v", "o", "ffn_up", "ffn_down"), required=True)
    parser.add_argument("--input", type=Path, help="optional true token-row .npy input")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.weights, allow_pickle=False) as archive:
        weight = np.asarray(archive[f"{args.matrix}_weight"])
        bias = np.asarray(archive[f"{args.matrix}_bias"])
    if weight.dtype != np.float32 or bias.dtype != np.float32:
        raise ValueError("fixture must already contain float32 weights/bias")
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("fixture shape mismatch")
    weight_path, bias_path = args.out / "weights.f32", args.out / "bias.f32"
    np.ascontiguousarray(weight, dtype="<f4").tofile(weight_path)
    np.ascontiguousarray(bias, dtype="<f4").tofile(bias_path)
    report = {
        "matrix": args.matrix,
        "shape_out_in": list(weight.shape),
        "source_weights": str(args.weights),
        "source_weights_sha256": sha(args.weights),
        "raw_weight_bytes": weight_path.stat().st_size,
        "raw_weight_sha256": sha(weight_path),
        "raw_bias_sha256": sha(bias_path),
        "all_weights_retained": True,
    }
    if args.input:
        hidden = np.load(args.input, allow_pickle=False)
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden[0]
        if hidden.ndim != 2 or hidden.shape[1] != weight.shape[1] or hidden.dtype not in (np.float32, np.float64):
            raise ValueError("input must be float32/float64 [tokens,in] (or [1,tokens,in])")
        np.ascontiguousarray(hidden, dtype="<f4").tofile(args.out / "input.f32")
        report["source_input_sha256"] = sha(args.input)
        report["input_shape"] = list(hidden.shape)
        report["source_input_dtype"] = str(hidden.dtype)
        report["probe_input_dtype"] = "float32"
    (args.out / "fixture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
