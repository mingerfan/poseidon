#!/usr/bin/env python3
"""Validate ResNet18 slot equations against dense FP64 and native logits."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from ResNet18 import (
    BOUNDARY,
    IMAGE_VALUES,
    SLOTS,
    dense,
    load_image,
    load_label,
    load_model,
    make_relu,
    network,
    pack_input,
)


def decode_out_pack(packs, shape):
    from imagenet_layout import unpack
    return unpack(packs, shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--relu-coefficients", type=Path, required=True)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--native-logits", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runtime-fixture", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("use a new report path")
    if args.runtime_fixture and args.runtime_fixture.exists():
        parser.error("use a new runtime fixture path")

    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    model = load_model(args.weights)
    image = load_image(args.values)
    label = load_label(args.labels)
    relu = make_relu(args.relu_coefficients, float)

    errors = {}
    with torch.no_grad():
        reference, expected = dense(model, image, relu)

        def observe(name, values, shape):
            actual = decode_out_pack(values, shape)
            expected_value = expected[name][0] if expected[name].ndim == 4 else expected[name].reshape(-1, 1, 1)
            actual = actual[:, :expected_value.shape[1], :expected_value.shape[2]]
            if not torch.isfinite(actual).all():
                raise AssertionError(f'non-finite stage {name}')
            error = float((actual - expected_value).abs().max())
            errors[name] = error
            print(name, 'max_error', error, flush=True)

        packed_input = pack_input(image, model)
        result_full = network(model, packed_input, relu, observe, clear=True)[0][:1000]
        result = result_full
        if not torch.isfinite(result_full).all():
            raise AssertionError('non-finite logits')

    native = json.loads(args.native_logits.read_text())["logits"]
    native_error = float((result_full - torch.tensor(native)).abs().max())
    dense_error = float((result_full - reference).abs().max())
    if max(errors.values()) > 1e-8:
        raise AssertionError(f"stage mismatch {errors}")
    if dense_error > 1e-8:
        raise AssertionError(f"dense mismatch {dense_error}")
    if native_error > 0.02:
        raise AssertionError(f"native reference mismatch {native_error}")

    source_sha256 = hashlib.sha256(Path(__file__).with_name("ResNet18.py").read_bytes()).hexdigest()
    layout_sha256 = hashlib.sha256(Path(__file__).with_name("imagenet_layout.py").read_bytes()).hexdigest()
    model_sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.weights.glob("*.txt"))
    }
    model_sources["d13"] = hashlib.sha256(args.relu_coefficients.read_bytes()).hexdigest()
    model_sources["frontend"] = source_sha256
    model_sources["layout"] = layout_sha256
    model_sha256 = hashlib.sha256(
        json.dumps(model_sources, sort_keys=True).encode()
    ).hexdigest()
    report = dict(
        status="PASS",
        scope="clear FP64 slot/dense/original native logits; not CKKS execution",
        frontend_sha256=source_sha256,
        layout_sha256=layout_sha256,
        model_sha256=model_sha256,
        weights=102,
        image=0,
        label=label,
        prediction=int(result.argmax()),
        native_prediction=max(range(len(native)), key=native.__getitem__),
        stage_max_errors=errors,
        dense_max_error=dense_error,
        native_max_error=native_error,
        logits=result_full.tolist(),
        slots=SLOTS,
        input_packs=len(packed_input),
        keys_generated=False,
        gpu_executed=False,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.runtime_fixture:
        fixture = dict(
            format_version=1,
            model="resnet18",
            seed=0,
            model_sha256=model_sha256,
            input=image.reshape(-1).tolist(),
            packed_inputs=[packed.tolist() for packed in packed_input],
            input_encoding="frontend-packed-slots",
            python_output=result_full.tolist(),
            native_output=native,
            label=label,
            output_encoding="actual-logits-first-1000-slots",
            bootstrap_requirement="native-no-hoist",
            semantic_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        )
        args.runtime_fixture.write_text(json.dumps(fixture, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
