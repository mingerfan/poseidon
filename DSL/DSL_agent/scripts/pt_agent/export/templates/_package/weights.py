"""Load only numeric NPZ tensors and enforce the public weight schema."""

import json
from pathlib import Path

import numpy as np


def load(path):
    schema = json.loads((Path(__file__).resolve().parent.parent/"config/weight_schema.json").read_text())
    weights = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(schema):
            raise ValueError("NPZ keys differ from weight_schema.json")
        for name, shape in schema.items():
            value = np.asarray(archive[name], dtype=np.float64)
            if list(value.shape) != shape or not np.isfinite(value).all():
                raise ValueError(f"weight {name} has an invalid shape or nonfinite values")
            weights[name] = value
    return weights
