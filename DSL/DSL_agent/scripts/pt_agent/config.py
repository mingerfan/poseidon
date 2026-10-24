"""Paths used by PT-Agent; importing this module has no side effects."""

from dataclasses import dataclass
import os
from pathlib import Path

from .paths import HECATE_BUILD, PROFILES_DIR


DEFAULT_HECATE_OPT = HECATE_BUILD / "bin/hecate-opt"
DEFAULT_OPERATOR_SPEC = PROFILES_DIR / "poseidon-operator-spec-v1.json"

# U1 system behaviour, inferred from inputs rather than exposed as model flags.
SAMPLE_SPLIT_SEED = 20261024
MIN_INPUT_SAMPLES = 223
SEQUENCE_MULTIPLE = 8
SEQUENCE_MAX_BUCKETS = 3
SEQUENCE_BUCKET_PERCENTILES = (50, 90, 99)
SEQUENCE_TRUNCATION_TRIGGER = 0.05
SEQUENCE_RATIO_TRIGGER = 4


@dataclass(frozen=True)
class Config:
    results: Path
    hecate_opt: Path
    operator_spec: Path


def load() -> Config:
    """Read explicit results storage and optional tool path overrides."""
    results = os.environ.get("PT_RESULTS")
    if not results or not results.strip():
        raise ValueError("PT_RESULTS must name the PT-Agent results directory")
    return Config(
        results=Path(results).expanduser().resolve(),
        hecate_opt=Path(os.environ.get("PT_HECATE_OPT", DEFAULT_HECATE_OPT)).expanduser(),
        operator_spec=Path(os.environ.get("PT_OPERATOR_SPEC", DEFAULT_OPERATOR_SPEC)).expanduser(),
    )
