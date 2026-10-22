"""Read the existing CIFAR records and frozen index/label selections."""

import json
from pathlib import Path

import numpy as np
from ..paths import WORK_RESULTS


RESULTS = WORK_RESULTS
RECORDS = RESULTS / "agent-cifar-offline-records-r1804/cifar10-records"
SELECTIONS = RESULTS / "agent-resnet20-original-accuracy-r1805/frozen-selections.json"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
HELDOUT = range(9000, 10000)
CONFIRM = range(8000, 9000)
REPORT = range(0, 8000)


def read_rows(split: str, indices) -> tuple[np.ndarray, np.ndarray]:
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    indices = list(indices)
    maximum = 50000 if split == "train" else 10000
    if any(not isinstance(i, (int, np.integer)) or not 0 <= i < maximum for i in indices):
        raise ValueError("record index outside split")
    pixels = np.empty((len(indices), 3, 32, 32), dtype=np.uint8)
    labels = np.empty(len(indices), dtype=np.int64)
    groups = {}
    for output, index in enumerate(indices):
        groups.setdefault(index//1000*1000, []).append((output, index % 1000))
    for shard, rows in groups.items():
        records = np.memmap(RECORDS / f"{split}-{shard:05d}.records", mode="r", dtype=np.uint8, shape=(1000, 3073))
        output, row = zip(*rows)
        selected = records[list(row)]
        pixels[list(output)] = selected[:, 1:].reshape(-1, 3, 32, 32)
        labels[list(output)] = selected[:, 0]
        del records
    return pixels, labels


def normalize(pixels_uint8, normalization=(MEAN, STD)) -> np.ndarray:
    mean, std = normalization
    pixels = np.asarray(pixels_uint8, dtype=np.float64)/255
    return (pixels-np.asarray(mean).reshape(1, 3, 1, 1))/np.asarray(std).reshape(1, 3, 1, 1)


def _selection(key, split, normalization=(MEAN, STD)):
    entries = json.loads(SELECTIONS.read_text())[key]
    for entry in entries:
        index = entry["index"]
        if entry["shard"] != f"{split}-{index//1000*1000:05d}.records" or entry["row"] != index % 1000:
            raise ValueError("selection shard/row disagrees with index")
    pixels, labels = read_rows(split, [entry["index"] for entry in entries])
    if any(entry["label"] != int(label) for entry, label in zip(entries, labels)):
        raise ValueError("selection label disagrees with record label")
    return normalize(pixels, normalization), labels


def calibration_set(normalization=(MEAN, STD)) -> tuple[np.ndarray, np.ndarray]:
    return _selection("calibration", "train", normalization)


def validation_set(n: int = 2000, normalization=(MEAN, STD)) -> tuple[np.ndarray, np.ndarray]:
    """Read increasing training indices from 10000, excluding calibration."""
    calibration = {entry["index"] for entry in json.loads(SELECTIONS.read_text())["calibration"]}
    indices = [index for index in range(10000, 50000) if index not in calibration]
    if type(n) is not int or not 0 <= n <= len(indices):
        raise ValueError("validation size exceeds the available training records")
    pixels, labels = read_rows("train", indices[:n])
    return normalize(pixels, normalization), labels


def encrypted_test_set(normalization=(MEAN, STD)) -> tuple[np.ndarray, np.ndarray]:
    return _selection("encrypted_test", "test", normalization)


def test_set(limit: int | None = None, normalization=(MEAN, STD)) -> tuple[np.ndarray, np.ndarray]:
    count = 10000 if limit is None else limit
    if not isinstance(count, int) or not 0 <= count <= 10000:
        raise ValueError("test limit must be 0..10000")
    pixels, labels = read_rows("test", range(count))
    return normalize(pixels, normalization), labels


def heldout_set(n: int = 1000, normalization=(MEAN, STD)):
    """Read the first n held-out test records, starting at index 9000."""
    if type(n) is not int or not 0 <= n <= len(HELDOUT):
        raise ValueError("heldout size must be 0..1000")
    pixels, labels = read_rows("test", HELDOUT[:n])
    return normalize(pixels, normalization), labels


def confirm_set(normalization=(MEAN, STD)):
    """Read confirmation-only test records 8000 through 8999."""
    pixels, labels = read_rows("test", CONFIRM)
    return normalize(pixels, normalization), labels


def report_test_set(normalization=(MEAN, STD)):
    """Read report-only test records 0 through 7999."""
    pixels, labels = read_rows("test", REPORT)
    return normalize(pixels, normalization), labels
