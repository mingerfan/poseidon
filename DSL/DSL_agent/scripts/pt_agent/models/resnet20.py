"""Reuse the existing B-owned model definition and approved public weights."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING
from ..paths import work_result

if TYPE_CHECKING:
    import torch
    from ..intake import Graph


WEIGHTS = work_result('agent-cifar-offline-records-r1804/public/resnet20-12fca82f.th')


def load_torch() -> torch.nn.Module:
    path = str(Path(__file__).resolve().parents[3] / "workloads/resnet20")
    if path not in sys.path:
        sys.path.insert(0, path)
    from model_reference import load_original_weights

    return load_original_weights(WEIGHTS)[0].float().eval()


def load_graph() -> Graph:
    import torch
    from ..intake import import_module

    return import_module(load_torch(), (torch.zeros(1, 3, 32, 32),), "resnet20")
