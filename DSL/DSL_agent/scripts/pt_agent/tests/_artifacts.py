"""Skip tests whose inputs exist only on a machine that ran the earlier rounds.

Pure logic tests run everywhere. Tests that read historical run directories, the CIFAR records, trained
weights or the compiler build are skipped (not failed) when those artifacts are absent, with the missing
paths in the skip reason.
"""

from pathlib import Path
import unittest

from pt_agent.paths import HECATE_BUILD, RESULTS_DIR, WORK_RESULTS

DATA = (WORK_RESULTS / "agent-cifar-offline-records-r1804/cifar10-records",
        WORK_RESULTS / "agent-resnet20-original-accuracy-r1805/frozen-selections.json")
WEIGHTS = (WORK_RESULTS / "agent-cifar-offline-records-r1804/public/resnet20-12fca82f.th",)
COMPILER = (HECATE_BUILD / "bin/hecate-opt",)
# Ordinary CPU CKKS execution still goes through the legacy SDK selector and a baseline module that is not
# part of this repository's history.
LEGACY_CPU = (WORK_RESULTS / "agent-normal-cpu-approved-overlay-staging-r2680/sdk/ordinary-cpu-selector.json",
              Path(__file__).resolve().parents[2] / "baseline/normal_cpu_overlay_sandbox.py")


def require(*groups):
    """Decorator for a TestCase class: skip it unless every listed path exists."""
    paths = [Path(p) for group in groups for p in (group if isinstance(group, (tuple, list)) else (group,))]
    missing = [str(p) for p in paths if not p.exists()]
    return unittest.skipIf(bool(missing), "missing local artifacts: " + ", ".join(missing[:3]))


__all__ = ["COMPILER", "DATA", "LEGACY_CPU", "RESULTS_DIR", "WEIGHTS", "WORK_RESULTS", "require"]
