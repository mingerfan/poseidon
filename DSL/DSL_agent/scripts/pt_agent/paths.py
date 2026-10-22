"""Filesystem locations derived from this checkout; importing this module has no side effects.

Layout convention (all relative to the workspace that contains the repository):

    <workspace>/Code/Poseidon        the repository (REPO)
    <workspace>/pt_agent_results     PT-Agent run directories (RESULTS_DIR)
    <workspace>/poseidon-work        toolchain builds, SDK and dataset records (WORK_ROOT)

POSEIDON_WORK_ROOT and POSEIDON_PLATFORM override the toolchain location, as before.
PT_RESULTS is deliberately not read here: inner (Nix --pure) code must not depend on it.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
WORKSPACE = REPO.parents[1]
RESULTS_DIR = WORKSPACE / "pt_agent_results"
WORK_ROOT = Path(os.environ.get("POSEIDON_WORK_ROOT") or WORKSPACE / "poseidon-work").expanduser()
PLATFORM = os.environ.get("POSEIDON_PLATFORM") or "aarch64-linux"
PLATFORM_DIR = WORK_ROOT / "platforms" / PLATFORM
WORK_RESULTS = PLATFORM_DIR / "results"


def pt_result(*parts: str) -> Path:
    """A path inside a PT-Agent results directory."""
    return RESULTS_DIR.joinpath(*parts)


def work_result(*parts: str) -> Path:
    """A path inside the platform results directory of poseidon-work."""
    return WORK_RESULTS.joinpath(*parts)
