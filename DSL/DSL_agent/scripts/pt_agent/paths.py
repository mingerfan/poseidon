"""Filesystem locations derived from this checkout; importing this module has no side effects.

Layout convention (all relative to the workspace that contains the repository):

    <workspace>/Code/Poseidon        the repository (REPO)
    <workspace>/pt_agent_results     PT-Agent run directories (RESULTS_DIR)
    <workspace>/poseidon-work        toolchain builds, SDK and dataset records (WORK_ROOT)

Every location can be overridden by an environment variable; tools/nixrun.py forwards these variables into
the pure Nix shell (see ENV_VARS):

    POSEIDON_WORK_ROOT     poseidon-work root                     (default <workspace>/poseidon-work)
    POSEIDON_PLATFORM      platform directory name                (default aarch64-linux)
    PT_AGENT_RESULTS_ROOT  PT-Agent results root                  (default <workspace>/pt_agent_results)
    PT_HECATE_BUILD        Hecate/DaCapo build directory with bin/hecate-opt
    PT_DACAPO_ROOT         DaCapo source tree: Python Hecate frontend, minimax coefficients, ResNet20-SiLU weights
    PT_DACAPO_UPSTREAM     unmodified upstream DaCapo checkout (expert DSL examples, R9a motivation only)

PT_RESULTS is deliberately not read here: it names the output directory of one outer run, and inner
(Nix --pure) code must not depend on it.
"""

import os
from pathlib import Path

ENV_VARS = ("POSEIDON_WORK_ROOT", "POSEIDON_PLATFORM", "PT_AGENT_RESULTS_ROOT", "PT_HECATE_BUILD", "PT_DACAPO_ROOT",
            "PT_DACAPO_UPSTREAM")


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value and value.strip() else default


REPO = Path(__file__).resolve().parents[4]
WORKSPACE = REPO.parents[1]
RESULTS_DIR = _env_path("PT_AGENT_RESULTS_ROOT", WORKSPACE / "pt_agent_results")
WORK_ROOT = _env_path("POSEIDON_WORK_ROOT", WORKSPACE / "poseidon-work")
PLATFORM = os.environ.get("POSEIDON_PLATFORM") or "aarch64-linux"
PLATFORM_DIR = WORK_ROOT / "platforms" / PLATFORM
WORK_RESULTS = PLATFORM_DIR / "results"

# Compiler toolchain (see DSL/DSL_agent/scripts/pt_agent/README.md for how to build it).
# Default since 2026-10-30: the same source with the Qwen25 R121 bootstrap-window change
# (STATUS "R121 自举段预算修正对齐"). Set PT_HECATE_BUILD to
# WORK_RESULTS/agent-current-dacapo-build-r1712/build to reproduce earlier results.
HECATE_BUILD = _env_path("PT_HECATE_BUILD", WORK_RESULTS / "agent-dacapo-bootwindow-20261030/build")
DACAPO_ROOT = _env_path("PT_DACAPO_ROOT", REPO / "third_party/ckks-runtime/third_party/dacapo")
DACAPO_UPSTREAM = _env_path("PT_DACAPO_UPSTREAM", REPO / "third_party/dacapo")
PROFILES_DIR = Path(__file__).resolve().parent / "compiler_profiles"


def pt_result(*parts: str) -> Path:
    """A path inside a PT-Agent results directory."""
    return RESULTS_DIR.joinpath(*parts)


def work_result(*parts: str) -> Path:
    """A path inside the platform results directory of poseidon-work."""
    return WORK_RESULTS.joinpath(*parts)
