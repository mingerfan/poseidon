"""Append-only JSON-lines run registry under PT_RESULTS."""

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
import secrets
import subprocess

from .config import load as load_config


REQUIRED_FIELDS = frozenset({
    "run_id", "round", "model", "mode", "git_rev", "metrics", "artifacts_dir", "ts",
})


def append(row: dict) -> None:
    """Append one complete run record; keep all earlier records."""
    if not isinstance(row, dict) or not REQUIRED_FIELDS.issubset(row):
        raise ValueError("Registry row requires: " + ", ".join(sorted(REQUIRED_FIELDS)))
    if not isinstance(row["metrics"], dict):
        raise ValueError("Registry metrics must be an object")
    line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
    path = load_config().results / "registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(line)
        stream.flush()


def load() -> list[dict]:
    """Read records in append order; report malformed records with line numbers."""
    path = load_config().results / "registry.jsonl"
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows = []
    with stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        for lineno, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not REQUIRED_FIELDS.issubset(row):
                    raise ValueError("incomplete run record")
                if not isinstance(row["metrics"], dict):
                    raise ValueError("metrics must be an object")
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            rows.append(row)
    return rows


def git_rev() -> str:
    """Return the current short revision and a dirty marker, or unknown."""
    try:
        directory = Path(__file__).resolve().parent
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=directory,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=directory,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return rev + ("-dirty" if dirty else "")


def new_run_id(round_name: str, model: str, mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{round_name}-{model}-{mode}-{stamp}-{secrets.token_hex(2)}"
