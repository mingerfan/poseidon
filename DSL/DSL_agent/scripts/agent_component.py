#!/usr/bin/env python3
"""Framework-facing single-job worker; legacy scripts/agent.py is unchanged."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/"baseline"))
from component_worker import main
if __name__ == "__main__":
    raise SystemExit(main())
