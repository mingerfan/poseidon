#!/usr/bin/env python3
"""Prepare the locked Agent CPU backend; --plan is the default."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))

if __name__ == "__main__":
    from agent_setup import main
    raise SystemExit(main())
