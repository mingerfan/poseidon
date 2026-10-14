#!/usr/bin/env python3
"""Explicit original-source calibration and approximation preparation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"baseline"))
from source_preparation_entry import main
if __name__=="__main__":raise SystemExit(main())
