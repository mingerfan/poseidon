#!/usr/bin/env python3
"""Semantic benchmark entry; defaults never contact a provider."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"baseline"))
if len(sys.argv)>1 and sys.argv[1] in ("baseline","free","directed","summary"):
    from semantic_benchmark_execution import main
elif len(sys.argv)>1 and sys.argv[1]=="helper-directed":
    from upstream_helper_benchmark import main
elif len(sys.argv)>1 and sys.argv[1]=="compiler-evidence":
    from compiler_evidence_cli import main
elif len(sys.argv)>1 and sys.argv[1]=="rejections":
    from rejection_benchmark_cli import main
else:
    from benchmark_runner import main
if __name__=="__main__":
    raise SystemExit(main())
