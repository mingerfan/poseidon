#!/usr/bin/env python3
"""Semantic benchmark entry; defaults never contact a provider."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"baseline"))
if len(sys.argv)>1 and sys.argv[1] in ("baseline","free","directed","summary"):
    from semantic_benchmark_execution import main
elif len(sys.argv)>1 and sys.argv[1]=="helper-directed":
    sys.path.insert(0,str(Path(__file__).resolve().parent/'baseline/benchmarks/tools'))
    from poseidon_benchmark_entries import helper_main as main
elif len(sys.argv)>1 and sys.argv[1]=="compiler-evidence":
    sys.path.insert(0,str(Path(__file__).resolve().parent/'baseline/benchmarks/tools'))
    from poseidon_benchmark_entries import compiler_main as main
elif len(sys.argv)>1 and sys.argv[1]=="rejections":
    sys.path.insert(0,str(Path(__file__).resolve().parent/'baseline/benchmarks/tools'))
    from poseidon_rejection_entries import main
else:
    from benchmark_runner import main
if __name__=="__main__":
    raise SystemExit(main())
