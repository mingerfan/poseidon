from pathlib import Path
import sys,shlex,unittest,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 import run as live;live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=180,environment="poseidon-cpu"))
import qualification_scope as q

import traceback
from unified_graph_lowering import candidate_source
for name in ("conv2_c1","bn1_c2","pool2_c2"):
 request=q.read(q.RESULTS/"agent-generation-origin-admission-r1677"/name/"request.json")
 try:
  source,record=candidate_source(request)
  print(name,"rule_source_ok",len(source))
 except Exception:
  print(name,"rule_source_error")
  traceback.print_exc(limit=12)
