"""Replay the exact audited Agent answer under final code; no new generation."""
import sys,json,shlex,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import VENV,enter_nix,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=600)
 from component_backend import qualify
 from benchmark_graph import digest
 from semantic_benchmark_execution import runtime_sources
 old=WORK/"results/agent-deepseek-page6_bf"
 candidate=json.loads((old/"attempt-00/response.txt").read_text())
 assert candidate["hecate_source"]==(old/"attempt-00/candidate.py").read_text()
 q=json.loads((old/"request.json").read_text());sources=runtime_sources()
 result=qualify(q,candidate)
 value=dict(source_hashes=sources,original_live_evidence=str(old),result=result,
  original_response_sha256=hashlib.sha256((old/"attempt-00/response.txt").read_bytes()).hexdigest(),
  new_agent_generation=False,paid_calls=0,same_model_not_additional_benchmark_case=True)
 value["binding"]=digest(value)
 path=WORK/"results/current-agent-witness-r198.json"
 with path.open("x") as f:json.dump(value,f,indent=2)
 print(json.dumps(result))
 return int(result["status"]!="passed")
if __name__=="__main__":raise SystemExit(main())
