"""Profile immutable provenance validation only; no source loading or backend execution."""
import argparse,sys,shlex,json,time,cProfile,pstats,io
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from hecate_python_env import WORK,VENV,enter_nix
def inside():
 from candidate_bundle import check_provenance
 package=WORK/"results/agent-public-tensor-v22-cpu-v2/tasks/fb485ba6c60e469c97541f097929d8a2/program"
 model=json.loads((package/"model.json").read_text());value=json.loads((package/"model-provenance.json").read_text())
 profiler=cProfile.Profile();now=time.monotonic();profiler.runcall(check_provenance,value,model)
 text=io.StringIO();pstats.Stats(profiler,stream=text).strip_dirs().sort_stats("cumulative").print_stats(20)
 (HERE/"provenance-profile-v1.log").write_text(text.getvalue())
 (HERE/"provenance-profile-v1.json").write_text(json.dumps({"seconds":time.monotonic()-now,"api":0,"compile":0,"fhe":0}))
 print(text.getvalue())
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");a=p.parse_args()
 if a.inside:inside()
 else:
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  raise SystemExit(enter_nix(command,seconds=120,keep_env=(),environment="poseidon-cpu"))
