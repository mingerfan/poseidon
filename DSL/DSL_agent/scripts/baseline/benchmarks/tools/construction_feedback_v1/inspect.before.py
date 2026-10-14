"""Read-only cost facts for two preserved, rejected historical candidates."""
import copy,json,os,shlex,sys,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def main():
 if not sys.argv[1:]:
  from hecate_python_env import VENV,enter_nix
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(command,seconds=60,keep_env=(),environment="poseidon-cpu")
 assert sys.argv[1:]==["--inside"]
 from unified_graph_contract import validate_request,validate_candidate
 from decorated_functions import validate
 from unified_capacity_layout import EXTENDED_ABIS
 from candidate_check_diagnostic import CandidateCheckError
 proposal=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json").read_text())
 c=proposal["cases"][0]
 request=json.loads(Path(c["request_file"]).read_text());p=validate_request(request)
 taskdir=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-tensor-axis-live-v1")/c["id"]/"tasks"/"a2f5ba0de3954fd3b3e8106aae6d6a07"
 rows=[]
 for index in (0,2):
  raw=json.loads((taskdir/f"response-{index}.json").read_text());candidate=json.loads(raw["raw"])
  try:validate_candidate(candidate,request)
  except CandidateCheckError as e:
   assert e.reason=="expanded_work_budget"
   observed=e.context["observed"]
  else:raise AssertionError("historical_invalid_candidate_accepted")
  plan=validate(candidate["hecate_source"],request["public_constants"],request["layout"]["output_ciphertexts"],
   input_names=tuple(s["dsl_name"] for s in request["layout"]["inputs"])+("zero_ct",),
   array_mutation=True,slot_period=p,
   capacity_layout=request["layout"] if request["layout"]["execution_abi"] in EXTENDED_ABIS else None,
   upstream_helpers="upstream_helpers" in request,upstream_request=request if "upstream_helpers" in request else None)
  functions=plan["functions"];counts=collections.Counter(functions["golden"]["calls"])
  costs=[dict(helper=n,calls=k,per_call_work=functions[n]["expanded_cost"]+1,
    total_work=k*(functions[n]["expanded_cost"]+1)) for n,k in counts.items()]
  assert functions["golden"]["expanded_cost"]==observed
  constant_count=len(request["public_constants"])
  rows.append(dict(round=index,observed=observed,limit=1024,public_binding_count=constant_count,
   native_helper_invocations=sum(counts.values()),native_call_work=sum(v["total_work"] for v in costs),
   repeated_registry_contribution=constant_count*sum(counts.values()),top_costs=sorted(costs,key=lambda v:-v["total_work"])[:3],
   candidate_still_rejected=True))
 result=dict(format="poseidon-construction-cost-readonly-v1",rows=rows,provider_calls=0,compiler_calls=0,encrypted_calls=0)
 with (HERE/"inspection.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2)
 print(json.dumps(result))
 return 0
if __name__=="__main__":raise SystemExit(main())
