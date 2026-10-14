"""Compare prior 11 guidance contracts against saved implementation bytes."""
import copy,hashlib,importlib.util,json,os,shlex,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def main():
 if sys.argv[1:]!=["--inside"]:
  from hecate_python_env import VENV,enter_nix
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
   [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=60,keep_env=(),environment="poseidon-cpu")
 from benchmark_graph import CAPACITY_FORMAT,canonical
 from benchmark_suite import Builder
 from compiler_configuration import configuration
 from unified_graph_contract import prepare
 from component_contract import reconstruct_request
 b=Builder([(2,2)]);m=b.node("mean",["input0"],axes=[-1],keepdims=True)
 g=b.finish(b.node("subtract",["input0",m]));g.update(format=CAPACITY_FORMAT,resource_profile="general-tensors-v1")
 file=HERE/"source-before/unified_graph_contract.py"
 source_sha=hashlib.sha256(file.read_bytes()).hexdigest()
 assert source_sha==json.loads((HERE/"plan.json").read_text())["source_before"][file.name]
 spec=importlib.util.spec_from_file_location("_trusted_preserved_v11_contract",file)
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 config=configuration("poseidon-cpu-eva-w45-v1");rows=[]
 for version in (None,*(f"explicit-v{i}"for i in range(1,12))):
  args=dict(chunk_period=4,generation_guidance=version)
  before=old.prepare(copy.deepcopy(g),config["ckks_config_sha256"],config,**args)
  after=prepare(copy.deepcopy(g),config["ckks_config_sha256"],config,**args)
  assert canonical(before)==canonical(after),"old_request_bytes_changed"
  assert reconstruct_request(after)==after,"old_request_reconstruction_changed"
  rows.append(dict(version=version,request_id=after["request_id"],canonical_sha256=hashlib.sha256(canonical(after)).hexdigest(),unchanged=True))
 current=prepare(copy.deepcopy(g),config["ckks_config_sha256"],config,chunk_period=4,generation_guidance="explicit-v12")
 assert current["request_id"]not in [r["request_id"]for r in rows]
 report=dict(format="poseidon-tensor-axis-historical-contract-compatibility-v1",status="passed",
  cases=len(rows),failed=0,skipped=0,prior_implementation_sha256=source_sha,
  requests=rows,current_version="explicit-v12",current_request_id=current["request_id"],
  provider_calls=0,compilation=False,encrypted_execution=False)
 report["binding"]=hashlib.sha256(canonical(report)).hexdigest()
 (HERE/"compatibility.final.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps({k:report[k]for k in("status","cases","failed","skipped","current_version","provider_calls")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
