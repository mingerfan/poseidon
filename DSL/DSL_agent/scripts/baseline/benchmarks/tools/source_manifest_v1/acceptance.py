"""Cross-check generated manifests using the existing pinned PyTorch intake."""
import hashlib,importlib.util,io,json,os,pathlib,shlex,shutil,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(HERE.parent/"einsum_live_v1"));import runner_scope as guard
from run import OwnedTree
sys.path.insert(0,str(BASE))
os.environ["POSEIDON_PLATFORM"]="aarch64-linux"
os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
spec=importlib.util.spec_from_file_location("source_manifest_preparer",ROOT/"scripts/prepare_pytorch_source.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
OUT=guard.RESULTS/"agent-source-manifest-acceptance-v1"
IDS=("einsum_matrix","einsum_public_weight","einsum_partition_composition")
def write(path,value):
 with pathlib.Path(path).open("x")as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
def measure(path):return int(subprocess.check_output(["du","-sb",str(path)],text=True,timeout=20).split()[0])
def tools():return {str(p.relative_to(ROOT)):guard.sha(p)for p in[ROOT/"scripts/prepare_pytorch_source.py",HERE/"acceptance.py",HERE/"test_manifest.py"]}
def boundary(plan):
 guard.require(time.time()<plan["deadline_wall"],"source_manifest_time_boundary")
 guard.require(tools()==plan["tools"],"frozen_manifest_tools")
 guard.require(measure(OUT)<=64*1024**2 and measure(guard.RESULTS)<=32*1024**3
  and shutil.disk_usage(guard.WORK).free>=4*1024**3,"source_manifest_space_boundary")
 guard.verify()
def outer():
 proposal=guard.verify();guard.require(not OUT.exists(),"fresh_source_manifest_acceptance")
 controls=guard.read(HERE/"controls-v2.json")
 guard.require(controls["status"]=="passed"and controls["source_sha256"]==guard.sha(ROOT/"scripts/prepare_pytorch_source.py")
  and controls["test_sha256"]==guard.sha(HERE/"test_manifest.py"),"current_metadata_controls")
 rows=[]
 for case in proposal["cases"]:
  if case["id"]not in IDS:continue
  path=pathlib.Path(case["source_manifest"]);original=guard.read(path);module=original["entry"].split(".")[0]+".py"
  helper=[path.parent/name for name in original["files"]if name!=module]
  result,root=cli.build(path.parent/module,entry=original["entry"],
   inputs=[v["name"]+":"+",".join(str(n)for n in v["shape"])for v in original["inputs"]],outputs=original["outputs"],
   helpers=helper,weights=[name+"="+v["file"]for name,v in original["weights"].items()],
   factory_args=original["factory"]["args"],factory_kwargs=original["factory"]["kwargs"],
   dtype_policy=original["dtype_policy"],approximations=original["approximations"],
   source_format=original["format"],resource_profile=case["resource_profile"])
  guard.require(result==original,"equivalent_existing_manifest")
  fresh=cli.save(result,path.parent/"source-cli-r1344.json",root)
  rows.append(dict(case=case["id"],manifest=str(fresh),original_manifest=str(path),
   original_manifest_sha256=guard.sha(path),manifest_sha256=guard.sha(fresh),logical_manifest_sha256=guard.digest(result),
   mathematical_manifest_unchanged=True))
 created=time.time();plan=dict(format="poseidon-source-manifest-acceptance-plan-v1",
  created_wall=created,deadline_wall=created+300,tools=tools(),cases=rows,max_new_bytes=64*1024**2,
  expected_identity_sha256=proposal["identity_sha256"],compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0)
 plan["binding"]=guard.digest(plan);OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 tree=OwnedTree().start()
 try:code=enter_nix(command,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
 finally:background=tree.finish()
 if code or background:write(OUT/"termination.json",dict(exit_code=code,background_processes=background,automatic_restart=False,
  provider_calls=0,compiler_calls=0,encrypted_calls=0))
 return code if code else int(background)
def inside():
 guard.require(os.environ.get("IN_NIX_SHELL")=="pure"and pathlib.Path(sys.prefix)==VENV,"pinned_metadata_acceptance")
 import numpy as np,torch
 from pytorch_weight_package import decode
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_backend import PoseidonBackend
 from application_model_source import FORMAT
 from unified_capacity_constants import request_version
 plan=guard.read(OUT/"plan.json");proposal=guard.verify();boundary(plan)
 actual=PoseidonBackend().identity_manifest()
 guard.require(actual==proposal["application_stage_identity"],"actual_frozen_stage_identity")
 comparisons=[]
 data_dir=OUT/"npy";data_dir.mkdir()
 for version in[(1,0),(2,0)]:
  for dtype,values in[("<f4",[.1,-.2,.3,.4]),("<f8",[.1,-.2,.3,.4]),
                     ("<i4",[1,-2,3,4]),("<i8",[2**53,-2**53,0,1]),("|b1",[True,False,True,False])]:
   for dims in[(),(2,2)]:
    boundary(plan);value=np.array(values[:1]if not dims else values,dtype=dtype).reshape(dims)
    stream=io.BytesIO();np.lib.format.write_array(stream,value,version=version,allow_pickle=False)
    raw=stream.getvalue();filename="weight_"+str(len(comparisons))+".npy";path=data_dir/filename;path.write_bytes(raw)
    item=cli.npy_record(path,"general-tensors-v1")
    expected=dict(file=filename,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),shape=list(value.shape),dtype=value.dtype.str)
    guard.require(item==expected and np.array_equal(decode(raw,item),value),"numpy_weight_parser_match")
    comparisons.append(dict(dtype=dtype,version=list(version),shape=list(dims),status="passed",sha256=item["sha256"]))
 write(OUT/"numpy-comparisons.json",dict(status="passed",tests=len(comparisons),rows=comparisons,numpy_version=np.__version__))
 app=application(WorkerContext.capture(OUT/"tasks"));rows=[]
 cases={case["id"]:case for case in proposal["cases"]}
 for source in plan["cases"]:
  boundary(plan);case=cases[source["case"]]
  guard.require(guard.sha(pathlib.Path(source["original_manifest"]))==source["original_manifest_sha256"]
   and guard.sha(pathlib.Path(source["manifest"]))==source["manifest_sha256"],"source_manifest_bytes")
  options=dict(case["options"]);options["capacity_packing_version"]=request_version(guard.request_read(case["full_request_file"]))
  task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=source["manifest"],trust_local_code=True),
   resource_profile=case["resource_profile"],options=options,
   limits=dict(generations=4,http_attempts=16,wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000))
  state=app.prepare_source(task);guard.require(state["status"]==state["phase"]=="queued"
    and state.get("intake",{}).get("reference_verified")is True,"independent_reference_verified")
  meta=app._checked_task_metadata(task)
  guard.require(meta["request"]==guard.request_read(case["full_request_file"]),"unchanged_prepared_mathematical_request")
  guard.require(meta["identity"]==actual and state["generations"]==state["http_attempts"]==0,"source_only_same_identity")
  row=dict(case=case["id"],task_id=task,status="prepared",request_id=meta["request"]["request_id"],
   logical_manifest_equal=True,mathematical_request_equal=True,source_captured=True,independent_reference_verified=True,
   provider_calls=0,compiler_calls=0,encrypted_calls=0)
  write(OUT/(case["id"]+".receipt.json"),row);rows.append(row);boundary(plan)
  print(json.dumps(dict(case=case["id"],status="source_prepared",provider_calls=0)),flush=True)
 report=dict(format="poseidon-source-manifest-acceptance-v1",status="passed",source_cases=len(rows),rows=rows,
  numpy_cross_checks=len(comparisons),seconds=time.time()-plan["created_wall"],plan_binding=plan["binding"],
  identity_manifest_sha256=actual["manifest_sha256"],
  versions=dict(python=sys.version.split()[0],numpy=np.__version__,torch=torch.__version__,torch_cuda=torch.version.cuda),
  provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,decryption_calls=0,
  original_manifest_and_request_unchanged=True,automatic_restart=False)
 report["binding"]=guard.digest(report);write(OUT/"report.json",report)
 print(json.dumps(dict(status="passed",source_cases=len(rows),numpy_cross_checks=len(comparisons),
  seconds=report["seconds"],identity_manifest_sha256=actual["manifest_sha256"],provider_calls=0)),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
