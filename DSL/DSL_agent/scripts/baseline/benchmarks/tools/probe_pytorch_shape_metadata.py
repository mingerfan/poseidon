"""Isolated development-only metadata patch probe; never an application success."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmarks.tools.audit_pytorch_shape_composition import sha,checked,preservation

WORKER='''import importlib,json,hashlib,random,sys
from pathlib import Path
import numpy as np,torch
from benchmark_graph import require,digest,samples
from pytorch_model_intake import validate_package,source_identity
from pytorch_weight_package import read,load_state
from staged_static_trace import trace as staged_trace,ARITHMETIC_POLICY,INDEX_POLICY
from pytorch_static_trace import trace
from pytorch_arithmetic_normalization import normalize,rename_record,bind_record,check_record
from pytorch_model_capture import capture
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as independent
from operator_reference import evaluate as high
request=json.loads(Path("/request.json").read_text());spec=request["manifest"]
actual_runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,
                    cuda=torch.version.cuda,hip=getattr(torch.version,"hip",None))
require(source_identity()==request["capture_identity"] and actual_runtime==request["runtime"],"Frozen prototype dependencies")
require(hashlib.sha256(Path("/app/staged_static_trace.py").read_bytes()).hexdigest()==request["staged_source_sha256"],"Staged metadata source")
files={name:Path("/model",name).read_text() for name in spec["files"]}
validate_package(spec,files)
module_name,entry=spec["entry"].split(".")
module=getattr(importlib.import_module(module_name),entry)(*spec["factory"]["args"],**spec["factory"]["kwargs"])
arrays,_=read("/model",spec["weights"],resource_profile="general-tensors-v1")
load_state(module,arrays,spec["weights"]);module=module.double().eval();torch.set_num_threads(1)
before={k:v.detach().clone() for k,v in module.state_dict().items()}
rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
# Old opt-in modes retain rejection; only the new metadata policy opens this shape query.
try:staged_trace(module,spec["inputs"],resource_profile="general-tensors-v1",policy=INDEX_POLICY)
except ValueError as error:
    require(str(error)=="Static shape unavailable for this operation","Old metadata behavior")
else:raise ValueError("Old index metadata behavior changed")
raw=staged_trace(module,spec["inputs"],resource_profile="general-tensors-v1",policy=ARITHMETIC_POLICY)
normalized,record=normalize(raw,resource_profile="general-tensors-v1")
canonical=trace(normalized,spec["inputs"],resource_profile="general-tensors-v1",policy=INDEX_POLICY)
record=rename_record(record,normalized,canonical)
original,lowered,binding=capture(canonical,spec["inputs"],spec["outputs"],
    approximations=spec["approximations"],resource_profile="general-tensors-v1",
    dtype_policy="strict-float64-v1",preserve_output_structure=True,record_state_sources=True,
    capture_policy=INDEX_POLICY,approximation_policy="node-module-default-v1",
    matrix_policy="static-tensor-matmul-v1")
record=bind_record(record,original,captured_fx=canonical);check_record(record,original)
maximum=0.
with torch.no_grad():
    for inputs in samples(lowered,16):
        module_result=module(*(torch.from_numpy(inputs[s["name"]]) for s in spec["inputs"])).detach().numpy()
        normalized_result=canonical(*(torch.from_numpy(inputs[s["name"]]) for s in spec["inputs"])).detach().numpy()
        expected=high(original,inputs)["result"]
        for actual in (module_result,normalized_result,
                       mathematical(lowered,inputs,resource_profile="general-tensors-v1")["result"],
                       independent(lowered,inputs,resource_profile="general-tensors-v1")["result"]):
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
        require(torch.equal(torch.get_rng_state(),rng) and random.getstate()==py_rng and
                all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),np_rng)),"Pure prototype forward")
        require(all(torch.equal(v,module.state_dict()[k]) for k,v in before.items()),"Prototype state changed")
report=dict(format="poseidon-shape-metadata-prototype-reference-v1",probes=16,
    manifest_sha256=digest(spec),original_sha256=digest(original),model_sha256=digest(lowered),
    metadata_policy=ARITHMETIC_POLICY,old_index_policy_still_rejected=True,maximum_reference_error=maximum,
    original_module_checked=True,normalized_module_checked=True,independent_math_torch_checked=True,
    arithmetic_record=record,capture_identity=request["capture_identity"],runtime=request["runtime"],
    staged_source_sha256=request["staged_source_sha256"],isolated=True,
    production_intake_integrated=False,compiled=False,encrypted_execution=False,agent_generated=False)
report["binding"]=digest(report)
for name,value in (("report.json",report),("original.json",original),("model.json",lowered)):
    Path("/out",name).write_text(json.dumps(value,sort_keys=True))
Path("/out/fx-source.txt").write_text(str(raw.graph))
'''
def run(review_dir,fixtures,output,prototype,staged_source,proposal_path):
    from pytorch_model_intake import IMPLEMENTATION,source_identity,prepare
    from trusted_pytorch_import import runtime_identity
    from hecate_python_env import VENV,WORK
    from pytorch_weight_package import read
    from process_resource_limits import limits
    started=time.monotonic()
    output,fixtures=Path(output),Path(fixtures)
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh prototype directory")
    review=checked(json.loads((Path(review_dir)/"report.json").read_text()))
    proposal=checked(json.loads(Path(proposal_path).read_text()))
    prototype=checked(json.loads(Path(prototype).read_text()))
    require(preservation(proposal)==review["frozen_inventories"] and source_identity()==review["capture_identity"],
            "Frozen proposal/capture")
    require(sha(staged_source)==prototype["staged_source_sha256"] and
            sha(ROOT/"scripts/baseline/pytorch_static_trace.py")==prototype["original_sha256"],
            "Exact staged/active metadata implementation")
    require(prototype["production_intake_integrated"] is False,"Development-only patch")
    plans=[row for row in review["rows"] if row["kind"]=="positive" and row["capture_status"]=="rejected"]
    require(len(plans)==9 and all(row["failure"]["code"]=="TENSOR_ARITHMETIC_RESULT_METADATA_MISSING" for row in plans),
            "Retained nine original shape gaps")
    frozen=preservation(proposal);runtime=runtime_identity();capture_id=source_identity()
    own=sha(__file__);rows=[]
    output.mkdir(mode=0o700)
    worker=output/"probe-worker.py";worker.write_text(WORKER)
    catalog=json.loads((fixtures/"catalog.json").read_text())
    cases={row["id"]:row for row in catalog["cases"]}
    for row in plans:
        require(time.monotonic()-started<90,"Prototype 90-second bound")
        name=row["id"];folder=output/name;folder.mkdir()
        spec,paths=prepare(fixtures/cases[name]["manifest"])
        _,weights=read((fixtures/cases[name]["manifest"]).parent,spec["weights"],resource_profile="general-tensors-v1")
        request=dict(manifest=spec,capture_identity=capture_id,runtime=runtime,
                     staged_source_sha256=prototype["staged_source_sha256"])
        payload=folder/"request.json";payload.write_bytes(canonical(request))
        (folder/"weights").mkdir()
        for file,raw in weights.items():(folder/"weights"/file).write_bytes(raw)
        lib=os.environ["HECATE_PYTHON_LIBRARY_PATH"]
        require(all(Path(p).is_relative_to("/nix/store") for p in lib.split(":")),"Pinned libraries")
        command=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
            "--ro-bind","/nix/store","/nix/store","--ro-bind",str(VENV),str(VENV),
            "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
            "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app","--dir","/model",
            "--ro-bind",str(payload),"/request.json","--bind",str(folder),"/out"]
        for file in IMPLEMENTATION:command+=["--ro-bind",str(ROOT/"scripts/baseline"/file),"/app/"+file]
        command+=["--ro-bind",str(staged_source),"/app/staged_static_trace.py",
                  "--ro-bind",str(worker),"/app/probe-worker.py"]
        for file,path in paths.items():command+=["--ro-bind",str(path),"/model/"+file]
        for file in weights:command+=["--ro-bind",str(folder/"weights"/file),"/model/"+file]
        for key,value in dict(PYTHONPATH="/app:/model",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
            LD_LIBRARY_PATH=lib,HOME="/nonexistent",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1").items():
            command+=["--setenv",key,value]
        command+=["--chdir","/out","--",str(VENV/"bin/python"),"-B","/app/probe-worker.py"]
        with (folder/"capture.log").open("x") as log:
            process=subprocess.Popen(command,env={"PATH":"/usr/bin:/bin"},stdout=log,stderr=subprocess.STDOUT,
                                     preexec_fn=limits,start_new_session=True)
            try:process.wait(timeout=min(180,max(1,90-(time.monotonic()-started))))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise
        require(process.returncode==0,"Prototype isolated check failed; retain original capture.log")
        ref=checked(json.loads((folder/"report.json").read_text()))
        require(ref["probes"]==16 and ref["manifest_sha256"]==digest(spec) and
                ref["capture_identity"]==capture_id and ref["staged_source_sha256"]==prototype["staged_source_sha256"],
                "Prototype reference identity")
        rows.append(dict(id=name,development_probe_status="passed",original_public_capture_status="rejected",
            production_intake_integrated=False,probes=16,old_index_policy_still_rejected=True,
            max_reference_error=ref["maximum_reference_error"],report_sha256=sha(folder/"report.json")))
    require(preservation(proposal)==frozen and source_identity()==capture_id and runtime_identity()==runtime and
            sha(__file__)==own,"Frozen identities changed")
    report=dict(format="poseidon-agent-shape-patch-prototype-probe-v1",planned=9,passed=len(rows),
        failed=0,not_run=9-len(rows),reference_groups=sum(row["probes"] for row in rows),
        original_public_capture_failures_retained=9,production_intake_integrated=False,
        rows=rows,prototype_binding=prototype["binding"],source_review_binding=review["binding"],
        frozen_proposal_binding=proposal["binding"],frozen_inventories=frozen,
        capture_identity=capture_id,runtime=runtime,probe_tool_sha256=own,
        worker_sha256=sha(worker),seconds=time.monotonic()-started,
        compiled=0,encrypted_execution=0,decrypted=0,new_agent_generation=0,paid_calls=0,
        backend_changes=0,reference_changes=0,production_source_changes=0,
        scope="Manually prepared Agent-only metadata patch in isolated development probe; not public V8/FHE/Agent success")
    report["binding"]=digest(report);(output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","capture_identity")},ensure_ascii=False))
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("review-dir","fixtures","output","prototype","staged-source","frozen-proposal"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(run(args.review_dir,args.fixtures,args.output,args.prototype,args.staged_source,args.frozen_proposal))
