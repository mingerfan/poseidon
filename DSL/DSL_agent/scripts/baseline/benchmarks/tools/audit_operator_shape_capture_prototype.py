"""Replay owned source failures with explicitly staged Agent files in an independent sandbox.

This development tool does not integrate staged code, rebind production evidence,
compile DSL, call a Provider, or import user model source in the host process.
"""
import argparse,copy,hashlib,json,math,os,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require,samples
PROFILE="general-tensors-v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked(value):
    require(type(value) is dict and value.get("binding")==digest({k:v for k,v in value.items() if k!="binding"}),"Bound development data")
    return value
def validate_prototype(directory):
    from pytorch_model_intake import source_identity
    value=checked(json.loads((directory/"prototype.json").read_text()))
    require(value["production_integrated"] is False and value["backend_changes"]==0 and value["reference_changes"]==0,"Staged-only prototype scope")
    current=source_identity()
    require(set(value["staged_hashes"])==set(value["original_hashes"])=={
        "pytorch_model_intake.py","pytorch_model_capture.py","pytorch_static_trace.py"},"Exact staged adapter files")
    require(all(current[name]==expected for name,expected in value["original_hashes"].items()),"Prototype source base changed")
    require(all(sha(directory/name)==expected for name,expected in value["staged_hashes"].items()),"Staged source changed")
    require(all(sha(directory/name)==expected for name,expected in value["patch_hashes"].items()),"Staged patch changed")
    expected=dict(current);expected.update(value["staged_hashes"])
    return value,current,expected
def sandbox(spec,paths,weight_files,output,source_hashes,prototype,seconds):
    from hecate_python_env import VENV
    from pytorch_model_intake import IMPLEMENTATION
    from process_resource_limits import limits
    from trusted_pytorch_import import runtime_identity
    require(not output.exists(),"Fresh isolated worker directory")
    output.mkdir(mode=0o700)
    payload=output/"capture-input.json"
    payload.write_bytes(canonical(dict(manifest=spec,resource_profile=PROFILE,runtime=runtime_identity(),source_hashes=source_hashes)))
    (output/"source-snapshot").mkdir()
    for name,path in paths.items():
        raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==spec["files"][name],"Owned source snapshot hash")
        (output/"source-snapshot"/name).write_bytes(raw)
    if weight_files:
        (output/"weight-snapshot").mkdir()
        for name,raw in weight_files.items():(output/"weight-snapshot"/name).write_bytes(raw)
    library=os.environ["HECATE_PYTHON_LIBRARY_PATH"]
    require(all(Path(p).is_relative_to("/nix/store") for p in library.split(":")),"Pinned library paths")
    command=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
        "--ro-bind","/nix/store","/nix/store","--ro-bind",str(VENV),str(VENV),
        "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
        "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app","--dir","/model",
        "--ro-bind",str(payload),"/request.json","--bind",str(output),"/out"]
    for name in IMPLEMENTATION:
        selected=prototype/name if (prototype/name).is_file() else ROOT/"scripts/baseline"/name
        require(sha(selected)==source_hashes[name],"Sandbox adapter binding")
        command+=["--ro-bind",str(selected),"/app/"+name]
    for name in paths:command+=["--ro-bind",str(output/"source-snapshot"/name),"/model/"+name]
    for name in weight_files:command+=["--ro-bind",str(output/"weight-snapshot"/name),"/model/"+name]
    for name,value in dict(PYTHONPATH="/app:/model",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
            LD_LIBRARY_PATH=library,HOME="/nonexistent",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1").items():
        command+=["--setenv",name,value]
    command+=["--chdir","/out","--",str(VENV/"bin/python"),"-B","/app/pytorch_model_intake.py","--worker"]
    with (output/"capture.log").open("x") as log:
        child=subprocess.Popen(command,env={"PATH":"/usr/bin:/bin"},stdout=log,stderr=subprocess.STDOUT,
            preexec_fn=limits,start_new_session=True)
        try:child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait();raise ValueError("Prototype isolated worker timeout")
    return child.returncode
def validate_report(report,original,lowered,spec,prototype,source_hashes):
    import trusted_pytorch_import as legacy
    from pytorch_model_intake import shim,check_reference,SHAPE_FORMAT,REFERENCES
    from pytorch_static_trace import policy_record,ARITHMETIC_POLICY
    from pytorch_arithmetic_normalization import check_record
    from model_resources import identity
    require(spec["format"]==SHAPE_FORMAT and report.get("format")==REFERENCES[SHAPE_FORMAT],"Prototype explicit v9 source scope")
    require(report.get("source_hashes")==source_hashes and report.get("manifest_sha256")==digest(spec)
        and report.get("original_sha256")==digest(original) and report.get("model_sha256")==digest(lowered),"Staged reference identity")
    require(report.get("runtime")==legacy.runtime_identity() and report.get("probes")==16
        and report.get("resource_profile")==identity(PROFILE),"Staged reference runtime/probe binding")
    for name in ("original_module_checked","normalized_module_checked","arithmetic_semantics_preserved","pure_forward_checked"):
        require(report.get(name) is True,"Staged reference scope: "+name)
    for name in ("encrypted_execution","agent_generated","runtime_layout_validated","approximation_accuracy_certified","source_precision_execution_compared"):
        require(report.get(name) is False,"Staged non-encrypted scope: "+name)
    require(report.get("paid_calls")==0 and report.get("reference_status")=="passed" and report.get("reference_dtype")=="float64"
        and report.get("mode")=="isolated_trusted_pytorch_reference"
        and report.get("comparison_tolerance")==dict(atol=1e-12,rtol=1e-12),"Staged strict independent reference")
    for name in ("max_decomposition_error","max_original_module_error","max_original_function_approximation_error",
                 "max_reference_tolerance_ratio","max_arithmetic_normalization_error","max_arithmetic_normalization_ratio"):
        value=report.get(name)
        require(type(value) in (int,float) and math.isfinite(value) and value>=0,"Finite reference metric")
    require(report["max_reference_tolerance_ratio"]<=1 and report["max_arithmetic_normalization_ratio"]<=1,"Staged numerical comparison")
    # Validate unchanged metadata with its retained validator; never rewrite the
    # worker report or pretend its staged implementation is production evidence.
    metadata=copy.deepcopy(report.get("capture_metadata"));arithmetic=metadata.pop("arithmetic_capture")
    require(arithmetic["resource_profile"]==PROFILE,"Arithmetic resource binding");check_record(arithmetic,original)
    expected=policy_record(spec["inputs"],PROFILE,policy=ARITHMETIC_POLICY)
    expected["version"]=prototype["source_metadata_policy"]
    require(canonical(metadata.pop("source_shape_capture"))==canonical(expected),"Staged shape policy binding")
    legacy.check_capture_metadata(metadata,original,shim(spec),resource_profile=PROFILE)
    try:check_reference(report,original,lowered,spec,resource_profile=PROFILE)
    except ValueError as error:
        require(str(error)=="Arithmetic reference source identity","Unexpected production rejection")
    else:raise ValueError("Staged reference was accepted as production evidence")
    return dict(staged_reference_checked=True,production_reference_rejected=True)
def guards(folder):
    from pytorch_model_intake import prepare
    require(not folder.exists(),"Fresh owned rejection fixture inventory")
    folder.mkdir()
    base=dict(format="poseidon-trusted-pytorch-v9",entry="models.make_model",
        inputs=[dict(name="x",shape=[1,1,5])],outputs=["result"],factory=dict(args=[],kwargs={}),
        weights={},dtype_policy="strict-float64-v1",
        approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))
    forward="        y = F.adaptive_avg_pool1d(x, 2)\n        return y.reshape(y.numel())\n"
    rows=[
        ("data_branch","        y = F.adaptive_avg_pool1d(x, 2)\n        if y.sum() > 0:\n            return y\n        return -y\n",base,"torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow"),
        ("unknown_result_metadata","        y = torch.flip(x, (2,))\n        return y.reshape(y.numel())\n",base,"ValueError: Static shape unavailable for this operation"),
        ("adaptive_upsample","        y = F.adaptive_avg_pool1d(x, 6)\n        return y.flatten()\n",base,"ValueError: Adaptive pool N,C,spatial downsampling shape"),
        ("adaptive_unbatched","        y = F.adaptive_avg_pool1d(x, 2)\n        return y.flatten()\n",{**base,"inputs":[dict(name="x",shape=[1,5])]},"ValueError: Adaptive pool N,C,spatial rank"),
        ("missing_norm_approximation","        y = F.instance_norm(x)\n        return y.reshape(y.numel())\n",base,"ValueError: Explicit approximation required for rsqrt at "),
        ("old_v8_metadata",forward,{**base,"format":"poseidon-trusted-pytorch-v8"},"ValueError: Static shape unavailable for this operation"),
        ("old_v8_functional","        return F.adaptive_avg_pool1d(x, 2).flatten()\n",{**base,"format":"poseidon-trusted-pytorch-v8"},"ValueError: Unsupported PyTorch call: ")]
    result=[]
    for name,body,spec,terminal in rows:
        d=folder/name;d.mkdir()
        source="import torch\nfrom torch.nn import functional as F\nclass Model(torch.nn.Module):\n    def forward(self, x):\n"+body+"\ndef make_model():\n    return Model().double().eval()\n"
        (d/"models.py").write_text(source);spec=copy.deepcopy(spec)
        spec["files"]={"models.py":sha(d/"models.py")};(d/"manifest.json").write_bytes(canonical(spec))
        prepare(d/"manifest.json")
        result.append(dict(id=name,manifest=str(d/"manifest.json"),expected_terminal=terminal))
    inventory={str(p.relative_to(folder)):sha(p) for p in folder.rglob("*") if p.is_file()}
    (folder/"inventory.json").write_bytes(canonical(dict(cases=result,files=inventory)))
    return result
def run(fixtures,prototype,previous,proposal_path,output):
    import numpy as np
    from hecate_python_env import VENV,WORK
    from pytorch_model_intake import prepare,source_identity
    from pytorch_weight_package import read
    from model_decomposition import decompose
    from operator_arithmetic_graph import decompose as declared_decompose,evaluate_reference
    from benchmark_math import evaluate as mathematics
    from benchmark_torch import evaluate as tensor_reference
    from benchmarks.tools.audit_pytorch_operator_shape_contexts import frozen
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh staged audit output")
    require(prototype.resolve().is_relative_to(ROOT/"scripts/baseline/benchmarks/tools/fixtures"),"Explicit development prototype")
    proposed=checked(json.loads(proposal_path.read_text()));bound=frozen(proposed)
    staged,current,expected=validate_prototype(prototype)
    old=checked(json.loads(previous.read_text()));catalog=checked(json.loads((fixtures/"catalog.json").read_text()))
    require(old["planned"]==39 and old["passed"]==21 and old["input_rejected"]==18 and old["unclassified"]==0,"Exact retained input gaps")
    require(old["capture_identity"]==current and old["proposal_binding"]==proposed["binding"],"Original rejection context identity")
    failed={r["id"] for r in old["rows"] if r["status"]=="rejected"}
    controls={"adaptive_pool_0_module_literal","adaptive_pool_1_module_literal","adaptive_pool_2_module_literal",
        "average_pool_control_0_functional_query","instance_norm_0_module_literal","padding_1_functional_query"}
    selected=[c for c in catalog["cases"] if c["id"] in failed|controls]
    require(len(selected)==24 and not(failed&controls),"Exact original failures and retained controls")
    preserved={str(p):sha(p) for d in (fixtures,previous.parent) for p in d.rglob("*") if p.is_file()}
    own=sha(__file__);output.mkdir(mode=0o700);(output/"captures").mkdir()
    rejects=guards(ROOT/"scripts/baseline/benchmarks/tools/fixtures/operator-shape-rejections-v1")
    started=time.monotonic();rows=[];negative=[];halt=None
    def remaining():
        left=150-(time.monotonic()-started);require(left>0,"Prototype audit time bound");return min(60,left)
    def save():
        result=dict(format="poseidon-operator-shape-prototype-audit-v1",prototype_binding=staged["binding"],
            production_integrated=False,production_source_identity=current,staged_source_identity=expected,
            proposal_binding=proposed["binding"],frozen_inventories=bound,original_audit_binding=old["binding"],
            planned_original_failures=18,original_failures_passed=sum(r["status"]=="passed" and r["scope"]=="original_failure" for r in rows),
            planned_controls=6,controls_passed=sum(r["status"]=="passed" and r["scope"]=="retained_control" for r in rows),
            planned_guards=len(rejects),guards_passed=sum(r["status"]=="rejected_as_expected" for r in negative),
            completed=len(rows)+len(negative),not_run=31-len(rows)-len(negative),
            isolated_module_reference_groups=sum(r.get("module_probes",0) for r in rows),
            independent_paired_groups=sum(r.get("paired_probes",0) for r in rows),
            production_reference_rejections=sum(r.get("production_reference_rejected",False) for r in rows),
            maximum_reference_error=max([r.get("reference_error",0) for r in rows] or [0]),rows=rows,guards=negative,
            halt=halt,seconds=time.monotonic()-started,tool_sha256=own,preserved_inventory=preserved,
            production_changes=0,backend_changes=0,reference_changes=0,compiled=0,hecate_tracing=0,
            encrypted_execution=0,decrypted=0,paid_calls=0,downloads=0,installs=0,agent_generated=0,
            scope="Explicit staged Agent intake probe; not public intake, program package or encrypted acceptance")
        result["binding"]=digest(result);(output/"report.json").write_bytes(canonical(result));return result
    save()
    try:
        for case in selected:
            require(source_identity()==current and validate_prototype(prototype)[2]==expected,"Capture sources changed")
            spec,paths=prepare(fixtures/case["manifest"])
            require(digest(spec)==case["manifest_sha256"],"Original manifest preserved")
            _,weights=read((fixtures/case["manifest"]).parent,spec["weights"],resource_profile=PROFILE)
            destination=output/"captures"/case["id"]
            code=sandbox(spec,paths,weights,destination,expected,prototype,remaining())
            row=dict(id=case["id"],scope="original_failure" if case["id"] in failed else "retained_control",
                exit=code,log_sha256=sha(destination/"capture.log"))
            if code:
                row.update(status="failed",terminal=(destination/"capture.log").read_text().splitlines()[-1])
            else:
                original=json.loads((destination/"original.json").read_text())
                report=json.loads((destination/"capture-reference.json").read_text());lowered,_=decompose(original,resource_profile=PROFILE)
                row.update(validate_report(report,original,lowered,spec,staged,expected))
                declared=json.loads((fixtures/case["graph"]).read_text())
                require(digest(declared)==case["graph_sha256"],"Original declared mathematics preserved")
                model,_=declared_decompose(declared,resource_profile=PROFILE);maximum=0.
                for inputs in samples(model,16):
                    reference=evaluate_reference(declared,inputs)["result"]
                    for value in (mathematics(lowered,inputs,resource_profile=PROFILE)["result"],
                                  tensor_reference(lowered,inputs,resource_profile=PROFILE)["result"]):
                        np.testing.assert_allclose(value,reference,atol=1e-12,rtol=1e-12)
                        maximum=max(maximum,float(np.max(np.abs(value-reference))))
                (destination/"lowered-audit.json").write_bytes(canonical(lowered))
                row.update(status="passed",module_probes=16,paired_probes=16,reference_error=maximum,
                    original_sha256=digest(original),lowered_sha256=digest(lowered),reference_sha256=sha(destination/"capture-reference.json"))
            rows.append(row);save()
        for case in rejects:
            spec,paths=prepare(Path(case["manifest"]));destination=output/"captures"/("guard_"+case["id"])
            code=sandbox(spec,paths,{},destination,expected,prototype,remaining())
            terminal=(destination/"capture.log").read_text().splitlines()[-1]
            match=code!=0 and (terminal.startswith(case["expected_terminal"]) if case["id"] in ("missing_norm_approximation","old_v8_functional") else terminal==case["expected_terminal"])
            if case["id"]=="old_v8_functional":
                match=match and re.fullmatch(r"ValueError: Unsupported PyTorch call: <built-in method adaptive_avg_pool1d of type object at 0x[0-9a-f]+>",terminal) is not None
            negative.append(dict(id=case["id"],status="rejected_as_expected" if match else "guard_failed",
                exit=code,terminal=terminal,log_sha256=sha(destination/"capture.log")));save()
    except Exception as error:
        halt=dict(layer="staged_audit",type=type(error).__name__,reason=str(error));save();raise
    require(frozen(proposed)==bound and source_identity()==current and validate_prototype(prototype)[2]==expected,"Final frozen identities")
    require(all(sha(p)==h for p,h in preserved.items()) and sha(__file__)==own,"Original evidence/tools preserved")
    result=save()
    print(json.dumps({k:result[k] for k in ("binding","completed","not_run","original_failures_passed","controls_passed","guards_passed",
        "isolated_module_reference_groups","independent_paired_groups","production_reference_rejections","maximum_reference_error","seconds","halt")}))
    return 0 if result["original_failures_passed"]==18 and result["controls_passed"]==6 and result["guards_passed"]==7 else 1
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fixtures","prototype","previous","proposal","output"):parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args();raise SystemExit(run(a.fixtures,a.prototype,a.previous,a.proposal,a.output))
