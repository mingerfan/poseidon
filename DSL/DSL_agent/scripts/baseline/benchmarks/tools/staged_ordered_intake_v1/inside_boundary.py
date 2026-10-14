"""Exercise the real facade.run boundary with retained public fixtures."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,"/app")
import numpy as np,torch
from platform_config import require_python_packages,loaded_libraries,identity
from benchmark_graph import digest
import pytorch_model_intake as facade
from candidate_bundle import load_model_provenance,audit_provenance
require_python_packages(torch,np)
assert sys.version_info[:3]==(3,10,14)
torch.set_num_threads(1)
output=Path(os.environ["POSEIDON_BOUNDARY_RESULTS"])
measurements=[]
started=time.monotonic()
for name in ("linear","norm","split_weight"):
    target=output/("capture-"+name)
    result=facade.run("/fixtures/"+name+"/manifest.json",target,
                      resource_profile="general-tensors-v1",trusted=True)
    assert result["status"]=="captured" and result["reference_verified"]
    assert result["encrypted_execution"] is False and result["agent_generated"] is False
    model=json.loads((target/"model.json").read_text())
    original=json.loads((target/"original.json").read_text())
    reference=json.loads((target/"reference-report.json").read_text())
    provenance=load_model_provenance(target,model,max_bytes=4*1024**2)
    audit_provenance(provenance,model)
    assert digest(original)==reference["original_sha256"]
    assert reference["runtime"]["torch"]=="2.0.1"
    assert reference["probes"]==16 and reference["reference_status"]=="passed"
    manifest=json.loads((target/"trusted-python/manifest.json").read_text())
    for declaration in manifest["weights"].values():
        path=target/"trusted-python"/declaration["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==declaration["sha256"]
    measurements.append(dict(case=name,result=result,finite_reference=reference["finite_reference"],
        model_sha256=digest(model),provenance_sha256=digest(provenance),
        source_hashes=reference["source_hashes"],retained_weights=len(manifest["weights"]),
        reloaded=True))
wrong=output/"must-not-exist-wrong-platform"
code="""from pytorch_model_intake import run
try: run('/fixtures/linear/manifest.json',__import__('sys').argv[1],trusted=True)
except RuntimeError as error:
    assert 'actual linux/aarch64' in str(error),str(error)
else: raise AssertionError('Wrong platform accepted')
"""
env=dict(os.environ,POSEIDON_PLATFORM="x86_64-linux")
denied=subprocess.run([sys.executable,"-B","-c",code,str(wrong)],env=env,
                      capture_output=True,text=True,timeout=15)
assert denied.returncode==0,denied.stderr
assert not wrong.exists()
report=dict(format="poseidon-staged-ordered-boundary-v1",status="passed",
    source_capture_runs=3,wrong_platform_rejections=1,source_packages_reloaded=3,
    measured=measurements,seconds=time.monotonic()-started,platform=identity(),
    runtime=dict(python=list(sys.version_info[:3]),torch=torch.__version__,numpy=np.__version__),
    mapped_libraries=loaded_libraries(),network_namespace=str(Path("/proc/self/ns/net").readlink()),
    parent_probe_marker_present=os.environ.get("POSEIDON_PROBE_SECRET")=="noncredential-test-marker",
    production_registered=False,hecate_traces=0,compilations=0,encrypted_executions=0,paid_calls=0)
(output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({k:report[k] for k in ("status","source_capture_runs","wrong_platform_rejections",
                                   "source_packages_reloaded","runtime")}))
