"""Staged real source sandbox and semantic rejection acceptance; no compiler/API."""
import sys,json,shlex,time,traceback,hashlib,copy,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
      [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(cmd,seconds=360,environment="poseidon-cpu"))
sys.path.insert(0,str(HERE/"component-stage"))
from hecate_python_env import WORK
from benchmark_graph import digest
from pytorch_model_intake import run
from candidate_bundle import check_provenance
from pytorch_ordered_contract import encode_weight_files,source_identity
from pytorch_weight_package import read
import application_capabilities
out=WORK/"results"/("agent-spatial-source-sandbox-stage-"+str(time.time_ns()));out.mkdir()
start=time.monotonic();rows=[]
batch=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-spatial-source-component-stage-1791728068527392953")
selected=[(n,batch/n/"source/manifest.json")for n in("transpose_1_0","structured_dict","nested_keywords","float32_public_weight")]
compat=BASE/"benchmarks/tools/pytorch_input_binding_v1/cases-final"
selected += [("compat_v26_"+n,compat/n/"manifest.json")for n in("keywords0","nested0")]
for name,original in selected:
    row=dict(case=name,status="running");rows.append(row)
    try:
        spec=json.loads(original.read_text());source=out/(name+"-source");source.mkdir()
        contents={f:(original.parent/f).read_text()for f in spec["files"]}
        if "spatial_component_fixtures.py"in contents:
            text=contents["spatial_component_fixtures.py"].replace("from pytorch_input_binding import CALL_FORMAT",
                'CALL_FORMAT="poseidon-pytorch-call-v1"')
            contents["spatial_component_fixtures.py"]=text
        for f,text in contents.items():(source/f).write_text(text)
        spec["files"]={f:hashlib.sha256(t.encode()).hexdigest()for f,t in contents.items()}
        for info in spec["weights"].values():
            f=info["file"]
            if not (source/f).exists():(source/f).write_bytes((original.parent/f).read_bytes())
        manifest=source/"manifest.json";manifest.write_text(json.dumps(spec,allow_nan=False))
        capture=out/name;receipt=run(manifest,capture,resource_profile="general-tensors-v1",trusted=True)
        high=json.loads((capture/"original.json").read_text());lower=json.loads((capture/"model.json").read_text())
        bound=json.loads((capture/"decomposition.json").read_text());reference=json.loads((capture/"reference-report.json").read_text())
        arrays,raw=read(source,spec["weights"],resource_profile="general-tensors-v1")
        from pytorch_ordered_contract import package_version
        provenance=dict(original=high,decomposition=bound,reference_report=reference,
            trusted_pytorch=dict(format=package_version(spec),manifest=spec,files=contents,
                weight_files=encode_weight_files(spec,raw,resource_profile="general-tensors-v1")))
        check_provenance(provenance,lower)
        (capture/"provenance.json").write_text(json.dumps(provenance,allow_nan=False))
        row.update(status="passed",receipt=receipt,plaintext=reference["finite_reference"],
                   manifest_sha256=digest(spec),source_hashes=spec["files"])
    except Exception as error:
        row.update(status="failed",error_type=type(error).__name__,error=str(error))
        (out/(name+"-failure.txt")).write_text(traceback.format_exc())
    print(json.dumps({k:v for k,v in row.items()if k not in("plaintext","source_hashes")}),flush=True)
suite=unittest.defaultTestLoader.loadTestsFromName("test_pytorch_spatial_source")
with(out/"semantics-tests.log").open("x")as stream:
    unit=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
caps=application_capabilities.input_sources()["trusted_pytorch"]
assert "poseidon-trusted-pytorch-v27"in caps["formats"]and caps["spatial_calls"]["version"]=="poseidon-trusted-pytorch-v27"
assert not any("-spatial-core-"in v or "-spatial-public-"in v for v in caps["formats"])
report=dict(format="poseidon-spatial-source-sandbox-stage-v1",result=str(out),
    passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]=="failed"for r in rows),rows=rows,
    unit_tests=dict(run=unit.testsRun,failed=len(unit.failures),errors=len(unit.errors),skipped=len(unit.skipped)),
    source_identity=source_identity(),elapsed_seconds=time.monotonic()-start,
    sandbox=dict(bubblewrap=True,network="unshare-all",candidate_resource_limits="unchanged"),
    production_registered=False,api_calls=0,compilations=0,encrypted_calls=0)
report["binding"]=digest(report);(out/"report.json").write_text(json.dumps(report,allow_nan=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items()if k not in("rows","source_identity")}),flush=True)
raise SystemExit(1 if report["failed"]or not unit.wasSuccessful()else 0)
