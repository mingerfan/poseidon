"""Real isolated captures; reuses immutable direct reference evidence."""
import hashlib,json,os,sys,time,traceback,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import WORK,VENV
from benchmark_graph import require,digest
from application_identity import current_identity_manifest
from pytorch_model_intake import run,package_format
from pytorch_ordered_contract import encode_weight_files
from candidate_bundle import check_provenance
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned isolated source environment")
rows=json.loads((HERE/"source-contexts-r2.json").read_text())
require(len(rows)==33,"Exact public dtype contexts")
sdk=current_identity_manifest();start=time.monotonic()
out=WORK/"results/agent-public-dtype-cast-source-v24-v1"
require(not out.exists(),"Fresh source capture results")
out.mkdir(mode=0o700)
write=lambda p,d:p.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n")
done=[];failed=[]
for row in rows:
    require(time.monotonic()-start<220,"Source capture wall boundary")
    require(shutil.disk_usage(WORK).free>=4*1024**3,"Source capture disk reserve")
    manifest=Path(row["manifest_path"]);spec=json.loads(manifest.read_text());directory=out/row["case"]
    previous=json.loads(Path(row["provenance"]).read_text())
    check_provenance(previous,json.loads(Path(row["model"]).read_text()))
    try:
        result=run(manifest,directory,resource_profile="general-tensors-v1",trusted=True)
        require(result["status"]=="captured"and result["reference_verified"]and
                not result["encrypted_execution"]and not result["agent_generated"],"Isolated source scope")
        model=json.loads((directory/"model.json").read_text())
        require(model==json.loads(Path(row["model"]).read_text()),"Source/direct mathematical graph equality")
        public={w["file"]:(manifest.parent/w["file"]).read_bytes()for w in spec["weights"].values()}
        provenance=dict(original=json.loads((directory/"original.json").read_text()),
            decomposition=json.loads((directory/"decomposition.json").read_text()),
            reference_report=json.loads((directory/"reference-report.json").read_text()),
            trusted_pytorch=dict(format=package_format(spec),manifest=spec,
                files={name:(manifest.parent/name).read_text()for name in spec["files"]},
                weight_files=encode_weight_files(spec,public,resource_profile="general-tensors-v1")))
        check_provenance(provenance,model);write(directory/"provenance.json",provenance)
        done.append(dict(case=row["case"],status="passed",model=str(directory/"model.json"),
            provenance=str(directory/"provenance.json"),manifest=str(manifest)))
    except Exception as exc:
        failed.append(dict(case=row["case"],error=type(exc).__name__,reason=str(exc)[:200]))
        (out/(row["case"]+"-failure-private.txt")).write_text(traceback.format_exc())
    require(current_identity_manifest()["manifest_sha256"]==sdk["manifest_sha256"],"No source mutation during capture")
record=dict(format="poseidon-public-dtype-isolated-capture-v1",source_contract="poseidon-trusted-pytorch-v24",
 sdk_identity=sdk["manifest_sha256"],stage_identities={k:sdk[k]["sha256"]for k in["generation","compilation","execution","validation"]},
 passed=len(done),failed=len(failed),skipped=0,not_run=33-len(done)-len(failed),cases=done,failures=failed,
 independent_reference_groups=sum(row["source_reference_groups"]for row in rows),
 independent_reference_values=sum(row["source_reference_values"]for row in rows),
 manual_math_groups=sum(row["independent_groups"]for row in rows),
 manual_max_absolute_error=max(row["independent_max_absolute_error"]for row in rows),
 seconds=time.monotonic()-start,compiler_calls=0,encrypted_calls=0,provider_calls=0)
record["binding"]=digest(record);write(out/"report.json",record)
with(HERE/"isolated-controls.json").open("x")as f:json.dump(record,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in record.items()if k not in["cases","stage_identities"]}))
raise SystemExit(0 if len(done)==33 else 1)
