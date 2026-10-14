"""No-HTTP rebind of the same three whole models after Python import migration."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from workspace_paths import WORK
from benchmark_graph import canonical,digest,require,samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from component_contract import prepare_task
from poseidon_backend_identity import identity
from check_poseidon_helpers import snapshot
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    prior_path=ROOT/"docs/baseline/poseidon-whole-models-live-proposal-r274.json"
    prior=json.loads(prior_path.read_text());target=ROOT/"docs/baseline/poseidon-whole-models-live-proposal-r284.json"
    out=WORK/"results/poseidon-whole-models-prepare-r284"
    require(not target.exists() and not out.exists(),"Fresh proposal and evidence required")
    old_core=prior["source_hashes"];new_core=snapshot()
    changed={p:dict(before=h,after=new_core.get(p)) for p,h in old_core.items() if new_core.get(p)!=h}
    added=set(new_core)-set(old_core)
    require(set(old_core)<=set(new_core) and set(changed)=={"scripts/baseline/model_import_cli.py","scripts/baseline/trusted_pytorch_import.py","scripts/baseline/candidate_bundle.py"} and
            added=={"scripts/baseline/test_poseidon_model_import.py"},"Unexpected live-core drift")
    changed.update({p:dict(before=None,after=new_core[p]) for p in added})
    require(identity()==prior["backend_identity"],"Backend binaries/dependencies changed")
    out.mkdir();proposal=copy.deepcopy(prior)
    proposal.update(status="pending_approval",source_hashes=new_core,evidence_root=str(out),
                    supersedes=dict(proposal=str(prior_path),binding=prior["binding"],
                                    proposal_sha256=sha(prior_path),prior_executed=False,
                                    reason="Poseidon model import environment, isolated PyTorch reference and source provenance"),
                    adapter_boundary=ROOT.joinpath("docs/baseline/poseidon-python-import-migration-r285.json").as_posix(),
                    preparation_paid_calls=0,source_changes=changed)
    proposal.pop("binding");rows=[]
    for case in proposal["cases"]:
        model_path=ROOT/case["model_path"];model=json.loads(model_path.read_text())
        require(sha(model_path)==case["model_file_sha256"] and digest(model)==case["model_sha256"],"Frozen model changed")
        request=prepare_task(model,dict(compiler_configuration="poseidon-cpu-eva-w45-v1",generation_guidance="explicit-v10"))
        prior_request_path=Path(case["prepare_evidence"])/"request.json"
        prior_request_raw=prior_request_path.read_bytes()
        require(sha(prior_request_path)==case["request_sha256"],"Prior request file changed")
        require(request["request_id"]==case["request_id"] and canonical(request)==canonical(json.loads(prior_request_raw)),
                "Public request changed")
        errors=[]
        for inputs in samples(model,16):
            a=mathematical(model,inputs);b=torch_reference(model,inputs)
            for name in a:
                np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
                errors.append(float(np.max(np.abs(a[name]-b[name]))))
        folder=out/case["id"];folder.mkdir();(folder/"request.json").write_bytes(prior_request_raw)
        case["prepare_evidence"]=str(folder)
        rows.append(dict(id=case["id"],request_id=request["request_id"],model_sha256=digest(model),
                         reference_groups=16,reference_max_absolute_error=max(errors),paid_calls=0))
    proposal["binding"]=digest(proposal);target.write_text(json.dumps(proposal,indent=2))
    (out/"report.json").write_bytes(canonical(dict(rows=rows,source_changes=changed,binding=proposal["binding"],
             models_unchanged=True,public_requests_unchanged=True,backend_unchanged=True,paid_calls=0)))
    print(json.dumps(dict(proposal=str(target),binding=proposal["binding"],cases=[r["id"] for r in rows],
                         maximum_generations=proposal["maximum_generations"],
                         maximum_http_attempts=proposal["maximum_http_attempts"],paid_calls=0)),flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
