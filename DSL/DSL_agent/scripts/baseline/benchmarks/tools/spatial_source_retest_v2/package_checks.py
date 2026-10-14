"""Post-generation package acceptance only; never produces a candidate."""
import json, os, sys
from pathlib import Path
from scope import ROOT, BASE, CASES, PROPOSAL, read, require, sha, canonical

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(canonical(value));os.chmod(tmp,0o600);tmp.replace(path)

def forward_samples(meta,samples):
    from pytorch_input_binding import materialize
    manifest=meta["model_provenance"]["trusted_pytorch"]["manifest"]
    require(manifest["format"]=="poseidon-trusted-pytorch-v27","structured_package_source")
    call=manifest["call"]
    return [dict(args=[materialize(v,b)for v in call["args"]],
        kwargs={k:materialize(v,b)for k,v in call["kwargs"].items()})for b in samples]

def one(output,name,phase):
    require(name in CASES and phase in ("new_inputs","repeat","cache"),"reviewed_package_phase")
    sys.path.insert(0,str(BASE))
    import numpy as np
    from hecate_python_env import VENV
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from application_model_source import FORMAT
    from candidate_bundle import check_bundle
    from candidate_contract import request_file_bytes
    from pytorch_source_contract import restore_outputs
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"pinned_package_worker")
    directory=Path(output)/name
    require(not (directory/(phase+".json")).exists(),"single_package_phase")
    proposal=read(PROPOSAL);case=next(c for c in proposal["cases"] if c["id"]==name)
    task=read(directory/"task-id.json")["task_id"]
    app=application(WorkerContext.capture(directory/"tasks"))
    meta=app._checked_task_metadata(task);package=app._dir(task)/"program"
    require(meta["request"]==read(case["request_file"]),
            "frozen_package_request")
    require(meta["identity"]==proposal["source_stage_identity"],"frozen_package_stage_identity")
    before=check_bundle(package);model=meta["request"]["model"]
    from program_execution import package_execution_metadata
    expected_structure=package_execution_metadata(package,before,model)["output_structure"]
    require(before["agent_generated"] is True and before["validation"]["level"]=="numerical",
            "new_agent_numerical_package")
    row=dict(case=name,phase=phase,status="failed",provider_calls=0,
             new_agent_generation=False,package_binding=before["binding"])
    try:
        if phase in ("new_inputs","repeat"):
            from benchmark_math import evaluate as math_ref
            from benchmark_torch import evaluate as torch_ref
            from numerical_results import compare
            samples=[];reference=[]
            group_count=3 if phase=="new_inputs" else 1
            for index in range(group_count):
                rng=np.random.default_rng(20261008+index)
                values={s["name"]:rng.uniform(-.2,.2,s["shape"]).astype(np.float64)
                        for s in model["inputs"]}
                a,b=math_ref(model,values),torch_ref(model,values)
                require(set(a)==set(b)=={o["name"] for o in model["outputs"]},
                        "independent_reference_outputs")
                for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
                samples.append({key:value.tolist() for key,value in values.items()})
                reference.append(np.concatenate([a[o["name"]].reshape(-1) for o in model["outputs"]]))
            if phase=="repeat":
                execution=app.execute(task,forward_samples(meta,samples),sample_format="forward_call")
            else:
                execution=app.execute(task,samples)
            actual=np.asarray([np.concatenate([np.asarray(s[o["name"]],dtype=np.float64).reshape(-1)
                for o in model["outputs"]]) for s in execution["outputs"]],dtype=np.float64)
            compared=compare(actual,np.asarray(reference,dtype=np.float64),1e-5,1e-4)
            require(execution["encrypted_execution"] is True and execution["executor_secret_key"] is False
                    and execution["agent_calls"]==execution["compiler_calls"]==0,"real_package_execution")
            require(compared["passed"],"frozen_numerical_tolerance")
            require(before==check_bundle(package),"reloaded_package_integrity")
            precheck=execution["resource_precheck"]
            require(execution["execution"]["resource_precheck"]==precheck,
                    "mandatory_execution_resource_precheck")
            evidence=Path(execution["evidence"])/"output"
            require(read(evidence/"resource-precheck.json")==precheck and
                    read(evidence/"execution-metadata.json")==execution["execution"],
                    "execution_resource_receipt")
            require(precheck["status"]=="assessed" and precheck["failure"] is None and
                    precheck["facts"]["known_retained_payload_lower_bytes"]<=
                    precheck["limits"]["memory_budget_bytes"],"resource_lower_bound")
            require(execution.get("output_structure")==expected_structure,"preserved_output_tree")
            if expected_structure is not None:
                require("structured_outputs" in execution and execution["structured_outputs"]==
                        [restore_outputs(values,expected_structure,allow_none=True) for values in execution["outputs"]],
                        "original_output_structure")
            row.update(status="passed",execution=execution,comparison=compared,input_groups=group_count,
                       sample_format="forward_call" if phase=="repeat" else "named",
                       compiled=False,encrypted_execution=True,package_reloaded=True,
                       resource_precheck_verified=True,original_application_accuracy_certified=False)
        else:
            origin=read(app._dir(task)/"source.json")
            source=dict(format=FORMAT,kind="trusted_pytorch",
                        manifest=str(ROOT/case["source_manifest"]),trust_local_code=True)
            again=app.submit_source(source,resource_profile="general-tensors-v1",
                    options=origin["requirements"]["options"],limits=origin["budget"])
            class NeverGenerate:
                kind="scripted_replay";agent_calls=0
                def generate(self,*args):raise AssertionError("cached_package_required")
            result=app.run(again,NeverGenerate())
            require(result["status"]=="succeeded" and result["package"]["reused"] is True
                    and result["package"]["binding"]==before["binding"]
                    and result["generations"]==result["http_attempts"]==0,"reusable_source_package")
            require(before==check_bundle(package),"cache_preserves_package")
            row.update(status="passed",result=result,package_reused=True,
                       compiled=False,encrypted_execution=False)
    except Exception as error:
        row.update(error_type=type(error).__name__,diagnostic=str(error)[:300])
        row.setdefault("failure_layer","package_acceptance")
    write(directory/(phase+".json"),row)
    return 0 if row["status"]=="passed" else 1
