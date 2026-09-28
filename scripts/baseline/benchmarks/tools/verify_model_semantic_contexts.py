"""Bounded, no-fee dual-reference validation of supplementary context graphs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,validate,signature
from build_model_semantic_contexts import read_suite,RECIPES,build

def checked_bundle(path,suite):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>4*1024**2:
        raise ValueError("Context bundle file gate")
    bundle=json.loads(path.read_text())
    body={k:v for k,v in bundle.items() if k!="bundle_sha256"}
    if digest(body)!=bundle["bundle_sha256"]:raise ValueError("Context bundle hash")
    binding=bundle["source_binding"]
    if hashlib.sha256((suite/"index.json").read_bytes()).hexdigest()!=binding["index_sha256"]:
        raise ValueError("Context suite changed")
    if hashlib.sha256((suite/"coverage.json").read_bytes()).hexdigest()!=binding["coverage_sha256"]:
        raise ValueError("Context ledger changed")
    allowed=[HERE/"build_model_semantic_contexts.py",BASE/"benchmark_graph.py",
             BASE/"benchmark_semantics.py",BASE/"benchmark_suite.py"]
    expected={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in allowed}
    if binding["generator_sources"]!=expected:raise ValueError("Context generator binding mismatch")
    for name,h in binding["generator_sources"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:raise ValueError("Unsafe source path")
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError("Context generator drift")
    rows,index,ledger=read_suite(suite)
    if build(rows,ledger,binding)!=bundle:raise ValueError("Context bundle differs from bound generator")
    parents={r["model"]["id"]:r for r in rows}
    extra=bundle["supplemental_models"]
    if not 1<=len(extra)<=256:raise ValueError("Supplementary model count")
    if len({r["model"]["id"] for r in extra})!=len(extra):raise ValueError("Duplicate supplementary model")
    for row in extra:
        model=row["model"];validate(model)
        if (digest(model),signature(model),signature(model,True))!=(row["model_sha256"],row["signature"],row["topology"]):
            raise ValueError("Supplementary identity mismatch")
        prov=row["provenance"]
        if prov["recipe"] not in RECIPES:raise ValueError("Unknown context recipe")
        parent=parents[prov["parent_id"]]
        if parent["model_sha256"]!=prov["parent_sha256"] or row["split"]!=parent["split"]:
            raise ValueError("Supplementary parent/split mismatch")
    return bundle,parents

def transformed_inputs(inputs,recipe):
    if recipe.startswith("input_affine"):return {k:-0.5*v+0.125 for k,v in inputs.items()}
    if recipe.startswith("input_negate"):return {k:-v for k,v in inputs.items()}
    return inputs

def transformed_first(value,recipe):
    if "output_residual" in recipe:return value*value+value
    if recipe=="output_negate":return -value
    return value

def execute(bundle,parents,out,binding,sources):
    import numpy as np
    import torch
    from benchmark_graph import samples
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as pytorch
    from benchmark_runner import ideal_reference
    from platform_config import require_python_packages,identity
    require_python_packages(torch,np);torch.set_num_threads(1)
    if out.exists():raise ValueError("Preserve existing context validation")
    out.mkdir(parents=True)
    def dump(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (out/"runner.py").write_bytes(Path(__file__).read_bytes())
    dump(out/"bundle.json",bundle)
    dump(out/"run.json",dict(binding=binding,source_hashes=sources,
                            bundle_sha256=bundle["bundle_sha256"],
                            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            reference_atol=1e-12,reference_rtol=1e-12,probes_per_model=16,
                            platform=identity(),numpy=np.__version__,
                            torch=torch.__version__,python=sys.version,paid_calls=0))
    start=time.monotonic();records=[]
    for row in bundle["supplemental_models"]:
        if time.monotonic()-start>120:raise RuntimeError("120 second reference budget exhausted; retain evidence")
        g=row["model"];prov=row["provenance"];parent=parents[prov["parent_id"]];recipe=prov["recipe"]
        result=dict(id=g["id"],model_sha256=row["model_sha256"],provenance=prov,
                    state="not_run",probes=0,max_reference_error=0.0,
                    max_composition_error=0.0,output_hashes=[],encrypted_execution=False)
        try:
            for inputs in samples(g):
                expected=mathematical(g,inputs);actual=pytorch(g,inputs)
                changed_inputs=transformed_inputs(inputs,recipe)
                composed=mathematical(parent["model"],changed_inputs)
                first=parent["model"]["outputs"][0]["name"]
                composed[first]=transformed_first(composed[first],recipe)
                for name,value in expected.items():
                    if value.shape!=actual[name].shape or not np.isfinite(value).all() or not np.isfinite(actual[name]).all():
                        raise ValueError("Reference nonfinite or shape mismatch")
                    np.testing.assert_allclose(actual[name],value,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(composed[name],value,atol=1e-12,rtol=1e-12)
                    result["max_reference_error"]=max(result["max_reference_error"],float(np.max(np.abs(value-actual[name]))))
                    result["max_composition_error"]=max(result["max_composition_error"],float(np.max(np.abs(value-composed[name]))))
                ideal=ideal_reference(parent,changed_inputs)
                if ideal is not None:
                    ideal=transformed_first(ideal,recipe)
                    if ideal.shape!=expected[first].shape:raise ValueError("Ideal reference shape")
                    result["ideal_function_max_abs_difference"]=max(result.get("ideal_function_max_abs_difference",0),
                                                                  float(np.max(np.abs(expected[first]-ideal))))
                    result["ideal_function_reference_scope"]="parent ideal function with explicit context input/output transformations"
                result["output_hashes"].append(dict(math=digest({n:v.tolist() for n,v in expected.items()}),
                                                    torch=digest({n:v.tolist() for n,v in actual.items()})))
                result["probes"]+=1
            result["state"]="passed"
        except (ValueError,AssertionError,TypeError,KeyError,IndexError,OverflowError) as error:
            result.update(state="failed",failure_layer="supplemental_dual_reference",error=str(error)[:2000])
        dump(out/(g["id"]+".json"),result);records.append(result)
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>64*1024**2:
            raise RuntimeError("64 MiB evidence budget exhausted; retain evidence")
    report=dict(binding=binding,planned=len(bundle["supplemental_models"]),
                passed=sum(r["state"]=="passed" for r in records),failed=sum(r["state"]=="failed" for r in records),
                skipped=0,probes=sum(r["probes"] for r in records),
                max_reference_error=max(r["max_reference_error"] for r in records),
                max_composition_error=max(r["max_composition_error"] for r in records),
                approximate_cases=sum("ideal_function_max_abs_difference" in r for r in records),
                result_hashes={r["id"]:hashlib.sha256((out/(r["id"]+".json")).read_bytes()).hexdigest() for r in records},
                seconds=time.monotonic()-start,encrypted_execution=False,paid_calls=0,original_models_unchanged=True)
    dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="result_hashes"}))
    return int(report["failed"]>0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True);p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--output",type=Path);p.add_argument("--execute",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    bundle,parents=checked_bundle(a.bundle,a.suite)
    from semantic_benchmark_execution import runtime_sources
    sources=runtime_sources()
    binding=digest(dict(bundle_sha256=bundle["bundle_sha256"],runtime_sources=sources,
                        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        probes=16,atol=1e-12,rtol=1e-12))
    if not a.execute:
        print(json.dumps(dict(binding=binding,supplementary_models=len(bundle["supplemental_models"]),
                              probes_per_model=16,max_wall_seconds=120,max_result_mib=64,paid_calls=0,
                              encrypted_execution=False,state="plan_only")))
        return 0
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if a.output is None or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("Platform results output required")
    if not a.inside:
        command=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
        return enter_nix('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(command),seconds=150)
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:p.error("Locked Nix Python required")
    status=execute(bundle,parents,a.output,binding,sources)
    if runtime_sources()!=sources:raise ValueError("Source changed during context validation")
    return status
if __name__=="__main__":raise SystemExit(main())
