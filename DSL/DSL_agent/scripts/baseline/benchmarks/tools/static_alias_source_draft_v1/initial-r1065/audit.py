"""Known-local FX and independent math acceptance for ordered alias normalization."""
import argparse,hashlib,json,math,os,sys,time,unittest
from pathlib import Path
import numpy as np,torch
from benchmark_graph import digest,require
from benchmark_math import evaluate as core_math
from benchmark_torch import evaluate as core_torch
from hecate_python_env import ROOT,VENV,WORK
from platform_config import require_python_packages
from strict_json_files import dump
from alias_source_reference import evaluate as reference
import fixtures,test_alias_source_plan as tests

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity():
    local=Path(__file__).parent
    files=[local/n for n in ("alias_source_plan.py","fx_alias_source_plan.py","alias_source_reference.py","fixtures.py","test_alias_source_plan.py","audit.py")]
    files += [ROOT/"scripts/baseline"/n for n in ("benchmark_graph.py","benchmark_math.py","benchmark_torch.py","model_decomposition.py","model_resources.py","platform_config.py")]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}
def samples(signature,seed):
    random=np.random.default_rng(seed);rows=[]
    for probe in range(16):
        inputs={}
        for spec in signature:
            s=spec["shape"];n=math.prod(s)
            if probe==0:a=np.zeros(s,dtype=np.float64)
            elif probe==1:a=np.linspace(-.15,.15,n,dtype=np.float64).reshape(s)
            elif probe==3:a=np.asarray([(-1)**i*.15 for i in range(n)],dtype=np.float64).reshape(s)
            else:a=random.uniform(-.15,.15,s).astype(np.float64)
            inputs[spec["name"]]=a
        rows.append(inputs)
    return rows
def outputs(result):
    values=result if type(result) is tuple else (result,)
    return {"output_"+str(i):x.detach().numpy().copy() for i,x in enumerate(values)}
def run(output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require_python_packages(torch,np)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh evidence")
    output.mkdir(mode=0o700);(output/"models").mkdir()
    frozen=identity();started=time.monotonic()
    with (output/"units.log").open("x") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    units=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    dump(output/"units-result.json",units)
    require(result.wasSuccessful() and not result.skipped,"Unit/negative/binding checks")
    rows=[]
    for index,case in enumerate(fixtures.CASES):
        require(time.monotonic()-started<180 and identity()==frozen,"Time and source identity")
        fixture,traced,plan,metadata=fixtures.capture(case)
        dump(output/"models"/(case["id"]+".json"),plan)
        dump(output/"models"/(case["id"]+".fx-mapping.json"),metadata)
        (output/"models"/(case["id"]+".fx.py")).write_text(traced.code)
        errors=dict(original=0.,fixture=0.,decomposition=0.,torch=0.,approximation=0.);observations=[];values=0
        for probe,inputs in enumerate(samples(plan["source_request"]["source"]["inputs"],15001+index)):
            args=[torch.from_numpy(inputs[n].copy()) for n in ("x","other") if n in inputs]
            native=outputs(fixture(*[a.clone() for a in args]));captured=outputs(traced(*[a.clone() for a in args]))
            original=reference(plan["source_request"],inputs,approximate=False)
            expected=reference(plan["source_request"],inputs,approximate=True)
            actual=core_math(plan["lowered_graph"],inputs);tensor=core_torch(plan["lowered_graph"],inputs)
            require(set(native)==set(captured)==set(original)==set(expected)==set(actual)==set(tensor),"Named outputs")
            for name in native:
                all_values=(native[name],captured[name],original[name],expected[name],actual[name],tensor[name])
                require(all(a.shape==native[name].shape and np.isfinite(a).all() for a in all_values),"Shape and finite values")
                require(np.allclose(native[name],original[name],rtol=1e-12,atol=1e-12),"Independent original source reference")
                require(np.allclose(native[name],captured[name],rtol=1e-12,atol=1e-12),"Native and fixture FX")
                require(np.allclose(actual[name],expected[name],rtol=1e-12,atol=1e-12),"Independent approximate alias reference")
                require(np.allclose(tensor[name],expected[name],rtol=1e-12,atol=1e-12),"Core Torch")
                for field,a,b in (("original",native[name],original[name]),("fixture",native[name],captured[name]),
                                   ("decomposition",actual[name],expected[name]),("torch",tensor[name],expected[name]),
                                   ("approximation",expected[name],original[name])):
                    errors[field]=max(errors[field],float(np.max(np.abs(a-b))))
                values+=int(native[name].size)
            observations.append(dict(probe=probe,inputs={n:v.tolist() for n,v in inputs.items()},
                original={n:v.tolist() for n,v in original.items()},approximate={n:v.tolist() for n,v in expected.items()},
                lowered_numpy={n:v.tolist() for n,v in actual.items()},lowered_torch={n:v.tolist() for n,v in tensor.items()}))
        dump(output/(case["id"]+".references.json"),dict(rows=observations))
        row=dict(id=case["id"],status="math_passed",configuration=case,reference_groups=16,output_values=values,
            source_request_sha256=plan["source_request_sha256"],plan_binding=plan["binding"],fx_binding=metadata["binding"],
            original_nodes=len(plan["mathematical_graph"]["nodes"]),expanded_nodes=len(plan["lowered_graph"]["nodes"]),
            alias_events=plan["events"],output_schema=metadata["output_schema"],maximum_errors=errors,
            original_function_accuracy_certified=False,production_intake_integrated=False)
        rows.append(row);dump(output/"progress.json",dict(rows=rows))
        print(json.dumps({k:row[k] for k in ("id","status","expanded_nodes","maximum_errors")}),flush=True)
    require(identity()==frozen,"Final identity")
    report=dict(format="poseidon-alias-source-draft-audit-v1",status="draft_math_validated",units=units,
        cases=len(rows),math_passed=len(rows),failed=0,skipped=0,rows=rows,
        reference_groups=sum(r["reference_groups"] for r in rows),output_values=sum(r["output_values"] for r in rows),
        source_hashes=frozen,runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,hip=torch.version.hip),
        trusted_local_fixture_fx_captures=len(rows),production_source_capture=0,production_intake_integrated=False,
        hecate_tracing=0,compiler_calls=0,encrypted_execution=0,decryption=0,agent_generation=0,paid_calls=0,http_attempts=0,
        installation=0,download=0,parameters_changed=False,existing_reference_changed=False,frozen_benchmark_modified=False,
        original_function_accuracy_certified=False,new_unique_benchmark_models_claimed=0,
        plaintext_tolerance=dict(atol=1e-12,rtol=1e-12),frozen_ckks_tolerance=dict(atol=1e-5,rtol=1e-4),
        seconds=time.monotonic()-started,scope="Agent-owned ordered storage/alias source draft and known-local FX fixtures; no production/FHE/Agent acceptance")
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","units","math_passed","reference_groups","output_values","seconds")}))
    return 0
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();raise SystemExit(run(args.output))
