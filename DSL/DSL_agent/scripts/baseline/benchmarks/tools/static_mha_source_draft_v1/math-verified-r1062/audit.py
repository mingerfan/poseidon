"""Known-fixture FX and independent plaintext acceptance for the MHA draft."""
import argparse,hashlib,json,math,os,sys,time,unittest
from pathlib import Path
import numpy as np,torch
from benchmark_graph import digest,require
from benchmark_math import evaluate as core_math
from benchmark_torch import evaluate as core_torch
from platform_config import require_python_packages
from hecate_python_env import ROOT,VENV,WORK
from strict_json_files import dump
from mha_source_reference import evaluate as reference
import fixtures,test_mha_source_plan as tests

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_identity():
    local=Path(__file__).parent
    paths=[local/n for n in ("mha_source_plan.py","mha_source_reference.py","fixtures.py","test_mha_source_plan.py","audit.py")]
    paths += [ROOT/"scripts/baseline"/n for n in ("benchmark_graph.py","model_resources.py","model_decomposition.py","benchmark_math.py","benchmark_torch.py","platform_config.py")]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def samples(signature,seed):
    result=[];rng=np.random.default_rng(seed)
    for index in range(16):
        row={}
        for inp in signature:
            s=inp["shape"];n=math.prod(s)
            if index==0:x=np.zeros(s,dtype=np.float64)
            elif index==1:x=np.linspace(-.15,.15,n,dtype=np.float64).reshape(s)
            elif index==3:x=np.asarray([(-1)**i*.15 for i in range(n)],dtype=np.float64).reshape(s)
            else:x=rng.uniform(-.15,.15,s).astype(np.float64)
            row[inp["name"]]=x
        result.append(row)
    return result
def native_outputs(value,plan):
    result={"attn_output":value[0].detach().numpy()}
    if plan["source_request"]["options"]["need_weights"]:result["attn_weights"]=value[1].detach().numpy()
    else:require(value[1] is None,"Static None output")
    return result
def run(output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require_python_packages(torch,np)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh evidence")
    output.mkdir(mode=0o700);(output/"models").mkdir()
    frozen=source_identity();start=time.monotonic()
    with (output/"units.log").open("x") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    units=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    dump(output/"units-result.json",units)
    require(result.wasSuccessful() and not result.skipped,"Unit/negative/binding checks")
    rows=[]
    for index,case in enumerate(fixtures.CASES):
        require(time.monotonic()-start<180 and source_identity()==frozen,"Time and identity")
        fixture,graph,plan=fixtures.capture(case,index)
        dump(output/"models"/(case["id"]+".json"),plan)
        (output/"models"/(case["id"]+".fx.py")).write_text(graph.code)
        observations=[];errors=dict(original=0.,fixture=0.,decomposition=0.,torch=0.,approximation=0.);values=0
        for probe,inputs in enumerate(samples(plan["mathematical_graph"]["inputs"],14001+index)):
            args=[torch.from_numpy(inputs[n].copy()) for n in ("q",) if n in inputs] if case.get("self_attention") else [torch.from_numpy(inputs[n].copy()) for n in ("q","k","v")]
            native=native_outputs(fixture(*args),plan);traced=native_outputs(graph(*args),plan)
            original=reference(plan["source_request"],inputs,approximate=False)
            expected=reference(plan["source_request"],inputs,approximate=True)
            actual=core_math(plan["lowered_graph"],inputs);tensor=core_torch(plan["lowered_graph"],inputs)
            require(set(native)==set(traced)==set(original)==set(expected)==set(actual)==set(tensor),"Named outputs")
            for name in native:
                outputs=(native[name],traced[name],original[name],expected[name],actual[name],tensor[name])
                require(all(x.shape==native[name].shape and np.isfinite(x).all() for x in outputs),"Shapes and finite values")
                require(np.allclose(native[name],original[name],rtol=1e-12,atol=1e-12),"Independent original MHA reference")
                require(np.allclose(traced[name],native[name],rtol=1e-12,atol=1e-12),"Fixture FX behavior")
                require(np.allclose(actual[name],expected[name],rtol=1e-12,atol=1e-12),"Independent declared approximation")
                require(np.allclose(tensor[name],expected[name],rtol=1e-12,atol=1e-12),"Independent Torch core graph")
                errors["original"]=max(errors["original"],float(np.max(np.abs(native[name]-original[name]))))
                errors["fixture"]=max(errors["fixture"],float(np.max(np.abs(native[name]-traced[name]))))
                errors["decomposition"]=max(errors["decomposition"],float(np.max(np.abs(actual[name]-expected[name]))))
                errors["torch"]=max(errors["torch"],float(np.max(np.abs(tensor[name]-expected[name]))))
                errors["approximation"]=max(errors["approximation"],float(np.max(np.abs(expected[name]-original[name]))))
                values+=int(native[name].size)
            observations.append(dict(probe=probe,inputs={k:v.tolist() for k,v in inputs.items()},
                original={k:v.tolist() for k,v in original.items()},approximation={k:v.tolist() for k,v in expected.items()},
                lowered_numpy={k:v.tolist() for k,v in actual.items()},lowered_torch={k:v.tolist() for k,v in tensor.items()}))
        dump(output/(case["id"]+".references.json"),dict(rows=observations))
        row=dict(id=case["id"],status="math_passed",configuration=case,reference_groups=16,output_values=values,
                 input_count=len(plan["mathematical_graph"]["inputs"]),named_output_count=len(plan["mathematical_graph"]["outputs"]),
                 original_nodes=len(plan["mathematical_graph"]["nodes"]),expanded_nodes=len(plan["lowered_graph"]["nodes"]),
                 output_schema=plan["output_schema"],mask_dispatch=plan["mask_dispatch"],
                 plan_binding=plan["binding"],source_request_sha256=plan["source_request_sha256"],
                 fx_code_sha256=sha(output/"models"/(case["id"]+".fx.py")),maximum_errors=errors,
                 production_intake_integrated=False,original_function_accuracy_certified=False)
        rows.append(row);dump(output/"progress.json",dict(rows=rows))
        print(json.dumps({k:row[k] for k in ("id","status","expanded_nodes","maximum_errors")}),flush=True)
    require(source_identity()==frozen,"Final source identity")
    report=dict(format="poseidon-mha-source-draft-audit-v1",status="draft_math_validated",cases=len(rows),math_passed=len(rows),failed=0,skipped=0,
        units=units,rows=rows,reference_groups=sum(r["reference_groups"] for r in rows),output_values=sum(r["output_values"] for r in rows),
        source_hashes=frozen,runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,hip=torch.version.hip),
        trusted_local_fixture_fx_captures=len(rows),production_source_capture=0,production_intake_integrated=False,
        hecate_tracing=0,compiler_calls=0,encrypted_execution=0,decryption=0,agent_generation=0,paid_calls=0,http_attempts=0,
        installation=0,download=0,parameters_changed=False,existing_reference_changed=False,
        original_function_accuracy_certified=False,frozen_benchmark_modified=False,new_unique_benchmark_models_claimed=0,
        plaintext_tolerance=dict(atol=1e-12,rtol=1e-12),frozen_ckks_tolerance=dict(atol=1e-5,rtol=1e-4),
        seconds=time.monotonic()-start,scope="Agent-owned source graph draft, known local MHA FX fixtures and independent mathematical references; no production/FHE/Agent acceptance")
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","units","math_passed","reference_groups","output_values","seconds")}))
    return 0
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();raise SystemExit(run(args.output))
