"""Independent plaintext acceptance for the isolated GELU source-variant draft."""
import argparse,hashlib,json,math,os,sys,time,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from benchmark_graph import digest,require
from benchmark_math import evaluate as core_math
from benchmark_torch import evaluate as core_torch
from platform_config import require_python_packages
from hecate_python_env import ROOT,VENV,WORK
from strict_json_files import dump
from gelu_source_plan import verify_plan,verify_call
from gelu_source_reference import original_scalar,check_domain
import test_gelu_source_plan as tests

CASES=[
    dict(id="none_functional_vector",variant="none",source="functional",shape=[4],basis="power"),
    dict(id="none_module_matrix",variant="none",source="module",shape=[2,3],basis="power"),
    dict(id="none_keyword_rank4",variant="none",source="input_keyword",shape=[1,2,2,3],basis="chebyshev"),
    dict(id="tanh_functional_vector",variant="tanh",source="functional",shape=[4],basis="power"),
    dict(id="tanh_module_matrix",variant="tanh",source="module",shape=[2,3],basis="power"),
    dict(id="tanh_keyword_rank4",variant="tanh",source="input_keyword",shape=[1,2,2,3],basis="chebyshev"),
]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def source_identity():
    local=Path(__file__).parent
    files=[local/name for name in ("gelu_source_plan.py","gelu_source_reference.py","test_gelu_source_plan.py","audit.py")]
    files += [ROOT/"scripts/baseline"/name for name in
              ("benchmark_graph.py","model_decomposition.py","model_resources.py",
               "benchmark_math.py","benchmark_torch.py","platform_config.py")]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def samples(shape,seed):
    rng=np.random.default_rng(seed);n=math.prod(shape)
    values=[np.zeros(shape,dtype=np.float64),np.linspace(-.8,.8,n,dtype=np.float64).reshape(shape),
            rng.uniform(-.8,.8,shape).astype(np.float64),
            np.array([(-1)**i for i in range(n)],dtype=np.float64).reshape(shape)]
    values += [rng.uniform(-.8,.8,shape).astype(np.float64) for _ in range(12)]
    return values

def scalar_polynomial(x,profile):
    coefficients=profile["coefficients"]
    if profile["basis"]=="power":
        # Direct monomial summation is independent of the lowering evaluator.
        return math.fsum(c*x**i for i,c in enumerate(coefficients))
    terms=[1.,x]
    for i in range(2,len(coefficients)):
        terms.append(2*x*terms[-1]-terms[-2])
    return math.fsum(c*t for c,t in zip(coefficients,terms))

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
    require(result.wasSuccessful() and not result.skipped,"Unit/negative/signature/identity checks")
    rows=[]
    for index,case in enumerate(CASES):
        require(time.monotonic()-start<180 and source_identity()==frozen,"Time and source binding")
        profile=tests.POWER if case["basis"]=="power" else tests.CHEB
        fixture,graph,call=tests.capture(case["variant"],case["source"],case["shape"],profile)
        verify_call(call,expected_source_call_sha256=call["source_call_sha256"])
        plan=call["plan"]
        verify_plan(plan,expected_source_request_sha256=plan["source_request_sha256"])
        dump(output/"models"/(case["id"]+".json"),call)
        (output/"models"/(case["id"]+".fx.py")).write_text(graph.code)
        observations=[];max_original=0.;max_fixture=0.;max_decomposition=0.;max_torch=0.;max_approximation=0.;max_variant=0.;values=0
        for probe,x in enumerate(samples(case["shape"],10301+index)):
            check_domain(x.reshape(-1).tolist(),profile)
            raw=F.gelu(torch.from_numpy(x.copy()),approximate=case["variant"]).numpy()
            fixture_value=fixture(torch.from_numpy(x.copy())).detach().numpy()
            traced_value=graph(torch.from_numpy(x.copy())).detach().numpy()
            original=np.array([original_scalar(float(v),case["variant"]) for v in x.reshape(-1)],dtype=np.float64).reshape(x.shape)
            requested=np.array([scalar_polynomial(float(v),profile) for v in x.reshape(-1)],dtype=np.float64).reshape(x.shape)
            normalized=core_math(plan["lowered_graph"],{"x":x})["result"]
            torch_graph=core_torch(plan["lowered_graph"],{"x":x})["result"]
            other_variant="none" if case["variant"]=="tanh" else "tanh"
            other=np.array([original_scalar(float(v),other_variant) for v in x.reshape(-1)],dtype=np.float64).reshape(x.shape)
            require(all(v.shape==x.shape and np.isfinite(v).all() for v in (raw,fixture_value,traced_value,original,requested,normalized,torch_graph)),"Reference shapes and finite values")
            require(np.allclose(raw,original,atol=1e-12,rtol=1e-12),"Pinned source function reference")
            require(np.allclose(fixture_value,raw,atol=1e-12,rtol=1e-12) and np.allclose(traced_value,raw,atol=1e-12,rtol=1e-12),"Source fixture/FX behavior")
            require(np.allclose(normalized,requested,atol=1e-12,rtol=1e-12),"Independent requested polynomial")
            require(np.allclose(torch_graph,requested,atol=1e-12,rtol=1e-12),"Independent Torch core graph")
            original_error=float(np.max(np.abs(raw-original)))
            fixture_error=max(float(np.max(np.abs(fixture_value-raw))),float(np.max(np.abs(traced_value-raw))))
            decomposition_error=float(np.max(np.abs(normalized-requested)))
            torch_error=float(np.max(np.abs(torch_graph-requested)))
            approximation_error=float(np.max(np.abs(requested-original)))
            variant_difference=float(np.max(np.abs(other-original)))
            max_original=max(max_original,original_error);max_fixture=max(max_fixture,fixture_error)
            max_decomposition=max(max_decomposition,decomposition_error);max_torch=max(max_torch,torch_error)
            max_approximation=max(max_approximation,approximation_error);max_variant=max(max_variant,variant_difference)
            values+=int(x.size)
            observations.append(dict(probe=probe,inputs=x.tolist(),source_original=original.tolist(),requested_polynomial=requested.tolist(),
                                     lowered_numpy=normalized.tolist(),lowered_torch=torch_graph.tolist(),
                                     original_error=original_error,decomposition_error=decomposition_error,
                                     approximation_error=approximation_error,variant_difference=variant_difference))
        dump(output/(case["id"]+".references.json"),dict(rows=observations))
        row=dict(id=case["id"],variant=case["variant"],source=case["source"],shape=case["shape"],basis=case["basis"],
                 status="math_passed",reference_groups=16,output_values=values,source_identity=plan["source_function"],
                 source_call_sha256=call["source_call_sha256"],plan_binding=plan["binding"],
                 fx_code_sha256=sha(output/"models"/(case["id"]+".fx.py")),
                 max_original_reference_error=max_original,max_fixture_fx_error=max_fixture,
                 max_decomposition_error=max_decomposition,max_core_torch_error=max_torch,
                 max_approximation_error=max_approximation,max_source_variant_difference=max_variant,
                 original_function_accuracy_certified=False,production_intake_integrated=False)
        rows.append(row);dump(output/"progress.json",dict(rows=rows))
        print(json.dumps({k:row[k] for k in ("id","status","max_decomposition_error","max_approximation_error","max_source_variant_difference")}),flush=True)
    require(source_identity()==frozen,"Final source identities")
    report=dict(format="poseidon-gelu-source-draft-audit-v1",status="draft_math_validated",cases=len(rows),
                math_passed=len(rows),failed=0,skipped=0,units=units,contexts_by_variant=dict(none=3,tanh=3),
                reference_groups=sum(r["reference_groups"] for r in rows),output_values=sum(r["output_values"] for r in rows),
                rows=rows,source_hashes=frozen,runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,hip=torch.version.hip),
                trusted_fixture_fx_captures=len(rows),production_source_capture=0,production_intake_integrated=False,
                source_variants_preserved=True,implicit_source_variant_substitution=False,
                original_function_accuracy_certified=False,parameters_changed=False,existing_reference_changed=False,new_original_variant_reference_added=True,
                new_hecate_tracing=0,compiler_calls=0,encrypted_execution=0,decryption=0,agent_generation=0,
                paid_calls=0,http_attempts=0,installation=0,download=0,frozen_benchmark_modified=False,
                new_unique_benchmark_models_claimed=0,seconds=time.monotonic()-start,
                plaintext_tolerance=dict(atol=1e-12,rtol=1e-12),frozen_ckks_tolerance=dict(atol=1e-5,rtol=1e-4),
                scope="isolated Agent draft, known local FX fixtures and independent plaintext references; no production import/FHE/Agent success")
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","units","math_passed","reference_groups","output_values","seconds")}))
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();raise SystemExit(run(args.output))
