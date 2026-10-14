"""Pinned mathematical differential checks for the standalone source-operation draft."""
import argparse,hashlib,json,os,sys,time,unittest
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from benchmark_graph import digest,require
from benchmark_math import evaluate as core_math
from benchmark_torch import evaluate as core_torch
from operator_reference import evaluate as high_reference
from source_ops_draft import public_division,functional_max_pool,verify_plan,implementation,source_call,verify_call
from hecate_python_env import ROOT,VENV,WORK
from platform_config import require_python_packages
from strict_json_files import dump
import test_source_ops_draft,test_fx_source_ops_draft
from test_fx_source_ops_draft import Fixture,capture
from fx_source_ops_draft import implementation as fx_implementation
APPROX=dict(method="polynomial",basis="power",domain=[-2.,2.],coefficients=[0.,0.,1.])
CASES=[
dict(id="division_scalar",kind="division",shape=[2,3],divisor=2.,call="torch.div"),
dict(id="division_broadcast",kind="division",shape=[1,2,3],divisor=[[[1.5,-2.,.5]]],call="Tensor.divide"),
dict(id="division_expansion",kind="division",shape=[2,1,2,3],divisor=[[[[2.,-4.,8.]],[[.5,1.25,-2.5]]]],call="operator.truediv"),
dict(id="pool1d_padding_dilation",kind="pool",shape=[1,2,7],rank=1,kernel=3,stride=2,padding=1,dilation=2),
dict(id="pool2d_channels",kind="pool",shape=[1,2,3,4],rank=2,kernel=[2,2],stride=[1,2],padding=0,dilation=1),
dict(id="pool2d_sparse_edges",kind="pool",shape=[1,1,5,5],rank=2,kernel=2,stride=2,padding=1,dilation=[2,1]),
dict(id="pool1d_standard_channels",kind="pool",shape=[2,2,5],rank=1,kernel=2,stride=1,padding=0,dilation=1),
dict(id="pool1d_sparse_edges",kind="pool",shape=[1,1,6],rank=1,kernel=2,stride=2,padding=1,dilation=2),
dict(id="pool2d_asymmetric",kind="pool",shape=[1,1,4,5],rank=2,kernel=[2,3],stride=[2,1],padding=[0,1],dilation=1)
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def independent_source_identity():
    paths=[Path(test_source_ops_draft.__file__),Path(test_fx_source_ops_draft.__file__)]
    paths += [ROOT/"scripts/baseline"/name for name in
              ("benchmark_math.py","benchmark_torch.py","operator_reference.py","platform_config.py")]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def samples(shape,seed):
    n=int(np.prod(shape));rng=np.random.default_rng(seed)
    out=[np.zeros(shape,dtype=np.float64),np.linspace(-.25,.25,n,dtype=np.float64).reshape(shape),
         rng.uniform(-.3,.3,size=shape).astype(np.float64),
         np.asarray([(-1)**i*.3 for i in range(n)],dtype=np.float64).reshape(shape)]
    out.extend(rng.uniform(-.3,.3,size=shape).astype(np.float64) for _ in range(12))
    return out
def run(output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require_python_packages(torch,np)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh result directory")
    output.mkdir(mode=0o700);(output/"models").mkdir()
    frozen=implementation();fx_frozen=fx_implementation();independent_frozen=independent_source_identity()
    audit_hash=sha(Path(__file__));start=time.monotonic()
    with (output/"units.log").open("x") as stream:
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_source_ops_draft,test_fx_source_ops_draft))
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    units=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    dump(output/"units-result.json",units)
    require(result.wasSuccessful() and not result.skipped,"Meaningful unit/negative/identity checks")
    rows=[]
    for index,case in enumerate(CASES):
        require(time.monotonic()-start<180 and implementation()==frozen and fx_implementation()==fx_frozen and independent_source_identity()==independent_frozen,"Draft freeze/time")
        if case["kind"]=="division":
            fixture=Fixture(case["call"],case["divisor"])
            traced,call=capture(fixture,case["shape"])
        else:
            fixture=Fixture(None,pool=(case["rank"],case["kernel"],case["stride"],case["padding"],case["dilation"]))
            traced,call=capture(fixture,case["shape"],APPROX)
        (output/"models"/(case["id"]+".fx.py")).write_text(traced.code)
        verify_call(call,expected_source_call_sha256=call["source_call_sha256"])
        plan=call["plan"]
        dump(output/"models"/(case["id"]+".source-call.json"),call)
        verify_plan(plan,expected_source_request_sha256=plan["source_request_sha256"])
        dump(output/"models"/(case["id"]+".json"),plan)
        outputs=[];max_original=0.;max_decomposed=0.;max_torch=0.;max_approx=0.;values=0
        for probe,x in enumerate(samples(case["shape"],9201+index)):
            typed={"x":x}
            actual=core_math(plan["lowered_graph"],typed)["result"]
            torch_lowered=core_torch(plan["lowered_graph"],typed)["result"]
            if case["kind"]=="division":
                exact=np.divide(x,np.asarray(case["divisor"],dtype=np.float64))
                native=torch.div(torch.from_numpy(x.copy()),torch.from_numpy(np.asarray(case["divisor"],dtype=np.float64))).numpy()
                reference=exact
                ideal=exact
            else:
                operator=F.max_pool1d if case["rank"]==1 else F.max_pool2d
                native=operator(torch.from_numpy(x.copy()),kernel_size=case["kernel"],stride=case["stride"],
                    padding=case["padding"],dilation=case["dilation"],ceil_mode=False,return_indices=False).numpy()
                ideal=high_reference(plan["mathematical_graph"],typed,ideal=True)["result"]
                reference=high_reference(plan["mathematical_graph"],typed,ideal=False)["result"]
            require(native.shape==ideal.shape==reference.shape==actual.shape==torch_lowered.shape,"Independent output shape")
            require(all(np.isfinite(v).all() for v in (native,ideal,reference,actual,torch_lowered)),"Finite outputs")
            fixture_output=fixture(torch.from_numpy(x.copy())).detach().numpy()
            traced_output=traced(torch.from_numpy(x.copy())).detach().numpy()
            require(fixture_output.shape==native.shape==traced_output.shape,"FX fixture output shape")
            require(np.allclose(fixture_output,native,rtol=1e-12,atol=1e-12) and
                    np.allclose(traced_output,native,rtol=1e-12,atol=1e-12),"Original fixture/FX math")
            original=float(np.max(np.abs(native-ideal)))
            decomposed=float(np.max(np.abs(actual-reference)))
            compared=float(np.max(np.abs(torch_lowered-reference)))
            approximate=float(np.max(np.abs(reference-native)))
            require(np.allclose(native,ideal,rtol=1e-12,atol=1e-12),"Original math reference")
            require(np.allclose(actual,reference,rtol=1e-12,atol=1e-12),"Independent decomposition reference")
            require(np.allclose(torch_lowered,reference,rtol=1e-12,atol=1e-12),"Independent Torch core reference")
            max_original=max(max_original,original);max_decomposed=max(max_decomposed,decomposed)
            max_torch=max(max_torch,compared);max_approx=max(max_approx,approximate);values+=int(reference.size)
            outputs.append(dict(probe=probe,inputs=x.tolist(),original=native.tolist(),approximation=reference.tolist(),
                lowered_numpy=actual.tolist(),lowered_torch=torch_lowered.tolist(),
                original_error=original,decomposition_error=decomposed,torch_reference_error=compared,approximation_error=approximate))
        dump(output/(case["id"]+".references.json"),dict(rows=outputs))
        row=dict(id=case["id"],kind=case["kind"],status="math_passed",contexts=1,reference_groups=16,output_values=values,
            plan_binding=plan["binding"],source_request_sha256=plan["source_request_sha256"],
            trusted_fixture_fx_capture=True,fx_code_sha256=sha(output/"models"/(case["id"]+".fx.py")),
            max_original_reference_error=max_original,max_decomposition_error=max_decomposed,
            max_core_torch_error=max_torch,max_approximation_error=max_approx,approximation_accuracy_certified=False,
            expanded_nodes=len(plan["lowered_graph"]["nodes"]),production_intake_integrated=False,
            compiler_calls=0,encrypted_execution=False,agent_generated=False,paid_calls=0)
        rows.append(row);dump(output/"progress.json",dict(rows=rows))
        print(json.dumps({k:row[k] for k in ("id","status","expanded_nodes","max_decomposition_error","max_approximation_error")}),flush=True)
    require(implementation()==frozen and fx_implementation()==fx_frozen and independent_source_identity()==independent_frozen and sha(Path(__file__))==audit_hash,"Draft identities unchanged")
    report=dict(format="poseidon-static-source-op-draft-audit-v2",cases=len(rows),math_passed=len(rows),failed=0,skipped=0,
        units=units,contexts_by_operation=dict(public_division=3,functional_pool=6),
        contexts_by_source_partition=dict(public_division=3,max_pool1d=3,max_pool2d=3),
        reference_groups=sum(r["reference_groups"] for r in rows),independent_source_identity=independent_frozen,
        output_values=sum(r["output_values"] for r in rows),rows=rows,implementation=frozen,fx_implementation=fx_frozen,audit_tool_sha256=audit_hash,
        trusted_fixture_fx_captures=len(rows),new_production_source_capture=0,
        runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,hip=torch.version.hip),
        production_intake_integrated=False,frozen_benchmark_modified=False,new_unique_model_count_claimed=False,
        new_source_capture=0,new_hecate_tracing=0,compiler_calls=0,encrypted_execution=False,decryption=0,agent_generation=0,
        paid_calls=0,http_attempts=0,installation=0,download=0,parameters_changed=False,
        seconds=time.monotonic()-start,scope="draft FX-call mapping from trusted local fixtures, mathematical graphs and independent references; no production input/FHE/Agent acceptance",
        plaintext_tolerance=dict(atol=1e-12,rtol=1e-12),frozen_ckks_tolerance=dict(atol=1e-5,rtol=1e-4))
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","math_passed","units","reference_groups","output_values","seconds")}))
    return 0
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();raise SystemExit(run(a.output))
