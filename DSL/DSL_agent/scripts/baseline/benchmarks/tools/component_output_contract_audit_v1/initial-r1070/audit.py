"""Pure component/output contract acceptance; no Provider or backend calls."""
import argparse,hashlib,json,math,os,sys,time,unittest
from pathlib import Path
import numpy as np,torch
from benchmark_graph import FORMAT,canonical,digest,require
from benchmark_math import evaluate as graph_math
from model_output_contract import build,verify,capture_tree,restore
from platform_config import require_python_packages
from hecate_python_env import ROOT,VENV,WORK
from strict_json_files import dump
import test_model_output_contract as tests

CASES=(
 dict(id="tensor_square",names=["out"],shape=[2],inputs=["x"]),
 dict(id="tensor_and_none",names=["out"],shape=[2],inputs=["x"]),
 dict(id="dict_nested_cache",names=["out","state"],shape=[2],inputs=["x"]),
 dict(id="scalar_tensor",names=["out"],shape=[2],inputs=["x"]),
 dict(id="multi_input_dict",names=["out","state"],shape=[2],inputs=["x","other"]),
 dict(id="transposed_list",names=["out"],shape=[2,2],inputs=["x"]),
)
def fixture(case):
    model=dict(format=FORMAT,id=case["id"],inputs=[dict(name=n,shape=case["shape"]) for n in case["inputs"]],
        constants={},nodes=[],outputs=[])
    def node(op,refs,attrs=None):
        v="value_"+str(len(model["nodes"]));model["nodes"].append(dict(id="node_"+str(len(model["nodes"])),op=op,inputs=refs,attrs=attrs or {},outputs=[v]));return v
    if case["id"] in ("tensor_square","tensor_and_none","dict_nested_cache"):
        y=node("square",["x"]);model["outputs"].append(dict(name="out",value=y))
        if case["id"]=="dict_nested_cache":
            model["constants"]["offset"]=.125;z=node("add",[y,"offset"]);model["outputs"].append(dict(name="state",value=z))
    elif case["id"]=="scalar_tensor":
        y=node("sum",["x"],dict(axes=[0],keepdims=True));model["outputs"].append(dict(name="out",value=y))
    elif case["id"]=="multi_input_dict":
        y=node("add",["x","other"]);z=node("subtract",["x","other"])
        model["outputs"]=[dict(name="out",value=y),dict(name="state",value=z)]
    else:
        y=node("transpose",["x"],dict(dim0=0,dim1=1));model["outputs"].append(dict(name="out",value=y))
    return model
def native(case,inputs):
    x=torch.from_numpy(inputs["x"].copy())
    if case["id"]=="tensor_square":return x.square()
    if case["id"]=="tensor_and_none":return x.square(),None
    if case["id"]=="dict_nested_cache":return {"logits":x.square(),"cache":([x.square()+.125,None],)}
    if case["id"]=="scalar_tensor":return x.sum(),None
    if case["id"]=="multi_input_dict":
        other=torch.from_numpy(inputs["other"].copy())
        return {"merged":x+other,"parts":[x-other,None]}
    return [None,x.transpose(0,1)]
def independent(case,inputs):
    x=inputs["x"]
    if case["id"] in ("tensor_square","tensor_and_none"):return dict(out=x*x)
    if case["id"]=="dict_nested_cache":return dict(out=x*x,state=x*x+.125)
    if case["id"]=="scalar_tensor":return dict(out=np.asarray([sum(float(v) for v in x)],dtype=np.float64))
    if case["id"]=="multi_input_dict":return dict(out=x+inputs["other"],state=x-inputs["other"])
    return dict(out=np.asarray([[x[i,j] for i in range(x.shape[0])] for j in range(x.shape[1])],dtype=np.float64))
def public_values(value):
    if type(value) is torch.Tensor:return value.item() if value.ndim==0 else value.detach().numpy().tolist()
    if value is None:return None
    if type(value) is tuple:return tuple(public_values(x) for x in value)
    if type(value) is list:return [public_values(x) for x in value]
    if type(value) is dict:return {k:public_values(x) for k,x in value.items()}
    raise ValueError("Known fixture structure")
def source_identity():
    tool=Path(__file__)
    paths=[tool,ROOT/"scripts/baseline/model_output_contract.py",ROOT/"scripts/baseline/test_model_output_contract.py"]
    paths += [ROOT/"scripts/baseline"/n for n in ("benchmark_graph.py","benchmark_math.py","model_resources.py","pytorch_source_contract.py","platform_config.py")]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def samples(case,seed):
    random=np.random.default_rng(seed);rows=[]
    for probe in range(16):
        values={}
        for name in case["inputs"]:
            shape=case["shape"];n=math.prod(shape)
            if probe==0:a=np.zeros(shape,dtype=np.float64)
            elif probe==1:a=np.linspace(-.15,.15,n).reshape(shape)
            elif probe==3:a=np.asarray([(-1)**i*.15 for i in range(n)],dtype=np.float64).reshape(shape)
            else:a=random.uniform(-.15,.15,shape).astype(np.float64)
            values[name]=a
        rows.append(values)
    return rows
def run(output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure Python")
    require_python_packages(torch,np)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh evidence")
    output.mkdir(mode=0o700);frozen=source_identity();started=time.monotonic()
    with (output/"units.log").open("x") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    units=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    dump(output/"units-result.json",units)
    require(result.wasSuccessful() and not result.skipped,"Component tests")
    rows=[]
    for index,case in enumerate(CASES):
        require(time.monotonic()-started<180 and source_identity()==frozen,"Time and identity")
        model=fixture(case);contract=None;errors=0.;values=0;observations=[]
        source_binding=digest(dict(case=case,audit_sha256=frozen[str(Path(__file__).relative_to(ROOT))]))
        for probe,inputs in enumerate(samples(case,16001+index)):
            original=native(case,inputs)
            leaves,tree=capture_tree(original,lambda v:type(v) is torch.Tensor,case["names"],scalar_tensor=lambda v:v.ndim==0)
            candidate=build(model,tree,source_binding=source_binding)
            if contract is None:contract=candidate
            require(candidate==contract,"Static source output contract")
            expected=independent(case,inputs);actual=graph_math(model,inputs)
            for name,tensor in zip(case["names"],leaves):
                raw=tensor.detach().numpy();raw=raw.reshape(1) if raw.ndim==0 else raw
                require(raw.shape==expected[name].shape==actual[name].shape,"ABI output shape")
                require(np.allclose(raw,expected[name],rtol=1e-12,atol=1e-12) and
                        np.allclose(actual[name],expected[name],rtol=1e-12,atol=1e-12),"Independent output reference")
                errors=max(errors,float(np.max(np.abs(raw-expected[name]))),float(np.max(np.abs(actual[name]-expected[name]))))
                values+=int(raw.size)
            restored=restore(contract,actual,expected_binding=contract["binding"],model=model,source_binding=source_binding)
            require(restored==public_values(original),"Caller container/None/scalar structure")
            observations.append(dict(probe=probe,reference={k:v.tolist() for k,v in expected.items()},restored=restored))
        verify(contract,expected_binding=contract["binding"],model=model,source_binding=source_binding)
        dump(output/(case["id"]+".model.json"),model);dump(output/(case["id"]+".output-contract.json"),contract)
        dump(output/(case["id"]+".references.json"),dict(rows=observations))
        row=dict(id=case["id"],status="reference_and_structure_passed",reference_groups=16,output_values=values,
            contract_binding=contract["binding"],maximum_error=errors,model_sha256=digest(model),source_binding=source_binding)
        rows.append(row);print(json.dumps(row),flush=True)
    require(source_identity()==frozen,"Final source identity")
    report=dict(format="poseidon-component-output-contract-audit-v1",status="component_contract_validated",
        units=units,rows=rows,contexts=len(rows),reference_groups=sum(r["reference_groups"] for r in rows),
        output_values=sum(r["output_values"] for r in rows),source_hashes=frozen,
        runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,hip=torch.version.hip),
        source_capture="known local native output fixtures only",production_source_capture=0,
        legacy_default_intake_unchanged=True,new_capture_intake_wired=False,new_package_envelope_default_wired=False,
        execution_packets_in_unit_tests="synthetic protocol fixtures; not encrypted execution",
        hecate_tracing=0,compiler_calls=0,encrypted_execution=0,decryption=0,agent_generation=0,paid_calls=0,http_attempts=0,
        installation=0,download=0,parameters_changed=False,existing_reference_changed=False,
        frozen_benchmark_modified=False,new_unique_benchmark_models_claimed=0,
        plaintext_tolerance=dict(atol=1e-12,rtol=1e-12),frozen_ckks_tolerance=dict(atol=1e-5,rtol=1e-4),seconds=time.monotonic()-started)
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","units","contexts","reference_groups","output_values","seconds")}))
    return 0
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();raise SystemExit(run(args.output))
