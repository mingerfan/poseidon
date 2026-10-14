"""Compare a manually selected generic rule expression with independent references.

Source/slot-mathematics developer check only; no frontend, compiler, Runtime,
Provider, or backend parameter changes. Candidate Python is interpreted as a
closed arithmetic AST and is never eval/exec'd.
"""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from unified_graph_lowering import lower
from unified_graph_contract import validate_candidate
from benchmark_math import evaluate as mathematical_reference
from benchmark_torch import evaluate as torch_reference

def interpret(program,request,inputs):
    module=ast.parse(program)
    assert len(module.body)==1 and isinstance(module.body[0],ast.FunctionDef)
    function=module.body[0];period=request["layout"]["input_slot_period"]
    env={k:np.asarray(v,dtype=np.float64) for k,v in request["public_constants"].items()}
    env["zero_ct"]=np.zeros(period,dtype=np.float64)
    for entry in request["layout"]["inputs"]:
        values=inputs[entry["name"]].reshape(-1)
        vector=np.zeros(period,dtype=np.float64)
        start=entry["offset"];size=entry["elements"]
        vector[:size]=values[start:start+size]
        env[entry["dsl_name"]]=vector
    def read(node):
        if isinstance(node,ast.Name):return env[node.id]
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -read(node.operand)
        if isinstance(node,ast.BinOp):
            left,right=read(node.left),read(node.right)
            if isinstance(node.op,ast.Add):return left+right
            if isinstance(node.op,ast.Sub):return left-right
            if isinstance(node.op,ast.Mult):return left*right
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="rotate":
            assert len(node.args)==1 and not node.keywords
            step=read(node.args[0]);assert type(step) is int
            return np.roll(read(node.func.value),-step)
        if isinstance(node,ast.List):return [read(x) for x in node.elts]
        raise AssertionError("Unsupported arithmetic AST")
    for statement in function.body:
        if isinstance(statement,ast.Assign):
            assert len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name)
            env[statement.targets[0].id]=read(statement.value)
        elif isinstance(statement,ast.Return):
            physical=read(statement.value);decoded={}
            for output in request["layout"]["outputs"]:
                flat=np.zeros(int(np.prod(output["shape"])),dtype=np.float64)
                for block in output["chunks"]:
                    start=block["offset"];size=block["elements"]
                    flat[start:start+size]=physical[block["ciphertext"]][:size]
                decoded[output["name"]]=flat.reshape(output["shape"])
            return decoded
        else:raise AssertionError("Unsupported statement")
    raise AssertionError("Missing return")

def run(request_path):
    request=json.loads(Path(request_path).read_text())
    assert request["compiler_configuration"]["backend"]=="poseidon_cpu"
    assert all(len([i for i in request["layout"]["inputs"] if i["name"]==s["name"]])==1
               for s in request["model"]["inputs"])
    assert all(len(s["chunks"])==1 for s in request["layout"]["outputs"])
    program=lower(request,balanced_power=True)
    checked=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=program),request)
    count=0;maximum=0.
    for case in range(16):
        inputs={}
        rng=np.random.default_rng(2026100700+case)
        for index,spec in enumerate(request["model"]["inputs"]):
            shape=spec["shape"];size=int(np.prod(shape))
            if case==0:array=np.zeros(shape,dtype=np.float64)
            elif case==1:array=(np.where(np.arange(size)%2,1.,-1.)*.125).reshape(shape)
            elif case==2:array=rng.uniform(-.2,.2,shape)
            elif case==3:array=(np.where(np.arange(size)%2,1.,-1.)*.25).reshape(shape)
            else:array=rng.uniform(-.25,.25,shape)
            inputs[spec["name"]]=array
        expected=mathematical_reference(request["model"],inputs)
        independent=torch_reference(request["model"],inputs)
        actual=interpret(program,request,inputs)
        assert set(actual)==set(expected)==set(independent)
        for name,value in expected.items():
            np.testing.assert_allclose(value,independent[name],atol=2e-12,rtol=2e-12)
            np.testing.assert_allclose(value,actual[name],atol=2e-12,rtol=2e-12)
            count+=value.size;maximum=max(maximum,float(np.max(np.abs(value-actual[name]))))
    return dict(format="poseidon-rule-expression-slot-probe-v1",status="passed",origin="manual_selected_rule_variant",
        request_sha256=hashlib.sha256(Path(request_path).read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(program.encode()).hexdigest(),program=program,
        groups=16,compared_values=count,maximum_plaintext_difference=maximum,
        static_candidate_accepted=True,reference_implementations=["benchmark_math","benchmark_torch"],
        python=sys.version.split()[0],numpy=np.__version__,torch=torch.__version__,
        frontend_tracings=0,compiler_calls=0,encrypted_executions=0,provider_calls=0,
        interpretation="finite slot-mathematics evidence only; compiler scale and encrypted correctness unverified")
if __name__=="__main__":
    print(json.dumps(run(sys.argv[1]),sort_keys=True))
