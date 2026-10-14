"""Development-only V1/V2 operator-reference audit; no compiler or provider.

Published source graphs remain unchanged. Supplemental mathematical contexts are
separate from the frozen Benchmark and from any encrypted/Agent acceptance.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
BASE=ROOT/"scripts/baseline"
if str(BASE) not in sys.path:
    sys.path.insert(0,str(BASE))

from benchmark_graph import canonical,digest,require,samples
from model_decomposition import FORMAT as V1,REGISTRY,REGISTRY_VERSION
from operator_graph import FORMAT as V2,decompose,evaluate_reference,implementation_sources
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from execution_input_domain import check_approximation_domains
from semantic_benchmark_execution import runtime_sources

FOLDERS=("operator-decomposition-v1","operator-extension-v2","operator-extension-v3",
         "operator-extension-v4","operator-extension-v5","capacity-normalization-regression-v1")
PROFILE="general-tensors-v1"
MAX_CASES=256
FORMAT="poseidon-operator-version-reference-audit-v1"

def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):
            value.update(block)
    return value.hexdigest()

def runtime_identity():
    """Record the packages and native CPU implementations actually loaded."""
    import numpy as np,torch,platform
    import numpy.core._multiarray_umath as numpy_native
    from workspace_paths import require_linux_backend
    require_linux_backend()
    require(sys.version_info[:3]==(3,10,14) and np.__version__=="1.25.2",
            "Pinned reference Python/NumPy versions")
    expected={"aarch64":"2.0.1","x86_64":"2.0.1+cpu"}
    require(platform.machine() in expected and torch.__version__==expected[platform.machine()] and
            torch.version.cuda is None and torch.version.hip is None,"Pinned platform CPU Torch version")
    files=[Path(sys.executable).resolve(),Path(np.__file__).resolve(),
           Path(numpy_native.__file__).resolve(),Path(torch.__file__).resolve(),
           Path(torch._C.__file__).resolve(),Path(torch.__file__).parent/"lib/libtorch_cpu.so"]
    require(all(path.is_file() for path in files),"Loaded reference package files")
    return dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,
                platform=platform.machine(),system=sys.platform,cpu_only=True,
                files={str(path):sha(path) for path in files})

def audit_binding(code,tool_hash,source_files,models,runtime,proposal):
    return digest(dict(code=code,tool=tool_hash,source_files=source_files,
                       model_signatures=[digest(item["model"]) for item in models],
                       profile=PROFILE,reference_tolerance=dict(atol=2e-12,rtol=2e-12),
                       runtime=runtime,frozen_proposal=proposal))

def shape(value):
    if type(value) is list:
        require(value and all(shape(item)==shape(value[0]) for item in value),"Public constant shape")
        return [len(value)]+shape(value[0])
    require(type(value) in (int,float),"Public constant value")
    return []

def structural_signature(model):
    """Ignore identifiers and public weight values, preserve ordered semantics."""
    refs={spec["name"]:["input",i] for i,spec in enumerate(model["inputs"])}
    public=[]
    def ref(name):
        if name not in refs:
            require(name in model["constants"],"Undefined structural reference")
            refs[name]=["public",len(public)]
            public.append(shape(model["constants"][name]))
        return refs[name]
    nodes=[]
    for index,node in enumerate(model["nodes"]):
        operands=[ref(name) for name in node["inputs"]]
        attrs=copy.deepcopy(node["attrs"])
        for key in ("approximation","exp","reciprocal","log"):
            if type(attrs.get(key)) is dict and "coefficients" in attrs[key]:
                attrs[key]["coefficients"]={"shape":shape(attrs[key]["coefficients"])}
        nodes.append(dict(op=node["op"],inputs=operands,attrs=attrs,arity=len(node["outputs"])))
        for ordinal,name in enumerate(node["outputs"]):
            refs[name]=["node",index,ordinal]
    outputs=[ref(item["value"]) for item in model["outputs"]]
    return digest(dict(inputs=[spec["shape"] for spec in model["inputs"]],public=public,
                       nodes=nodes,outputs=outputs))

def node(op,refs,attrs,output="y",identifier="compute"):
    return dict(id=identifier,op=op,inputs=refs,attrs=attrs,outputs=[output])

def fixture(op,context):
    shapes=([2],[2,2],[1,2,2])
    dims=shapes[context]
    value="x"
    model=dict(format=V1,id="operator_version_"+op+"_"+str(context),
               inputs=[dict(name="x",shape=dims)],constants={},nodes=[],
               outputs=[dict(name="result",value="y")])
    attrs={};refs=[value]
    if op=="dot":
        width=context+2
        model["inputs"]=[dict(name="x",shape=[width]),dict(name="rhs",shape=[width])]
        refs=["x","rhs"]
    elif op in ("reciprocal","rsqrt","sqrt","log"):
        model["constants"]["one"]=1.
        model["nodes"]=[node("square",["x"],{},"squared","positive_square"),
                        node("add",["squared","one"],{},"positive","positive_shift")]
        refs=["positive"]
        if op=="log":
            attrs=dict(approximation=dict(method="polynomial",basis="power",
                       domain=[1.,4.],coefficients=[0.,1.,-.125]))
        else:
            attrs=dict(approximation=dict(method="goldschmidt" if op=="reciprocal" else "newton",
                                         domain=[1.,4.],iterations=context+1))
    elif op=="exp":
        attrs=dict(approximation=dict(method="taylor",degree=context+2,domain=[-1.,1.]))
    elif op=="dropout":
        attrs=dict(training=False)
        if context:
            model["nodes"]=[node("square",["x"],{},"squared","square_prefix")]
            refs=["squared"]
    elif op in ("relu","gelu","silu","sigmoid","tanh"):
        attrs=dict(approximation=dict(method="polynomial",basis="power",
                                     domain=[-1.,1.],coefficients=[.125,.5,.125]))
    elif op=="rope":
        width=2*(context+1)
        model["inputs"]=[dict(name="x",shape=[width])]
        attrs=dict(position=context+1,theta=100.+100.*context)
    elif op=="softmax":
        attrs=dict(axis=len(dims)-1,exp=dict(method="taylor",degree=2,domain=[-1.,1.]),
                   reciprocal=dict(method="goldschmidt",iterations=2,domain=[.1,12.]))
    elif op=="rms_norm":
        dims=([2],[2,4],[1,2,4])[context]
        model["inputs"]=[dict(name="x",shape=dims)]
        model["constants"]["gain"]=[1.,-.5,.25,1.25][:dims[-1]]
        refs=["x","gain"]
        attrs=dict(eps=1.,approximation=dict(method="newton",iterations=2,domain=[1.,4.]))
    elif op=="variance":
        attrs=dict(axes=[len(dims)-1],keepdims=True)
    elif op=="global_avg_pool":
        dims=([1,2,3],[2,1,2,3],[1,2,2,2])[context]
        model["inputs"]=[dict(name="x",shape=dims)]
    else:
        raise ValueError("No supplemental fixture for registered operator "+op)
    model["nodes"].append(node(op,refs,attrs))
    return dict(model=model,origin="supplemental_reference_context",source=None,primary_operator=op)

def cases(root=ROOT):
    rows=[];counts={name:set() for name in REGISTRY}
    for folder in FOLDERS:
        directory=root/"scripts/baseline/cases"/folder
        require(directory.is_dir(),"Published source case folder")
        for path in sorted(directory.rglob("*.json")):
            require(not path.is_symlink() and path.stat().st_size<=4*1024**2,"Source JSON budget")
            model=json.loads(path.read_text())
            if type(model) is not dict or model.get("format")!=V1:
                continue
            rows.append(dict(model=model,source=str(path),origin="existing_published_graph"))
            signature=structural_signature(model)
            for op in set(n["op"] for n in model["nodes"])&set(REGISTRY):
                counts[op].add(signature)
    for op in REGISTRY:
        for context in range(3):
            if len(counts[op])>=3:
                break
            item=fixture(op,context)
            signature=structural_signature(item["model"])
            if signature in counts[op]:
                continue
            rows.append(item);counts[op].add(signature)
        require(len(counts[op])>=3,"Reference contexts missing for "+op)
    require(1<=len(rows)<=MAX_CASES,"Version-audit case budget")
    return rows,counts

def preservation(proposal_path):
    if proposal_path is None:
        return None
    proposal=json.loads(Path(proposal_path).read_text())
    require(digest({k:v for k,v in proposal.items() if k!="binding"})==proposal["binding"],
            "Proposal content binding")
    require(runtime_sources()==proposal["source_hashes"],"Frozen runtime sources changed")
    from application_backend import PoseidonBackend
    from trusted_pytorch_import import source_identity
    require(PoseidonBackend().identity()==proposal["backend_identity"],"Frozen backend identity changed")
    require(source_identity()==proposal["capture_identity"] and
            implementation_sources()==proposal["matrix_identity"],"Frozen input identity changed")
    for key in ("case_hashes","tool_hashes"):
        require(all(sha(path)==expected for path,expected in proposal[key].items()),"Frozen "+key)
    return dict(binding=proposal["binding"],source_files=len(proposal["source_hashes"]),
                case_files=len(proposal["case_hashes"]),tools=len(proposal["tool_hashes"]),matched=True)

def check(item,profile=PROFILE,probes=16):
    """Independent reference comparison; no DSL converter computes expectations."""
    import numpy as np
    original=copy.deepcopy(item["model"])
    new=copy.deepcopy(original);new["format"]=V2
    first,binding_first=decompose(original,resource_profile=profile)
    second,binding_second=decompose(new,resource_profile=profile)
    require(original==item["model"] and first["inputs"]==second["inputs"],
            "Original input and graph preservation")
    require([o["name"] for o in first["outputs"]]==[o["name"] for o in second["outputs"]],
            "Named output order")
    maximum=original_function_error=0.
    logical=samples(first,probes)
    for inputs in logical:
        expected=evaluate_reference(original,inputs)
        expected_new=evaluate_reference(new,inputs)
        actual_first=mathematical(first,inputs)
        actual_second=mathematical(second,inputs)
        torch_second=tensor_reference(second,inputs)
        ideal=evaluate_reference(original,inputs,ideal=True)
        require(list(expected)==list(expected_new)==list(actual_first)==list(actual_second)==list(torch_second),
                "Independent named output sets")
        for name in expected:
            for actual in (expected_new[name],actual_first[name],actual_second[name],torch_second[name]):
                require(actual.shape==expected[name].shape and np.isfinite(actual).all(),
                        "Reference output shape/nonfinite")
                np.testing.assert_allclose(actual,expected[name],atol=2e-12,rtol=2e-12)
                maximum=max(maximum,float(np.max(np.abs(actual-expected[name]))))
            original_function_error=max(original_function_error,float(np.max(np.abs(expected[name]-ideal[name]))))
    check_approximation_domains(original,logical)
    check_approximation_domains(new,logical)
    matrix=any(node["op"] in ("dot","matmul") for node in original["nodes"])
    byte_equal=canonical(first)==canonical(second)
    require(matrix or byte_equal,"Non-matrix graph changed under V2")
    return dict(id=original["id"],source=item["source"],origin=item["origin"],
                operators=sorted({n["op"] for n in original["nodes"]}&set(REGISTRY)),
                structural_signature=structural_signature(original),status="passed",
                input_groups=probes,reference_count=5,core_graph_bytes_equal=byte_equal,
                max_absolute_decomposition_error=maximum,
                max_original_function_approximation_error=original_function_error,
                approximation_accuracy_certified=False,
                original_sha256=digest(original),v2_original_sha256=digest(new),
                v1_core_sha256=digest(first),v2_core_sha256=digest(second),
                v1_binding=binding_first["binding_sha256"],v2_binding=binding_second["binding_sha256"],
                compiled=False,encrypted_execution=False,agent_generated=False)

def run(output,proposal_path=None):
    import numpy as np,torch
    output=Path(output)
    require(not output.exists(),"Fresh version-audit output")
    before=preservation(proposal_path)
    data,planned=cases()
    source_files={item["source"]:sha(item["source"]) for item in data if item["source"]}
    code=runtime_sources()
    own=sha(__file__)
    runtime=runtime_identity()
    binding=audit_binding(code,own,source_files,data,runtime,before)
    output.mkdir(mode=0o700);(output/"models").mkdir()
    started=time.monotonic();rows=[]
    covered={op:set() for op in REGISTRY}
    torch.set_num_threads(1)
    for index,item in enumerate(data):
        (output/"models"/(str(index).zfill(3)+".json")).write_bytes(canonical(item["model"]))
        try:
            row=check(item)
        except (ValueError,TypeError,KeyError,IndexError,AssertionError,FloatingPointError,OverflowError) as error:
            row=dict(id=item["model"]["id"],source=item["source"],origin=item["origin"],
                     status="failed",layer="reference_or_decomposition",
                     operators=sorted({n["op"] for n in item["model"]["nodes"]}&set(REGISTRY)),
                     error_type=type(error).__name__,error=str(error)[:700],
                     compiled=False,encrypted_execution=False,agent_generated=False)
        else:
            for op in row["operators"]:covered[op].add(row["structural_signature"])
        rows.append(row)
        (output/"progress.json").write_bytes(canonical(dict(binding=binding,planned=len(data),rows=rows)))
    require(runtime_sources()==code and sha(__file__)==own and
            all(sha(p)==h for p,h in source_files.items()),"Audit source drift")
    require(runtime_identity()==runtime,"Loaded reference implementation drift")
    after=preservation(proposal_path)
    require(after==before,"Frozen proposal changed during audit")
    result=dict(format=FORMAT,binding=binding,
                planned=len(data),passed=sum(r["status"]=="passed" for r in rows),
                failed=sum(r["status"]=="failed" for r in rows),skipped=0,not_run=0,
                rows=rows,registered_operators=len(REGISTRY),registry_version=REGISTRY_VERSION,
                planned_distinct_contexts={op:len(v) for op,v in planned.items()},
                passed_distinct_contexts={op:len(v) for op,v in covered.items()},
                context_target=3,independent_input_groups=sum(r.get("input_groups",0) for r in rows),
                unique_structural_signatures=len({r["structural_signature"] for r in rows if r["status"]=="passed"}),
                max_absolute_decomposition_error=max([r.get("max_absolute_decomposition_error",0.) for r in rows]+[0.]),
                source_files=source_files,source_hashes=code,tool_sha256=own,
                frozen_proposal=after,seconds=time.monotonic()-started,
                runtime=runtime,
                reference_tolerance=dict(atol=2e-12,rtol=2e-12),
                ckks_tolerance=dict(atol=1e-5,rtol=1e-4),
                approximation_accuracy_certified=False,
                existing_frozen_benchmark_changed=False,source_records_rewritten=False,
                compiled=False,hecate_tracing=False,encrypted_execution=False,agent_generated=False,paid_calls=0,
                claim="Reference/decomposition and declared-domain compatibility only; no FHE or Agent coverage")
    result["status"]="passed" if not result["failed"] and all(len(v)>=3 for v in covered.values()) else "failed"
    (output/"report.json").write_bytes(canonical(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","source_files","source_hashes")},ensure_ascii=False))
    return 0 if result["status"]=="passed" else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frozen-proposal",type=Path)
    args=parser.parse_args()
    return run(args.output,args.frozen_proposal)

if __name__=="__main__":
    raise SystemExit(main())
