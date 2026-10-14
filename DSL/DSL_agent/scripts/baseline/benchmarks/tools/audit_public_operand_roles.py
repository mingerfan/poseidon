"""Dev-only public/cipher operand-role probe; no production normalization."""
import copy,json,sys,hashlib,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,samples,require
from operator_graph import FORMAT,decompose,evaluate_reference
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from benchmarks.tools.audit_operator_graph_versions import preservation,runtime_identity,structural_signature

def manual_graph(model):
    """Separate mathematical prototype; never edits or replaces the source model."""
    require(type(model) is dict and model["format"]==FORMAT and len(model["nodes"])==1,
            "Manual role probe single-node model")
    result=copy.deepcopy(model);node=result["nodes"][0]
    require(node["op"] in ("add","subtract","multiply") and node["attrs"]=={} and
            len(node["inputs"])==2,"Manual role probe arithmetic")
    public=set(result["constants"]);first,last=node["inputs"]
    require(first not in public or last not in public,"Two public operands")
    if first in public:
        if node["op"] in ("add","multiply"):
            node["inputs"]=[last,first]
        else:
            node["op"]="add";node["inputs"]=["probe_negative",first]
            result["nodes"].insert(0,dict(id="probe_negate",op="negate",inputs=[last],attrs={},
                                          outputs=["probe_negative"]))
    return result

def fixture(op,kind,side):
    scalar=kind=="scalar"
    model=dict(format=FORMAT,id="role_"+op+"_"+kind+"_"+side,
     inputs=[dict(name="x",shape=[2,2])],
     constants=dict(public=.375 if scalar else [[.125,-.25],[.375,-.5]]),
     nodes=[dict(id="arithmetic",op=op,inputs=["public","x"] if side=="left" else ["x","public"],
                 attrs={},outputs=["y"])],outputs=[dict(name="result",value="y")])
    return model

def run(output,proposal):
    import numpy as np
    from semantic_benchmark_execution import runtime_sources
    output=Path(output);require(not output.exists(),"Fresh role audit")
    before=preservation(proposal);source=runtime_sources();runtime=runtime_identity()
    own=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(mode=0o700);(output/"models").mkdir();rows=[];maximum=0.;started=time.monotonic()
    for op in ("add","subtract","multiply"):
        for kind in ("scalar","tensor"):
            for side in ("left","right"):
                model=fixture(op,kind,side);original=copy.deepcopy(model)
                normalized=manual_graph(model)
                require(original==model and original["constants"]==normalized["constants"],"Original graph and weights preserved")
                path=output/"models"/(model["id"]+".json");path.write_bytes(canonical(model))
                (output/"models"/(model["id"]+"-manual.json")).write_bytes(canonical(normalized))
                result=dict(id=model["id"],operator=op,public_kind=kind,public_side=side,
                            original_graph_sha256=digest(model),manual_graph_sha256=digest(normalized),
                            original_preserved=True,mathematical_groups=16,compiled=False,
                            encrypted_execution=False,agent_generated=False,
                            source_kind="declared_graph_and_separate_manual_prototype")
                try:
                    old,_=decompose(model,resource_profile="general-tensors-v1")
                except ValueError as error:
                    result.update(original_admission="rejected",failure_layer="agent_input_decomposition",
                                  reason=str(error))
                    require(side=="left" and "First operand must be encrypted" in str(error),"Expected role rejection")
                    old=None
                else:
                    result["original_admission"]="accepted"
                core,_=decompose(normalized,resource_profile="general-tensors-v1")
                local=0.
                for values in samples(core,16):
                    public=np.asarray(model["constants"]["public"],dtype=np.float64)
                    first,last=(public,values["x"]) if side=="left" else (values["x"],public)
                    independent={"add":np.add,"subtract":np.subtract,"multiply":np.multiply}[op](first,last)
                    high=evaluate_reference(model,values)["result"]
                    actual=mathematical(core,values)["result"]
                    torch=tensor_reference(core,values)["result"]
                    for expected in (high,actual,torch):
                        require(expected.shape==independent.shape and np.isfinite(expected).all(),"Role result shape")
                        np.testing.assert_allclose(expected,independent,atol=2e-12,rtol=2e-12)
                        local=max(local,float(np.max(np.abs(expected-independent))))
                    if old is not None:
                        np.testing.assert_allclose(mathematical(old,values)["result"],independent,atol=2e-12,rtol=2e-12)
                result.update(manual_reference="passed",maximum_reference_error=local,
                              production_adapter_changed=False);maximum=max(maximum,local);rows.append(result)
    # A graph computing solely on public operands stays outside this proposed rule.
    rejects=0
    for op in ("add","subtract","multiply"):
        bad=fixture(op,"scalar","left");bad["constants"]["another_public"]=.5
        bad["nodes"][0]["inputs"]=["public","another_public"]
        try:manual_graph(bad)
        except ValueError:rejects+=1
        else:raise AssertionError("All-public candidate was normalized")
    require(runtime_sources()==source and preservation(proposal)==before and runtime_identity()==runtime,
            "Frozen production identity changed")
    report=dict(format="poseidon-public-operand-role-reference-probe-v1",
     planned=12,original_admitted=sum(r["original_admission"]=="accepted" for r in rows),
     original_rejected=sum(r["original_admission"]=="rejected" for r in rows),
     manual_mathematical_passed=12,manual_reference_failed=0,independent_input_groups=192,
     all_public_rejected=rejects,max_absolute_reference_error=maximum,rows=rows,
     frozen_proposal=before,runtime=runtime,tool_sha256=own,seconds=time.monotonic()-started,
     production_adapter_changed=False,paid_calls=0,hecate_tracing=False,compiled=False,
     encrypted_execution=False,agent_generated=False,
     boundary="Manual mathematical normalization is not original-model admission, FHE acceptance or Agent success")
    report["binding"]=digest(report)
    (output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","runtime")},ensure_ascii=False))
    return 0

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frozen-proposal",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(run(args.output,args.frozen_proposal))
