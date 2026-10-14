"""Independent plaintext audit of full-size frozen ResNet stem public layout."""
import argparse,hashlib,json,math,os,resource,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import samples,canonical,digest,require,validate
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    from hecate_python_env import WORK,VENV
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned Python")
    require(not a.output.exists() and a.output.parent.resolve().is_relative_to(WORK/"results"),"Fresh output")
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,125))
    import numpy as np
    from unified_capacity_constants import packing,CHANNEL_VERSION
    from unified_request_storage import compact_request,_decode_tiles,_decode_runs
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as pytorch
    from semantic_benchmark_execution import runtime_sources
    path=WORK/"results/poseidon-resnet20-source-r533/stem-model.json"
    raw=path.read_bytes();g=json.loads(raw);checked=validate(g);period=4096
    frozen=runtime_sources();start=time.monotonic();packed=packing(g,dict(input_slot_period=period),CHANNEL_VERSION)
    wire=compact_request(dict(layout=dict(input_slot_period=period),public_constants=packed["vectors"],
        constant_origins=dict(capacity_tensor_packing=packed["descriptor"])))
    for k,v in wire["public_constants"].items():
        decoded=_decode_tiles(v,period) if type(v) is dict and v["format"]=="capacity-public-vector-tiles-v2" else _decode_runs(v["runs"],period) if type(v) is dict else v
        require(canonical(decoded)==canonical(packed["vectors"][k]),"Bit-exact public vector storage")
    vectors={k:np.asarray(v,dtype=np.float64) for k,v in packed["vectors"].items()}
    max_layout=0.;max_references=0.;values=0;input_groups=16
    def routed(node,parts,role,size):
        out=[np.zeros(period) for _ in range(math.ceil(size/period))]
        for b in packed["bindings"]:
            if b["node_id"]!=node["id"] or b["role"]!=role:continue
            term=np.roll(parts[b["input_chunk"]],-b["baby_step"])*vectors[b["key"]]
            out[b["output_chunk"]]+=np.roll(term,-b["giant_step"])
        return out
    a.output.mkdir()
    (a.output/"public-plan.json").write_bytes(canonical(packed["descriptor"]))
    actuals=[];expecteds=[]
    for probe in samples(g,input_groups):
        env={name:[np.pad(probe[name].reshape(-1)[offset:offset+period],(0,period-len(probe[name].reshape(-1)[offset:offset+period])))
                    for offset in range(0,probe[name].size,period)] for name in probe}
        for n in g["nodes"]:
            size=math.prod(checked["shapes"][n["outputs"][0]]);parts=env[n["inputs"][0]]
            mode=packed["tensor_strategies"][n["id"]]
            if mode=="spatial_bsgs":out=routed(n,parts,"tensor_route",size)
            elif mode=="channel_affine":
                out=[]
                for i,part in enumerate(parts):
                    keys={b["role"]:b["key"] for b in packed["bindings"] if b["node_id"]==n["id"] and b.get("chunk")==i}
                    out.append(part*vectors[keys["bn_gain"]]+vectors[keys["bn_offset"]])
            elif mode=="aligned_suffix":
                width=packed["suffix_windows"][n["id"]]["width"];summed=[]
                for part in parts:
                    value=part.copy()
                    for bit in range(width.bit_length()-1):value=value+np.roll(value,-(1<<bit))
                    summed.append(value)
                out=routed(n,summed,"suffix_gather",size)
            else:raise ValueError("Unexpected public strategy")
            env[n["outputs"][0]]=out
        expected=mathematical(g,probe);other=pytorch(g,probe)
        for name in expected:
            np.testing.assert_allclose(expected[name],other[name],rtol=1e-12,atol=1e-12)
            max_references=max(max_references,float(np.max(np.abs(expected[name]-other[name]))))
        result=np.concatenate(env[g["outputs"][0]["value"]])[:size];reference=expected[g["outputs"][0]["name"]].reshape(-1)
        np.testing.assert_allclose(result,reference,rtol=1e-12,atol=1e-12)
        max_layout=max(max_layout,float(np.max(np.abs(result-reference))));values+=reference.size
        actuals.append(result);expecteds.append(reference)
    np.save(a.output/"public-layout.npy",np.asarray(actuals),allow_pickle=False)
    np.save(a.output/"reference.npy",np.asarray(expecteds),allow_pickle=False)
    require(runtime_sources()==frozen,"Source drift")
    report=dict(format="poseidon-full-stem-public-layout-v1",model_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes=frozen,developer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        period=period,configured_runtime=False,packing_version=CHANNEL_VERSION,groups=input_groups,compared_values=values,
        max_reference_error=max_references,max_layout_error=max_layout,passed=True,
        convolution_input_shape=g["inputs"][0]["shape"],convolution_output_shape=checked["shapes"][g["nodes"][0]["outputs"][0]],
        vectors=len(packed["vectors"]),bindings=len(packed["bindings"]),wire_bytes=len(canonical(wire)),
        compiled=False,encrypted_execution=False,new_agent_generation=False,paid_calls=0,seconds=time.monotonic()-start)
    (a.output/"report.json").write_bytes(canonical(report));print(json.dumps({k:v for k,v in report.items() if k!="source_hashes"}))
    return 0
if __name__=="__main__":sys.exit(main())
