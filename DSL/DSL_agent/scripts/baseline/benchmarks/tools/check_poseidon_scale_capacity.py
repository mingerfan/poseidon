"""Fixed-parameter RuntimePlan diagnostics; no compiler/Agent acceptance claim."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from workspace_paths import WORK
from benchmark_suite import Builder
from benchmark_graph import canonical,digest,require,samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from component_contract import prepare_task
from candidate_contract import trace_payload_json,validate_candidate
from poseidon_artifact_gate import PRIMES,CONTEXT,sha,inspect,bind_periodic_constants
from poseidon_runtime_paths import SPEC
from poseidon_candidate_sandbox import run
from poseidon_backend_identity import identity
from poseidon_validation import execute
from numerical_results import compare
from check_poseidon_helpers import snapshot

def value(id,level,scale,*,kind="ciphertext",components=2):
    return dict(id=id,kind=kind,level=level,scale_log2=scale,components=components,
                context=CONTEXT,ntt=True,place=dict(kind="host",rank=0))

def plan(level,plain_scale):
    # Both intermediates mathematically equal the same input x; their nominal
    # scales differ. Their ciphertext product implements the frozen square model.
    a=90;b=45+plain_scale;product=a+b
    values=[value("0",12,45),value("1",12,45),value("2",level,45,kind="plaintext",components=1),
            value("3",level,plain_scale,kind="plaintext",components=1),value("4",level,45),
            value("5",level,a),value("6",level,b),value("7",level,product,components=3),
            value("8",level,product)]
    initial=[dict(kind="encode",ordinal=0,output="2",payload=dict(kind="inline",values=[1.])),
             dict(kind="encode",ordinal=1,output="3",payload=dict(kind="inline",values=[1.]))]
    def op(ordinal,kind,inputs,output,attrs=None):
        item=dict(kind="compute",ordinal=ordinal,op=kind,inputs=inputs,output=output,place=dict(kind="host",rank=0))
        if attrs is not None:item["attrs"]=attrs
        return item
    execution=[op(2,"mod_switch",["0"],"4",dict(target_level=level)),
               op(3,"mul_cp",["4","2"],"5"),op(4,"mul_cp",["4","3"],"6"),
               op(5,"mul_cc",["5","6"],"7"),op(6,"relinearize",["7"],"8")]
    current="8";scale=product
    for count in range(level):
        scale-=60;target=level-count-1;output=str(9+count)
        values.append(value(output,target,scale))
        execution.append(op(7+count,"rescale",[current],output,dict(target_level=target,target_scale_log2=scale)))
        current=output
    spec_sha=sha(SPEC.read_bytes())
    return dict(format_version=1,plan_id="1",target=dict(capability_version=1,device_counts=[0],
            operator_spec=dict(id="poseidon-agent-cpu-v1",version=1,source_sha256=spec_sha),
            target_id="poseidon-ckks-cpu",world_size=1),
        values=values,external_inputs=["0","1"],final_outputs=[current],
        initialization=initial,execution=execution,finalization=[])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    require(not a.output.exists() and a.output.resolve().parent==(WORK/"results").resolve(),"Fresh result root")
    a.output.mkdir();start=time.monotonic();sources=snapshot();backend=identity()
    b=Builder([(2,)]);model=b.finish(b.node("square",["input0"]));request=prepare_task(model)
    candidate=dict(schema=1,request_id=request["request_id"],hecate_source='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [x*x]\n')
    validate_candidate(candidate,request)
    groups=samples(model,4);reference=[];inputs=[]
    period=request["layout"]["input_slot_period"]
    for sample in groups:
        x=sample["input0"].reshape(-1);ref=mathematical(model,sample);other=torch_reference(model,sample)
        np.testing.assert_allclose(ref["output0"],other["output0"],atol=1e-12,rtol=1e-12)
        reference.append(ref["output0"].reshape(-1))
        inputs.append([np.pad(x,(0,period-len(x))).tolist(),[0.]*period])
    reference=np.asarray(reference)
    (a.output/"model.json").write_bytes(canonical(model));(a.output/"request.json").write_bytes(canonical(request))
    frozen=dict(format="poseidon-runtime-scale-diagnostic-v1",backend="poseidon_cpu",backend_identity=backend,
                sources=sources,model_sha256=digest(model),request_id=request["request_id"],
                compiler_calls=0,agent_calls=0,paid_calls=0,parameters_changed=False,
                tolerance=dict(atol=1e-5,rtol=1e-4),max_wall_seconds=600,max_artifact_bytes=64*1024**2,
                cases=[dict(id="unequal_scales_fit",level=2,plain_scale=15,product_scale=150,expected="numeric_pass"),
                       dict(id="above_180_fit",level=3,plain_scale=75,product_scale=210,expected="numeric_pass"),
                       dict(id="insufficient_capacity",level=1,plain_scale=15,product_scale=150,expected="gate_reject_and_backend_fail_or_bad_numeric")])
    (a.output/"plan.json").write_bytes(canonical(frozen));rows=[]
    for case in frozen["cases"]:
        require(time.monotonic()-start<600 and snapshot()==sources,"Time/source boundary")
        folder=a.output/case["id"];folder.mkdir();raw=folder/"raw.json";raw.write_bytes(canonical(plan(case["level"],case["plain_scale"])))
        binding=bind_periodic_constants(raw,folder/"unused",folder/"bound",period)
        payload=folder/"payload.json";payload.write_text(trace_payload_json(request,candidate))
        (folder/"inputs.json").write_bytes(canonical(inputs))
        gate=None;gate_error=None
        try:gate=inspect(folder/"bound/plan.json",SPEC,request,bundle=folder/"bound/bundle")
        except ValueError as error:gate_error=str(error)
        record=dict(**case,gate=gate,gate_error=gate_error,layout_binding=binding,
                    modulus_capacity_bits=math.prod(PRIMES[:case["level"]+1]).bit_length())
        if case["expected"]=="numeric_pass":
            require(gate is not None,"Valid fixed-parameter plan rejected")
            metadata=execute(payload,folder,request,run,folder/"execute.log")
            comparison=compare(np.load(folder/"decrypted.npy",allow_pickle=False),reference,1e-5,1e-4)
            require(comparison["passed"],"Valid fixed-parameter execution numerical failure")
            record.update(status="passed",execution=metadata,comparison=comparison,encrypted_execution=True)
        else:
            require(gate_error is not None and "capacity" in gate_error,"Invalid plan must be rejected by normal gate")
            # Explicit negative diagnostic only: invoke the pinned native verifier
            # to cross-check rejection/numerical corruption. This is never an
            # application bypass or a program-package acceptance.
            code=run(payload,folder,["/poseidon-agent-cpu","/out/bound/plan.json","/operator-spec.json",
                     "/out/bound/bundle","/out/inputs.json","/out/execution.json"],folder/"execute.log",150)
            record["native_exit_code"]=code
            record["native_diagnostic"]=(folder/"execute.log").read_text()[-2000:]
            if code==0:
                report=json.loads((folder/"execution.json").read_text());require(report["q"]==list(PRIMES[:13]) and report["p"]==list(PRIMES[13:]),"Native parameter mismatch")
                actual=np.asarray(report["outputs"])[:,0,:2];comparison=compare(actual,reference,1e-5,1e-4)
                require(not comparison["passed"],"Unexpected correct execution outside pinned capacity")
                record.update(native_outcome="returned_incorrect_numeric",comparison=comparison)
            else:record["native_outcome"]="rejected"
            record.update(status="passed_negative_diagnostic",agent_or_compiler_program_accepted=False)
        rows.append(record)
        report=dict(**frozen,rows=rows,passed=len(rows),not_run=len(frozen["cases"])-len(rows),seconds=time.monotonic()-start)
        (a.output/"report.json").write_bytes(canonical(report))
        print(case["id"],record["status"],record.get("native_outcome","encrypted_numeric"),flush=True)
        require(sum(p.stat().st_size for p in a.output.rglob("*") if p.is_file())<=64*1024**2,"Artifact boundary")
    require(identity()==backend and snapshot()==sources,"Final backend/source binding")
    return 0
if __name__=="__main__":raise SystemExit(main())
