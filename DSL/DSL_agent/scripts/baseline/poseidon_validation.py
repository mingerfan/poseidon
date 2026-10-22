"""Poseidon validation operations shared by the existing adapter and migration checks."""
import json
import numpy as np
from pathlib import Path
from poseidon_runtime_paths import SPEC
from poseidon_artifact_gate import inspect,bind_periodic_constants,require,PRIMES
from cipher_abi import has_zero_argument,logical_input_names
from hecate_python_env import digest

def wide_layout(layout):
    from unified_capacity_layout import WIDE_ABI,validate_layout
    if layout.get("execution_abi")!=WIDE_ABI:return None
    validate_layout(layout)
    return layout

def execution_contract(layout):
    if wide_layout(layout) is None:return None
    return dict(format="poseidon-agent-inputs-v2",execution_abi=layout["execution_abi"],
        input_slot_period=layout["input_slot_period"],decode_slots=layout["input_slot_period"],
        input_ciphertexts=len(logical_input_names(layout))+int(has_zero_argument(layout)),
        output_ciphertexts=layout["output_ciphertexts"])

def input_payload(inputs,layout):
    contract=execution_contract(layout)
    return dict(contract,samples=inputs.tolist()) if contract is not None else inputs.tolist()

def gate(output,request):
    plan=output/"lowered._hecate_golden.runtime-plan.json"
    source_bundle=output/"lowered._hecate_golden.bundle"
    raw=inspect(plan,SPEC,request,bundle=source_bundle)
    binding=bind_periodic_constants(plan,source_bundle,output/"bound",request["layout"]["input_slot_period"],
                                    capacity_layout=wide_layout(request["layout"]))
    bound=inspect(output/"bound/plan.json",SPEC,request,bundle=output/"bound/bundle")
    return dict(raw=raw,binding=binding,bound=bound)

def prepare_inputs(result,output,request):
    layout=request["layout"];period=layout["input_slot_period"]
    with np.load(result/"arrays.npz",allow_pickle=False) as data:
        inputs=data["inputs"].copy()
    if inputs.ndim==2:inputs=inputs[:,None,:]
    require(inputs.ndim==3 and inputs.shape[1:]==(len(logical_input_names(layout)),period),"Physical input binding")
    if has_zero_argument(layout):
        inputs=np.concatenate([inputs,np.zeros((len(inputs),1,period))],axis=1)
    (output/"inputs.json").write_text(json.dumps(input_payload(inputs,layout),allow_nan=False))
    return {str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}

def execute(payload,output,request,native_run,log):
    from application_planning.compiled_resources import execution_storage_precheck, PRECHECK_FILE, CompiledResourceIntegrityRejected
    resource_precheck=execution_storage_precheck(output,request)
    from poseidon_artifact_gate import read as artifact_read, sha as artifact_sha
    precheck_sha256=artifact_sha(artifact_read(output/PRECHECK_FILE))
    code=native_run(payload,output,["/poseidon-agent-cpu","/out/bound/plan.json","/operator-spec.json",
        "/out/bound/bundle","/out/inputs.json","/out/execution.json"],log,150)
    try:
        if artifact_sha(artifact_read(output/PRECHECK_FILE))!=precheck_sha256:
            raise CompiledResourceIntegrityRejected()
    except (ValueError, OSError) as error:
        raise CompiledResourceIntegrityRejected() from error
    require(code==0,f"Isolated Poseidon execution failed (exit {code}); see execute.log")
    report=json.loads((output/"execution.json").read_text())
    require(report["backend"]=="poseidon_cpu" and report["encrypted_execution"] is True and report["bootstrap"] is False,"Execution identity")
    require(report["q"]==list(PRIMES[:13]) and report["p"]==list(PRIMES[13:]) and report["security"]=="tc128" and report["executor_secret_key"] is False,"Actual runtime parameters/key boundary")
    contract=execution_contract(request["layout"])
    decode=256 if contract is None else contract["decode_slots"]
    if contract is not None:
        expected=dict(format="poseidon-agent-execution-v2",execution_abi=contract["execution_abi"],
            input_slot_period=contract["input_slot_period"],decoded_slots=decode,
            input_ciphertexts=contract["input_ciphertexts"],output_ciphertexts=contract["output_ciphertexts"])
        require(all(type(report.get(k)) is type(v) and report.get(k)==v for k,v in expected.items()),"Wide execution protocol identity")
    else:
        require("format" not in report and "execution_abi" not in report,"Unexpected execution protocol")
    slots=np.asarray(report.pop("outputs"),dtype=np.float64)
    require(slots.ndim==3 and slots.shape[1:]==(request["layout"]["output_ciphertexts"],decode) and np.isfinite(slots).all(),"Decoded ciphertext shape")
    selected=np.asarray([[sample[c,s] for c,s in request["layout"]["output_selectors"]] for sample in slots])
    np.save(output/"decrypted.npy",selected,allow_pickle=False)
    report["resource_precheck"]=resource_precheck
    (output/"execution-metadata.json").write_text(json.dumps(report,indent=2))
    if "execution_abi" in request["layout"] and len(request["layout"]["output_shape"])>1:
        from packed_input_abi import decode_output
        np.save(output/"decrypted-logical.npy",decode_output(selected,request["layout"]),allow_pickle=False)
    return report

def audit_gate(output,request,record):
    import tempfile
    raw=inspect(output/"lowered._hecate_golden.runtime-plan.json",SPEC,request,bundle=output/"lowered._hecate_golden.bundle")
    bound=inspect(output/"bound/plan.json",SPEC,request,bundle=output/"bound/bundle")
    require(raw==record["raw"] and bound==record["bound"],"Artifact gate receipt differs")
    with tempfile.TemporaryDirectory() as tmp:
        regenerated=Path(tmp)/"bound"
        binding=bind_periodic_constants(output/"lowered._hecate_golden.runtime-plan.json",
            output/"lowered._hecate_golden.bundle",regenerated,request["layout"]["input_slot_period"],
            capacity_layout=wide_layout(request["layout"]))
        require(binding==record["binding"],"Layout binding differs")
        expected={str(p.relative_to(regenerated)):digest(p) for p in regenerated.rglob("*") if p.is_file()}
        actual={str(p.relative_to(output/"bound")):digest(p) for p in (output/"bound").rglob("*") if p.is_file()}
        require(expected==actual,"Periodic layout artifacts differ")
