"""Bounded manual migration probe; no provider and no Agent success claims."""
import json,tempfile,sys
from pathlib import Path
import numpy as np
from hecate_python_env import ROOT,WORK,VENV,digest
from poseidon_runtime_paths import PROFILE,SPEC
from compiler_configuration import configuration
from run_model_batch import prepare_case
from fx_to_hecate import translate
from candidate_contract import make_request,trace_payload_json,validate_candidate
from poseidon_candidate_sandbox import run
from poseidon_runtime_compile import command
def main():
    result=Path(tempfile.mkdtemp(prefix="poseidon-linear-probe-r232-",dir=WORK/"results"))
    print(result,flush=True)
    descriptor=json.loads((ROOT/"scripts/baseline/cases/linear-example.json").read_text())
    if '--mlp' in sys.argv:
        fixed=json.loads((ROOT/'scripts/baseline/golden_cases/fixtures.json').read_text())
        descriptor=dict(schema=2,id='migration-mlp4x4x2',input_shape=[4],
            constants={k:fixed[k] for k in ('mlp_hidden_weight','mlp_hidden_bias','linear_weight','linear_bias')},
            nodes=[dict(id='hidden',op='linear',inputs=['x'],weight='mlp_hidden_weight',bias='mlp_hidden_bias'),
                   dict(id='squared',op='square',inputs=['hidden']),
                   dict(id='out',op='linear',inputs=['squared'],weight='linear_weight',bias='linear_bias')],output='out')
    model,shape=prepare_case(descriptor,result)
    if '--mlp' in sys.argv:
        from python_compiler_smoke import prepare_reference
        prepare_reference('mlp4x4x2',result)
    (result/'model.json').write_text(json.dumps(descriptor,indent=2))
    translated=translate(model,shape)
    config=configuration("poseidon-cpu-eva-w45-v1")
    request=make_request(translated,descriptor,digest(PROFILE),str(model),compiler_configuration=config)
    candidate=dict(schema=1,request_id=request["request_id"],hecate_source=translated["hecate_source"])
    validate_candidate(candidate,request)
    (result/"request.json").write_text(json.dumps(request,indent=2))
    payload=result/"payload.json";payload.write_text(trace_payload_json(request,candidate))
    output=result/"output";output.mkdir()
    for name,argv in (("trace",[str(VENV/"bin/python"),"/app/candidate_trace.py"]),("compile",command())):
        rc=run(payload,output,argv,result/(name+".log"),150)
        print(name,rc,flush=True)
        if rc:raise RuntimeError((result/(name+".log")).read_text()[-6000:])
    from poseidon_validation import gate,prepare_inputs,execute
    from seal_cpu_golden import compare
    evidence=gate(output,request)
    frozen=prepare_inputs(result,output,request)
    execution=execute(payload,output,request,run,result/"execute.log")
    with np.load(result/"arrays.npz",allow_pickle=False) as data:
        comparison=compare(np.load(output/"decrypted.npy",allow_pickle=False),data["reference"],1e-5,1e-4)
    assert all(digest(output/k)==v for k,v in frozen.items()),"Artifact mutated"
    report=dict(backend="poseidon_cpu",agent_calls=0,gate=evidence,execution=execution,comparison=comparison)
    (result/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(comparison),flush=True)
    assert comparison["passed"],"Numerical failure"

if __name__=="__main__":main()
