"""Manual repair-direction checks, exact-arithmetic diagnostics and adapter lifecycle."""
import argparse,ast,hashlib,json,shlex,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");a=p.parse_args()
    from hecate_python_env import enter_nix,VENV
    if not a.inside:return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=600)
    import numpy as np
    from workspace_paths import RESULTS
    from benchmark_graph import digest
    from benchmark_runner import dump
    from component_contract import VERSION
    from candidate_bundle import export_bundle
    from unified_public_contract import normalize
    from semantic_benchmark_execution import runtime_sources
    root=RESULTS/"validation-adapter-r167";output=root/"followup";output.mkdir()
    frozen=runtime_sources()
    def flat(source,request,inputs):
        period=request["layout"]["input_slot_period"]
        expanded=normalize(source,request);fn=ast.parse(expanded["source"]).body[0]
        def vector(v):
            v=np.asarray(v,dtype=np.float64).reshape(-1)
            assert len(v) in (1,period)
            return np.repeat(v,period) if len(v)==1 else v
        rows=[]
        for batch in inputs:
            values={k:vector(v) for k,v in expanded["constants"].items()}
            values.update({s["dsl_name"]:vector(batch[i]) for i,s in enumerate(request["layout"]["inputs"])})
            values["zero_ct"]=np.zeros(period)
            def expr(n):
                if type(n) is ast.Name:return values[n.id]
                if type(n) is ast.Constant:return vector(n.value)
                if type(n) is ast.UnaryOp:return -expr(n.operand)
                if type(n) is ast.BinOp:
                    x,y=expr(n.left),expr(n.right)
                    if type(n.op) is ast.Add:return x+y
                    if type(n.op) is ast.Sub:return x-y
                    assert type(n.op) is ast.Mult;return x*y
                if type(n) is ast.Call:
                    assert type(n.func) is ast.Attribute and n.func.attr=="rotate"
                    from hecate_contract import rotation_literal
                    return np.roll(expr(n.func.value),-rotation_literal(n.args[0]))
                if type(n) in (ast.List,ast.Tuple):return [expr(v) for v in n.elts]
                raise ValueError("Unexpected flat arithmetic")
            for statement in fn.body:
                if type(statement) is ast.Assign:values[statement.targets[0].id]=expr(statement.value)
                else:
                    assert type(statement) is ast.Return
                    values_out=expr(statement.value)
                    if type(statement.value) not in (ast.List,ast.Tuple):values_out=[values_out]
                    rows.append([values_out[c][i] for c,i in request["layout"]["output_selectors"]])
        return np.asarray(rows)
    rows=[]
    for name in ("construct_083_2","construct_116_0"):
        spec=json.loads((BASE/"cases/validation-adapter-r167"/(name+".json")).read_text())
        original=json.loads((root/"acceptance"/name/"job.json").read_text())
        prior=json.loads((root/"acceptance"/name/"result.json").read_text());old=Path(prior["evidence"])
        assert original["request"]==spec["request"]
        assert hashlib.sha256(original["candidate"]["hecate_source"].encode()).hexdigest()==spec["original_source_sha256"]
        with np.load(old/"arrays.npz",allow_pickle=False) as data:
            # Diagnostic arithmetic uses the already-frozen runner inputs and reference.
            predicted=flat(original["candidate"]["hecate_source"],spec["request"],data["inputs"])
            corrected=flat(spec["candidate"]["hecate_source"],spec["request"],data["inputs"])
            diagnostic=dict(original_arithmetic_max_error=float(np.max(np.abs(predicted-data["reference"]))),
                            corrected_arithmetic_max_error=float(np.max(np.abs(corrected-data["reference"]))))
            if (old/"attempt-00/output/decrypted.npy").is_file():
                diagnostic["original_fhe_vs_arithmetic_max_error"]=float(np.max(np.abs(
                    np.load(old/"attempt-00/output/decrypted.npy",allow_pickle=False)-predicted)))
            assert diagnostic["corrected_arithmetic_max_error"]<1e-12
        folder=output/name;folder.mkdir()
        job=dict(format=VERSION,action="qualify",request=spec["request"],candidate=spec["candidate"]);dump(folder/"job.json",job)
        with (folder/"worker.log").open("x") as log:
            p=subprocess.run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/agent_component.py"),"--job",str(folder/"job.json")],
                             stdout=subprocess.PIPE,stderr=log,text=True,timeout=300)
        result=json.loads(p.stdout);dump(folder/"result.json",result)
        rows.append(dict(name=name,manual_repair=True,new_agent_generation=False,diagnostic=diagnostic,result=result))
        dump(output/"progress.json",dict(rows=rows))
    selected=root/"acceptance/unified-native-copy-2"
    result=json.loads((selected/"result.json").read_text())
    manifest=export_bundle(Path(result["evidence"]),output/"composition-bundle")
    with (output/"bundle-replay.log").open("x") as log:
        p=subprocess.run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/dsl_bundle.py"),"replay",
            "--bundle",str(output/"composition-bundle"),"--execute"],stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert p.returncode==0
    # Cancel only after the extracted validation adapter created an attempt.
    known=set(RESULTS.glob("candidate-replay-*"))
    log=(output/"cancel.log").open("x")
    proc=subprocess.Popen([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/agent_component.py"),
        "--job",str(selected/"job.json")],stdout=subprocess.PIPE,stderr=log,text=True)
    started=time.monotonic();active=None
    while time.monotonic()-started<30 and proc.poll() is None:
        active=next((p for p in set(RESULTS.glob("candidate-replay-*"))-known if (p/"attempt-00").is_dir()),None)
        if active is not None:break
        time.sleep(.01)
    if proc.poll() is None:proc.terminate()
    stdout,_=proc.communicate(timeout=30);log.close()
    assert active is not None
    cancelled=json.loads(stdout);assert proc.returncode==130 and cancelled["status"]=="cancelled"
    cleanup=json.loads((active/"key-cleanup-outcome.json").read_text())
    assert cleanup["complete"] and not (active/"private-keys").exists()
    assert runtime_sources()==frozen
    report=dict(format="poseidon-adapter-followup-r168",rows=rows,composition_bundle_binding=manifest["binding"],
        composition_bundle_replay="passed",active_adapter_cancellation="passed",cancelled_evidence=str(active),
        new_paid_calls=0,new_agent_generation=False,source_hashes=frozen)
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_hashes",)}))
    return any(r["result"]["status"]!="passed" for r in rows)
if __name__=="__main__":raise SystemExit(main())
