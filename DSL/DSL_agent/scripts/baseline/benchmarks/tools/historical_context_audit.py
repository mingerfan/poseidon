"""Verify a previously audited context without crediting it to current runtime.

No historical candidate is executed or recompiled. All originally audited files
must remain identical; the reference implementations themselves must still match
the old binding before their independent numerical recomputation is permitted.
"""
from pathlib import Path
import numpy as np
from audit_unified_candidate import read,sha,verify_files
from benchmark_graph import samples,digest,require
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from seal_cpu_golden import compare
BASE=Path(__file__).resolve().parents[2]
def verify_saved(folder,saved,sources):
    for name in ("benchmark_graph.py","benchmark_math.py","benchmark_torch.py","seal_cpu_golden.py"):
        require(sha(BASE/name)==sources["scripts/baseline/"+name],"Historical reference source changed")
    verify_files(folder,saved["files"])
    report=read(folder/"report.json")
    require(report["status"]=="passed" and report["agent_calls"]==0 and
            not report["llm_generation_validated"],"Not manual historical pass")
    verify_files(folder,report["frozen_hashes"])
    successes=[a for a in report["attempts"] if a.get("status")=="passed"]
    require(len(successes)==1,"Ambiguous historical attempt")
    attempt=successes[0];out=folder/("attempt-%02d"%attempt["index"])/"output"
    verify_files(out,attempt["artifact_hashes"])
    require(all(attempt[k] for k in ("compiled","executed","numerically_correct")) and
            attempt["trace"]["frontend"]=="real_Hecate" and attempt["execution"]["encrypted_execution"],
            "Historical execution evidence incomplete")
    require(read(out/"execution.json")==attempt["execution"],"Historical execution binding")
    require(read(folder/"key-cleanup-outcome.json")["complete"],"Cleanup incomplete")
    model=read(folder/"model.json");request=read(folder/"request.json")
    require(model==request["model"] and digest(model)==saved["model_sha256"],"Historical model binding")
    refs=[]
    for probe in samples(model,4):
        expected=mathematical(model,probe);second=torch_reference(model,probe)
        for name in expected:np.testing.assert_allclose(expected[name],second[name],atol=1e-12,rtol=1e-12)
        refs.append(np.concatenate([expected[o["name"]].reshape(-1) for o in model["outputs"]]))
    with np.load(folder/"arrays.npz",allow_pickle=False) as arrays:
        require(np.array_equal(np.stack(refs),arrays["reference"]),"Historical reference mismatch")
        comparison=compare(np.load(out/"decrypted.npy",allow_pickle=False),arrays["reference"],1e-5,1e-4)
    require(comparison==saved["comparison"]==attempt["comparison"] and comparison["passed"],
            "Historical numerical mismatch")
    return saved
