"""Small offline Poseidon migration cohort using the existing candidate runner."""
import json,tempfile
from pathlib import Path
from hecate_python_env import ROOT,WORK,digest
from run_candidate import parse_args,inside
CASES=[
 ("native-helper-loop","arithmetic-alias-chain.json","native_loops/helper_repeated.py","--native-public-loops"),
 ("native-array-transpose","arithmetic-alias-chain.json","native_arrays/matrix_transpose.py","--native-arrays"),
 ("native-augmented-subtract","arithmetic-alias-chain.json","native_augmented/subtract.py","--native-scalar-augmented"),
 ("native-plain-left","arithmetic-alias-chain.json","native_augmented/plain_left.py","--native-scalar-augmented"),
]
def main():
    root=Path(tempfile.mkdtemp(prefix="poseidon-regression-r232-",dir=WORK/"results"))
    report=dict(backend="poseidon_cpu",agent_calls=0,cases=[],runner_sha256=digest(Path(__file__)))
    print(root,flush=True)
    for name,model,source,flag in CASES:
        args=parse_args(["--inside","--case",str(ROOT/"scripts/baseline/cases"/model),
            "--golden-file",str(ROOT/"scripts/baseline/golden_cases"/source),flag,"--max-repairs","0"])
        terminal={}
        code=inside(args,report_sink=lambda r,p:terminal.update(report=r,evidence=str(p)))
        report["cases"].append(dict(name=name,exit_code=code,evidence=terminal["evidence"],
            status=terminal["report"]["status"],attempts=terminal["report"]["attempts"]))
        (root/"report.json").write_text(json.dumps(report,indent=2))
    return int(any(row["exit_code"]!=0 for row in report["cases"]))
if __name__=="__main__":raise SystemExit(main())
