"""Explicit offline model import; input Python is parsed, never executed."""
import argparse,hashlib,json,os,shlex,sys
from pathlib import Path
from benchmark_graph import digest, samples
from benchmark_runner import strict_file,dump
from model_decomposition import decompose
from restricted_model_python import parse
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--graph",type=Path);g.add_argument("--python-manifest",type=Path)
    p.add_argument("--output",type=Path);p.add_argument("--write",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    original_path=a.graph or a.python_manifest
    spec=strict_file(original_path,131072)
    files={}
    if a.python_manifest:
        root=original_path.resolve().parent
        for name,h in spec.get("files",{}).items():
            rel=Path(name)
            if rel.is_absolute() or len(rel.parts)!=1 or rel.suffix!=".py" or not rel.stem.isidentifier():
                p.error("Only explicitly listed flat Python modules")
            path=root/rel
            if path.is_symlink() or not path.is_file() or path.stat().st_size>65536:p.error("Python module file gate")
            files[name]=path.read_text()
        original,lowered,binding=parse(spec,files)
    else:
        original=spec;lowered,binding=decompose(original)
    if not a.write:
        print(json.dumps(dict(mode="plan",nodes=len(lowered["nodes"]),model_sha256=digest(lowered),
                              approximation_profiles=binding["approximation_profiles"],paid_calls=0)))
        return 0
    if not a.output or a.output.exists():p.error("New output directory required; preserve existing files")
    from hecate_python_env import enter_nix,VENV
    if not a.inside:
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve().parents[1]/"prepare_model.py"),*sys.argv[1:],"--inside"]
        return enter_nix('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=180)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure Python required")
    import numpy as np
    import torch
    from platform_config import require_python_packages
    require_python_packages(torch,np);torch.set_num_threads(1)
    from operator_reference import evaluate as high
    from benchmark_math import evaluate as core
    from benchmark_torch import evaluate as torch_ref
    execution_error=0.;approximation_error=0.
    for inputs in samples(lowered,16):
        expected=high(original,inputs);actual=core(lowered,inputs);independent=torch_ref(lowered,inputs)
        ideal=high(original,inputs,ideal=True)
        for name in expected:
            np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(independent[name],expected[name],atol=1e-12,rtol=1e-12)
            execution_error=max(execution_error,float(np.max(np.abs(actual[name]-expected[name]))))
            approximation_error=max(approximation_error,float(np.max(np.abs(expected[name]-ideal[name]))))
    # The local package is an import artifact, not an Agent-generated DSL bundle.
    sources={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
             for n in ("model_decomposition.py","restricted_model_python.py","operator_reference.py","model_import_cli.py")}
    report=dict(schema=1,mode="offline_model_import",model_sha256=digest(lowered),
                original_sha256=digest(original),source_hashes=sources,probes=16,
                reference_status="passed",max_decomposition_error=execution_error,
                max_original_function_approximation_error=approximation_error,
                approximation_accuracy_certified=False,encrypted_execution=False,agent_generated=False,paid_calls=0)
    a.output.mkdir(parents=True)
    dump(a.output/"model.json",lowered);dump(a.output/"original.json",original)
    dump(a.output/"decomposition.json",binding);dump(a.output/"reference-report.json",report)
    if files:
        (a.output/"python").mkdir()
        dump(a.output/"python"/"manifest.json",spec)
        for name,source in files.items():(a.output/"python"/name).write_text(source)
    print(json.dumps(report));return 0
