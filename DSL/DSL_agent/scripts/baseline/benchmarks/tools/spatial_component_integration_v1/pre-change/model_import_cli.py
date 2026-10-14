"""Offline graph import, static Python parsing or explicitly trusted FX capture."""
import argparse,hashlib,json,os,shlex,sys
from pathlib import Path
from benchmark_graph import digest, samples
from strict_json_files import strict_file,dump
from operator_graph import FORMAT as OPERATOR_V2
from operator_arithmetic_graph import decompose, evaluate_reference, FORMAT as OPERATOR_V3, implementation_sources
from model_resources import PROFILES
from restricted_model_python import parse
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--trusted-pytorch-manifest",type=Path);g.add_argument("--model-package",type=Path);g.add_argument("--graph",type=Path);g.add_argument("--python-manifest",type=Path)
    p.add_argument("--trust-local-code",action="store_true")
    p.add_argument("--resource-profile",choices=tuple(PROFILES),default=None)
    p.add_argument("--output",type=Path);p.add_argument("--write",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.trusted_pytorch_manifest:
        from pytorch_model_intake import prepare,run
        spec,_=prepare(a.trusted_pytorch_manifest)
        if not a.trust_local_code:p.error("Explicit --trust-local-code required")
        if not a.write:
            print(json.dumps(dict(mode="plan",trusted_code=True,files=spec["files"],paid_calls=0)))
            return 0
    if (a.model_package or a.trusted_pytorch_manifest) and not a.inside:
        from hecate_python_env import enter_nix,VENV
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve().parents[1]/"prepare_model.py"),*sys.argv[1:],"--inside"]
        return enter_nix('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=180,environment="poseidon-cpu")
    if a.trusted_pytorch_manifest:
        from hecate_python_env import VENV
        if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure Python required")
        if not a.output:p.error("Fresh output directory required")
        print(json.dumps(run(a.trusted_pytorch_manifest,a.output,
                             resource_profile=a.resource_profile or "legacy-v1",trusted=True)))
        return 0
    original_path=a.graph or a.python_manifest or a.model_package
    from model_resources import resolve,identity
    selected=a.resource_profile or "legacy-v1"
    package_binding=None
    if a.model_package:
        from model_weight_package import load
        spec,selected,package_binding=load(a.model_package)
        if a.resource_profile is not None and a.resource_profile!=selected:p.error("Package/profile mismatch")
    else:spec=strict_file(original_path,resolve(selected).json_bytes)
    if spec.get("format") in (OPERATOR_V2,OPERATOR_V3) and not a.inside:
        from hecate_python_env import enter_nix,VENV
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve().parents[1]/"prepare_model.py"),*sys.argv[1:],"--inside"]
        return enter_nix('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=180,environment="poseidon-cpu")
    files={}
    if a.python_manifest:
        if selected!="legacy-v1":p.error("Restricted Python resource extension pending; use operator graph/package")
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
        original=spec;lowered,binding=decompose(original,resource_profile=selected)
    if package_binding is not None:
        binding["weight_package"]=package_binding
        binding.pop("binding_sha256",None);binding["binding_sha256"]=digest(binding)
    if not a.write:
        print(json.dumps(dict(mode="plan",nodes=len(lowered["nodes"]),model_sha256=digest(lowered),
                              approximation_profiles=binding["approximation_profiles"],resource_profile=identity(selected),runtime_layout_validated=False,paid_calls=0)))
        return 0
    if not a.output or a.output.exists():p.error("New output directory required; preserve existing files")
    from hecate_python_env import enter_nix,VENV
    if not a.inside:
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve().parents[1]/"prepare_model.py"),*sys.argv[1:],"--inside"]
        return enter_nix('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=180,environment="poseidon-cpu")
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure Python required")
    import numpy as np
    import torch
    from platform_config import require_python_packages
    require_python_packages(torch,np);torch.set_num_threads(1)
    high=evaluate_reference
    from benchmark_math import evaluate as core
    from benchmark_torch import evaluate as torch_ref
    execution_error=0.;approximation_error=0.
    for inputs in samples(lowered,16):
        expected=high(original,inputs);actual=core(lowered,inputs,resource_profile=selected);independent=torch_ref(lowered,inputs,resource_profile=selected)
        ideal=high(original,inputs,ideal=True)
        for name in expected:
            np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(independent[name],expected[name],atol=1e-12,rtol=1e-12)
            execution_error=max(execution_error,float(np.max(np.abs(actual[name]-expected[name]))))
            approximation_error=max(approximation_error,float(np.max(np.abs(expected[name]-ideal[name]))))
    # The local package is an import artifact, not an Agent-generated DSL bundle.
    sources={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
             for n in ("model_decomposition.py","model_resources.py","model_weight_package.py","benchmark_graph.py","benchmark_math.py","benchmark_torch.py","restricted_model_python.py","operator_reference.py","model_import_cli.py")}
    if original.get("format") in (OPERATOR_V2,OPERATOR_V3):
        sources.update(implementation_sources(original["format"]))
    report=dict(schema=1,mode="offline_model_import",model_sha256=digest(lowered),
                original_sha256=digest(original),source_hashes=sources,probes=16,
                resource_profile=identity(selected),runtime_layout_validated=False,reference_status="passed",max_decomposition_error=execution_error,
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
