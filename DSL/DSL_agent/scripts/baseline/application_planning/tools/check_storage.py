"""Developer check of the existing Poseidon ciphertext storage API, not FHE."""
import argparse,json,os,shlex,subprocess,sys,time,hashlib,shutil
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from hecate_python_env import WORK,VENV,enter_nix
from sandbox_policy import limits

def run(output):
    from application_backend import PoseidonBackend
    from poseidon_runtime_paths import read_paths
    from poseidon_artifact_gate import PRIMES,CONTEXT
    from application_planning.resources import native_address_limit
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned allocation worker")
    require(not output.exists() and output.resolve().parent==(WORK/"results").resolve(),"Fresh allocation output")
    require(shutil.disk_usage(WORK).free>=4*1024**3,"Allocation disk reserve")
    total=int(subprocess.check_output(["du","-sb",str(WORK/"results")],text=True,timeout=30).split()[0])
    require(total<=32*1024**3,"Allocation global disk budget")
    output.mkdir()
    source=ROOT/"scripts/baseline/application_planning/native/input_storage_probe.cpp"
    paths=read_paths()
    build=Path(paths["poseidon_build"])
    library=build/"libposeidon_shared.so"
    require(library.is_file(),"Existing Poseidon library")
    compiler=Path(shutil.which("g++")).resolve()
    version=subprocess.check_output([str(compiler),"-dumpfullversion"],text=True,timeout=10).strip()
    require(version=="13.2.0","Pinned GCC version")
    identity=PoseidonBackend().identity()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    binding=dict(format="poseidon-input-allocation-probe-v1",source_sha256=sha(source),
        tool_sha256=sha(Path(__file__)),library_sha256=sha(library),compiler_sha256=sha(compiler),
        compiler_version=version,backend_identity=identity,context_id=CONTEXT,primes=list(PRIMES),
        memory_limit=native_address_limit(),compile_seconds=120,execute_seconds=60)
    binding["binding"]=digest(binding)
    (output/"binding.json").write_bytes(canonical(binding))
    executable=output/"input_storage_probe"
    command=[str(compiler),"-O2","-std=c++17","-I"+str(ROOT/"src"),"-I"+str(build),
        str(source),str(library),"-Wl,-rpath,"+str(build),"-o",str(executable)]
    started=time.monotonic()
    with (output/"compile.log").open("x") as stream:
        process=subprocess.run(command,stdout=stream,stderr=stream,timeout=120,preexec_fn=limits)
    report=dict(format="poseidon-input-allocation-report-v1",binding=binding["binding"],
        status="compile_failed",native_probe_compiled=process.returncode==0,dsl_compiled=False,
        encrypted_execution=False,paid_calls=0,seconds=time.monotonic()-started)
    if process.returncode==0:
        with (output/"execute.err").open("x") as err:
            result=subprocess.run([str(executable),*map(str,PRIMES)],text=True,stdout=subprocess.PIPE,
                stderr=err,timeout=60,preexec_fn=limits)
        (output/"execute.out").write_text(result.stdout)
        report.update(status="allocation_failed",returncode=result.returncode)
        if result.returncode==0:
            actual=json.loads(result.stdout)
            require(actual==dict(input_coefficient_bytes=32768*13*2*8,components=2,data_moduli=13,
                poly_degree=32768,coefficient_size=8,separate_allocations=True,encrypted_execution=False),
                "Actual allocation differs from request analysis")
            report.update(status="passed",allocation=actual,executable_sha256=sha(executable))
        loaded=subprocess.check_output(["ldd",str(executable)],text=True,timeout=10)
        (output/"linked-libraries.txt").write_text(loaded)
        require("not found" not in loaded,"Allocation dynamic library unresolved")
        require(str(library) in loaded,"Allocation loaded a different Poseidon library")
    require(identity==PoseidonBackend().identity() and sha(source)==binding["source_sha256"]
        and sha(library)==binding["library_sha256"],"Allocation source or SDK drift")
    report["seconds"]=time.monotonic()-started
    (output/"report.json").write_bytes(canonical(report))
    print(json.dumps(report),flush=True)
    return 0 if report["status"]=="passed" else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=parser.parse_args()
    if not a.inside:
        argv=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--output",str(a.output.resolve())]
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(argv)
        return enter_nix(command,seconds=300,environment="poseidon-cpu")
    return run(a.output.resolve())
if __name__=="__main__":raise SystemExit(main())
