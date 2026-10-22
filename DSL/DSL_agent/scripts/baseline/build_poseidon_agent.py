"""Build the pinned RuntimePlan backend with an existing isolated SDK and Poseidon CPU library."""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path
from hecate_python_env import ROOT,WORK,VENV,enter_nix
from platform_config import configuration
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poseidon-build",type=Path,required=True,help="Existing native build containing libposeidon_shared.so")
    p.add_argument("--apply",action="store_true");p.add_argument("--inside",action="store_true")
    a=p.parse_args();platform=configuration()["id"]
    core=a.poseidon_build.resolve()
    if not core.is_relative_to(WORK.resolve()) or not (core/"libposeidon_shared.so").is_file():
        raise ValueError("Existing Poseidon CPU build must be inside selected platform work root")
    compiler=WORK/"build-dacapo/runtime-plan-v1";executor=WORK/"build-poseidon/agent-runtime-v1"
    manifest=WORK/"deps/poseidon-runtime-v1.json"
    source=ROOT/"third_party/ckks-runtime-agent/third_party/dacapo"
    plan=dict(format="poseidon-runtime-toolchain-v1",platform=platform,compiler_build=str(compiler),
              executor_build=str(executor),poseidon_build=str(core))
    if not a.apply:
        print(json.dumps(dict(toolchain=plan,downloads=0,installs=0,compile_jobs=2,link_jobs=1,
            requires="existing LLVM/MLIR 18.1.2 GCC 13.2.0 GMP 6.3.0 isolated SDK",maximum_seconds=7200,
            maximum_new_bytes=4*1024**3),indent=2));return 0
    if manifest.exists() and json.loads(manifest.read_text())!=plan:
        raise ValueError("Existing toolchain manifest differs; preserve it and explicitly review configuration switch")
    if not a.inside:
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--apply","--poseidon-build",str(core)]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=7200,environment="poseidon-cpu")
    if not os.environ.get("IN_NIX_SHELL"):raise ValueError("Pinned isolated SDK required")
    if subprocess.check_output(["llvm-config","--version"],text=True).strip()!="18.1.2":raise ValueError("LLVM version")
    start=time.monotonic()
    shared=["-GNinja","-DCMAKE_BUILD_TYPE=Release","-DCMAKE_JOB_POOLS=compile=2;link=1",
            "-DCMAKE_JOB_POOL_COMPILE=compile","-DCMAKE_JOB_POOL_LINK=link"]
    commands=[
      ["cmake","-S",str(source),"-B",str(compiler),*shared,"-DCMAKE_C_COMPILER="+os.environ["CC"],
       "-DCMAKE_CXX_COMPILER="+os.environ["CXX"],"-DLLVM_DIR="+os.environ["LLVM_DIR"],
       "-DMLIR_DIR="+os.environ["MLIR_DIR"],"-DHECATE_LINK_ALL_MLIR_LIBS=OFF"],
      ["cmake","--build",str(compiler),"--parallel","2","--target","hecate-opt","HecateFrontend"],
      ["cmake","-S",str(ROOT/"src/poseidon/tools/agent_runtime"),"-B",str(executor),*shared,
       "-DCMAKE_CXX_COMPILER="+os.environ["CXX"],"-DPOSEIDON_CPU_BUILD="+str(core)],
      ["cmake","--build",str(executor),"--parallel","2"],
      ["ctest","--test-dir",str(executor),"--output-on-failure"]]
    initial=sum(p.stat().st_size for d in (compiler,executor) if d.exists() for p in d.rglob("*") if p.is_file())
    for command in commands:
        remaining=7200-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError("Build budget exhausted")
        subprocess.run(command,check=True,timeout=remaining)
        size=sum(p.stat().st_size for d in (compiler,executor) for p in d.rglob("*") if p.is_file())
        if size-initial>4*1024**3:raise RuntimeError("Build disk budget exceeded; artifacts preserved")
    manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(json.dumps(plan,indent=2)+"\n")
    return 0
if __name__=="__main__":raise SystemExit(main())
