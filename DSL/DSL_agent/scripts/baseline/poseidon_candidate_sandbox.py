"""Isolated real Hecate/RuntimePlan/Poseidon commands. No SEAL library mounts."""
import os
import subprocess
from pathlib import Path
from poseidon_sandbox_policy import MODULES, limits
from hecate_python_env import ROOT, VENV, WORK
from workspace_paths import WORK_BASE
from platform_config import configuration
from poseidon_runtime_paths import SOURCE,DACAPO,BUILD,CPU_BUILD,POSEIDON_BUILD,PROFILE,SPEC

def command(payload, output, argv, keys=None, *, helpers=False):
    if keys is not None:
        raise ValueError("Poseidon execution owns ephemeral keys; legacy key directories are unsupported")
    for path, kind in ((payload,"payload"),(output,"output")):
        if not path.resolve().is_relative_to(WORK/"results") or not path.exists():
            raise ValueError("Invalid sandbox "+kind)
    library=os.environ["HECATE_PYTHON_LIBRARY_PATH"]
    if not all(Path(p).is_relative_to("/nix/store") for p in library.split(":")):
        raise ValueError("Non-Nix library search path")
    cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL",
         "--clearenv","--ro-bind","/nix/store","/nix/store","--ro-bind",str(VENV),str(VENV),
         "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app",
         "--ro-bind",str(payload),"/payload.json","--bind",str(output),"/out",
         "--ro-bind",str(DACAPO/"python/hecate"),"/frontend",
         "--ro-bind",str(BUILD/"lib"),"/hecate/build/lib",
         "--ro-bind",str(BUILD/"lib"),str(BUILD/"lib"),
         "--ro-bind",str(BUILD/"bin/hecate-opt"),"/hecate-opt",
         "--ro-bind",str(PROFILE),"/profile.json","--ro-bind",str(SPEC),"/operator-spec.json",
         "--ro-bind",str(CPU_BUILD/"poseidon_agent_cpu"),"/poseidon-agent-cpu",
         "--ro-bind",str(POSEIDON_BUILD/"libposeidon_shared.so"),str(POSEIDON_BUILD/"libposeidon_shared.so")]
    for name in MODULES:
        cmd+=["--ro-bind",str(ROOT/"scripts/baseline"/name),"/app/"+name]
    if helpers:
        from poly_dependencies import verify,TARGET
        verify()
        cmd+=["--ro-bind",str(TARGET),"/poly-deps","--ro-bind",str(DACAPO/"python/poly"),"/upstream-poly",
              "--ro-bind",str(ROOT/"scripts/baseline/poseidon-upstream-helpers-v1.json"),"/app/poseidon-upstream-helpers-v1.json",
              "--dir","/app/upstream_adapters"]
        for name in ("__init__.py","batch_norm.py","batch_norm_node.py","concat_node.py","periodic_ring.py","spatial_node.py","spatial_mapped.py","fused_conv_bn.py","downsample_node.py","virtual_ring.py","virtual_node.py","chunk_virtual_node.py","fixed_polynomial.py"):
            cmd+=["--ro-bind",str(ROOT/"scripts/baseline/upstream_adapters"/name),"/app/upstream_adapters/"+name]
    variables=dict(PATH="/nonexistent",LD_LIBRARY_PATH=library,HECATE="/hecate",HECATE_BUILD="/hecate/build",
        POSEIDON_WORK_ROOT=str(WORK_BASE),POSEIDON_PLATFORM=configuration()["id"],HOME="/nonexistent",
        PYTHONPATH="/app:/frontend",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",LANG="C.UTF-8")
    for name,value in variables.items():cmd+=["--setenv",name,value]
    return cmd+["--chdir","/out","--",*argv]

def run(payload,output,argv,log,seconds=60,keys=None,*,helpers=False):
    cmd=command(payload,output,argv,keys,helpers=helpers)
    with log.open("w") as stream:
        result=subprocess.run(["/usr/bin/timeout","-k","3s",str(seconds),*cmd],
            env={"PATH":"/usr/bin:/bin","POSEIDON_PROBE_SECRET":"must-not-survive"},
            stdout=stream,stderr=subprocess.STDOUT,timeout=seconds+10,
            close_fds=True,preexec_fn=limits)
    return result.returncode
