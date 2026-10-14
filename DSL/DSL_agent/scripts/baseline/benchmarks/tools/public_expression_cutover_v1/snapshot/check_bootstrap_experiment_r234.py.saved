"""Run the separately approved bootstrap experiment; no Agent configuration changes."""
import hashlib,json,math,os,resource,shutil,subprocess,time
from pathlib import Path
ROOT=Path("/home/lhohy/Code/Poseidon")
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
OUT=WORK/"results/poseidon-bootstrap-experiment-r234"
BUILD=WORK/"build-poseidon/bootstrap-experiment-r234"
SRC=ROOT/"src/poseidon/tests/bootstrapping/agent_bootstrap_experiment"
LIBBUILD=WORK/"build-poseidon/bootstrap-cpu-r217/poseidon_build"
LIB=LIBBUILD/"libposeidon_shared.so"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
auth=json.loads((OUT/"authorization.json").read_text())
proposal=ROOT/"docs/baseline/poseidon-native-bootstrap-proposal-r233.json"
assert auth["approved"] and sha(proposal)==auth["proposal_file_sha256"]
cfg=auth["approved_scope"]; limits=cfg["limits"]
assert math.prod(cfg["q"]+cfg["p"]).bit_length()==876<=881
assert len(set(cfg["q"]+cfg["p"]))==18
assert all((v-1)%65536==0 for v in cfg["q"]+cfg["p"])
assert sha(LIB)=="dcfb32471bbadf12e66835fc9b8dac4c06106582cf884862f3ac8a9612e154e4"
assert not BUILD.exists() and not (OUT/"report.json").exists()
assert int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024>7*1024**3
start=time.monotonic(); deadline=start+limits["max_wall_seconds"]
shutil.copytree(SRC,OUT/"source-snapshot")
shutil.copy2(__file__,OUT/"runner.py")
report={"format":"poseidon-bootstrap-experiment-result-v1","status":"running",
 "authorization_sha256":sha(OUT/"authorization.json"),"proposal_sha256":auth["proposal_sha256"],
 "parameters":cfg,"library_sha256":sha(LIB),"source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in SRC.iterdir() if p.is_file()},
 "steps":[],"paid_calls":0,"download_bytes":0,"runtime_integration":False,"agent_generated":False}
def size():
 return sum(p.stat().st_size for base in (BUILD,OUT) if base.exists() for p in base.rglob("*") if p.is_file())
def save():
 report["elapsed_seconds"]=time.monotonic()-start;report["disk_bytes"]=size()
 (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
def bounded(cmd,name,preexec=None,env=None):
 if size()>limits["max_disk_bytes"]: raise RuntimeError("disk budget")
 remaining=int(deadline-time.monotonic())-5
 if remaining<=0: raise RuntimeError("wall budget")
 stamp=time.monotonic()
 with (OUT/(name+".log")).open("w") as log:
  result=subprocess.run(["/usr/bin/timeout","-k","3s",str(remaining),*cmd],
   stdout=log,stderr=subprocess.STDOUT,timeout=remaining+4,preexec_fn=preexec,env=env)
 report["steps"].append({"name":name,"returncode":result.returncode,"seconds":time.monotonic()-stamp})
 save();print(name,result.returncode,flush=True)
 if size()>limits["max_disk_bytes"]:raise RuntimeError("disk budget")
 return result.returncode
def sandbox_limits():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 resource.setrlimit(resource.RLIMIT_AS,(limits["max_address_space_bytes"],)*2)
 seconds=max(1,int(deadline-time.monotonic())-5)
 resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds))
 resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,)*2)
 resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
try:
 save()
 if bounded(["cmake","-G","Ninja","-S",str(SRC),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release",
             "-DPOSEIDON_ROOT="+str(ROOT),"-DPOSEIDON_CPU_BUILD="+str(LIBBUILD)],"configure"):
  raise RuntimeError("configure failed")
 if bounded(["cmake","--build",str(BUILD),"-j2"],"build"):raise RuntimeError("build failed")
 binary=BUILD/"agent_bootstrap_experiment"
 report["binary_sha256"]=sha(binary);shutil.copy2(binary,OUT/"probe-binary")
 library=os.environ["HECATE_PYTHON_LIBRARY_PATH"]
 assert all(Path(p).is_relative_to("/nix/store") for p in library.split(":"))
 cmd=["/usr/bin/bwrap","--unshare-all","--new-session","--die-with-parent","--cap-drop","ALL","--clearenv",
 "--ro-bind","/nix/store","/nix/store","--ro-bind",str(binary),"/probe",
 "--ro-bind",str(LIB),str(LIB),"--proc","/proc","--dev","/dev","--tmpfs","/tmp",
 "--setenv","HOME","/nonexistent","--setenv","PATH","/nonexistent",
 "--setenv","OMP_NUM_THREADS","1","--setenv","OPENBLAS_NUM_THREADS","1",
 "--setenv","LD_LIBRARY_PATH",library,"--chdir","/tmp","--","/probe"]
 code=bounded(cmd,"execute",sandbox_limits,{"PATH":"/usr/bin:/bin"})
 log=(OUT/"execute.log").read_text()
 report["context_admitted"]="context_admitted=true" in log
 report["comparisons"]=[line for line in log.splitlines() if line.startswith("comparison ")]
 report["diagnostics"]=[line for line in log.splitlines() if line.startswith(("state=","blocked_","bootstrap_config "))]
 report["status"]="passed" if code==0 and "all_four_groups_passed=true" in log else "failed"
 report["numerical_passed"]=report["status"]=="passed"
 report["launcher_peak_child_rss_kib"]=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
 report["memory_note"]="Launcher accounting does not measure the isolated probe peak."
except Exception as exc:
 report["status"]="blocked";report["error"]=str(exc)
finally: save()
print(json.dumps({k:v for k,v in report.items() if k not in ("parameters","source_hashes")},indent=2))

raise SystemExit(0 if report["status"]=="passed" else 1)
