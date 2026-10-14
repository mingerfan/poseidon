import os,sys,json,time,subprocess,signal
from pathlib import Path
pre=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-common-pytorch-offline-prepared-r1040")
tools=Path(__file__).parent
scope=json.loads((pre/"approved-backend-scope-r1082.json").read_text())
assert os.environ.get("IN_NIX_SHELL")=="pure" and sys.version_info[:3]==(3,10,14)
start=time.monotonic()
def run(tool,log,terminal):
    remaining=int(min(scope["deadline_wall"]-time.time(),scope["wall_seconds"]-(time.monotonic()-start)))-3
    if remaining<=0:raise RuntimeError("Approved batch time exhausted")
    began=time.monotonic()
    with (pre/log).open("x") as output:
        child=subprocess.Popen([sys.executable,"-B",str(tools/tool)],stdout=output,stderr=subprocess.STDOUT,
            start_new_session=True)
        try:code=child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    (pre/terminal).write_text(json.dumps(dict(exit_code=code,seconds=time.monotonic()-began),indent=2)+"\n")
    return code
code=run("backend.py","backend-r1082.log","backend-terminal-r1082.json")
out=pre.parent/"agent-common-pytorch-backend-r1082"
report=json.loads((out/"report.json").read_text()) if (out/"report.json").is_file() else None
audit=None
if report and report["status"]=="complete":
    audit=run("audit.py","audit-r1083.log","audit-terminal-r1083.json")
terminal=dict(backend_exit_code=code,audit_exit_code=audit,seconds=time.monotonic()-start,
    paid_calls=0,single_nix_environment=True)
(pre/"job-terminal-r1082.json").write_text(json.dumps(terminal,indent=2)+"\n")
print(json.dumps(terminal));raise SystemExit(audit if audit not in (None,0) else code)
