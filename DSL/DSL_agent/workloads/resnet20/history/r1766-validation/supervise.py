"""Host-side owner proof and bounded offline SDK preflight (no native/API work)."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).parent
ROOT = Path('/home/lhohy/Code/Poseidon-generalization')
BASE = ROOT / 'DSL/DSL_agent/scripts/baseline'
WORK = Path('/home/lhohy/poseidon-work/platforms/aarch64-linux')
RESULT = WORK / 'results/agent-resnet20-reference-r1766'
SDK = WORK / 'results/agent-current-sdk-release-r1725/sdk.json'
sys.path.insert(0, str(BASE))
from shared_execution import Coordinator, process, retained_bytes

assert not RESULT.exists()
coordinator = Coordinator('/home/lhohy/poseidon-work/coordination/v1', [WORK / 'results'])
state, usage = coordinator.snapshot()
assert not [name for name, batch in state['batches'].items() if batch['status'] not in
            ('succeeded', 'failed', 'cancelled', 'interrupted')]
assert 'resnet20-reference-r1766' not in state['batches']
assert usage['observed_bytes'] + 64*1024**2 <= 48*1024**3
assert shutil.disk_usage(WORK).free >= 4*1024**3 + 64*1024**2
paths = [p for p in BASE.rglob('*') if p.is_file() and not p.is_symlink() and
         p.suffix in ('.py', '.json') and '__pycache__' not in p.parts and
         not {'fixtures', 'gold', 'benchmarks'}.intersection(p.relative_to(BASE).parts)]
paths += [ROOT/'DSL/DSL_agent/docs/generic/shared-candidate-live-r1745.json', ROOT/'DSL/DSL_agent/docs/generic/proposals/linear-single-generation-r1737/body.json', WORK / 'results/candidate-replay-z0go45dl/request.json',
          WORK / 'results/agent-legacy-cli-acceptance-r1735/report.json',
          HERE / 'check_reference.py', Path(__file__), SDK,
          Path(json.loads(SDK.read_text())['source_lock']), Path(json.loads(SDK.read_text())['resource_policy'])]
ASSETS=Path('/home/lhohy/Code/Poseidon-resnet20/DSL/DSL_agent/workloads/resnet20')
paths += [ASSETS/'public_assets.py',ASSETS/'model_reference.py',ASSETS/'UPSTREAM_LICENSE']
DOWNLOAD=WORK/'results/agent-resnet20-assets-recovery-r1759/downloads'
for i in range(4):
    p=DOWNLOAD/('completed-%02d.json'%i);paths.append(p);record=json.loads(p.read_text());folder=DOWNLOAD/record['asset']['id'];paths.append(folder/'stream.json');paths += [folder/part['name']for part in record['stream']['parts']]
frozen = dict(nonce=secrets.token_hex(16), source={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted(set(paths))})
plan_path = HERE / 'pinned-plan.json'
with plan_path.open('x') as stream:
    json.dump(frozen, stream, sort_keys=True, indent=2)
env = {name: os.environ[name] for name in ('HOME', 'USER', 'LOGNAME', 'LANG') if name in os.environ}
env.update(PATH='/usr/bin:/bin', POSEIDON_WORK_ROOT='/home/lhohy/poseidon-work',
           POSEIDON_PLATFORM='aarch64-linux', PYTHONDONTWRITEBYTECODE='1',
           POSEIDON_SDK_MANIFEST=str(SDK),
           NIX_BUILD_SHELL='/nix/store/q36009ll4v7ay3xv4na856w715nfrbn8-bash-5.2p26/bin/bash')
command = ['/usr/bin/bwrap', '--unshare-user', '--unshare-net', '--die-with-parent',
           '--new-session', '--bind', '/', '/', '--proc', '/proc', '--dev', '/dev', '--',
           'bash', 'scripts/baseline/nix_portable.sh', 'nix-shell', '--pure',
           '--keep', 'POSEIDON_WORK_ROOT', '--keep', 'POSEIDON_PLATFORM',
           '--keep', 'POSEIDON_SDK_MANIFEST', '--option', 'substitute', 'false',
           '--max-jobs', '0', '--option', 'builders', '',
           '--option', 'allow-import-from-derivation', 'false', '--argstr', 'platform',
           'aarch64-linux', 'src/poseidon/tools/dacapo/poseidon-cpu-shell.nix',
           '--run', shlex.join(['python3', '-B', str(HERE / 'check_reference.py'), str(plan_path)])]
log_path = HERE / 'pinned-host.log'
started = time.monotonic()
proof = None
with log_path.open('x') as log:
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while child.poll() is None:
            assert time.monotonic() - started < 360, 'Host SDK preflight wall budget'
            assert log_path.stat().st_size <= 16*1024**2
            assert shutil.disk_usage(WORK).free >= 4*1024**3
            ready = RESULT / 'owner-ready.json'
            if ready.is_file() and proof is None:
                value = json.loads(ready.read_text())
                assert value['nonce'] == frozen['nonce']
                owner = value['owner']
                assert process(owner['pid']) == owner, 'Host process identity differs from worker'
                pid = owner['pid']
                ancestry = []
                for _ in range(64):
                    ancestry.append(pid)
                    if pid == child.pid:
                        break
                    stat = Path(f'/proc/{pid}/stat').read_text()
                    pid = int(stat[stat.rindex(')')+2:].split()[1])
                    assert pid > 1, 'Worker is not descended from the launched Nix process'
                else:
                    raise AssertionError('Worker ancestry depth budget')
                proof = dict(owner=owner, launched_pid=child.pid, ancestry=ancestry,
                             nonce=value['nonce'], verified_from_host=True)
                with (RESULT / 'host-owner-proof.json').open('x') as stream:
                    json.dump(proof, stream, sort_keys=True, indent=2)
                with (RESULT / 'owner-ack.json').open('x') as stream:
                    json.dump(dict(nonce=value['nonce'], owner_verified=True), stream)
            if RESULT.exists():
                assert retained_bytes([RESULT], reject_links=True) <= 64*1024**2
            time.sleep(.1)
        code = child.wait()
    except BaseException:
        # Only the process group created by this supervisor is eligible to stop.
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        raise
if RESULT.exists():
    shutil.copyfile(log_path, RESULT / 'host.log')
    shutil.copyfile(plan_path, RESULT / 'host-plan.json')
print(json.dumps(dict(returncode=code, elapsed_seconds=time.monotonic()-started,
                      owner_verified=proof is not None, result=str(RESULT))))
if code:
    print(log_path.read_text()[-12000:])
sys.exit(code)
