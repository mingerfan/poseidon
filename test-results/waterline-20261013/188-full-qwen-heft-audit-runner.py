#!/usr/bin/env python3
"""One authorized CPU-only full compile plus immediate pass-boundary audit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, default=os.environ.get(
    'POSEIDON_188_WORKSPACE', '/home/xuming/poseidon-waterline-20261013'))
parser.add_argument('--output-dir', type=Path, default=os.environ.get('POSEIDON_HEFT_AUDIT_OUT'),
                    help='fresh output directory; never reuse an earlier audit directory')
parser.add_argument('--print-command', action='store_true',
                    help='validate the fresh destination and print commands without writing or launching')
args = parser.parse_args()
BASE = args.workspace.expanduser().resolve()
OUT = (args.output_dir or BASE / 'qwen3-plans-bsgs-q50-heft-audit/gpu4').expanduser().resolve()
ORIGINAL = BASE / 'qwen3-plans-bsgs-q50/gpu4'
STATUS = OUT / 'status.json'
AUDITOR = Path(__file__).resolve().with_name('full-qwen-heft-stream-audit.py')
START = time.monotonic()

def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def stamp(path):
    st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino}

state = {'scope': 'one CPU-only Q50 logical40 full compile; immediate before/after the same Waterline pass invocation',
         'started_at': datetime.datetime.now().astimezone().isoformat(), 'supervisor_pid': os.getpid(),
         'nice': os.getpriority(os.PRIO_PROCESS, 0), 'numerical_model_validation': False,
         'complete': False, 'passed': False, 'stage': 'preflight'}

def update(**changes):
    state.update(changes, elapsed_seconds=time.monotonic()-START)
    write_json(STATUS, state)
    print(json.dumps(changes), flush=True)

if not AUDITOR.is_file():
    raise SystemExit('missing sibling auditor: ' + str(AUDITOR))
if OUT == ORIGINAL.resolve():
    raise SystemExit('refusing the original plan directory as audit output')
if OUT.exists():
    # Allow scripts uploaded into a fresh directory, but no earlier evidence or
    # partial compiler outputs. In particular, never truncate status/log/binary.
    allowed = {Path(__file__).resolve(), AUDITOR.resolve()}
    conflicts = []
    for path in OUT.iterdir():
        same_auditor = (path.name == AUDITOR.name and path.is_file()
                        and digest(path) == digest(AUDITOR))
        if path.resolve() not in allowed and not same_auditor:
            conflicts.append(path.name)
    if conflicts:
        raise SystemExit('refusing existing audit artifacts: ' + ', '.join(sorted(conflicts)))
generated = json.loads((ORIGINAL / 'command.json').read_text())
generated = generated.copy()
generated[generated.index('-o') + 1] = str(OUT / 'qwen3.mlir')
generated += ['--mlir-print-ir-before=plan-plaintext-waterline',
              '--mlir-print-ir-after=plan-plaintext-waterline']
if args.print_command:
    print(json.dumps({'output_directory': str(OUT), 'compiler_argv': generated,
        'auditor_source': str(AUDITOR), 'audit_argv': [sys.executable, str(OUT/AUDITOR.name),
        '--dump', str(OUT/'compile.log'), '--summary', str(OUT/'canonical-summary.json'),
        '--spool', str(OUT/'before-canonical.jsonl')], 'launched': False}, indent=2))
    sys.exit(0)
OUT.mkdir(parents=True, exist_ok=True)
# Exclusive creation also prevents two simultaneous starts in an empty OUT.
with STATUS.open('x') as reservation:
    reservation.write(json.dumps(state, indent=2) + '\n')

try:
    write_json(OUT / 'command.json', generated)
    if AUDITOR.resolve() != (OUT / AUDITOR.name).resolve():
        if not (OUT / AUDITOR.name).exists():
            shutil.copyfile(AUDITOR, OUT / AUDITOR.name)
    original_cmd = json.loads((ORIGINAL / 'command.json').read_text())
    cmd = json.loads((OUT / 'command.json').read_text())
    expected = original_cmd.copy()
    expected[expected.index('-o') + 1] = str(OUT / 'qwen3.mlir')
    expected += ['--mlir-print-ir-before=plan-plaintext-waterline', '--mlir-print-ir-after=plan-plaintext-waterline']
    if cmd != expected or '--waterline=40' not in cmd:
        raise RuntimeError('audit command differs from authorized original command changes')
    inputs = [BASE / 'toolchain/bin/hecate-opt', BASE / 'profiles/compiler-profile.json',
              BASE / 'profiles/operator-spec.json', BASE / 'profiles/communication-profile.json']
    binding = {str(p): {'sha256': digest(p), **stamp(p)} for p in inputs}
    trace = BASE / 'qwen3-traced-bsgs/trace_qwen3.mlirbc'
    trace_before = stamp(trace)
    if 'sha256:' + binding[str(inputs[2])]['sha256'] not in next(x for x in cmd if x.startswith('--runtime-plan-operator-spec-sha256=')):
        raise RuntimeError('current OperatorSpec does not match the saved original command')
    write_json(OUT / 'input-bindings.json', {'files': binding, 'trace_stat_only': trace_before,
        'source_command': str(ORIGINAL / 'command.json'), 'changes': ['output path', 'before/after Waterline IR print flags'],
        'large_input_or_bundle_hashed': False})
    with (OUT / 'compile.log').open('wb') as log:
        process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(OUT / 'compile.time.log'), *cmd],
                                   cwd=BASE, stdout=log, stderr=log)
        time.sleep(.2)
        children = Path(f'/proc/{process.pid}/task/{process.pid}/children')
        child_ids = children.read_text().split() if children.exists() else []
        update(stage='compiling', time_pid=process.pid, compiler_pids=[int(p) for p in child_ids])
        rc = process.wait()
    (OUT / 'compile.exit-status').write_text(str(rc)+'\n')
    update(compiler_returncode=rc)
    if rc:
        update(stage='compile_failed', complete=True)
        sys.exit(1)
    unchanged = all(digest(p) == binding[str(p)]['sha256'] for p in inputs) and stamp(trace) == trace_before
    if not unchanged:
        raise RuntimeError('compiler/profile/source changed during compile')
    update(stage='canonical_comparison', inputs_unchanged_during_compile=True)
    audit_cmd = [sys.executable, str(OUT/'full-qwen-heft-stream-audit.py'), '--dump', str(OUT/'compile.log'),
                 '--summary', str(OUT/'canonical-summary.json'), '--spool', str(OUT/'before-canonical.jsonl')]
    with (OUT/'canonical-audit.log').open('wb') as log:
        audit_rc = subprocess.run(audit_cmd, cwd=BASE, stdout=log, stderr=log).returncode
    update(canonical_returncode=audit_rc)
    if audit_rc:
        update(stage='canonical_comparison_failed', complete=True)
        sys.exit(1)
    update(stage='binding_original_binary')
    binary = 'qwen3._hecate_qwen3_0_6b.runtime-plan.bin'
    old, new = ORIGINAL/binary, OUT/binary
    old_sha, new_sha = digest(old), digest(new)
    same = old_sha == new_sha
    write_json(OUT/'runtime-plan-binding.json', {'original': {**stamp(old), 'sha256': old_sha},
        'audit': {**stamp(new), 'sha256': new_sha}, 'identical': same, 'bundle_hashed': False})
    update(stage='complete' if same else 'binary_binding_mismatch', complete=True, passed=same,
           binary_identical_to_original=same, numerical_model_validation=False)
    sys.exit(0 if same else 1)
except Exception as error:
    update(stage='failed', complete=True, error=str(error))
    raise
