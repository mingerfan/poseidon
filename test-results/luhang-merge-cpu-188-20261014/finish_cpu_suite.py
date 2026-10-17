#!/usr/bin/env python3
"""Finish the default correctness suite without the extra dnum timing sweeps."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

base = Path('/home/xuming/poseidon-luhang-merge-cpu-20261014')
logs = base / 'logs'
driver = int((logs / 'driver.pid').read_text())
child = 160626
stat_path = Path(f'/proc/{child}/stat')
ticks = os.sysconf('SC_CLK_TCK')
timed_out = False
while True:
    fields = stat_path.read_text().rsplit(') ', 1)[1].split()
    assert int(fields[1]) == driver
    elapsed = float(Path('/proc/uptime').read_text().split()[0]) - int(fields[19]) / ticks
    if fields[0] == 'Z':
        exit_code = os.waitstatus_to_exitcode(int(fields[49]))
        break
    if elapsed > 1200:
        timed_out = True
        os.killpg(child, signal.SIGTERM)
    time.sleep(2)

report = json.loads((logs / 'results.json').read_text())
report['tests'].append({'name': 'test_ckks_bootstrap_cmp',
    'command': [str(base / 'build-cpu/bin/test_ckks_bootstrap_cmp')],
    'exit_code': 124 if timed_out else exit_code, 'timeout': timed_out,
    'seconds': round(elapsed, 3), 'log': 'test_ckks_bootstrap_cmp.log'})
(logs / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
# The paused driver keeps its child's exit status available until this point.
os.kill(driver, signal.SIGTERM)
os.kill(driver, signal.SIGCONT)

def execute(name, command, timeout):
    start = time.monotonic()
    with (logs / (name + '.log')).open('w') as output:
        output.write('COMMAND: ' + json.dumps(command) + '\n')
        output.flush()
        process = subprocess.run(command, cwd=base, stdout=output,
                                 stderr=subprocess.STDOUT, timeout=timeout)
    return {'name': name, 'command': command, 'exit_code': process.returncode,
            'timeout': False, 'seconds': round(time.monotonic() - start, 3),
            'log': name + '.log'}

build = base / 'build-cpu'
command = ['c++', '-std=c++20', '-O3', '-DNDEBUG', '-fopenmp', '-pthread',
    '-I' + str(base / 'src'), '-I' + str(build),
    '-I' + str(base / 'deps/msgsl-src/include'),
    '-I' + str(base / 'deps/spdlog-1.17.0/include'),
    str(base / 'examples/ckks/test_ckks_bootstrap_dnum.cpp'),
    '-L' + str(build), '-lposeidon_shared', '-lgmpxx', '-lgmp',
    '-Wl,-rpath,' + str(build), '-o', str(build / 'bin/test_ckks_bootstrap_dnum')]
report['extra_build'] = execute('build_bootstrap_dnum', command, 300)
if report['extra_build']['exit_code'] == 0:
    report['tests'].append(execute('test_ckks_bootstrap_dnum_show-chains',
        [str(build / 'bin/test_ckks_bootstrap_dnum'), '--show-chains'], 60))
else:
    report['tests'].append(report['extra_build'])
report['scope'] = 'All 14 default CPU example executables; extra CF-real algebra/in-place checks; dnum source compilation and parameter-chain checks. The optional dnum timing sweeps are outside this correctness run.'
report['finished'] = datetime.datetime.now().astimezone().isoformat()
(logs / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
(logs / 'results-before-audit.json').write_text(json.dumps(report, indent=2) + '\n')
audit = subprocess.run(['python3', str(base / 'run_cpu_suite.py'), str(base), '--audit-existing'])
(logs / 'finish.status').write_text(str(audit.returncode) + '\n')
