#!/usr/bin/env python3
"""Profile a later segment after the initial preparation burst has passed."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'poseidon-qwen-prefix-20261019'
nsys='/usr/local/cuda-12.2/nsight-systems-2023.2.3/bin/nsys'
folder=ROOT/'nsys-late';folder.mkdir(exist_ok=False)
runner=ROOT/'build-host-markers/qwen'
command=json.loads((SOURCE/'runs-nvml/updated/command.json').read_text())
command[0]=str(runner);command[4]=str(folder/'report.json');command[5]=str(folder/'decoded.json')
env=dict(os.environ,**json.loads((SOURCE/'runs-nvml/updated/environment.json').read_text()))
for key in ('POSEIDON_RUNTIME_TRACE','POSEIDON_TRACE'):env.pop(key,None)
env['POSEIDON_RUNTIME_TRACE']=str(folder/'boot')
profile=[nsys,'profile','--trace=cuda,nvtx,osrt','--osrt-threshold=100000','--sample=none',
    '--cpuctxsw=none','--cuda-memory-usage=false','--stats=false','--delay=110','--duration=20',
    '--kill=sigterm','--wait=primary','-o',str(folder/'profile')]+command
loader=importlib.util.spec_from_file_location('cpu_probe',ROOT/'run-cpu-probe.py')
probe=importlib.util.module_from_spec(loader);loader.loader.exec_module(probe)
observer=probe.Observer()
try:
    assert not observer.applications(),'Other compute processes are active'
    probe.save(folder/'command.json',profile)
    started=time.monotonic()
    with (folder/'profile.log').open('w') as log:
        p=subprocess.Popen(profile,env=env,stdout=log,stderr=subprocess.STDOUT)
        probe.save(folder/'launch.json',dict(pid=p.pid,monotonic=started,
            scope='20-second later diagnostic segment; intentionally terminated by nsys'))
        code=p.wait()
    probe.save(folder/'status.json',dict(exit_code=code,wall_seconds=time.monotonic()-started))
    # Some nsys versions propagate the intentionally sent target SIGTERM (143).
    if code not in (0,143):raise RuntimeError(f'nsys returned {code}; inspect profile.log')
    assert (folder/'profile.nsys-rep').is_file()
    print('Late Nsight capture finished',flush=True)
    with (folder/'export.log').open('w') as log:
        subprocess.run([nsys,'export','--type=sqlite','-o',str(folder/'profile.sqlite'),
            str(folder/'profile.nsys-rep')],stdout=log,stderr=subprocess.STDOUT,check=True)
    probe.run_case(observer,'api-probe-late',probe=True,delay=110,duration=20,executable=runner)
finally:observer.close()
print('All later diagnostics finished',flush=True)
