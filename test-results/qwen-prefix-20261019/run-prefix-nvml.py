#!/usr/bin/env python3
"""Repeat the bounded graph with a persistent, low-cost GPU observer."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from nvml_sampler import Observer

ROOT = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location('original_runner', ROOT/'run-prefix.py')
original = importlib.util.module_from_spec(loader)
loader.loader.exec_module(original)
save, digest = original.save, original.digest
BASE = original.BASE


def run_case(observer, variant, window):
    folder = ROOT/'runs-nvml'/variant
    folder.mkdir(parents=True, exist_ok=False)
    prefix = ROOT/'plans/prefix4-2000000'
    fixture = json.loads((prefix/'fixture.json').read_text())
    fixture['waterline'] = json.loads((prefix/'prefix-summary.json').read_text())['config']
    fixture['waterline']['lookahead_groups'] = window
    save(folder/'fixture.json', fixture)
    bundle = BASE/'qwen3-plans-bsgs-q50/gpu4/qwen3._hecate_qwen3_0_6b.bundle'
    executable = ROOT/'build-observed-v2'/variant/'bin/poseidon_runtime_gpu_qwen_e2e'
    command = [str(executable), str(prefix/'prefix.runtime-plan.bin'), str(BASE/'profiles/operator-spec.json'),
               str(folder/'fixture.json'), str(folder/'report.json'), str(folder/'decoded.json'),
               '--bundle', str(bundle)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3',
                       POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB='24576',
                       POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='24576',
                       OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    for variable in ('POSEIDON_RUNTIME_TRACE', 'POSEIDON_TRACE'):
        environment.pop(variable, None)
    save(folder/'command.json', command)
    save(folder/'environment.json', {key: environment[key] for key in (
        'CUDA_VISIBLE_DEVICES','POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB',
        'POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')})
    save(folder/'bindings.json', dict(executable_sha256=digest(executable),
         plan_sha256=digest(prefix/'prefix.runtime-plan.bin'), fixture_sha256=digest(folder/'fixture.json'),
         spec_sha256=digest(BASE/'profiles/operator-spec.json'), manifest_sha256=digest(bundle/'manifest.bin'),
         bundle_data=dict(path=str(bundle/'data.bin'),bytes=(bundle/'data.bin').stat().st_size,
                          mtime_ns=(bundle/'data.bin').stat().st_mtime_ns)))
    idle = 0
    while idle < 3:
        apps = observer.applications()
        idle = 0 if apps else idle+1
        if apps:
            print('Waiting for idle GPUs:', json.dumps(apps), flush=True)
        time.sleep(2)
    print('Starting', variant, 'window', window, 'persistent NVML', flush=True)
    begin = time.monotonic()
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    save(folder/'launch.json', dict(pid=process.pid,monotonic=begin,variant=variant,window=window,
                                   timestamp=datetime.datetime.now().astimezone().isoformat()))
    stop, failures, markers = threading.Event(), [], []

    def monitor():
        try:
            with (folder/'samples.jsonl').open('x') as stream:
                while not stop.is_set():
                    started = time.monotonic()
                    sample = observer.sample()
                    sample['elapsed'] = sample['monotonic']-begin
                    sample['timestamp'] = datetime.datetime.now().astimezone().isoformat()
                    try:
                        status = Path(f'/proc/{process.pid}/status').read_text()
                    except FileNotFoundError:
                        status = None
                    sample['host'] = None if status is None else {
                        line.split(':')[0]+'_bytes':int(line.split()[1])*1024
                        for line in status.splitlines() if line.split(':')[0] in ('VmRSS','VmHWM')}
                    stream.write(json.dumps(sample)+'\n')
                    stream.flush()
                    stop.wait(max(0,2-(time.monotonic()-started)))
        except Exception as error:
            failures.append(repr(error))
            print('Observer failed:', repr(error), flush=True)
            if process.poll() is None:
                process.terminate()

    thread = threading.Thread(target=monitor)
    thread.start()
    with (folder/'run.log').open('x') as stream:
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            text = line.strip()
            if text in ('PREFIX_EXECUTION_BEGIN','PREFIX_EXECUTION_END','FORWARD begin') or text.startswith('FORWARD complete'):
                markers.append(dict(monotonic=time.monotonic(),text=text))
                save(folder/'markers.json',markers)
                print(variant,text,flush=True)
    code = process.wait()
    end = time.monotonic()
    stop.set()
    thread.join()
    report = json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else None
    result = dict(variant=variant,window=window,exit_code=code,wall_seconds=end-begin,
                  report_passed=bool(report and report['passed']),telemetry_errors=failures,pid=process.pid,
                  observer='NVML initialized once; 2-second persistent sampling; no SMI or pmon subprocesses')
    save(folder/'status.json',result)
    print('Finished',json.dumps(result),flush=True)
    return result


def main():
    observer = Observer()
    try:
        check = [observer.sample() for _ in range(3)]
        save(ROOT/'nvml-observer-check.json',check)
        print('Observer check seconds:',[s['query_seconds'] for s in check],flush=True)
        results = [run_case(observer,n,w) for n,w in (('updated',128),('previous',8))]
        save(ROOT/'run-status-nvml.json',results)
        return 0 if all(r['report_passed'] and not r['telemetry_errors'] for r in results) else 1
    finally:
        observer.close()


if __name__ == '__main__':
    raise SystemExit(main())
