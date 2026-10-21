#!/usr/bin/env python3
"""Run the same bounded four-GPU Qwen graph with two Runtime versions and telemetry."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

OUT = Path(__file__).resolve().parent
BASE = Path('/home/xuming/poseidon-waterline-20261013')
DEVICES = '0,1,2,3'


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def query(arguments):
    return subprocess.check_output(['nvidia-smi', '-i', DEVICES] + arguments,
                                   text=True, timeout=15)


def applications():
    output = query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
    return [dict(uuid=row[0].strip(), pid=int(row[1]), memory_mib=int(row[2]))
            for row in csv.reader(output.splitlines()) if row]


def wait_for_idle():
    consecutive = 0
    while consecutive < 3:
        processes = applications()
        if processes:
            consecutive = 0
            print('Waiting for idle GPUs:', json.dumps(processes), flush=True)
        else:
            consecutive += 1
        time.sleep(2)


def run_case(variant, window, plan, fixture, bundle):
    folder = OUT / 'runs' / variant
    folder.mkdir(parents=True, exist_ok=False)
    fixture_data = json.loads(fixture.read_text())
    fixture_data['waterline'] = json.loads((plan.parent / 'prefix-summary.json').read_text())['config']
    fixture_data['waterline']['lookahead_groups'] = window
    save(folder / 'fixture.json', fixture_data)
    executable = OUT / 'build-observed-v2' / variant / 'bin/poseidon_runtime_gpu_qwen_e2e'
    command = [str(executable), str(plan), str(BASE / 'profiles/operator-spec.json'),
               str(folder / 'fixture.json'), str(folder / 'report.json'), str(folder / 'decoded.json'),
               '--bundle', str(bundle)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICES,
                       POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB='24576',
                       POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='24576',
                       OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    for variable in ('POSEIDON_RUNTIME_TRACE', 'POSEIDON_TRACE'):
        environment.pop(variable, None)
    save(folder / 'command.json', command)
    save(folder / 'environment.json', {key: environment[key] for key in (
        'CUDA_VISIBLE_DEVICES', 'POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB',
        'POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')})
    save(folder / 'bindings.json', dict(executable_sha256=digest(executable),
         plan_sha256=digest(plan), fixture_sha256=digest(folder / 'fixture.json'),
         spec_sha256=digest(BASE / 'profiles/operator-spec.json'),
         manifest_sha256=digest(bundle / 'manifest.bin'),
         bundle_data=dict(path=str(bundle / 'data.bin'), bytes=(bundle / 'data.bin').stat().st_size,
                          mtime_ns=(bundle / 'data.bin').stat().st_mtime_ns)))
    wait_for_idle()
    print('Starting', variant, 'window', window, flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    save(folder / 'launch.json', dict(pid=process.pid, monotonic=started,
         timestamp=datetime.datetime.now().astimezone().isoformat(), variant=variant, window=window))
    stop = threading.Event()
    failures = []
    markers = []

    def monitor():
        try:
            with (folder / 'samples.jsonl').open('x') as stream:
                while not stop.is_set():
                    begin = time.monotonic()
                    gpu = query(['--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm',
                                 '--format=csv,noheader,nounits'])
                    devices = []
                    for row in csv.reader(gpu.splitlines()):
                        devices.append(dict(index=int(row[0]), uuid=row[1].strip(), gpu_util=int(row[2]),
                            memory_util=int(row[3]), memory_mib=int(row[4]), power_w=float(row[5]), clock_mhz=int(row[6])))
                    host = None
                    path = Path('/proc') / str(process.pid)
                    try:
                        process_status = (path / 'status').read_text()
                    except FileNotFoundError:
                        process_status = None
                    if process_status is not None:
                        host = {}
                        for line in process_status.splitlines():
                            key, _, value = line.partition(':')
                            if key in ('VmRSS', 'VmHWM'):
                                host[key + '_bytes'] = int(value.split()[0]) * 1024
                    sample = dict(monotonic=time.monotonic(), elapsed=time.monotonic() - started,
                                  timestamp=datetime.datetime.now().astimezone().isoformat(),
                                  query_seconds=time.monotonic() - begin, gpu=devices,
                                  applications=applications(), host=host)
                    stream.write(json.dumps(sample) + '\n')
                    stream.flush()
                    stop.wait(max(0, 1 - (time.monotonic() - begin)))
        except Exception as error:
            failures.append(repr(error))

    pmon_error = (folder / 'pmon.stderr').open('x')
    pmon = subprocess.Popen(['stdbuf', '-oL', 'nvidia-smi', 'pmon', '-i', DEVICES,
                             '-s', 'um', '-d', '1'], stdout=subprocess.PIPE,
                            stderr=pmon_error, text=True, bufsize=1)

    def collect_pmon():
        with (folder / 'pmon.jsonl').open('x') as stream:
            for line in pmon.stdout:
                stream.write(json.dumps(dict(monotonic=time.monotonic(), text=line.rstrip('\n'))) + '\n')
                stream.flush()

    observer = threading.Thread(target=monitor)
    pmon_reader = threading.Thread(target=collect_pmon)
    observer.start()
    pmon_reader.start()
    with (folder / 'run.log').open('x') as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            stripped = line.strip()
            if stripped in ('PREFIX_EXECUTION_BEGIN', 'PREFIX_EXECUTION_END', 'FORWARD begin') or stripped.startswith('FORWARD complete'):
                event = dict(monotonic=time.monotonic(), text=stripped)
                markers.append(event)
                print(variant, stripped, flush=True)
    code = process.wait()
    finished = time.monotonic()
    stop.set()
    observer.join()
    if pmon.poll() is None:
        pmon.terminate()
    pmon.wait(timeout=15)
    pmon_reader.join()
    pmon_error.close()
    save(folder / 'markers.json', markers)
    report = json.loads((folder / 'report.json').read_text()) if (folder / 'report.json').exists() else None
    result = dict(variant=variant, window=window, exit_code=code,
                  wall_seconds=finished - started, report_passed=bool(report and report['passed']),
                  telemetry_errors=failures, pid=process.pid)
    save(folder / 'status.json', result)
    print('Finished', json.dumps(result), flush=True)
    return result


def main():
    print('Preparing bounded original four-GPU Qwen plan.', flush=True)
    source_plans = list((BASE / 'qwen3-plans-bsgs-q50/gpu4').glob('*.runtime-plan.bin'))
    assert len(source_plans) == 1
    source = source_plans[0]
    bundle = source.with_name(source.name.removesuffix('.runtime-plan.bin') + '.bundle')
    (OUT / 'plans').mkdir(exist_ok=True)
    prefix = OUT / 'plans/prefix4-2000000'
    helper_command = [str(OUT / 'make-prefix'), str(source), str(BASE / 'profiles/operator-spec.json'),
                      str(BASE / 'qwen3-runner-bsgs-q50.json'), str(prefix), '2000000']
    save(OUT / 'prefix-command.json', helper_command)
    with (OUT / 'prefix-generation.log').open('x') as log:
        subprocess.run(helper_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert json.loads((prefix / 'prefix-summary.json').read_text())['execution_instructions'] <= 2000000
    assert (OUT / 'build-observed-v2/commands.json').exists(), 'observed runners are not built'
    save(OUT / 'source-bindings.json', dict(source_plan_sha256=digest(source),
         full_fixture_sha256=digest(BASE / 'qwen3-runner-bsgs-q50.json'),
         prefix_plan_sha256=digest(prefix / 'prefix.runtime-plan.bin')))
    # New runs first: any first-touch OS-cache effect does not favor the update.
    results = []
    for variant, window in (('updated', 128), ('previous', 8)):
        results.append(run_case(variant, window, prefix / 'prefix.runtime-plan.bin',
                                prefix / 'fixture.json', bundle))
    save(OUT / 'run-status.json', results)
    return 0 if all(r['exit_code'] == 0 and r['report_passed'] and not r['telemetry_errors'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
