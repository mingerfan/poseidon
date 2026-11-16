#!/usr/bin/env python3
"""Run fresh-process synthetic key preparation probes, without storing credentials.

Run on the GPU host; all paths are explicit. Each case includes validation outside
stage timers. The default long-chain cases are probes, not a boot benchmark.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def gpu_state():
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                           '--format=csv,noheader'], capture_output=True, text=True, check=True)
    return {'gpu': result.stdout.strip(), 'apps': apps.stdout.strip()}


def environment():
    return {
        'hostname': os.uname().nodename,
        'uname': list(os.uname()),
        'cpu_max': text('/sys/fs/cgroup/cpu.max'),
        'memory_max': text('/sys/fs/cgroup/memory.max'),
        'cpuset_effective': text('/sys/fs/cgroup/cpuset.cpus.effective'),
        'affinity': sorted(os.sched_getaffinity(0)),
        'cpuinfo': text('/proc/cpuinfo'),
        'cuda_version': subprocess.run(['/usr/local/cuda/bin/nvcc', '--version'],
                                      capture_output=True, text=True).stdout,
        'initial_gpu': gpu_state(),
        'note': 'No GPU reset, cache drop, or cold physical disk assertion. Synthetic keys only.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--threads', default='1', help='unset to use libgomp default')
    parser.add_argument('--wait', choices=['DEFAULT', 'ACTIVE', 'PASSIVE'], default='PASSIVE')
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--affinity', default='0-6', help='all to retain caller affinity')
    parser.add_argument('probe_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    # This lock coordinates ONLY processes using this exact file/inode.
    # The device UUID and fresh nvidia-smi check below are independently recorded.
    with (root / '.probe.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = root / args.case
        output.mkdir(exist_ok=False)
        state = gpu_state()
        if state['apps']:
            raise RuntimeError(f'GPU is busy: {state}')
        if int(state['gpu'].split(',')[2]) > 512:
            raise RuntimeError(f'Unexpected VRAM usage: {state}')
        metadata = environment()
        env = os.environ.copy()
        # Ensure no inherited optional tensor-table or threading experiment leaks.
        for key in list(env):
            if key.startswith(('POSEIDON_NTT_', 'POSEIDON_INTT_', 'OMP_', 'GOMP_')):
                env.pop(key)
        for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            env[key] = '1'
        if args.threads != 'unset':
            env['OMP_NUM_THREADS'] = args.threads
        env.update(OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', CUDA_VISIBLE_DEVICES='0')
        if args.wait != 'DEFAULT':
            env['OMP_WAIT_POLICY'] = args.wait
        if args.wait == 'PASSIVE':
            env['GOMP_SPINCOUNT'] = '0'
        command = [args.binary, *[x for x in args.probe_args if x != '--']]
        if args.affinity != 'all':
            command = ['taskset', '-c', args.affinity, *command]
        metadata.update(command=command, selected_environment={k: v for k, v in env.items()
                        if k.startswith(('OMP_', 'GOMP_', 'POSEIDON_', 'CUDA_', 'OPENBLAS_', 'MKL_', 'NUMEXPR_'))},
                        timeout_s=args.timeout)
        metadata['cpu_stat_before'] = text('/sys/fs/cgroup/cpu.stat')
        start = time.perf_counter()
        with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            try:
                completed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                           timeout=args.timeout)
                metadata['exit_code'] = completed.returncode
                metadata['timed_out'] = False
            except subprocess.TimeoutExpired:
                metadata['exit_code'] = None
                metadata['timed_out'] = True
        metadata['process_wall_s'] = time.perf_counter() - start
        metadata['cpu_stat_after'] = text('/sys/fs/cgroup/cpu.stat')
        metadata['final_gpu'] = gpu_state()
        (output / 'launch.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'case': args.case, 'exit_code': metadata['exit_code'],
                          'timed_out': metadata['timed_out'], 'wall_s': metadata['process_wall_s']}), flush=True)
        if metadata['timed_out'] or metadata['exit_code']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
