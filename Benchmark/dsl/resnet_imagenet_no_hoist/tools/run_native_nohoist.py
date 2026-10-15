"""Run an isolated native baseline on an idle GPU; preserve every log."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def idle_gpu(gpu):
    output = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    uuid, free = (item.strip() for item in output.strip().split(','))
    tasks = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'], text=True).splitlines()
    if uuid in [item.strip() for item in tasks] or int(free) < 28672:
        raise RuntimeError('GPU occupied or less than 28 GiB free')
    return uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bin_dir', type=Path)
    parser.add_argument('log', type=Path)
    parser.add_argument('--gpu', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--model', choices=('bootstrap', '18', '50'), required=True)
    parser.add_argument('--blocks', type=int, choices=range(17), default=0)
    args = parser.parse_args()
    if args.model == '18' and args.blocks > 8:
        parser.error('ResNet18 has only 8 blocks')
    binary = args.bin_dir / ('imagenet_nohoist_bootstrap_probe' if args.model == 'bootstrap'
                             else f'resnet{args.model}_native_nohoist')
    command = [str(binary), '--run'] if args.model == 'bootstrap' else [
        str(binary), '--infer', '0', str(args.blocks)]
    if not binary.is_file():
        raise FileNotFoundError(binary)
    env = os.environ.copy()
    for key in ('POSEIDON_RESCALE_META_BITS', 'POSEIDON_RESCALE_MODEL_BITS',
                'POSEIDON_DSL_SCALE_SHIFT', 'POSEIDON_GPU_DNUM'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), POSEIDON_NTT_ALGO='fourstep',
               POSEIDON_GPU_LINEAR_TRANSFORM_MODE='no_hoist',
               POSEIDON_TRIDENT_RESNET18_ROOT='/home/hanshubin/Trident/resnet18',
               POSEIDON_TRIDENT_RESNET50_ROOT='/home/hanshubin/Trident/resnet50')
    uuid = idle_gpu(args.gpu)
    # Exclusive creation refuses to overwrite even a failed experiment's evidence.
    with args.log.open('x') as log:
        log.write(json.dumps(dict(command=command, gpu_uuid=uuid, pool_gib=26,
                                  scope='native baseline, not DSL equivalence')) + '\n')
        log.flush()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f'\nNATIVE_BASELINE_PROCESS_EXIT code={result.returncode}\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
