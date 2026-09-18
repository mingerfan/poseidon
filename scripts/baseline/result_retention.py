"""Remove only regenerated SEAL key files from completed local experiments.

Reports, inputs, weights, DSL, IR, HEVM/CST and decrypted outputs are retained.
The CLI defaults to a read-only plan and refuses cleanup while runners exist.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

PREFIXES = ('agent-deepseek-', 'candidate-replay-', 'fx-batch-', 'seal-cpu-golden-', 'multi-input-golden-')
TERMINAL = {'passed', 'failed', 'repair_budget_exhausted', 'provider_failed', 'provider_exhausted',
            'infrastructure_failed', 'input_failed'}
KEYS = {'gal.seal', 'relin.seal', 'sec.seal', 'pub.seal', 'parm.seal'}


def bounded_json(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 8*1024**2:
        raise ValueError('Expected bounded regular metadata file')
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def plan_run(run, results):
    results = Path(results).resolve()
    run = Path(run).absolute()
    if run.is_symlink() or run.resolve() != run or run.parent != results or not run.name.startswith(PREFIXES):
        raise ValueError('Cleanup target must be an explicit experiment directly below results')
    report, digest = bounded_json(run/'report.json')
    if report.get('status') not in TERMINAL:
        raise ValueError('Experiment is not terminal')
    keys = run/'private-keys'
    if not keys.exists() and not keys.is_symlink():
        return dict(run=str(run), report_sha256=digest, files=[], bytes=0)
    if keys.is_symlink() or not keys.is_dir():
        raise ValueError('Key directory must not be a link')
    parameters, pdigest = bounded_json(run/'parameters.json')
    if (parameters.get('seal_version') != '4.0.0' or parameters.get('polynomial_degree') != 32768
            or parameters.get('security_check') != 'tc128' or parameters.get('parameters_set') is not True):
        raise ValueError('Unrecognized generated key parameters')
    files = []
    for path in sorted(keys.iterdir()):
        info = path.lstat()
        if path.name not in KEYS or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('Unexpected key-directory entry; preserve everything')
        files.append(dict(name=path.name, bytes=info.st_size, inode=info.st_ino, device=info.st_dev))
    return dict(run=str(run), report_sha256=digest, parameters_sha256=pdigest,
                files=files, bytes=sum(f['bytes'] for f in files))


def cleanup_run(run, results):
    plan = plan_run(run, results)
    if not plan['files']:
        return plan
    run = Path(plan['run'])
    audit = run/'key-cleanup.json'
    # Do not overwrite prior audit or any historical result report.
    with audit.open('x') as stream:
        json.dump(dict(plan, started_unix=time.time(), policy='regenerable_keys_only',
                       original_random_keys_recoverable=False,
                       regenerate='seal_golden_keys NEW_EMPTY_DIRECTORY <rotation_steps from parameters.json>'), stream, indent=2)
    directory = os.open(run/'private-keys', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    removed = []
    try:
        if bounded_json(run/'report.json')[1] != plan['report_sha256']:
            raise ValueError('Report changed during cleanup')
        for item in plan['files']:
            info = os.stat(item['name'], dir_fd=directory, follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                    (info.st_dev, info.st_ino, info.st_size) != (item['device'], item['inode'], item['bytes'])):
                raise ValueError('Key changed during cleanup')
            os.unlink(item['name'], dir_fd=directory)
            removed.append(item)
    finally:
        os.close(directory)
        with (run/'key-cleanup-outcome.json').open('x') as stream:
            json.dump(dict(removed=removed, freed_bytes=sum(i['bytes'] for i in removed),
                           complete=len(removed)==len(plan['files']), finished_unix=time.time()), stream, indent=2)
    (run/'private-keys').rmdir()
    return plan


def main():
    from hecate_python_env import WORK
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args=parser.parse_args()
    if args.apply:
        processes=subprocess.run(['ps','-eo','args='],capture_output=True,text=True,check=True,timeout=10).stdout
        if any(name in processes for name in ('run_candidate.py','run_agent_batch.py','run_model_batch.py',
                                              'seal_cpu_golden.py','run_multi_input_goldens.py')):
            raise ValueError('Stop experiment runners before retrospective cleanup')
    results=WORK/'results'
    plans, skipped=[],[]
    for run in sorted(results.iterdir()):
        if not run.name.startswith(PREFIXES) or not (run/'private-keys').exists():
            continue
        try:
            plan=cleanup_run(run,results) if args.apply else plan_run(run,results)
            plans.append(plan)
        except (OSError,ValueError) as error:
            skipped.append(dict(run=str(run),reason=type(error).__name__))
    print(json.dumps(dict(mode='applied' if args.apply else 'plan',runs=len(plans),
                         bytes=sum(p['bytes'] for p in plans),skipped=skipped),indent=2))


if __name__=='__main__':
    main()
