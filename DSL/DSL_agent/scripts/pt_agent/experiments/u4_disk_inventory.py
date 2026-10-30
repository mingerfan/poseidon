"""U4 §7.9.5: read-only directory inventory after a disk admission stop."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tarfile

from ..paths import REPO, RESULTS_DIR, WORK_RESULTS


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def inventory(out):
    roots = (WORK_RESULTS, RESULTS_DIR)
    rows = []
    by_name = defaultdict(list)
    errors = []
    scripts = set()
    packages = set()
    frozen_files = set()
    seen_inodes = set()
    for root in roots:
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                continue
            status = directory.stat()
            row = dict(path=str(directory), root=str(root), name=directory.name,
                       size_bytes=0, allocated_bytes=0, files=0,
                       modified=stamp(status.st_mtime), latest_modified=stamp(status.st_mtime),
                       references=defaultdict(set))
            latest = status.st_mtime
            for base, dirs, files in os.walk(directory, followlinks=False,
                                             onerror=lambda exc: errors.append(str(exc))):
                base = Path(base)
                for name in dirs + files:
                    path = base/name
                    try:
                        info = path.lstat()
                    except OSError as exc:
                        errors.append(str(exc))
                        continue
                    latest = max(latest, info.st_mtime)
                    identity = (info.st_dev, info.st_ino)
                    if identity not in seen_inodes:
                        seen_inodes.add(identity)
                        row['allocated_bytes'] += info.st_blocks*512
                    if name in files:
                        row['files'] += 1
                        row['size_bytes'] += info.st_size
                    if path.is_symlink():
                        continue
                    if name == 'run.sh' and 'server_jobs' in path.parts:
                        scripts.add(path)
                    if name == 'compile_argv.json' and base.name == 'config':
                        packages.add(base.parent)
                    if directory.name.startswith('r8e7-reconfirm-') and path.suffix in (
                            '.json', '.jsonl', '.py', '.sh', '.md') and path.is_file():
                        frozen_files.add(path)
            row['latest_modified'] = stamp(latest)
            rows.append(row)
            by_name[directory.name].append(row)

    sources = defaultdict(int)
    tokens = re.compile(r'[A-Za-z0-9_.-]+')

    def reference(text, category, source):
        sources[category] += 1
        for name in set(tokens.findall(text)) & by_name.keys():
            for row in by_name[name]:
                row['references'][category].add(source)

    def read(path, category):
        try:
            with path.open(errors='replace') as stream:
                for line in stream:
                    reference(line, category, str(path))
        except OSError as exc:
            errors.append(str(exc))

    agent = REPO/'DSL/DSL_agent/scripts/pt_agent'
    # Current configuration and its location defaults; no credentials are read.
    for path in (agent/'paths.py', agent/'config.py'):
        read(path, 'current_configuration')
    for path in (agent/'compiler_profiles').rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.py', '.sh'):
            read(path, 'current_configuration')
    for path in scripts:
        read(path, 'server_job')
        for row in rows:
            if path.is_relative_to(Path(row['path'])):
                row['references']['server_job'].add(str(path)+': containing directory for relative paths')
    for package in packages:
        for row in rows:
            if package.is_relative_to(Path(row['path'])):
                row['references']['p2_package'].add(str(package)+': containing directory for relative paths')
        for sub in ('config', '_package'):
            for path in (package/sub).rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.py', '.sh'):
                    read(path, 'p2_package')
        for name in ('run.sh', 'model.py', 'trace_model.py', 'manifest.json', 'MANIFEST.json'):
            if (package/name).is_file():
                read(package/name, 'p2_package')
    docs = REPO/'DSL/DSL_agent/docs/pt_agent'
    for name in ('STATUS.md', 'PLAN.md'):
        read(docs/name, 'status_plan')
    for path in frozen_files:
        read(path, 'frozen_result')
    for row in rows:
        if row['name'].startswith('r8e7-reconfirm-'):
            row['references']['frozen_result'].add('frozen result directory itself')
        if out.is_relative_to(Path(row['path'])):
            row['references']['active_stop'].add(str(out)+': current preserved stop evidence')
    # Read the frozen tree directly from Git without checking it out or writing it.
    command = ['git', 'archive', 'pt-agent-freeze-v1', 'DSL/DSL_agent/scripts/pt_agent']
    with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for member in archive:
                if member.isfile() and Path(member.name).suffix in ('.py', '.json', '.sh', '.md'):
                    stream = archive.extractfile(member)
                    for line in stream:
                        reference(line.decode(errors='replace'), 'frozen_tag',
                                  'pt-agent-freeze-v1:'+member.name)
        stderr = proc.stderr.read().decode(errors='replace')
        code = proc.wait()
        if code:
            errors.append('git archive: '+stderr)

    totals = defaultdict(int)
    for row in rows:
        refs = row['references']
        if any(refs.get(key) for key in ('current_configuration', 'p2_package', 'server_job', 'active_stop')):
            category = 'must_retain'
        elif any(refs.get(key) for key in ('status_plan', 'frozen_result', 'frozen_tag')):
            category = 'retain'
        else:
            category = 'cleanup_candidate_requires_user_approval'
        row['classification'] = category
        row['references'] = {key: sorted(values) for key, values in refs.items()}
        totals[category+'_bytes'] += row['size_bytes']
        totals[category+'_allocated_bytes'] += row['allocated_bytes']
        totals[category+'_directories'] += 1
    report = dict(round='U4', section='7.9.5.4', generated=datetime.now(timezone.utc).isoformat(),
                  roots=list(map(str, roots)), read_only=True, deleted_files=0,
                  directory_size='recursive file sizes; allocated bytes count each inode once',
                  reference_matching='directory-name tokens in the requested configuration, packages, jobs, documents and frozen results',
                  free_bytes=shutil.disk_usage(RESULTS_DIR).free,
                  totals=dict(totals), directories=rows, errors=errors,
                  deletion='candidate classification grants no permission; individual user approval required')
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    report = inventory(args.out)
    print(json.dumps(dict(inventory=str(args.out), **report['totals'], errors=len(report['errors']))))
    return 0 if not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
