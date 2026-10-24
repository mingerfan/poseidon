"""Plan and, with explicit approval, install an isolated intake environment.

The system-Python --plan supervisor uses the existing, read-only Nix interpreter
to obtain CPython 3.10 wheel tags and its bundled dependency resolver.  Wheel
payloads and both venv directories are never opened for writing during --plan.
"""

import argparse
import fcntl
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import ssl
import subprocess
import sys
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from ..paths import PLATFORM_DIR, REPO, pt_result


PROFILE = REPO / 'src/poseidon/tools/dacapo/intake-python-aarch64.profile.json'
# User-approved upper bound: 133 MiB plus 10%, rounded down.
DOWNLOAD_LIMIT = 133 * 1024**2 * 110 // 100


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MetadataIndex:
    """Bounded, cached metadata GETs. Wheel URLs are recorded, never fetched."""

    def __init__(self, profile, out, *, ca_file=None, prior_requests=()):
        self.profile, self.out = profile, out
        self.cache = {}
        self.requests = list(prior_requests)
        self.bytes = sum(row['bytes'] for row in self.requests)
        for row in self.requests:
            if row['complete']:
                suffix = row['url'].removeprefix(profile['index'] + '/')
                self.cache[suffix] = json.loads(Path(row['path']).read_text())
        context = ssl.create_default_context(cafile=str(ca_file) if ca_file else None)
        self.opener = build_opener(NoRedirect(), HTTPSHandler(context=context))

    def get(self, name, version=None):
        suffix = f'{name}/{version}/json' if version else f'{name}/json'
        if suffix in self.cache:
            return self.cache[suffix]
        if len(self.requests) >= self.profile['metadata_request_limit']:
            raise RuntimeError('Metadata request allowance exhausted')
        url = self.profile['index'] + '/' + suffix
        if not url.startswith('https://pypi.org/pypi/'):
            raise ValueError('Planning only reads the public PyPI JSON index')
        row = dict(url=url, complete=False, bytes=0)
        self.requests.append(row)
        write(self.out / 'metadata-requests.json', self.requests)
        maximum = min(self.profile['metadata_response_limit_bytes'],
                      self.profile['metadata_total_limit_bytes'] - self.bytes)
        if maximum <= 0:
            raise RuntimeError('Metadata byte allowance exhausted')
        try:
            request = Request(url, headers={'Accept': 'application/json'})
            with self.opener.open(request, timeout=self.profile['metadata_timeout_seconds']) as response:
                raw = response.read(maximum + 1)
            self.bytes += len(raw)
            row.update(bytes=len(raw), complete=True)
            destination = self.out / 'metadata' / (name + ('-' + version if version else '') + '.json')
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(raw)
            row['path'] = str(destination)
            if len(raw) > maximum:
                raise RuntimeError('Metadata byte allowance exceeded')
            data = json.loads(raw)
        except Exception as exc:
            row.update(complete=False, error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write(self.out / 'metadata-requests.json', self.requests)
        self.cache[suffix] = data
        return data


def resolve(profile, base, index):
    from importlib.metadata import version as installed_version
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import InvalidVersion, Version
    from pip._vendor.resolvelib import BaseReporter, Resolver
    from pip._vendor.resolvelib.providers import AbstractProvider

    actual_python = '.'.join(map(str, sys.version_info[:3]))
    if actual_python != profile['python'] or platform.machine() != 'aarch64':
        raise RuntimeError('Plan requires the existing pinned CPython 3.10.14 aarch64 interpreter')
    if base['python'] != actual_python or base['target'] != profile['target']:
        raise ValueError('Base lock target differs from the intake profile')
    base_pins = {canonicalize_name(row['name']): row['version'] for row in base['wheels']}
    installed = {name: installed_version(name) for name in base_pins}
    if installed != base_pins:
        raise RuntimeError('Main environment versions differ from the base lock')
    if base_pins['torch'] != '2.0.1' or base_pins['numpy'] != '1.25.2':
        raise ValueError('Intake requires torch 2.0.1 and numpy 1.25.2')
    tag_order = {tag: rank for rank, tag in enumerate(sys_tags())}
    environment = default_environment()

    def wheel(files):
        choices = []
        for item in files:
            if item.get('packagetype') != 'bdist_wheel' or item.get('yanked'):
                continue
            if item.get('requires_python') and not SpecifierSet(item['requires_python']).contains(actual_python):
                continue
            _, _, _, tags = parse_wheel_filename(item['filename'])
            ranks = [tag_order[tag] for tag in tags if tag in tag_order]
            if ranks:
                choices.append((min(ranks), item['filename'], item))
        return min(choices, key=lambda row: row[:2])[2] if choices else None

    @dataclass(frozen=True)
    class Candidate:
        name: str
        version: object
        extras: frozenset

    class Provider(AbstractProvider):
        def __init__(self):
            self.documents = {}

        def identify(self, requirement_or_candidate):
            return canonicalize_name(requirement_or_candidate.name)

        def get_preference(self, identifier, resolutions, candidates, information, backtrack_causes):
            return (identifier not in base_pins, identifier)

        def find_matches(self, identifier, requirements, incompatibilities):
            reqs = list(requirements[identifier])
            excluded = set(incompatibilities[identifier])
            extras = frozenset(extra for req in reqs for extra in req.extras)
            exact = [part.version for req in reqs for part in req.specifier
                     if part.operator == '==' and '*' not in part.version]
            if exact:
                versions = {Version(value) for value in exact}
            else:
                project = index.get(identifier)
                self.documents[(identifier, str(Version(project['info']['version'])))] = project
                versions = set()
                for value, files in project['releases'].items():
                    try:
                        version = Version(value)
                    except InvalidVersion:
                        continue
                    if not version.is_prerelease and not version.is_devrelease and wheel(files):
                        versions.add(version)
            def matches():
                for version in sorted(versions, reverse=True):
                    candidate = Candidate(identifier, version, extras)
                    if candidate in excluded or not all(req.specifier.contains(version) for req in reqs):
                        continue
                    key = (identifier, str(version))
                    if key not in self.documents:
                        self.documents[key] = index.get(*key)
                    document = self.documents[key]
                    python = document['info'].get('requires_python')
                    if python and not SpecifierSet(python).contains(actual_python):
                        continue
                    if wheel(document['urls']):
                        yield candidate
            return matches

        def is_satisfied_by(self, requirement, candidate):
            return (self.identify(requirement) == candidate.name
                    and requirement.specifier.contains(candidate.version)
                    and requirement.extras.issubset(candidate.extras))

        def get_dependencies(self, candidate):
            document = self.documents[(candidate.name, str(candidate.version))]
            dependencies = []
            for text in document['info'].get('requires_dist') or []:
                req = Requirement(text)
                if req.url:
                    raise ValueError('Direct-URL dependencies require a separate profile')
                if req.marker is None or any(req.marker.evaluate(dict(environment, extra=extra))
                                              for extra in ('', *sorted(candidate.extras))):
                    dependencies.append(req)
            return dependencies

    requirements = [f'{name}=={version}' for name, version in sorted(base_pins.items())]
    requirements += profile['requirements']
    provider = Provider()
    result = Resolver(provider, BaseReporter()).resolve([Requirement(text) for text in requirements], max_rounds=1000)
    wheels = []
    for name, candidate in sorted(result.mapping.items()):
        document = provider.documents[(name, str(candidate.version))]
        selected = wheel(document['urls'])
        dependencies = provider.get_dependencies(candidate)
        if not all(provider.is_satisfied_by(req, result.mapping[provider.identify(req)]) for req in dependencies):
            raise RuntimeError('Resolved dependency closure is incomplete')
        wheels.append(dict(name=name, version=str(candidate.version), filename=selected['filename'],
            bytes=selected['size'], sha256=selected['digests']['sha256'], url=selected['url'],
            evidence=f"{profile['index']}/{name}/{candidate.version}/json",
            requires_dist=document['info'].get('requires_dist') or [],
            active_requires_dist=[str(req) for req in dependencies], extras=sorted(candidate.extras)))
    return dict(schema_version=1, profile=profile['name'], python=actual_python, target=profile['target'],
        marker_environment=environment, bootstrap=base['bootstrap'], requirements=requirements,
        wheels=wheels, wheel_count=len(wheels), total_download_bytes=sum(row['bytes'] for row in wheels),
        main_versions=installed, metadata_requests=len(index.requests), metadata_bytes=index.bytes,
        wheel_payload_requests=0, installation_approved=False)


def inside(args):
    profile = json.loads(args.profile.read_text())
    base = json.loads((args.profile.parent / profile['base_lock']).read_text())
    prior = json.loads((args.out / 'prior-requests.json').read_text()) if (args.out / 'prior-requests.json').exists() else []
    index = MetadataIndex(profile, args.out, ca_file=args.ca_file, prior_requests=prior)
    result = resolve(profile, base, index)
    write(args.out / 'resolved-lock.json', result)
    return 0


def _limited():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**3, 8 * 1024**3))


def _nix(args, tail, log_name, seconds=600):
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun',
               '--seconds', str(seconds), '--', '-m', 'pt_agent.tools.intake_python_env',
               '--profile', str(args.profile), '--out', str(args.out), *tail]
    with (args.out / log_name).open('w') as log:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                       stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=seconds + 120, preexec_fn=_limited)


def freeze_main(args):
    result = subprocess.run([sys.executable, '-B', '-m', 'pip', '--isolated', 'freeze', '--all'],
                            check=True, capture_output=True, text=True, timeout=60)
    (args.out / args.freeze_name).write_text(result.stdout)


def installed_inside(args):
    profile = json.loads(args.profile.read_text())
    lock = json.loads((args.profile.parent / profile['lock']).read_text())
    venv = PLATFORM_DIR / 'venvs' / profile['name']
    if venv.name != 'intake-2.0.1-cpu' or venv == PLATFORM_DIR / 'venvs/hecate-2.0.1-cpu':
        raise ValueError('Only the independent intake venv may be installed')
    if not os.environ.get('IN_NIX_SHELL') or tuple(sys.version_info[:3]) != (3, 10, 14):
        raise RuntimeError('Installation requires pinned Nix CPython 3.10.14')
    python = str(venv / 'bin/python')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'],
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CONFIG_FILE=os.devnull,
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.inside_install:
        if venv.exists():
            raise FileExistsError('Preserve existing intake environment')
        subprocess.run([sys.executable, '-B', '-m', 'venv', str(venv)], check=True, timeout=120)
        command = [python, '-B', '-m', 'pip', '--isolated', 'install', '--no-index',
                   '--no-cache-dir', '--no-deps', '--only-binary=:all:',
                   '--report', str(args.out / 'pip-install-report.json')]
        command += [str(args.out / 'wheels' / row['filename']) for row in lock['wheels']]
        subprocess.run(command, env=env, check=True, timeout=600)
    subprocess.run([python, '-B', '-m', 'pip', '--isolated', 'check'], env=env, check=True, timeout=60)
    packages = subprocess.run([python, '-B', '-m', 'pip', '--isolated', 'list', '--format=json'],
                              env=env, check=True, capture_output=True, text=True, timeout=60)
    normalize = lambda name: name.lower().replace('_', '-')
    actual = {normalize(row['name']): row['version'] for row in json.loads(packages.stdout)}
    expected = {normalize(row['name']): row['version'] for row in lock['wheels']}
    expected.update({name: lock['bootstrap'][name] for name in ('pip', 'setuptools')})
    if actual != expected:
        raise RuntimeError(f'Intake installed versions differ from plan: {actual}')
    write(args.out / 'installed-packages.json', actual)
    frozen = subprocess.run([python, '-B', '-m', 'pip', '--isolated', 'freeze', '--all'],
                            env=env, check=True, capture_output=True, text=True, timeout=60)
    (args.out / 'intake-pip-freeze.txt').write_text(frozen.stdout)
    probe = subprocess.run([python, '-B', '-c',
        'import json, numpy, torch, transformers, onnx, onnxruntime; '
        'assert torch.__version__ == "2.0.1" and numpy.__version__ == "1.25.2"; '
        'assert torch.version.cuda is None; '
        'print(json.dumps({m.__name__: m.__version__ for m in '
        '(numpy,torch,transformers,onnx,onnxruntime)}))'],
        env=env, capture_output=True, text=True, timeout=120)
    (args.out / 'import-probe.log').write_text(probe.stdout + probe.stderr)
    probe.check_returncode()
    write(args.out / 'verification.json', dict(status='verified', versions=actual,
                                              venv=str(venv), imports=json.loads(probe.stdout)))


def install_or_verify(args):
    from ..registry import append, git_rev, new_run_id

    profile = json.loads(args.profile.read_text())
    lock = json.loads((args.profile.parent / profile['lock']).read_text())
    versions = {row['name']: row['version'] for row in lock['wheels']}
    if (not versions['protobuf'].startswith('4.25.')
            or not versions['huggingface-hub'].startswith('0.24.') or 'hf-xet' in versions):
        raise ValueError('Installation requires the user-approved protobuf/hub pins and no hf-xet')
    if sum(row['bytes'] for row in lock['wheels']) > DOWNLOAD_LIMIT:
        raise ValueError('Plan exceeds the approved 133 MiB plus 10% download allowance')
    args.out = (args.out or pt_result('u1-intake-install-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'profile.json', profile)
    write(args.out / 'resolved-lock.json', lock)
    report = dict(status='failed', api_calls=0, round='U1-1.2', download_limit_bytes=DOWNLOAD_LIMIT,
                  wheel_requests=0, downloaded_bytes=0, installations=0)
    receipts = []
    lock_path = PLATFORM_DIR / 'venvs' / ('.' + profile['name'] + '.lock')
    with lock_path.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _nix(args, ['--inside-freeze', '--freeze-name', 'main-pip-freeze-before.txt'], 'freeze-before.log', 120)
        try:
            if args.install_approved:
                if (PLATFORM_DIR / 'venvs' / profile['name']).exists():
                    raise FileExistsError('Preserve existing intake environment')
                if shutil.disk_usage(PLATFORM_DIR).free < 10 * 1024**3:
                    raise RuntimeError('Installation requires at least 10 GiB free disk')
                cache = args.out / 'wheels'
                cache.mkdir()
                opener = build_opener(NoRedirect(), HTTPSHandler(context=ssl.create_default_context()))
                started = time.monotonic()
                for row in lock['wheels']:
                    if time.monotonic() - started > 1800:
                        raise TimeoutError('Approved wheel acquisition exceeded 1800 seconds')
                    url = urlparse(row['url'])
                    if (url.scheme, url.hostname) != ('https', 'files.pythonhosted.org'):
                        raise ValueError('Wheel URL must use the resolved public PyPI file host')
                    if Path(row['filename']).name != row['filename'] or not row['filename'].endswith('.whl'):
                        raise ValueError('Invalid wheel filename')
                    receipt = dict(name=row['name'], url=row['url'], bytes=0, complete=False)
                    receipts.append(receipt)
                    report['wheel_requests'] += 1
                    write(args.out / 'download-requests.json', receipts)
                    partial = cache / (row['filename'] + '.partial')
                    print('Downloading ' + row['filename'], flush=True)
                    with opener.open(row['url'], timeout=60) as response, partial.open('xb') as output:
                        while True:
                            chunk = response.read(min(1024**2, DOWNLOAD_LIMIT - report['downloaded_bytes'] + 1))
                            if not chunk:
                                break
                            receipt['bytes'] += len(chunk)
                            report['downloaded_bytes'] += len(chunk)
                            output.write(chunk)
                            if report['downloaded_bytes'] > DOWNLOAD_LIMIT:
                                raise RuntimeError('Downloaded payload exceeded approved budget')
                            if time.monotonic() - started > 1800:
                                raise TimeoutError('Approved acquisition wall time exhausted')
                    receipt['complete'] = True
                    partial.rename(cache / row['filename'])
                    write(args.out / 'download-requests.json', receipts)
                _nix(args, ['--inside-install'], 'install.log', 900)
                report['installations'] = 1
            _nix(args, ['--inside-verify'], 'verify.log', 300)
            report.update(status='verified', verification=json.loads((args.out / 'verification.json').read_text()))
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            write(args.out / 'download-requests.json', receipts)
            _nix(args, ['--inside-freeze', '--freeze-name', 'main-pip-freeze-after.txt'], 'freeze-after.log', 120)
            report['main_freeze_equal'] = ((args.out / 'main-pip-freeze-before.txt').read_text()
                                           == (args.out / 'main-pip-freeze-after.txt').read_text())
            if not report['main_freeze_equal']:
                report.update(status='failed', error='Main pip freeze changed during intake installation')
            write(args.out / 'REPORT.json', report)
            os.environ['PT_RESULTS'] = str(pt_result())
            append(dict(run_id=new_run_id('U1', 'intake', 'install' if args.install_approved else 'verify'),
                round='U1', model='intake', mode='environment-install' if args.install_approved else 'environment-verify',
                git_rev=git_rev(), metrics=report, artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT=' + str(args.out / 'REPORT.json'), flush=True)
    return 0 if report['status'] == 'verified' else 1


def plan(args):
    from ..registry import append, git_rev, new_run_id

    profile = json.loads(args.profile.read_text())
    lock_path = args.profile.parent / profile['lock']
    if lock_path.exists():
        raise FileExistsError('Preserve the existing intake lock; a refresh needs a new review')
    out = (args.out or pt_result('u1-intake-plan-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    main_venv = PLATFORM_DIR / 'venvs/hecate-2.0.1-cpu'
    intake_venv = PLATFORM_DIR / 'venvs' / profile['name']
    if intake_venv == main_venv:
        raise ValueError('Intake and main venv paths must differ')
    write(out / 'profile.json', profile)
    prior = []
    if args.prior_plan is not None:
        prior = json.loads((args.prior_plan / 'metadata-requests.json').read_text())
        write(out / 'prior-requests.json', prior)
    paths = ssl.get_default_verify_paths()
    candidates = ([Path(paths.cafile)] if paths.cafile else [])
    candidates += [Path(directory) / 'ca-certificates.crt'
                   for directory in (paths.capath or '').split(os.pathsep) if directory]
    ca_file = next((path for path in candidates if path.is_file()), None)
    if ca_file is None:
        raise RuntimeError('System CA bundle is unavailable; preserve certificate verification')
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(profile['plan_wall_seconds']),
        '--', '-m', 'pt_agent.tools.intake_python_env', '--inside-plan',
        '--profile', str(args.profile), '--out', str(out), '--ca-file', str(ca_file)]
    write(out / 'argv.json', command)
    print('PLAN_DIR=' + str(out), flush=True)
    started = time.monotonic()
    with (out / 'plan.log').open('w') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[2], timeout=profile['plan_wall_seconds'] + 120)
    report = dict(round='U1-1.1', complete=False, status='failed', exit_code=completed.returncode,
        seconds=time.monotonic() - started, profile_path=str(args.profile), lock_path=str(lock_path),
        main_venv=str(main_venv), intake_venv=str(intake_venv),
        intake_venv_exists=intake_venv.exists(), installations=0, wheel_payload_requests=0, api_calls=0,
        prior_plan=str(args.prior_plan) if args.prior_plan else None, prior_metadata_requests=len(prior),
        ca_file=str(ca_file))
    if completed.returncode == 0:
        resolved = json.loads((out / 'resolved-lock.json').read_text())
        with lock_path.open('x') as stream:
            stream.write(json.dumps(resolved, indent=2) + '\n')
        report.update(complete=True, status='awaiting-user-approval', wheel_count=resolved['wheel_count'],
            total_download_bytes=resolved['total_download_bytes'], metadata_requests=resolved['metadata_requests'],
            metadata_bytes=resolved['metadata_bytes'])
    write(out / 'REPORT.json', report)
    os.environ['PT_RESULTS'] = str(pt_result())
    append(dict(run_id=new_run_id('U1', 'intake', 'plan'), round='U1', model='intake', mode='environment-plan',
        git_rev=git_rev(), metrics=report, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT=' + str(out / 'REPORT.json'), flush=True)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--inside-plan', action='store_true')
    mode.add_argument('--install-approved', action='store_true')
    mode.add_argument('--verify-existing', action='store_true')
    mode.add_argument('--inside-install', action='store_true')
    mode.add_argument('--inside-verify', action='store_true')
    mode.add_argument('--inside-freeze', action='store_true')
    parser.add_argument('--profile', type=Path, default=PROFILE)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--ca-file', type=Path, help='Inside only: existing system CA bundle, passed read-only')
    parser.add_argument('--prior-plan', type=Path, help='Retain and count metadata attempts from a failed plan')
    parser.add_argument('--freeze-name', choices=('main-pip-freeze-before.txt', 'main-pip-freeze-after.txt'))
    args = parser.parse_args()
    args.profile = args.profile.resolve()
    if args.inside_plan:
        if args.out is None or args.ca_file is None:
            parser.error('--inside-plan requires explicit output and existing CA paths')
        return inside(args)
    if args.inside_freeze:
        if args.out is None or args.freeze_name is None:
            parser.error('--inside-freeze requires output and freeze filename')
        freeze_main(args)
        return 0
    if args.inside_install or args.inside_verify:
        if args.out is None:
            parser.error('Inside install/verify requires explicit output')
        installed_inside(args)
        return 0
    if args.install_approved or args.verify_existing:
        return install_or_verify(args)
    return plan(args)


if __name__ == '__main__':
    sys.exit(main())
