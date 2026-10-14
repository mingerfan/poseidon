"""Finite developer qualification with an explicit scope and executable wrapper.

Driver(scope, entry_script) retains the original common acceptance isolation and
resource gates. Each child re-enters entry_script, which must construct the same
reviewed scope. Optional scope hooks select public source descriptors and rule
expressions only; this driver does not rewrite Provider candidates.
"""
import argparse, fcntl, json, os, shlex, signal, subprocess, sys, time
from pathlib import Path

class Driver:

    def __init__(self, scope, entry_script):
        self.scope = scope
        self.HERE = scope.HERE
        self.ROOT = scope.ROOT
        self.BASE = scope.BASE
        self.WORK = scope.WORK
        self.RESULTS = scope.RESULTS
        self.entry_script = Path(entry_script)
        scope.require(self.entry_script.is_absolute() and self.entry_script == self.entry_script.resolve() and self.entry_script.is_file() and (not self.entry_script.is_symlink()), 'Explicit regular driver entry script')

    def _before_launch(self):
        before = getattr(self.scope, 'before_launch', None)
        if before is not None:
            self.scope.require(callable(before), 'Callable scope launch guard')
            before()

    def candidate_origin(self, plan=None):
        origin = getattr(self.scope, 'CANDIDATE_ORIGIN', getattr(self.scope, 'candidate_origin', None))
        if origin is None:
            origin = (plan or {}).get('candidate_origin', 'rule_baseline')
        self.scope.require(isinstance(origin, str) and bool(origin), 'Declared candidate origin')
        return origin

    def allowed_env(self):
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        names = ('PATH', 'IN_NIX_SHELL', 'POSEIDON_PLATFORM', 'POSEIDON_WORK_ROOT', 'HECATE_PYTHON_LIBRARY_PATH', 'LD_LIBRARY_PATH', 'LANG')
        env = {n: os.environ[n] for n in names if n in os.environ}
        from platform_config import require_platform
        profile = require_platform()
        env.update(POSEIDON_WORK_ROOT=str(WORK.parents[1]), POSEIDON_PLATFORM=profile['id'], PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONPATH=str(BASE), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HOME='/nonexistent')
        return env

    def unstarted_attempt(self, previous):
        """Only explicitly observed pre-worker failures can retain the original budget."""
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        failure = scope.read(previous / 'failure-before-sdk.json')
        scope.require(failure.get('bound_plan_sha256') == scope.sha(previous / 'plan.json'), 'Original failed plan observation')
        counts = ('compiler_calls', 'encrypted_executions', 'provider_calls', 'http_attempts')
        scope.require(all((type(failure.get(n)) is int and failure[n] == 0 for n in counts)), 'No completed or unknown backend execution to repeat')
        namespace_log = previous / 'namespace.log'
        allowed = {'plan.json', 'failure-before-sdk.json', 'tool-snapshot'}
        zero_case_boundary = False
        if failure.get('code') == 'namespace_work_root_filesystem':
            extras = {'namespace.log', 'report.json', 'namespace-receipt.json', 'staged-sources.json'}
            scope.require(set(failure.get('files', {})) == extras and all(((previous / n).is_file() and (not (previous / n).is_symlink()) and (scope.sha(previous / n) == h) for (n, h) in failure['files'].items())), 'Exact pre-case failure files')
            prior = scope.read(previous / 'plan.json')
            report = scope.read(previous / 'report.json')
            receipt = scope.read(previous / 'namespace-receipt.json')
            zero_case_boundary = failure.get('nix_entered') is True and failure.get('backend_worker_started') is True and (failure.get('phase_workers_started') == 0) and (report['rows'] == []) and (report['compilations'] == report['completed_phases'] == 0) and (report['pending'] == [dict(case=n, phase=p, status='not_run') for n in scope.CASES for p in scope.PHASES]) and (report['binding'] == prior['binding']) and (receipt['actual_nix_shell'] == 'pure') and (receipt['project_credential_path_absent'] is True) and (receipt['production_applied'] is False) and (receipt['staged_modules'] == len(scope.read(HERE / 'plan.json')['targets'])) and (scope.read(previous / 'staged-sources.json') == prior['bindings']['inside_sources']) and namespace_log.read_text().endswith('ValueError: Offline disk reserve\n')
            scope.require(zero_case_boundary, 'Exact zero-case checkpoint')
            allowed.update(extras)
        if namespace_log.exists():
            known_rejection = namespace_log.is_file() and failure.get('namespace_log_sha256') == scope.sha(namespace_log) and (failure.get('code') == 'namespace_argument_count_exceeded' and namespace_log.read_bytes() == b'bwrap: Exceeded maximum number of arguments 9000\n' or (failure.get('code') == 'namespace_command_in_option_fd' and scope.sha(namespace_log) == 'ea34153e8c450472668997677f571603e955212b806e2b2bc724848ecc3321af'))
            scope.require(namespace_log.is_file() and (not namespace_log.is_symlink()) and (namespace_log.stat().st_size == 0 or known_rejection or zero_case_boundary), 'No observed namespace execution')
            allowed.add('namespace.log')
        scope.require(set((p.name for p in previous.iterdir())) == allowed and failure.get('exit_code') == 1, 'Only pre-worker artifacts')
        missing_platform = failure.get('code') == 'explicit_platform_missing' and failure.get('nix_entered') is False and (not namespace_log.exists())
        argv_overflow = failure.get('code') == 'namespace_argv_e2big' and failure.get('nix_entered') is True and (failure.get('backend_worker_started') is False) and namespace_log.exists()
        argument_rejection = failure.get('code') in ('namespace_argument_count_exceeded', 'namespace_command_in_option_fd') and failure.get('nix_entered') is True and (failure.get('backend_worker_started') is False) and namespace_log.exists()
        scope.require(missing_platform or argv_overflow or argument_rejection or zero_case_boundary, 'Explicit unstarted failure required')
        return failure

    def spawn_namespace(self, arguments, log):
        """Send the exact public argument vector through bwrap's documented --args FD."""
        self._before_launch()
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        scope.require(arguments and arguments[0] == '/usr/bin/bwrap', 'Pinned sandbox executable')
        encoded = [os.fsencode(argument) for argument in arguments[1:]]
        scope.require(all((b'\x00' not in argument for argument in encoded)), 'NUL in namespace argument')
        separator = arguments.index('--') if '--' in arguments else len(arguments)
        options = encoded[:separator - 1]
        command = arguments[separator:]
        scope.require(not command or len(command) > 1, 'Namespace command required')
        descriptor = os.memfd_create('poseidon-bwrap-arguments', os.MFD_CLOEXEC)
        try:
            with os.fdopen(os.dup(descriptor), 'wb') as stream:
                stream.write(b'\x00'.join(options) + b'\x00')
            os.lseek(descriptor, 0, os.SEEK_SET)
            return subprocess.Popen([arguments[0], '--args', str(descriptor), *command], pass_fds=(descriptor,), env={'PATH': '/usr/bin:/bin'}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            os.close(descriptor)

    def outer(self, output, continuation=None):
        self._before_launch()
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        scope.require(continuation is None or getattr(scope, 'ALLOW_CONTINUATION', True), 'Scope forbids budget continuation')
        if hasattr(scope, 'validate_output'):
            scope.validate_output(output)
        plan = scope.plan()
        sys.path.insert(0, str(BASE))
        sys.path.insert(0, str(HERE.parent / 'staged_component_cutover_v1'))
        from guarded_apply_joint import existing_guard
        scope.require(not existing_guard.active_processes(), 'Existing live/candidate process')
        with (WORK / 'deps/nix-portable-v012/launcher.lock').open('r+') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        output = Path(output)
        scope.require(output.is_absolute() and output == output.resolve() and (output.parent == RESULTS) and (not output.exists()), 'Fresh offline result directory')
        scope.require(scope.measure(RESULTS) <= scope.SCOPE['global_bytes'], 'Existing global artifact boundary')
        import shutil
        scope.require(shutil.disk_usage(WORK).free >= scope.SCOPE['reserve_bytes'], 'Existing disk reserve')
        plan.pop('binding')
        started_wall = time.time()
        plan.update(deadline_wall=started_wall + scope.SCOPE['wall_seconds'], initial_result_directories=sorted((p.name for p in RESULTS.iterdir())), readonly_result_entries=sorted((p.name for p in RESULTS.iterdir())))
        if getattr(scope, 'UNIQUE_PHASES', False):
            plan['started_wall'] = started_wall
        if continuation is not None:
            previous = Path(continuation)
            scope.require(previous == previous.resolve() and previous.parent == RESULTS and previous.is_dir(), 'Original finite budget directory')
            prior = scope.read(previous / 'plan.json')
            failure = scope.read(previous / 'failure-before-sdk.json')
            scope.require(prior['binding'] == __import__('hashlib').sha256(scope.canonical({k: v for (k, v) in prior.items() if k != 'binding'})).hexdigest(), 'Original finite plan integrity')
            scope.require(prior['scope'] == scope.SCOPE and prior['cases'] == list(scope.CASES) and (prior['phases'] == list(scope.PHASES)), 'Same original cases and caps')
            self.unstarted_attempt(previous)
            revised = {str(Path(getattr(scope, '__file__', HERE / 'runtime_scope.py')).resolve()), str(self.entry_script), str(Path(__file__).resolve())}
            for (path, want) in prior['bindings']['files'].items():
                frozen = previous / 'tool-snapshot' / Path(path).name if path in revised else Path(path)
                scope.require(scope.sha(frozen) == want, 'Original unchanged fixture/tool snapshot')
            scope.require(prior['bindings']['original_production'] == plan['bindings']['original_production'] and prior['bindings']['inside_sources'] == plan['bindings']['inside_sources'] and (prior['bindings']['source_reference'] == plan['bindings']['source_reference']), 'Same original source and independent reference')
            scope.require(time.time() < prior['deadline_wall'], 'Original finite budget expired')
            plan.update(deadline_wall=prior['deadline_wall'], initial_result_directories=prior['initial_result_directories'], continued_budget=dict(root=str(previous), plan_sha256=scope.sha(previous / 'plan.json'), failure_sha256=scope.sha(previous / 'failure-before-sdk.json')))
        plan['binding'] = __import__('hashlib').sha256(scope.canonical(plan)).hexdigest()
        output.mkdir(mode=448)
        scope.write(output / 'plan.json', plan)
        from hecate_python_env import enter_nix, VENV
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ' + shlex.join([str(VENV / 'bin/python'), '-B', str(self.entry_script), '--namespace', str(output)])
        code = enter_nix(command, seconds=max(1, int(plan['deadline_wall'] - time.time())), keep_env=())
        scope.source_binding()
        return code

    def namespace(self, output):
        self._before_launch()
        if hasattr(self.scope, 'validate_runtime_output'):
            self.scope.validate_runtime_output(output)
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        output = Path(output)
        scope.resource_boundary(output, scope.read(output / 'plan.json'))
        sys.path.insert(0, str(BASE))
        from hecate_python_env import VENV
        scope.require(os.environ.get('IN_NIX_SHELL') == 'pure' and Path(sys.prefix) == VENV, 'Actual pinned Nix worker')
        output = Path(output)
        plan = scope.read(output / 'plan.json')
        scope.check_plan(plan)
        cmd = ['/usr/bin/bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL', '--clearenv']
        for name in ('/usr', '/bin', '/sbin', '/lib', '/lib64', '/nix/store'):
            if Path(name).exists():
                cmd += ['--ro-bind', name, name]
        for name in ('/etc/ld.so.cache', '/etc/localtime', '/etc/nsswitch.conf', '/etc/passwd', '/etc/group'):
            if Path(name).exists():
                cmd += ['--ro-bind', name, name]
        cmd += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--ro-bind', '/sys/devices/system/cpu', '/sys/devices/system/cpu', '--dir', str(ROOT)]
        for name in ('scripts', 'src', 'include', 'cmake', 'third_party', 'docs', '.agents', '.git'):
            p = ROOT / name
            if p.exists():
                cmd += ['--ro-bind', str(p), str(p)]
        for name in ('CMakeLists.txt', 'README.md', 'LICENSE', 'AGENTS.md', '.gitmodules', '.gitignore'):
            p = ROOT / name
            if p.is_file():
                cmd += ['--ro-bind', str(p), str(p)]
        cmd += ['--tmpfs', str(BASE)]
        targets = scope.read(HERE / 'plan.json')['targets']
        for p in sorted(BASE.iterdir()):
            selected = HERE / 'targets' / p.name if p.name in targets else p
            cmd += ['--ro-bind', str(selected), str(p)]
        for name in targets:
            if not (BASE / name).exists():
                cmd += ['--ro-bind', str(HERE / 'targets' / name), str(BASE / name)]
        cmd += ['--remount-ro', str(BASE)]
        cmd += ['--ro-bind', str(WORK), str(WORK)]
        for name in ('deps', 'cache', 'build-dacapo', 'build-poseidon', 'venvs'):
            p = WORK / name
            if p.exists():
                cmd += ['--ro-bind', str(p), str(p)]
        locks = WORK / 'cache/agent-native-slots'
        scope.require(locks.is_dir() and locks == locks.resolve(), 'Existing native lock directory')
        cmd += ['--bind', str(locks), str(locks)]
        cmd += ['--bind', str(RESULTS), str(RESULTS)]
        for (key, value) in self.allowed_env().items():
            cmd += ['--setenv', key, value]
        cmd = self.mount_page(cmd, plan['readonly_result_entries'], 0, output, str(VENV / 'bin/python'))
        with (output / 'namespace.log').open('xb') as log:
            return self.wait_namespace(cmd, log, plan)

    def mount_page(self, arguments, entries, start, output, python_executable):
        """Partition option setup; every old result receives one read-only bind."""
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        scope.require(type(start) is int and 0 <= start <= len(entries), 'Readonly mount offset')
        scope.require(len(set(entries)) == len(entries) and all((isinstance(n, str) and n not in ('', '.', '..') and (Path(n).name == n) for n in entries)), 'Readonly evidence inventory')
        prefix = ['--chdir', str(ROOT), '--', python_executable, '-B', str(self.entry_script)]
        capacity = (8500 - len(arguments) - len(prefix) - 3) // 3
        scope.require(capacity > 0, 'Finite mount page capacity')
        end = min(start + capacity, len(entries))
        result = list(arguments)
        for name in entries[start:end]:
            path = str(RESULTS / name)
            result += ['--ro-bind', path, path]
        tail = ['--inside', str(output)] if end == len(entries) else ['--mount-layer', str(output), str(end)]
        result += prefix + tail
        scope.require(len(result) <= 8500, 'Namespace option bound')
        return result

    def stop_worker(self, child):
        """Stop the owned worker and descendants even across new sessions."""
        identities = {}
        for item in Path('/proc').iterdir():
            if not item.name.isdigit():
                continue
            try:
                fields = (item / 'stat').read_text().rsplit(')', 1)[1].split()
                identities[int(item.name)] = (int(fields[1]), fields[19])
            except (OSError, ValueError, IndexError):
                continue
        selected = {child.pid}
        while True:
            descendants = {pid for pid, (parent, _) in identities.items() if parent in selected}
            if descendants <= selected:
                break
            selected |= descendants
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in sorted(selected, reverse=True):
                try:
                    current = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
                    if pid in identities and current[19] == identities[pid][1]:
                        os.kill(pid, sig)
                except (OSError, IndexError):
                    pass
            if sig == signal.SIGTERM:
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        child.wait(timeout=5)

    def wait_phase(self, child, output, plan):
        """Keep the original phase and batch caps active during execution."""
        deadline = min(time.monotonic() + self.scope.SCOPE['phase_seconds'],
                       time.monotonic() + max(0, plan['deadline_wall'] - time.time()))
        try:
            while child.poll() is None:
                self.scope.resource_boundary(output, plan)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(child.args, self.scope.SCOPE['phase_seconds'])
                try:
                    return child.wait(timeout=min(3, remaining))
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
            return child.returncode
        except BaseException:
            self.stop_worker(child)
            raise

    def wait_namespace(self, arguments, log, plan):
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        child = self.spawn_namespace(arguments, log)
        try:
            code = child.wait(timeout=max(1, int(plan['deadline_wall'] - time.time())))
        except subprocess.TimeoutExpired:
            self.stop_worker(child)
            code = 124
        scope.check_plan(plan)
        return code

    def mount_layer(self, output, start):
        self._before_launch()
        if hasattr(self.scope, 'validate_runtime_output'):
            self.scope.validate_runtime_output(output)
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        output = Path(output)
        scope.resource_boundary(output, scope.read(output / 'plan.json'))
        sys.path.insert(0, str(BASE))
        from hecate_python_env import VENV
        scope.require(os.environ.get('IN_NIX_SHELL') == 'pure' and Path(sys.prefix) == VENV, 'Actual mount-layer SDK')
        scope.require(not (ROOT / '.env').exists(), 'Credential path excluded by initial namespace')
        output = Path(output)
        plan = scope.read(output / 'plan.json')
        scope.check_plan(plan)
        scope.require(time.time() < plan['deadline_wall'], 'Original namespace deadline')
        cmd = ['/usr/bin/bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL', '--clearenv', '--bind', '/', '/', '--proc', '/proc', '--dev', '/dev']
        for (key, value) in self.allowed_env().items():
            cmd += ['--setenv', key, value]
        cmd = self.mount_page(cmd, plan['readonly_result_entries'], int(start), output, str(VENV / 'bin/python'))
        return self.wait_namespace(cmd, sys.stdout, plan)

    def boundary(self, output, plan):
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        scope.expected_namespace(plan)
        scope.resource_boundary(output, plan)

    def inside(self, output):
        self._before_launch()
        if hasattr(self.scope, 'validate_runtime_output'):
            self.scope.validate_runtime_output(output)
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        output = Path(output)
        scope.resource_boundary(output, scope.read(output / 'plan.json'))
        sys.path.insert(0, str(BASE))
        from hecate_python_env import VENV
        scope.require(os.environ.get('IN_NIX_SHELL') == 'pure' and Path(sys.prefix) == VENV, 'Pinned namespace')
        output = Path(output)
        plan = scope.read(output / 'plan.json')
        expected = scope.expected_namespace(plan)
        scope.require(not (ROOT / '.env').exists(), 'Project credential file excluded')
        from semantic_benchmark_execution import runtime_sources
        actual = runtime_sources()
        scope.require(actual == expected, 'Complete staged runtime source inventory')
        scope.write(output / 'staged-sources.json', actual)
        scope.write(output / 'namespace-receipt.json', dict(format='poseidon-ordered-common-namespace-v1', project_credential_path_absent=True, network_namespace=str(Path('/proc/self/ns/net').readlink()), actual_nix_shell=os.environ['IN_NIX_SHELL'], staged_modules=len(scope.read(HERE / 'plan.json')['targets']), production_applied=False, provider_calls=0))
        rows = []
        started = time.monotonic()

        def save(status):
            completed = {(r['case'], r['phase']) for r in rows}
            pending = [dict(case=n, phase=p, status='not_run') for n in scope.CASES for p in scope.PHASES if (n, p) not in completed]
            data = dict(format='poseidon-ordered-common-offline-report-v1', binding=plan['binding'], status=status, rows=rows, pending=pending, seconds=time.monotonic() - started, new_agent_generation=False, candidate_origin=self.candidate_origin(plan), provider_calls=0, http_attempts=0, production_applied=False, compilations=sum((r.get('compiled') is True for r in rows)), completed_phases=sum((r['status'] == 'passed' for r in rows)))
            (output / 'report.json').write_bytes(scope.canonical(data))
        save('running')
        for name in scope.CASES:
            for phase in scope.PHASES:
                if phase != 'qualification' and (not any((r['case'] == name and r['phase'] == 'qualification' and (r['status'] == 'passed') for r in rows))):
                    continue
                self.boundary(output, plan)
                scope.require(runtime_sources() == actual, 'Staged source binding changed')
                destination = output / name
                destination.mkdir(exist_ok=True)
                report = destination / (phase + '.json')
                scope.require(not report.exists(), 'Fresh phase receipt')
                command = [sys.executable, '-B', str(self.entry_script), '--case', str(output), name, phase]
                with (destination / (phase + '.log')).open('xb') as stream:
                    child = subprocess.Popen(command, env=self.allowed_env(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    aborted = None
                    try:
                        code = self.wait_phase(child, output, plan)
                    except subprocess.TimeoutExpired:
                        code, aborted = 124, 'phase_timeout'
                    except ValueError as error:
                        code, aborted = 1, str(error)[:300]
                row = scope.read(report) if report.exists() else dict(case=name, phase=phase, status='failed', failure_layer='environment', code='missing_phase_receipt')
                if row.get('case') != name or row.get('phase') != phase:
                    row.update(status='failed', failure_layer='integrity', code='phase_receipt_identity')
                if aborted is not None:
                    row.update(status='failed', failure_layer='resources', code=aborted)
                row['exit_code'] = code
                if code != 0:
                    row['status'] = 'failed'
                rows.append(row)
                save('running')
                if row.get('failure_layer') in ('integrity', 'environment', 'resources'):
                    save('stopped')
                    return 1
                self.boundary(output, plan)
        scope.require(runtime_sources() == actual, 'Final staged source binding changed')
        status = 'passed' if len(rows) == len(scope.CASES) * len(scope.PHASES) and all((r['status'] == 'passed' for r in rows)) else 'failed'
        save(status)
        print(json.dumps(dict(status=status, evidence=str(output), completed=len(rows))), flush=True)
        return 0 if status == 'passed' else 1

    def reserve_phase(self, directory, name, phase, plan):
        scope = self.scope
        if getattr(scope, 'UNIQUE_PHASES', False):
            scope.require(directory.is_dir() and directory == directory.resolve(), 'Reviewed phase directory')
            scope.require(not (directory / (phase + '.json')).exists(), 'Fresh phase receipt')
            scope.write(directory / (phase + '-started.json'), dict(
                format='poseidon-offline-phase-start-v1', binding=plan['binding'],
                case=name, phase=phase, started_wall=time.time()))

    def one(self, output, name, phase):
        self._before_launch()
        if hasattr(self.scope, 'validate_runtime_output'):
            self.scope.validate_runtime_output(output)
        scope = self.scope
        HERE = self.HERE
        ROOT = self.ROOT
        BASE = self.BASE
        WORK = self.WORK
        RESULTS = self.RESULTS
        scope.require(name in scope.CASES and phase in scope.PHASES, 'Reviewed case and phase')
        output = Path(output)
        plan = scope.read(output / 'plan.json')
        self.boundary(output, plan)
        sys.path.insert(0, str(BASE))
        import numpy as np
        from application_execution.context import WorkerContext
        from application_execution.backend import application
        from application_model_source import FORMAT
        from hecate_python_env import VENV
        from candidate_bundle import check_bundle
        scope.require(os.environ.get('IN_NIX_SHELL') == 'pure' and Path(sys.prefix) == VENV, 'Pinned case worker')
        scope.require(name in scope.CASES and phase in scope.PHASES, 'Reviewed case and phase')
        output = Path(output)
        plan = scope.read(output / 'plan.json')
        self.boundary(output, plan)
        directory = output / name
        self.reserve_phase(directory, name, phase, plan)
        context = WorkerContext.capture(directory / 'tasks')
        app = application(context)
        source = scope.case_source(name) if hasattr(scope, 'case_source') else dict(format=FORMAT, kind='trusted_pytorch', manifest=str(HERE / 'fixtures' / name / 'manifest.json'), trust_local_code=True)
        row = dict(case=name, phase=phase, status='failed', provider_calls=0, new_agent_generation=False, candidate_origin=self.candidate_origin(plan))
        try:
            if phase == 'qualification':
                options = scope.options_for_case(name) if hasattr(scope, 'options_for_case') else dict(chunk_period=4, generation_guidance='explicit-v11', capacity_packing_version='capacity-public-tensors-v9', public_weight_policy='descriptors-v1')
                limits = dict(generations=1, http_attempts=1, wall_seconds=170, api_timeout_seconds=1200, max_tokens=384000)
                task = app.submit_source(source, resource_profile=plan['resource_profile'], options=options, limits=limits)
                state = app.prepare_source(task)
                scope.require(state['status'] == 'queued', 'Actual source capture/admission')
                meta = app._checked_task_metadata(task)
                q = meta['request']
                expected = plan['bindings']['source_reference'][name]
                from benchmark_graph import digest
                scope.require(digest(q['model']) == expected['model_sha256'] and digest(meta['model_provenance']['original']) == expected['original_sha256'], 'Exact independent-reference model')
                if hasattr(scope, 'candidate_source'):
                    selected = scope.candidate_source(q)
                else:
                    from unified_graph_lowering import candidate_source
                    selected = candidate_source(q)
                program = selected[0] if isinstance(selected, tuple) else selected
                scope.require(isinstance(program, str), 'Rule source text required')
                if hasattr(scope, 'validate_candidate'):
                    scope.validate_candidate(name, q, program, plan)
                candidate_sha = __import__('hashlib').sha256(program.encode()).hexdigest()
                scope.write(directory / 'qualification-binding.json', dict(request=q, request_sha256=digest(q), candidate_sha256=candidate_sha, identity=meta['identity'], model_source_sha256=expected['source_package_sha256'], candidate_origin=self.candidate_origin(plan), agent_generated=False))

                class RuleReplay:
                    kind = 'scripted_replay'
                    agent_calls = 0

                    def generate(self, request, feedback=None):
                        scope.require(feedback == [] and request == q, 'One fixed rule expression')
                        return json.dumps(dict(schema=1, request_id=q['request_id'], hecate_source=program))
                result = app.run(task, RuleReplay())
                scope.write(directory / 'task-id.json', dict(task_id=task))
                row.update(result=result, source_capture_supervisor_executed=True, request_id=q['request_id'], candidate_sha256=candidate_sha, compiled=bool((result.get('acceptance') or {}).get('compiled')), encrypted_execution=bool((result.get('acceptance') or {}).get('encrypted_execution')))
                if result['status'] != 'succeeded':
                    failure = result.get('failure') or {}
                    row['failure_layer'] = failure.get('layer', 'case_acceptance')
                    row['failure_reason'] = failure.get('reason_code')
                    if row['failure_reason'] == 'environment_failure':
                        row['failure_layer'] = 'environment'
                scope.require(result['status'] == 'succeeded' and result['acceptance']['numerically_validated'], 'Real numerical qualification')
                scope.require(result['generation']['agent_generated'] is False and result['http_attempts'] == 0, 'Rule provenance')
                package = app._dir(task) / 'program'
                manifest = check_bundle(package)
                scope.require(manifest['validation']['level'] == 'numerical', 'Numerical package')
                row.update(status='passed', package=str(package), package_binding=manifest['binding'], input_groups=result['acceptance']['input_groups'], compared_values=result['acceptance']['compared_values'], maximum_absolute_error=result['acceptance']['maximum_absolute_error'])
            else:
                task = scope.read(directory / 'task-id.json')['task_id']
                meta = app._checked_task_metadata(task)
                package = app._dir(task) / 'program'
                before = check_bundle(package)
                model = meta['request']['model']
                if phase == 'new_inputs':
                    from benchmark_math import evaluate as math_ref
                    from benchmark_torch import evaluate as torch_ref
                    from numerical_results import compare
                    samples = []
                    reference = []
                    for i in range(3):
                        rng = np.random.default_rng(20261007 + i)
                        values = {s['name']: rng.uniform(-0.2, 0.2, s['shape']) for s in model['inputs']}
                        (a, b) = (math_ref(model, values), torch_ref(model, values))
                        for key in a:
                            np.testing.assert_allclose(a[key], b[key], atol=2e-12, rtol=2e-12)
                        samples.append({k: v.tolist() for (k, v) in values.items()})
                        reference.append(np.concatenate([a[o['name']].reshape(-1) for o in model['outputs']]))
                    result = app.execute(task, samples)
                    actual = np.asarray([np.concatenate([np.asarray(s[o['name']]).reshape(-1) for o in model['outputs']]) for s in result['outputs']])
                    comparison = compare(actual, np.asarray(reference), 1e-05, 0.0001)
                    scope.require(result['encrypted_execution'] is True and result['executor_secret_key'] is False and (result['agent_calls'] == result['compiler_calls'] == 0), 'Real package execution boundary')
                    scope.require(comparison['passed'] and before == check_bundle(package), 'Frozen numerical/package tolerance')
                    row.update(status='passed', execution=result, comparison=comparison, input_groups=3, compiled=False, encrypted_execution=True, package_reloaded=True)
                else:
                    origin = scope.read(app._dir(task) / 'source.json')
                    again = app.submit_source(source, resource_profile=plan['resource_profile'], options=origin['requirements']['options'], limits=origin['budget'])

                    class NeverGenerate:
                        kind = 'scripted_replay'
                        agent_calls = 0

                        def generate(self, *args):
                            raise AssertionError('Cached package required')
                    result = app.run(again, NeverGenerate())
                    scope.require(result['status'] == 'succeeded' and result['package']['reused'] is True and (result['package']['binding'] == before['binding']) and (result['generations'] == result['http_attempts'] == 0), 'Reusable source package')
                    row.update(status='passed', result=result, package_reused=True, compiled=False, encrypted_execution=False)
        except Exception as error:
            row.update(error_type=type(error).__name__, diagnostic=str(error)[:300])
            row.setdefault('failure_layer', 'case_acceptance')
        scope.write(directory / (phase + '.json'), row)
        return 0 if row['status'] == 'passed' else 1

    def main(self, argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--plan', action='store_true')
        group.add_argument('--execute', type=Path)
        parser.add_argument('--continue-budget', type=Path)
        group.add_argument('--namespace', type=Path, help=argparse.SUPPRESS)
        group.add_argument('--inside', type=Path, help=argparse.SUPPRESS)
        group.add_argument('--mount-layer', nargs=2, help=argparse.SUPPRESS)
        group.add_argument('--case', nargs=3, help=argparse.SUPPRESS)
        args = parser.parse_args(argv)
        if args.plan:
            print(json.dumps(self.scope.plan(), sort_keys=True, indent=2))
            return 0
        if args.execute:
            return self.outer(args.execute, args.continue_budget)
        if args.namespace:
            return self.namespace(args.namespace)
        if args.inside:
            return self.inside(args.inside)
        if args.mount_layer:
            return self.mount_layer(*args.mount_layer)
        return self.one(*args.case)
