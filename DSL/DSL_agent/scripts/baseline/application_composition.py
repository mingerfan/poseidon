"""Explicit composition task entry over normal shared admission and CPU context.

Composition never enters synthesize/prepare_task or invents a parent generation
receipt. New qualification is owned by the parent and leaves child history intact.
The existing application factory and legacy bundle formats are unchanged.
"""
import copy
from pathlib import Path
import signal
import uuid

from application_component import Controller, Stopped, TERMINAL
from application_shared import SharedApplication, input_binding
from application_execution.backend import ContextBackend
from application_artifact_budget import OperationalStopped, ArtifactMonitor
import cpu_composition_package as package

need, digest = package.need, package.digest


def is_composition(folder):
    return package.loads(package.regular(Path(folder) / 'manifest.json', 1024**2)).get('format') == package.FORMAT


class CompositionBackend(ContextBackend):
    def check_package(self, folder, level):
        if not is_composition(folder):
            return super().check_package(folder, level)
        self.context.check_current()
        manifest, _ = package.check_bundle(folder, level=level)
        self.context.check_current()
        return manifest['binding']

    def execute(self, folder, samples, *, artifact_observer=None, sample_format='named'):
        if not is_composition(folder):
            return super().execute(folder, samples, artifact_observer=artifact_observer, sample_format=sample_format)
        need(sample_format == 'named', 'composition requires named tensor samples')
        from cpu_composition_execution import execute_prepared
        from hecate_python_env import WORK
        self.context.check_current()
        manifest, request = package.check_execution_dependencies(folder)
        context = composition_context(self.context, request)
        def check():
            need(composition_context(self.context, request) == context, 'composition execution context changed')
        report = execute_prepared(folder, samples,
            evidence=WORK / 'results' / ('poseidon-program-execution-' + uuid.uuid4().hex),
            checkpoint=check, artifact_observer=artifact_observer)
        check()
        report['run_context'] = context
        return report


def composition_context(worker, request):
    worker.check_current()
    need(request['authority'] == package.authority(), 'composition current source context')
    return package.seal(dict(format='poseidon-composition-request-context-v1', worker_binding=worker.binding,
        request_binding=request['binding'], reuse_key=package.reuse_key(request),
        authority_sha256=digest(request['authority']), profile=package.PROFILE))


class CompositionApplication(SharedApplication):
    def _operation(self, action, binding, body, **options):
        authority = package.authority()
        def checked():
            from cpu_composition_verified import current_operation, VerificationOperation
            need(package.authority() == authority, 'composition source changed before operation')
            operation = current_operation()
            if operation is not None:
                need(operation.worker is self.backend.context and operation.scope_owner is self,
                     'composition verification application/worker owner')
                return body()
            class FailedOperationResult(Exception):
                def __init__(self, result): self.result = result
            try:
                need(self._shared_active is not None, 'shared composition verification scope required')
                with VerificationOperation(self.backend.context,
                                           checkpoint=self._shared_active.batch.checkpoint,
                                           scope_owner=self) as operation:
                    result = body()
                    if operation.invalid:
                        need(type(result) is dict and result.get('status') not in
                             {'succeeded', 'executed', 'passed', 'queued'}, 'invalid successful verification operation')
                        raise FailedOperationResult(result)
                    need(package.authority() == authority, 'composition source changed after operation')
                    return result
            except FailedOperationResult as failure:
                return failure.result
        return super()._operation(action, dict(binding, composition_authority_sha256=digest(authority)),
                                  checked, **options)

    def submit_composition(self, parent_raw, *, source_node_id, children):
        """Trusted host-only inputs; normal shared 4GiB reservation precedes writes."""
        entries = copy.deepcopy(children)
        need(type(entries) is list, 'composition child entries')
        for row in entries:
            need(type(row) is dict and 'folder' in row, 'composition entry folder')
            row['folder'] = str(row['folder'])
        binding = dict(task_store=str(self.store), package_type=package.FORMAT,
            parent_sha256=package.sha(parent_raw), source_node_id=source_node_id,
            children=input_binding(entries))
        return self._operation('application-submit', binding,
            lambda: self._submit_composition(parent_raw, source_node_id, entries))

    def _submit_composition(self, parent_raw, source_node_id, entries):
        data = package.derive(parent_raw, source_node_id, entries)
        request = data['request']
        context = composition_context(self.backend.context, request)
        task = uuid.uuid4().hex; path = self._dir(task); path.mkdir(mode=0o700)
        self._task_created(path)
        # These original Controller ceilings are never Provider authorization.
        # Shared stages allow API=0, compiler=0 and one encrypted invocation.
        bounds = dict(generations=1, http_attempts=1, wall_seconds=900, api_timeout_seconds=1, max_tokens=1)
        meta = package.seal(dict(kind=package.REQUEST, request=request, context=context,
            budget=bounds, identity=request['authority'], identity_policy='composition-v1',
            operational_limits=copy.deepcopy(self.global_operational_limits),
            validation_level='numerical', entries=[dict(row, folder=str(row['folder'])) for row in entries],
            execution_budget=dict(API=0, HTTP=0, compile=0, encrypted=1, retries=0)))
        package.write(path / 'source-parent.json', parent_raw)
        self._write(path / 'task.json', meta)
        self._write(path / 'state.json', dict(task_id=task, status='queued', phase='queued',
            validation_level='numerical', generations=0, http_attempts=0, ledger=[], failure=None,
            package=None, acceptance=None, generation=None, reuse_key=package.reuse_key(request)))
        try:
            monitor = ArtifactMonitor(self, path, self.global_operational_limits, initialize=True)
            monitor.check(force=True)
        except OperationalStopped as stop:
            state = self._load(path / 'state.json')
            state.update(status=stop.status, phase='complete', failure=stop.public_failure)
            self._write(path / 'state.json', state)
        return task

    def _identity_matches(self, meta, *, delivery=False):
        if meta.get('kind') != package.REQUEST:
            return super()._identity_matches(meta, delivery=delivery)
        return meta['identity'] == meta['request']['authority'] == package.authority() and \
               composition_context(self.backend.context, meta['request']) == meta['context']

    def _package_task_metadata(self, origin, requested, level):
        if requested.get('kind') != package.REQUEST:
            return super()._package_task_metadata(origin, requested, level)
        prior = self._checked_task_metadata(origin)
        need(prior['kind'] == package.REQUEST and level == prior['validation_level'] == 'numerical' and
             package.reuse_key(prior['request']) == package.reuse_key(requested['request']) and
             prior['request'] == requested['request'] and self._identity_matches(prior), 'composition cached task binding')
        return prior

    def _execution_input(self, task, samples, sample_format):
        try:
            return self._composition_execution_input(task, samples, sample_format)
        except (ValueError, TypeError, KeyError, IndexError, OSError):
            return dict(original_contract_validation_required=True)

    def _composition_execution_input(self, task, samples, sample_format):
        meta = self._checked_task_metadata(task)
        if meta.get('kind') != package.REQUEST:
            return super()._execution_input(task, samples, sample_format)
        need(sample_format == 'named', 'composition sample format')
        from program_execution import normalize_samples
        request = meta['request']
        parent_raw = package.regular(self._dir(task) / 'source-parent.json')
        need(package.sha(parent_raw) == request['source_parent_raw_sha256'], 'prebinding parent source')
        model = package.selected_model(package.loads(parent_raw), request['source_node_id'],
                                       request['layout']['outputs'][0]['name'])
        need(package.digest(model) == request['model_sha256'], 'prebinding logical model')
        # Input prebinding grants no package grade. The checked result/backend
        # runs after shared admission, inside the one verification operation.
        normalized = normalize_samples(dict(model=model, layout=request['layout']), samples)
        return input_binding([{name: value.tolist() for name, value in row.items()} for row in normalized])

    def result(self, task):
        meta_path = self._dir(task) / 'task.json'
        try:
            kind = self._checked_task_metadata(task).get('kind') if meta_path.exists() else None
        except (ValueError, TypeError, KeyError, OSError):
            result = self.status(task)
            result.update(status='service_error', package=None, acceptance=None,
                failure=dict(code='task_integrity_rejected', layer='integrity'))
            return result
        if kind != package.REQUEST:
            return super().result(task)
        if self.status(task)['status'] == 'succeeded':
            # execute() reaches result() after the original operational alarm
            # and monitor are installed. Seed from the task's original readonly
            # children before checking its writable program copy. Cold writable
            # inputs still take the original complete verification path.
            try:
                from cpu_composition_verified import current_operation
                operation = current_operation()
                if operation is not None:
                    need(operation.worker is self.backend.context and operation.scope_owner is self,
                         'composition result verification owner')
                    meta = self._checked_task_metadata(task)
                    if operation.request_binding is None:
                        derived = package.derive(package.regular(self._dir(task) / 'source-parent.json'),
                            meta['request']['source_node_id'], meta['entries'])
                        need(package.canonical(derived['request']) == package.canonical(meta['request']),
                             'composition result source request changed')
                    else:
                        need(operation.request_binding == meta['request']['binding'],
                             'composition result operation request changed')
            except (ValueError, TypeError, KeyError, OSError):
                result = self.status(task)
                result.update(status='service_error', package=None, acceptance=None,
                    failure=dict(code='composition_source_integrity_rejected', layer='delivery'))
                return result
        result = super().result(task)
        if result['status'] == 'succeeded':
            try:
                state = self._load(self._dir(task) / 'state.json')
                meta = self._checked_task_metadata(task)
                folder = self._dir(result['package']['task_id']) / 'program'
                receipt = package.bound(package.loads(package.regular(folder / 'qualification/qualification.json')))
                need(state['reuse_key'] == package.reuse_key(meta['request']) and state['generation'] is None and
                     state['acceptance'] == acceptance(receipt), 'composition result acceptance binding')
            except (ValueError, TypeError, KeyError, OSError):
                result.update(status='service_error', package=None, acceptance=None,
                    failure=dict(code='composition_acceptance_integrity_rejected', layer='delivery'))
        return result

    def run(self, task, provider=None):
        if self._checked_task_metadata(task).get('kind') != package.REQUEST:
            return super().run(task, provider)
        need(provider is None, 'composition has no Provider')
        return self.run_composition(task)

    def run_composition(self, task):
        if self.status(task)['status'] in TERMINAL:
            return self.result(task)
        return self._operation('application-run', self._task_input(task),
            lambda: self._run_composition(task), task=task, seconds=900, stages=dict(encrypted=1))

    def _run_composition(self, task):
        from cpu_composition_execution import qualify, export_qualified
        from hecate_python_env import WORK
        path = self._dir(task)
        with self._lock(path):
            state = self._load(path / 'state.json'); meta = self._checked_task_metadata(task)
            need(state['status'] == 'queued' and meta['kind'] == package.REQUEST, 'composition queued task')
            need(self._identity_matches(meta), 'composition task source identity changed')
            state.update(status='running', phase='preflight'); self._write(path / 'state.json', state)
            control = None; previous = None
            try:
                control = Controller(self, path, state, meta['budget'], operational_limits=self.global_operational_limits)
                need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'worker already owns alarm')
                def alarm(*_):
                    try:
                        control.check()
                    except (Stopped, OperationalStopped):
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        raise
                previous = signal.signal(signal.SIGALRM, alarm)
                signal.setitimer(signal.ITIMER_REAL, .1, .1)
                control.check(force=True)
                # A new source request is rederived before either cache lookup
                # or backend execution. A stale child path cannot alias a cache.
                derived = package.derive(package.regular(path / 'source-parent.json'),
                    meta['request']['source_node_id'], meta['entries'])
                need(derived['request'] == meta['request'], 'composition input changed after submit')
                for other in sorted(self.store.glob('*/state.json')):
                    control.check()
                    if other.parent == path:
                        continue
                    prior = self._load(other)
                    if prior['status'] != 'succeeded' or prior.get('reuse_key') != state['reuse_key']:
                        continue
                    self._package_task_metadata(other.parent.name, meta, 'numerical')
                    origin = prior['package']['task_id']; self._package_task_metadata(origin, meta, 'numerical')
                    binding = self.backend.check_package(self._dir(origin) / 'program', 'numerical')
                    need(binding == prior['package']['binding'], 'composition cached package changed')
                    receipt = package.bound(package.loads(package.regular(self._dir(origin) / 'program' /
                        'qualification/qualification.json')))
                    need(prior['acceptance'] == acceptance(receipt), 'composition cached acceptance changed')
                    state.update(status='succeeded', phase='complete', acceptance=acceptance(receipt),
                        package=dict(task_id=origin, binding=binding, reused=True), generation=None)
                    control.save(); return self.result(task)
                state['phase'] = 'composing'; control.save()
                prepared = path / 'prepared'
                exported = package.export_prepared(package.regular(path / 'source-parent.json'),
                    meta['request']['source_node_id'], meta['entries'], prepared, checkpoint=control.check)
                need(exported['request_binding'] == meta['request']['binding'], 'composition export request')
                evidence = WORK / 'results' / ('poseidon-program-execution-' + uuid.uuid4().hex)
                state['phase'] = 'qualifying'; control.save()
                session = self._shared_active
                need(session is not None, 'shared composition encrypted lease required')
                def observe(artifact):
                    session.observe(artifact)
                    control.register_artifact(artifact)
                with session.batch.stage('encrypted'):
                    qualified = qualify(prepared, evidence=evidence, checkpoint=control.check,
                                        artifact_observer=observe)
                final = export_qualified(prepared, evidence, path / 'program', checkpoint=control.check)
                need(self._identity_matches(meta), 'composition context changed after qualification')
                state.update(status='succeeded', phase='complete', failure=None, generation=None,
                    package=dict(task_id=task, binding=final['binding'], reused=False),
                    acceptance=acceptance(qualified))
            except (Stopped, OperationalStopped) as stop:
                state.update(status=stop.status, phase='complete', package=None, acceptance=None,
                    failure=getattr(stop, 'public_failure', dict(code=stop.status, layer='resource')))
            except Exception:
                state.update(status='service_error', phase='complete', package=None, acceptance=None,
                    failure=dict(code='composition_qualification_rejected', layer='delivery', retry_policy='never_automatic'))
            finally:
                if previous is not None:
                    signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
                self._write(path / 'state.json', state)
        return self.result(task)


def acceptance(receipt):
    return dict(type=package.FORMAT, qualification_binding=receipt['binding'],
        encrypted_execution=True, numerically_validated=True, samples=4,
        scope='selected source convolution only', parent_agent_generated=False,
        derived_from_genuine_children=True, agent_calls=0, compiler_calls=0)


def application(context):
    """Explicit factory; the legacy application(context) remains unchanged."""
    from application_execution.context import WorkerContext
    need(isinstance(context, WorkerContext), 'composition worker context type')
    context.persist()
    return CompositionApplication(context.task_store, CompositionBackend(context))
