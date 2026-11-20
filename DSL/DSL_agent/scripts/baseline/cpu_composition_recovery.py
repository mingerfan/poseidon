"""Explicit host API to reverify one fully written, interrupted CPU parent.

This API remains outside the historical nine compositor SOURCES. A separately
frozen caller owns source41, approval/limits, host ownership and immutable input
snapshots. Expected digests select evidence; they do not confer qualification.

The original normal 4GiB Shared operation, current WorkerContext, original
numerical package reader and original acceptance/result paths remain mandatory.
No run/execute/qualify call, Provider, compiler, encrypted stage, key creation or
native invocation occurs in this API. Original run failure and ledger records
are retained. A successful return is a NEW explicit recovery result.
"""

from pathlib import Path
import copy
import hashlib
import json
import math
import os
import re
import signal
import stat
import time
import uuid

FORMAT = 'poseidon-completed-parent-recovery-input-v1'
RESULT = 'poseidon-explicit-completed-parent-recovery-result-v1'
WALL_SECONDS = 360
GIB = 1024**3
KINDS = {'api', 'build', 'compile', 'encrypted', 'link', 'qualify'}
EXPECTED_KEYS = {'format', 'name', 'record_sha256', 'operation_raw_sha256',
    'scope_plan_raw_sha256', 'program_manifest_raw_sha256', 'qualification_raw_sha256',
    'worker_context_binding', 'helper_source_sha256', 'host_pid_namespace_inode', 'state_before_snapshot'}


class RecoveryRejected(ValueError):
    pass


def need(ok, reason):
    if not ok:
        raise RecoveryRejected(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _expected(value):
    need(type(value) is dict and set(value) == EXPECTED_KEYS and value['format'] == FORMAT, 'recovery input schema')
    need(type(value['name']) is str and re.fullmatch('shared-component-[A-Za-z0-9_-]{1,80}', value['name']),
         'original scope name')
    for key in EXPECTED_KEYS - {'format', 'name', 'host_pid_namespace_inode', 'state_before_snapshot'}:
        need(_hash(value[key]), 'exact expected digest: ' + key)
    need(type(value['host_pid_namespace_inode']) is int and value['host_pid_namespace_inode'] > 0,
         'host PID namespace identity')
    item = value['state_before_snapshot']
    need(type(item) is dict and set(item) == {'path', 'sha256', 'bytes'} and type(item['path']) is str
         and _hash(item['sha256']) and type(item['bytes']) is int and 1 <= item['bytes'] <= 4 * 1024**2,
         'caller-preserved original state snapshot')
    return json.loads(canonical(value))


def _completed_scope(record, expected):
    """Pure data rejection surface; this does not prove the owner is dead."""
    need(type(record) is dict and digest(record) == expected['record_sha256'], 'original complete scope record digest')
    need(record.get('status') == 'interrupted' and record.get('active') is None and
         record.get('uncertain_stage') is None, 'explicitly settled interrupted scope with no uncertain stage')
    need(set(record.get('used', {})) == KINDS and set(record['plan'].get('stages', {})) == KINDS,
         'original stage schema')
    for key in KINDS:
        wanted = int(key == 'encrypted')
        need(type(record['used'][key]) is int and record['used'][key] == wanted and
             type(record['plan']['stages'][key]) is int and record['plan']['stages'][key] == wanted,
             'exact original single encrypted allocation/use and all other stages zero')
    last = record.get('last_stage')
    need(type(last) is dict and last.get('kind') == 'encrypted' and
         type(last.get('number')) is int and last['number'] == 1 and
         last.get('outcome') == 'completed' and last.get('execution_uncertain') is False,
         'original native stage must be known completed')
    need(type(record['plan']['reserve_bytes']) is int and record['plan']['reserve_bytes'] == 4 * GIB,
         'original normal4GiB scope')
    need(record['plan_binding'] == digest(record['plan']), 'original scope plan binding')
    need(type(record.get('owner')) is dict and set(record['owner']) == {'pid', 'start_ticks', 'boot_id'} and
         type(record['owner']['pid']) is int and record['owner']['pid'] > 1 and
         type(record['owner']['start_ticks']) is int and record['owner']['start_ticks'] > 0 and
         type(record['owner']['boot_id']) is str, 'original owner tuple')
    for item in (record, last):
        for key in ('started', 'finished'):
            need(type(item.get(key)) in (int, float) and math.isfinite(item[key]), 'finite original scope clock')
    need(record['started'] <= last['started'] <= last['finished'] <= record['finished'], 'native completion before settled scope end')
    return record


def _recoverable_state(state, task):
    need(type(state) is dict and state.get('task_id') == task and state.get('validation_level') == 'numerical', 'original numerical task state')
    need(state.get('package') is None and state.get('acceptance') is None and state.get('generation') is None,
         'recovery never replaces an already delivered or synthetic qualification')
    need(type(state.get('generations')) is int and state['generations'] == 0 and
         type(state.get('http_attempts')) is int and state['http_attempts'] == 0 and state.get('ledger') == [],
         'composition has no generation/HTTP history')
    if state.get('status') == 'running':
        need(state.get('phase') == 'qualifying' and state.get('failure') is None, 'original interrupted qualification state')
    else:
        need(state.get('status') == 'service_error' and state.get('phase') == 'complete' and
             state.get('failure') == dict(code='interrupted_outcome_unknown', layer='worker'),
             'only original explicit interrupted recovery state is eligible')


def _failed_committed_state(state, verification_binding, *, status, failure):
    """Return a conservative replacement only for this call's written success.

    Pure data: never writes, touches a historical proof, or mutates its input.
    An unrelated marker or any already unsuccessful state is left untouched.
    The caller still must re-raise the original exception after writing a match.
    """
    need(_hash(verification_binding), 'this recovery verification binding')
    if type(state) is not dict or state.get('status') != 'succeeded':
        return None
    marker = state.get('completed_parent_recovery')
    if type(marker) is not dict or marker.get('verification_binding') != verification_binding:
        return None
    need(type(status) is str and status in {'cancelled', 'timed_out', 'budget_exhausted',
         'service_error', 'unsupported', 'validation_failed'} and type(failure) is dict,
         'original unsuccessful terminal classification')
    replacement = copy.deepcopy(state)
    replacement.update(status=status, phase='complete', package=None, acceptance=None,
                       failure=copy.deepcopy(failure))
    return replacement


def _regular(path, limit=16 * 1024**2, *, private=False):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and
         all(not p.is_symlink() for p in (path, *path.parents)), 'direct bounded recovery evidence path')
    need(not set(path.parts).intersection({'.env', '.ssh', '.aws'}) and path.suffix not in {'.pem', '.key'},
         'private credential/key material is outside recovery')
    info = path.stat()
    need(stat.S_ISREG(info.st_mode) and info.st_size <= limit, 'bounded regular recovery evidence')
    if private:
        need(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600, 'private caller state snapshot')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    after = path.stat()
    need(len(data) <= limit and (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'recovery evidence changed while reading')
    return data


def _json(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            need(key not in value, 'duplicate recovery JSON key')
            value[key] = item
        return value
    def nonfinite(_):
        raise RecoveryRejected('nonfinite recovery JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def _bound(value):
    need(type(value) is dict and _hash(value.get('binding')) and
         value['binding'] == digest({k: v for k, v in value.items() if k != 'binding'}), 'recovery evidence self binding')
    return value


def _snapshot_reference(expected, task_dir):
    item = expected['state_before_snapshot']
    path = Path(item['path'])
    need(not path.is_relative_to(task_dir), 'original state snapshot must already exist outside mutable task')
    raw = _regular(path, 4 * 1024**2, private=True)
    need(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'caller original state snapshot exact bytes')
    return raw


def _sync_new(path, raw):
    need(type(raw) is bytes, 'archive bytes')
    need(all(not p.is_symlink() for p in (path, *path.parents)), 'direct recovery archive path')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _native_copies(program, native):
    expected = {}
    for path in sorted((program / 'qualification').iterdir()):
        need(path.is_file() and not path.is_symlink(), 'qualified evidence regular files')
        raw = _regular(path)
        need(raw == _regular(native / path.name), 'qualified evidence is the exact original registered native output')
        expected[path.name] = sha(raw)
    need({'qualification.json', 'report.json', 'execution.json', 'inputs.json', 'payload.json',
          'generated-input-guard.json', 'native-command.json', 'native.log'} <= set(expected),
         'complete original native evidence set')
    need({p.name for p in native.iterdir()} == set(expected), 'exact original native evidence inventory')
    return expected


def recover_completed_parent(app, task, expected_failed_scope):
    """Reverify and explicitly adopt one complete parent; never replay native.

    expected_failed_scope is a caller-pinned FORMAT record with EXPECTED_KEYS.
    Its state_before_snapshot is a private preexisting immutable-by-custody raw
    copy outside the task. The trusted launcher must freeze it and source41,
    prove host ownership, and separately authorize/enforce all worker limits.
    This function changes no limit and consumes no encrypted/compiler/API stage.
    """
    from application_composition import CompositionApplication
    import cpu_composition_package as package
    expected = _expected(expected_failed_scope)
    need(type(app) is CompositionApplication and type(task) is str and re.fullmatch('[0-9a-f]{32}', task), 'original composition application/task')
    need(Path(__file__).name not in package.SOURCES, 'separate versioned recovery API source')
    helper_raw = _regular(Path(__file__).resolve())
    need(sha(helper_raw) == expected['helper_source_sha256'], 'caller-pinned recovery source41')
    task_dir = app._dir(task)
    before_raw = _snapshot_reference(expected, task_dir)
    need(_regular(task_dir / 'state.json', 4 * 1024**2) == before_raw, 'live original state still equals preserved snapshot')
    _recoverable_state(_json(before_raw), task)
    # Only cheap raw metadata and the preexisting snapshot are read before
    # original admission. No task/CO/evidence write precedes _operation.
    binding = dict(app._task_input(task), completed_parent_recovery=dict(
        format=FORMAT, recovery_input_sha256=digest(expected), helper_source_sha256=sha(helper_raw),
        original_run_returned_successfully=False, new_native_calls=0))
    returned = app._operation('application-run', binding,
        lambda: _inside(app, task, expected, before_raw), task=task, stages={}, seconds=WALL_SECONDS)
    need(type(returned) is dict and returned.get('format') == RESULT and returned.get('status') == 'succeeded',
         'new explicit recovery did not finish its original Shared operation')
    return returned


def _inside(app, task, expected, before_raw):
    from application_component import Controller, Stopped
    from application_artifact_budget import OperationalStopped
    from application_composition import acceptance
    from cpu_composition_verified import current_operation
    from shared_execution import process
    import cpu_composition_package as package
    task_dir, program = app._dir(task), app._dir(task) / 'program'
    session = app._shared_active
    need(session is not None and app._shared_action == 'application-run' and app._shared_task == task,
         'original admitted task operation')
    need(session.batch.record()['plan']['reserve_bytes'] == 4 * GIB and
         all(type(v) is int and v == 0 for v in session.batch.record()['plan']['stages'].values()),
         'original4GiB and strictly zero-stage new operation')
    operation = current_operation()
    need(operation is not None and operation.worker is app.backend.context and operation.scope_owner is app,
         'original operation-local verification scope')
    need(os.stat('/proc/self/ns/pid').st_ino == expected['host_pid_namespace_inode'], 'caller-pinned real host PID namespace')
    need(_regular(task_dir / 'state.json', 4 * 1024**2) == before_raw and
         _snapshot_reference(expected, task_dir) == before_raw, 'frozen original state at admitted boundary')
    state = _json(before_raw)
    _recoverable_state(state, task)
    meta = app._checked_task_metadata(task)
    need(meta['kind'] == package.REQUEST and meta['validation_level'] == 'numerical' and
         meta['execution_budget'] == dict(API=0, HTTP=0, compile=0, encrypted=1, retries=0), 'original task contract')
    request = package.bound(meta['request'])
    need(meta['identity'] == request['authority'] == package.authority() and app._identity_matches(meta) and
         meta['context']['worker_binding'] == expected['worker_context_binding'] == app.backend.context.binding,
         'current original meta/Context/source9 authority')
    app.backend.context.check_current()
    coordinator = session.batch.coordinator
    data, _ = coordinator.snapshot()
    original = copy.deepcopy(data['batches'][expected['name']])
    _completed_scope(original, expected)
    need(original['owner']['boot_id'] == process()['boot_id'] and process(original['owner']['pid']) != original['owner'],
         'original owner is actually dead in the pinned host namespace')
    operation_path = task_dir / (expected['name'] + '.json')
    op_raw = _regular(operation_path)
    need(sha(op_raw) == expected['operation_raw_sha256'], 'original task/shared operation raw receipt')
    op = _bound(_json(op_raw))
    expected_input = dict(app._task_input(task), composition_authority_sha256=digest(package.authority()))
    need(op['format'] == 'poseidon-application-shared-operation-v1' and op['action'] == 'application-run'
         and op['task_store'] == str(app.store) and op['task_id'] == task and op['batch_name'] == expected['name']
         and op['plan_binding'] == original['plan_binding'] and op['owner'] == original['owner']
         and op['operation_input'] == expected_input, 'original run bound this exact task and current request')
    need(original['artifacts'][0]['path'] == op['scope'] and Path(op['scope']).name == expected['name'], 'original scope owner path')
    scope_plan_raw = _regular(Path(op['scope']) / 'plan.json')
    need(sha(scope_plan_raw) == expected['scope_plan_raw_sha256'], 'original scope/plan raw snapshot')
    scope_plan = _json(scope_plan_raw)
    old_authority = scope_plan['authority']
    need(scope_plan['plan'] == original['plan'] and scope_plan['plan_binding'] == original['plan_binding']
         and digest(old_authority) == original['plan']['source_binding']
         and old_authority['action'] == op['action'] and old_authority['input_sha256'] == digest(op['operation_input'])
         and old_authority['host'] == session.frozen['host']
         and old_authority['extra_authority'] == session.frozen['extra_authority'], 'original full Shared source/SDK/input authority')
    native_records = [row for row in original['artifacts'] if Path(row['path']).name.startswith('poseidon-program-execution-')]
    need(len(original['artifacts']) == 2 and len(native_records) == 1, 'one exact original registered native evidence directory')
    native = Path(native_records[0]['path'])
    for row in original['artifacts']:
        path = Path(row['path']); info = path.stat()
        need(path.is_dir() and not path.is_symlink() and info.st_dev == row['device'] and info.st_ino == row['inode'],
             'original artifact physical ownership')
    need(len(original.get('growth_scopes', [])) == 1 and original['growth_scopes'][0]['path'] == str(task_dir),
         'original run monitored this exact task')
    need(sha(_regular(program / 'manifest.json')) == expected['program_manifest_raw_sha256'] and
         sha(_regular(program / 'qualification/qualification.json')) == expected['qualification_raw_sha256'],
         'caller-pinned original complete program and qualification')
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'normal recovery owns no preexisting timer')
    bounds = dict(meta['budget'], wall_seconds=min(meta['budget']['wall_seconds'], WALL_SECONDS))
    control = Controller(app, task_dir, copy.deepcopy(state), bounds,
                         operational_limits=app.global_operational_limits)
    previous = None
    proof = None
    try:
        def alarm(*_):
            try:
                control.check()
            except (Stopped, OperationalStopped):
                signal.setitimer(signal.ITIMER_REAL, 0)
                raise
        previous = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, .1, .1)
        control.check(force=True)
        derived = package.derive(package.regular(task_dir / 'source-parent.json'),
            request['source_node_id'], meta['entries'])
        need(package.canonical(derived['request']) == package.canonical(request), 'original readonly9 seed exact request')
        # This is the original complete numerical reader. It rechecks every
        # file, source/ABI/plan/bundle and the raw CKKS/input/math receipts.
        checked_binding = app.backend.check_package(program, 'numerical')
        manifest = package.bound(package.loads(package.regular(program / 'manifest.json')))
        receipt = package.bound(package.loads(package.regular(program / 'qualification/qualification.json')))
        need(manifest['binding'] == checked_binding and manifest['request_binding'] == request['binding'] and
             package.canonical(manifest['validation']) == package.canonical(dict(
                level='numerical', encrypted_execution=True, numerically_validated=True)) and
             receipt['parent_request_binding'] == request['binding'], 'full original numerical verification result')
        native_files = _native_copies(program, native)
        need(receipt['actual_execution_raw_sha256'] == native_files['execution.json'] and
             receipt['actual_execution_report_sha256'] == native_files['report.json'] and
             receipt['generated_input_guard_sha256'] == native_files['generated-input-guard.json'], 'original completed encrypted provenance')
        need(app._identity_matches(meta) and sha(_regular(Path(__file__).resolve())) == expected['helper_source_sha256'],
             'unchanged task/compositor/new helper sources')
        need(sha(_regular(program / 'manifest.json')) == expected['program_manifest_raw_sha256'] and
             sha(_regular(program / 'qualification/qualification.json')) == expected['qualification_raw_sha256'] and
             dict(app._task_input(task), composition_authority_sha256=digest(package.authority())) == expected_input,
             'full reader verified the exact pinned program, qualification and task metadata')
        after, _ = coordinator.snapshot()
        need(after['batches'][expected['name']] == original and
             _regular(operation_path) == op_raw and _regular(Path(op['scope']) / 'plan.json') == scope_plan_raw and
             _regular(task_dir / 'state.json', 4 * 1024**2) == before_raw,
             'original state and historical scope remain untouched during verification')
        control.check(force=True)
        parent = task_dir / 'completed-parent-recoveries'
        parent.mkdir(mode=0o700, exist_ok=True)
        need(not parent.is_symlink() and parent.is_dir(), 'direct new recovery archive directory')
        archive = parent / uuid.uuid4().hex
        archive.mkdir(mode=0o700)
        _sync_new(archive / 'state.before.json', before_raw)
        proof = dict(format='poseidon-completed-parent-recovery-verification-v1',
            status='verified_before_state_commit', execution_authority=False,
            original_run_returned_successfully=False, explicit_recovery=True,
            task_store=str(app.store), task_id=task, expected_input_sha256=digest(expected),
            preserved_external_state=expected['state_before_snapshot'], state_before_raw_sha256=sha(before_raw),
            old_scope_name=expected['name'], old_scope_record_sha256=digest(original),
            old_operation_raw_sha256=sha(op_raw), old_scope_plan_raw_sha256=sha(scope_plan_raw),
            original_package_binding=manifest['binding'], qualification_binding=receipt['binding'],
            numerical_reader='original CompositionBackend.check_package(level=numerical)',
            native_evidence=str(native), original_native_files=native_files,
            context_binding=app.backend.context.binding, authority_sha256=digest(request['authority']),
            new_helper_source_sha256=expected['helper_source_sha256'], new_shared_scope=session.batch.name,
            new_encrypted_API_compile_stages=0, acceptance=acceptance(receipt),
            scope='selected source convolution only', SDK_B_A5_A6_Goal_completion=False)
        proof['binding'] = digest(proof)
        _sync_new(archive / 'verification.json', canonical(proof))
        control.check(force=True)
        need(_regular(task_dir / 'state.json', 4 * 1024**2) == before_raw, 'original state unchanged before atomic commit')
        need(_native_copies(program, native) == native_files and
             sha(_regular(program / 'manifest.json')) == expected['program_manifest_raw_sha256'] and
             sha(_regular(program / 'qualification/qualification.json')) == expected['qualification_raw_sha256'],
             'original native files and complete package remain frozen before commit')
        committed = copy.deepcopy(state)
        committed.update(status='succeeded', phase='complete', failure=None, generation=None,
            package=dict(task_id=task, binding=manifest['binding'], reused=False), acceptance=acceptance(receipt),
            operational_usage=control.artifacts.usage(),
            completed_parent_recovery=dict(format=RESULT, verification_binding=proof['binding'],
                state_before_raw_sha256=sha(before_raw), original_run_returned_successfully=False,
                verification_path=str(archive / 'verification.json')))
        app._write(task_dir / 'state.json', committed)
        delivered = app.result(task)
        need(delivered['status'] == 'succeeded' and delivered['package'] == committed['package'] and
             delivered['acceptance'] == acceptance(receipt) and delivered['generation'] is None,
             'original result must independently deliver the same qualified package')
        need(sha(_regular(program / 'manifest.json')) == expected['program_manifest_raw_sha256'] and
             sha(_regular(program / 'qualification/qualification.json')) == expected['qualification_raw_sha256'],
             'original result delivered the exact pinned program')
        control.check(force=True)
        final, _ = coordinator.snapshot()
        need(final['batches'][expected['name']] == original and all(v == 0 for v in session.batch.record()['used'].values()),
             'historical scope unchanged and no new execution stage used')
        result = dict(format=RESULT, status='succeeded', explicit_recovery=True,
            original_run_returned_successfully=False, original_result=delivered,
            verification_binding=proof['binding'], verification_path=str(archive / 'verification.json'),
            old_scope_name=expected['name'], old_scope_record_sha256=digest(original),
            new_shared_scope=session.batch.name, helper_source_sha256=expected['helper_source_sha256'],
            new_native_calls=0, new_API_calls=0, new_compile_calls=0,
            selected_Conv_scope_only=True, SDK_or_B_certification=False,
            A5_complete=False, A6_complete=False, Goal_complete=False)
        result['binding'] = digest(result)
        return result
    except BaseException as error:
        # Stopped/OperationalStopped inherit BaseException, so the original
        # SharedApplication Exception fallback cannot clear this late commit.
        # Do not let a periodic alarm reenter this narrow cleanup.
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if proof is not None:
                if isinstance(error, (Stopped, OperationalStopped)):
                    status = error.status
                    failure = getattr(error, 'public_failure', dict(code=status, layer='resource'))
                elif isinstance(error, KeyboardInterrupt):
                    status = 'cancelled'
                    failure = dict(code='cancelled', layer='worker')
                else:
                    status = 'service_error'
                    failure = dict(code='completed_parent_recovery_did_not_complete', layer='worker',
                                   retry_policy='never_automatic')
                current = app._load(task_dir / 'state.json')
                replacement = _failed_committed_state(current, proof['binding'], status=status, failure=failure)
                if replacement is not None:
                    app._write(task_dir / 'state.json', replacement)
        except BaseException as cleanup_error:
            # Preserve the original exception object/classification and expose
            # cleanup failure as its cause. Neither branch returns success.
            raise error from cleanup_error
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
