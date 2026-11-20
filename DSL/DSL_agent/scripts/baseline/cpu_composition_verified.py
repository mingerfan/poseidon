"""Private, operation-local reuse of fully verified public Conv derivations.

Only a full original derivation on a controlled readonly source view can mint a
reusable proof. Every use still reads the complete source inventory and bytes,
checks the original current WorkerContext, compositor and SDK, and returns a new
copy. Nothing here grants compiled or numerical grade, or survives an operation.
Readonly views depend on trusted host custody; they are not host immutability.
"""
import base64
import copy
import os
from pathlib import Path
import secrets
import stat
from threading import get_ident, local
from functools import wraps
import time

import cpu_composition_package as package

_ACTIVE = local()
MAX_METADATA = 16 * 1024**2
MAX_ROOTS = 32


class VerificationRejected(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise VerificationRejected(reason)


def _fail_closed(method):
    @wraps(method)
    def checked(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self._invalidate()
            raise
    return checked


class _Proof:
    """An opaque object, not a serialized statement supplied by a caller."""
    __slots__ = ('__issuer', '__key', '__kind')

    def __init__(self, issuer, key, kind):
        object.__setattr__(self, '_Proof__issuer', issuer)
        object.__setattr__(self, '_Proof__key', key)
        object.__setattr__(self, '_Proof__kind', kind)

    def __setattr__(self, *_):
        raise TypeError('Immutable verification proof')

    def __reduce_ex__(self, _):
        raise TypeError('Verification proof cannot be serialized')

    def __copy__(self):
        raise TypeError('Verification proof cannot be copied')

    def __deepcopy__(self, _):
        raise TypeError('Verification proof cannot be copied')


class VerifiedDerivation(_Proof):
    __slots__ = ()


class VerifiedMathReference(_Proof):
    __slots__ = ()


def _walk(root):
    return sorted(Path(root).rglob('*'))


def _scan(entries):
    """Full raw reads, never an mtime/inode substitute for content hashes."""
    need(type(entries) is list and 1 <= len(entries) <= 9, 'composition source count')
    semantic, physical = [], []
    total = count = 0
    for entry in entries:
        need(type(entry) is dict and set(entry) == {'folder', 'binding', 'kernel_positions'}, 'source entry fields')
        root = Path(entry['folder'])
        need(root.is_absolute() and root == root.resolve() and root.is_dir() and
             all(not p.is_symlink() for p in (root, *root.parents)), 'direct source root')
        before = _walk(root)
        files = {}
        for path in before:
            need(not path.is_symlink(), 'source symlink')
            info = path.lstat()
            need(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), 'source entry type')
            if stat.S_ISREG(info.st_mode):
                name = package.intake.relative(str(path.relative_to(root)))
                raw = package.regular(path)
                files[name] = package.sha(raw)
                total += len(raw); count += 1
                need(total <= package.MAX_BYTES and count <= package.MAX_FILES, 'source snapshot budget')
        need(_walk(root) == before, 'source inventory changed while reading')
        manifest = package.bound(package.loads(package.regular(root / 'manifest.json', 1024**2)))
        need(manifest.get('format') == 'poseidon-dsl-bundle-v6' and
             manifest.get('model_provenance') == 'not_supplied' and manifest['binding'] == entry['binding'],
             'public v6 source snapshot domain')
        need(type(manifest['files']) is dict and 1 <= len(manifest['files']) <= 1024 and
             all(files.get(name) == expected for name, expected in manifest['files'].items()),
             'declared source bytes')
        semantic.append(dict(manifest_raw_sha256=files['manifest.json'],
            files=manifest['files'], program_binding=entry['binding'],
            kernel_positions=copy.deepcopy(entry['kernel_positions'])))
        physical.append(dict(root=str(root), files=files,
            readonly=bool(os.statvfs(root).f_flag & os.ST_RDONLY),
            entry=dict(entry, folder=str(root))))
    return dict(semantic=semantic, physical=physical, bytes=total, files=count)


def _encode_data(data):
    value = copy.deepcopy({key: item for key, item in data.items() if key != 'inputs'})
    value['entries'] = [dict(row, folder=str(row['folder'])) for row in value['entries']]
    value['plan_v1'] = base64.b64encode(data['plan_v1']).decode('ascii')
    value['released']['raw'] = base64.b64encode(data['released']['raw']).decode('ascii')
    value['linked']['bundle_raw'] = base64.b64encode(data['linked']['bundle_raw']).decode('ascii')
    value['inputs'] = []
    for row in data['inputs']:
        item = copy.deepcopy({key: val for key, val in row.items() if key != 'read_blob'})
        for key in ('plan_raw', 'spec_raw', 'bundle_raw'):
            item[key] = base64.b64encode(row[key]).decode('ascii')
        value['inputs'].append(item)
    raw = package.canonical(value)
    need(len(raw) <= MAX_METADATA, 'operation derivation metadata bound')
    return raw


def _decode_data(raw, physical):
    value = package.loads(raw)
    value['plan_v1'] = base64.b64decode(value['plan_v1'], validate=True)
    value['released']['raw'] = base64.b64decode(value['released']['raw'], validate=True)
    value['linked']['bundle_raw'] = base64.b64decode(value['linked']['bundle_raw'], validate=True)
    for index, row in enumerate(value['inputs']):
        for key in ('plan_raw', 'spec_raw', 'bundle_raw'):
            row[key] = base64.b64decode(row[key], validate=True)
        root = Path(physical[index]['root']); hashes = physical[index]['files']
        def read_blob(content, root=root, hashes=hashes):
            package.link.content_name(content)
            name = 'artifacts/bound/bundle/data/' + content[7:] + '.bin'
            result = package.regular(root / name)
            need(hashes.get(name) == package.sha(result) == content[7:], 'proof source blob changed')
            return result
        row['read_blob'] = read_blob
    return value


def _math_sources():
    import benchmark_graph, benchmark_math, benchmark_torch
    from hecate_python_env import ROOT
    baseline = ROOT / 'DSL/DSL_agent/scripts/baseline'
    result = {}
    for module in (benchmark_graph, benchmark_math, benchmark_torch):
        path = Path(module.__file__).resolve()
        need(path == baseline / (module.__name__ + '.py'), 'original mathematical module origin')
        result[module.__name__] = package.sha(package.regular(path))
    return result


def compute_references(model, child_models):
    """Independent real arithmetic only; never an encrypted-output substitute."""
    import numpy as np
    from benchmark_graph import samples
    from benchmark_math import evaluate
    from benchmark_torch import evaluate as torch_evaluate
    from cpu_composition_execution import comparison
    need(type(child_models) is list and 2 <= len(child_models) <= 9, 'reference child count')
    # Operate on private canonical snapshots, never caller-owned mutable arrays.
    model = package.loads(package.canonical(model))
    child_models = package.loads(package.canonical(child_models))
    sources = _math_sources()
    probes = samples(model, 4)
    def rows(values):
        return np.asarray([[x for output in model['outputs'] for x in
            np.asarray(row[output['name']], dtype=np.float64).flat] for row in values], dtype='<f8')
    full = rows([evaluate(model, probe) for probe in probes])
    torch = rows([torch_evaluate(model, probe) for probe in probes])
    full_check = comparison(torch, full, 1e-12, 1e-12)
    need(full_check['failures'] == 0, 'full parent independent mathematical difference')
    summed = None
    for child in child_models:
        need(child['inputs'] == model['inputs'] and
             [row['name'] for row in child['outputs']] == [row['name'] for row in model['outputs']],
             'reference logical signature')
        current = rows([torch_evaluate(child, probe) for probe in probes])
        need(current.shape == full.shape, 'reference child output shape')
        # Explicit original child order. np.sum would change association.
        summed = current.copy() if summed is None else summed + current
    child_check = comparison(summed, full, 1e-12, 1e-12)
    need(child_check['failures'] == 0, 'ordered child leftfold mathematical difference')
    logical = [{spec['name']: np.asarray(probe[spec['name']], dtype='<f8').tolist()
                for spec in model['inputs']} for probe in probes]
    probe_bits = [[dict(name=spec['name'], shape=list(np.asarray(probe[spec['name']]).shape),
                       dtype='little-endian float64',
                       raw_sha256=package.sha(np.asarray(probe[spec['name']], dtype='<f8').tobytes(order='C')))
                   for spec in model['inputs']] for probe in probes]
    need(_math_sources() == sources, 'mathematical source changed')
    value = dict(format='poseidon-parent-and-ordered-child-math-reference-v1',
        model_sha256=package.digest(model), ordered_child_model_sha256=[package.digest(c) for c in child_models],
        probes=logical, reference=full.tolist(), full_math_comparison=full_check,
        child_leftfold_comparison=child_check, math_sources=sources,
        probes_sha256=package.digest(logical), probes_f64_sha256=package.digest(probe_bits),
        addition_order='stable float64 leftfold in request child order',
        forwards=dict(parent_numpy=4, parent_torch=4, child_torch=4 * len(child_models)),
        encrypted_execution=False, numerical_parent_qualification=False)
    raw = package.canonical(value)
    need(len(raw) <= MAX_METADATA, 'mathematical reference metadata bound')
    return raw


def _decode_references(raw):
    import numpy as np
    result = package.loads(raw)
    # Return independent arrays with immutable backing and independent dicts.
    for row in result['probes']:
        for name, value in row.items():
            array = np.asarray(value, dtype='<f8')
            row[name] = np.frombuffer(array.tobytes(), dtype='<f8').reshape(array.shape)
    reference = np.asarray(result['reference'], dtype='<f8')
    result['reference'] = np.frombuffer(reference.tobytes(), dtype='<f8').reshape(reference.shape)
    return result


class VerificationOperation:
    """A private read epoch for one pinned worker and one composition request."""
    def __init__(self, worker, *, request_binding=None, checkpoint=lambda: None, scope_owner=None):
        from application_execution.context import WorkerContext
        need(isinstance(worker, WorkerContext), 'pinned verification WorkerContext')
        need(request_binding is None or type(request_binding) is str, 'verification request binding')
        self.worker, self.request_binding, self.checkpoint = worker, request_binding, checkpoint
        self.scope_owner = scope_owner
        self.owner = (os.getpid(), get_ident())
        self.issuer = object(); self.nonce = secrets.token_hex(16)
        self.active = False; self.invalid = False
        self.semantic_key = None
        self.records = {}; self.roots = {}; self.math = {}
        self.metrics = dict(full_derivations=0, original_child_readers=0, derivation_hits=0,
            math_forward_batches=0, math_hits=0, full_file_scans=0, guard_calls=0,
            guard_CPU_seconds=0.0, guard_wall_seconds=0.0, final_file_scans=0)

    def __enter__(self):
        need(getattr(_ACTIVE, 'operation', None) is None, 'one verification operation per thread')
        need(not self.active and not self.invalid, 'fresh verification operation')
        try:
            self.active = True
            self.worker.check_current()
            from poseidon_runtime_paths import checked_sdk
            self.authority = package.canonical(package.authority())
            self.sdk = package.canonical(checked_sdk())
            baseline = Path(package.__file__).resolve().parent
            self.readonly_code = all(os.statvfs(baseline / name).f_flag & os.ST_RDONLY for name in package.SOURCES)
            _ACTIVE.operation = self
            self._guard()
            return self
        except BaseException:
            self._invalidate()
            self.active = False
            if getattr(_ACTIVE, 'operation', None) is self:
                del _ACTIVE.operation
            raise

    def _invalidate(self):
        self.invalid = True
        self.records.clear(); self.math.clear()

    @_fail_closed
    def _check_owner(self):
        need(self.active and not self.invalid and self.owner == (os.getpid(), get_ident()),
             'verification operation owner/lifetime')

    def _guard(self):
        before_cpu, before_wall = time.process_time(), time.monotonic()
        try:
            self._check_owner()
            self.checkpoint()
            self.worker.check_current()
            from poseidon_runtime_paths import checked_sdk
            need(package.canonical(package.authority()) == self.authority and
                 package.canonical(checked_sdk()) == self.sdk, 'verification current source/stage/SDK changed')
            self.metrics['guard_calls'] += 1
        except BaseException:
            self._invalidate()
            raise
        finally:
            self.metrics['guard_CPU_seconds'] += time.process_time() - before_cpu
            self.metrics['guard_wall_seconds'] += time.monotonic() - before_wall

    def _scan(self, entries):
        try:
            scanned = _scan(entries)
            self.metrics['full_file_scans'] += 1
            for row in scanned['physical']:
                root = row['root']
                need(root not in self.roots or self.roots[root] == row,
                     'verification source inventory or bytes changed')
                need(root in self.roots or len(self.roots) < MAX_ROOTS, 'verification source root bound')
                self.roots[root] = row
            return scanned
        except BaseException:
            self._invalidate()
            raise

    def _token(self, kind, key):
        cls = VerifiedDerivation if kind == 'derivation' else VerifiedMathReference
        return cls(self.issuer, key, kind)

    @_fail_closed
    def _record(self, token, kind):
        self._guard()
        return self._record_after_guard(token, kind)

    def _record_after_guard(self, token, kind):
        """Internal continuation of an entry guard, never a public skip flag."""
        self._check_owner()
        cls = VerifiedDerivation if kind == 'derivation' else VerifiedMathReference
        need(type(token) is cls and token._Proof__issuer is self.issuer and
             token._Proof__kind == kind, 'foreign verification proof')
        record = self.records.get(token._Proof__key) if kind == 'derivation' else self.math.get(token._Proof__key)
        need(record is not None and record['token'] is token, 'unknown verification proof')
        if kind == 'derivation':
            current = self._scan([row['entry'] for row in record['physical']])
            need(current['physical'] == record['physical'], 'verified source bytes changed')
        return record

    @_fail_closed
    def derive(self, parent_raw, node_id, entries):
        self._guard()
        entries = [dict(copy.deepcopy(row), folder=str(row['folder'])) for row in entries]
        scanned = self._scan(entries)
        key = package.digest(dict(parent_raw_sha256=package.sha(parent_raw), node_id=node_id,
                                  children=scanned['semantic'], authority_sha256=package.sha(self.authority)))
        need(self.semantic_key in (None, key), 'operation model/child order changed')
        record = self.records.get(key)
        if record is not None:
            self.metrics['derivation_hits'] += 1
            result = _decode_data(record['payload'], record['physical'])
        else:
            data = package._derive_full(parent_raw, node_id, entries)
            self.metrics['full_derivations'] += 1
            self.metrics['original_child_readers'] += len(entries)
            need(self.request_binding in (None, data['request']['binding']), 'operation composition request changed')
            self.request_binding = data['request']['binding']
            self.semantic_key = key
            after = self._scan(entries)
            need(scanned == after, 'source changed during full original derivation')
            # A cold writable source retains the original complete path. Only
            # the controlled readonly source domain can seed reusable proofs.
            if self.readonly_code and all(row['readonly'] for row in scanned['physical']):
                record = dict(token=self._token('derivation', key), payload=_encode_data(data),
                    physical=copy.deepcopy(scanned['physical']), semantic=scanned['semantic'])
                self.records[key] = record
                result = _decode_data(record['payload'], record['physical'])
            else:
                result = data
        need(result['request']['binding'] == self.request_binding, 'operation request binding')
        self._guard()
        return result

    @_fail_closed
    def derivation_proof(self, request):
        self._guard()
        package.bound(request)
        need(request['binding'] == self.request_binding, 'operation request proof binding')
        for record in self.records.values():
            original_request = package.loads(record['payload'])['request']
            if original_request['binding'] == request['binding']:
                need(package.canonical(request) == package.canonical(original_request),
                     'exact verified composition request')
                return record['token']
        return None

    @_fail_closed
    def references(self, request, model, child_models):
        token = self.derivation_proof(request)
        need(package.digest(model) == request['model_sha256'] and
             [package.digest(m) for m in child_models] == [row['model_sha256'] for row in request['children']],
             'mathematical model/order binding')
        if token is None:
            # A writable cold source never seeds D or T. Keep full arithmetic.
            result = _decode_references(compute_references(model, child_models))
            need(result['model_sha256'] == request['model_sha256'] and
                 result['ordered_child_model_sha256'] == [row['model_sha256'] for row in request['children']],
                 'computed reference exact model/order binding')
            self._guard()
            return result
        self._record_after_guard(token, 'derivation')
        key = package.digest(dict(request_binding=request['binding'], model=request['model_sha256'],
            children=[row['model_sha256'] for row in request['children']], sources=_math_sources()))
        if key not in self.math:
            raw = compute_references(model, child_models)
            value = package.loads(raw)
            need(value['model_sha256'] == request['model_sha256'] and
                 value['ordered_child_model_sha256'] == [row['model_sha256'] for row in request['children']],
                 'computed reference exact model/order binding')
            self.math[key] = dict(token=self._token('math', key), payload=raw)
            self.metrics['math_forward_batches'] += 1
        else:
            self.metrics['math_hits'] += 1
        result = _decode_references(self.math[key]['payload'])
        self._guard()
        return result

    @_fail_closed
    def describe(self, token):
        record = self._record(token, 'derivation')
        return dict(kind='verified-derivation', request_binding=self.request_binding,
            payload_sha256=package.sha(record['payload']), operation_nonce=self.nonce,
            prepared_only=True, qualification_authority=False, owner=list(self.owner))

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._guard()
                # Recheck every admitted input root at the final boundary.
                for root, row in list(self.roots.items()):
                    checked = _scan([row['entry']])
                    self.metrics['final_file_scans'] += 1
                    need(checked['physical'][0] == row, 'operation final source snapshot changed')
        finally:
            self.active = False
            self._invalidate(); self.roots.clear()
            if getattr(_ACTIVE, 'operation', None) is self:
                del _ACTIVE.operation
        return False


def current_operation():
    operation = getattr(_ACTIVE, 'operation', None)
    if operation is not None:
        # Lookup exposes no proof/grade. Derive/reference methods still check
        # full current source and SDK at entry and exit; proof queries guard too.
        operation._check_owner()
    return operation


def references(request, model, child_models):
    operation = current_operation()
    if operation is not None:
        return operation.references(request, model, child_models)
    return _decode_references(compute_references(model, child_models))
