"""R25: interpret one saved Orbit packed graph on three fixed training probes.

Only NumPy is imported by the numeric helpers.  No model, compiler, native
frontend, optimizer, or cryptographic runtime is loaded by this worker.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import struct
import threading
import time
import traceback

from .r10_paper_metrics import StageRecorder, write_json


ROUND = 'R25'
MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-saved-packed-graph-three-probe-check'
SCOPE = 'R22 original unoptimized packed graph on train:0,train:1,train:2; finite-input numerical agreement only'
CRITERION = '1e-12+1e-12*abs(reference)'
GIB, CHECK_SECONDS, SLOT_COUNT = 1024**3, 900, 65536
OP_KINDS = ('constant', 'rotate', 'mul', 'add', 'negate')
ARITY = dict(input=0, constant=0, rotate=1, mul=2, add=2, negate=1)
EXPECTED_CONSTANT_COUNT = 1547
EXPECTED_PROGRAM = dict(node_count=10113, edge_count=13342, output='10111',
                        op_counts=dict(input=1, constant=2712, rotate=1382,
                                       mul=2920, add=3022, negate=76),
                        static_live_peak=17, static_terminal_ids=['10111'])
COUNTER_FIELDS = (
    'graph_load_attempts', 'graph_load_completed',
    'constant_load_attempts', 'constant_load_completed',
    'probe_load_attempts', 'probe_load_completed',
    'packed_forward_attempts', 'packed_forward_completed',
    'math_check_attempts', 'math_check_completed', 'op_attempts', 'op_completed',
)
COUNT_FIELDS = COUNTER_FIELDS
ZERO_FIELDS = (
    'llm_calls', 'downloads', 'installations', 'new_weights', 'new_datasets',
    'fit_calls', 'agent_calls', 'tuner_calls', 'count_runs', 'emit_runs',
    'trace_runs', 'compiler_runs', 'optimizer_runs', 'solver_runs', 'keygen_runs',
    'real_ckks_attempts', 'real_ckks_completed', 'heir_calls', 'heldout_samples',
    'confirm_samples', 'report_samples', 'freeze_attempts', 'raw_cifar_reads',
    'torch_model_builds', 'torch_model_forwards', 'checkpoint_reads', 'poly_text_reads',
)
FALSE_FIELDS = ('backend_ready', 'security_validated', 'accuracy_validated',
                'formal_matrix_complete')
UNMEASURED_FIELDS = ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency',
                     'accuracy_acceptance')
STAGE_FIELDS = ('source_checked', 'native_packed_equivalence_verified')
_CONSTANT_PROGRESS = None


class WorkerFailure(RuntimeError):
    def __init__(self, message, status='error'):
        super().__init__(message)
        self.status = status


def _initial_result():
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None,
                  failed_phase=None, native_packed_equivalence_scope=SCOPE,
                  program_metadata=None, constants_metadata=None, probe_rows=[],
                  pool_rows=[], cache_nbytes=0, observed_pool_peak=0,
                  observed_pool_nbytes_peak=0, active_probe_index=None,
                  active_sample_id=None, active_node=None, active_op=None,
                  phase_artifact_bytes=0, case_artifact_bytes=0,
                  budget_monitor_completed=False,
                  op_attempted_by_kind=dict.fromkeys(OP_KINDS, 0),
                  op_completed_by_kind=dict.fromkeys(OP_KINDS, 0))
    for names, value in ((COUNTER_FIELDS, 0), (ZERO_FIELDS, 0),
                         (FALSE_FIELDS, False), (STAGE_FIELDS, False),
                         (UNMEASURED_FIELDS, None)):
        result.update(dict.fromkeys(names, value))
    return result


def _numeric_modules():
    import numpy as np
    return np


def _initialize_numpy():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    np = _numeric_modules()
    if np.__version__ != '1.25.2':
        raise WorkerFailure('R25 requires the existing NumPy 1.25.2', 'dependency_missing')
    return dict(numpy=np.__version__, dtype='float64', threads=1,
                torch_loaded=False, native_frontend_loaded=False)


def _read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise WorkerFailure('An explicit JSON source must be an object', 'source_failed')
    return value


def _same_path(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _file_record(path):
    path = Path(path).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise WorkerFailure('Missing explicit source file: ' + str(path), 'source_failed')
    return dict(path=str(path), size_bytes=path.stat().st_size)


def _size(path):
    total = 0
    for parent, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def _budget_observation(args, started=None):
    work = Path(args.work_dir).resolve()
    if started is None:
        started = getattr(args, '_started_ns', None)
    now = time.perf_counter_ns()
    return dict(ts_ns=now, case_root=str(work.parent.parent),
                case_artifact_bytes=_size(work.parent.parent),
                disk_free_bytes=shutil.disk_usage(work).free,
                elapsed_ns=None if started is None else now - started)


def _budget_error(row):
    if row['case_artifact_bytes'] > 4 * GIB:
        return 'artifact_budget_exceeded', 'R25 case artifacts exceed 4 GiB'
    if row['disk_free_bytes'] < 10 * GIB:
        return 'disk_reserve', 'R25 disk reserve is below 10 GiB'
    if row['elapsed_ns'] is not None and row['elapsed_ns'] >= CHECK_SECONDS * 1_000_000_000:
        return 'timeout', 'R25 inner allowance exhausted'
    return None


def _checkpoint(args, result):
    work = Path(args.work_dir).resolve()
    row = _budget_observation(args)
    result.update(phase_artifact_bytes=_size(work), case_artifact_bytes=row['case_artifact_bytes'],
                  last_budget_observation=row)
    write_json(work / 'phase_result.json', result)
    failure = _budget_error(row)
    if failure:
        raise WorkerFailure(failure[1], failure[0])


@contextmanager
def _phase(args, recorder, result, name, **metadata):
    result['failed_phase'] = name
    _checkpoint(args, result)
    with recorder.span(name, **metadata):
        yield
    _checkpoint(args, result)


def _terminate_self():
    os.kill(os.getpid(), signal.SIGTERM)


@contextmanager
def _budget_guard(args, result):
    work, stop = Path(args.work_dir).resolve(), threading.Event()
    started = getattr(args, '_started_ns', time.perf_counter_ns())

    def watch():
        while not stop.wait(1):
            try:
                observation = _budget_observation(args, started)
                with (work / 'budget-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(observation) + '\n')
                    stream.flush()
                failure = _budget_error(observation)
                if failure is None:
                    continue
                record = dict(status=failure[0], error=failure[1], **observation)
            except BaseException as exc:
                record = dict(status='record_error', error=f'{type(exc).__name__}: {exc}'[:2000],
                              ts_ns=time.perf_counter_ns())
            record.update(termination_origin='worker_budget_monitor',
                          counters={name: result.get(name) for name in COUNTER_FIELDS},
                          active_probe_index=result.get('active_probe_index'),
                          active_node=result.get('active_node'), active_op=result.get('active_op'))
            try:
                write_json(work / 'budget-fatal.json', record)
            finally:
                _terminate_self()
            return

    thread = threading.Thread(target=watch, name='r25-budget-observer', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)
        result['budget_monitor_completed'] = not thread.is_alive()
        if thread.is_alive():
            raise WorkerFailure('R25 resource observer did not stop', 'record_error')


def load_program(graph_path) -> dict:
    """Validate saved JSON and compute a last-use plan, without numeric imports."""
    graph = _read_json(graph_path)
    nodes, edges = graph.get('nodes'), graph.get('edges')
    if (not isinstance(nodes, list) or len(nodes) < 2 or not isinstance(edges, list)
            or any(not isinstance(node, dict) for node in nodes)):
        raise WorkerFailure('TDAG needs node and edge lists', 'graph_failed')
    ids = [node.get('id') for node in nodes]
    expected_ids = ['arg0'] + [str(index) for index in range(len(nodes) - 1)]
    if ids != expected_ids or graph.get('inputs') != ['arg0'] or graph.get('outputs') != [ids[-1]]:
        raise WorkerFailure('TDAG needs ordered unique SSA IDs and single original I/O', 'graph_failed')
    positions = {name: index for index, name in enumerate(ids)}
    incoming = {name: [] for name in ids}
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise WorkerFailure('TDAG edges must have source/target fields', 'graph_failed')
        source, target = edge.get('source'), edge.get('target')
        if (not isinstance(source, str) or not isinstance(target, str)
                or source not in positions or target not in positions
                or (source, target) in seen or positions[source] >= positions[target]):
            raise WorkerFailure('TDAG has duplicate, missing, or unordered dependencies', 'graph_failed')
        seen.add((source, target))
        incoming[target].append(source)
    counts, constant_ids = Counter(), set()
    for index, node in enumerate(nodes):
        name, kind, description = node['id'], node.get('op'), node.get('op_descr')
        if kind not in ARITY or not isinstance(description, dict):
            raise WorkerFailure('Unsupported TDAG operation or attributes', 'graph_failed')
        if (kind == 'input') != (index == 0) or len(incoming[name]) != ARITY[kind]:
            raise WorkerFailure('TDAG operation arity/input differs', 'graph_failed')
        incoming[name].sort(key=positions.__getitem__)
        if kind == 'constant':
            value = description.get('value')
            if type(value) is not int or value < 0:
                raise WorkerFailure('Constant value must be a nonnegative file ID', 'graph_failed')
            constant_ids.add(value)
        elif kind == 'rotate' and type(description.get('offset')) is not int:
            raise WorkerFailure('Rotation offset must be a signed integer', 'graph_failed')
        counts[kind] += 1
    last_use = dict(positions)
    for index, node in enumerate(nodes):
        for source in incoming[node['id']]:
            last_use[source] = index
    last_use[ids[-1]] = len(nodes)
    live, peak = {'arg0'}, 1
    plan = [dict(node_id='arg0', live_before_release=1)]
    for index, node in enumerate(nodes[1:], 1):
        name = node['id']
        if any(source not in live for source in incoming[name]):
            raise WorkerFailure('Static last-use plan lost an operand', 'graph_failed')
        live.add(name)
        peak = max(peak, len(live))
        plan.append(dict(node_id=name, live_before_release=len(live)))
        for source in incoming[name]:
            if last_use[source] == index:
                live.remove(source)
        if last_use[name] == index:
            live.remove(name)
    if live != {ids[-1]}:
        raise WorkerFailure('Static last-use plan must retain only the output', 'graph_failed')
    metadata = dict(node_count=len(nodes), edge_count=len(edges), op_counts=dict(counts),
                    input='arg0', output=ids[-1], constant_ids=sorted(constant_ids),
                    static_live_peak=peak, static_terminal_ids=[ids[-1]],
                    static_live_plan=plan, operand_order='original node index',
                    operation_attributes_preserved=True)
    return dict(nodes=nodes, edges=edges, incoming=incoming, last_use=last_use,
                input='arg0', output=ids[-1], metadata=metadata)


def read_constants(path, slot_count, expected_count) -> tuple:
    """Read one original binary file with bounded numeric prefix allocations.

    Metadata entries describe the original raw length; ignored long tails are
    skipped, and their numeric finiteness is deliberately left unobserved.
    """
    np = _numeric_modules()
    if type(slot_count) is not int or slot_count < 1 or type(expected_count) is not int or expected_count < 1:
        raise WorkerFailure('Positive integer slot/count arguments are required', 'constant_failed')
    constants, entries = {}, []
    count, active_id, active_length, last_offset = None, None, None, 0
    path = Path(path)

    def progress(**extra):
        if _CONSTANT_PROGRESS is not None:
            _CONSTANT_PROGRESS(dict(entries_read=len(entries), count=count, entries=list(entries),
                cache_nbytes=sum(value.nbytes for value in constants.values()),
                active_constant_id=active_id, active_raw_length=active_length,
                current_file_offset=last_offset, **extra))

    try:
        with path.open('rb') as stream:
            file_bytes = os.fstat(stream.fileno()).st_size

            def int64():
                raw = stream.read(8)
                if len(raw) != 8:
                    raise WorkerFailure('Short binary constant header', 'constant_failed')
                return struct.unpack('<q', raw)[0]

            count = int64()
            last_offset = stream.tell()
            if count != expected_count:
                raise WorkerFailure('Binary constant count differs from the graph', 'constant_failed')
            for identifier in range(count):
                active_id, active_length = identifier, None
                length = active_length = int64()
                offset = last_offset = stream.tell()
                if length <= 0 or length > (file_bytes - offset) // 8:
                    raise WorkerFailure('Constant length is nonpositive or exceeds its file', 'constant_failed')
                prefix_length = min(length, slot_count)
                prefix = np.fromfile(stream, dtype=np.dtype('<f8'), count=prefix_length)
                last_offset = stream.tell()
                if prefix.size != prefix_length:
                    raise WorkerFailure('Short binary constant prefix', 'constant_failed')
                if not bool(np.isfinite(prefix).all()):
                    raise WorkerFailure('Effective constant values must be finite', 'nonfinite')
                if prefix_length == slot_count:
                    expanded = prefix
                else:
                    expanded = np.empty(slot_count, dtype=np.float64)
                    for start in range(0, slot_count, prefix_length):
                        end = min(start + prefix_length, slot_count)
                        expanded[start:end] = prefix[:end - start]
                expanded.flags.writeable = False
                constants[identifier] = expanded
                stream.seek(offset + 8 * length)
                last_offset = stream.tell()
                entries.append(dict(id=identifier, raw_length=length, prefix_values=prefix_length,
                                    skipped_values=length - prefix_length, expanded_slots=slot_count,
                                    nbytes=int(expanded.nbytes), effective_values_finite=True,
                                    skipped_tail_finite=None if length > slot_count else True))
                del prefix, expanded
                if (identifier + 1) % 32 == 0 or identifier + 1 == count:
                    progress()
            consumed_bytes = stream.tell()
    except BaseException as exc:
        try:
            progress(error=f'{type(exc).__name__}: {exc}'[:2000])
        except BaseException:
            # A record/budget failure must not replace the original parse error.
            pass
        raise
    metadata = dict(path=str(path.resolve()), file_bytes=file_bytes, count=count,
                    slot_count=slot_count, cache_nbytes=sum(value.nbytes for value in constants.values()),
                    readonly=True, entries=entries, opened_files=1,
                    consumed_bytes=consumed_bytes, trailing_bytes=file_bytes - consumed_bytes,
                    format='little-endian int64 count; int64 length; float64 payload',
                    expansion='expanded[j] = raw[j % raw_length]',
                    long_tail_values_read=False)
    return constants, metadata


def _require_vector(value, slots):
    np = _numeric_modules()
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype('float64') or value.shape != (slots,):
        raise WorkerFailure('A packed value must be float64 with the original slot count', 'shape_failed')
    if not bool(np.isfinite(value).all()):
        raise WorkerFailure('Packed arithmetic produced a nonfinite value', 'nonfinite')
    return value


def pack_input(image) -> object:
    np = _numeric_modules()
    value = np.asarray(image)
    if value.dtype != np.dtype('float64') or value.shape not in ((3, 32, 32), (1, 3, 32, 32)):
        raise WorkerFailure('Input must be an original float64 RGB image', 'shape_failed')
    if not bool(np.isfinite(value).all()):
        raise WorkerFailure('Input contains nonfinite values', 'nonfinite')
    flattened = value.reshape(-1, order='C') / 32.0
    block = np.zeros(4096, dtype=np.float64)
    block[:3072] = flattened
    packed = np.tile(block, 16)
    return _require_vector(packed, SLOT_COUNT)


def decode_output(packed) -> object:
    np = _numeric_modules()
    value = _require_vector(packed, SLOT_COUNT)
    with np.errstate(over='raise', invalid='raise'):
        try:
            result = (value[:10] * 32.0).reshape(1, 10)
        except FloatingPointError as exc:
            raise WorkerFailure('Decoded logits are nonfinite', 'nonfinite') from exc
    if not bool(np.isfinite(result).all()):
        raise WorkerFailure('Decoded logits are nonfinite', 'nonfinite')
    return result


def _check_math(reference, observed):
    np = _numeric_modules()
    reference, observed = np.asarray(reference), np.asarray(observed)
    if (reference.dtype != np.dtype('float64') or observed.dtype != np.dtype('float64')
            or reference.shape != (1, 10) or observed.shape != reference.shape):
        raise WorkerFailure('Reference and decoded logits must be float64 [1,10]', 'shape_failed')
    if not bool(np.isfinite(reference).all() and np.isfinite(observed).all()):
        raise WorkerFailure('Reference/decoded logits must be finite', 'nonfinite')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            difference = observed - reference
            error = np.abs(difference)
            allowance = 1e-12 + 1e-12 * np.abs(reference)
            ratio = error / allowance
    except FloatingPointError as exc:
        raise WorkerFailure('Mathematical comparison produced a nonfinite value', 'nonfinite') from exc
    if not bool(np.isfinite(difference).all() and np.isfinite(ratio).all()):
        raise WorkerFailure('Mathematical comparison produced a nonfinite value', 'nonfinite')
    return dict(passed=bool(np.all(error <= allowance)), finite=True,
                reference_shape=list(reference.shape), observed_shape=list(observed.shape),
                reference_dtype=str(reference.dtype), observed_dtype=str(observed.dtype),
                criterion=CRITERION, difference=difference.tolist(),
                max_absolute_error=float(error.max()), max_tolerance_ratio=float(ratio.max()))


def execute_program(program, constants, packed, args, recorder, result) -> object:
    """Execute the validated source order; retain only values with future users.

    The pure helper accepts a smaller packed vector for deterministic unit
    fixtures.  The real run fixes SLOT_COUNT and its source graph separately.
    Pool statistics count SSA references, including references to shared cache.
    """
    np = _numeric_modules()
    if not isinstance(packed, np.ndarray) or packed.ndim != 1:
        raise WorkerFailure('Packed input must be a vector', 'shape_failed')
    slots = packed.size
    _require_vector(packed, slots)
    for identifier in program['metadata']['constant_ids']:
        if identifier not in constants:
            raise WorkerFailure('A graph constant ID is absent', 'constant_failed')
        _require_vector(constants[identifier], slots)
        if constants[identifier].flags.writeable:
            raise WorkerFailure('Shared constants must remain read-only', 'constant_failed')
    pool = {program['input']: packed}
    peak, nbytes_peak, released = 1, int(packed.nbytes), 0
    attempts_before, completed_before = result['op_attempts'], result['op_completed']
    result.update(active_node=program['input'], active_op='input', current_pool_count=1,
                  current_pool_nbytes=int(packed.nbytes))
    try:
        for index, node in enumerate(program['nodes'][1:], 1):
            name, kind = node['id'], node['op']
            result.update(active_node=name, active_op=kind)
            operands = [pool[source] for source in program['incoming'][name]]
            result['op_attempts'] += 1
            result['op_attempted_by_kind'][kind] += 1
            with np.errstate(over='ignore', invalid='ignore'):
                if kind == 'constant':
                    value = constants[node['op_descr']['value']]
                elif kind == 'add':
                    value = operands[0] + operands[1]
                elif kind == 'mul':
                    value = operands[0] * operands[1]
                elif kind == 'negate':
                    value = -operands[0]
                elif kind == 'rotate':
                    value = np.roll(operands[0], -node['op_descr']['offset'])
                else:
                    raise WorkerFailure('Unsupported packed operation', 'graph_failed')
            result['op_completed'] += 1
            result['op_completed_by_kind'][kind] += 1
            pool[name] = value
            count, nbytes = len(pool), sum(item.nbytes for item in pool.values())
            peak, nbytes_peak = max(peak, count), max(nbytes_peak, nbytes)
            result.update(current_pool_count=count, current_pool_nbytes=nbytes,
                          observed_pool_peak=max(result['observed_pool_peak'], peak),
                          observed_pool_nbytes_peak=max(result['observed_pool_nbytes_peak'], nbytes_peak))
            _require_vector(value, slots)
            if count != program['metadata']['static_live_plan'][index]['live_before_release']:
                raise WorkerFailure('Runtime pool differs from static last-use plan', 'graph_failed')
            for source in program['incoming'][name]:
                if program['last_use'][source] == index:
                    del pool[source]
                    released += 1
            if program['last_use'][name] == index:
                del pool[name]
                released += 1
            result.update(current_pool_count=len(pool),
                          current_pool_nbytes=sum(item.nbytes for item in pool.values()))
            del operands, value
            if index % 256 == 0:
                _checkpoint(args, result)
        if set(pool) != {program['output']} or peak != program['metadata']['static_live_peak']:
            raise WorkerFailure('Runtime pool must finish with only the expected output', 'graph_failed')
        return pool[program['output']]
    finally:
        result['pool_rows'].append(dict(probe_index=result.get('active_probe_index'),
            attempted=result['op_attempts'] - attempts_before,
            completed=result['op_completed'] - completed_before, peak_live=peak,
            peak_live_nbytes=nbytes_peak, terminal_ids=list(pool),
            released_references=released, pool_is_new=True,
            bytes_scope='SSA references; shared cache references may count the same storage'))


def _check_sources(args):
    records = {name: _read_json(getattr(args, name + '_result')) for name in ('r13', 'r22', 'r24')}
    for name, record in records.items():
        if record.get('round') != name.upper() or record.get('model') != MODEL or record.get('status') != 'ok':
            raise WorkerFailure('Saved source identity/status differs: ' + name, 'source_failed')
    r13, r22, r24 = (records[name] for name in ('r13', 'r22', 'r24'))
    if (r22.get('mode') != 'orbit-resnet20-trace-tdag-preflight'
            or any(r22.get(name) is not True for name in ('orbit_model_trace_passed', 'tdag_parse_passed'))
            or any(r22.get(key, {}).get(phase) != 1 for key in ('attempted', 'completed') for phase in ('trace', 'parse'))):
        raise WorkerFailure('The original R22 trace/parse must have completed once', 'source_failed')
    trace, parse = (r22.get('phase_results', {}).get(name, {}) for name in ('trace', 'parse'))
    files = trace.get('constant_files') or []
    if (trace.get('status') != 'ok' or parse.get('status') != 'ok'
            or parse.get('parser_metadata', {}).get('max_slot') != SLOT_COUNT
            or not _same_path(parse.get('tdag'), args.graph)
            or len(files) != 1 or not _same_path(files[0].get('path'), args.constants)
            or files[0].get('size_bytes') != 784347640
            or Path(args.constants).stat().st_size != 784347640
            or Path(args.graph).resolve() != Path(args.r22_result).resolve().parent / 'work/parse-worker/tdag.json'
            or Path(args.constants).resolve() != Path(args.r22_result).resolve().parent / 'work/trace-worker/constants/_hecate_ResNetSiLU64k.cst'):
        raise WorkerFailure('R22 original graph/constants path or slot metadata differs', 'source_failed')
    artifacts = (r13.get('worker') or {}).get('artifacts') or {}
    for key, name in (('probes', 'training_probes'), ('probe_source', 'training_probe_source')):
        if not _same_path(artifacts.get(name), getattr(args, key)):
            raise WorkerFailure('Only the original R13 saved probes may be read', 'source_failed')
    worker = r24.get('worker') or {}
    expected_counts = dict(original_forward_attempts=3, original_forward_completed=3,
        approx_forward_attempts=3, approx_forward_completed=3, activation_attempts=57,
        activation_completed=57, math_check_attempts=57, math_check_completed=57)
    if (r24.get('mode') != 'orbit-native-polynomial-physical-scale-check'
            or any(worker.get(name) is not True for name in
                   ('original_reference_passed', 'native_polynomial_math_passed', 'physical_scaled_forward_completed'))
            or any(worker.get(name) != value for name, value in expected_counts.items())
            or not _same_path(r24.get('source_records', {}).get('records', {}).get('r22'), args.r22_result)
            or not _same_path(r24.get('source_records', {}).get('records', {}).get('r13'), args.r13_result)
            or worker.get('probe_source') != (r13.get('worker') or {}).get('source')):
        raise WorkerFailure('Saved R24 physical-unit references or sample provenance differ', 'source_failed')
    probes = worker.get('approximate_probes')
    if (not isinstance(probes, list) or len(probes) != 3
            or [row.get('probe_index') for row in probes] != [0, 1, 2]
            or [row.get('sample_id') for row in probes] != ['train:0', 'train:1', 'train:2']):
        raise WorkerFailure('Three ordered original R24 approximate references are required', 'source_failed')
    paths = {name: _file_record(getattr(args, name)) for name in
             ('graph', 'constants', 'probes', 'probe_source', 'r13_result', 'r22_result', 'r24_result')}
    summary = dict(paths=paths, reference_source='R24 worker.approximate_probes[].approximate',
                   original_mlir_metadata=trace.get('mlir'), parser_metadata=parse.get('parser_metadata'),
                   source_times_included=False, training_preprocessing_verified=False,
                   coefficient_source_equivalence_verified=False)
    return summary, r13, r22, r24


def _load_probes(args, r13, result):
    np = _numeric_modules()
    source = _read_json(args.probe_source)
    if (source != (r13.get('worker') or {}).get('source') or source.get('split') != 'train'
            or source.get('indices') != [0, 1, 2]
            or source.get('sample_ids') != ['train:0', 'train:1', 'train:2']
            or source.get('normalization') != {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}):
        raise WorkerFailure('Original training probe source/normalization differs', 'source_failed')
    with np.load(args.probes, allow_pickle=False) as archive:
        if set(archive.files) != {'input', 'labels', 'indices'}:
            raise WorkerFailure('Saved NPZ fields differ', 'source_failed')
        images, labels, indices = (np.array(archive[name]) for name in ('input', 'labels', 'indices'))
    if (images.dtype != np.dtype('float64') or images.shape != (3, 3, 32, 32)
            or not bool(np.isfinite(images).all()) or labels.shape != (3,) or indices.shape != (3,)
            or labels.dtype.kind not in ('i', 'u') or indices.dtype.kind not in ('i', 'u')
            or labels.tolist() != source.get('labels') or indices.tolist() != [0, 1, 2]):
        raise WorkerFailure('Original training tensor shape/dtype/identity differs', 'shape_failed')
    return images, labels, source


def _cached_references(r24):
    np = _numeric_modules()
    references = []
    for row in r24['worker']['approximate_probes']:
        value = np.asarray(row.get('approximate'))
        if value.dtype != np.dtype('float64') or value.shape != (1, 10) or not bool(np.isfinite(value).all()):
            raise WorkerFailure('R24 cached references must be finite float64 [1,10]', 'source_failed')
        references.append(value)
    return references


def run_checks(args, recorder, result) -> None:
    global _CONSTANT_PROGRESS
    with _phase(args, recorder, result, 'source_records'):
        summary, r13, r22, r24 = _check_sources(args)
        result.update(source_checked=True, source_summary=summary)
        write_json(Path(args.work_dir) / 'source-summary.json', summary)
    with _phase(args, recorder, result, 'numeric_environment'):
        result['numeric_environment'] = _initialize_numpy()
        references = _cached_references(r24)
    with _phase(args, recorder, result, 'graph_load'):
        result['graph_load_attempts'] += 1
        _checkpoint(args, result)
        program = load_program(args.graph)
        result['graph_load_completed'] += 1
        metadata = program['metadata']
        if (any(metadata.get(name) != value for name, value in EXPECTED_PROGRAM.items())
                or metadata['constant_ids'] != list(range(EXPECTED_CONSTANT_COUNT))):
            raise WorkerFailure('Saved original R22 graph structure/counts differ', 'graph_failed')
        result['program_metadata'] = dict(metadata, slots=SLOT_COUNT,
                                         static_live_nbytes=metadata['static_live_peak'] * SLOT_COUNT * 8,
                                         original_nodes=program['nodes'], original_edges=program['edges'])
        write_json(Path(args.work_dir) / 'program-metadata.json', result['program_metadata'])
    with _phase(args, recorder, result, 'constant_load'):
        result['constant_load_attempts'] += 1
        _checkpoint(args, result)
        previous = _CONSTANT_PROGRESS

        def progress(row):
            result['constants_progress'] = row
            result['cache_nbytes'] = int(row['cache_nbytes'])
            _checkpoint(args, result)

        _CONSTANT_PROGRESS = progress
        try:
            constants, metadata = read_constants(args.constants, SLOT_COUNT, EXPECTED_CONSTANT_COUNT)
        finally:
            _CONSTANT_PROGRESS = previous
        result['constant_load_completed'] += 1
        result.update(constants_metadata=metadata, cache_nbytes=int(metadata['cache_nbytes']))
        if (set(constants) != set(range(EXPECTED_CONSTANT_COUNT))
                or metadata['cache_nbytes'] != EXPECTED_CONSTANT_COUNT * SLOT_COUNT * 8
                or any(value.flags.writeable for value in constants.values())):
            raise WorkerFailure('Expanded constant cache identity/size/readonly differs', 'constant_failed')
        write_json(Path(args.work_dir) / 'constants-metadata.json', metadata)
    with _phase(args, recorder, result, 'probe_load'):
        result['probe_load_attempts'] += 1
        _checkpoint(args, result)
        images, labels, source = _load_probes(args, r13, result)
        result['probe_load_completed'] += 1
        result['probe_source'] = source
    for index in range(3):
        result.update(active_probe_index=index, active_sample_id=source['sample_ids'][index],
                      active_node=None, active_op=None)
        with _phase(args, recorder, result, 'packed_forward', probe_index=index,
                    sample_id=source['sample_ids'][index]):
            packed = pack_input(images[index])
            before = result['op_attempts']
            result['packed_forward_attempts'] += 1
            _checkpoint(args, result)
            output = execute_program(program, constants, packed, args, recorder, result)
            result['packed_forward_completed'] += 1
            if result['op_attempts'] - before != len(program['nodes']) - 1:
                raise WorkerFailure('A packed execution did not visit every original node', 'graph_failed')
            observed = decode_output(output)
            row = dict(probe_index=index, sample_id=source['sample_ids'][index],
                       label=int(labels[index]), reference=references[index].tolist(),
                       observed=observed.tolist(), packed_shape=list(output.shape),
                       packed_dtype=str(output.dtype), packed_finite=True,
                       packed_maxabs=float(_numeric_modules().abs(output).max()))
            result['probe_rows'].append(row)
            del output, packed
        with _phase(args, recorder, result, 'math_check', probe_index=index,
                    sample_id=source['sample_ids'][index]):
            result['math_check_attempts'] += 1
            _checkpoint(args, result)
            checked = _check_math(references[index], observed)
            result['math_check_completed'] += 1
            row.update(checked)
            _checkpoint(args, result)
            if not checked['passed']:
                raise WorkerFailure('Packed decoded output differs from the fixed R24 reference', 'math_failed')
        del observed
    with _phase(args, recorder, result, 'final_counts_and_cache'):
        count = len(program['nodes']) - 1
        expected = dict(graph_load_attempts=1, graph_load_completed=1,
                        constant_load_attempts=1, constant_load_completed=1,
                        probe_load_attempts=1, probe_load_completed=1,
                        packed_forward_attempts=3, packed_forward_completed=3,
                        math_check_attempts=3, math_check_completed=3,
                        op_attempts=3 * count, op_completed=3 * count)
        by_kind = {kind: EXPECTED_PROGRAM['op_counts'].get(kind, 0) * 3 for kind in OP_KINDS}
        if (any(result.get(name) != value for name, value in expected.items())
                or result['op_attempted_by_kind'] != by_kind or result['op_completed_by_kind'] != by_kind
                or len(result['probe_rows']) != 3 or any(not row.get('passed') for row in result['probe_rows'])
                or any(value.flags.writeable for value in constants.values())
                or any(row['terminal_ids'] != [program['output']] for row in result['pool_rows'])):
            raise WorkerFailure('Complete counters/pool/constant checks differ', 'graph_failed')
        result['constants_remained_readonly'] = True
        result['native_packed_equivalence_verified'] = True
        write_json(Path(args.work_dir) / 'probe-results.json', result['probe_rows'])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work-dir', 'graph', 'constants', 'probes', 'probe-source',
                 'r13-result', 'r22-result', 'r24-result'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if any((args.work_dir / name).exists() for name in ('phase_result.json', 'events.jsonl')):
        raise ValueError('R25 preserves previous worker evidence')
    resource.setrlimit(resource.RLIMIT_AS, (4 * GIB, 4 * GIB))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * GIB, 8 * GIB))
    saved_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    result, recorder = _initial_result(), StageRecorder(args.work_dir / 'events.jsonl')
    args._started_ns = time.perf_counter_ns()
    write_json(args.work_dir / 'phase_result.json', result)
    try:
        with recorder.span('r25.checks'):
            (args.work_dir / 'worker-source.py').write_text(Path(__file__).read_text())
            with _budget_guard(args, result):
                run_checks(args, recorder, result)
        result.update(status='ok', failed_phase=None)
    except WorkerFailure as exc:
        result.update(status=exc.status, error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    except (ImportError, FileNotFoundError) as exc:
        result.update(status='dependency_missing', error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    except MemoryError as exc:
        result.update(status='resource_limit', error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}'[:2000])
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, saved_sigterm)
        result.update(elapsed_ns=time.perf_counter_ns() - args._started_ns,
                      resource_ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      elapsed_scope='worker entry through checks and monitor cleanup; before terminal write')
        try:
            _checkpoint(args, result)
        except BaseException as exc:
            result['terminal_budget_error'] = f'{type(exc).__name__}: {exc}'[:2000]
            if result['status'] == 'ok':
                result.update(status=getattr(exc, 'status', 'error'), error=result['terminal_budget_error'])
            write_json(args.work_dir / 'phase_result.json', result)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
