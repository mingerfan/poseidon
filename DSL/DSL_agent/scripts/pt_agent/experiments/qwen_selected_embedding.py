"""Explicit compact embedding of actual saved polynomial selections.

Preparation inspects source JSON, file metadata and one stored NPY header.
The six real packed embedding projections are a separately scheduled run.
The original generation and square embedding entry points remain unchanged.
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time
import zipfile

from . import qwen_selector_validation as base
from . import qwen_monotone_selector as monotone
from .qwen_embedding_validation import map_stored_member
from .r10_paper_metrics import StageRecorder, read_events, summarize_events


MEMBER = 'original.model.embed_tokens.weight.npy'
SHAPE = (151936, 896)
SLOTS = 16384
RSS_BYTES = 2 * 1024 ** 3
SECONDS = 900


def inspect_member(path, member, np, expected_shape=SHAPE):
    """Read ZIP directory/NPY header metadata, without mapping tensor values."""
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if entry.filename == member]
        if len(entries) != 1:
            raise ValueError('Exactly one unchanged original embedding member is required')
        entry = entries[0]
    if entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1 or entry.file_size != entry.compress_size:
        raise ValueError('Only the original unencrypted stored NPY member is supported')
    with path.open('rb') as stream:
        stream.seek(entry.header_offset)
        header = stream.read(30)
        if len(header) != 30:
            raise ValueError('The original ZIP header is incomplete')
        name_size, extra_size = struct.unpack_from('<HH', header, 26)
        member_start = entry.header_offset + 30 + name_size + extra_size
        stream.seek(member_start)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('The saved NPY format version is unsupported')
        offset = stream.tell()
    dtype = np.dtype(dtype)
    nbytes = math.prod(shape) * dtype.itemsize
    if (tuple(shape) != tuple(expected_shape) or dtype != np.dtype('<f8') or dtype.hasobject
            or offset - member_start + nbytes != entry.file_size or offset + nbytes > path.stat().st_size):
        raise ValueError('Original embedding shape/dtype or stored-member bounds differ')
    return dict(path=str(path), member=member, shape=list(shape), dtype=str(dtype),
        fortran_order=fortran, offset_bytes=offset, tensor_bytes=nbytes,
        archive_bytes=path.stat().st_size, member_bytes=entry.file_size,
        tensor_values_read=False, tensor_mapping_created=False, integrity_checks=0)


def check_sources(args, np):
    selector_path = args.selector_result.resolve()
    selector = json.loads(selector_path.read_text())
    own_path = Path(selector['own_output_result']).resolve()
    own = json.loads(own_path.read_text())
    if (selector.get('status') != 'ok' or selector.get('variant') != monotone.VARIANT
            or selector.get('phase') != 'evaluate' or selector.get('config') != monotone.config()
            or selector.get('selection_attempts') != 6 or selector.get('selection_completed') != 6
            or selector.get('greedy_policy_passed') is not True
            or selector.get('all_comparison_contracts_passed') is not True
            or selector.get('coefficient_fits') != 0 or selector.get('candidate_frozen') is not False):
        raise ValueError('The actual six accepted, unfrozen analytic training selections are required')
    original_rows = base.source_rows(own_path)
    if selector.get('source_rows') != original_rows:
        raise ValueError('The original training-logit source identities differ')
    model_info = args.model_info.resolve()
    if Path(own['worker']['source']['model_info']).resolve() != model_info:
        raise ValueError('The original model source differs from the actual selector trajectories')
    if args.weights_archive.resolve() != model_info / 'graph/weights.npz':
        raise ValueError('The original model archive must remain at its saved model-info path')
    info = json.loads((model_info / 'model_info.json').read_text())
    config = info['source']['config']
    if (config.get('num_hidden_layers') != 24 or config.get('hidden_size') != SHAPE[1]
            or config.get('vocab_size') != SHAPE[0] or info.get('task') != 'causal_lm'):
        raise ValueError('The unchanged complete trained Qwen model identity differs')
    rows = selector.get('rows', [])
    if [(row.get('sample_id'), row.get('step')) for row in rows] != [(row['sample_id'], row['step']) for row in original_rows]:
        raise ValueError('The actual six training selection rows differ')
    selected = []
    for index, (row, original) in enumerate(zip(rows, original_rows)):
        filename = 'selection-' + str(index) + '.npy'
        path = selector_path.parent / filename
        if (row.get('completed') is not True or row.get('one_hot_file') != filename or not path.is_file()
                or row.get('quality', {}).get('token_agreement') is not True
                or row.get('quality', {}).get('actual_selected_token') != original['own_output_token']
                or row.get('contract', {}).get('comparison_gap_contract_passed') is not True):
            raise ValueError('An actual saved selection vector is missing or renamed')
        selected.append(dict(sample_id=original['sample_id'], step=original['step'], path=str(path),
            size_bytes=path.stat().st_size, original_token=original['own_output_token'],
            recorded_quality=row['quality']))
    return dict(selector_result=str(selector_path), own_result=str(own_path), model_info=str(model_info),
        weights_archive=str(args.weights_archive.resolve()), selections=selected,
        member_metadata=inspect_member(args.weights_archive, MEMBER, np),
        source_scope='Three raw unlabelled training sentences; no task instructions, chat wrapping or standard task quality claim')


def read_selection(row, np, expected_size=None):
    value = np.load(row['path'], mmap_mode='r', allow_pickle=False)
    size = SHAPE[0] if expected_size is None else expected_size
    if value.dtype != np.float64 or value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError('An unchanged finite float64 actual selection is required')
    token = int(value.argmax())
    if token != row['recorded_quality']['actual_selected_token']:
        raise ValueError('The actual saved selection token differs from its recorded observation')
    return value


def clear_mixture(selection, matrix, np, *, block_rows=256):
    """Independent block dot products retain every actual mixture weight."""
    if (selection.dtype != np.float64 or matrix.dtype != np.float64 or matrix.ndim != 2
            or selection.shape != (matrix.shape[0],) or not np.isfinite(selection).all()
            or type(block_rows) is not int or block_rows < 1):
        raise ValueError('The clear mixture requires matching original float64 values')
    output = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, matrix.shape[0], block_rows):
        part = matrix[start:start + block_rows]
        if not np.isfinite(part).all():
            raise ValueError('An original embedding value is nonfinite')
        output += selection[start:start + block_rows] @ part
    if not np.isfinite(output).all():
        raise ValueError('The independent clear mixture is nonfinite')
    return output


def compact_embedding(backend, selection, original_matrix):
    """Explicit adapter; the original generation entry remains untouched."""
    from ..kernels.compact_linear import embedding_from_selection
    return embedding_from_selection(backend, selection, original_matrix)


def packed_selection(backend, weights, np):
    from ..kernels.layout import layout_blocks, multi_flat_layout
    layout = multi_flat_layout(backend.slots, len(weights))
    vectors = []
    for start, stop, _ in layout_blocks(layout):
        vector = np.zeros(backend.slots, dtype=np.float64)
        vector[:stop-start] = weights[start:stop]
        if backend.batch is not None:
            if backend.batch != 1:
                raise ValueError('This explicit selection adapter supports one diagnostic sample')
            vector = vector[None, :]
        vectors.append(vector)
    backend.bind(vectors)
    return dict(one_hot=[backend.input(index) for index in range(len(vectors))], layout=layout,
                metadata=dict(vocabulary=len(weights)))


@contextmanager
def resident_guard(supervisor_pid, record):
    from .qwen_own_output_packed import _resident_snapshot
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise ValueError('A dedicated RSS guard timer is required')
    record.update(limit_bytes=RSS_BYTES, peak_observed_bytes=0, stopped=False)
    def check(*_):
        value = _resident_snapshot(supervisor_pid) * 1024
        record['peak_observed_bytes'] = max(record['peak_observed_bytes'], value)
        if value > RSS_BYTES:
            record['stopped'] = True
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise MemoryError('The selected-embedding subtree exceeded its2GiB RSS allowance')
    signal.signal(signal.SIGALRM, check)
    try:
        check()
        signal.setitimer(signal.ITIMER_REAL, .1, .1)
        yield
        check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def worker(args):
    np = base._numeric()
    out = args.out.resolve()
    recorder = StageRecorder(out / 'events.jsonl')
    result = dict(status='running', phase='selected-embedding', variant='actual-selection-compact-embedding-v1',
        started_ns=time.perf_counter_ns(), rows=[], resource_guard={}, fit_stages=[],
        actual_selection_reads=0, original_weight_members_mapped=0, full_weight_copies=0,
        clear_mixture_attempts=0, clear_mixture_completed=0, packed_embedding_attempts=0, packed_embedding_completed=0,
        source_checked=False, candidate_frozen=False, coefficient_fits=0,
        original_generation_defaults_changed=False, integrated_encrypted_generation_verified=False,
        mock_forwards_scope='Full Qwen decoder forwards; standalone compact embedding calls have their own counters',
        **{key: 0 for key in base.ZERO_FIELDS}, compiled_bootstraps=None, predicted_latency=None,
        CKKS_latency=None, accuracy_acceptance=None)
    checkpoint = lambda: base.save(out / 'result.json', result)
    matrix = None
    try:
        with resident_guard(args.supervisor_pid, result['resource_guard']), recorder.span('actual_selected_embedding'):
            with recorder.span('source_checks'):
                result['source'] = check_sources(args, np)
                result['source_checked'] = True
            with recorder.span('map_original_embedding'):
                matrix, mapping = map_stored_member(args.weights_archive, MEMBER)
                if tuple(matrix.shape) != SHAPE or matrix.flags.writeable:
                    raise ValueError('The original complete embedding must remain a readonly mapping')
                result.update(mapping=mapping, original_weight_members_mapped=1)
            from .qwen_mock_progress import ObservedStreamingMockBackend
            from .qwen_precision_quality import strict_math
            for index, source in enumerate(result['source']['selections']):
                weights = read_selection(source, np)
                result['actual_selection_reads'] += 1
                row = dict(sample_id=source['sample_id'], step=source['step'], actual_selected_token=int(weights.argmax()), completed=False)
                result['rows'].append(row)
                with recorder.span('independent_clear_mixture', sample_id=source['sample_id'], step=source['step']):
                    result['clear_mixture_attempts'] += 1
                    checkpoint()
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        reference = clear_mixture(weights, matrix, np)
                    result['clear_mixture_completed'] += 1
                backend = ObservedStreamingMockBackend(SLOTS)
                try:
                    with recorder.span('compact_packed_embedding', sample_id=source['sample_id'], step=source['step']):
                        selection = packed_selection(backend, weights, np)
                        result['packed_embedding_attempts'] += 1
                        checkpoint()
                        with np.errstate(over='raise', invalid='raise', divide='raise'):
                            values, layout = compact_embedding(backend, selection, matrix)
                        if len(values) != 1 or layout.size != SHAPE[1]:
                            raise ValueError('The original896-dimensional output layout differs')
                        observed = np.asarray(values[0].value[:SHAPE[1]])
                        row.update(strict_math=strict_math(reference, observed, np),
                            hard_greedy_embedding_max_difference=float(np.abs(observed-matrix[row['actual_selected_token']]).max()),
                            stats=backend.stats(), retained_constant_arrays=sum(value.data is not None for value in backend._constants.values()),
                            constant_identity_entries=len(backend._constant_objects))
                        row['reference_file'] = 'mixture-' + str(index) + '.npy'
                        row['packed_file'] = 'packed-' + str(index) + '.npy'
                        np.save(out / row['reference_file'], reference, allow_pickle=False)
                        np.save(out / row['packed_file'], observed, allow_pickle=False)
                        result['packed_embedding_completed'] += 1
                        checkpoint()
                finally:
                    backend.release_batch()
                if not row['strict_math']['passed'] or row['retained_constant_arrays'] or row['constant_identity_entries']:
                    raise ValueError('The actual mixed embedding failed original strict arithmetic or streaming retention policy')
                row['completed'] = True
                checkpoint()
            result.update(status='ok', actual_selected_embedding_math_verified=True,
                scope='Six actual saved approximate selection vectors through the original compact embedding; full generated-model and crypto execution unmeasured')
    except BaseException as exc:
        result.update(status='error', error=dict(type=type(exc).__name__, message=str(exc)))
    finally:
        if matrix is not None:
            matrix._mmap.close()
        result.update(ended_ns=time.perf_counter_ns(), worker_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            timing=summarize_events(read_events(out / 'events.jsonl')))
        result['worker_seconds'] = (result['ended_ns'] - result['started_ns']) / 1e9
        checkpoint()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'worker'), required=True)
    for flag in ('out', 'selector-result', 'model-info', 'weights-archive'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--supervisor-pid', type=int)
    args = parser.parse_args(argv)
    if args.phase == 'worker':
        if args.supervisor_pid is None:
            parser.error('The explicit supervisor PID is required')
        return 0 if worker(args)['status'] == 'ok' else 1
    args.out = args.out.resolve()
    protected = [args.selector_result.resolve().parent, args.model_info.resolve()]
    if any(args.out == source or source in args.out.parents for source in protected):
        raise ValueError('New diagnostics must preserve original selection and model directories')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.phase == 'prepare':
        source = check_sources(args, base._numeric())
        base.save(args.out / 'preparation.json', dict(status='prepared', source=source,
            real_packed_embeddings_started=0, model_forwards=0, count_runs=0,
            planned_limits=dict(rss_gib=2, as_gib=6, total_seconds=900),
            original_library_defaults_changed=False, actual_mixed_embedding_quality_measured=False))
        return 0
    from .r10_paper_metrics import run_process
    from .. import paths
    command = [str(paths.ORBIT_ROOT.parent / 'orbit-venv/bin/python'), '-B', '-m', __spec__.name,
        '--phase', 'worker', '--out', str(args.out), '--selector-result', str(args.selector_result),
        '--model-info', str(args.model_info), '--weights-archive', str(args.weights_archive),
        '--supervisor-pid', str(os.getpid())]
    base.save(args.out / 'command.json', command)
    observation = run_process(command, cwd=Path(__file__).resolve().parents[2], log=args.out / 'worker.log',
        resources=args.out / 'resources.jsonl', timeout=SECONDS, mem_gib=6, fsize_gib=.1,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                 MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'))
    base.save(args.out / 'supervision.json', dict(observation=observation, total_seconds=SECONDS))
    observed = json.loads((args.out / 'result.json').read_text()) if (args.out / 'result.json').is_file() else {}
    extra = {key: observed.get(key) for key in ('clear_mixture_attempts','clear_mixture_completed',
        'packed_embedding_attempts','packed_embedding_completed','actual_selected_embedding_math_verified')}
    base.record_run(args.out, 'selected-embedding', paths.RESULTS_DIR / 'registry.jsonl',
                    variant='actual-compact-embedding', extra_metrics=extra, run_id=args.out.name)
    return 0 if observation['exit_code'] == 0 and not observation['timed_out'] and not observation['remaining_pids'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
