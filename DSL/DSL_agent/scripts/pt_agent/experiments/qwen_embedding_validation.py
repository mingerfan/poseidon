"""Single-member original Qwen embedding mapping and streaming diagnostics.

The input is an exact one-hot for an original-tokenizer training-text token.
This checks one original-weight linear projection; no Qwen forward, calibration,
polynomial selection, model accuracy, compilation or encrypted execution runs.
"""

import argparse
import gc
import json
from pathlib import Path
import resource
import struct
import zipfile

from .r10_paper_metrics import StageRecorder, write_json


def map_stored_array(path, member, *, required_rank=None, _entry=None):
    """Map one uncompressed NPY member without extracting or materializing it.

    Read ZIP/NPY format metadata only; do not invoke CRC or digest validation.
    """
    import numpy as np
    path = Path(path).resolve()
    if _entry is None:
        with zipfile.ZipFile(path) as archive:
            matches = [item for item in archive.infolist() if item.filename == member]
        if len(matches) != 1:
            raise ValueError('The original archive requires exactly one selected member')
        item = matches[0]
    else:
        item = _entry
        if item.filename != member:
            raise ValueError('The selected archive metadata names a different member')
    if item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1 or item.compress_size != item.file_size:
        raise ValueError('Direct original-weight mapping requires an unencrypted stored NPY member')
    with path.open('rb') as stream:
        stream.seek(item.header_offset)
        local_header = stream.read(30)
        if len(local_header) != 30:
            raise ValueError('The ZIP local header is incomplete')
        filename_length, extra_length = struct.unpack_from('<HH', local_header, 26)
        member_start = item.header_offset + 30 + filename_length + extra_length
        stream.seek(member_start)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Only existing NPY versions 1 and 2 are supported')
        offset = stream.tell()
    dtype = np.dtype(dtype)
    if (dtype != np.dtype('<f8') or dtype.hasobject or any(size < 1 for size in shape)
            or required_rank is not None and len(shape) != required_rank):
        raise ValueError('The original selected member requires float64 arrays with the required dimensions')
    import math
    nbytes = math.prod(shape)*dtype.itemsize
    if offset-member_start+nbytes != item.file_size or offset+nbytes > path.stat().st_size:
        raise ValueError('The selected member dimensions do not fit its recorded storage')
    matrix = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape, order='F' if fortran else 'C')
    return matrix, dict(path=str(path), member=member, shape=list(shape), dtype=str(dtype),
        fortran_order=fortran, offset_bytes=offset, member_bytes=item.file_size,
        tensor_bytes=nbytes, archive_bytes=path.stat().st_size, direct_readonly_mapping=True,
        full_tensor_copies=0, other_weight_members_loaded=0, integrity_checks=0)


def map_stored_member(path, member):
    return map_stored_array(path, member, required_rank=2)


def _selection(backend, vocabulary, token_id, *, bind):
    import numpy as np
    from ..kernels.layout import layout_blocks, multi_flat_layout
    layout = multi_flat_layout(backend.slots, vocabulary)
    vectors = []
    for start, stop, _ in layout_blocks(layout):
        vector = np.zeros(backend.slots, dtype=np.float64)
        if start <= token_id < stop:
            vector[token_id-start] = 1.
        vectors.append(vector)
    if bind:
        backend.bind(vectors)
    values = [backend.input(index) for index in range(len(vectors))]
    return dict(one_hot=values, layout=layout, metadata=dict(vocabulary=vocabulary))


def tokenize_probe(args, recorder, result):
    from transformers import AutoTokenizer
    with recorder.span('tokenize_original_training_text'):
        source = json.loads(args.source_rows.read_text())
        if (not Path(source['source']).name.startswith('train-') or source['column'] != 'sentence'
                or source['row_number_base'] != 0 or not source['source_rows']):
            raise ValueError('The original saved text must retain its SST-2 training provenance')
        with args.samples.open() as stream:
            text = json.loads(stream.readline())['text']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('A nonempty original training text is required')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
        encoded = tokenizer(text, padding=False, truncation=False)
        ids = encoded['input_ids']
        if not ids or any(type(value) is not int for value in ids):
            raise ValueError('The original tokenizer must return native integer token IDs')
        result['token_probe'] = dict(token_id=ids[0], token_ids=ids, text=text,
            original_tokenizer=str(args.tokenizer.resolve()), source_samples=str(args.samples.resolve()),
            original_data_source=source['source'], original_training_row=source['source_rows'][0],
            original_text_index=0, original_column='sentence', normalization='unchanged native text',
            tokenization=dict(padding=False, truncation=False), text_rows_read=1,
            purpose='exact original-token one-hot embedding mathematical regression',
            calibration=False, accuracy_evaluation=False)
    result['status'] = 'ok'


def run(args, recorder, result):
    import numpy as np
    from ..backends import CountBackend
    from ..backends.streaming_mock import StreamingMockBackend
    variant = getattr(args, 'linear_variant', 'original-streaming')
    if variant == 'compact-rectangular':
        from ..kernels.compact_linear import embedding_from_selection, POLICY
    elif variant == 'original-streaming':
        from ..kernels.encrypted_generation import embedding_from_selection
        from ..kernels.streaming_linear import POLICY
    else:
        raise ValueError('Unknown explicitly selected embedding schedule')
    result['linear_variant'] = variant
    with recorder.span('source_check'):
        info = json.loads(args.model_info.read_text())
        config = json.loads(args.config.read_text())
        if (config.get('num_hidden_layers') != 24 or config.get('hidden_size') != 896
                or config.get('vocab_size') != 151936 or config.get('tie_word_embeddings') is not True
                or info['source']['config']['num_hidden_layers'] != 24):
            raise ValueError('The original full Qwen2-0.5B identity differs')
        token_probe = json.loads(args.token_record.read_text())['token_probe']
        token = token_probe['token_id']
        if (type(token) is not int or not 0 <= token < config['vocab_size']
                or token_probe['calibration'] is not False or token_probe['text_rows_read'] != 1
                or not Path(token_probe['original_data_source']).name.startswith('train-')):
            raise ValueError('The original text token must belong to the unchanged vocabulary')
        if getattr(args, 'expected_token', None) is not None and token != args.expected_token:
            raise ValueError('The saved original-tokenizer training token differs from the diagnostic source')
        result['token_probe'] = token_probe
        result['configured_token'] = dict(source=str(args.config.resolve()), bos_token_id=config.get('bos_token_id'),
            eos_token_id=config.get('eos_token_id'), semantic='auxiliary original model identity metadata')
    with recorder.span('map_original_embedding'):
        matrix, mapping = map_stored_member(args.weights_archive, args.member)
        if tuple(matrix.shape) != (config['vocab_size'], config['hidden_size']):
            raise ValueError('The original embedding member shape differs')
        result['mapping'], result['mapping_completed'], result['original_weight_members_mapped'] = mapping, 1, 1
        result['streaming_policy'] = dict(POLICY)
    write_json(args.work_dir/'worker-result.json', result)
    with recorder.span('count_original_embedding'):
        result['count_attempts'] = 1
        write_json(args.work_dir/'worker-result.json', result)
        backend = CountBackend(16384)
        selection = _selection(backend, matrix.shape[0], token, bind=False)
        values, output_layout = embedding_from_selection(backend, selection, matrix)
        backend.output(values)
        result['count_stats'] = backend.stats()
        result['count_constant_identity_entries'] = len(backend._constant_objects)
        result['count_retained_array_constants'] = sum(value.data is not None for value in backend._constants.values())
        if result['count_constant_identity_entries'] or result['count_retained_array_constants']:
            raise ValueError('The streaming Count unexpectedly retained constant arrays')
        result['count_completed'] = 1
        del values, selection, backend
        gc.collect()
    write_json(args.work_dir/'worker-result.json', result)
    with recorder.span('math_original_embedding'):
        result['math_attempts'] = 1
        write_json(args.work_dir/'worker-result.json', result)
        backend = StreamingMockBackend(16384)
        selection = _selection(backend, matrix.shape[0], token, bind=True)
        values, output_layout = embedding_from_selection(backend, selection, matrix)
        if len(values) != 1 or output_layout.size != config['hidden_size']:
            raise ValueError('The original embedding output layout differs')
        observed = np.asarray(values[0].value[:config['hidden_size']])
        reference = np.asarray(matrix[token])
        errors = np.abs(observed-reference)
        tolerances = 1e-12+1e-12*np.abs(reference)
        passed = bool(np.isfinite(observed).all() and np.all(errors <= tolerances))
        result['math'] = dict(passed=passed, maximum_absolute_error=float(errors.max()),
            maximum_tolerance_ratio=float((errors/tolerances).max()),
            criterion='abs(observed-reference)<=1e-12+1e-12*abs(reference)',
            output_elements=len(observed), all_finite=bool(np.isfinite(observed).all()),
            observed=observed.tolist(), reference=reference.tolist(),
            reference_semantics='the original embedding row selected by an original-tokenizer training-text token',
            constant_identity_entries=len(backend._constant_objects),
            retained_array_constants=sum(value.data is not None for value in backend._constants.values()))
        if not passed or result['math']['constant_identity_entries'] or result['math']['retained_array_constants']:
            raise ValueError('Original-weight streaming embedding mathematical check failed')
        result['math_completed'] = 1
    result['status'] = 'ok'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('tokenize','validate'), default='validate')
    parser.add_argument('--work-dir', required=True, type=Path)
    for flag in ('model-info','config','weights-archive','token-record','tokenizer','samples','source-rows'):
        parser.add_argument('--'+flag, type=Path)
    parser.add_argument('--member')
    parser.add_argument('--linear-variant', choices=('original-streaming','compact-rectangular'),
                        default='original-streaming')
    parser.add_argument('--expected-token', type=int)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', mapping_completed=0, count_attempts=0, count_completed=0,
        math_attempts=0, math_completed=0, full_qwen_forwards=0, original_model_loads=0,
        original_weight_members_mapped=0, real_text_reads=0, calibration_samples=0,
        heldout_samples=0, confirmation_samples=0, report_samples=0, api_calls=0,
        polynomial_fits=0, keygen_runs=0, ckks_executions=0, compiler_runs=0,
        accuracy_acceptance=None, compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None)
    recorder = StageRecorder(args.work_dir/'events.jsonl')
    try:
        with recorder.span('qwen_embedding_diagnostic'):
            if args.phase == 'tokenize':
                tokenize_probe(args, recorder, result)
                result['real_text_reads'] = 1
            else:
                run(args, recorder, result)
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:2000])
    result['worker_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.work_dir/'worker-result.json', result)
    print(json.dumps({key:result[key] for key in ('status','count_completed','math_completed','worker_maxrss_kib')}))
    return 0 if result['status']=='ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
