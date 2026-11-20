"""Independent floating references for a frozen program and its RuntimePlan.

Bootstrap, modulus switching, rescaling and relinearization are mathematical
identities in this oracle. These operations do not perform encrypted execution.
All paths are explicit; only the three saved training probes are consumed.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time

from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json
from .r31_lattigo_input_worker import check_sources
from .r31_lattigo_runtime_plan import content_filename, plan_inventory

SCOPE = ('Independent float64 replay of the frozen R28 program and actual R29 RuntimePlan on '
         'saved normalized train:0, train:1 and train:2; boot/mod_switch/rescale/relinearize '
         'are mathematical identities only. No CKKS execution or model-quality selection.')


def check_math(reference, observed):
    import numpy as np
    a, b = np.asarray(reference), np.asarray(observed)
    if a.dtype != np.float64 or b.dtype != np.float64 or a.shape != b.shape:
        raise ValueError('Independent math dtype/shape differs')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Independent math produced nonfinite values')
    delta = np.abs(a-b); allowed = 1e-12+1e-12*np.abs(a)
    return dict(passed=bool(np.all(delta <= allowed)), shape=list(a.shape),
                max_absolute_error=float(delta.max(initial=0)),
                max_error_ratio=float((delta/allowed).max(initial=0)),
                criterion='1e-12+1e-12*abs(reference)')


def replay_plan(plan, manifest, bundle, packed, *, bootstrap_observer=None, checkpoint=None,
                bootstrap_vector_observer=None):
    """Last-use float64 replay; one raw plaintext in cache, no cipher emulation."""
    import numpy as np
    inventory = plan_inventory(plan, manifest)
    slots = inventory['slot_count']; data = np.asarray(packed)
    if data.dtype != np.float64 or data.shape != (slots,) or not np.isfinite(data).all():
        raise ValueError('Packed reference input must be finite original float64 slots')
    desc = {v['id']: v for v in plan['values']}
    encodes = {op['output']: op['payload']['content'] for op in plan['initialization']}
    last = {operand: i for i, op in enumerate(plan['execution']) for operand in op['inputs']}
    pool = {plan['external_inputs'][0]: data.copy()}; final = plan['final_outputs'][0]
    cache_id = None; cache_array = None; reads = 0; counts = Counter(); rows = []
    peak = len(pool); peak_bytes = sum(a.nbytes for a in pool.values())
    def plain(name):
        nonlocal cache_id, cache_array, reads
        content = encodes[name]
        if content != cache_id:
            path = Path(bundle)/'data'/content_filename(content)
            if path.stat().st_size != slots*8:
                raise ValueError('Raw plaintext byte length differs from original manifest')
            cache_array = None
            cache_array = np.fromfile(path, dtype='<f8', count=slots)
            if cache_array.shape != (slots,) or not np.isfinite(cache_array).all():
                raise ValueError('Invalid raw plaintext values')
            cache_array.flags.writeable = False; cache_id = content; reads += 1
        return cache_array
    for index, op in enumerate(plan['execution']):
        if checkpoint is not None and index % 256 == 0: checkpoint()
        kind = op['op']; inputs = op['inputs']; a = pool[inputs[0]]
        b = None if len(inputs) == 1 else (plain(inputs[1]) if desc[inputs[1]]['kind'] == 'plaintext' else pool[inputs[1]])
        with np.errstate(over='raise', invalid='raise'):
            if kind in ('add_cc', 'add_cp'): value = a+b
            elif kind in ('mul_cc', 'mul_cp'): value = a*b
            elif kind == 'rotate': value = np.roll(a, -op['attrs']['steps'])
            elif kind == 'negate': value = -a
            elif kind in ('rescale', 'mod_switch', 'relinearize', 'boot'):
                if kind == 'boot':
                    row = dict(index=len(rows), ordinal=op['ordinal'], input=inputs[0], output=op['output'],
                        input_level=desc[inputs[0]]['level'], target_level=desc[op['output']]['level'],
                        minimum=float(a.min()), maximum=float(a.max()), absmax=float(np.abs(a).max()),
                        scope='Pre-bootstrap mathematical values, not decrypted ciphertext values')
                    rows.append(row)
                    if bootstrap_observer is not None: bootstrap_observer(row)
                    if bootstrap_vector_observer is not None:
                        view=a.view(); view.flags.writeable=False
                        bootstrap_vector_observer(row,view)
                value = a.copy()
            else: raise ValueError('Unsupported independent operation: '+kind)
        if value.dtype != np.float64 or value.shape != (slots,) or not np.isfinite(value).all():
            raise ValueError('Nonfinite/malformed reference at ordinal '+str(op['ordinal']))
        pool[op['output']] = value; counts[kind] += 1
        peak = max(peak, len(pool)); peak_bytes = max(peak_bytes, sum(v.nbytes for v in pool.values()))
        for name in set(inputs):
            if last[name] == index and name != final: pool.pop(name, None)
        if op['output'] not in last and op['output'] != final: pool.pop(op['output'], None)
    if set(pool) != {final}: raise ValueError('Unexpected terminal floating values')
    return pool[final], dict(inventory=inventory, op_counts=dict(sorted(counts.items())),
        execution_completed=sum(counts.values()), plaintext_file_reads=reads, plaintext_cache_capacity=1,
        peak_live_arrays=peak, peak_live_array_bytes=peak_bytes, bootstrap_inputs=rows,
        reference_scope=SCOPE, actual_ckks_operations=0)


def run(args):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    args.work_dir.mkdir(parents=True, exist_ok=False)
    rec = StageRecorder(args.work_dir/'events.jsonl'); began = time.perf_counter_ns()
    result = dict(round='R31', mode='frozen-runtime-plan-independent-training-math', status='running',
        source_checked=False, training_samples=0, samples_completed=0, model_info_load_attempts=0,
        model_info_load_completed=0, probe_load_attempts=0, probe_load_completed=0,
        tensor_attempts=0, tensor_completed=0, mock_attempts=0, mock_completed=0,
        replay_attempts=0, replay_completed=0, comparisons=[], samples=[], error=None,
        mathematical_equivalence_verified=False, bootstrap_domain_acceptance=None,
        keygen_runs=0, encryption_runs=0, native_bootstrap_runs=0, decryption_runs=0,
        llm_calls=0, heldout_samples=0, confirm_samples=0, report_samples=0,
        extra_integrity_checks=0, security_acceptance=None, accuracy_acceptance=None, scope=SCOPE)
    def persist(): write_json(args.work_dir/'math-result.json', result)
    def checkpoint():
        if time.perf_counter_ns()-began > args.seconds*1e9: raise TimeoutError('Math preparation allowance exhausted')
        if shutil.disk_usage(args.work_dir).free < 10*1024**3: raise RuntimeError('Math preparation disk reserve exhausted')
        persist()
    persist()
    try:
        with rec.span('original_source_provenance'):
            result['source'] = check_sources(args)
            if Path(result['source']['model_info']).resolve() != args.model_info.resolve():
                raise ValueError('Explicit ModelInfo differs from original source')
            result['source_checked'] = True
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from ..models.info import ModelInfo
        from ..dsl.parser import parse
        from ..dsl.typecheck import check
        from ..lower import lower, pack_inputs, decode_outputs
        from ..tensor_sim import run_tensor
        from ..backends.mock import MockBackend
        from ..backends.base import _Plain, _vector
        from ..kernels.layout import layout_record
        class ReferenceMock(MockBackend):
            # Sequential constant names retain original arithmetic and avoid
            # the default backend's content-digest deduplication.
            def const(self, vec):
                data = _vector(vec, self.slots, broadcast=True)
                self.reference_constants = getattr(self, 'reference_constants', 0)+1
                return _Plain(self, data, 'reference-'+str(self.reference_constants))
        result['model_info_load_attempts'] += 1
        with rec.span('saved_original_model_info'):
            info = ModelInfo.load(args.model_info); weights = info.graph.weights
            result['model_info_load_completed'] += 1
            program = parse(args.program.read_text())
            checked = check(program, {k:v.shape for k,v in weights.items()}, graph=info.graph)
            if not checked.ok: raise ValueError('Original frozen program failed typecheck: '+str(checked.diags))
            plan = json.loads(args.plan.read_text()); manifest = json.loads((args.bundle/'manifest.json').read_text())
            result['inventory'] = plan_inventory(plan, manifest)
            if result['inventory']['bootstrap_count'] != 90 or result['inventory']['execution_count'] != 17063:
                raise ValueError('Expected original frozen 90-bootstrap compiled plan')
        result['probe_load_attempts'] += 1
        with rec.span('saved_training_probes'):
            with np.load(args.probes, allow_pickle=False) as archive:
                images = archive['input']; indices = archive['indices']
            if images.dtype != np.float64 or images.shape != (3,3,32,32) or indices.tolist() != [0,1,2] or not np.isfinite(images).all():
                raise ValueError('Saved original three training probes differ')
            result['probe_load_completed'] += 1; result['training_samples'] = 3
        for index in range(3):
            checkpoint(); sample_dir = args.work_dir/('train-'+str(index)); sample_dir.mkdir()
            sample = dict(sample_id='train:'+str(index), index=index, split='train', comparisons=[], status='running')
            result['samples'].append(sample); image = images[index:index+1]
            with rec.span('original_input_pack', sample=index):
                blocks = pack_inputs(program, [image])
                if len(blocks) != 1 or blocks[0].shape != (1,16384): raise ValueError('Original pack shape differs')
                packed = blocks[0][0]; input_path = sample_dir/'input.bin'
                packed.astype('<f8',copy=False).tofile(input_path)
                sample.update(input=str(input_path), packing=dict(layout='mux',gap=1,scalar_multiplier=1,logical_slots=16384))
            result['tensor_attempts'] += 1; persist()
            with rec.span('frozen_program_tensor', sample=index):
                tensor = run_tensor(program, weights, image, cache=False, act_mode='program')
                result['tensor_completed'] += 1
                if tensor.dtype != np.float64 or tensor.shape != (1,10) or not np.isfinite(tensor).all():
                    raise ValueError('Frozen tensor output differs')
            result['mock_attempts'] += 1; persist()
            backend = ReferenceMock(slots=16384, batch=1, stream_constants=True)
            try:
                with rec.span('frozen_program_packed_mock', sample=index):
                    backend.bind(blocks); lowered, layouts = lower(program, weights, backend, act_mode='program')
                    decoded = decode_outputs(backend.outputs, layouts)
                    if len(decoded) != 1: raise ValueError('Expected one original model output')
                    mock = decoded[0].copy(); result['mock_completed'] += 1
                    decoded_indices = decode_outputs([np.arange(16384,dtype=np.float64)[None,:]], layouts)[0].reshape(-1)
                    if decoded_indices.shape != (10,) or not np.array_equal(decoded_indices,decoded_indices.astype(np.int64)):
                        raise ValueError('Output decoder did not select ten scalar slots')
                    slots = decoded_indices.astype(np.int64).tolist()
                    sample['decode_indices'] = slots; sample['output_layouts'] = [layout_record(lay) for lay in layouts]
                    sample['mock_constant_naming'] = 'Sequential identifiers; original arithmetic; no digest deduplication'
                    del lowered
            finally: backend.release_batch(); del backend; gc.collect()
            comparison = check_math(tensor, mock); comparison.update(sample=index,pair='tensor_vs_packed_mock')
            sample['comparisons'].append(comparison); result['comparisons'].append(comparison); persist()
            if not comparison['passed']: raise ValueError('Frozen program Tensor/Mock math mismatch')
            result['replay_attempts'] += 1; persist()
            boot_file = sample_dir/'bootstrap-inputs.jsonl'
            def boot_row(row):
                with boot_file.open('a') as stream:
                    stream.write(json.dumps(dict(row,sample_id=sample['sample_id']),allow_nan=False)+'\n'); stream.flush()
            with rec.span('compiled_runtime_plan_independent_replay', sample=index):
                replay, replay_summary = replay_plan(plan, manifest, args.bundle, packed,
                    bootstrap_observer=boot_row, checkpoint=checkpoint)
                result['replay_completed'] += 1
                replay_logits = replay[slots][None,:]
                if len(replay_summary['bootstrap_inputs']) != 90: raise ValueError('Expected 90 independent bootstrap inputs')
                sample['replay_summary'] = replay_summary
                sample['bootstrap_inputs'] = str(boot_file)
                sample['bootstrap_input_absmax'] = max(r['absmax'] for r in replay_summary['bootstrap_inputs'])
            for name, reference in (('packed_mock_vs_runtime_plan',mock),('tensor_vs_runtime_plan',tensor)):
                comparison = check_math(reference,replay_logits); comparison.update(sample=index,pair=name)
                sample['comparisons'].append(comparison); result['comparisons'].append(comparison); persist()
                if not comparison['passed']: raise ValueError('Compiled RuntimePlan independent math mismatch: '+name)
            expected = sample_dir/'expected.json'
            write_json(expected, dict(indices=slots,values=mock.reshape(-1).tolist(),scope=SCOPE,
                sample_id=sample['sample_id'],tensor_values=tensor.reshape(-1).tolist(),
                replay_values=replay_logits.reshape(-1).tolist(),math_comparisons=sample['comparisons']))
            sample.update(status='ok',expected=str(expected)); result['samples_completed'] += 1
            persist(); del tensor,mock,replay,replay_logits,blocks,packed; gc.collect()
        result.update(status='ok',mathematical_equivalence_verified=True)
    except BaseException as exc: result.update(status='error',error=type(exc).__name__+': '+str(exc))
    result['elapsed_ns'] = time.perf_counter_ns()-began
    result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['timing'] = summarize_events(read_events(args.work_dir/'events.jsonl'))
    persist(); return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','bundle','program','model-info','probes','probe-source','r13-result','compilation-result','work-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=1800)
    args=p.parse_args(argv); result=run(args)
    print(json.dumps(dict(status=result['status'],samples_completed=result['samples_completed'],error=result['error'])))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__': raise SystemExit(main())
