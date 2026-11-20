"""Continue the accepted packed prefill using its actual output token.

The first packed result is cached evidence.  This phase executes only the
second complete packed forward, against the saved own-output Tensor result.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import signal

from . import qwen_precision_quality as fixed
from .qwen_autoregressive_quality import advance_state
from .r10_paper_metrics import StageRecorder, write_json


def source_gate(args):
    own = json.loads(args.own_output_result.read_text())
    cached = json.loads(args.prefill_result.read_text())
    w, first = own.get('worker') or {}, cached.get('worker') or {}
    if (own.get('status') != 'ok' or own.get('job') != 'quality-own-tensor'
            or w.get('phase') != 'own-output-tensor' or w.get('own_output_generation_completed') is not True
            or w.get('approximate_tokens_fed_back') is not True
            or any(w.get(k) != n for k,n in dict(tensor_forward_attempts=6,tensor_forward_completed=6,
                state_update_attempts=3,state_update_completed=3).items())):
        raise ValueError('The original complete actual own-output Tensor trajectory is required')
    if (cached.get('status') != 'ok' or cached.get('job') != 'quality-mock'
            or first.get('mock_forward_attempts') != 1 or first.get('mock_forward_completed') != 1
            or first.get('independent_packed_math_passed') is not True
            or first.get('strict_tensor_mock_math',{}).get('passed') is not True
            or first.get('count_mock_schedule_matched') is not True
            or first.get('mock_sample',{}).get('sample_id') != 'train:0'
            or first.get('mock_sample',{}).get('step') != 0):
        raise ValueError('The accepted actual train:0 prefill packed result is required')
    for worker in (w, first):
        source = worker.get('source') or {}
        if (Path(source['count_result']).resolve() != args.count_result.resolve()
                or Path(source['model_info']).resolve() != args.model_info.resolve()
                or Path(source['original_hf_reference']).resolve() != args.reference.resolve()):
            raise ValueError('Both cached stages must use the same original model and fixed candidate')
    if args.prefill_logits.resolve() != (args.prefill_result.parent/'worker/mock-logits.npy').resolve():
        raise ValueError('Explicit logits must be the actual cached packed prefill output')
    rows = w.get('sample_rows') or []
    selected = [row for row in rows if row.get('sample_id') == 'train:0']
    if len(rows) != 6 or [row.get('step') for row in selected] != [0,1]:
        raise ValueError('Preserve exactly the actual first training trajectory and its two output records')
    output = Path(selected[1]['logits_file'])
    if output.name != str(output):
        raise ValueError('Own-output reference filename must remain local to its worker')
    reference = args.own_output_result.parent/'worker'/output
    for path in (reference,args.prefill_logits):
        if not path.is_file():
            raise ValueError('A required complete saved logits vector is absent')
    return selected, reference, dict(own_output_result=str(args.own_output_result.resolve()),
        cached_prefill_result=str(args.prefill_result.resolve()),
        cached_prefill_logits=str(args.prefill_logits.resolve()),tensor_step1_reference=str(reference.resolve()),
        clock_scope='Cached first packed step plus one newly executed second packed step; cold two-step generation clock unmeasured',
        tensor_source_training_quality_passed=w.get('training_quality_passed'),
        tensor_source_step1_domains_contained=selected[1].get('all_domains_contained'))


def handoff(rows, prefill_logits, np):
    if (prefill_logits.dtype != np.float64 or prefill_logits.shape != (1,1,151936)
            or not np.isfinite(prefill_logits).all()):
        raise ValueError('Cached actual packed logits must retain finite original float64 vocabulary output')
    token = int(prefill_logits.argmax())
    if token != rows[0]['approximate_greedy_token']:
        raise ValueError('Actual packed prefill greedy token differs from the preserved own-output Tensor source')
    current = advance_state(rows[0]['input_state'], token, vocabulary_size=prefill_logits.shape[-1])
    fields = ('prefix_token_ids','prefix_length','attention_mask','sel')
    if any(current[key] != rows[1]['input_state'][key] for key in fields):
        raise ValueError('Actual packed-output-derived state differs from the saved own-output second input')
    return current, dict(actual_packed_greedy_token=token,mock_argmax_matches_tensor=True,
        actual_second_input_matches_tensor=True,source='Actual cached packed prefill argmax, original step0 state and original embedding lookup',
        sample_id='train:0',step=1,prefix_length=current['prefix_length'])


def _resident_snapshot(supervisor_pid):
    from .r10_paper_metrics import _scan
    table = _scan()
    ancestor = os.getpid(); visited = set()
    while ancestor in table and ancestor not in visited and ancestor != supervisor_pid:
        visited.add(ancestor); ancestor = table[ancestor]['ppid']
    if ancestor != supervisor_pid:
        raise ValueError('RSS supervisor must be an ancestor of this isolated worker')
    family = {supervisor_pid}
    while True:
        children = {pid for pid,row in table.items() if row['ppid'] in family}
        if children.issubset(family):break
        family.update(children)
    return sum(table[pid].get('vmrss_kib',0) for pid in family if pid in table)


@contextmanager
def resident_guard(supervisor_pid, observation):
    if signal.getitimer(signal.ITIMER_REAL) != (0.,0.):
        raise RuntimeError('The independent resident guard requires an unused timer')
    old = signal.getsignal(signal.SIGALRM)
    observation.update(resident_limit_gib=5,samples=0,peak_tree_rss_kib=0,stopped=False,
        scope='Explicit diagnostic supervisor and all observed descendants, including Nix launchers')
    def check(signum=None,frame=None):
        rss = _resident_snapshot(supervisor_pid)
        observation['samples'] += 1; observation['peak_tree_rss_kib'] = max(observation['peak_tree_rss_kib'],rss)
        if rss > 5*1024**2:
            observation['stopped'] = True
            signal.setitimer(signal.ITIMER_REAL,0.)
            raise MemoryError('Own-output packed continuation exceeds5GiB resident allowance')
    signal.signal(signal.SIGALRM,check)
    try:
        check();signal.setitimer(signal.ITIMER_REAL,.1,.1)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);signal.signal(signal.SIGALRM,old)


def run_check(args, recorder, result, np):
    from .qwen_mock_progress import ObservedStreamingMockBackend, progress_records
    from ..kernels.compact_linear import compact_transformer_projections
    from ..lower import lower, pack_inputs, decode_outputs
    with recorder.span('cached_packed_output_state_handoff'):
        rows, reference_path, summary = source_gate(args)
        prefill = np.load(args.prefill_logits,allow_pickle=False,mmap_mode='r')
        current, metadata = handoff(rows,prefill,np)
        result.update(handoff=metadata,cached_sources=summary)
        info, program, _ = fixed._load(args,result)
        arrays = fixed.arrays_for(info,current,np)
        reference = np.load(reference_path,allow_pickle=False,mmap_mode='r')
        result['actual_input'] = dict(prefix_length=current['prefix_length'],
            input_shapes={key:list(value.shape) for key,value in arrays.items()})
    backend = ObservedStreamingMockBackend(program.header.slots,batch=1)
    result['mock_forward_attempts'] = 1
    try:
        with recorder.span('actual_own_output_step1_packed_math',sample_id='train:0',step=1):
            backend.bind(pack_inputs(program,[arrays[param.name] for param in program.params]))
            with progress_records(backend,program,args.work_dir,result),compact_transformer_projections():
                values,layouts = lower(program,info.graph.weights,backend)
            observed = decode_outputs([value.value for value in values],layouts)[0]
            result['strict_tensor_mock_math'] = fixed.strict_math(reference,observed,np)
            result['mock_stats'] = backend.stats()
            result['output_layout_blocks'] = [dict(start=start,end=end,valid_size=block.size)
                for layout in layouts for start,end,block in layout.blocks]
            counted = json.loads(args.count_result.read_text())['worker']['count_stats']
            result['count_mock_schedule_matched'] = all(result['mock_stats'][key] == counted[key]
                for key in ('counts','ops_total','distinct_consts','rot_steps'))
            result['retained_constant_arrays'] = sum(item.data is not None for item in backend._constants.values())
            result['constant_identity_entries'] = len(backend._constant_objects)
            result['mock_forward_completed'] = 1
            np.save(args.work_dir/'own-step1-mock-logits.npy',observed,allow_pickle=False)
    finally:
        result['partial_mock_stats'] = backend.stats()
        backend.release_batch()
    if (not result['strict_tensor_mock_math']['passed'] or not result['count_mock_schedule_matched']
            or result['retained_constant_arrays'] or result['constant_identity_entries']):
        raise ValueError('Actual-output second packed forward must retain original strict arithmetic and streaming schedule')
    result.update(status='ok',independent_packed_math_passed=True,cached_first_step_continuation_verified=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work-dir','count-result','model-info','reference','own-output-result','prefill-result','prefill-logits'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--supervisor-pid',type=int,required=True)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True,exist_ok=True)
    if (args.work_dir/'worker-result.json').exists():
        raise FileExistsError('Preserve the original packed continuation result')
    result = dict(status='running',phase='own-output-step1-packed',model='Qwen2-0.5B',
        source_checked=False,candidate_frozen=False,mock_forward_attempts=0,mock_forward_completed=0,
        tensor_forward_attempts=0,tensor_forward_completed=0,api_calls=0,fresh_hf_forwards=0,
        approximation_selection_runs=0,coefficient_reconstruction_calls=None,
        coefficient_policy='Original deterministic evaluators reconstruct fixed selected coefficients; call count unmeasured',
        emit_runs=0,compiler_runs=0,keygen_runs=0,ckks_executions=0,
        heldout_samples=0,confirm_samples=0,report_samples=0,independent_packed_math_passed=False,
        cached_first_step_continuation_verified=False,full_ciphertext_model_binding_verified=False,
        compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None,resident_guard={},
        scope='One new train:0 second packed forward from accepted actual cached packed prefill argmax; cold two-step clock and integrated encrypted selector/state remain unmeasured')
    recorder = StageRecorder(args.work_dir/'events.jsonl')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    try:
        import numpy as np
        with resident_guard(args.supervisor_pid,result['resident_guard']),recorder.span('qwen_cached_prefill_step1_packed'):
            run_check(args,recorder,result,np)
    except BaseException as exc:
        result.update(status='error',error_type=type(exc).__name__,error=str(exc)[:3000])
    result['worker_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.work_dir/'worker-result.json',result)
    print(json.dumps(dict(status=result['status'],mock_forward_completed=result['mock_forward_completed'])))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':
    raise SystemExit(main())
