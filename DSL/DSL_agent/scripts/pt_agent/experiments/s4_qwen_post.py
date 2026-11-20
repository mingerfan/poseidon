"""Post-freeze Qwen report and semantic count, isolated from the Agent process."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .. import paths
from . import s4_qwen_block as s4
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json
from .s2_heir_serial import record_registry

MODULE = 'pt_agent.experiments.s4_qwen_post'
read = s4.read


def eligibility(source):
    frozen = source.get('frozen') is True
    mathematical = (source.get('phases', {}).get('math') or {}).get('math_passed') is True
    return dict(count=frozen, report=frozen and mathematical,
                compiled=source.get('compile_accepted') is True, strict_mathematics=mathematical)


def report(args, info, program, recorder):
    import numpy as np
    from ..analyzer import reference_forward, approximation_points
    from ..planners.agent_tools import AgentContext
    from ..transformer_validation import validate
    from ..metrics import passes
    from .s4_qwen_cold import BlockSamples
    source = BlockSamples(args.source_case / 'model', info, report=True)
    with recorder.span('isolated_fixed_report_inputs'):
        arrays, labels = source.get('report')
    if labels is not None or len(arrays['inputs_embeds']) != 160:
        raise ValueError('Keep all160 original unlabelled report texts')
    with recorder.span('original_canonical_block_plaintext_reference'):
        reference = []
        for index in range(0, 160, 20):
            values = reference_forward(info.graph, {key: value[index:index+20] for key, value in arrays.items()})
            reference.append(values[info.graph.outputs[0]])
        exact = np.concatenate(reference)
    keys = [name for _, name in approximation_points(info.graph)]
    ctx = AgentContext(graph=info.graph, weights=info.graph.weights,
        calibration=read(args.source_case / 'model/calibration.json'),
        weight_shapes={key: value.shape for key, value in info.graph.weights.items()},
        val_images=arrays['inputs_embeds'], val_exact_logits=exact, val_labels=None, val_exact_top1=None,
        probe_images=arrays['inputs_embeds'][:4], probe_act_inputs=[], needed_margin={}, act_tensors=keys,
        model_name=info.graph.name, normalization=None, model_info=info, samples=source, val_inputs=arrays,
        val_split='report', prompt_version='r8-v11', placement='program-plan', tune_search='sweep')
    captured = []
    with recorder.span('frozen_tensor_report'):
        measured = validate(program, ctx, engine='tensor', stop_on_divergence=False, output_callback=captured.append)
    if len(captured) != 1 or captured[0].shape != exact.shape:
        raise ValueError('Original fixed report output is incomplete')
    approx = captured[0]
    finite = bool(np.isfinite(approx).all())
    np.savez(args.work_dir / 'report-outputs.npz', reference=exact, approximate=approx)
    return dict(status='ok', report_completed=True, report_samples=160, sample_accesses=source.accesses,
        task_policy=info.threshold, task_policy_passed=bool(finite and not measured['diverged'] and passes(measured, info.threshold)),
        validation_metrics=measured, finite=finite,
        source_program=str(args.source_case / 'session/program.ptdsl'),
        reference_scope='Original canonical float64 layer0 graph, matching the BERT isolated-report protocol; not complete-decoder language-model accuracy',
        full_decoder_LM_quality=None, report_feedback_to_Agent=False, CKKS_runs=0, API_calls=0)


def count(args, info, program, recorder):
    from ..plan import plan_program, verify_program
    from .s3_bert_components import aggregate
    with recorder.span('fresh_semantic_Count_placement'):
        plan, dag, placed = plan_program(program, info.graph.weights, profile='qwen48', frontend_normalize=True)
        boots, statements, input_owned = aggregate(program, dag.prog_op, placed.boot_nodes)
        if sum(boots.values()) != plan['boots'] or placed.boots != plan['boots']:
            raise ValueError('Semantic attribution differs from actual plan')
        write_json(args.work_dir / 'refresh-plan.json', plan)
        write_json(args.work_dir / 'statement-counts.json', statements)
        replay = verify_program(program, info.graph.weights, plan, profile='qwen48')
    compiled = read(args.source_case / 'result.json')['compiled_bootstraps']
    return dict(status='ok', label='arranger', boots=boots, total=replay['boots'],
        placement=program.header.placement, source=str(args.source_case / 'session/program.ptdsl'),
        compiled_reference_total=compiled, difference_from_compiled_reference=None if compiled is None else replay['boots']-compiled,
        compiled_reference_matches=None if compiled is None else replay['boots'] == compiled, plan_total=plan['boots'],
        feasible=None if compiled is None else replay['boots'] == compiled,
        CountExpr_replay=replay, input_owned_bootstraps=input_owned,
        statement_counts=str(args.work_dir / 'statement-counts.json'),
        attribution_scope='Same normalized cost DAG value ownership as S3-BERT; actual discrepancies are preserved',
        sample_reads=0, API_calls=0, CKKS_runs=0, compiler_attempts=0)


def worker(args):
    from .r29_local_compile_matrix import native_metadata_commands
    result = dict(status='running', error=None, phase=args.worker, API_calls=0, CKKS_runs=0)
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    try:
        gate = read(args.gate)
        if not gate['eligibility'][args.worker] or Path(gate['source_case']).resolve() != args.source_case.resolve():
            raise ValueError('Require the closed frozen block and the phase-specific mathematical acceptance')
        with native_metadata_commands(read(args.metadata_template)):
            from .bert_sst2_worker import initialize_numeric
            initialize_numeric()
            from .s4_qwen_cold import checked_program, original_numeric_scope
            args.model_dir, args.session_dir = args.source_case / 'model', args.source_case / 'session'
            info, program = checked_program(args)
            with original_numeric_scope():
                result.update(report(args, info, program, recorder) if args.worker == 'report' else count(args, info, program, recorder))
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
    result['timing'] = summarize_events(read_events(recorder.path))
    write_json(args.work_dir / 'result.json', result)
    return int(result['status'] != 'ok')


def main(args):
    if args.worker:
        return worker(args)
    from .a5_a6_rule63_search import guarded_process
    if args.resume_wait:
        previous = read(args.out / 'progress.json')
        if previous.get('status') != 'waiting_for_original_cold_chain' or (args.out / 'result.json').exists():
            raise ValueError('Only untouched post-processing wait metadata may resume')
    else:
        args.out.mkdir(exist_ok=False)
    while time.time() < s4.DEADLINE:
        if ((args.source_case / 'completion.json').is_file() and (args.publisher / 'result.json').is_file()
                and (args.prior_case is None or (args.prior_case / 'result.json').is_file())):
            break
        write_json(args.out / 'progress.json', dict(status='waiting_for_original_cold_chain', source=str(args.source_case)))
        time.sleep(5)
    source = read(args.source_case / 'result.json')
    state = dict(status='running', source_case=str(args.source_case), phases={},
                 compile_accepted=source.get('compile_accepted') is True,
                 eligibility=eligibility(source),
                 API_calls=0, compiler_attempts=0, CKKS_runs=0)
    if not state['eligibility']['count'] or time.time() >= s4.DEADLINE:
        state.update(status='not_run', reason='Publication cutoff reached' if time.time() >= s4.DEADLINE else
                     'No compiled frozen candidate from the one authorized additional cold run',
                     report=dict(status='not_run'), semantic_count=dict(status='not_run'))
        write_json(args.out / 'result.json', state)
    else:
        with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            from .s4_qwen_cold import frozen_state
            frozen_state(args.source_case)
            write_json(args.out / 'closed-source-gate.json', state)
            for phase in ('report', 'count'):
                if not state['eligibility'][phase]:
                    state['phases'][phase] = dict(status='not_run', reason='Strict post-freeze mathematics did not complete')
                    continue
                budget = s4.remaining_before_deadline(1800)
                if shutil.disk_usage(args.out).free < 10*s4.GIB:
                    raise RuntimeError('Post-freeze work requires the original10GiB disk reserve')
                case = args.out / (args.out.name + '-' + phase)
                case.mkdir()
                started = time.perf_counter_ns()
                state.update(active_phase=phase)
                write_json(args.out / 'progress.json', state)
                argv = [sys.executable, '-B', '-m', s4.MODULE, '--enter', '--seconds', str(budget),
                    '--python', str(paths.INTAKE_PYTHON), '--', '-B', '-m', MODULE, '--worker', phase,
                    '--source-case', str(args.source_case), '--work-dir', str(case),
                    '--gate', str(args.out / 'closed-source-gate.json'), '--metadata-template', str(args.metadata_template)]
                observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=case / 'worker.log',
                    resources=case / 'resources.jsonl', timeout=budget, mem_gib=24, fsize_gib=2)
                write_json(case / 'process.json', observed)
                value = read(case / 'result.json') if (case / 'result.json').is_file() else dict(status='error',error='Missing worker result')
                state['phases'][phase] = value
                record_registry(case, 'S4-QWEN-ISOLATED-POST', 'Qwen2-0.5B-layer0', phase,
                                value['status'], dict(API_calls=0, compiler_attempts=0))
                persist_case_completion(case, round_label='S4-QWEN-ISOLATED-POST', started_ns=started,
                    status=value['status'], timing=value.get('timing') or dict(status='error',attribution_ready=False,spans=[],diagnostics=[]),
                    required_artifact_bytes=artifact_bytes(case))
                if value['status'] != 'ok' or not (observed['exit_code'] == 0 and not observed['timed_out']
                        and not observed['remaining_pids'] and observed['signal'] is None):
                    state.update(status='error', error='One post-freeze phase failed; no retry')
                    break
            else:
                state['status'] = 'ok'
            write_json(args.out / 'result.json', state)
    for root in (args.paper, args.mirror):
        write_json(root / 'S4_QWEN_ISOLATED_POST.json', state)
        for phase, name in (('report', 'qwen_block_isolated_report.json'), ('count', 'qwen_block_fig2_components.json')):
            if phase in state.get('phases', {}):
                write_json(root / name, state['phases'][phase])
        target = root / 'EXPERIMENT_DELIVERY.json'
        delivery = read(target)
        delivery['S4_Qwen_isolated_post'] = dict(status=state['status'], source=str(args.out / 'result.json'))
        delivery['links']['S4_Qwen_isolated_post'] = str(args.paper / 'S4_QWEN_ISOLATED_POST.json')
        write_json(target, delivery)
    return int(state['status'] == 'error')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('report', 'count'))
    parser.add_argument('--resume-wait', action='store_true')
    parser.add_argument('--prior-case', type=Path)
    for key in ('source-case', 'metadata-template', 'out', 'paper', 'mirror', 'publisher', 'gate', 'work-dir'):
        parser.add_argument('--'+key, type=Path)
    raise SystemExit(main(parser.parse_args()))
