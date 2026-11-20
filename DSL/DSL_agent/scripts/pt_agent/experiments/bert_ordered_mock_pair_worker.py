"""Internal worker for one guarded original/ordered BERT held-out pair.

The outer supplies all paths and a closed-source admission record. This worker
has no provider, registry, compiler or report entry. It verifies an execution
optimization on an existing frozen program; it does not measure a cold chain.
"""
import argparse
import json
from pathlib import Path
import time

from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json


COUNT = 40
SCHEMA = 'bert-ordered-original-mock-pair-admission-v1'


def read(path):
    return json.loads(Path(path).read_text())


def admission(args):
    proof = read(args.source_admission)
    flags = ('source_closed', 'owner_queue_closed', 'all_source_births_closed',
             'source_registry_unique_eight_fields')
    if (proof.get('schema') != SCHEMA or proof.get('status') != 'ok'
            or any(proof.get(key) is not True for key in flags)
            or proof.get('logical_heldout_rows') != COUNT
            or proof.get('report_read_capability') is not False):
        raise ValueError('Require actual outer closed-source admission for this held-out pair')
    for key in ('model_dir', 'program', 'frozen', 'model', 'inputs', 'metadata_template'):
        if Path(proof.get(key, '')).resolve() != getattr(args, key).resolve():
            raise ValueError('Keep the original admitted worker path: '+key)
    source = Path(proof['source_case']).resolve()
    if (args.model_dir.resolve() != source/'model'
            or args.program.resolve() != source/'session/program.ptdsl'
            or args.frozen.resolve() != source/'session/FROZEN.json'
            or args.work_dir.exists() or args.work_dir.resolve().is_relative_to(source)):
        raise ValueError('Preserve the original frozen source and use a new private worker directory')
    if not (args.model_dir/'samples/heldout.npz').is_file():
        raise ValueError('The original stored held-out inputs are required; do not rebuild them')
    marker = read(args.frozen)
    if (Path(marker.get('program', '')).resolve() != args.program.resolve()
            or marker.get('split_version') != 'sst2-native-v1'):
        raise ValueError('Require the original frozen program on its fixed SST-2 partition')
    return proof


def command_context(template):
    from .r29_local_compile_matrix import native_metadata_commands
    return native_metadata_commands(read(template))


def runtime(args):
    from .bert_sst2_worker import initialize_numeric, NumericSplits, strict_math
    initialize_numeric()
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context, VAL_BATCH, MOCK_VAL_BATCH
    from ..dsl.parser import parse
    from ..dsl.interface import require_interface
    from ..dsl.typecheck import check
    from ..transformer_validation import validate
    from .ordered_mock_validation import parallel_mock_validate

    info = ModelInfo.load(args.model_dir)
    if (Path(info.source['local_path']).resolve() != args.model.resolve()
            or Path(info.source['inputs_path']).resolve() != args.inputs.resolve()
            or info.task != 'classification' or info.threshold != {'acc_drop_pp': .5}
            or info.split.get('version') != 'sst2-native-v1'
            or info.split.get('heldout') != list(range(1000, 2000))
            or min(VAL_BATCH, MOCK_VAL_BATCH) != 10):
        raise ValueError('Keep the original model, task policy, partition and ten-row Mock batches')
    samples = NumericSplits(args.model_dir, info)
    original_get = samples.get
    accesses = []

    def heldout_only(split, n=None):
        if split != 'heldout' or n != COUNT:
            raise ValueError('This execution pair reads only the fixed first40 held-out rows')
        accesses.append(dict(split=split, logical_rows=n))
        return original_get(split, n)

    samples.get = heldout_only
    ctx = build_context(info, read(args.model_dir/'calibration.json'), samples=samples, n_val=COUNT)
    ctx.prompt_version, ctx.placement, ctx.tune_search = 'r8-v11', 'program-plan', 'sweep'
    program = parse(args.program.read_text())
    require_interface(program, info)
    checked = check(program, {key: value.shape for key, value in info.graph.weights.items()},
                    graph=info.graph, prompt_version='r8-v11')
    if not checked.ok or len(ctx.val_images) != COUNT or samples.local is not None:
        raise ValueError('Require the original typechecked program and stored forty-row inputs')
    if samples.raw.report_authorized or samples.raw.accesses:
        raise ValueError('Do not authorize report access or read raw sentences for this pair')
    return dict(ctx=ctx, program=program, validate=validate, parallel=parallel_mock_validate,
                strict=strict_math, accesses=accesses, logical_positions=info.split['heldout'][:COUNT],
                report_capability=False, stored_inputs_only=True)


def compare_pair(original, ordered, original_output, ordered_output,
                 original_diagnostics, ordered_diagnostics, strict):
    if (original.get('total') != COUNT or ordered.get('total') != COUNT
            or original.get('images_done') != COUNT or ordered.get('images_done') != COUNT
            or original != ordered):
        raise ValueError('Original and ordered full metrics/domain rows differ')
    if (len(original_diagnostics) != 1 or original_diagnostics != ordered_diagnostics
            or original_diagnostics[0].get('total') != COUNT
            or len(original_diagnostics[0].get('samples', [])) != COUNT):
        raise ValueError('Original domain diagnostics or event ordering differ')
    checked = strict(original_output, ordered_output)
    if checked.get('passed') is not True:
        raise ValueError('Ordered Mock differs from original Mock at the unchanged mathematical tolerance')
    return dict(output_mathematics=checked, full_metrics_equal=True,
                original_domain_diagnostics_equal=True, original_event_indices_equal=True,
                original_batch_size=10, logical_rows=COUNT)


def worker(args):
    proof = admission(args)
    args.work_dir.mkdir(parents=True)
    target = args.work_dir/'pair-result.json'
    recorder = StageRecorder(args.work_dir/'pair-events.jsonl')
    began = time.perf_counter_ns()
    result = dict(schema='bert-ordered-original-mock-pair-result-v1', status='running',
        pair_accepted=False, error=None, source_admission=str(args.source_admission), source=proof,
        logical_heldout_rows=COUNT, requested_workers=3, original_mock_batch_size=10,
        attempted=dict(tensor=0, original_mock=0, ordered_mock=0),
        completed=dict(tensor=0, original_mock=0, ordered_mock=0), measurements={},
        new_agent_steps=0, confirmation_calls=0, report_samples=0, native_test_reads=0,
        API=0, Count=0, compiler_runs=0, keygen_runs=0, ckks_runs=0,
        downloads=0, installations=0, integrity_checks=0, speedup=None,
        cold_chain_measurement=False, qualifies_two_hour_claim=False,
        timing_scope='Warm execution-equivalence pair; Tensor then original Mock then ordered Mock. Cache order differs; raw spans establish no cold speedup claim.')
    write_json(target, result)
    try:
        with command_context(args.metadata_template):
            with recorder.span('restore_original_frozen_heldout40_context'):
                api = runtime(args)
            if (api['report_capability'] is not False or api['stored_inputs_only'] is not True
                    or api['accesses'] != [dict(split='heldout', logical_rows=COUNT)]):
                raise ValueError('Retain one stored held-out access and no report capability')
            outputs, diagnostics, metrics = {}, {}, {}
            scheduling = []

            def telemetry(value):
                # Helper emits only scalar scheduling, clock and cleanup data.
                scheduling.append(dict(value))
                with (args.work_dir/'ordered-batch-events.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(observed_ns=time.perf_counter_ns(), **value))+'\n')
                    stream.flush()

            for phase in result['attempted']:
                result['attempted'][phase] += 1
                write_json(target, result)
                captured, recorded = [], []
                phase_started = time.perf_counter_ns()
                with recorder.span(phase):
                    call = api['parallel'] if phase == 'ordered_mock' else api['validate']
                    extra = dict(workers=3, telemetry=telemetry) if phase == 'ordered_mock' else {}
                    measured = call(api['program'], api['ctx'], engine='tensor' if phase == 'tensor' else 'mock',
                        output_callback=captured.append, diagnostic_callback=recorded.append, **extra)
                if len(captured) != 1 or measured.get('total') != COUNT:
                    raise ValueError('Keep one complete original forty-row output and metric record')
                outputs[phase], diagnostics[phase], metrics[phase] = captured[0], recorded, measured
                result['measurements'][phase] = dict(elapsed_s=(time.perf_counter_ns()-phase_started)/1e9,
                    metrics=measured, diagnostics=recorded, logical_rows=COUNT)
                result['completed'][phase] += 1
                write_json(target, result)
            with recorder.span('unchanged_original_and_ordered_comparisons'):
                starts = [v for v in scheduling if v.get('event') == 'batch_started']
                received = [v for v in scheduling if v.get('event') == 'batch_received']
                expected = [(i, i+10) for i in range(0, COUNT, 10)]
                if ([(v.get('start'), v.get('stop')) for v in starts] != expected
                        or [(v.get('start'), v.get('stop')) for v in received] != expected
                        or len({v.get('pid') for v in starts}) != 4
                        or any(v.get('kind') != 'ok' for v in received)
                        or [v.get('event') for v in scheduling[:3]] != ['batch_started']*3
                        or any(v.get('event') in ('sequential_fallback', 'unused_prefetch_cleanup',
                                                 'owned_cleanup_error') for v in scheduling)):
                    raise ValueError('The real ordered four-batch path must execute without fallback')
                result['ordered_execution_observed'] = True
                for key in ('feasible', 'agree'):
                    if (key not in metrics['tensor'] or key not in metrics['original_mock']
                            or metrics['tensor'][key] != metrics['original_mock'][key]):
                        raise ValueError('Original final Tensor/Mock summary check differs: '+key)
                result['pair_checks'] = compare_pair(metrics['original_mock'], metrics['ordered_mock'],
                    outputs['original_mock'], outputs['ordered_mock'], diagnostics['original_mock'],
                    diagnostics['ordered_mock'], api['strict'])
            result.update(status='ok', pair_accepted=True, sample_accesses=api['accesses'],
                          logical_split_positions=api['logical_positions'],
                          stored_inputs_only=api['stored_inputs_only'],
                          report_read_capability=api['report_capability'])
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc))
    result['elapsed_s'] = (time.perf_counter_ns()-began)/1e9
    result['timing'] = summarize_events(read_events(recorder.path))
    write_json(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-admission', 'model-dir', 'program', 'frozen', 'model', 'inputs',
                 'metadata-template', 'work-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-worker', action='store_true')
    args = parser.parse_args(argv)
    if not args.run_worker:
        parser.error('An actual outer admission and explicit internal worker execution are required')
    result = worker(args)
    print(json.dumps(dict(status=result['status'], pair_accepted=result['pair_accepted'])))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
