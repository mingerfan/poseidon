"""Original Qwen first-block adaptation of the existing BERT three-hour driver.

The inclusive clock, search boundary, one C2 and compilation order are reused.
Only the original-model intake, block sample provider and output metric differ.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

from .. import paths
from . import bert_fresh_cold_3h as cold
from . import bert_sst2 as bert
from . import s4_qwen_block as s4
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

MODULE = 'pt_agent.experiments.s4_qwen_cold'
MODEL = 'Qwen2-0.5B-layer0'
SPLIT_VERSION = 's4-qwen-original-block-training-v1'
read = s4.read


@contextmanager
def original_numeric_scope():
    """Reuse the established Qwen operand-lifetime and projection adapters."""
    from .. import plan as planner, transformer_validation as validation, cost
    from ..planners import agent_tools
    from ..backends.streaming_mock import StreamingMockBackend
    from ..kernels.compact_linear import compact_transformer_projections
    from .qwen_prefill_compile import _planner_namespace, _projection_backend_scope
    from .qwen_agent_resources import private_cost_namespace
    private = _planner_namespace()
    costs = private_cost_namespace(planner=private)
    cost_names = ('estimate', 'estimate_normalized', 'estimate_fast', 'estimate_normalized_fast')
    original_costs = {(module, name): getattr(module, name)
                      for module in (cost, agent_tools) for name in cost_names}
    for module, name in original_costs:
        setattr(module, name, costs[name])
    old_plan, old_verify, old_mock = planner.plan_program, planner.verify_program, validation.MockBackend
    planner.plan_program, planner.verify_program = private['plan_program'], private['verify_program']
    validation.MockBackend = StreamingMockBackend
    try:
        with _projection_backend_scope(), compact_transformer_projections():
            yield
    finally:
        planner.plan_program, planner.verify_program, validation.MockBackend = old_plan, old_verify, old_mock
        for (module, name), function in original_costs.items():
            setattr(module, name, function)


def original_embedding(model, ids):
    """Read only selected BF16 rows from the original public safetensors file."""
    import numpy as np
    file = Path(model) / 'model.safetensors'
    with file.open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        header = json.loads(stream.read(size))
    entry = header['model.embed_tokens.weight']
    if entry['dtype'] != 'BF16' or entry['shape'] != [151936, 896]:
        raise ValueError('Retain the original BF16151936x896 embedding source')
    mapped = np.memmap(file, mode='r', dtype='<u2', offset=8+size+entry['data_offsets'][0],
                       shape=tuple(entry['shape']))
    bits = mapped[np.asarray(ids, dtype=np.int64)].astype(np.uint32) << 16
    values = bits.view(np.float32).astype(np.float64)
    del mapped
    return values


class BlockSamples:
    def __init__(self, model_dir, info, *, report=False):
        self.directory, self.info, self.report = Path(model_dir), info, report
        self.accesses = []

    def get(self, split, n=None):
        import numpy as np
        if split not in ('calibration', 'heldout', 'confirm', 'report') or split == 'report' and not self.report:
            raise ValueError('Report inputs are unavailable to the Agent')
        indices = self.info.split[split] if n is None else self.info.split[split][:n]
        self.accesses.append(dict(partition=split, indices=indices, count=len(indices)))
        cached = self.directory / 'samples' / (split + '.npz')
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as archive:
                return {key: np.array(archive[key][:len(indices)]) for key in self.info.graph.inputs}, None
        from transformers import AutoTokenizer
        from ..data.samples import encode_texts
        texts = read(self.directory / 'samples/texts.json')
        tokenizer = AutoTokenizer.from_pretrained(self.info.source['local_path'], local_files_only=True,
                                                  trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = encode_texts(tokenizer, [texts[index] for index in indices], self.info.sequence, task='causal_lm')
        arrays = dict(inputs_embeds=original_embedding(self.info.source['local_path'], encoded['input_ids']),
                      attention_mask=np.asarray(encoded['attention_mask'], dtype=np.float64))
        if split != 'report' and n is None:
            np.savez(cached, **arrays)
        return arrays, None


def prepare(args, recorder, result):
    import numpy as np
    import torch
    from ..analyzer import calibrate, reference_forward
    from ..data.samples import read as read_samples
    from ..intake.probe import import_model
    from ..metrics import compare, passes
    from ..models.info import metric_for
    from .qwen_rule63_block_compile import slice_layer0
    samples = read_samples(args.inputs)
    with recorder.span('fresh_original_HF_intake'):
        imported = import_model(args.model, samples, observer=recorder)
    original = imported.info
    if len(imported.original.module.model.layers) != 24 or original.sequence['length'] != 32:
        raise ValueError('Retain the original24-layer Qwen model and32-token intake')
    block = slice_layer0(original.graph)
    partition = dict(original.split)
    expected = read(args.original_info / 'model_info.json')['split']
    if any(partition[key] != expected[key] for key in ('calibration', 'heldout', 'confirm', 'report')):
        raise ValueError('Original Qwen partitions changed during fresh intake')
    texts = [json.loads(line)['text'] for line in args.inputs.read_text().splitlines() if line.strip()]
    needed = args.heldout - len(partition['heldout'])
    if needed < 0:
        raise ValueError('Keep every original heldout sample')
    training = Path(read(args.inputs.parent / 'source_rows.json')['source'])
    # Read an existing training shard through the previously prepared Parquet environment.
    with recorder.span('additional_fixed_training_heldout'):
        script = ('import json,sys; import pyarrow.parquet as pq; '
            't=pq.read_table(sys.argv[1],columns=["sentence"]).column("sentence"); '
            'print(json.dumps(t.slice(223,int(sys.argv[2])).to_pylist()))')
        observed = subprocess.run([str(args.data_python), '-B', '-c', script, str(training), str(needed)],
            capture_output=True, text=True, check=True, timeout=120,
            env=bert.child_environment(args, args.work_dir))
        extra = json.loads(observed.stdout)
    if len(extra) != needed or any(not isinstance(text, str) or not text.strip() for text in extra):
        raise ValueError('Additional real training rows are incomplete')
    texts.extend(extra)
    partition['heldout'] = list(partition['heldout']) + list(range(223, 223+needed))
    partition.update(version=SPLIT_VERSION,
        sample_ids=['SST2-training-row:' + str(index) for index in range(len(texts))])
    all_indices = [index for key in ('calibration', 'heldout', 'confirm', 'report') for index in partition[key]]
    if len(set(all_indices)) != len(all_indices) or sorted(all_indices) != list(range(len(texts))):
        raise ValueError('Four Qwen partitions must remain disjoint')
    metric, policy = metric_for('regression', has_labels=False)
    info = replace(original, graph=block, inputs=original.inputs[:2], task='regression',
        metric=metric, threshold=policy, split=partition,
        source=dict(original.source, inputs_path=str(args.inputs), benchmark_scope='original-decoder-layer0',
                    training_source=str(training), additional_heldout_rows=list(range(223, 223+needed))))
    args.model_dir.mkdir()
    (args.model_dir / 'samples').mkdir()
    write_json(args.model_dir / 'samples/texts.json', texts)
    provider = BlockSamples(args.model_dir, info)
    arrays, _ = provider.get('calibration')
    original_arrays, _ = imported.samples.get('calibration', 1)
    if any(not np.array_equal(arrays[name][:1], original_arrays[name]) for name in block.inputs):
        raise ValueError('Direct original BF16 embedding lookup differs from the fresh HF input provider')
    checks = []
    with recorder.span('fresh_original_layer0_reference_probes'):
        for index in range(3):
            values = {key: value[index:index+1] for key, value in arrays.items()}
            with torch.no_grad():
                output = imported.original.module.model(
                    inputs_embeds=torch.from_numpy(values['inputs_embeds']),
                    attention_mask=torch.from_numpy(values['attention_mask']), use_cache=False,
                    output_hidden_states=True, return_dict=True).hidden_states[1].cpu().numpy()
            normalized = reference_forward(block, values)[block.outputs[0]]
            measured = compare(dict(values=output, task='regression'), normalized, None)
            checks.append(dict(index=index, metrics=measured, passed=passes(measured, policy),
                scope='Original HF block output versus canonical float64 graph; original HF RMSNorm rounds internally'))
            if not checks[-1]['passed']:
                raise ValueError('Original block reference fidelity policy rejected the fresh import')
    with recorder.span('fresh_23_training_calibration'):
        calibration = calibrate(block, arrays, batch=1)
        if any(isinstance(value, (int, float)) and not math.isfinite(value)
               for row in calibration.values() for value in row.values()):
            raise ValueError('Nonfinite calibration; no Agent request sent')
        write_json(args.model_dir / 'calibration.json', calibration)
    with recorder.span('save_original_block_and_prepare_heldout'):
        info.save(args.model_dir)
        provider.get('heldout')
    result.update(import_completed=True, reference_checks=checks,
        split_counts={key: len(partition[key]) for key in ('calibration', 'heldout', 'confirm', 'report')},
        sample_accesses=provider.accesses, task=info.task, task_policy=policy,
        metric_scope='Hidden-state block relative error; full-decoder next-token accuracy unmeasured',
        original_model_freshly_loaded=True, calibration_reused=False, candidate_reused=False)


def context(args):
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context
    info = ModelInfo.load(args.model_dir)
    result = build_context(info, read(args.model_dir / 'calibration.json'),
                           samples=BlockSamples(args.model_dir, info), n_val=args.heldout)
    result.prompt_version, result.placement, result.tune_search = 'r8-v11', 'program-plan', 'sweep'
    return result


def agent_step(args, recorder, result):
    from ..autonomous_budget import RenewingLLMClient
    from ..planners.transformer_session import load_state, output_tokens, session_step
    from .bert_sst2_worker import RecordedRenewingClient
    with recorder.span('original_internal_context'):
        ctx = context(args)
    path = args.session_dir / 'SESSION.json'
    phase = load_state(path)['phase'] if path.is_file() else 'evaluate'
    client = None
    if phase == 'evaluate':
        client = RecordedRenewingClient(RenewingLLMClient(budget_path=args.ledger,
            allowance_path=args.allowance, recorder=recorder, artifact_dir=args.session_dir / 'api',
            model='deepseek-flash', reasoning_effort='high', max_tokens=output_tokens('r8-v11'), timeout=600),
            args.session_dir / 'calls', recorder=recorder)
    with recorder.span('agent_' + phase):
        measured = session_step(ctx, args.session_dir, client=client, max_iters=10,
            tune_rule='v8-6.25', confirm_rule='c2', tune_policy='all', stale_limit=2,
            focus_retune=True, numeric_lock=True)
    result.update(next_step=measured['next_step'], step=measured,
                   llm_calls=client.calls if client else 0, frozen=(args.session_dir / 'FROZEN.json').is_file())


def checked_program(args):
    from ..models.info import ModelInfo
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..dsl.interface import require_interface
    info = ModelInfo.load(args.model_dir)
    program = parse((args.session_dir / 'program.ptdsl').read_text())
    require_interface(program, info)
    if not check(program, {key: value.shape for key, value in info.graph.weights.items()},
                 graph=info.graph, prompt_version='r8-v11').ok:
        raise ValueError('Frozen block typecheck failed')
    return info, program


def math_worker(args, recorder, result):
    from ..analyzer import reference_forward
    from ..transformer_validation import execute
    from .bert_sst2_worker import strict_math
    info, program = checked_program(args)
    arrays, _ = BlockSamples(args.model_dir, info).get('calibration', 3)
    checks = []
    for index in range(3):
        inputs = {key: value[index:index+1] for key, value in arrays.items()}
        with recorder.span('calibration_math', sample=index):
            exact = reference_forward(info.graph, inputs)[info.graph.outputs[0]]
            tensor_exact = execute(program, info.graph.weights, inputs, engine='tensor', exact=True)
            mock_exact = execute(program, info.graph.weights, inputs, engine='mock', exact=True)
            tensor_approx = execute(program, info.graph.weights, inputs, engine='tensor', exact=False)
            mock_approx = execute(program, info.graph.weights, inputs, engine='mock', exact=False)
        rows = [strict_math(exact, tensor_exact), strict_math(exact, mock_exact), strict_math(tensor_approx, mock_approx)]
        checks.append(dict(index=index, checks=rows))
        if any(row['passed'] is not True for row in rows):
            raise ValueError('Original strict1e-12+1e-12*abs(ref) mathematics failed')
    result.update(math_passed=True, frozen_math=checks, strict_checks=9)


def emit_worker(args, recorder, result):
    from ..backends import RefreshBackend
    from ..backends.streaming_count import StreamingCountBackend
    from ..backends.streaming_emit import StreamingEmitBackend
    from ..lower import lower
    from ..plan import plan_program
    info, program = checked_program(args)
    with recorder.span('program_bootstrap_placement'):
        plan, dag, placement = plan_program(program, info.graph.weights, profile='qwen48')
        write_json(args.work_dir / 'refresh-plan.json', plan)
        del dag, placement
        gc.collect()
    with recorder.span('Count_plan_replay'):
        counter = StreamingCountBackend(16384, boot_out_level=plan['levels']['boot_out_level'])
        active = RefreshBackend(counter, policy='plan', plan=plan, **plan['levels'])
        lower(program, info.graph.weights, active, act_mode='program')
        stats = counter.stats()
        if stats['counts']['boot'] != plan['boots']:
            raise ValueError('Fresh block Count/plan bootstrap count differs')
        del active, counter
        gc.collect()
    with recorder.span('trusted_Emit_and_write'):
        backend = StreamingEmitBackend(16384, directory=args.work_dir / 'emit', frontend=args.frontend,
                                       max_constant_bytes=48 * s4.GIB)
        try:
            active = RefreshBackend(backend, policy='plan', plan=plan, **plan['levels'])
            lower(program, info.graph.weights, active, act_mode='program')
            emitted = backend.stats()
            if any(stats[key] != emitted[key] for key in ('counts', 'ops_total', 'distinct_consts')):
                raise ValueError('Fresh block complete Count/Emit statistics differ')
            backend.write()
        finally:
            if not backend.finished:
                backend.close()
    result.update(planned_bootstraps=plan['boots'], count_replay=stats, emit_stats=emitted)


def native_worker(args, recorder, result):
    from .. import toolchain
    from .r29_local_compile_matrix import compiler_argv, plan_observation
    from .r30_common_model_e2e import diagnostic_compile_argv
    host = toolchain.host_paths()
    emit = args.work_dir / 'emit'
    environment = dict(os.environ, HECATE=str(host['hecate']), HECATE_BUILD=str(host['hecate_build']),
                       LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'])
    if args.phase == 'trace':
        argv = [sys.executable, '-B', str(emit / 'candidate_trace.py')]
    else:
        argv = compiler_argv(read(args.metadata_template), 'pars', binary=host['hecate_opt'],
            source=emit / 'candidate_trace.mlir', profile=args.profile, output=emit / 'lowered.mlir')
        argv = diagnostic_compile_argv(argv, elide_final_ir=True)
    write_json(args.work_dir / (args.phase + '-argv.json'), argv)
    with recorder.span('native_' + args.phase):
        process = run_process(argv, cwd=emit, log=args.work_dir / (args.phase + '-native.log'),
            resources=args.work_dir / (args.phase + '-native-resources.jsonl'), timeout=args.remaining_seconds,
            mem_gib=24, fsize_gib=16, env=environment)
    result['process'] = process
    write_json(args.work_dir / (args.phase + '-process.json'), process)
    if not cold._clean(process):
        raise RuntimeError('Original native ' + args.phase + ' failed')
    if args.phase == 'compile':
        plans = list(emit.glob('lowered.*.runtime-plan.json'))
        if len(plans) != 1:
            raise ValueError('One actual complete block RuntimePlan required')
        result.update(plan_observation(read(plans[0]), read(args.profile)), runtime_plan=str(plans[0]))
        if result['compiled_bootstraps'] != read(args.work_dir / 'refresh-plan.json')['boots']:
            raise ValueError('Actual compiled block bootstrap count differs from the original plan')


def worker(args):
    from .r29_local_compile_matrix import native_metadata_commands
    args.model_dir, args.session_dir = args.source_case / 'model', args.source_case / 'session'
    args.work_dir.mkdir(parents=True, exist_ok=True)
    recorder = StageRecorder(args.work_dir / (args.phase + '-events.jsonl'))
    result = dict(status='running', error=None, phase=args.phase, model=MODEL, report_samples=0, llm_calls=0)
    try:
        with native_metadata_commands(read(args.metadata_template)):
            from .bert_sst2_worker import initialize_numeric
            initialize_numeric()
            with original_numeric_scope():
                if args.phase == 'prepare':
                    prepare(args, recorder, result)
                elif args.phase == 'checkpoint':
                    prepared = read(args.source_case / 'phases/001-prepare/worker_result.json')
                    result.update(checkpoint_semantic_coverage_passed=prepared['import_completed'] and
                        all(row['passed'] for row in prepared['reference_checks']),
                        scope='Fresh original model intake and three original HF block reference checks')
                elif args.phase in ('step', 'confirm'):
                    agent_step(args, recorder, result)
                    result['phase'] = 'step'
                elif args.phase == 'math':
                    math_worker(args, recorder, result)
                elif args.phase == 'emit':
                    emit_worker(args, recorder, result)
                else:
                    native_worker(args, recorder, result)
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
    result['timing'] = summarize_events(read_events(recorder.path))
    write_json(args.work_dir / 'worker_result.json', result)
    return result


def check_worker(phase, result):
    if result.get('status') != 'ok' or result.get('model') != MODEL or result.get('phase') != phase:
        raise ValueError('S4 worker identity or requested stage differs')
    if result.get('report_samples') != 0:
        raise ValueError('No independent report data enters the cold Agent or compilation')


def frozen_state(case):
    from .qwen_agent_continuation import session_outcome
    outcome = session_outcome(Path(case) / 'session')
    if outcome['status'] != 'ok' or outcome['frozen'] is not True:
        raise ValueError('Original v11/C2 did not accept the block candidate')
    return dict(frozen=True, outcome='frozen', program=str(Path(case) / 'session/program.ptdsl'))


def source_gate(args):
    if args.out.exists() or args.checkpoint_out.exists() or args.out.parent != paths.RESULTS_DIR.resolve():
        raise ValueError('Fresh direct result directories required')
    config = read(args.model / 'config.json')
    if config.get('num_hidden_layers') != 24 or config.get('hidden_size') != 896 or config.get('model_type') != 'qwen2':
        raise ValueError('Original trained Qwen2-0.5B required')
    if args.heldout not in (256, 1000) or args.heldout != read(args.probe)['heldout_decision']['requested_heldout_samples']:
        raise ValueError('Fix heldout sample count from the completed timing probe before the cold start')
    if s4.remaining_before_deadline(10800) < 10800:
        raise TimeoutError('Less than a full three-hour S4 cold allowance remains before20:00')
    return dict(model=str(args.model), inputs=str(args.inputs), layer=0, sequence_capacity=32,
        fresh_HF_intake=True, warm_model_or_calibration_reuse=False, warm_session_or_candidate_reuse=False,
        calibration_samples=23, heldout_samples=args.heldout, confirm_samples=20, report_samples=0,
        independent_report_available_samples=160, heldout_decision=read(args.probe)['heldout_decision'],
        output_metric='regression_relative', output_policy=dict(relative_error=.01),
        output_scope='Original layer0 hidden state, full-decoder language-model quality unmeasured')


def command(args, phase, work, remaining):
    inner = ['-B', '-m', MODULE, '--worker', '--phase', phase,
        '--source-case', str(args.out), '--work-dir', str(work), '--model', str(args.model),
        '--inputs', str(args.inputs), '--original-info', str(args.original_info),
        '--heldout', str(args.heldout), '--data-python', str(args.data_python),
        '--metadata-template', str(args.metadata_template), '--profile', str(args.profile),
        '--frontend', str(args.frontend), '--ledger', str(args.ledger), '--allowance', str(args.allowance),
        '--remaining-seconds', str(max(1, int(remaining)-15))]
    return [sys.executable, '-B', '-m', s4.MODULE, '--enter', '--seconds', str(max(1, int(remaining))),
             '--python', str(args.intake_python), '--', *inner]


def paper_record(args, result, observation):
    from .a_group_runs import append_record, empty_record
    from .a_group_profile_sum_policy import profile_sum
    compiled = result['phases'].get('compile') or next((row.get('worker_result') or {}
        for row in reversed(result['processes']) if row['phase'] == 'compile'), {})
    process = compiled.get('process') or {}
    failure = result.get('termination') or {}
    timeout = failure.get('cause') == 'wall_budget' or 'TimeoutError' in (result.get('error') or '') or any(
        (row.get('observation') or {}).get('timed_out') or (row.get('observation') or {}).get('exit_code') == 124
        for row in result['processes'])
    status = ('ok' if result['compile_accepted'] else 'timeout' if timeout else 'resource_limit'
        if failure.get('cause') in ('physical_memory', 'resource_guard') else 'compiler_crash'
        if process.get('exit_code') not in (None, 0) else 'driver_error')
    stages = Counter()
    for row in result['timing']['spans']:
        if row.get('exclusive_ns') is not None:
            stages[row['stage_id']] += row['exclusive_ns'] / 1e9
    stages['persistence_and_other'] = observation['completion_elapsed_seconds'] - sum(stages.values())
    peak = 0
    for row in result['processes']:
        file = Path(row['resources']) if row.get('resources') else None
        if file and file.is_file():
            for line in file.read_text().splitlines():
                sample = json.loads(line)
                peak = max(peak, sum(value.get('vmrss_kib', 0)*1024 for value in sample.get('processes', [])))
    policy = profile_sum(result.get('prediction'))
    row = empty_record(run_id=args.out.name, task='S4-Q1', model='qwen2_0_5b_block', system='arranger',
        param_group='ours-N16-boot5', status=status, status_detail=result.get('error'), budget_s=10800,
        compile_s=(process['ended_ns']-process['started_ns'])/1e9 if process.get('ended_ns') else None,
        generation_e2e_s=observation['completion_elapsed_seconds'] if result['compile_accepted'] else None,
        observed_complete_case_s=observation['completion_elapsed_seconds'],
        peak_rss_gib=peak/s4.GIB if peak else None, boots_compiled=result.get('compiled_bootstraps'),
        exec_ops=compiled.get('execution_steps'), pred_latency_s=policy['pred_latency_s'],
        pred_table=str(args.profile), prediction_policy=policy, api_calls=result['api_calls'],
        stages_s=dict(stages), source=str(args.out / 'result.json'), finished_utc=datetime.now(timezone.utc).isoformat(),
        cold=True, full_model=False, layer=0, sequence_capacity=32, heldout_samples=args.heldout,
        calibration_samples=23, confirm_samples=20, report_samples=0,
        usable_within_three_hours=observation['persisted_pipeline_within_three_hours'],
        usable_within_two_hours=observation['persisted_pipeline_within_two_hours'],
        three_hour_record=str(args.out / 'three-hour-record.json'), paper_eligible=time.time() <= s4.DEADLINE,
        quality=dict(metric='regression_relative', split='confirmation20',
                     criterion='relative_error<=0.01; original calibrated/analytical range policies',
                     source=str(args.out / 'session/AGENT.json'), full_decoder_LM_quality_measured=False))
    write_json(args.out / 'paper-record.json', row)
    if args.paper_runs:
        append_record(args.paper_runs, row)


def namespace():
    source = vars(cold)
    values = dict(source, MODULE=MODULE, ROUND='S4-Q1-FRESH-QWEN-BLOCK-3H', MODEL=MODEL,
                  MODE='fresh-original-Qwen2-layer0-v11-C2-three-hour-cold-compile', __file__=__file__)
    for name, value in source.items():
        if isinstance(value, FunctionType) and value.__globals__ is source:
            fn = FunctionType(value.__code__, values, value.__name__, value.__defaults__, value.__closure__)
            fn.__kwdefaults__ = value.__kwdefaults__
            values[name] = fn
    state_policy = dict(cold.serial.STATE_POLICY, split_version=SPLIT_VERSION)
    values.update(source_gate=source_gate, command=command, paper_record=paper_record,
        _read_worker=lambda args, phase, work: read(work / 'worker_result.json') if (work / 'worker_result.json').is_file() else None,
        serial=SimpleNamespace(STATE_POLICY=state_policy, serial_lease=cold.serial.serial_lease,
                               frozen_state=frozen_state, durable_json=cold.serial.durable_json),
        original=SimpleNamespace(child_environment=bert.child_environment, check_worker=check_worker,
            account_api_calls=bert.account_api_calls, __file__=bert.__file__))
    return values


def parser():
    cli = cold.parser()
    cli.add_argument('--worker', action='store_true')
    cli.add_argument('--phase', choices=('prepare', 'checkpoint', 'step', 'confirm', 'math', 'emit', 'trace', 'compile'))
    cli.add_argument('--heldout', type=int, choices=(256, 1000), required=True)
    cli.add_argument('--original-info', type=Path, required=True)
    cli.add_argument('--probe', type=Path)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute() if key in ('intake_python', 'data_python') else value.resolve())
    if args.worker:
        result = worker(args)
    else:
        with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = namespace()['execute'](args)
    print(json.dumps(dict(status=result['status'], error=result.get('error'))), flush=True)
    return int(result['status'] != 'ok')


if __name__ == '__main__':
    raise SystemExit(main())
