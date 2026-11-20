"""Bounded, original-model Qwen Agent continuation and diagnostic code generation.

The saved full model and its existing 23/20/20/160 split remain unchanged.
Report data has no reader in this entry.  A compiled two-token graph is an
engineering artifact; standard LM quality and encrypted generation stay unset.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .completion_metrics import artifact_bytes, persist_case_completion

ROUND = 'autonomous-qwen-continuation'
MODEL = 'Qwen2-0.5B'
MODE = 'original-full-decoder-agent-two-token-codegen'
MODULE = 'pt_agent.experiments.qwen_agent_continuation'
GIB = 1024**3
PARTITIONS = dict(calibration=23, heldout=20, confirm=20, report=160)
PHASE_SECONDS = dict(calibration=900, prompt=900, step=3600, placement=7200,
                     emit=7200, trace=1800, compile=7200)
FORMAL = ('accuracy_acceptance', 'standard_lm_quality', 'predicted_latency', 'CKKS_latency',
          'security_acceptance', 'full_generation_quality_acceptance')
ZERO = ('report_samples', 'keygen_runs', 'real_ckks_attempts', 'real_ckks_completed',
        'intermediate_decryptions', 'new_downloads', 'new_installations', 'new_weights')


def read_json(path):
    return json.loads(Path(path).read_text())


def _same(left, right):
    return Path(left).resolve() == Path(right).resolve()


def source_gate(args):
    """Read small JSON and filesystem metadata; never read text/weight payloads."""
    path = args.model_info / 'model_info.json'
    info = read_json(path)
    source = info['source']; config = source.get('config', {})
    if (info.get('task') != 'causal_lm' or info.get('metric') != 'lm_next_token'
            or info.get('threshold') != {'agreement_pct': 99.0, 'logit_rel_err': .01}
            or config.get('num_hidden_layers') != 24 or config.get('hidden_size') != 896
            or config.get('vocab_size') != 151936 or not _same(source['local_path'], args.model)
            or info.get('input_handling', {}).get('single_lm_head') is not True
            or info.get('sequence', {}).get('length') != 32):
        raise ValueError('Retain the supplied original full Qwen model, sequence and inferred policy')
    split = info['split']; indices = []
    if len(split.get('sample_ids', [])) != 223:
        raise ValueError('Exactly the original223 stored input IDs are required')
    for name, count in PARTITIONS.items():
        row = split.get(name)
        if (not isinstance(row, list) or len(row) != count or any(type(i) is not int for i in row)):
            raise ValueError('Original stored partition differs: '+name)
        indices.extend(row)
    if sorted(indices) != list(range(223)):
        raise ValueError('Original partitions must cover223 disjoint inputs')
    rows = read_json(args.source_rows)
    if (Path(rows.get('source', '')).name != 'train-00000-of-00001.parquet'
            or rows.get('column') != 'sentence' or rows.get('row_number_base') != 0
            or rows.get('source_rows') != list(range(223))):
        raise ValueError('Use the original223 unlabelled SST2 training sentences')
    if (args.source_rows.resolve() != args.inputs.parent/'source_rows.json'
            or any(value.rsplit(':', 1)[0] != args.inputs.name for value in split['sample_ids'])):
        raise ValueError('Preserve the original sample file and sibling row provenance')
    prepared_path = args.inputs.parent/'worker_result.json'
    prepared = read_json(prepared_path)
    if (prepared.get('status') != 'ok' or prepared.get('phase') != 'prepare-text'
            or prepared.get('sample_count') != 223 or not _same(prepared['inputs_jsonl'], args.inputs)
            or not _same(prepared['source_rows'], args.source_rows)):
        raise ValueError('The saved original text preparation paths differ')
    graph_path = args.model_info / info['graph'] / 'graph.json'
    graph = read_json(graph_path)
    kinds = [row[1] for row in graph['nodes']]
    if kinds.count('rmsnorm') != 49 or kinds.count('softmax') != 24 or kinds.count('act') != 24:
        raise ValueError('All24 original decoder layers and49 normalization nodes must remain')
    archive = args.model_info / info['graph'] / 'weights.npz'
    for item in (args.model, args.inputs, args.source_rows, args.metadata_template, args.frontend,
                 args.profile, graph_path, archive, args.coefficients):
        if not item.exists():
            raise FileNotFoundError(str(item))
    linked = read_json(args.linked_result)
    if (linked.get('status') != 'ok' or linked.get('linked_packed_math_verified') is not True
            or linked.get('same_backend_embedding_handle_consumed') is not True
            or linked.get('candidate_frozen') is not False
            or not _same(linked['model_source']['model_info'], args.model_info)
            or not _same(linked['source']['coefficients'], args.coefficients)):
        raise ValueError('Retain the previously measured same-model packed own-output chain')
    return dict(model_info=str(path.resolve()), original_model=str(args.model.resolve()),
        samples=str(args.inputs.resolve()), source_rows=str(args.source_rows.resolve()),
        original_text_preparation=str(prepared_path.resolve()),
        split_counts=PARTITIONS.copy(), split_indices={name: split[name] for name in PARTITIONS},
        weights_archive=dict(path=str(archive.resolve()), bytes=archive.stat().st_size, values_read=False),
        original_layers=24, graph_operation_counts={kind: kinds.count(kind) for kind in set(kinds)},
        previous_linked_math=str(args.linked_result.resolve()),
        previous_linked_math_scope='Cached packed prefill plus one new packed selector/embedding/forward; cold clock excluded',
        new_generation_scope='Two complete forwards with packed polynomial selection and same-backend embedding feedback; no KV cache',
        report_reader_available=False, extra_integrity_checks=0)


class TokenSplits:
    """Original tokenizer plus readonly original embedding; report access rejected."""
    def __init__(self, info, inputs, tokenizer, access):
        self.info, self.inputs, self.tokenizer, self.access = info, Path(inputs), tokenizer, access

    def get(self, split, n=None):
        if split not in ('calibration', 'heldout', 'confirm'):
            raise ValueError('This continuation never reads report samples')
        indices = self.info.split[split] if n is None else self.info.split[split][:n]
        from ..data.samples import read_selected_text, encode_texts
        import numpy as np
        raw = read_selected_text(self.inputs, self.info.split['sample_ids'], indices)
        texts, labels = raw.select(indices)
        if labels is not None:
            raise ValueError('The original Qwen inputs are unlabelled sentences')
        encoded = encode_texts(self.tokenizer, texts, self.info.sequence, task=self.info.task)
        ids = np.asarray(encoded['input_ids']); mask = np.asarray(encoded['attention_mask'], dtype=np.float64)
        embedding = self.info.graph.weights['original.model.embed_tokens.weight']
        if (ids.shape != (len(indices), 32) or mask.shape != ids.shape
                or ids.min() < 0 or ids.max() >= len(embedding) or not (mask.sum(1) > 0).all()):
            raise ValueError('Original tokenizer/embedding boundary differs')
        selection = np.zeros_like(mask); selection[np.arange(len(mask)), mask.sum(1).astype(int)-1] = 1.
        self.access.append(dict(partition=split, indices=list(indices), count=len(indices)))
        return dict(inputs_embeds=np.asarray(embedding[ids], dtype=np.float64),
                    attention_mask=mask, sel=selection), None


def _load_model(args, result):
    from ..models.mapped_info import load_mapped_info
    from transformers import AutoTokenizer
    info = load_mapped_info(args.model_info)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    result['weight_mapping'] = info.mapped_weights
    return info, TokenSplits(info, args.inputs, tokenizer, result.setdefault('sample_accesses', []))


def _context(args, result):
    from ..planners.agent_tools import build_context
    info, samples = _load_model(args, result)
    calibration = read_json(args.calibration)
    context = build_context(info, calibration, samples=samples, n_val=20)
    context.prompt_version, context.placement, context.tune_search = 'r8-v11', 'program-plan', 'sweep'
    return context


def session_outcome(directory):
    """Classify the unchanged generic state machine without interpreting report data."""
    directory = Path(directory); state_path = directory/'SESSION.json'
    if not state_path.exists():
        return dict(status='not_started', done=False, frozen=False, next_step='evaluate', calls=0)
    state = read_json(state_path)
    if (state.get('prompt_version') != 'r8-v11' or state.get('max_iters') != 10
            or state.get('placement') != 'program-plan' or state.get('confirm_rule') != 'c2'):
        raise ValueError('Retain original v11 max10/program-plan/C2 session policy')
    phase = state['phase']
    if phase not in ('evaluate', 'tune', 'confirm', 'done'):
        raise ValueError('Unknown original Agent phase')
    out = dict(status='running', done=phase == 'done', frozen=False, next_step=phase,
               calls=state['calls'], iteration=state['iteration'])
    if phase != 'done':
        return out
    agent = read_json(directory/'AGENT.json')
    selected = agent['confirm']['selected']; frozen_path = directory/'FROZEN.json'
    if state.get('failed') or agent.get('status') == 'error':
        return dict(out, status='error', error=agent.get('error') or state.get('failed'))
    if selected is None:
        if frozen_path.exists() or agent.get('status') != 'confirm_failed':
            raise ValueError('Unselected terminal session has inconsistent frozen artifacts')
        return dict(out, status='quality_rejected')
    frozen = read_json(frozen_path); source = (directory/'program.ptdsl').read_text()
    semantic = lambda text: ast.dump(ast.parse(text), include_attributes=False)
    passed = [row for row in agent['confirm']['evals'] if row.get('passed') is True
              and row.get('iteration') == selected['iteration'] and row.get('origin') == selected['origin']]
    if (agent.get('status') != 'feasible' or semantic(source) != semantic(selected['source'])
            or not _same(frozen['program'], directory/'program.ptdsl') or len(passed) != 1
            or passed[0].get('tail', {}).get('passed') is not True):
        raise ValueError('Frozen program must retain the original selected source and passing C2 tail')
    return dict(out, status='ok', frozen=True, selected=selected,
                quality_scope='Internal23cal/20heldout/20confirm; no standard LM corpus or report')


def _calibrate(args, recorder, result):
    from ..analyzer import calibrate
    info, samples = _load_model(args, result)
    arrays, _ = samples.get('calibration')
    with recorder.span('original23_training_calibration_batch1'):
        measured = calibrate(info.graph, arrays, batch=1)
    bad = [name for name, row in measured.items()
           if any(type(value) in (float, int) and not math.isfinite(value) for value in row.values())]
    result['nonfinite_calibration_points'] = bad
    if bad:
        raise ValueError('Original generic calibration retained nonfinite observations; no Agent request sent')
    write_json(args.calibration, measured)
    lengths = arrays['attention_mask'].sum(1).astype(int)
    eligible = [i for i, length in enumerate(lengths) if 1 <= length <= 30]
    if not eligible:
        raise ValueError('No original calibration prompt has room for two generated tokens')
    index = info.split['calibration'][eligible[0]]
    generation_input = dict(index=index, sample_id=info.split['sample_ids'][index],
        partition='calibration', prompt_length=int(lengths[eligible[0]]), sequence_length=32,
        selection_rule='First stored calibration prompt with two free token positions')
    write_json(args.calibration.parent/'generation-input.json', generation_input)
    result.update(calibration_completed=True, calibration_samples=23, generation_input=generation_input)


def _agent_phase(args, recorder, result):
    from ..planners.transformer_session import load_state, output_tokens, session_step
    context = _context(args, result)
    outcome = session_outcome(args.session)
    if outcome['done']:
        result.update(session=outcome, status=outcome['status'], candidate_frozen=outcome['frozen'])
        return
    if args.worker_phase == 'prompt':
        from ..planners.prompts import build_system_prompt, initial_message
        from ..autonomous_budget import audit_public_messages
        state_path = args.session/'SESSION.json'
        messages = (load_state(state_path)['messages'] if state_path.exists() else
                    [dict(role='system', content=build_system_prompt(context)),
                     dict(role='user', content=initial_message(context))])
        write_json(args.work_dir/'public-api-request.json', dict(
            endpoint='https://api.deepseek.com/chat/completions', messages=messages,
            message_audit=audit_public_messages(messages), raw_sentences_included=False,
            raw_weights_included=False, report_samples=0,
            provenance='User-authorized DSL instructions, original public architecture and internal calibration/heldout aggregates'))
        result.update(prompt_completed=True, api_required=outcome['next_step'] == 'evaluate', session=outcome)
        return
    client = None
    if outcome['next_step'] == 'evaluate':
        from ..autonomous_budget import RenewingLLMClient
        from .bert_sst2_worker import RecordedRenewingClient
        client = RecordedRenewingClient(RenewingLLMClient(budget_path=args.ledger,
            allowance_path=args.allowance, recorder=recorder, artifact_dir=args.session/'api',
            model='deepseek-flash', reasoning_effort='high', max_tokens=output_tokens('r8-v11'), timeout=600),
            args.session/'calls', recorder=recorder)
    with recorder.span('original_v11_'+outcome['next_step']):
        result['step'] = session_step(context, args.session, client=client, max_iters=10,
            tune_rule='v8-6.25', confirm_rule='c2', tune_policy='all', stale_limit=2,
            focus_retune=True, numeric_lock=True)
    result['api_calls'] = client.calls if client else 0
    result['session'] = session_outcome(args.session)
    result['candidate_frozen'] = result['session']['frozen']
    if result['session']['done']:
        result['status'] = result['session']['status']


def _program(args, result):
    outcome = session_outcome(args.session)
    if not outcome['frozen']:
        raise ValueError('Code generation requires the actual confirmed generic Agent program')
    result['candidate_frozen'] = True
    from ..dsl.parser import parse
    from ..dsl.typecheck import check
    from ..dsl.interface import require_interface
    info, _ = _load_model(args, result)
    program = parse((args.session/'program.ptdsl').read_text())
    require_interface(program, info)
    checked = check(program, {key: value.shape for key, value in info.graph.weights.items()},
                    graph=info.graph, prompt_version='r8-v11')
    if not checked.ok:
        raise ValueError('The confirmed full program must pass its original interface and typecheck')
    return info, program


@contextmanager
def _existing_inputs(backend, values):
    """Reuse real symbolic handles; preserve the initial encrypted argument list."""
    sentinel = object()
    old_input = backend.__dict__.get('input', sentinel)
    old_output = backend.__dict__.get('output', sentinel)
    captured = []
    backend.input = lambda index: values[index]
    backend.output = lambda outputs: captured.extend(outputs)
    try:
        yield captured
    finally:
        for name, old in (('input', old_input), ('output', old_output)):
            if old is sentinel:
                delattr(backend, name)
            else:
                setattr(backend, name, old)


@contextmanager
def _compact_generation():
    from ..kernels import encrypted_generation, compact_linear
    previous = encrypted_generation.embedding_from_selection
    encrypted_generation.embedding_from_selection = compact_linear.embedding_from_selection
    try:
        with compact_linear.compact_transformer_projections():
            yield
    finally:
        encrypted_generation.embedding_from_selection = previous


def lower_two_tokens(program, weights, backend, refresh_plan, prompt_length, coefficients, *, recorder=None):
    """Compose existing full forward/selection/embedding kernels on one backend."""
    from ..backends import RefreshBackend
    from ..kernels.encrypted_generation import EncryptedPrefix, generate_two
    from ..kernels.compact_linear import compact_transformer_projections
    from ..lower import lower
    if [p.name for p in program.params] != ['inputs_embeds', 'attention_mask', 'sel']:
        raise ValueError('Retain original input order and client embedding boundary')
    # Each896-component token occupies one original16384-slot ciphertext.
    if program.header.slots != 16384 or tuple(program.params[0].shape) != (1, 32, 896):
        raise ValueError('The existing two-token bridge requires the original32x896 token layout')
    tokens = [[backend.input(i)] for i in range(32)]
    initial_masks = [backend.input(32), backend.input(33)]
    state = EncryptedPrefix(backend, tokens, prompt_length, 896, 32)
    forwards = []
    def forward(current):
        if not current.selections:
            masks = initial_masks
        else:
            # Public length/masks change with a fixed server step; token values stay symbolic.
            public = current.public_masks(); zero = current.token_blocks[0][0] * backend.const([0.])
            import numpy as np
            masks = []
            for name in ('attention_mask', 'sel'):
                vector = np.zeros(backend.slots, dtype=np.float64); vector[:32] = public[name][0]
                masks.append(zero + backend.const(vector))
        values = [token[0] for token in current.token_blocks] + masks
        active = RefreshBackend(backend, policy='plan', plan=refresh_plan, **refresh_plan['levels'])
        with _existing_inputs(backend, values) as captured, compact_transformer_projections():
            output, layouts = lower(program, weights, active, act_mode='program')
        if len(layouts) != 1:
            raise ValueError('Retain one full vocabulary output')
        forwards.append(dict(step=len(forwards), prefix_length=current.length,
                             output_blocks=len(output), refresh=active.stats()))
        if len(captured) != len(output):
            raise ValueError('Capture the refreshed native output handles exactly once')
        return captured
    with _compact_generation():
        generated = generate_two(state, forward, weights['original.model.embed_tokens.weight'],
            bound=32., normalized_gap=2.**-27, coefficients=coefficients)
    backend.output([item for row in generated['token_selections'] for item in row])
    return dict(generated['metadata'], forwards=forwards, same_backend=True,
        selector_quality_for_new_candidate=None, standard_lm_quality=None,
        scope='Unmodified full24-layer forward twice, existing cubic52 selector and original embedding; diagnostic compilation only')


def _codegen(args, recorder, result):
    if args.worker_phase == 'placement':
        info, program = _program(args, result)
        from ..plan import plan_program
        with recorder.span('unchanged_generic_program_plan'):
            plan, backend, placement = plan_program(program, info.graph.weights, profile='qwen48')
            write_json(args.codegen/'refresh-plan.json', plan)
            result.update(placement_completed=True, planned_prefill_bootstraps=plan['boots'])
            del backend, placement; gc.collect()
        return
    if args.worker_phase == 'emit':
        info, program = _program(args, result)
        from ..backends.streaming_emit import StreamingEmitBackend
        from .qwen_monotone_selector import analytic_coefficients, config
        artifact = read_json(args.coefficients)
        if artifact.get('config', artifact) != config() or artifact.get('format') != 1:
            raise ValueError('Retain the existing explicitly recorded cubic52 selector variant')
        import numpy as np
        coefficients = analytic_coefficients(np)
        if artifact.get('stages') != [stage.tolist() for stage in coefficients]:
            raise ValueError('Original cubic52 polynomial coefficient values differ')
        backend = StreamingEmitBackend(program.header.slots, directory=args.codegen/'emit',
            frontend=args.frontend, max_constant_bytes=int(args.case_gib*GIB-GIB))
        try:
            with recorder.span('full_two_token_streaming_emit'):
                result['generation'] = lower_two_tokens(program, info.graph.weights, backend,
                    read_json(args.codegen/'refresh-plan.json'),
                    read_json(args.calibration.parent/'generation-input.json')['prompt_length'], coefficients)
                result['emit_stats'] = backend.stats()
                result['trace_source'] = str(backend.write().resolve())
                result['emit_completed'] = True
        finally:
            if not backend.finished:
                backend.close()
        return
    from .. import toolchain
    host = toolchain.host_paths(); emit = args.codegen/'emit'
    env = dict(os.environ, LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'],
               HECATE=str(host['hecate']), HECATE_BUILD=str(host['hecate_build']))
    if args.worker_phase == 'trace':
        command = [sys.executable, '-B', str(emit/'candidate_trace.py')]
    else:
        from .r29_local_compile_matrix import compiler_argv
        command = compiler_argv(read_json(args.metadata_template), 'pars', binary=host['hecate_opt'],
            source=emit/'candidate_trace.mlir', profile=args.profile, output=emit/'lowered.mlir')
    write_json(args.work_dir/'native-argv.json', [str(item) for item in command])
    observed = run_process(command, cwd=emit, log=args.work_dir/'native.log',
        resources=args.work_dir/'native-resources.jsonl', timeout=PHASE_SECONDS[args.worker_phase],
        mem_gib=args.memory_gib, fsize_gib=8, env=env)
    result['process'] = observed
    if observed['exit_code'] or observed['timed_out'] or observed['signal'] is not None or observed['remaining_pids']:
        raise RuntimeError('Native '+args.worker_phase+' did not finish cleanly')
    if args.worker_phase == 'trace':
        mlir = emit/'candidate_trace.mlir'
        if not mlir.is_file() or not mlir.stat().st_size:
            raise ValueError('The native save returned no complete candidate_trace.mlir')
        result.update(trace_completed=True, mlir=dict(path=str(mlir), bytes=mlir.stat().st_size))
    else:
        plans = list(emit.glob('lowered.*.runtime-plan.json'))
        if len(plans) != 1:
            raise ValueError('Expected one complete two-token RuntimePlan')
        plan = read_json(plans[0])
        result.update(compile_completed=True, runtime_plan=str(plans[0]),
            compiled_operation_count=len(plan['execution']),
            compiled_bootstraps=sum(row.get('op') == 'boot' for row in plan['execution']))


def worker(args):
    # All paths are explicit argv; the worker never reads the run registry or RESULTS.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    args.work_dir.mkdir(parents=True, exist_ok=False)
    recorder = StageRecorder(args.work_dir/'events.jsonl')
    result = dict(round=ROUND, model=MODEL, phase=args.worker_phase, status='running', api_calls=0,
                  candidate_frozen=False, **dict.fromkeys(ZERO, 0), **dict.fromkeys(FORMAL))
    started = time.perf_counter_ns()
    write_json(args.work_dir/'worker-result.json', result)
    try:
        result['source'] = source_gate(args)
        from .r29_local_compile_matrix import native_metadata_commands
        with native_metadata_commands(read_json(args.metadata_template)):
            import torch
            torch.set_num_threads(1); torch.set_num_interop_threads(1)
            with recorder.span('qwen_'+args.worker_phase):
                if args.worker_phase == 'calibration':
                    _calibrate(args, recorder, result)
                elif args.worker_phase in ('prompt', 'step'):
                    _agent_phase(args, recorder, result)
                else:
                    _codegen(args, recorder, result)
            if result['status'] == 'running':
                result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:4000])
    result.update(elapsed_ns=time.perf_counter_ns()-started,
                  worker_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(args.work_dir/'worker-result.json', result)
    return 0 if result['status'] in ('ok', 'quality_rejected') else 1


def phase_command(args, phase, work):
    seconds = PHASE_SECONDS[phase]
    argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds+60),
            '--python', str(args.intake_python), '--', '-B', '-m', MODULE,
            '--worker-phase', phase, '--work-dir', str(work)]
    for name in ('model', 'model_info', 'inputs', 'source_rows', 'metadata_template', 'frontend', 'profile',
                 'linked_result', 'coefficients', 'calibration', 'session', 'codegen', 'ledger', 'allowance'):
        argv.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
    return argv+['--memory-gib', str(args.memory_gib), '--case-gib', str(args.case_gib)]+(['--server-mode'] if args.server_mode else [])


def _budget(args, out):
    size = artifact_bytes(out); free = shutil.disk_usage(out).free
    if size+1024**2 > args.case_gib*GIB or free < 10*GIB:
        raise RuntimeError('Qwen case artifact allowance or10GiB disk reserve exhausted')
    return dict(case_bytes=size, free_bytes=free)


def _append_registry(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def register_api_calls(ledger, registry, work, *, before, after, git_rev):
    """Register only this outer ledger interval, retaining earlier unique call rows."""
    from ..autonomous_budget import call_records
    rows = call_records(ledger)
    if not 0 <= before <= after <= len(rows):
        raise ValueError('Invalid explicit Qwen ledger interval')
    registered, retained = [], []
    registry = Path(registry); registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX); stream.seek(0)
        existing = {row.get('metrics', {}).get('global_call') for line in stream if line.strip()
                    for row in [json.loads(line)] if row.get('mode') == 'llm-call'}
        stream.seek(0, os.SEEK_END)
        for number in range(before+1, after+1):
            if number in existing:
                retained.append(number); continue
            call = rows[number-1]
            row = dict(run_id=Path(work).parent.parent.name+'-qwen-api-'+str(number), round=ROUND,
                model=MODEL, mode='llm-call', git_rev=git_rev,
                metrics=dict(call, global_call=number, source_ledger=str(Path(ledger).resolve()), prompt_version='r8-v11'),
                artifacts_dir=str(work), ts=call['ts'])
            stream.write(json.dumps(row, allow_nan=False)+'\n'); registered.append(number)
        stream.flush(); os.fsync(stream.fileno())
    return dict(registered=registered, retained=retained, source_ledger=str(Path(ledger).resolve()))


def write_server_job(args, target):
    """Write a portable larger-host fallback; this function starts no process."""
    source = source_gate(args); target = Path(target).resolve(); target.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-B', '-m', MODULE, '--out', str(target/'actual-run'),
            '--through', 'compile', '--server-mode', '--memory-gib', '112', '--rss-gib', '112',
            '--case-gib', '512', '--seconds', '172800']
    for name in ('model', 'model_info', 'inputs', 'source_rows', 'metadata_template', 'frontend', 'profile',
                 'linked_result', 'coefficients', 'intake_python', 'ledger', 'allowance'):
        argv += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    script = target/'run.sh'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+
        shlex.quote(str(Path(__file__).resolve().parents[2]))+'\nexec '+shlex.join(argv)+'\n')
    script.chmod(0o755)
    write_json(target/'job.json', dict(status='prepared_not_executed', argv=argv, source=source,
        required_hardware=dict(architecture='aarch64-linux', physical_ram_gib=128, address_space_gib=112, resident_gib=112,
                               work_disk_gib=512, free_disk_reserve_gib=10, threads=1),
        hardware_scope='Finite planning allowance; successful full DAG/trace/native memory remains unmeasured',
        artifact_transport='Mount the explicit original model/input/toolchain paths from argv; payloads are not copied into this package',
        stages=['original23 calibration', 'v11 Agent/heldout20/C2confirm20', 'confirmed program-plan',
                'existing streaming Emit of two complete24-layer forwards and selectors', 'trace once', 'compile once'],
        downloads=0, installations=0, model_modifications=0, report_samples=0, api_calls=0,
        candidate_frozen=False, full_generation_quality_acceptance=None, encrypted_execution=False))


def execute(args):
    from .. import paths
    from ..autonomous_budget import call_records
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    args.session, args.codegen, args.calibration = out/'session', out/'codegen', out/'calibration.json'
    args.codegen.mkdir(); started = time.perf_counter_ns(); deadline = time.monotonic()+args.seconds
    recorder = StageRecorder(out/'events.jsonl')
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', phases=[],
        source_checked=False, candidate_frozen=False, api_calls=0, termination=None,
        attempted=dict.fromkeys(PHASE_SECONDS, 0), completed=dict.fromkeys(PHASE_SECONDS, 0),
        through=args.through, compiled_bootstraps=None, complete_prediction=None,
        scope='Cold continuation from saved original import; historical import/reference clocks excluded',
        generation_protocol=dict(model_forward_calls=2, generated_tokens=2, kv_cache_used=False,
            actual_two_token_execution_completed=False), **dict.fromkeys(ZERO, 0), **dict.fromkeys(FORMAL))
    stop = threading.Event(); failures = []
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    def interrupt(sig, frame):
        if not failures:
            result['termination'] = dict(cause='incoming_signal', signal=sig)
        raise InterruptedError('Qwen continuation interrupted')
    for sig in old:
        signal.signal(sig, interrupt)
    def monitor():
        from .r10_paper_metrics import _scan
        while not stop.wait(.5):
            try:
                observation = _budget(args, out)
                table = _scan(); family = {os.getpid()}
                while True:
                    extra = {pid for pid, row in table.items() if row['ppid'] in family}
                    if extra.issubset(family): break
                    family.update(extra)
                rss = sum(table[pid].get('vmrss_kib', 0) for pid in family if pid in table)*1024
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:')))*1024
                observation.update(ts_ns=time.perf_counter_ns(), subtree_rss_bytes=rss, available_bytes=available)
                with (out/'guard-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(observation)+'\n')
                if rss > args.rss_gib*GIB or available < GIB or time.monotonic() >= deadline:
                    raise RuntimeError('Resident memory, host reserve or total wall allowance exhausted')
            except BaseException as exc:
                if not failures:
                    failures.append(str(exc)); result['termination'] = dict(cause='own_budget_guard', error=str(exc))
                    try:
                        write_json(out/'guard-failure.json', result['termination'])
                    finally:
                        os.kill(os.getpid(), signal.SIGUSR1)
                return
    thread = threading.Thread(target=monitor, daemon=True)
    before, git_rev = None, 'unknown'
    try:
        before = len(call_records(args.ledger))
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                             capture_output=True, text=True)
        git_rev = git.stdout.strip() if git.returncode == 0 else 'unknown'
        thread.start()
        with recorder.span('qwen_continuation'):
            result['source'] = source_gate(args); result['source_checked'] = True
            source_dir = out/'source'; source_dir.mkdir()
            shutil.copyfile(__file__, source_dir/'entry-source.py')
            root = Path(__file__).resolve().parents[1]
            for relative in ('models/mapped_info.py', 'data/samples.py', 'planners/transformer_session.py',
                'planners/agent.py', 'planners/agent_tools.py', 'planners/confirm.py',
                'transformer_validation.py', 'analyzer.py', 'backends/streaming_emit.py',
                'kernels/encrypted_generation.py', 'kernels/compact_linear.py', 'plan.py'):
                shutil.copyfile(root/relative, source_dir/relative.replace('/', '_'))
            shutil.copyfile(args.model_info/'model_info.json', source_dir/'original-model-info.json')
            for name in ('metadata_template', 'source_rows'):
                shutil.copyfile(getattr(args, name), source_dir/(name+'.json'))
            if args.through != 'metadata':
                phases = ['calibration', 'prompt']
                def launch(phase):
                    if failures: raise RuntimeError(failures[0])
                    _budget(args, out)
                    remaining = deadline-time.monotonic()
                    if remaining <= 0: raise TimeoutError('Complete Qwen allowance exhausted')
                    work = out/'phases'/(str(len(result['phases'])+1).zfill(3)+'-'+phase)
                    argv = phase_command(args, phase, work)
                    write_json(out/('command-'+str(len(result['phases'])+1)+'.json'), argv)
                    result['attempted'][phase] += 1
                    write_json(out/'result.json', result)
                    with recorder.span('phase_'+phase):
                        observed = run_process(argv, cwd=Path(__file__).resolve().parents[2],
                            log=out/(work.name+'.log'), resources=out/(work.name+'-resources.jsonl'),
                            timeout=min(PHASE_SECONDS[phase]+120, remaining), mem_gib=args.memory_gib,
                            fsize_gib=8, env=dict(os.environ))
                    path = work/'worker-result.json'; child = read_json(path) if path.is_file() else None
                    result['phases'].append(dict(phase=phase, process=observed, worker=child))
                    write_json(out/'result.json', result)
                    if (observed['exit_code'] or observed['timed_out'] or observed['signal'] is not None
                            or observed['remaining_pids'] or not child or child['status'] not in ('ok', 'quality_rejected')):
                        raise RuntimeError('First failed Qwen phase: '+phase)
                    result['completed'][phase] += 1
                    return child
                for phase in phases: launch(phase)
                if args.through != 'prompt':
                    for _ in range(21):
                        child = launch('step'); outcome = child['session']
                        if outcome['done']:
                            result.update(status=outcome['status'], candidate_frozen=outcome['frozen'], session=outcome)
                            break
                    else:
                        raise RuntimeError('Original max10 session exceeded21 finite tool steps')
                    if result['candidate_frozen'] and args.through == 'compile':
                        for phase in ('placement', 'emit', 'trace', 'compile'):
                            child = launch(phase)
                        result['compiled_bootstraps'] = child['compiled_bootstraps']
                        result['compiled_operation_count'] = child['compiled_operation_count']
                        result['two_token_codegen_completed'] = True
            if result['status'] == 'running': result['status'] = 'prepared'
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:4000])
    finally:
        stop.set()
        if thread.is_alive(): thread.join(timeout=2)
        for sig, handler in old.items(): signal.signal(sig, handler)
        # Explicit outer paths; each real call keeps one original eight-field registry row.
        try:
            after = len(call_records(args.ledger))
            result['api_calls'] = after-before if before is not None else None
            if before is None: raise RuntimeError('Initial API ledger observation unavailable')
            result['api_registration'] = register_api_calls(args.ledger, paths.RESULTS_DIR/'registry.jsonl',
                out/'phases'/'qwen-accounting', before=before, after=after, git_rev=git_rev)
        except BaseException as exc:
            result.update(status='error', accounting_error=type(exc).__name__+': '+str(exc))
        result['git_rev'] = git_rev
        result['recorded_pre_persistence_elapsed_seconds'] = (time.perf_counter_ns()-started)/1e9
        result['timing'] = summarize_events(read_events(recorder.path))
        result['artifact_bytes'] = artifact_bytes(out)
        try:
            _budget(args, out)
        except BaseException as exc:
            result.update(status='error', terminal_budget_error=str(exc))
        result['metrics'] = {key: result[key] for key in ('status', 'candidate_frozen', 'api_calls',
            'compiled_bootstraps', 'accuracy_acceptance', 'predicted_latency', 'CKKS_latency', 'security_acceptance')}
        for _ in range(12):
            write_json(out/'result.json', result)
            amount = artifact_bytes(out)
            if result['artifact_bytes'] == amount: break
            result['artifact_bytes'] = amount
        _append_registry(paths.RESULTS_DIR/'registry.jsonl', dict(run_id=out.name, round=ROUND,
            model=MODEL, mode=MODE, git_rev=git_rev, metrics=result['metrics'], artifacts_dir=str(out),
            ts=datetime.now(timezone.utc).isoformat()))
        persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
            timing=result['timing'], required_artifact_bytes=artifact_bytes(out))
    return 0 if result['status'] in ('ok', 'prepared', 'quality_rejected') else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-phase', choices=tuple(PHASE_SECONDS))
    p.add_argument('--through', choices=('metadata', 'prompt', 'agent', 'compile'), default='agent')
    for name in ('out', 'work-dir', 'model', 'model-info', 'inputs', 'source-rows', 'metadata-template',
                 'frontend', 'profile', 'linked-result', 'coefficients', 'calibration', 'session', 'codegen',
                 'intake-python', 'ledger', 'allowance'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--memory-gib', type=float, default=12.)
    p.add_argument('--rss-gib', type=float, default=14.)
    p.add_argument('--case-gib', type=int, default=32)
    p.add_argument('--seconds', type=int, default=21600)
    p.add_argument('--server-mode', action='store_true')
    p.add_argument('--make-server-job', type=Path)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if not (0 < args.memory_gib <= (112 if args.server_mode else 24)
            and 0 < args.rss_gib <= (112 if args.server_mode else 14)
            and 2 <= args.case_gib <= (512 if args.server_mode else 64) and 1 <= args.seconds <= 172800):
        p.error('Use local AS<=24GiB/RSS<=14GiB with finite artifacts and wall time; server allowances are separate')
    if not args.worker_phase:
        from .. import paths
        defaults = dict(model=paths.RESULTS_DIR/'preparation-20261115/models/Qwen2-0.5B',
            model_info=paths.RESULTS_DIR/'qwen-normalization-rebased-20261115/work/import/model_info',
            inputs=paths.RESULTS_DIR/'r10-paper-qwen-import-20261115/worker/text-input/samples.jsonl',
            source_rows=paths.RESULTS_DIR/'r10-paper-qwen-import-20261115/worker/text-input/source_rows.json',
            metadata_template=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json',
            frontend=paths.DACAPO_ROOT/'python/hecate/hecate/expr.py', profile=paths.PROFILES_DIR/'qwen48/compiler-config.json',
            linked_result=paths.RESULTS_DIR/'qwen-linked-packed-generation-20261116/result.json',
            coefficients=paths.RESULTS_DIR/'qwen-bounded-cubic-selector52-20261116/prepare/coefficients.json',
            intake_python=paths.INTAKE_PYTHON, ledger=paths.RESULTS_DIR/'llm_calls.jsonl',
            allowance=paths.RESULTS_DIR/'autonomous-api-allowance.json')
        for name, value in defaults.items():
            if getattr(args, name) is None: setattr(args, name, value)
        if args.out is None: p.error('Choose a fresh explicit --out')
    else:
        for name in ('work_dir', 'model', 'model_info', 'inputs', 'source_rows', 'metadata_template', 'frontend',
                     'profile', 'linked_result', 'coefficients', 'calibration', 'session', 'codegen', 'ledger', 'allowance'):
            if getattr(args, name) is None: p.error('Inner phase requires explicit --'+name.replace('_', '-'))
    for name, value in vars(args).items():
        if isinstance(value, Path):
            # Keep the existing host-visible symlink: its /nix/store target is available only inside Nix.
            setattr(args, name, value.expanduser().absolute() if name == 'intake_python' else value.resolve())
    if args.make_server_job:
        write_server_job(args, args.make_server_job)
        return 0
    if args.server_mode and not args.worker_phase:
        total = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemTotal:')))*1024
        if total < 120*GIB:
            p.error('The explicit server job requires at least120GiB physical RAM; no child started')
    return worker(args) if args.worker_phase else execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
