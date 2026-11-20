"""Count existing BERT placements by semantic statement; no compiler or provider."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .. import paths
from .r10_paper_metrics import run_process, write_json

CATEGORIES = ('act', 'softmax', 'attention', 'norm', 'other')
VARIANTS = ('rule63', 'softmax_only', 'arranger')


def read(path):
    return json.loads(Path(path).read_text())


def softmax_only(rule, frozen):
    """Replace only softmax calls, rebinding their input and mask to rule names."""
    def calls(text):
        tree = ast.parse(text)
        return [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and
                isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and
                node.value.func.id == 'softmax']
    original, replacement = calls(rule), calls(frozen)
    if not original or len(original) != len(replacement):
        raise ValueError('Require the same nonempty ordered set of softmax statements')
    lines = rule.splitlines(keepends=True)
    for old, new in reversed(list(zip(original, replacement))):
        if len(old.value.args) != 1 or len(new.value.args) != 1:
            raise ValueError('Require one explicit softmax score input')
        call = deepcopy(new.value)
        call.args = deepcopy(old.value.args)
        old_mask = [x for x in old.value.keywords if x.arg == 'mask']
        new_mask = [x for x in call.keywords if x.arg == 'mask']
        if len(old_mask) != 1 or len(new_mask) != 1:
            raise ValueError('Require one existing mask reference per softmax')
        new_mask[0].value = deepcopy(old_mask[0].value)
        statement = ast.Assign(targets=deepcopy(old.targets), value=call)
        ast.fix_missing_locations(statement)
        ending = '\n' if lines[old.end_lineno-1].endswith('\n') else ''
        lines[old.lineno-1:old.end_lineno] = [' '*old.col_offset + ast.unparse(statement) + ending]
    return ''.join(lines)


def category(kind):
    return {'act': 'act', 'softmax': 'softmax', 'matmul': 'attention',
            'layernorm': 'norm', 'rmsnorm': 'norm'}.get(kind, 'other')


def aggregate(program, owners, nodes):
    statements = {op.id: op for op in program.ops}
    counts = Counter(owners[node] for node in nodes)
    if any(owner is not None and owner not in statements for owner in counts):
        raise ValueError('Placement contains an unknown DSL statement owner')
    totals = dict.fromkeys(CATEGORIES, 0)
    rows = []
    for op in program.ops:
        name = category(op.kind)
        totals[name] += counts[op.id]
        rows.append(dict(id=op.id, line=op.lineno, kind=op.kind, category=name, boots=counts[op.id]))
    totals['other'] += counts[None]
    if counts[None]:
        rows.append(dict(id=None, line=None, kind='input', category='other', boots=counts[None]))
    return totals, rows, counts[None]


def initialize():
    from .bert_sst2_worker import initialize_numeric
    initialize_numeric()


def count_worker(args):
    initialize()
    from .r29_local_compile_matrix import native_metadata_commands
    with native_metadata_commands(read(args.metadata_template)):
        from ..models.info import ModelInfo
        from ..dsl.parser import parse
        from ..dsl.typecheck import check
        from ..dsl.interface import require_interface
        from ..dsl.ir import canonical
        from ..plan import plan_program, verify_program
        info = ModelInfo.load(args.model_dir)
        rule, frozen = args.rule.read_text(), args.frozen.read_text()
        text = softmax_only(rule, frozen) if args.variant == 'softmax_only' else (
            rule if args.variant == 'rule63' else frozen)
        program = parse(text)
        if args.variant == 'softmax_only':
            original = parse(rule)
            if program.header != original.header or program.params != original.params or program.outputs != original.outputs:
                raise ValueError('Only softmax statements may change')
            before, after = canonical(original), canonical(program)
            for old, new, op in zip(before['ops'], after['ops'], original.ops):
                if op.kind != 'softmax' and old != new:
                    raise ValueError('A non-softmax statement changed')
            if len(before['ops']) != len(after['ops']):
                raise ValueError('The original statement set must remain fixed')
        require_interface(program, info)
        checked = check(program, {k: v.shape for k, v in info.graph.weights.items()},
                        graph=info.graph, prompt_version='r8-v11')
        if not checked.ok:
            raise ValueError('Original or mixed program did not typecheck: ' + str(checked.diags))
        args.out.mkdir(exist_ok=False)
        source = args.out / 'program.ptdsl'
        source.write_text(text)
        started = time.perf_counter_ns()
        plan, dag, placed = plan_program(program, info.graph.weights, profile='qwen48', frontend_normalize=True)
        totals, statements, input_owned = aggregate(program, dag.prog_op, placed.boot_nodes)
        if sum(totals.values()) != placed.boots or placed.boots != plan['boots']:
            raise ValueError('Semantic count partition differs from the actual selected plan')
        write_json(args.out / 'refresh-plan.json', plan)
        write_json(args.out / 'statement-counts.json', statements)
        write_json(args.out / 'bootstrap-source-nodes.json',
                   [dict(node=n, statement_id=dag.prog_op[n]) for n in placed.boot_nodes])
        replay = verify_program(program, info.graph.weights, plan, profile='qwen48')
        expected = {'rule63': 12713, 'arranger': 7251}.get(args.variant)
        row = dict(label=args.variant, boots=totals, total=replay['boots'], placement=program.header.placement,
                   source=str(args.rule if args.variant == 'rule63' else args.frozen if args.variant == 'arranger' else source),
                   program=str(source), feasible=True, status='ok', count_wall_s=(time.perf_counter_ns()-started)/1e9,
                   plan_total=plan['boots'], CountExpr_replay=replay, compiled_reference_total=expected,
                   difference_from_compiled_reference=None if expected is None else replay['boots']-expected,
                   compiled_reference_matches=None if expected is None else replay['boots']==expected,
                   placement_variant=plan['placement'], raw_candidates=plan.get('candidates'), merged_nodes=len(plan['merged']),
                   input_owned_bootstraps=input_owned, statement_counts=str(args.out / 'statement-counts.json'),
                   attribution_scope='DSL statement owning the refreshed value in the normalized cost DAG; '
                                     'same ownership used by transformer_cost.statement_costs',
                   native_compile_executed=False, API=0, CKKS=0, sample_reads=0)
        if expected is not None and replay['boots'] != expected:
            row['feasible'] = False
        write_json(args.out / 'result.json', row)


def quality_worker(args):
    initialize()
    import numpy as np
    from .r29_local_compile_matrix import native_metadata_commands
    with native_metadata_commands(read(args.metadata_template)):
        from ..models.info import ModelInfo
        from ..dsl.parser import parse
        from ..analyzer import reference_forward
        from ..tensor_sim import run_tensor
        info = ModelInfo.load(args.model_dir)
        if info.threshold != {'acc_drop_pp': 0.5} or info.split['heldout'] != list(range(1000, 2000)):
            raise ValueError('Keep the original model policy and fixed 1000-row heldout partition')
        program = parse(args.program.read_text())
        sample_file = args.model_dir / 'samples/heldout.npz'
        with np.load(sample_file, allow_pickle=False) as saved:
            labels = saved['labels']
            inputs = {p.name: saved[p.name] for p in program.params}
        if len(labels) != 1000 or any(len(x) != 1000 or x.dtype != np.float64 for x in inputs.values()):
            raise ValueError('Keep all original float64 heldout inputs')
        args.out.mkdir(exist_ok=False)
        reference, approximate = [], []
        for start in range(0, 1000, 20):
            batch = {k: v[start:start+20] for k, v in inputs.items()}
            exact = reference_forward(info.graph, batch)[info.graph.outputs[0]]
            observed = run_tensor(program, info.graph.weights, batch, cache=False)
            if exact.shape != observed.shape or exact.dtype != np.float64 or observed.dtype != np.float64:
                raise ValueError('Original and approximate tensor output signatures differ')
            reference.append(exact)
            approximate.append(observed)
            write_json(args.out / 'progress.json', dict(status='running', completed=start+len(exact), total=1000))
        exact, observed = np.concatenate(reference), np.concatenate(approximate)
        finite = bool(np.isfinite(exact).all() and np.isfinite(observed).all())
        original_top1 = float(100*np.mean(exact.argmax(1)==labels))
        approximate_top1 = float(100*np.mean(observed.argmax(1)==labels)) if finite else None
        agree = int(np.sum(exact.argmax(1)==observed.argmax(1))) if finite else None
        drop = original_top1-approximate_top1 if finite else None
        result = dict(status='ok', split='heldout', total=1000, all_finite=finite,
                      agreement_count=agree, agreement_pct=agree/10 if finite else None,
                      original_top1_pct=original_top1, approximate_top1_pct=approximate_top1,
                      acc_drop_pp=drop, policy=info.threshold,
                      quality_accepted=bool(finite and drop <= info.threshold['acc_drop_pp']),
                      sample_source=str(sample_file), source_program=str(args.program),
                      source_model=str(args.model_dir), evaluation_engine='float64 tensor',
                      samples_returned_to_Agent=False, confirmation_reads=0, report_reads=0, API=0, CKKS=0)
        np.savez(args.out / 'outputs.npz', reference=exact, approximate=observed, labels=labels)
        write_json(args.out / 'result.json', result)


def worker_command(args, phase, out, *, variant=None, program=None):
    argv = [str(args.python), '-B', '-m', __spec__.name, '--worker', phase,
            '--out', str(out), '--model-dir', str(args.model_dir), '--metadata-template', str(args.metadata_template)]
    if phase == 'count':
        argv += ['--variant', variant, '--rule', str(args.rule), '--frozen', str(args.frozen)]
    else:
        argv += ['--program', str(program)]
    return argv


def publication_state(args, state, complete=False):
    """Update only delivery metadata after the S2 publisher has exited."""
    for root in (args.paper, args.mirror):
        path = root / 'EXPERIMENT_DELIVERY.json'
        value = read(path)
        value.update(status='current_required_experiment_scope_complete' if complete else 'additional_S3_BERT_work_active',
                     current_required_experiment_scope_complete=complete,
                     S3_BERT_Fig2=dict(status=state['status'], source=str(args.out / 'result.json'),
                                      progress=str(args.out / 'progress.json')),
                     last_metadata_revision_utc=datetime.now(timezone.utc).isoformat())
        if complete:
            value['links']['BERT_Fig2_components'] = str(args.paper / 'bert_fig2_components.json')
        write_json(path, value)


def main(args):
    if args.worker:
        return count_worker(args) if args.worker == 'count' else quality_worker(args)
    from .completion_metrics import persist_case_completion, artifact_bytes
    from .s2_heir_serial import record_registry
    started = time.perf_counter_ns()
    args.out.mkdir(exist_ok=False)
    state = dict(round='S3-BERT-FIG2', status='waiting_for_S2_delivery', rows=[], processes=[],
                 API=0, CKKS=0, compiler_runs=0, report_reads=0, confirmation_reads=0)
    write_json(args.out / 'progress.json', state)
    while not (args.predecessor / 'result.json').is_file():
        if datetime.now(timezone.utc) >= datetime(2026, 11, 18, 12, tzinfo=timezone.utc):
            raise TimeoutError('S3 did not start before the data cutoff')
        time.sleep(5)
    if read(args.predecessor / 'result.json')['status'] != 'ok':
        raise ValueError('S2 source delivery requires closure before new heavy work')
    publication_state(args, state)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LD_', 'NIX_', '__NIX_'))
           and k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open() as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for variant in VARIANTS:
            state.update(status='counting', active_variant=variant)
            write_json(args.out / 'progress.json', state)
            publication_state(args, state)
            child = args.out / variant
            obs = run_process(worker_command(args, 'count', child, variant=variant),
                              cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / (variant+'.log'),
                              resources=args.out / (variant+'-resources.jsonl'), timeout=3600,
                              mem_gib=14, fsize_gib=4, env=env)
            state['processes'].append(dict(variant=variant, observation=obs))
            if obs['exit_code'] != 0 or obs['timed_out'] or obs['remaining_pids']:
                row = dict(label=variant, boots=dict.fromkeys(CATEGORIES), total=None, placement=None,
                           source=str(child), feasible=False, status='driver_error', error='Count worker did not close cleanly')
            else:
                row = read(child / 'result.json')
            state['rows'].append(row)
            write_json(args.out / 'progress.json', state)
        middle = state['rows'][1]
        if middle['status'] == 'ok':
            state.update(status='quality', active_variant='softmax_only')
            write_json(args.out / 'progress.json', state)
            child = args.out / 'softmax_only-quality'
            obs = run_process(worker_command(args, 'quality', child, program=Path(middle['program'])),
                              cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / 'quality.log',
                              resources=args.out / 'quality-resources.jsonl', timeout=1800,
                              mem_gib=4, fsize_gib=4, env=env)
            state['processes'].append(dict(variant='softmax_only-quality', observation=obs))
            if obs['exit_code'] == 0 and not obs['timed_out'] and obs['remaining_pids'] == []:
                middle['quality'] = read(child / 'result.json')
                middle['feasible'] = middle['feasible'] and middle['quality']['quality_accepted']
            else:
                middle.update(feasible=False, quality=dict(status='driver_error', source=str(child)))
    completed = all(row['status'] == 'ok' for row in state['rows']) and middle.get('quality', {}).get('status') == 'ok'
    state.update(status='ok' if completed else 'completed_with_errors', active_variant=None, source=str(args.out / 'result.json'),
                 count_method='Normalized cost DAG and unchanged native-equivalent placement; actual CountExpr plan replay',
                 quality_split='fixed heldout[1000:2000], not confirmation or report',
                 rounds_returned_to_Agent=False)
    write_json(args.out / 'result.json', state)
    record_registry(args.out, 'S3-BERT-FIG2', 'bert_tiny', 'semantic-component-count-and-heldout', state['status'],
                    dict(totals={r['label']:r['total'] for r in state['rows']}, API=0, CKKS=0))
    persist_case_completion(args.out, round_label='S3-BERT-FIG2', started_ns=started, status=state['status'],
            timing=dict(status='ok', attribution_ready=False, spans=[], diagnostics=[]), required_artifact_bytes=artifact_bytes(args.out))
    for root in (args.paper, args.mirror):
        write_json(root / 'bert_fig2_components.json', state)
    publication_state(args, state, complete=completed)
    print(json.dumps(dict(status=state['status'], totals={r['label']:r['total'] for r in state['rows']})), flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=('count', 'quality'))
    p.add_argument('--variant', choices=VARIANTS)
    for key in ('out', 'rule', 'frozen', 'model-dir', 'metadata-template', 'predecessor', 'paper', 'mirror', 'program'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--python', type=Path, default=paths.WORKSPACE / 'baselines/orion-venv/bin/python')
    args = p.parse_args()
    required = ('out', 'model_dir', 'metadata_template') + (('rule', 'frozen', 'variant') if args.worker == 'count'
               else ('program',) if args.worker == 'quality' else ('rule', 'frozen', 'predecessor', 'paper', 'mirror'))
    if any(getattr(args, key) is None for key in required):
        p.error('Explicit original source paths are required: ' + ','.join(required))
    return args


if __name__ == '__main__':
    main(arguments())
