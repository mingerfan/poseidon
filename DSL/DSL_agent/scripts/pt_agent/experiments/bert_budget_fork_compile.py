"""Compile one accepted warm BERT budget-stop fork with explicit intake lineage.

The original intake/checkpoint keeps its original paths. The new copied model
metadata and read-only array references are validated as a distinct source.
Original calibration3/strict9 math and all trusted compiler bodies are reused.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
from types import SimpleNamespace

from .. import paths
from . import bert_compile_first_deferred_report as base
from . import bert_retained_budget_confirm as fork
from . import bert_serial_continuation as serial
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events

ROUND = 'BERT-BUDGET-FORK-COMPILE'
MODEL = base.MODEL
MODE = 'warm-original-c2-budget-fork-calibration3-explicit-intake-bridge-compile'
MODULE = 'pt_agent.experiments.bert_budget_fork_compile'
BRIDGE_FIELDS = ('original_source_case', 'continuation_case', 'boundary_snapshot')


def _same(value, expected):
    return isinstance(value, (str, Path)) and Path(value).resolve() == Path(expected).resolve()


def _one(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv):
        raise ValueError('Original confirmation command has an ambiguous required path: '+flag)
    return Path(argv[argv.index(flag)+1]).resolve()


def _inner(binding):
    return bool(getattr(binding, 'worker_phase', None) or getattr(binding, 'delegate_compile', False))


def _proof_path(binding):
    explicit = getattr(binding, 'admission_proof', None)
    if explicit is not None:
        return Path(explicit).resolve()
    if _inner(binding):
        raise ValueError('Every child requires an explicit outer-admission metadata proof')
    if getattr(binding, 'out', None) is None:
        raise ValueError('Outer admission requires its own metadata output case')
    return Path(binding.out)/'outer-admission.json'


def fork_gate(case, checkpoint_audit, binding):
    """Original control metadata plus new reference identities; payload reads0."""
    case = Path(case).resolve(); checkpoint_audit = Path(checkpoint_audit).resolve()
    if checkpoint_audit.name != 'result.json' or not checkpoint_audit.is_file():
        raise ValueError('Use the actual original semantic-checkpoint result.json')
    # Inner workers derive source locations only from explicit argv/proof.
    # Global registry and configured result-root admission belong to the outer.
    root = binding.original_source_case.parent.resolve()
    if not _inner(binding) and root != paths.RESULTS_DIR.resolve():
        raise ValueError('Outer source root differs from the configured result directory')
    if case.parent != root or binding.original_source_case.parent != root or binding.continuation_case.parent != root:
        raise ValueError('Preserve separate direct result cases for fork, source and continuation')
    proof = base.read(_proof_path(binding)) if _inner(binding) else None
    if proof is not None and (proof.get('round') != ROUND or proof.get('model') != MODEL
            or proof.get('mode') != MODE or proof.get('schema_version') != 1
            or proof.get('global_registry_checked_by_system_outer') is not True
            or not _same(proof.get('proof_path'), _proof_path(binding))
            or not _same(proof.get('source_case'), case)
            or not _same(proof.get('original_source_case'), binding.original_source_case)
            or not _same(proof.get('continuation_case'), binding.continuation_case)
            or not _same(proof.get('boundary_snapshot'), binding.boundary_snapshot)
            or not _same(proof.get('checkpoint_audit'), checkpoint_audit)):
        raise ValueError('Explicit outer admission is not bound to this child source')
    result, completion = base.read(case/'result.json'), base.read(case/'completion.json')
    if (result.get('round') != fork.ROUND or result.get('mode') != fork.MODE or result.get('model') != MODEL
            or result.get('status') not in ('ok', 'quality_rejected')
            or completion.get('status') != result['status'] or completion.get('required_artifacts_persisted') is not True
            or completion.get('registry_persisted') is not True
            or result.get('admission_passed') is not True
            or type(result.get('attempted_confirm')) is not int or result['attempted_confirm'] != 1
            or type(result.get('completed_confirm')) is not int or result['completed_confirm'] != 1
            or any(type(result.get(key)) is not int or result[key] != 0 for key in fork.ZERO)):
        raise ValueError('One durably closed original C2 budget-fork result is required')
    source = base.read(case/'source.json'); recovery = base.read(case/'recovery.json')
    if (result.get('source') != source or recovery.get('source') != source
            or not _same(source.get('source_case'), binding.original_source_case)
            or not _same(source.get('continuation_case'), binding.continuation_case)
            or not _same(source.get('boundary_snapshot'), binding.boundary_snapshot)
            or not _same(result.get('source_case'), binding.original_source_case)
            or not _same(result.get('continuation_case'), binding.continuation_case)
            or recovery.get('only_session_change') != 'phase:evaluate->confirm'
            or recovery.get('max_iters_retained') != 10 or recovery.get('full_candidate_pool_retained') is not True
            or recovery.get('original_source_changed') is not False
            or recovery.get('independent_model_samples_shell') is not True
            or recovery.get('selected_by_report') is not False or recovery.get('new_generation_or_tune_commands') != 0
            or not _same(recovery.get('new_confirm_cache_directory'), case/'model/samples')):
        raise ValueError('Complete original/fork/boundary provenance is required')
    work = case/'phases/001-confirm'
    argv = base.read(work/'command.json')
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ValueError('The saved original confirmation argv must be a string list')
    old_args = SimpleNamespace(source_case=binding.original_source_case, continuation_case=binding.continuation_case,
        boundary_snapshot=binding.boundary_snapshot, out=case, metadata_template=_one(argv, '--metadata-template'),
        intake_python=Path(argv[argv.index('--python')+1]).absolute(), data_python=Path(argv[argv.index('--data-python')+1]).absolute(),
        ledger=_one(argv, '--ledger'), allowance=_one(argv, '--allowance'))
    if argv[1:] != fork.command(old_args, source, work)[1:]:
        raise ValueError('The saved C2 command differs from the original once-only worker contract')
    current_source = fork.source_gate(old_args)
    if current_source != source:
        raise ValueError('The original closed source or boundary changed after C2')
    bound = fork.branch_bindings(old_args)
    if result.get('branch_bindings') != bound:
        raise ValueError('Fork metadata or graph/calibration/heldout reference identities changed')
    # The accepted original import wrote these files locally. Follow only the
    # fork's declared references, never a later replacement source-file alias.
    for name in ('graph/graph.json', 'graph/weights.npz', 'samples/calibration.npz', 'samples/heldout.npz'):
        original_file = binding.original_source_case/'model'/name
        if not original_file.is_file() or original_file.resolve() != original_file.absolute():
            raise ValueError('Original graph/weight/cache reference was replaced by a foreign alias: '+name)
    expected_claim = root/'bert-budget-stop-c2-claims'/(binding.original_source_case.name+'.json')
    claim_path = Path(proof['claim_path']).resolve() if proof is not None else expected_claim
    if claim_path != expected_claim:
        raise ValueError('Explicit outer claim path differs from the canonical original-source claim')
    claim = base.read(claim_path)
    if (not _same(result.get('exclusive_c2_claim'), claim_path)
            or not _same(claim.get('source_case'), binding.original_source_case)
            or not _same(claim.get('branch'), case) or not _same(claim.get('boundary'), binding.boundary_snapshot)
            or claim.get('purpose') != fork.MODE or type(claim.get('retries')) is not int or claim['retries'] != 0):
        raise ValueError('The unique original-source C2 claim is not bound to this fork')
    fork._closed_process(result.get('process'), successful=True)
    worker = base.read(work/'worker_result.json')
    if worker != result.get('worker'):
        raise ValueError('Fork raw confirmation worker differs from the terminal observation')
    fork.original.check_worker('step', worker)
    step = worker.get('step') or {}
    if (step.get('step') != 'confirm' or step.get('next_step') != 'done'
            or step.get('session_calls') != 3 or worker.get('llm_calls') != 0 or worker.get('report_samples') != 0):
        raise ValueError('Fork must contain only the original completed confirmation step')
    before = copy.deepcopy(base.read(binding.boundary_snapshot)); before['phase'] = 'confirm'
    outcome = fork.accepted_branch(old_args, before)
    from ..planners.transformer_session import load_state, json_safe
    state = load_state(case/'session/SESSION.json'); agent = base.read(case/'session/AGENT.json')
    if json_safe(state.get('confirmation')) != agent.get('confirm'):
        raise ValueError('SESSION and original C2 result disagree')
    selected = (agent.get('confirm') or {}).get('selected')
    if selected is not None and selected not in before['pool']:
        raise ValueError('Original C2 selected a program outside the unchanged full saved pool')
    eligible = outcome.get('frozen') is True
    if (result.get('selection_outcome') != outcome or result.get('outcome') != outcome.get('outcome')
            or result.get('status') != ('ok' if eligible else 'quality_rejected')
            or result.get('frozen') is not eligible or result.get('frozen_accepted') is not eligible
            or worker.get('frozen') is not eligible):
        raise ValueError('Fork C2 terminal acceptance and original freeze readback differ')
    rows = [proof.get('registry_row')] if proof is not None else []
    if proof is None:
        with (root/'registry.jsonl').open() as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            for line in stream:
                if case.name not in line:
                    continue
                row = json.loads(line)
                if row.get('run_id') == case.name:
                    rows.append(row)
    if (len(rows) != 1 or not isinstance(rows[0],dict) or set(rows[0]) != serial.REGISTRY_FIELDS
            or rows[0].get('run_id') != case.name
            or rows[0].get('round') != fork.ROUND or rows[0].get('mode') != fork.MODE
            or rows[0].get('model') != MODEL or not _same(rows[0].get('artifacts_dir'), case)
            or rows[0].get('metrics') != result.get('metrics')):
        raise ValueError('Fork needs exactly one matching original eight-field registry row')
    original_info = binding.original_source_case/'model/model_info.json'
    audited = serial.accepted_checkpoint(SimpleNamespace(checkpoint_out=checkpoint_audit.parent),
        dict(model_info=str(original_info), model=source['model']))
    admission = dict(schema_version=1, round=ROUND, model=MODEL, mode=MODE,
        proof_path=str(_proof_path(binding)), source_case=str(case),
        original_source_case=str(binding.original_source_case), continuation_case=str(binding.continuation_case),
        boundary_snapshot=str(binding.boundary_snapshot), checkpoint_audit=str(checkpoint_audit),
        claim_path=str(claim_path), claim=claim, registry_row=rows[0],
        global_registry_checked_by_system_outer=True, original_source=source, fork_bindings=bound,
        selected_outcome=outcome, snapshot_calls=3, original_actual_session_calls=4)
    if proof is not None and proof != admission:
        raise ValueError('Explicit outer admission metadata differs from the revalidated child source')
    return dict(eligible=eligible, outcome=outcome, source=source, bindings=bound, original_args=old_args,
        semantic_checkpoint=audited, fork_case=case, original_model_info=original_info,
        copied_model_info=case/'model/model_info.json', metadata_template=old_args.metadata_template,
        registry_run_id=rows[0]['run_id'], outer_admission=admission)


def metadata_source(case, checkpoint_audit, binding):
    gate = fork_gate(case, checkpoint_audit, binding)
    if not gate['eligible']:
        raise ValueError('The normally rejected C2 fork has no frozen program for compilation')
    if not _inner(binding):
        target = _proof_path(binding)
        # The delegated outer creates its fresh case before source admission.
        # Read-only preflights before that point do not create output artifacts.
        if target.parent.is_dir():
            if target.parent != Path(binding.out) or target.name != 'outer-admission.json':
                raise ValueError('Outer admission proof must belong to its fresh metadata case')
            if target.exists():
                if base.read(target) != gate['outer_admission']:
                    raise ValueError('Preserve a different prior outer-admission proof')
            else:
                with target.open('x') as stream:
                    json.dump(gate['outer_admission'], stream, indent=2, allow_nan=False)
                    stream.write('\n');stream.flush();os.fsync(stream.fileno())
    case = gate['fork_case']; info_path = gate['copied_model_info']; info = base.read(info_path)
    sample = base.read(case/'model/samples/source.json'); split = info['split']; calibration = case/'model/samples/calibration.npz'
    source, outcome = gate['source'], gate['outcome']
    return dict(source_case=str(case), model_info=str(info_path), program=outcome['program'], session=str(case/'session'),
        original_model=source['model'], checkpoint_audit=str(Path(checkpoint_audit).resolve()),
        original_checkpoint=source['original_checkpoint'], frozen=outcome, session_state=serial.session_state(case),
        selected_source=Path(outcome['program']).read_text(), inputs=info['inputs'], sequence=info['sequence'],
        task=info['task'], metric=info['metric'], task_policy=info['threshold'], split_version=split['version'],
        split=split, sample_source=sample, calibration=dict(path=str(calibration), bytes=calibration.stat().st_size,
            partition='calibration', row_positions=[0,1,2], sample_indices=split['calibration'][:3], stored_rows=1000),
        semantic_checkpoint_accepted=gate['semantic_checkpoint']['checkpoint_semantic_coverage_passed'],
        report_task_acceptance=None, report_validation_metrics=None, report_deferred=True,
        report_outcome_used_for_selection=False, api_feedback_permitted=False,
        fork_provenance=dict(variant=MODE, original_source_case=str(binding.original_source_case),
            original_prepare_result=str(binding.original_source_case/'phases/001-prepare/worker_result.json'),
            original_model_info=str(gate['original_model_info']), copied_fork_model_info=str(info_path),
            original_checkpoint_audit=str(Path(checkpoint_audit).resolve()), original_audit_scope='Original model_info path; no claim that the copied fork path is identical',
            continuation_case=str(binding.continuation_case), boundary_snapshot=str(binding.boundary_snapshot),
            fork_result=str(case/'result.json'), fork_completion=str(case/'completion.json'), fork_registry_run_id=gate['registry_run_id'],
            outer_admission_proof=str(_proof_path(binding)), registry_verification_scope='System outer only; inner revalidates the explicit admitted row and original source metadata without reading a global registry',
            original_metadata_equal=True, graph_and_existing_array_references_equal=True,
            original_graph_weights=str(binding.original_source_case/'model/graph/weights.npz'),
            weight_binding_scope='Same canonical original graph/weights file; no new payload/value/digest verification. Original source immutability remains required.',
            metadata_template=str(gate['metadata_template']), inherited_search_calls=3, excluded_original_calls=[4],
            snapshot_calls=3, completed_search_rounds=3, original_actual_session_calls=source['original_calls'],
            original_fourth_call_charged=True, excluded_iteration_state=source['original_source_state'],
            original_search_and_partial_tune_result=str(binding.continuation_case/'result.json'),
            original_search_and_partial_tune_completion=str(binding.continuation_case/'completion.json'),
            original_search_and_partial_work_discarded=False,
            original_max_iters_retained=10, original_C2_repeated=False, intake_repeated=False,
            bridge_scope='Copied metadata semantic equality plus canonical readonly graph/cache references; original prepare/audit paths retained separately',
            clock_scope='Warm frozen-fork compilation only; original search/continuation/C2 clocks remain separate'))


def _options(binding):
    result = []
    for name in BRIDGE_FIELDS:
        result += ['--'+name.replace('_','-'), str(getattr(binding, name))]
    return result+['--admission-proof',str(_proof_path(binding))]


def namespace(binding):
    """Clone only trusted ordinary functions; imported context managers stay original."""
    values = dict(vars(base), ROUND=ROUND, MODE=MODE, MODULE=MODULE, __file__=__file__)
    for name, function in vars(base).items():
        if isinstance(function, types.FunctionType) and function.__globals__ is vars(base):
            clone = types.FunctionType(function.__code__, values, name, function.__defaults__, function.__closure__)
            clone.__kwdefaults__ = copy.copy(function.__kwdefaults__); values[name] = clone
    values['metadata_source'] = lambda case, audit: metadata_source(case, audit, binding)
    original_metadata = base.compiler_metadata
    def compiler_metadata(args):
        source = values['metadata_source'](args.source_case, args.checkpoint_audit)
        if not _same(args.metadata_template, source['fork_provenance']['metadata_template']):
            raise ValueError('Preserve the original C2 metadata template under the explicit fork bridge')
        return original_metadata(args)
    values['compiler_metadata'] = compiler_metadata
    original_argv = values['_argv']
    values['_argv'] = lambda args, phase, proof, remaining: original_argv(args, phase, proof, remaining)+_options(binding)
    original_namespace = values['compile_namespace']
    def compile_namespace(seconds=base.original.OVERALL_SECONDS):
        compiled = original_namespace(seconds)
        phase_command = compiled['phase_command']
        compiled['phase_command'] = lambda args, phase, work: phase_command(args, phase, work)+_options(binding)
        return compiled
    values['compile_namespace'] = compile_namespace
    previous_save = values['save']
    def save(path, result):
        if Path(path).name == 'result.json' and Path(path).parent == getattr(binding, 'out', None):
            for module, name in ((base,'original-compile-first-source.py'), (fork,'original-budget-confirm-source.py')):
                target = Path(path).parent/name
                if not target.exists(): shutil.copyfile(Path(module.__file__), target)
        return previous_save(path, result)
    values['save'] = save
    return values


def observed_terminal(args, started, *, status, error=None, gate=None):
    if args.out.exists(): raise FileExistsError('Preserve the previous fork compilation observation')
    args.out.mkdir(); recorder = StageRecorder(args.out/'events.jsonl')
    with recorder.span('fork_compile_metadata_observation'):
        shutil.copyfile(Path(__file__), args.out/'driver-source.py')
    result = dict(round=ROUND, model=MODEL, mode=MODE, status=status, error=error,
        outcome='quality_rejected' if status=='quality_rejected' else 'admission_error', source_case=str(args.source_case),
        original_source_case=str(args.original_source_case), admission='Original C2 fork was normally rejected' if gate else error,
        attempted=dict(math=0,compile=0), completed=dict(math=0,compile=0), processes=[],
        math_passed=False, compile_completed=False, compile_accepted=False, report_deferred=True,
        report_task_acceptance=None, accuracy_acceptance=None, security_acceptance=None, CKKS_latency=None,
        compiled_bootstraps=None, predicted_latency_us=None, **dict.fromkeys(base.ZERO,0))
    result['timing'] = summarize_events(read_events(recorder.path))
    result['metrics'] = {key:result[key] for key in ('status','outcome','attempted','completed','math_passed','compile_completed','compile_accepted',*base.ZERO)}
    base.save(args.out/'result.json', result)
    from ..registry import git_rev
    row = dict(run_id=args.out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=git_rev(), metrics=result['metrics'],
               artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    persist_case_completion(args.out, round_label=ROUND, started_ns=started, status=status,
        timing=result['timing'], required_artifact_bytes=artifact_bytes(args.out))
    return result


def parser():
    cli = base.parser()
    for name in BRIDGE_FIELDS: cli.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    cli.add_argument('--admission-proof',type=Path)
    return cli


def main(argv=None):
    cli = parser(); args = cli.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value,Path): setattr(args,name,value.expanduser().absolute() if name=='intake_python' else value.expanduser().resolve())
    if not 1 <= args.remaining_seconds <= base.OVERALL_SECONDS: cli.error('Use the original finite1..12000second compile allowance')
    if _inner(args) and args.admission_proof is None:
        cli.error('Every worker/delegate requires explicit --admission-proof from the admitted system outer')
    values = namespace(args)
    if args.worker_phase:
        if args.work_dir is None: cli.error('Worker requires explicit work directory')
        if args.worker_phase == 'math': result = values['math_worker'](args)
        else:
            if any(getattr(args,name) is None for name in ('math_result','model_info','program')): cli.error('Compiler worker requires actual math and source paths')
            if base.read(args.math_result).get('compiler_inputs') != values['compiler_metadata'](args): raise ValueError('Original compiler parameters differ')
            result = values['compile_namespace'](args.remaining_seconds)['worker'](args)
    elif args.delegate_compile:
        if args.out is None or args.math_result is None: cli.error('Delegate requires fresh output and actual math result')
        if base.read(args.math_result).get('compiler_inputs') != values['compiler_metadata'](args): raise ValueError('Original compiler parameters differ')
        result = values['compile_namespace'](args.remaining_seconds)['execute'](args)
    else:
        if args.out is None or args.compile_out is None: cli.error('Outer requires explicit fresh metadata and compiler cases')
        started = time.perf_counter_ns()
        try: gate = fork_gate(args.source_case,args.checkpoint_audit,args)
        except Exception as exc:
            result = observed_terminal(args, started, status='error', error=type(exc).__name__+': '+str(exc))
        else:
            result = (values['execute'](args) if gate['eligible'] else
                      observed_terminal(args, started, status='quality_rejected', gate=gate))
    print(json.dumps({key:result.get(key) for key in ('status','error','math_passed','compile_completed','compiled_bootstraps')}),flush=True)
    return 0 if result['status'] in ('ok','quality_rejected') else 1


if __name__ == '__main__': raise SystemExit(main())
