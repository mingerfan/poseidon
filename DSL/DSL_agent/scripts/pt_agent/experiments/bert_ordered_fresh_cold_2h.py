"""A distinct fresh BERT two-hour experiment after a closed execution pair.

All original model intake, Agent decisions, C2, calibration mathematics and
native compilation use the existing inclusive-clock driver.  The optional
ordered execution entry is supplied only to new internal step processes.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
from types import FunctionType, SimpleNamespace

from .. import paths
from . import bert_fresh_cold_2h as original
from . import bert_ordered_mock_pair as pair
from .archive_producer_closure import _process
from .autonomous_serial_batch import acquire_lease
from .closed_success_raw_trace_archive import producer_births


MODULE = 'pt_agent.experiments.bert_ordered_fresh_cold_2h'
ROUND = 'BERT-ORDERED-FRESH-COLD-2H'
MODE = 'fresh-original-model-v11-ordered-ten-row-mock-c2-compile-no-report'
STEP_MODULE = 'pt_agent.experiments.bert_ordered_tuner_step'
read, save = original.read, original.save


def pair_qualification(args):
    case = args.pair_case
    if case is None or case.parent != paths.RESULTS_DIR.resolve() or case in (args.out, args.checkpoint_out):
        raise ValueError('An independent canonical direct closed execution-pair case is required')
    result, _ = pair.closed_case(case, ('ok',))
    pair.reference.registry_row(case, result)
    if (result.get('round') != pair.ROUND or result.get('mode') != pair.MODE
            or result.get('model') != original.MODEL or result.get('pair_accepted') is not True
            or result.get('attempted_pair') != 1 or result.get('completed_pair') != 1
            or result.get('guard_trigger') is not None or result.get('incoming_signal') is not None
            or result.get('monitor_stopped') is not True
            or (result.get('heavy_lease') or {}).get('released') is not True):
        raise ValueError('Keep a successful fully closed actual execution pair')
    source, worker = result['source'], result['worker']
    pair.check_worker(SimpleNamespace(out=case), worker, source)
    sampling = result.get('resource_sampling') or {}
    peak = sampling.get('max_sampled_subtree_rss_gib')
    if (sampling.get('complete_samples') is not True
            or sampling.get('sampled_rss_policy_passed') is not True
            or type(peak) not in (int, float) or not 0 < peak <= 14
            or type(sampling.get('saved_numeric_births_closed')) is not int
            or sampling['saved_numeric_births_closed'] < 1):
        raise ValueError('The execution pair requires actual complete RSS and producer evidence')
    supervisor = read(case/'supervisor.json')['process']
    current = _process(supervisor['pid'])
    if current is not None and current.get('starttime') == supervisor['starttime']:
        raise ValueError('The actual pair owner must exit before the new cold experiment')
    count = producer_births(case)
    for name in ('model', 'inputs', 'metadata_template'):
        if Path(source[name]).resolve() != getattr(args, name).resolve():
            raise ValueError('Keep the same original provided input and compiler metadata: '+name)
    return dict(schema='bert-ordered-tuner-step-admission-v1', status='ok',
        pair_case=str(case), pair_result=str(case/'result.json'), pair_accepted=True,
        pair_owner_closed=True, all_pair_producers_closed=True, pair_producer_births=count,
        pair_registry_unique_eight_fields=True, sampled_rss_policy_passed=True,
        qualification_rows=40, full_heldout_rows=1000, original_mock_batch_size=10,
        workers=3, report_read_capability=False, cold_chain_measurement=False,
        model=str(args.model), inputs=str(args.inputs), metadata_template=str(args.metadata_template),
        paired_program_supplied_to_new_Agent=False, warm_session_or_candidate_reuse=False,
        actual_full_1000_memory_and_two_hour_outcome_unmeasured=True)


def command(args, phase, work, remaining, *, base):
    proof = args._ordered_execution_admission
    target = args.out/'ordered-execution-admission.json'
    if target.is_file():
        if read(target) != proof:raise ValueError('The admitted execution policy changed')
    else:
        save(target, proof)
        shutil.copyfile(Path(original.__file__), args.out/'original-two-hour-driver.py')
        save(args.out/'supervisor.json', dict(process=args._heavy_owner, pid=args._heavy_owner['pid'],
            scope='Direct global heavy-lease owner; original fresh clock begins before source admission'))
    argv = base(args, phase, work, remaining)
    if phase == 'step':
        old = 'pt_agent.experiments.bert_sst2_recorded'
        if argv.count(old) != 1:raise ValueError('One original recorded step worker is required')
        argv[argv.index(old)] = STEP_MODULE
        argv += ['--ordered-qualification', str(target)]
    return argv


def paper_record(args, result, record, *, base):
    # Preserve the original classifier output, then separately name an actual
    # inner GNU-timeout exit if this distinct attempt encounters it again.
    from .a_group_runs import append_record
    saved_args = copy.copy(args)
    saved_args.paper_runs = args.out/'original-classifier-paper-row.jsonl'
    base(saved_args, result, record)
    rows = [json.loads(line) for line in saved_args.paper_runs.read_text().splitlines() if line.strip()]
    if len(rows) != 1:raise ValueError('One original scientific paper row is required')
    row = rows[0]
    row['execution_optimization_qualification'] = str(args.out/'ordered-execution-admission.json')
    row['original_classifier_row'] = str(saved_args.paper_runs)
    if result['processes']:
        last = result['processes'][-1]
        observation = last.get('observation') or {}
        argv = last.get('argv') or []
        if (row['status'] == 'driver_error' and last.get('phase') == 'step'
                and argv.count('pt_agent.tools.nixrun') == 1 and argv.count('--seconds') == 1
                and observation.get('exit_code') == 124 and observation.get('timed_out') is False
                and observation.get('signal') is None and observation.get('signals_sent') == []
                and observation.get('remaining_pids') == []):
            seconds = int(argv[argv.index('--seconds')+1])
            elapsed = (observation['ended_ns']-observation['started_ns'])/1e9
            if seconds <= elapsed < seconds+15:
                row.update(status='timeout', timeout_scope='inner_search_phase_allocation',
                    inner_allocated_seconds=seconds, inner_observed_seconds=elapsed,
                    complete_7200_seconds_consumed=record['elapsed_seconds'] >= 7200,
                    classification_evidence='Original nixrun GNU timeout exited124 at its actual allocated phase duration; outer timeout false')
    append_record(args.paper_runs, row)


def read_worker(args, phase, work, *, base):
    value = base(args, phase, work)
    if phase == 'step' and value is not None and value.get('status') == 'ok':
        observation = read(work/'ordered-step-result.json')
        if (observation.get('schema') != 'bert-ordered-tuner-step-result-v1'
                or observation.get('status') != 'ok'
                or (observation.get('timing') or {}).get('attribution_ready') is not True
                or observation.get('report_samples') != 0
                or observation.get('confirmation_execution_changed') is not False
                or observation.get('cold_chain_measurement') is not False
                or observation.get('qualifies_two_hour_claim') is not False):
            raise ValueError('A successful new step must also persist complete original-tool attribution')
        value = dict(value, ordered_tool_observation=observation,
                     ordered_tool_observation_source=str(work/'ordered-step-result.json'))
    return value


def namespace():
    """Keep the original driver code in a private dictionary for the new case."""
    source = vars(original)
    values = dict(source, MODULE=MODULE, ROUND=ROUND, MODE=MODE, __file__=__file__)
    for name, value in source.items():
        if isinstance(value, FunctionType) and value.__globals__ is source:
            clone = FunctionType(value.__code__, values, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = copy.copy(value.__kwdefaults__)
            values[name] = clone
    base_command, base_record = values['command'], values['paper_record']
    base_read = values['_read_worker']
    base_execute = values['execute']

    def source_gate(args):
        source = original.source_gate(args)
        proof = pair_qualification(args)
        args._ordered_execution_admission = proof
        return dict(source, execution_optimization_qualification=proof,
                    original_search_and_confirmation_policy_preserved=True)

    def parser():
        cli = original.parser()
        cli.add_argument('--pair-case', type=Path)
        return cli

    def execute(args):
        with (paths.RESULTS_DIR/'autonomous-heavy-experiments.lock').open('a') as lease:
            acquire_lease(lease, 60)
            if original.serial.foreign_heavy_processes():
                raise RuntimeError('Another numerical or compiler experiment remains active')
            args._heavy_owner = _process(os.getpid())
            if args._heavy_owner is None:raise ValueError('Require the actual new cold owner identity')
            try:return base_execute(args)
            finally:fcntl.flock(lease, fcntl.LOCK_UN)

    values.update(source_gate=source_gate, parser=parser, execute=execute,
        command=lambda *args: command(*args, base=base_command),
        paper_record=lambda *args: paper_record(*args, base=base_record),
        _read_worker=lambda *args: read_worker(*args, base=base_read))
    return values


def main(argv=None):
    return namespace()['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
