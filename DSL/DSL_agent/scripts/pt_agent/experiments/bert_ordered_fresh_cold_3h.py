"""A separate original three-hour cold BERT trial after execution qualification.

Only new internal tuning workers use the ordered ten-row Mock executor. The
original three-hour clock, source intake, committed candidate boundary, C2,
calibration mathematics, native compilation and persistence remain delegated.
This entry has no automatic launcher or access to independent report results.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
from types import FunctionType

from .. import paths
from . import bert_fresh_cold_3h as original
from . import bert_ordered_fresh_cold_2h as qualified
from .archive_producer_closure import _process
from .autonomous_serial_batch import acquire_lease


MODULE = 'pt_agent.experiments.bert_ordered_fresh_cold_3h'
ROUND = 'BERT-ORDERED-FRESH-COLD-3H'
MODE = 'fresh-original-model-v11-ordered-ten-row-mock-three-hour-c2-compile-no-report'
STEP_MODULE = 'pt_agent.experiments.bert_ordered_tuner_step_3h'
read, save = original.read, original.save


def pair_qualification(args):
    proof = qualified.pair_qualification(args)
    proof.pop('actual_full_1000_memory_and_two_hour_outcome_unmeasured')
    return dict(proof, actual_full_1000_memory_and_three_hour_outcome_unmeasured=True,
                cold_clock_budget_seconds=10800,
                pair_scope='Closed warm40 execution equivalence only; full1000 memory and fresh three-hour completion require actual new measurements')


def command(args, phase, work, remaining, *, base):
    target = args.out/'ordered-execution-admission.json'
    proof = args._ordered_execution_admission
    if target.is_file():
        if read(target) != proof:
            raise ValueError('The admitted execution policy changed')
    else:
        save(target, proof)
        shutil.copyfile(Path(original.__file__), args.out/'original-three-hour-driver.py')
        shutil.copyfile(Path(__file__).with_name('bert_ordered_tuner_step_3h.py'),
                        args.out/'ordered-three-hour-step-entry.py')
        save(args.out/'supervisor.json', dict(process=args._heavy_owner,
            pid=args._heavy_owner['pid'],
            scope='Direct global heavy-lease owner; original fresh three-hour clock begins before source admission'))
    argv = base(args, phase, work, remaining)
    if phase == 'step':
        worker = 'pt_agent.experiments.bert_sst2_recorded'
        if (argv.count(worker) != 1 or argv.count('--step-seconds') != 1
                or argv[argv.index('--step-seconds')+1] != '10800'):
            raise ValueError('Keep the original recorded three-hour step allowance')
        argv[argv.index(worker)] = STEP_MODULE
        argv += ['--ordered-qualification', str(target)]
    return argv


def paper_record(args, result, record, *, base):
    from .a_group_runs import append_record
    delegated = copy.copy(args)
    delegated.paper_runs = args.out/'original-classifier-paper-row.jsonl'
    base(delegated, result, record)
    rows = [json.loads(line) for line in delegated.paper_runs.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError('One original three-hour classifier row is required')
    row = dict(rows[0],
        execution_optimization_qualification=str(args.out/'ordered-execution-admission.json'),
        original_classifier_row=str(delegated.paper_runs))
    append_record(args.paper_runs, row)


def namespace():
    source = vars(original)
    values = dict(source, MODULE=MODULE, ROUND=ROUND, MODE=MODE, __file__=__file__)
    for name, value in source.items():
        if isinstance(value, FunctionType) and value.__globals__ is source:
            clone = FunctionType(value.__code__, values, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = copy.copy(value.__kwdefaults__)
            values[name] = clone
    base_command, base_record = values['command'], values['paper_record']
    base_read, base_execute = values['_read_worker'], values['execute']

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
            if args._heavy_owner is None:
                raise ValueError('Require the actual new cold owner identity')
            try:
                return base_execute(args)
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)

    values.update(source_gate=source_gate, parser=parser, execute=execute,
        command=lambda *args: command(*args, base=base_command),
        paper_record=lambda *args: paper_record(*args, base=base_record),
        _read_worker=lambda *args: qualified.read_worker(*args, base=base_read))
    return values


def main(argv=None):
    return namespace()['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
