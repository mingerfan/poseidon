"""Continue S5 after pre-intake argument parsing failures in the new wrapper."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import time

from . import s5_serial as serial
from . import s5_tuner_off as off
from .r10_paper_metrics import write_json


MODELS = serial.MODEL_NAMES[:3]


def admission():
    rows = []
    for model in MODELS:
        case = serial.case_for(model)
        result = serial.read(case / 'result.json')
        completion = serial.read(case / 'completion.json')
        processes = result.get('processes', [])
        if (result.get('status') != 'error' or completion.get('status') != 'error'
                or result.get('attempted') != {'intake': 1} or result.get('completed')
                or result.get('api_delta') != 0 or len(processes) != 1):
            raise ValueError('Only the three untouched pre-intake S5 entry failures may continue')
        process = processes[0]
        observation = process.get('observation') or {}
        intake = case / 'work/intake'
        if (process['phase'] != 'intake' or process.get('worker_result') is not None
                or observation.get('exit_code') != 2 or observation.get('timed_out')
                or observation.get('remaining_pids') or observation.get('signals_sent')
                or 'Every inner path is explicit argv' not in (intake / 'worker.log').read_text()
                or (intake / 'model_info').exists() or (intake / 'training_probes.npz').exists()):
            raise ValueError('A completed model or numeric attempt must not be rerun')
        rows.append(dict(model=model, source=str(case / 'result.json'),
            status='driver_error', failure_phase='argument_parsing_before_intake',
            original_case_preserved=True, model_imports=0, sample_reads=0, api_calls=0,
            scientific_attempts=0, entry_elapsed_s=completion['complete_case_elapsed_seconds']))
    return rows


def repaired_case(model):
    original = serial.case_for(model)
    return original.with_name(original.name + '-entry-fixed')


def main(args):
    evidence = admission()
    pending = dict(status='waiting_for_original_S5_BERT_closure',
        reason='Mechanical argument abbreviation correction before CNN intake',
        original_entry_failures=evidence, prior_batch=str(args.prior_batch),
        queued_models=list(MODELS), additional_BERT_runs=0,
        scientific_models_started=0, updated_utc=datetime.now(timezone.utc).isoformat())
    for root in (args.paper, args.mirror):
        write_json(root / 'S5_CNN_ENTRY_REPAIR_PROGRESS.json', pending)
    while not (args.prior_batch / 'result.json').is_file():
        if time.time() >= serial.s4.DEADLINE:
            raise TimeoutError('Publication deadline before the untouched CNN entry continuation')
        time.sleep(2)
    prior = serial.read(args.prior_batch / 'result.json')
    bert_rows = [row for row in prior.get('rows', []) if row['model'] == 'bert_tiny']
    if len(bert_rows) != 1 or not (args.wait_case / 'completion.json').is_file():
        raise ValueError('The single original S5 BERT and its report must close before CNN continuation')
    order = dict(resnet20=0, vgg16=1, alexnet=2, bert_tiny=3)
    def publication(selected, state):
        display = copy.copy(selected)
        display.wait_case = serial.read(args.prior_batch / 'progress.json')['source_Qwen']
        visible = dict(state, rows=sorted(state['rows'] + bert_rows, key=lambda row: order[row['model']]),
            reports=dict(prior.get('reports', {}), **state.get('reports', {})),
            original_entry_failures=evidence, prior_batch=str(args.prior_batch),
            scientific_order='BERT then ResNet20/VGG16/AlexNet after pre-intake argument correction',
            ordered_models=list(serial.MODEL_NAMES))
        serial.publication(display, visible)
        for root in (selected.paper, selected.mirror):
            write_json(root / 'S5_CNN_ENTRY_REPAIR_PROGRESS.json', visible)
    namespace = off.clone_module(serial, dict(MODEL_NAMES=MODELS, case_for=repaired_case, publication=publication))
    return namespace['main'](args)


if __name__ == '__main__':
    parser = serial.cli()
    parser.add_argument('--prior-batch', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
