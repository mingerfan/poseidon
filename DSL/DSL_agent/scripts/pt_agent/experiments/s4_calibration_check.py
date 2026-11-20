"""Calibration-only BERT regression for the S4 softmax inverse fix."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .. import paths
from .s4_qwen_block import read
from .r10_paper_metrics import write_json, StageRecorder, read_events, summarize_events
from .completion_metrics import persist_case_completion, artifact_bytes
from .s2_heir_serial import record_registry


def main(args):
    from .r29_local_compile_matrix import native_metadata_commands
    started = time.perf_counter_ns()
    args.out.mkdir(exist_ok=False)
    recorder = StageRecorder(args.out / 'events.jsonl')
    result = dict(status='running', error=None, started_ns=started, model='bert-tiny', calibration_samples=1000,
                  heldout_samples=0, confirm_samples=0, report_samples=0, API_calls=0, compiler_attempts=0)
    try:
        with native_metadata_commands(read(args.metadata_template)):
            from .bert_sst2_worker import initialize_numeric
            initialize_numeric()
            import numpy as np
            from ..models.info import ModelInfo
            from ..analyzer import calibrate
            info = ModelInfo.load(args.bert_case / 'model')
            with np.load(args.bert_case / 'model/samples/calibration.npz', allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key]) for key in info.graph.inputs}
            if any(len(value) != 1000 for value in arrays.values()):
                raise ValueError('Retain the original full1000 training calibration inputs')
            with recorder.span('BERT_original1000_calibration_only'):
                measured = calibrate(info.graph, arrays, batch=20)
            original = read(args.bert_case / 'model/calibration.json')
            differences = []
            other_differences = []
            for key in sorted(original):
                old, new = original[key], measured[key]
                row = dict(key=key, original=old, current=new)
                if old != new:
                    differences.append(row)
                    if not key.endswith('::inv'):
                        bad = [field for field in old if isinstance(old[field], (int, float)) and
                               abs(new[field]-old[field]) > 1e-12+1e-12*abs(old[field])]
                        if bad:
                            other_differences.append(dict(row, failed_fields=bad))
            if set(original) != set(measured) or other_differences:
                raise ValueError('Calibration changed fields outside the declared softmax inverse fix')
            for row in measured.values():
                if any(not np.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
                    raise ValueError('Recomputed BERT calibration is nonfinite')
            write_json(args.out / 'calibration.json', measured)
            result.update(status='ok', original_calibration=str(args.bert_case / 'model/calibration.json'),
                recomputed_calibration=str(args.out / 'calibration.json'), differences=differences,
                inverse_definition='sum(exp((scores-shift)*2**(-steps))*mask)/inv_scale using public R0 normalization',
                other_fields_unchanged_with_original_math_tolerance=True,
                existing_BERT_frozen_candidate_reused_for_evaluation=False)
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
    result['timing'] = summarize_events(read_events(recorder.path))
    write_json(args.out / 'result.json', result)
    print(json.dumps(dict(status=result['status'], error=result['error'],
                          changed_points=len(result.get('differences', [])))))
    return int(result['status'] != 'ok')


def close(args):
    result = read(args.out / 'result.json')
    record_registry(args.out, 'S4-SOFTMAX-CALIBRATION-CHECK', 'BERT-tiny', 'original1000-training-calibration-only',
                    result['status'], dict(API_calls=0, compiler_attempts=0))
    persist_case_completion(args.out, round_label='S4-SOFTMAX-CALIBRATION-CHECK', started_ns=result['started_ns'],
        status=result['status'], timing=result['timing'], required_artifact_bytes=artifact_bytes(args.out))
    return int(result['status'] != 'ok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--close', action='store_true')
    for key in ('out', 'bert-case', 'metadata-template'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(close(args) if args.close else main(args))
