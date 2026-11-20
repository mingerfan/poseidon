"""A new BERT step entry with ordered Mock execution and tool attribution.

The outer must first qualify the separate forty-row execution pair.  Only this
new Python worker binds a private copy of the original tuner function.  Its
search, numerical acceptance, Tensor checks and confirmation remain delegated.
"""
import argparse
from contextlib import contextmanager
import copy
import importlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shutil
import sys
import time
from types import FunctionType

from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json


SCHEMA = 'bert-ordered-tuner-step-admission-v1'
TUNER = 'pt_agent.planners.transformer_tuner'


def scalar_observation(value, names):
    result = {}
    for name in names:
        item = value.get(name)
        if type(item) is bool:result[name] = item
        elif isinstance(item, Integral):result[name] = int(item)
        elif isinstance(item, Real) and math.isfinite(item):result[name] = float(item)
        else:result[name] = None
    return result


def admission(argv, path):
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument('--phase', required=True, choices=('step',))
    for name in ('model', 'inputs', 'work-dir', 'session-dir'):
        cli.add_argument('--'+name, type=Path, required=True)
    args, _ = cli.parse_known_args(argv)
    proof = json.loads(Path(path).read_text())
    required = ('pair_accepted', 'pair_owner_closed', 'all_pair_producers_closed',
                'pair_registry_unique_eight_fields', 'sampled_rss_policy_passed')
    if (proof.get('schema') != SCHEMA or proof.get('status') != 'ok'
            or any(proof.get(key) is not True for key in required)
            or proof.get('qualification_rows') != 40
            or proof.get('full_heldout_rows') != 1000
            or proof.get('original_mock_batch_size') != 10
            or proof.get('workers') != 3
            or proof.get('report_read_capability') is not False
            or proof.get('cold_chain_measurement') is not False):
        raise ValueError('Require the closed real pair and unchanged full-validation policy')
    for name in ('model', 'inputs'):
        if Path(proof.get(name, '')).resolve() != getattr(args, name).resolve():
            raise ValueError('Keep the qualified original input path: '+name)
    return args, proof


class ToolRecords:
    def __init__(self, directory, counters):
        self.directory = Path(directory)
        self.recorder = StageRecorder(self.directory/'ordered-tool-events.jsonl')
        self.counters = counters
        self.started_ns = time.perf_counter_ns()
        self.initial_calls = counters.estimate_calls()
        self.initial_timing = counters.timing_snapshot()
        self.calls = dict(tune=0, validation_tensor=0, validation_mock=0,
                          exact_check=0, cached_estimate=0)
        self.batch_events = {}

    def batch_event(self, call, value):
        event = value.get('event', 'unknown')
        self.batch_events[event] = self.batch_events.get(event, 0)+1
        row = dict(value, validation_call=call, observed_ns=time.perf_counter_ns(),
                   scope='Owned child execution overlaps other batches; do not add to parent wall time')
        with (self.directory/'ordered-batch-events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    def finish(self, *, status, error=None):
        final_calls = self.counters.estimate_calls()
        final_timing = self.counters.timing_snapshot()
        if type(final_calls) is not int or final_calls < self.initial_calls:
            raise ValueError('The original monotonic estimate counter changed')
        delta = {key: final_timing[key]-value for key, value in self.initial_timing.items()}
        if any(not math.isfinite(value) or value < 0 for value in delta.values()):
            raise ValueError('Keep the original cumulative tool clocks without resets')
        value = dict(schema='bert-ordered-tuner-step-result-v1', status=status, error=error,
            started_ns=self.started_ns, ended_ns=time.perf_counter_ns(),
            calls=self.calls, ordered_batch_events=self.batch_events,
            original_estimate_calls=final_calls-self.initial_calls,
            original_counter_scope='This parent worker only; child execute work has separate overlapping batch records',
            original_timing_delta_s=delta,
            original_timing_scope='Original cumulative counters; plan_s is nested inside estimate_s; not an exclusive partition',
            timing=summarize_events(read_events(self.recorder.path)),
            confirmation_execution_changed=False, report_samples=0, compiler_feedback_to_Agent=False,
            algorithm='Original transformer tune code, defaults, arguments, ladders and acceptance; full heldout1000 and original10-row Mock batches',
            full_model_memory_qualified_by_pair=False,
            cold_chain_measurement=False, qualifies_two_hour_claim=False)
        write_json(self.directory/'ordered-step-result.json', value)
        return value


def _clone(function, namespace):
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    clone.__annotations__ = dict(function.__annotations__)
    clone.__qualname__ = function.__qualname__
    clone.__module__ = function.__module__
    clone.__doc__ = function.__doc__
    return clone


@contextmanager
def private_tuner_binding(module, records, parallel, timed):
    """Bind one original-code clone in this isolated step process, then restore."""
    original = module.tune
    if getattr(original, '_ordered_step_adapter', False):
        raise ValueError('Only one ordered execution adapter may own a step')
    namespace = dict(original.__globals__)
    original_validate = namespace['validate']
    original_exact = namespace['exact_check']
    original_estimate = namespace['cached_estimate']
    timed_parallel = timed('validation_s')(parallel)

    def validation(program, ctx, *args, **kwargs):
        engine = kwargs.get('engine', 'tensor')
        if engine not in ('tensor', 'mock'):
            raise ValueError('Keep the original tuner validation engines')
        if ctx.val_split != 'heldout' or len(ctx.val_images) != 1000:
            raise ValueError('The new tuner entry retains the complete internal heldout1000 context')
        key = 'validation_'+engine
        records.calls[key] += 1
        call = records.calls['validation_tensor']+records.calls['validation_mock']
        n = args[0] if args else kwargs.get('n')
        with records.recorder.span('tuner_validation_'+engine, validation_call=call,
                requested_n=n, full_context_rows=1000, original_mock_batch_size=10) as event:
            if engine == 'tensor':
                result = original_validate(program, ctx, *args, **kwargs)
            else:
                result = timed_parallel(program, ctx, *args, workers=3,
                    telemetry=lambda value: records.batch_event(call, value), **kwargs)
            event['metadata']['original_validation_observation'] = scalar_observation(result,
                ('feasible','agree','total','images_done','max_u','max_u_calibrated','max_u_analytic','n_diverged'))
            return result

    def exact(*args, **kwargs):
        records.calls['exact_check'] += 1
        with records.recorder.span('tuner_exact_check'):
            return original_exact(*args, **kwargs)

    def estimate(*args, **kwargs):
        records.calls['cached_estimate'] += 1
        before = records.counters.estimate_calls()
        with records.recorder.span('tuner_cached_estimate') as event:
            try:
                result = original_estimate(*args, **kwargs)
                event['metadata']['original_cost_observation'] = scalar_observation(result,
                    ('boots_pred','ops_total','max_depth','latency_pred_us'))
                event['metadata']['cost_scope'] = 'Original tuner estimate result; native compiled counts are measured later'
                return result
            finally:
                event['metadata']['actual_estimate_calls'] = records.counters.estimate_calls()-before

    namespace.update(validate=validation, exact_check=exact, cached_estimate=estimate)
    clone = _clone(original, namespace)

    def entry(*args, **kwargs):
        records.calls['tune'] += 1
        with records.recorder.span('original_tune_algorithm'):
            return clone(*args, **kwargs)

    entry._ordered_step_adapter = True
    entry._original_code_clone = clone
    module.tune = entry
    try:
        yield clone
    finally:
        module.tune = original


def delegate(argv, metadata_template, qualification):
    args, proof = admission(argv, qualification)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir/'ordered-step-result.json').exists():
        raise ValueError('Preserve the previous ordered step result')
    from . import bert_sst2_recorded as recorded
    from .r29_local_compile_matrix import native_metadata_commands
    template = json.loads(Path(metadata_template).read_text())
    records = None
    code, error = None, None
    write_json(args.work_dir/'ordered-execution-admission.json', proof)
    shutil.copyfile(Path(__file__), args.work_dir/'ordered-step-entry.py')
    try:
        with native_metadata_commands(template):
            worker = importlib.import_module(recorded.WORKER)
            worker.initialize_numeric()
            tuner = importlib.import_module(TUNER)
            counters = importlib.import_module('pt_agent.tensor_sim')
            from .ordered_mock_validation import parallel_mock_validate
            records = ToolRecords(args.work_dir, counters)
            with records.recorder.span('ordered_step_delegate'):
                with private_tuner_binding(tuner, records, parallel_mock_validate, counters.timed):
                    code = recorded.worker_delegate(list(argv), metadata_template)
            return code
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
        raise
    finally:
        if records is not None:
            records.finish(status='ok' if code == 0 and error is None else 'error', error=error)


def main(argv=None):
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument('--delegate-worker', action='store_true', required=True)
    cli.add_argument('--metadata-template', type=Path, required=True)
    cli.add_argument('--ordered-qualification', type=Path, required=True)
    args, remaining = cli.parse_known_args(sys.argv[1:] if argv is None else argv)
    return delegate(remaining, args.metadata_template.resolve(), args.ordered_qualification.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
