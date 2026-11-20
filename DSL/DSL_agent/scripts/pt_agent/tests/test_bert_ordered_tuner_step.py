"""Private binding and attribution tests; no real models, data or API calls."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_ordered_tuner_step as entry


def fixture(root):
    proof = dict(schema=entry.SCHEMA, status='ok', pair_accepted=True,
        pair_owner_closed=True, all_pair_producers_closed=True,
        pair_registry_unique_eight_fields=True, sampled_rss_policy_passed=True,
        qualification_rows=40, full_heldout_rows=1000, original_mock_batch_size=10,
        workers=3, report_read_capability=False, cold_chain_measurement=False,
        model=str(root/'model'), inputs=str(root/'inputs'))
    path = root/'qualification.json';path.write_text(json.dumps(proof))
    argv = ['--phase', 'step', '--model', proof['model'], '--inputs', proof['inputs'],
            '--work-dir', str(root/'work'), '--session-dir', str(root/'session')]
    return argv, path, proof


class Counters:
    def __init__(self):
        self.calls = 0
        self.clocks = dict(estimate_s=0., validation_s=0., plan_s=0.)

    def estimate_calls(self):return self.calls
    def timing_snapshot(self):return dict(self.clocks)

    def timed(self, name):
        def wrap(function):
            def run(*args, **kwargs):
                try:return function(*args, **kwargs)
                finally:self.clocks[name] += .25
            return run
        return wrap


def tuner_fixture(counters, *, failure=False):
    observed = []

    def validate(program, ctx, *args, **kwargs):
        observed.append(('original_validate', args, kwargs))
        return dict(engine=kwargs.get('engine', 'tensor'), source='original')

    def exact(*args, **kwargs):
        observed.append(('original_exact', args, kwargs))
        return dict(ok=True)

    def estimate(*args, **kwargs):
        observed.append(('original_estimate', args, kwargs))
        if not kwargs.get('cache', True):
            counters.calls += 1;counters.clocks['estimate_s'] += .5
            counters.clocks['plan_s'] += .1
        return dict(cost=7, boots_pred=12, ops_total=28, max_depth=11, latency_pred_us=70.)

    namespace = dict(validate=validate, exact_check=exact, cached_estimate=estimate)
    exec('def tune(program, ctx, *, budget_s=2400, n=200, fail=False):\n'
         '    cost=cached_estimate(program, cache=False)\n'
         '    check=exact_check(program, ctx, engine="tensor")\n'
         '    tensor=validate(program, ctx, n, engine="tensor", cache=False)\n'
         '    mock=validate(program, ctx, engine="mock", stop_on_divergence=False)\n'
         '    if fail: raise ValueError("fixture failure")\n'
         '    return (budget_s, cost, check, tensor, mock)\n', namespace)
    module = SimpleNamespace(tune=namespace['tune'])

    def parallel(program, ctx, *args, workers, telemetry, **kwargs):
        observed.append(('parallel', args, kwargs, workers))
        telemetry(dict(event='batch_started', start=0, stop=10, pid=123))
        return dict(engine='mock', source='parallel')

    return module, parallel, observed


class OrderedTunerStepTests(unittest.TestCase):
    def test_admission_preserves_fixed_policy_and_paths(self):
        with TemporaryDirectory() as folder:
            argv, path, proof = fixture(Path(folder))
            args, actual = entry.admission(argv, path)
            self.assertEqual(actual, proof)
            self.assertEqual(args.phase, 'step')
            for key, value in [('full_heldout_rows', 40), ('pair_owner_closed', False),
                               ('sampled_rss_policy_passed', False), ('workers', 2),
                               ('report_read_capability', True), ('model', '/different')]:
                changed = dict(proof, **{key:value});path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):entry.admission(argv, path)

    def test_report_and_confirmation_are_not_step_commands(self):
        with TemporaryDirectory() as folder:
            argv, path, _ = fixture(Path(folder))
            for phase in ('report', 'confirm', 'prepare'):
                changed = list(argv);changed[1] = phase
                with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                    entry.admission(changed, path)

    def test_original_code_and_defaults_use_private_globals_and_restore(self):
        with TemporaryDirectory() as folder:
            counter = Counters();records = entry.ToolRecords(folder, counter)
            module, parallel, observed = tuner_fixture(counter)
            original = module.tune; namespace = dict(original.__globals__)
            ctx = SimpleNamespace(val_split='heldout', val_images=range(1000))
            with entry.private_tuner_binding(module, records, parallel, counter.timed) as clone:
                self.assertIs(clone.__code__, original.__code__)
                self.assertEqual(clone.__defaults__, original.__defaults__)
                self.assertEqual(clone.__kwdefaults__, original.__kwdefaults__)
                self.assertIsNot(clone.__globals__, original.__globals__)
                measured = module.tune(object(), ctx)
                self.assertEqual(measured[0], 2400)
                self.assertEqual(measured[3]['source'], 'original')
                self.assertEqual(measured[4]['source'], 'parallel')
            self.assertIs(module.tune, original)
            self.assertEqual(original.__globals__, namespace)
            self.assertEqual(observed[2][1], (200,))
            self.assertEqual(observed[3][3], 3)
            result = records.finish(status='ok')
            self.assertEqual(result['calls'], dict(tune=1, validation_tensor=1,
                validation_mock=1, exact_check=1, cached_estimate=1))
            self.assertEqual(result['original_estimate_calls'], 1)
            self.assertEqual(result['original_timing_delta_s'],
                dict(estimate_s=.5, validation_s=.25, plan_s=.1))
            self.assertTrue(result['timing']['attribution_ready'])
            self.assertFalse(result['qualifies_two_hour_claim'])

    def test_failure_restores_binding_and_keeps_complete_error_spans(self):
        with TemporaryDirectory() as folder:
            counter = Counters();records = entry.ToolRecords(folder, counter)
            module, parallel, _ = tuner_fixture(counter);original = module.tune
            ctx = SimpleNamespace(val_split='heldout', val_images=range(1000))
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                with entry.private_tuner_binding(module, records, parallel, counter.timed):
                    module.tune(object(), ctx, fail=True)
            self.assertIs(module.tune, original)
            result = records.finish(status='error', error='fixture failure')
            self.assertEqual(result['status'], 'error')
            self.assertTrue(result['timing']['attribution_ready'])
            self.assertEqual(result['timing']['spans'][0]['status'], 'error')

    def test_reduced_context_and_confirmation_rejected_before_original_validate(self):
        with TemporaryDirectory() as folder:
            counter = Counters();records = entry.ToolRecords(folder, counter)
            module, parallel, observed = tuner_fixture(counter)
            with entry.private_tuner_binding(module, records, parallel, counter.timed):
                for split, count in [('heldout', 40), ('confirm', 1000), ('report', 1000)]:
                    ctx = SimpleNamespace(val_split=split, val_images=range(count))
                    with self.assertRaises(ValueError):module.tune(object(), ctx)
            self.assertFalse(any(item[0] in ('original_validate','parallel') for item in observed))

    def test_nested_binding_rejected_without_losing_original_owner(self):
        with TemporaryDirectory() as folder:
            counter = Counters();records = entry.ToolRecords(folder, counter)
            module, parallel, _ = tuner_fixture(counter);original = module.tune
            with entry.private_tuner_binding(module, records, parallel, counter.timed):
                owner = module.tune
                with self.assertRaises(ValueError):
                    with entry.private_tuner_binding(module, records, parallel, counter.timed):pass
                self.assertIs(module.tune, owner)
            self.assertIs(module.tune, original)

    def test_counter_reset_is_not_silently_reported_as_zero(self):
        with TemporaryDirectory() as folder:
            counter = Counters();counter.calls=2
            records = entry.ToolRecords(folder, counter);counter.calls=1
            with self.assertRaises(ValueError):records.finish(status='ok')

    def test_cost_observations_record_actual_estimate_values_without_feedback_edits(self):
        with TemporaryDirectory() as folder:
            counter=Counters();records=entry.ToolRecords(folder,counter)
            module,parallel,_=tuner_fixture(counter)
            ctx=SimpleNamespace(val_split='heldout',val_images=range(1000))
            with entry.private_tuner_binding(module,records,parallel,counter.timed):
                actual=module.tune(object(),ctx)
            self.assertEqual(actual[1],dict(cost=7,boots_pred=12,ops_total=28,max_depth=11,latency_pred_us=70.))
            value=records.finish(status='ok')
            cost=[row for row in value['timing']['spans']if row['stage_id']=='tuner_cached_estimate']
            self.assertEqual(cost[0]['metadata']['original_cost_observation'],
                             dict(boots_pred=12,ops_total=28,max_depth=11,latency_pred_us=70.))
            self.assertEqual(entry.scalar_observation({'x':float('nan'),'y':True},('x','y','missing')),
                             dict(x=None,y=True,missing=None))

    def delegate_fixture(self, root, *, exit_code=0, raise_error=False):
        argv, qualification, proof = fixture(root)
        template = root/'template.json';template.write_text('{}')
        counters = Counters();tuner, parallel, _ = tuner_fixture(counters)
        original = tuner.tune
        context_active = []
        imports = []

        @contextmanager
        def context(value):
            context_active.append(True)
            try:yield
            finally:context_active.pop()

        def initialize():
            self.assertTrue(context_active)

        modules = {'pt_agent.experiments.bert_sst2_worker':SimpleNamespace(initialize_numeric=initialize),
                   entry.TUNER:tuner, 'pt_agent.tensor_sim':counters}

        def import_module(name):
            self.assertTrue(context_active)
            imports.append(name)
            return modules[name]

        def original_entry(actual, metadata):
            self.assertEqual(actual, argv)
            self.assertEqual(metadata, template)
            self.assertTrue(context_active)
            self.assertIsNot(tuner.tune, original)
            ctx = SimpleNamespace(val_split='heldout', val_images=range(1000))
            tuner.tune(object(), ctx)
            if raise_error:raise RuntimeError('delegate fixture error')
            return exit_code

        with patch('pt_agent.experiments.r29_local_compile_matrix.native_metadata_commands', context), \
                patch('pt_agent.experiments.bert_sst2_recorded.worker_delegate', original_entry), \
                patch('pt_agent.experiments.ordered_mock_validation.parallel_mock_validate', parallel), \
                patch.object(entry.importlib, 'import_module', import_module):
            if raise_error:
                with self.assertRaisesRegex(RuntimeError, 'delegate fixture error'):
                    entry.delegate(argv, template, qualification)
            else:self.assertEqual(entry.delegate(argv, template, qualification), exit_code)
        self.assertIs(tuner.tune, original)
        self.assertFalse(context_active)
        self.assertEqual(imports, list(modules))
        return json.loads((root/'work/ordered-step-result.json').read_text())

    def test_delegate_preserves_worker_argv_and_metadata_context_before_import(self):
        with TemporaryDirectory() as folder:
            result = self.delegate_fixture(Path(folder))
            self.assertEqual(result['status'], 'ok')
            self.assertTrue(result['timing']['attribution_ready'])
            self.assertEqual(result['calls']['validation_mock'], 1)

    def test_original_worker_failure_remains_error_and_binding_is_restored(self):
        with TemporaryDirectory() as folder:
            result = self.delegate_fixture(Path(folder), exit_code=1)
            self.assertEqual(result['status'], 'error')
            self.assertTrue(result['timing']['attribution_ready'])

    def test_delegate_exception_persists_attribution_and_restores_binding(self):
        with TemporaryDirectory() as folder:
            result = self.delegate_fixture(Path(folder), raise_error=True)
            self.assertEqual(result['status'], 'error')
            self.assertIn('delegate fixture error', result['error'])
            self.assertTrue(result['timing']['attribution_ready'])


if __name__ == '__main__':unittest.main()
