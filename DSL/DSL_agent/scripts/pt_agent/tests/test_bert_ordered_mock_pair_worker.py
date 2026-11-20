"""Small closed-source and numerical-admission fixtures, with no model loads."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.experiments import bert_ordered_mock_pair_worker as entry
from pt_agent.experiments.bert_sst2_worker import strict_math


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    source = root/'closed-source'
    model_dir = source/'model'
    program = source/'session/program.ptdsl'
    frozen = source/'session/FROZEN.json'
    write(frozen, dict(program=str(program), split_version='sst2-native-v1'))
    program.write_text('only a path fixture; never executed or parsed as a program\n')
    (model_dir/'samples').mkdir(parents=True)
    (model_dir/'samples/heldout.npz').write_text('presence-only fixture; never read as an array\n')
    args = SimpleNamespace(source_admission=root/'new-case/source-admission.json',
        model_dir=model_dir, program=program, frozen=frozen, model=root/'original-model',
        inputs=root/'original-inputs', metadata_template=root/'existing-template.json',
        work_dir=root/'new-case/work')
    proof = dict(schema=entry.SCHEMA, status='ok', source_case=str(source),
        source_closed=True, owner_queue_closed=True, all_source_births_closed=True,
        source_registry_unique_eight_fields=True, logical_heldout_rows=40,
        report_read_capability=False)
    proof.update({k: str(getattr(args, k)) for k in
                  ('model_dir', 'program', 'frozen', 'model', 'inputs', 'metadata_template')})
    write(args.source_admission, proof)
    return args, proof


def fake_api(calls, *, fault=None):
    values = np.arange(80, dtype=np.float64).reshape(40, 2)/80
    metric = dict(total=40, images_done=40, feasible=True, agree=40,
                  rows=[dict(domain=(-1., 1.), max_u=.5)])
    report = dict(total=40, samples=[dict(sample_index=i, event_index=i//10) for i in range(40)])

    def measured(program, ctx, *, engine, output_callback, diagnostic_callback, **kwargs):
        calls.append(engine)
        result = copy.deepcopy(metric)
        if fault == 'tensor_summary' and engine == 'tensor':result['feasible'] = False
        if fault == 'short_count':result['total'] = 39
        if fault != 'missing_output':output_callback(values.copy())
        diagnostic_callback(copy.deepcopy(report))
        return result

    def parallel(program, ctx, *, telemetry, **kwargs):
        if fault == 'fallback':
            telemetry(dict(event='sequential_fallback', reason='fixture'))
        else:
            for i in (0, 10, 20):telemetry(dict(event='batch_started', start=i, stop=i+10, pid=100+i))
            telemetry(dict(event='batch_received', start=0, stop=10, kind='ok'))
            telemetry(dict(event='batch_started', start=30, stop=40, pid=130))
            for i in (10, 20, 30):telemetry(dict(event='batch_received', start=i, stop=i+10, kind='ok'))
        def output(value):
            if fault == 'math':value = value+1e-5
            kwargs['output_callback'](value)
        def diagnostic(value):
            if fault == 'diagnostic':value['samples'][10]['event_index'] = 99
            kwargs['diagnostic_callback'](value)
        changed = dict(kwargs, output_callback=output, diagnostic_callback=diagnostic)
        result = measured(program, ctx, **changed)
        if fault == 'metrics':result['rows'][0]['max_u'] = .8
        return result

    return dict(ctx=SimpleNamespace(), program=object(), validate=measured,
                parallel=parallel, strict=strict_math,
                accesses=[dict(split='heldout', logical_rows=40)],
                logical_positions=list(range(1000, 1040)), report_capability=False,
                stored_inputs_only=True)


class PairWorkerTests(unittest.TestCase):
    def run_fixture(self, root, *, fault=None, update=None):
        args, proof = fixture(root)
        calls = []
        api = fake_api(calls, fault=fault)
        if update:api.update(update)
        with patch.object(entry, 'runtime', return_value=api),\
             patch.object(entry, 'command_context', return_value=nullcontext()):
            result = entry.worker(args)
        return result, calls, args

    def test_complete_pair_keeps_scope_and_original_frozen_source(self):
        with TemporaryDirectory() as folder:
            result, calls, args = self.run_fixture(Path(folder))
            self.assertEqual(result['status'], 'ok')
            self.assertTrue(result['pair_accepted'])
            self.assertTrue(result['ordered_execution_observed'])
            self.assertEqual(calls, ['tensor', 'mock', 'mock'])
            self.assertEqual(result['attempted'], dict(tensor=1, original_mock=1, ordered_mock=1))
            self.assertEqual(result['attempted'], result['completed'])
            self.assertFalse(result['qualifies_two_hour_claim'])
            self.assertFalse(result['cold_chain_measurement'])
            self.assertIsNone(result['speedup'])
            self.assertEqual(result['compiler_runs'], 0)
            self.assertEqual(result['API'], 0)
            self.assertEqual(result['report_samples'], 0)
            self.assertFalse((args.frozen.parent/'REPORT_ACCESS.json').exists())
            self.assertEqual(json.loads(args.frozen.read_text()),
                dict(program=str(args.program), split_version='sst2-native-v1'))

    def test_open_or_unverified_source_is_rejected_before_worker_creation(self):
        for key in ('source_closed', 'owner_queue_closed', 'all_source_births_closed',
                    'source_registry_unique_eight_fields'):
            with self.subTest(key=key), TemporaryDirectory() as folder:
                args, proof = fixture(Path(folder));proof[key] = False;write(args.source_admission, proof)
                with patch.object(entry, 'runtime') as runtime, self.assertRaises(ValueError):entry.worker(args)
                runtime.assert_not_called()
                self.assertFalse(args.work_dir.exists())

    def test_full_metrics_and_original_diagnostic_indices_are_both_required(self):
        for fault in ('metrics', 'diagnostic'):
            with self.subTest(fault=fault), TemporaryDirectory() as folder:
                result, _, _ = self.run_fixture(Path(folder), fault=fault)
                self.assertEqual(result['status'], 'error')
                self.assertFalse(result['pair_accepted'])

    def test_unchanged_original_mathematical_tolerance_rejects_output_difference(self):
        with TemporaryDirectory() as folder:
            result, _, _ = self.run_fixture(Path(folder), fault='math')
            self.assertEqual(result['status'], 'error')
            self.assertFalse(result['pair_accepted'])
            self.assertIn('mathematical tolerance', result['error'])

    def test_original_final_tensor_mock_summary_still_required(self):
        with TemporaryDirectory() as folder:
            result, _, _ = self.run_fixture(Path(folder), fault='tensor_summary')
            self.assertEqual(result['status'], 'error')
            self.assertIn('Original final Tensor/Mock', result['error'])

    def test_sequential_fallback_does_not_qualify_as_real_parallel_execution(self):
        with TemporaryDirectory() as folder:
            result, _, _ = self.run_fixture(Path(folder), fault='fallback')
            self.assertEqual(result['status'], 'error')
            self.assertFalse(result['pair_accepted'])

    def test_no_report_capability_or_other_split_is_admitted(self):
        for update in ({'report_capability': True}, {'stored_inputs_only': False},
                       {'accesses': [dict(split='confirm', logical_rows=40)]}):
            with self.subTest(update=update), TemporaryDirectory() as folder:
                result, calls, _ = self.run_fixture(Path(folder), update=update)
                self.assertEqual(result['status'], 'error')
                self.assertEqual(calls, [])

    def test_reduced_row_count_and_missing_outputs_are_rejected(self):
        for fault in ('short_count', 'missing_output'):
            with self.subTest(fault=fault), TemporaryDirectory() as folder:
                result, calls, _ = self.run_fixture(Path(folder), fault=fault)
                self.assertEqual(result['status'], 'error')
                self.assertFalse(result['pair_accepted'])
                self.assertEqual(calls, ['tensor'])


if __name__ == '__main__':
    unittest.main()
