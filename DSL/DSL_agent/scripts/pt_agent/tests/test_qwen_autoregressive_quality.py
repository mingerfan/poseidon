"""Small synthetic generation state/protocol tests; no trained model or data."""

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import qwen_autoregressive_quality as own
from pt_agent.experiments import qwen_diagnostic_jobs as jobs
from pt_agent.experiments.r10_paper_metrics import StageRecorder


def state(length=2, step=0):
    return dict(step=step, prefix_token_ids=[2, 3]+[0]*30,
        prefix_length=length, attention_mask=[[1.]*length+[0.]*(32-length)],
        sel=[[0.]*(length-1)+[1.]+[0.]*(32-length)])


class StateCases(unittest.TestCase):
    def test_update_is_immutable_and_moves_only_actual_prefix_state(self):
        original = state()
        saved = copy.deepcopy(original)
        updated = own.advance_state(original, 7, vocabulary_size=151936)
        self.assertEqual(original, saved)
        self.assertEqual(updated['prefix_token_ids'][:5], [2, 3, 7, 0, 0])
        self.assertEqual(updated['prefix_length'], 3)
        self.assertEqual(updated['step'], 1)
        self.assertEqual(updated['attention_mask'][0][:5], [1., 1., 1., 0., 0.])
        self.assertEqual(updated['sel'][0][:5], [0., 0., 1., 0., 0.])
        updated['prefix_token_ids'][0] = 8
        self.assertEqual(original, saved)

    def test_invalid_capacity_token_mask_and_selector_are_rejected(self):
        for token in (-1, 151936, True, 2.5):
            with self.subTest(token=token), self.assertRaises(ValueError):
                own.advance_state(state(), token, vocabulary_size=151936)
        for update in (dict(prefix_length=32), dict(step='0'), dict(attention_mask=[[1.]*32]),
                       dict(sel=[[1.]+[0.]*31]), dict(prefix_token_ids=[-1]+[0]*31)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                own.advance_state(dict(state(), **update), 7, vocabulary_size=151936)

    def test_job_command_uses_fixed_diagnostic_budget_and_no_fresh_reference(self):
        args = SimpleNamespace(job='quality-own-tensor', out=Path('/synthetic/out'),
            linear_variant='compact-rectangular', model_info=Path('/synthetic/model'),
            original_hf_reference=Path('/synthetic/hf.json'), count_result=Path('/synthetic/count.json'))
        record = jobs.job_definition(args)
        self.assertEqual((record['memory_gib'], record['timeout']), (6, 960))
        self.assertEqual(record['command'][record['command'].index('--seconds')+1], '900')
        self.assertIn('pt_agent.experiments.qwen_autoregressive_quality', record['command'])
        self.assertNotIn('pt_agent.experiments.qwen_reference_worker', record['command'])
        from pt_agent.experiments import qwen_server_job as server
        with mock.patch.object(jobs, 'main', return_value=0) as dispatch:
            self.assertEqual(server.main(['--out', '/synthetic/out', '--stage', 'quality-own-tensor']), 0)
        forwarded = dispatch.call_args.args[0]
        self.assertEqual(forwarded[forwarded.index('--job')+1], 'quality-own-tensor')

    def test_completion_accepts_quality_rejection_and_rejects_incomplete_feedback(self):
        record = own.initial_result()
        record.update(status='ok', source_checked=True, own_output_generation_completed=True,
            all_finite=True, approximate_tokens_fed_back=True, training_quality_passed=False,
            tensor_forward_attempts=6, tensor_forward_completed=6, state_update_attempts=3,
            state_update_completed=3, generated_token_count=6, sample_rows=[{}]*6, trajectory_rows=[{}]*3)
        jobs.check_completed(record, 'quality-own-tensor', 'compact-rectangular')
        for key, value in (('state_update_completed', 2), ('tensor_forward_attempts', 7),
                           ('approximate_tokens_fed_back', False), ('fresh_hf_forwards', 1),
                           ('accuracy_acceptance', True), ('candidate_frozen', True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                jobs.check_completed(dict(record, **{key:value}), 'quality-own-tensor', 'compact-rectangular')


class SyntheticGenerationCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        cls.np = np

    def fixture(self, directory, *, divergence=False, fail_second=False, nonfinite=False):
        np = self.np
        work = Path(directory)/'worker'
        work.mkdir()
        args = SimpleNamespace(work_dir=work, reference=Path(directory)/'reference.json')
        initial = state()
        original_second = own.advance_state(initial, 7, vocabulary_size=151936)
        # Poison the saved second input.  It remains a comparison artifact and
        # must never initialize the second execution.
        original_second['prefix_token_ids'][2] = 99
        trajectories = []
        for index in range(3):
            rows = []
            for step, values in enumerate((initial, original_second)):
                name = 'ref-'+str(index)+'-'+str(step)+'.npy'
                reference = np.zeros((1, 1, 151936), dtype=np.float64)
                reference[0, 0, 7 if step == 0 else 8] = 2.
                np.save(Path(directory)/name, reference, allow_pickle=False)
                rows.append(dict(copy.deepcopy(values), logits_file=name))
            trajectories.append(dict(sample_id='train:'+str(index), steps=rows))
        info = SimpleNamespace(graph=SimpleNamespace(weights={}), task='causal_lm',
            threshold={'agreement_pct':99., 'logit_rel_err':.01})
        op = SimpleNamespace(id='tiny', lineno=1, kind='act',
            attrs=dict(fn='identity', method='exact', domain=(-100., 100.)))
        program = SimpleNamespace(ops=[op])
        seen = []

        def load(arguments, result):
            result.update(source_checked=True, threshold=dict(info.threshold),
                decomposition_variant='unit-width-public-square-v2', newton_budget=7)
            return info, program, dict(trajectories=trajectories)

        def arrays(model, values, numpy):
            seen.append(copy.deepcopy(values))
            return dict(ids=np.asarray(values['prefix_token_ids'], dtype=np.float64)[None, :],
                length=np.asarray([values['prefix_length']], dtype=np.float64))

        def execute(program, weights, arrays, *, point_evaluator, step_observer):
            length = int(arrays['length'][0])
            if fail_second and length == 3:
                raise RuntimeError('synthetic second-forward failure')
            data = arrays['ids'][:, length-1:length]
            point_evaluator(op, 'act', data, op.attrs, None)
            logits = np.zeros((1, 1, 151936), dtype=np.float64)
            if length == 2:
                token = 5 if divergence else 7
            else:
                self.assertEqual(int(arrays['ids'][0, 2]), 5 if divergence else 7)
                token = 8
            logits[0, 0, token] = 2.
            if nonfinite:
                logits[0, 0, 0] = np.nan
            step_observer(op, logits)
            return logits

        return args, seen, load, arrays, execute

    def execute_fixture(self, directory, **options):
        args, seen, load, arrays, execute = self.fixture(directory, **options)
        result = own.initial_result()
        with mock.patch.object(own.fixed, '_load', side_effect=load), \
             mock.patch.object(own.fixed, 'arrays_for', side_effect=arrays), \
             mock.patch.object(own, '_run', side_effect=execute), \
             mock.patch.object(own, '_polynomial', side_effect=lambda value, attrs:value.copy()):
            own.run_validation(args, StageRecorder(args.work_dir/'events.jsonl'), result, self.np)
        return args, seen, result

    def test_three_real_loops_feed_actual_output_and_ignore_saved_second_input(self):
        with tempfile.TemporaryDirectory() as directory:
            args, seen, result = self.execute_fixture(directory)
            self.assertEqual([row['prefix_length'] for row in seen], [2, 3]*3)
            self.assertEqual([row['prefix_token_ids'][2] for row in seen[1::2]], [7]*3)
            self.assertTrue(result['training_quality_passed'])
            self.assertFalse(result['original_hf_trajectory_matched'])
            self.assertEqual(result['tensor_forward_completed'], 6)
            self.assertEqual(result['state_update_completed'], 3)
            self.assertEqual([row['generated_tokens'] for row in result['trajectory_rows']], [[7, 8]]*3)
            self.assertEqual(len(result['point_rows']), 6)
            self.assertEqual(len(list(args.work_dir.glob('own-output-*-logits.npy'))), 6)
            jobs.check_completed(result, 'quality-own-tensor', 'compact-rectangular')

    def test_divergent_approximate_token_is_still_fed_back_without_new_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            _, seen, result = self.execute_fixture(directory, divergence=True)
            self.assertEqual([row['prefix_token_ids'][2] for row in seen[1::2]], [5]*3)
            self.assertEqual(result['status'], 'ok')
            self.assertFalse(result['training_quality_passed'])
            self.assertEqual(result['fresh_hf_forwards'], 0)
            self.assertEqual([row['generated_tokens'] for row in result['trajectory_rows']], [[5, 8]]*3)
            jobs.check_completed(result, 'quality-own-tensor', 'compact-rectangular')

    def test_partial_second_forward_preserves_completed_first_and_state_update(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, load, arrays, execute = self.fixture(directory, fail_second=True)
            result = own.initial_result()
            with mock.patch.object(own.fixed, '_load', side_effect=load), \
                 mock.patch.object(own.fixed, 'arrays_for', side_effect=arrays), \
                 mock.patch.object(own, '_run', side_effect=execute), \
                 mock.patch.object(own, '_polynomial', side_effect=lambda value, attrs:value.copy()), \
                 self.assertRaisesRegex(RuntimeError, 'second-forward'):
                own.run_validation(args, StageRecorder(args.work_dir/'events.jsonl'), result, self.np)
            self.assertEqual((result['tensor_forward_attempts'], result['tensor_forward_completed']), (2, 1))
            self.assertEqual((result['state_update_attempts'], result['state_update_completed']), (1, 1))
            self.assertEqual(result['generated_token_count'], 1)
            self.assertTrue(result['approximate_tokens_fed_back'])
            self.assertIsNone(result['training_quality_passed'])
            self.assertFalse(result['own_output_generation_completed'])
            partial = json.loads((args.work_dir/'worker-result.json').read_text())
            self.assertEqual(partial['tensor_forward_attempts'], 2)
            self.assertEqual(partial['tensor_forward_completed'], 1)

    def test_nonfinite_output_stops_before_token_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, load, arrays, execute = self.fixture(directory, nonfinite=True)
            result = own.initial_result()
            with mock.patch.object(own.fixed, '_load', side_effect=load), \
                 mock.patch.object(own.fixed, 'arrays_for', side_effect=arrays), \
                 mock.patch.object(own, '_run', side_effect=execute), \
                 mock.patch.object(own, '_polynomial', side_effect=lambda value, attrs:value.copy()), \
                 self.assertRaisesRegex(ValueError, 'nonfinite'):
                own.run_validation(args, StageRecorder(args.work_dir/'events.jsonl'), result, self.np)
            self.assertEqual((result['tensor_forward_attempts'], result['tensor_forward_completed']), (1, 0))
            self.assertEqual(result['state_update_attempts'], 0)
            self.assertIsNone(result['training_quality_passed'])


if __name__ == '__main__':
    unittest.main()
