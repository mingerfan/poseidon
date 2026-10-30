"""A1 §6.9: per-sample first violations stay outside Agent feedback."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent import domain_criteria as criteria, transformer_validation as validation
from pt_agent import transformer_tensor
from pt_agent.dsl.parser import parse
from pt_agent.tests._artifacts import RESULTS_DIR, require


SOURCE = '''@fhe.program(slots=1024)
def sm(x: Cipher[1,1,2]):
 a=pack(x,layout="token")
 p=softmax(a,exp_method="cheb",exp_domain=(-1.,1.),exp_deg=7,inv_method="remez",inv_domain=(.3,6.),inv_deg=7)
 return p
'''


def descriptors():
    points = [SimpleNamespace(op=SimpleNamespace(id=0), role=role) for role in ('exp', 'inv')]
    rows = [dict(program_line=4, graph_tensor='p::'+role, role=role, domain=(-1., 1.),
                 domain_source='calibrated' if role == 'exp' else 'analytic')
            for role in ('exp', 'inv')]
    return points, rows


def context(values):
    values = np.asarray(values, dtype=float).reshape(-1, 1, 2)
    info = SimpleNamespace(inputs=[dict(name='x', shape=[1, 1, 2], kind='tokens')],
                           task='regression', threshold={}, input_handling={})
    return SimpleNamespace(prompt_version='r8-v8', model_info=info, weights={}, weight_shapes={},
        val_images=values, val_inputs=dict(x=values), val_exact_logits=np.zeros_like(values),
        val_labels=None, val_split='heldout', act_tensors=['p::exp', 'p::inv'])


class ViolationOrderTests(unittest.TestCase):
    def test_calibrated_then_analytic_is_ordinary_infeasible(self):
        tracker = criteria.ViolationOrder(*descriptors(), 1)
        tracker.observe(0, 'exp', np.array([[2., 0.]]))
        tracker.observe(0, 'inv', np.array([[3.]]))
        self.assertFalse(tracker.analytic_domain_violation)
        chain = tracker.report(engine='tensor', split='heldout')['samples'][0]['chain']
        self.assertEqual([event['role'] for event in chain], ['exp', 'inv'])
        self.assertEqual(chain[0]['obs_max'], 2.)

    def test_analytic_first_stops_and_calibrated_095_does_not_mask_it(self):
        tracker = criteria.ViolationOrder(*descriptors(), 1)
        tracker.observe(0, 'exp', np.array([[.95, -.95]]))
        tracker.observe(0, 'inv', np.array([[1.1]]))
        self.assertTrue(tracker.analytic_domain_violation)
        self.assertEqual(tracker.report(engine='tensor', split='heldout')['samples'][0]
                         ['first_violation']['role'], 'inv')

    def test_samples_are_independent_and_masked_values_do_not_cross_samples(self):
        tracker = criteria.ViolationOrder(*descriptors(), 2)
        tracker.observe(0, 'exp', np.array([[2., 99.], [.5, 99.]]),
                        np.array([[True, False], [True, False]]))
        tracker.observe(0, 'inv', np.array([[2.], [2.]]))
        report = tracker.report(engine='tensor', split='heldout')
        self.assertTrue(report['analytic_domain_violation'])
        self.assertEqual([s['first_violation']['role'] for s in report['samples']], ['exp', 'inv'])
        self.assertEqual(report['samples'][0]['first_violation']['obs_max'], 2.)

    def test_nonfinite_and_batch_offsets_are_recorded_without_retaining_arrays(self):
        tracker = criteria.ViolationOrder(*descriptors(), 22)
        tracker.observe(0, 'exp', np.array([[np.nan, .2], [.2, .3]]), start=20)
        tracker.observe(0, 'inv', np.array([[np.inf], [np.inf]]), start=20)
        report = tracker.report(engine='mock', split='confirm')
        self.assertFalse(report['samples'][0]['observed'])
        self.assertEqual([s['first_violation']['role'] for s in report['samples'][20:]], ['exp', 'inv'])
        self.assertTrue(report['analytic_domain_violation'])
        self.assertTrue(report['samples'][20]['first_violation']['nonfinite'])
        self.assertEqual(report['samples'][21]['first_violation']['max_u'], float('inf'))

    def test_repeated_point_preserves_first_position_and_aggregates_range(self):
        tracker = criteria.ViolationOrder(*descriptors(), 1)
        tracker.observe(0, 'exp', np.array([[2.]]))
        tracker.observe(0, 'inv', np.array([[3.]]))
        tracker.observe(0, 'exp', np.array([[-4.]]))
        chain = tracker.report(engine='mock', split='heldout')['samples'][0]['chain']
        self.assertEqual([event['role'] for event in chain], ['exp', 'inv'])
        self.assertEqual((chain[0]['obs_min'], chain[0]['obs_max'], chain[0]['max_u']), (-4., 2., 4.))

    def test_optional_capture_is_nested_and_does_not_add_return_fields(self):
        with criteria.capture_violations() as outer:
            criteria.record_violations(dict(value=1))
            with criteria.capture_violations() as inner:
                criteria.record_violations(dict(value=2))
            criteria.record_violations(dict(value=3))
        self.assertEqual(outer, [dict(value=1), dict(value=3)])
        self.assertEqual(inner, [dict(value=2)])


class ExecutionObserverTests(unittest.TestCase):
    def test_tensor_mask_keeps_each_sample_and_preserves_public_observations(self):
        source = SOURCE.replace('x: Cipher[1,1,2]', 'x: Cipher[1,1,2], m: Mask[1,2]')
        source = source.replace(' p=softmax(a,', ' am=pack(m,layout="flat")\n p=softmax(a,mask=am,')
        program = parse(source)
        arrays = dict(x=np.array([[[.1, 10.]], [[10., .2]]]), m=np.array([[1., 0.], [0., 1.]]))
        old, new, events = [], [], []
        def observer(name, data, cfg):
            new.append((name, data.copy(), cfg))
        observer.sample_event = lambda op, role, data, mask: events.append(
            (role, np.asarray(data).copy(), None if mask is None else mask.copy()))
        expected = validation.execute(program, {}, arrays, exact=True,
            observer=lambda name, data, cfg: old.append((name, data.copy(), cfg)))
        actual = validation.execute(program, {}, arrays, exact=True, observer=observer)
        np.testing.assert_array_equal(actual, expected)
        for before, after in zip(old, new):
            self.assertEqual(before[0], after[0]); self.assertEqual(before[2], after[2])
            np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(events[0][0], 'exp')
        np.testing.assert_array_equal(events[0][1][:, 0, :], arrays['x'][:, 0, :])
        np.testing.assert_array_equal(events[0][2][:, 0, :], arrays['m'].astype(bool))

    def test_multihead_events_follow_each_engines_actual_operation_order(self):
        source = SOURCE.replace('Cipher[1,1,2]', 'Cipher[1,2,2,2]').replace('layout="token"', 'layout="score"')
        program = parse(source); points = validation.points(program)
        rows = [dict(program_line=point.op.lineno, graph_tensor=point.key,
                     domain=(-1., 1.) if point.role == 'exp' else (.1, 1.),
                     domain_source='calibrated' if point.role == 'exp' else 'analytic')
                for point in points]
        x = np.zeros((1, 2, 2, 2)); x[:, 1, 0, :] = 2.
        records = {}
        for engine in ('tensor', 'mock'):
            tracker = criteria.ViolationOrder(points, rows, 1); events = []
            def observer(name, data, cfg):
                pass
            def sample_event(op, role, data, mask):
                events.append(role); tracker.observe(op, role, data, mask)
            observer.sample_event = sample_event
            validation.execute(program, {}, dict(x=x), engine=engine, exact=True, observer=observer)
            records[engine] = (events, tracker.report(engine=engine, split='heldout'))
        self.assertEqual(records['tensor'][0], ['exp', 'inv'])
        self.assertEqual(records['mock'][0], ['exp', 'inv']*4)
        self.assertEqual(records['tensor'][1]['samples'][0]['first_violation']['role'], 'exp')
        self.assertEqual(records['mock'][1]['samples'][0]['first_violation']['role'], 'inv')

    def test_validate_classifies_each_sample_and_preserves_feedback_field_set(self):
        ctx = context([[2., .1], [.1, .2]])
        program = parse(SOURCE); reports = []
        # A deliberately wrong exponential makes inv exceed its analytical
        # interval. Only the first sample has an earlier exp-domain violation.
        def polynomial(value, attrs, exact=False):
            return np.full_like(value, -1. if attrs['fn'] == 'exp' else 1.)
        with patch.object(transformer_tensor, 'polynomial', side_effect=polynomial):
            with criteria.capture_violations() as captured:
                result = validation.validate(program, ctx, diagnostic_callback=reports.append)
            ctx.prompt_version = 'r8-v7'
            legacy = validation.validate(program, ctx)
        self.assertFalse(result['feasible']); self.assertTrue(result['analytic_domain_violation'])
        self.assertEqual([row['first_violation']['role'] for row in reports[0]['samples']], ['exp', 'inv'])
        self.assertEqual(captured, reports)
        self.assertEqual(set(result)-set(legacy),
                         {'max_u_calibrated', 'max_u_analytic', 'analytic_domain_violation'})
        self.assertNotIn('samples', result); self.assertNotIn('chain', result)
        with self.assertRaisesRegex(RuntimeError, '§6.7'):
            criteria.require_analytic_coverage(result)

    def test_validate_contaminated_analytic_point_does_not_stop(self):
        ctx = context([[2., .1]])
        def polynomial(value, attrs, exact=False):
            return np.full_like(value, -1. if attrs['fn'] == 'exp' else 1.)
        with patch.object(transformer_tensor, 'polynomial', side_effect=polynomial):
            result = validation.validate(parse(SOURCE), ctx)
        self.assertFalse(result['feasible']); self.assertGreater(result['max_u_analytic'], 1)
        self.assertFalse(result['analytic_domain_violation'])
        criteria.require_analytic_coverage(result)


MODEL = RESULTS_DIR/'A1-bert-prepare-20261027T123333-7967/model'
SESSION = RESULTS_DIR/'A1-bert-session-20261028T033706-5ca7'


@require(MODEL/'model_info.json', MODEL/'calibration.json', MODEL/'samples/heldout.npz',
         SESSION/'sources/iteration-6.ptdsl', SESSION/'agent_trace.jsonl')
class RetainedFourteenthResponseTests(unittest.TestCase):
    def test_fourteenth_response_is_divergent_without_new_feedback_fields(self):
        from pt_agent.data.samples import StoredSplits
        from pt_agent.models.info import ModelInfo
        from pt_agent.planners.transformer_session import json_safe
        info = ModelInfo.load(MODEL)
        ctx = validation.context(info, json.loads((MODEL/'calibration.json').read_text()),
                                 samples=StoredSplits(MODEL/'samples', info))
        ctx.prompt_version = 'r8-v8'
        program = parse((SESSION/'sources/iteration-6.ptdsl').read_text())
        with criteria.capture_violations() as reports:
            result = validation.validate(program, ctx)
        self.assertTrue(result['diverged']); self.assertFalse(result['feasible'])
        self.assertFalse(result['analytic_domain_violation'])
        criteria.require_analytic_coverage(result)
        self.assertTrue(all(sample['first_violation'] is not None and
                            sample['first_violation']['domain_source'] == 'calibrated'
                            for sample in reports[0]['samples']))
        recorded = json.loads((SESSION/'agent_trace.jsonl').read_text().splitlines()[5])['fb']['validation']
        expected = dict(recorded, analytic_domain_violation=False)
        outward = {key: value for key, value in result.items() if key not in ('rows', 'feasible', 'error')}
        self.assertEqual(json_safe(outward), expected)


if __name__ == '__main__':
    unittest.main()
