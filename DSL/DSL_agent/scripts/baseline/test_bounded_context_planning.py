"""Failure-path controls for pre-output representation selection, not live evidence."""
import copy
import unittest
from unittest.mock import patch

import bounded_context_planning as planner
from delivery_component import bound


class ContextPreparationTests(unittest.TestCase):
    def setUp(self):
        self.profile=bound(dict(preparation_options=dict(generation_guidance='explicit-v21'), provider_configuration={'reasoning_effort':'low'}))
        self.constraint={'binding':'fixture'}
        self.model={'id':'arbitrary'}
        self.calls=[]
        self.patchers=[patch.object(planner,'verify_constraints'),patch.object(planner,'select',return_value=self.profile),
                       patch.object(planner,'verify'),patch('application_planning.layout.candidate_options',return_value=[{}]),
                       patch.object(planner,'prepare_plan',side_effect=self.produce),patch('deepseek_provider.public_request',side_effect=copy.deepcopy)]
        for p in self.patchers:p.start();self.addCleanup(p.stop)

    def produce(self, model, constraints, **kw):
        self.calls.append(kw)
        guidance=kw['options']['generation_guidance']
        host={'request_id':'fixture-'+str(guidance)}
        if guidance is not None:host['generation_guidance']={'version':guidance}
        return host,{'status':'prepared','selection':{'score':(1,2)}}

    def test_context_failure_uses_bounded_optional_fallback(self):
        def project(host):
            if 'generation_guidance' in host:raise ValueError('original wire bound')
            return host
        with patch('deepseek_provider.public_request',side_effect=project):
            host,plan,report=planner.prepare(self.model,self.constraint)
        self.assertNotIn('generation_guidance',host);self.assertEqual(len(report['candidates']),2)
        self.assertEqual(report['candidates'][0]['failure'],'original wire bound')

    def test_explicit_guidance_is_never_silently_removed(self):
        with patch('deepseek_provider.public_request',side_effect=ValueError('too large')):
            with self.assertRaises(planner.PreparationRejected)as e:planner.prepare(self.model,self.constraint,options={'generation_guidance':'explicit-v21'})
        self.assertEqual(len(e.exception.report['candidates']),1)

    def test_backend_block_stays_blocked(self):
        with patch.object(planner,'prepare_plan',return_value=({'request_id':'blocked'},{'status':'backend_blocked','selection':{'score':(0,)}})):
            with self.assertRaises(planner.PreparationRejected)as e:planner.prepare(self.model,self.constraint)
        self.assertTrue(all(r['status']=='backend_blocked'for r in e.exception.report['candidates']))

    def test_host_limit_is_enforced_before_public_projection(self):
        with patch('public_weight_context.TRUSTED_BYTES',1),patch('deepseek_provider.public_request')as project:
            with self.assertRaises(planner.PreparationRejected):planner.prepare(self.model,self.constraint)
        project.assert_not_called()

    def test_candidate_and_wall_expansion_rejected(self):
        for n in (0,9,True,1.0):
            with self.assertRaises(ValueError):planner.prepare(self.model,self.constraint,candidate_limit=n)
        with self.assertRaises(ValueError):planner.prepare(self.model,self.constraint,wall_seconds=121)

    def test_single_candidate_does_not_gain_a_retry(self):
        planner.prepare(self.model,self.constraint,candidate_limit=1)
        self.assertEqual(len(self.calls),1)

    def test_model_name_does_not_choose_representation(self):
        a=planner.prepare(self.model,self.constraint)[2]
        b=planner.prepare({'id':'resnet20'},self.constraint)[2]
        self.assertEqual(a['selected'],b['selected'])

    def test_projection_identity_change_is_rejected(self):
        with patch('deepseek_provider.public_request',return_value={'request_id':'changed'}):
            with self.assertRaises(planner.PreparationRejected)as e:planner.prepare(self.model,self.constraint)
        self.assertIn('identity',e.exception.report['candidates'][0]['failure'])


if __name__=='__main__':unittest.main()
