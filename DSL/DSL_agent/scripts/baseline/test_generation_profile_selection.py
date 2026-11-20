"""Offline structural/profile controls, never a real-generation certificate."""
import copy
import unittest
from benchmark_suite import Builder
from component_contract import prepare_task
from deepseek_provider import public_request
from delivery_component import bound
from generation_protocol import Config
from generation_profile_selection import select,verify


def model(shape=(2,),branch=False):
    b=Builder([shape]);x=b.node('square',['input0'])
    if branch:x=b.node('add',['input0',x])
    return b.finish(x)


class GenerationProfileTests(unittest.TestCase):
    def test_small_unbranched_model_keeps_high_hint(self):
        g=model();p=select(g);self.assertEqual(p['provider_configuration']['reasoning_effort'],'high');verify(p,g)

    def test_branching_uses_lower_bounded_hint(self):
        g=model(branch=True);p=select(g);self.assertEqual(p['provider_configuration']['reasoning_effort'],'low')
        self.assertEqual(p['preparation_options'],dict(generation_guidance='explicit-v21'))

    def test_many_nodes_are_classified_by_structure(self):
        b=Builder([(2,)]);v='input0'
        for _ in range(8):v=b.node('negate',[v])
        p=select(b.finish(v));self.assertEqual(p['provider_configuration']['reasoning_effort'],'low')

    def test_renaming_model_does_not_change_profile_choice(self):
        a=model(branch=True);b=copy.deepcopy(a);b['id']='resnet20_not_a_profile_key'
        x,y=select(a),select(b)
        self.assertEqual(x['public_features'],y['public_features']);self.assertEqual(x['provider_configuration'],y['provider_configuration'])

    def test_no_token_retry_or_api_expansion(self):
        p=select(model(branch=True));c=Config(**p['provider_configuration'])
        self.assertEqual((c.max_tokens,c.max_calls,c.provider_retries,c.timeout_seconds),(32768,1,0,1200))
        self.assertIsNone(p['actual_reasoning_or_DSL_token_partition']);self.assertFalse(p['completion_guaranteed']);self.assertEqual(p['API_calls'],0)

    def test_expanded_or_invalid_caps_are_rejected(self):
        for n in [0,32769,True,1.0]:
            with self.assertRaises(ValueError):select(model(),maximum_output_tokens=n)
        with self.assertRaises(ValueError):select(model(),api_timeout_seconds=1201)

    def test_modified_config_cannot_reuse_a_frozen_profile(self):
        g=model(branch=True);p=select(g);p.pop('binding');p['provider_configuration']['reasoning_effort']='high';p=bound(p)
        with self.assertRaises(ValueError):verify(p,g)

    def test_existing_guidance_is_actually_present_in_public_request(self):
        g=model(branch=True);p=select(g)
        request=prepare_task(g,dict(p['preparation_options'],compiler_configuration='poseidon-cpu-eva-w45-current-v1'))
        wire=public_request(request)
        self.assertEqual(wire['generation_guidance']['version'],'explicit-v21')
        self.assertIn('bounded_expression_rule',wire['generation_guidance'])
        self.assertFalse(p['DSL_generated'])


if __name__=='__main__':unittest.main()
