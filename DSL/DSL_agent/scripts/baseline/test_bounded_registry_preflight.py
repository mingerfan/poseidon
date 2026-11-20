"""New grouping semantics and one-call planning-boundary controls."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from benchmark_math import evaluate
from test_statistical_branch_preparation import graph,node,output
from statistical_branch_preparation_v2 import prepare
from bounded_registry_preflight import prepare as preflight

COUNTS=dict(forwards=0,values=0,max_absolute_error=0.,real_registry_calls=0)


class GroupedPoolPreflightTests(unittest.TestCase):
    def equivalent(self,g,group,tile):
        original=copy.deepcopy(g);new,trace=prepare(g,pool_channels_per_part=group,pool_tile_elements=tile)
        self.assertEqual(original,g);self.assertEqual(trace['pool_channels_per_part'],group)
        for phase in range(2):
            xs={s['name']:(((np.arange(np.prod(s['shape']))+phase*3+i)%13)-6).astype(np.float64).reshape(s['shape'])/8.
                for i,s in enumerate(g['inputs'])}
            a=evaluate(g,xs);b=evaluate(new,xs);COUNTS['forwards']+=2
            for name,value in a.items():
                error=np.abs(value-b[name]);COUNTS['values']+=error.size;COUNTS['max_absolute_error']=max(COUNTS['max_absolute_error'],float(error.max()))
                self.assertTrue(np.all(error<=1e-12+1e-12*np.abs(value)))
        return trace

    def test_grouped_batched_1d_pool_with_uneven_channel_tail(self):
        g=graph([2,3,9]);a=node(g,'avg_pool1d',['x'],kernel=[3],stride=[2],padding=[0],count_include_pad=False)
        b=node(g,'mean',[a],axes=[2],keepdims=False);self.equivalent(output(g,a,b),2,8)

    def test_grouped_conv_square_pool_and_centered_variance(self):
        g=graph([1,2,6,8]);g['constants']['w']=(np.arange(4*2*3*3).reshape(4,2,3,3)/512.).tolist()
        a=node(g,'conv2d',['x','w'],groups=1,stride=[1,1],padding=[1,1],dilation=[1,1]);a=node(g,'square',[a])
        a=node(g,'avg_pool2d',[a],kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True)
        b=node(g,'mean',[a],axes=[3],keepdims=True);b=node(g,'subtract',[a,b]);b=node(g,'square',[b])
        b=node(g,'mean',[b],axes=[2,3],keepdims=False);self.equivalent(output(g,a,b),2,8)

    def test_grouped_axis_transform_and_shared_public_outputs(self):
        g=graph([1,5,6,8]);a=node(g,'avg_pool2d',['x'],kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True)
        b=node(g,'permute',[a],dims=[0,2,3,1]);b=node(g,'mean',[b],axes=[1],keepdims=False);b=node(g,'mean',[b],axes=[1],keepdims=False)
        self.equivalent(output(g,a,b),4,16)

    def test_invalid_pool_group_does_not_mutate_graph(self):
        g=graph([1,3,6,8]);a=node(g,'avg_pool2d',['x'],kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True);output(g,a)
        before=copy.deepcopy(g)
        for value in [0,True,17]:
            with self.assertRaises(ValueError):prepare(g,pool_channels_per_part=value)
        self.assertEqual(g,before)

    def test_preflight_uses_one_original_registry_without_full_factory(self):
        g=graph([1,2,4]);a=node(g,'square',['x']);output(g,a)
        import unified_graph_contract as contract
        with patch.object(contract,'registry',return_value={'p0':0.})as registry,patch('application_policy.prepare',side_effect=AssertionError('Full factory forbidden')):
            public,receipt=preflight(g,period=4096,packing_version='capacity-public-tensors-v8')
        self.assertEqual(registry.call_count,1);self.assertEqual(public,{'p0':0.})
        self.assertEqual(receipt['original_registry_calls'],1);self.assertFalse(receipt['backend_admission']);self.assertFalse(receipt['real_CKKS'])

    def test_preflight_propagates_original_guard_and_invalid_options_do_not_call_it(self):
        g=graph([1,2,4]);a=node(g,'square',['x']);output(g,a)
        import unified_graph_contract as contract
        with patch.object(contract,'registry',side_effect=ValueError('Original registry bound'))as registry:
            with self.assertRaisesRegex(ValueError,'Original registry bound'):preflight(g,period=4096,packing_version='capacity-public-tensors-v8')
            for period,version in [(True,'capacity-public-tensors-v8'),(8192,'capacity-public-tensors-v8'),(4096,'unregistered')]:
                with self.assertRaises(ValueError):preflight(g,period=period,packing_version=version)
        self.assertEqual(registry.call_count,1)


if __name__=='__main__':unittest.main()
