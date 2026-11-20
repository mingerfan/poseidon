import copy
import unittest
import numpy as np
from benchmark_math import evaluate
from spatial_channel_normalization import normalize


def model(op, shape, attrs, constants=None):
    constants = constants or {}
    return dict(format='poseidon-model-graph-v2', resource_profile='general-tensors-v1', id='test',
        inputs=[dict(name='x', shape=shape)], constants=constants,
        nodes=[dict(id='node', op=op, inputs=['x', *constants], attrs=attrs, outputs=['result'])],
        outputs=[dict(name='output', value='result')])


class SpatialNormalizationTests(unittest.TestCase):
    def equivalent(self, original, part=1):
        before=copy.deepcopy(original)
        normalized, proof=normalize(original, channels_per_part=part)
        self.assertEqual(original,before)
        self.assertFalse(proof['DSL_generated'])
        shape=original['inputs'][0]['shape']
        for phase in range(3):
            x=(((np.arange(np.prod(shape))*7+phase*3)%17)-8).astype(np.float64).reshape(shape)/8.
            expected=evaluate(original,{'x':x});actual=evaluate(normalized,{'x':x})
            for key in expected:
                np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=1e-12)
        return normalized

    def test_grouped_conv1d_unbatched_stride_dilation_bias(self):
        w=(np.arange(8*2*3,dtype=np.float64).reshape(8,2,3)-20)/64
        self.equivalent(model('conv1d',[4,13],dict(groups=2,stride=[2],padding=[2],dilation=[2]),
            dict(w=w.tolist(),b=[i/16 for i in range(8)])),part=3)

    def test_grouped_conv2d_batched_odd_shape(self):
        w=(np.arange(6*2*3*2,dtype=np.float64).reshape(6,2,3,2)-40)/128
        self.equivalent(model('conv2d',[2,4,5,7],dict(groups=2,stride=[2,1],padding=[1,0],dilation=[1,2]),dict(w=w.tolist())),part=2)

    def test_depthwise_multiplier(self):
        w=np.arange(6*1*3*3,dtype=np.float64).reshape(6,1,3,3)/128
        self.equivalent(model('conv2d',[1,3,5,5],dict(groups=3,stride=[1,1],padding=[1,1],dilation=[1,1]),dict(w=w.tolist())))

    def test_pool1d_divisors_and_partial_windows(self):
        for include in (False,True):
            self.equivalent(model('avg_pool1d',[2,3,9],dict(kernel=[3],stride=[2],padding=[1],count_include_pad=include)))

    def test_pool2d_unequal_axes_padding_and_tails(self):
        for include in (False,True):
            self.equivalent(model('avg_pool2d',[2,4,5,7],dict(kernel=[3,2],stride=[2,2],padding=[1,1],count_include_pad=include)),part=3)

    def test_concat_tree_and_name_collision(self):
        g=model('avg_pool1d',[17,7],dict(kernel=[2],stride=[2],padding=[0],count_include_pad=True))
        g['nodes'][0]['id']='spatial0'
        g['nodes'][0]['outputs']=['spatial1v0'];g['outputs'][0]['value']='spatial1v0'
        result=self.equivalent(g)
        self.assertTrue(all(len(n['inputs'])<=8 for n in result['nodes']))
        self.assertEqual(len({n['id']for n in result['nodes']}),len(result['nodes']))

    def test_expansion_rejected_without_input_mutation(self):
        g=model('avg_pool1d',[4,7],dict(kernel=[2],stride=[2],padding=[0],count_include_pad=True))
        old=copy.deepcopy(g)
        with self.assertRaisesRegex(ValueError,'node budget'):normalize(g,maximum_nodes=2)
        self.assertEqual(g,old)

    def test_single_channel_is_not_partitioned(self):
        g=model('avg_pool1d',[1,7],dict(kernel=[2],stride=[2],padding=[0],count_include_pad=True))
        result,receipt=normalize(g)
        self.assertEqual(g,result);self.assertEqual(receipt['changes'],[])


if __name__=='__main__':unittest.main()
