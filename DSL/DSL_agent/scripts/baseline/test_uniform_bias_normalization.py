import copy
import unittest
import numpy as np
from benchmark_math import evaluate
from uniform_bias_normalization import normalize


def graph(shape,weights,bias,attrs):
    rank=len(weights.shape)-2
    constants=dict(w=weights.tolist())
    if bias is not None:constants['b']=bias
    return dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='bias_fixture',
        inputs=[dict(name='x',shape=shape)],constants=constants,nodes=[dict(id='conv',op='conv'+str(rank)+'d',
            inputs=['x','w']+(['b']if bias is not None else[]),attrs=attrs,outputs=['value'])],outputs=[dict(name='output0',value='value')])


class UniformBiasTests(unittest.TestCase):
    def equivalent(self,g):
        before=copy.deepcopy(g);result,receipt=normalize(g);self.assertEqual(g,before)
        for phase in range(3):
            shape=g['inputs'][0]['shape'];x=(((np.arange(np.prod(shape))*7+phase*3)%17)-8).astype(np.float64).reshape(shape)/8.
            expected=evaluate(g,{'x':x});actual=evaluate(result,{'x':x})
            for name in expected:np.testing.assert_array_equal(actual[name],expected[name])
        self.assertFalse(receipt['DSL_generated']);return result,receipt

    def simple(self,bias=None):
        return graph([2,13],np.arange(2*2*3,dtype=np.float64).reshape(2,2,3)/64.,bias,
            dict(groups=1,stride=[2],padding=[1],dilation=[1]))

    def test_uniform_unbatched_conv1d(self):
        result,receipt=self.equivalent(self.simple([.125,.125]));self.assertEqual(len(receipt['changes']),1)
        self.assertEqual([n['op']for n in result['nodes']],['conv1d','add'])

    def test_batched_grouped_conv2d_padding_stride_dilation(self):
        self.equivalent(graph([2,2,5,7],np.arange(4*1*2*3,dtype=np.float64).reshape(4,1,2,3)/128.,[.1]*4,
            dict(groups=2,stride=[2,1],padding=[1,1],dilation=[1,2])))

    def test_depthwise_nonbinary_bias(self):
        self.equivalent(graph([1,4,4,5],np.arange(4*1*3*3,dtype=np.float64).reshape(4,1,3,3)/128.,[-.1]*4,
            dict(groups=4,stride=[1,1],padding=[1,1],dilation=[1,1])))

    def test_nonuniform_bias_is_preserved(self):
        g=self.simple([.125,.25]);result,receipt=self.equivalent(g);self.assertEqual(result,g);self.assertEqual(receipt['changes'],[])

    def test_signed_zero_distinction_is_preserved(self):
        g=self.simple([0.,-0.]);result,receipt=self.equivalent(g);self.assertEqual(result,g);self.assertEqual(receipt['changes'],[])

    def test_absent_bias_is_preserved(self):
        g=self.simple();result,receipt=self.equivalent(g);self.assertEqual(result,g);self.assertEqual(receipt['changes'],[])

    def test_equal_scalar_is_reused_across_convolutions(self):
        g=self.simple([.125,.125]);g['nodes'].append(dict(id='conv2',op='conv1d',inputs=['value','w','b'],
            attrs=dict(groups=1,stride=[1],padding=[1],dilation=[1]),outputs=['value2']));g['outputs'][0]['value']='value2'
        result,receipt=self.equivalent(g);self.assertEqual(len(receipt['changes']),2)
        self.assertEqual(receipt['changes'][0]['exact_scalar'],receipt['changes'][1]['exact_scalar'])

    def test_limits_preserve_the_original(self):
        g=self.simple([.125,.125]);before=copy.deepcopy(g)
        for n in [0,True,1025]:
            with self.assertRaises(ValueError):normalize(g,maximum_nodes=n)
        with self.assertRaisesRegex(ValueError,'node budget'):normalize(g,maximum_nodes=1)
        self.assertEqual(g,before)

    def test_model_name_does_not_select_a_different_transformation(self):
        a=self.simple([.125,.125]);b=copy.deepcopy(a);b['id']='resnet20_not_a_bias_rule'
        _,x=normalize(a);_,y=normalize(b);self.assertEqual(x['changes'],y['changes'])


if __name__=='__main__':unittest.main()
