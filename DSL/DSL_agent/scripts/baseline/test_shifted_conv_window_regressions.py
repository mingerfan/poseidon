"""Independent mathematical controls for convolution layout preparation."""
import copy
import unittest
import numpy as np

from benchmark_graph import validate,samples
from benchmark_math import evaluate
from shifted_conv_window_normalization import normalize


def model(shape=(1,2,5,6),outc=4,groups=1,kernel=(2,3),padding=(1,1),stride=(1,1),dilation=(1,1),bias=True):
    c=shape[-3];kh,kw=kernel
    w=[[[[((o+2*i+3*y+5*x)%7-3)/32 for x in range(kw)]for y in range(kh)]for i in range(c//groups)]for o in range(outc)]
    constants={'w':w};refs=['input0','w']
    if bias:constants['b']=[(i-2)/64 for i in range(outc)];refs.append('b')
    g=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='synthetic_conv',inputs=[dict(name='input0',shape=list(shape))],
        constants=constants,nodes=[dict(id='conv',op='conv2d',inputs=refs,attrs=dict(groups=groups,padding=list(padding),stride=list(stride),dilation=list(dilation)),outputs=['y'])],outputs=[dict(name='out',value='y')])
    validate(g);return g


class Im2ColTests(unittest.TestCase):
    def compare(self,g):
        before=copy.deepcopy(g);new,receipt=normalize(g);self.assertEqual(g,before);self.assertEqual(new['inputs'],g['inputs']);self.assertEqual(new['outputs'],g['outputs'])
        for sample in samples(g,4):
            a,b=evaluate(g,sample),evaluate(new,sample)
            for key in a:np.testing.assert_allclose(b[key],a[key],atol=1e-12,rtol=1e-12)
        return new,receipt

    def test_batched_rectangular_kernel_and_padding(self):self.compare(model())
    def test_unbatched_input(self):self.compare(model(shape=(2,5,6)))
    def test_depthwise_and_original_bias(self):self.compare(model(outc=2,groups=2))
    def test_grouped_channel_multiplier(self):self.compare(model(shape=(1,4,5,6),outc=6,groups=2))
    def test_asymmetric_stride(self):self.compare(model(stride=(2,1)))
    def test_dilated_kernel(self):self.compare(model(dilation=(2,2)))
    def test_pointwise_no_padding_no_bias(self):self.compare(model(kernel=(1,1),padding=(0,0),bias=False))
    def test_batch_groups_preserved(self):self.compare(model(shape=(2,2,5,6)))
    def test_aliased_named_outputs_and_downstream_use(self):
        g=model();g['nodes'].append(dict(id='res',op='add',inputs=['y','y'],attrs={},outputs=['z']));g['outputs']+=[dict(name='alias',value='y'),dict(name='sum',value='z')];self.compare(g)
    def test_shape_budget_retains_original_operator(self):
        g=model();new,record=normalize(g,maximum_patch_elements=1);self.assertEqual(new,g);self.assertEqual(record['unchanged'][0]['reason'],'patch_tensor_budget')
    def test_node_budget_and_invalid_caps_fail_without_mutation(self):
        g=model();old=copy.deepcopy(g)
        with self.assertRaises(ValueError):normalize(g,maximum_nodes=1)
        for n in [0,65537,True]:
            with self.assertRaises(ValueError):normalize(g,maximum_patch_elements=n)
        self.assertEqual(g,old)
    def test_model_name_and_identifier_collisions_do_not_choose_route(self):
        a=model();a['constants']['imcol0']=0.0;b=copy.deepcopy(a);b['id']='resnet20_is_not_a_dispatch_key'
        x,r=self.compare(a);y,s=self.compare(b);self.assertEqual(r['changes'],s['changes']);self.assertEqual(x['nodes'],y['nodes'])


if __name__=='__main__':unittest.main()
