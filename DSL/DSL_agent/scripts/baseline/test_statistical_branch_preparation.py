"""Independent numerical and shape controls for generic branch rewrites."""
import copy
import unittest
import numpy as np
from benchmark_math import evaluate
from benchmark_graph import validate
from reduction_branch_normalization import fuse_means,shard_retained_axes
from pool_channel_normalization import normalize as pool_channels
from broadcast_projection_fusion import normalize as projections


COUNTS=dict(forwards=0,compared_values=0,max_absolute_error=0.)

def graph(shape,extra=()):
    return dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='branch_fixture',
        inputs=[dict(name='x',shape=shape),*extra],constants={},nodes=[],outputs=[])

def node(g,op,refs,**attrs):
    name='n'+str(len(g['nodes']));out=name+'v0'
    g['nodes'].append(dict(id=name,op=op,inputs=list(refs),attrs=attrs,outputs=[out]));return out

def output(g,*values):
    g['outputs']=[dict(name='output'+str(i),value=v)for i,v in enumerate(values)];return g


class StatisticalBranchTests(unittest.TestCase):
    def equivalent(self,g,transform):
        before=copy.deepcopy(g);new,receipt=transform(g)
        self.assertEqual(g,before)
        self.assertEqual(validate(new)['output_shapes'],validate(g)['output_shapes'])
        for phase in range(3):
            xs={s['name']:(((np.arange(np.prod(s['shape']))*7+phase*3+i)%29)-14).astype(np.float64).reshape(s['shape'])/16.
                for i,s in enumerate(g['inputs'])}
            expected=evaluate(g,xs);actual=evaluate(new,xs);COUNTS['forwards']+=2
            for name,ref in expected.items():
                error=np.abs(actual[name]-ref);COUNTS['compared_values']+=error.size
                COUNTS['max_absolute_error']=max(COUNTS['max_absolute_error'],float(np.max(error)))
                self.assertTrue(np.all(error<=1e-12+1e-12*np.abs(ref)))
        self.assertFalse(receipt['DSL_generated']);return new,receipt

    def test_consecutive_means_drop_axes(self):
        g=graph([2,3,5]);a=node(g,'mean',['x'],axes=[2],keepdims=False);b=node(g,'mean',[a],axes=[1],keepdims=False)
        new,_=self.equivalent(output(g,b),fuse_means)
        self.assertEqual(sum(n['op']=='mean'for n in new['nodes']),1)

    def test_consecutive_means_mixed_keepdims_and_negative_axis(self):
        for first,second in [(False,True),(True,False),(True,True)]:
            g=graph([2,3,5]);a=node(g,'mean',['x'],axes=[-1],keepdims=first)
            b=node(g,'mean',[a],axes=[1],keepdims=second);self.equivalent(output(g,b),fuse_means)

    def test_reducing_a_previously_reduced_singleton(self):
        g=graph([2,3,5]);a=node(g,'mean',['x'],axes=[2],keepdims=True)
        b=node(g,'mean',[a],axes=[1,2],keepdims=True);self.equivalent(output(g,b),fuse_means)

    def test_parent_mean_is_retained_when_it_is_a_public_output(self):
        g=graph([2,3,5]);a=node(g,'mean',['x'],axes=[2],keepdims=False)
        b=node(g,'mean',[a],axes=[1],keepdims=False);new,_=self.equivalent(output(g,a,b),fuse_means)
        self.assertEqual(sum(n['op']=='mean'for n in new['nodes']),2)

    def test_three_consecutive_means(self):
        g=graph([2,3,4,5]);a='x'
        for axis in [3,2,1]:a=node(g,'mean',[a],axes=[axis],keepdims=False)
        self.equivalent(output(g,a),fuse_means)

    def test_scalar_shape_after_full_reduction(self):
        g=graph([2,3]);a=node(g,'mean',['x'],axes=[0,1],keepdims=False)
        b=node(g,'mean',[a],axes=[0],keepdims=False);self.equivalent(output(g,b),fuse_means)

    def test_shard_smallest_retained_axis(self):
        g=graph([1,3,4,5]);a=node(g,'mean',['x'],axes=[2,3],keepdims=False)
        _,receipt=self.equivalent(output(g,a),shard_retained_axes)
        self.assertEqual(receipt['changes'][0]['retained_axis'],1)
        self.assertEqual(receipt['changes'][0]['partitions'],3)

    def test_shard_sum_negative_axis_and_kept_dimensions(self):
        g=graph([2,3,4]);a=node(g,'sum',['x'],axes=[-1],keepdims=True)
        self.equivalent(output(g,a),shard_retained_axes)

    def test_shard_full_reduction_is_unchanged(self):
        g=graph([2,3]);a=node(g,'mean',['x'],axes=[0,1],keepdims=False)
        new,receipt=self.equivalent(output(g,a),shard_retained_axes)
        self.assertEqual(receipt['changes'],[]);self.assertEqual(new,g)

    def test_shard_concat_fanout_remains_at_most_eight(self):
        g=graph([11,3]);a=node(g,'sum',['x'],axes=[1],keepdims=False)
        new,_=self.equivalent(output(g,a),shard_retained_axes)
        self.assertTrue(all(len(n['inputs'])<=8 for n in new['nodes']))

    def test_pool_split_preserves_dense_convolution_and_public_weights(self):
        g=graph([1,2,5,7]);g['constants']['w']=(np.arange(4*2*3*3).reshape(4,2,3,3)/512.).tolist()
        a=node(g,'conv2d',['x','w'],groups=1,stride=[1,1],padding=[1,1],dilation=[1,1])
        conv=copy.deepcopy(g['nodes'][0]);a=node(g,'avg_pool2d',[a],kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True)
        new,receipt=self.equivalent(output(g,a),pool_channels)
        self.assertEqual([n for n in new['nodes']if n['op']=='conv2d'],[conv]);self.assertEqual(new['constants'],g['constants'])
        self.assertEqual(len(receipt['changes']),1)

    def test_pool_one_dimensional_batched_axis(self):
        g=graph([2,3,9]);a=node(g,'avg_pool1d',['x'],kernel=[3],stride=[2],padding=[1],count_include_pad=False)
        self.equivalent(output(g,a),pool_channels)

    def test_cipher_broadcast_projects_right_aligned_nonunit_axis(self):
        g=graph([2,3,5],[dict(name='z',shape=[3,1])]);a=node(g,'subtract',['x','z'])
        a=node(g,'slice',[a],axis=1,start=1,stop=3,step=1)
        _,receipt=self.equivalent(output(g,a),projections)
        self.assertIn('elementwise_broadcast_projection',[r['kind']for r in receipt['actions']])

    def test_cipher_broadcast_preserves_absent_leading_axis(self):
        g=graph([2,3,5],[dict(name='z',shape=[5])]);a=node(g,'multiply',['x','z'])
        a=node(g,'slice',[a],axis=1,start=0,stop=2,step=1);self.equivalent(output(g,a),projections)

    def test_projected_transform_becomes_a_view_only_for_singleton_moves(self):
        for op,attrs in [('permute',dict(dims=[0,2,1])),('transpose',dict(dim0=1,dim1=2))]:
            g=graph([2,3,4]);a=node(g,op,['x'],**attrs);a=node(g,'slice',[a],axis=2,start=1,stop=2,step=1)
            new,receipt=self.equivalent(output(g,a),projections)
            self.assertNotIn('permute',[n['op']for n in new['nodes']]);self.assertIn('transform_projection_to_view',[r['kind']for r in receipt['actions']])

    def test_projected_nontrivial_transform_is_preserved(self):
        g=graph([2,3,4]);a=node(g,'permute',['x'],dims=[0,2,1]);a=node(g,'slice',[a],axis=2,start=0,stop=2,step=1)
        new,_=self.equivalent(output(g,a),projections);self.assertIn('permute',[n['op']for n in new['nodes']])

    def test_node_budgets_and_invalid_limits_reject_without_input_mutation(self):
        g=graph([2,3,5]);a=node(g,'mean',['x'],axes=[2],keepdims=False);a=node(g,'mean',[a],axes=[1],keepdims=False);output(g,a)
        before=copy.deepcopy(g)
        for transform in [fuse_means,shard_retained_axes,pool_channels,projections]:
            for limit in [0,True,1025]:
                with self.assertRaises(ValueError):transform(g,maximum_nodes=limit)
        with self.assertRaisesRegex(ValueError,'node budget'):shard_retained_axes(g,maximum_nodes=1)
        self.assertEqual(g,before)


if __name__=='__main__':unittest.main()
