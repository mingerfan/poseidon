"""Independent original-reference controls for public graph window tiling."""
import copy
import unittest
import numpy as np
from benchmark_math import evaluate
from spatial_window_normalization import normalize


def graph(shape, kernel, stride, padding=None, include=True):
    rank=len(kernel)
    return dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='pool_fixture',
        inputs=[dict(name='x',shape=shape)],constants={},nodes=[dict(id='pool',op='avg_pool'+str(rank)+'d',inputs=['x'],
        attrs=dict(kernel=kernel,stride=stride,padding=[0]*rank if padding is None else padding,count_include_pad=include),
        outputs=['value'])],outputs=[dict(name='output0',value='value')])


class WindowNormalizationTests(unittest.TestCase):
    def equivalent(self,g,limit):
        before=copy.deepcopy(g);result,receipt=normalize(g,maximum_tile_output_elements=limit)
        self.assertEqual(g,before);self.assertEqual(result['constants'],g['constants'])
        for phase in range(3):
            shape=g['inputs'][0]['shape'];x=(((np.arange(np.prod(shape))*7+phase*3)%17)-8).astype(np.float64).reshape(shape)/8.
            expected=evaluate(g,{'x':x});actual=evaluate(result,{'x':x})
            for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
        self.assertFalse(receipt['DSL_generated'])
        self.assertTrue(all(t['output_elements']<=limit for n in receipt['changes']for t in n['tiles']))
        return result,receipt

    def test_unbatched_pool1d_overlapping_windows(self):
        self.equivalent(graph([2,17],[2],[1],include=False),8)

    def test_batched_pool1d_with_unused_tail(self):
        self.equivalent(graph([2,2,19],[3],[3]),8)

    def test_pool2d_square_disjoint_image(self):
        result,receipt=self.equivalent(graph([1,1,32,32],[2,2],[2,2]),64)
        self.assertEqual(len(receipt['changes'][0]['tiles']),4)

    def test_pool2d_overlap_unequal_axes_and_batches(self):
        self.equivalent(graph([2,3,7,11],[2,3],[1,2]),12)

    def test_pool2d_gaps_between_kernels(self):
        self.equivalent(graph([3,9,15],[2,2],[3,4]),6)

    def test_include_pad_flags_preserved_on_valid_windows(self):
        for include in (False,True):
            _,receipt=self.equivalent(graph([1,2,7,9],[3,2],[2,3],include=include),6)
            self.assertTrue(receipt['changes'])

    def test_padded_operator_is_preserved_without_approximation(self):
        g=graph([1,3,9,11],[3,3],[2,2],padding=[1,1],include=False)
        result,receipt=self.equivalent(g,8)
        self.assertEqual(result,g);self.assertEqual(receipt['unchanged'][0]['reason'],'padding_requires_boundary_adapter')

    def test_model_without_pool_is_unchanged(self):
        g=graph([2,9],[2],[2]);g['nodes'][0].update(op='square',attrs={})
        result,receipt=self.equivalent(g,4)
        self.assertEqual(result,g);self.assertEqual(receipt['changes'],[])

    def test_leading_dimension_budget_does_not_drop_channels(self):
        g=graph([2,5,11],[2],[1]);result,receipt=self.equivalent(g,4)
        self.assertEqual(result,g);self.assertEqual(receipt['unchanged'][0]['reason'],'leading_dimensions_exceed_tile_size')

    def test_inner_axis_only_tiles_keep_original_name(self):
        self.equivalent(graph([1,1,2,33],[2,2],[2,1]),4)

    def test_more_than_eight_tiles_use_bounded_concat_tree(self):
        result,_=self.equivalent(graph([1,257],[2],[2]),4)
        self.assertTrue(all(len(n['inputs'])<=8 for n in result['nodes']))

    def test_names_and_shared_consumers_are_preserved(self):
        g=graph([1,33],[2],[2]);g['nodes'][0]['id']='window0'
        g['nodes'] += [dict(id='consumer',op='square',inputs=['value'],attrs={},outputs=['child']),
            dict(id='join',op='add',inputs=['value','child'],attrs={},outputs=['final'])]
        g['outputs'].append(dict(name='output1',value='final'))
        result,_=self.equivalent(g,4)
        self.assertEqual(len({n['id']for n in result['nodes']}),len(result['nodes']))

    def test_expansion_limit_rejects_without_mutating_input(self):
        g=graph([1,65],[2],[2]);before=copy.deepcopy(g)
        with self.assertRaisesRegex(ValueError,'node budget'):normalize(g,maximum_tile_output_elements=2,maximum_nodes=2)
        self.assertEqual(g,before)

    def test_invalid_or_expanded_limits_are_rejected(self):
        g=graph([1,17],[2],[2])
        for value in (0,257,True,1.0):
            with self.assertRaises(ValueError):normalize(g,maximum_tile_output_elements=value)
        with self.assertRaises(ValueError):normalize(g,maximum_nodes=1025)

    def test_renaming_model_does_not_choose_a_different_partition(self):
        a=graph([1,1,32,32],[2,2],[2,2]);b=copy.deepcopy(a);b['id']='resnet20_not_a_route'
        _,x=normalize(a);_,y=normalize(b)
        self.assertEqual(x['changes'],y['changes'])

    def test_small_operator_is_byte_equivalent(self):
        g=graph([1,5],[2],[2]);result,receipt=normalize(g)
        self.assertEqual(result,g);self.assertEqual(receipt['unchanged'][0]['reason'],'already_within_tile_size')


if __name__=='__main__':unittest.main()
