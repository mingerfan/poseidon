"""Independent reductions, shape conventions and original boundary controls."""
import copy
import unittest
import numpy as np
from benchmark_math import evaluate
from test_statistical_branch_preparation import graph,node,output
from singleton_reduction_normalization import normalize

COUNTS=dict(forwards=0,values=0,max_absolute_error=0.)


class SingletonReductionTests(unittest.TestCase):
    def equivalent(self,shape,axes,keep,op='mean'):
        g=graph(shape);a=node(g,op,['x'],axes=axes,keepdims=keep);output(g,a);before=copy.deepcopy(g)
        new,receipt=normalize(g);self.assertEqual(g,before)
        for phase in range(3):
            x=(((np.arange(np.prod(shape))*7+phase)%19)-9).astype(np.float64).reshape(shape)/8.
            ref=evaluate(g,{'x':x})['output0'];actual=evaluate(new,{'x':x})['output0'];COUNTS['forwards']+=2
            self.assertEqual(actual.shape,ref.shape);error=np.abs(actual-ref);COUNTS['values']+=error.size
            COUNTS['max_absolute_error']=max(COUNTS['max_absolute_error'],float(error.max()))
            self.assertTrue(np.all(error<=1e-12+1e-12*np.abs(ref)))
        return new,receipt

    def test_trailing_singleton_sum_uses_complete_axes(self):
        new,trace=self.equivalent([1,3,5,1],[1,2],True,'sum')
        self.assertEqual(new['nodes'][0]['attrs']['axes'],[0,1,2,3]);self.assertEqual(len(trace['changes']),1)

    def test_dropped_axes_restore_retained_singleton_shape(self):
        new,_=self.equivalent([1,3,5,1],[1,2],False)
        self.assertEqual(new['nodes'][-1]['op'],'reshape');self.assertEqual(new['nodes'][-1]['attrs']['shape'],[1,1])

    def test_partial_nonunit_reduction_preserves_original_suffix_route(self):
        new,trace=self.equivalent([1,1,3,5],[3],True)
        self.assertEqual(trace['changes'],[]);self.assertEqual(new['nodes'][0]['attrs']['axes'],[3])

    def test_singleton_only_tensor_and_negative_axes(self):
        self.equivalent([1,1,1],[-2],False,'sum')
        self.equivalent([1,3,1,5],[-1,-3],True)

    def test_already_complete_reduction_and_multiple_named_outputs(self):
        _,trace=self.equivalent([2,3],[0,1],False);self.assertEqual(trace['changes'],[])
        g=graph([1,2,3,1]);a=node(g,'mean',['x'],axes=[1,2],keepdims=False)
        b=node(g,'square',['x']);output(g,a,b);before=copy.deepcopy(g);new,_=normalize(g)
        self.assertEqual(new['outputs'],before['outputs']);self.assertEqual(new['nodes'][-1],before['nodes'][-1])

    def test_limits_reject_without_mutation(self):
        g=graph([1,2,3,1]);a=node(g,'mean',['x'],axes=[1,2],keepdims=False);output(g,a);before=copy.deepcopy(g)
        for limit in [0,True,1025,1]:
            with self.assertRaises(ValueError):normalize(g,maximum_nodes=limit)
        self.assertEqual(g,before)


if __name__=='__main__':unittest.main()
