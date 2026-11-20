"""Independent arithmetic, shape, boundary and fallback controls."""
import copy
import unittest
import numpy as np
from benchmark_math import evaluate
from test_statistical_branch_preparation import graph,node,output
from dyadic_pool_normalization import normalize

COUNTS=dict(forwards=0,values=0,max_absolute_error=0.)

def pooled(shape,**override):
    g=graph(shape);a=node(g,'avg_pool2d',['x'],**dict(dict(kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True),**override))
    return output(g,a)


class DyadicPoolTests(unittest.TestCase):
    def equivalent(self,g):
        old=copy.deepcopy(g);new,trace=normalize(g);self.assertEqual(g,old)
        for phase in range(3):
            xs={s['name']:(((np.arange(np.prod(s['shape']))*7+phase*3+i)%23)-11).astype(np.float64).reshape(s['shape'])/16.
                for i,s in enumerate(g['inputs'])}
            ref=evaluate(g,xs);actual=evaluate(new,xs);COUNTS['forwards']+=2
            for name,v in ref.items():
                error=np.abs(v-actual[name]);COUNTS['values']+=error.size;COUNTS['max_absolute_error']=max(COUNTS['max_absolute_error'],float(error.max()))
                self.assertTrue(np.all(error<=1e-12+1e-12*np.abs(v)))
        self.assertFalse(trace['backend_or_memory_certified']);return new,trace

    def test_unbatched_columns_and_width_two_boundary(self):
        for width in [2,8]:self.equivalent(pooled([1,4,width]))

    def test_batched_multichannel_even_height_not_power_two(self):
        self.equivalent(pooled([2,3,6,16],count_include_pad=False))

    def test_composed_pools_shared_branch_and_multiple_outputs(self):
        g=pooled([1,2,8,16]);first=g['outputs'][0]['value'];second=node(g,'avg_pool2d',[first],kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True)
        square=node(g,'square',[first]);self.equivalent(output(g,first,second,square))

    def test_unsupported_geometry_is_retained_exactly(self):
        for g in [pooled([1,2,6,8],kernel=[3,3],stride=[1,1]),pooled([1,2,6,8],padding=[1,1]),pooled([1,2,5,8]),pooled([1,2,6,10])]:
            new,trace=self.equivalent(g);self.assertEqual(new,g);self.assertEqual(trace['changes'],[])

    def test_masks_are_public_binary_and_original_constants_survive(self):
        g=pooled([1,2,4,8]);g['constants']['retained_original']=[.125,-.25]
        new,trace=self.equivalent(g);self.assertEqual(new['constants']['retained_original'],g['constants']['retained_original'])
        for key,value in new['constants'].items():
            if key=='retained_original':continue
            if type(value)is float:self.assertEqual(value,.25)
            else:self.assertTrue(set(np.asarray(value).reshape(-1))<={0.,1.})
        self.assertEqual(len(trace['changes'][0]['stages']),2)

    def test_limits_and_model_labels_do_not_change_geometry_selection(self):
        g=pooled([1,2,4,8]);old=copy.deepcopy(g)
        for limit in [0,True,1025,1]:
            with self.assertRaises(ValueError):normalize(g,maximum_nodes=limit)
        self.assertEqual(g,old);_,a=normalize(g);g['id']='any_network_label';_,b=normalize(g)
        self.assertEqual(a['changes'],b['changes'])


if __name__=='__main__':unittest.main()
