"""Pure DAG identity and numerical controls for independent schedules."""
import copy
import unittest
import numpy as np
from benchmark_math import evaluate
from benchmark_graph import canonical,validate
from test_statistical_branch_preparation import graph,node,output
from topological_metadata_scheduling import normalize,restore

COUNTS=dict(forwards=0,values=0,max_absolute_error=0.)


class TopologicalMetadataTests(unittest.TestCase):
    def equivalent(self,g):
        before=copy.deepcopy(g);new,trace=normalize(g)
        self.assertEqual(g,before);self.assertEqual(restore(new,trace),before)
        for phase in range(2):
            xs={s['name']:(((np.arange(np.prod(s['shape']))+phase*3+i)%17)-8).astype(np.float64).reshape(s['shape'])/8.
                for i,s in enumerate(g['inputs'])}
            a=evaluate(g,xs);b=evaluate(new,xs);COUNTS['forwards']+=2
            for name,ref in a.items():
                np.testing.assert_array_equal(ref,b[name]);COUNTS['values']+=ref.size
        self.assertFalse(trace['native_memory_certified']);return new,trace

    def test_independent_layouts_are_grouped_before_spatial_nodes(self):
        g=graph([1,2,6,8]);parts=[]
        for lo in [0,2]:
            a=node(g,'slice',['x'],axis=2,start=lo,stop=lo+4,step=1)
            a=node(g,'avg_pool2d',[a],kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True);parts.append(a)
        a=node(g,'concat',parts,axis=2);new,_=self.equivalent(output(g,a))
        self.assertEqual([n['op']for n in new['nodes'][:2]],['slice','slice'])

    def test_shared_multi_input_broadcast_and_statistical_branch(self):
        g=graph([2,3,5],[dict(name='z',shape=[1,3,1])]);a=node(g,'square',['x'])
        b=node(g,'mean',[a],axes=[2],keepdims=True);b=node(g,'subtract',[a,b]);b=node(g,'square',[b])
        c=node(g,'add',['x','z']);c=node(g,'mean',[c],axes=[2],keepdims=False)
        self.equivalent(output(g,b,c,a))

    def test_split_producer_duplicate_uses_and_output_aliases(self):
        g=graph([2,8]);g['nodes']=[dict(id='split0',op='split',inputs=['x'],attrs=dict(axis=1,sections=[3,5]),outputs=['left','right'])]
        a=node(g,'square',['right']);b=node(g,'add',['left','left']);a=node(g,'concat',[b,a],axis=1)
        new,_=self.equivalent(output(g,a,'left','left'));self.assertEqual(sum(n['op']=='split'for n in new['nodes']),1)

    def test_changed_payload_or_order_receipt_cannot_restore(self):
        g=graph([2,3]);a=node(g,'square',['x']);b=node(g,'negate',['x']);output(g,a,b)
        new,trace=normalize(g);changed=copy.deepcopy(new);changed['nodes'][0]['op']='negate'
        with self.assertRaises(ValueError):restore(changed,trace)
        bad=copy.deepcopy(trace);bad['original_order'].reverse()
        with self.assertRaises(ValueError):restore(new,bad)

    def test_deterministic_idempotent_and_model_label_independent(self):
        g=graph([2,3]);a=node(g,'square',['x']);b=node(g,'reshape',['x'],shape=[3,2]);c=node(g,'mean',[a],axes=[1],keepdims=False);output(g,c,b)
        first,trace=normalize(g);again,_=normalize(first);self.assertEqual(first,again)
        renamed=copy.deepcopy(g);renamed['id']='a_model_label_does_not_dispatch';second,other=normalize(renamed)
        self.assertEqual(first['nodes'],second['nodes']);self.assertEqual(trace['scheduled_order'],other['scheduled_order'])

    def test_limits_and_logical_memory_metadata(self):
        g=graph([2,3]);a=node(g,'square',['x']);b=node(g,'negate',['x']);output(g,a,b);before=copy.deepcopy(g)
        for value in [0,True,1025,1]:
            with self.assertRaises(ValueError):normalize(g,maximum_nodes=value)
        self.assertEqual(g,before);_,trace=normalize(g)
        for field in ['original_liveness','scheduled_liveness']:
            self.assertFalse(trace[field]['native_ciphertext_or_RSS_bound']);self.assertGreater(trace[field]['peak_logical_elements'],0)


if __name__=='__main__':unittest.main()
