"""Algebraic and boundary controls against the independent original evaluator."""
import copy
import unittest
import numpy as np
from benchmark_math import evaluate
from projection_fusion import normalize


def graph():
    return dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='projection_fixture',
        inputs=[dict(name='x',shape=[2,3]),dict(name='y',shape=[2,5])],constants={},
        nodes=[dict(id='cat',op='concat',inputs=['x','y'],attrs=dict(axis=1),outputs=['joined'])],
        outputs=[dict(name='output0',value='selected')])

def node(g,op,refs,attrs,output):
    g['nodes'].append(dict(id='n'+str(len(g['nodes'])),op=op,inputs=refs,attrs=attrs,outputs=[output]))

def select(g,source='joined',start=3,stop=8,step=1):
    node(g,'slice',[source],dict(axis=1,start=start,stop=stop,step=step),'selected');return g


class ProjectionFusionTests(unittest.TestCase):
    def equivalent(self,g):
        before=copy.deepcopy(g);result,receipt=normalize(g)
        self.assertEqual(g,before);self.assertEqual(result['constants'],g['constants'])
        for phase in range(3):
            inputs={spec['name']:(((np.arange(np.prod(spec['shape']))*7+phase*3+i)%17)-8).astype(np.float64).reshape(spec['shape'])/8.
                    for i,spec in enumerate(g['inputs'])}
            expected=evaluate(g,inputs);actual=evaluate(result,inputs)
            for name in expected:np.testing.assert_array_equal(actual[name],expected[name])
        self.assertFalse(receipt['DSL_generated']);return result,receipt

    def test_exact_concat_operand_becomes_a_view(self):
        result,receipt=self.equivalent(select(graph()))
        self.assertEqual(result['outputs'][0]['value'],result['nodes'][0]['outputs'][0])
        self.assertEqual(result['nodes'][0]['op'],'reshape')

    def test_projection_crosses_a_concat_boundary(self):
        _,receipt=self.equivalent(select(graph(),start=1,stop=6))
        self.assertIn('concat_projection',[r['kind']for r in receipt['actions']])

    def test_square_commutes_with_static_projection(self):
        g=graph();node(g,'square',['joined'],{},'squared');self.equivalent(select(g,'squared'))

    def test_negation_and_power_preserve_each_selected_cell(self):
        for op,attrs in [('negate',{}),('power',dict(exponent=4))]:
            g=graph();node(g,op,['joined'],attrs,'unary');self.equivalent(select(g,'unary',1,7))

    def test_matching_binary_cipher_shapes(self):
        for op in ['add','subtract','multiply']:
            g=graph();node(g,'square',['joined'],{},'right');node(g,op,['joined','right'],{},'binary')
            _,receipt=self.equivalent(select(g,'binary',2,6))
            self.assertIn('elementwise_binary_projection',[r['kind']for r in receipt['actions']])

    def test_public_scalar_is_preserved_exactly(self):
        g=graph();g['constants']['factor']=.25;node(g,'multiply',['joined','factor'],{},'scaled')
        self.equivalent(select(g,'scaled'))

    def test_public_array_broadcast_uses_original_operator(self):
        g=graph();g['constants']['vector']=[.25];node(g,'multiply',['joined','vector'],{},'scaled')
        _,receipt=self.equivalent(select(g,'scaled'))
        self.assertIn('materialized_slice',[r['kind']for r in receipt['actions']])

    def test_cipher_broadcast_uses_original_operator(self):
        g=graph();g['inputs'].append(dict(name='z',shape=[1,8]));node(g,'add',['joined','z'],{},'broadcast')
        self.equivalent(select(g,'broadcast',0,5))

    def test_negative_bounds_are_normalized_statically(self):
        self.equivalent(select(graph(),start=-5,stop=-1))

    def test_nonunit_steps_are_kept_intact(self):
        for start,stop,step in [(0,8,2),(7,0,-2)]:self.equivalent(select(graph(),start=start,stop=stop,step=step))

    def test_multioutput_producer_is_copied_once(self):
        g=graph();g['inputs']=[dict(name='x',shape=[2,8])];g['nodes']=[dict(id='splitter',op='split',inputs=['x'],
            attrs=dict(axis=1,sections=[3,5]),outputs=['left','right'])]
        node(g,'concat',['left','right'],dict(axis=1),'joined');select(g)
        g['outputs'].append(dict(name='output1',value='left'))
        result,_=self.equivalent(g);self.assertEqual(sum(n['op']=='split'for n in result['nodes']),1)

    def test_shared_views_and_name_collisions(self):
        g=graph();g['nodes'][0]['id']='projection0';node(g,'square',['joined'],{},'square')
        select(g,'square');g['outputs'].append(dict(name='output1',value='joined'))
        result,_=self.equivalent(g);self.assertEqual(len({n['id']for n in result['nodes']}),len(result['nodes']))

    def test_conv_is_not_replaced_by_a_model_specific_route(self):
        g=graph();g['inputs']=[dict(name='x',shape=[1,2,5,5])];g['constants']['w']=(np.arange(4*2*3*3).reshape(4,2,3,3)/128.).tolist()
        g['nodes']=[];node(g,'conv2d',['x','w'],dict(groups=1,stride=[1,1],padding=[1,1],dilation=[1,1]),'joined')
        self.equivalent(select(g,start=1,stop=3))

    def test_reduction_axes_are_not_guessed(self):
        g=graph();g['inputs']=[dict(name='x',shape=[2,8,3])];g['nodes']=[]
        node(g,'mean',['x'],dict(axes=[2],keepdims=False),'joined');self.equivalent(select(g))

    def test_invalid_limits_and_expansion_are_rejected_without_mutation(self):
        g=select(graph(),start=1,stop=6);before=copy.deepcopy(g)
        for limit in [0,True,1025]:
            with self.assertRaises(ValueError):normalize(g,maximum_nodes=limit)
        with self.assertRaisesRegex(ValueError,'node budget'):normalize(g,maximum_nodes=1)
        self.assertEqual(g,before)

    def test_model_renaming_does_not_select_another_fusion(self):
        a=select(graph());b=copy.deepcopy(a);b['id']='resnet20_is_only_a_label'
        _,x=normalize(a);_,y=normalize(b);self.assertEqual(x['actions'],y['actions'])


if __name__=='__main__':unittest.main()
