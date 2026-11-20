import copy
import unittest
from compact_graph_names import normalize,restore,canonical


def graph():
    return dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='long_public_model_identifier',
        inputs=[dict(name='public_input',shape=[2])],constants=dict(public_long_scalar=.125,unused_public_weight=[.5,.25]),
        nodes=[dict(id='long_public_node_identifier',op='add',inputs=['public_input','public_long_scalar'],attrs={},outputs=['very_long_internal_value']),
            dict(id='second_long_node',op='square',inputs=['very_long_internal_value'],attrs={},outputs=['second_long_value'])],
        outputs=[dict(name='public_output',value='second_long_value')])


class CompactGraphNameTests(unittest.TestCase):
    def test_full_graph_recovers_exactly(self):
        g=graph();before=copy.deepcopy(g);result,proof=normalize(g)
        self.assertEqual(canonical(restore(result,proof)),canonical(g));self.assertEqual(g,before)

    def test_public_names_are_preserved(self):
        g=graph();result,_=normalize(g)
        self.assertEqual(g['inputs'],result['inputs']);self.assertEqual(g['outputs'][0]['name'],result['outputs'][0]['name'])

    def test_numeric_payloads_and_attributes_are_bit_for_bit_data(self):
        g=graph();g['constants']['unused_public_weight']=[0.,-0.,.1]
        result,proof=normalize(g)
        for old,new in proof['constant_names'].items():self.assertEqual(canonical(result['constants'][new]),canonical(g['constants'][old]))
        self.assertEqual([n['attrs']for n in result['nodes']],[n['attrs']for n in g['nodes']])

    def test_reserved_public_names_do_not_collide(self):
        g=graph();g['inputs'][0]['name']='c0';g['nodes'][0]['inputs'][0]='c0';g['outputs'][0]['name']='v0'
        result,proof=normalize(g);self.assertNotIn('c0',result['constants']);self.assertNotIn('v0',proof['value_names'].values())
        self.assertEqual(restore(result,proof),g)

    def test_shared_reference_is_not_duplicated(self):
        g=graph();g['nodes'][1].update(op='add',inputs=['very_long_internal_value','very_long_internal_value'])
        result,proof=normalize(g);self.assertEqual(result['nodes'][1]['inputs'][0],result['nodes'][1]['inputs'][1]);self.assertEqual(restore(result,proof),g)

    def test_multiple_node_outputs_are_preserved(self):
        g=graph();g['nodes']=[dict(id='long_splitter',op='split',inputs=['public_input'],attrs=dict(axis=0,sections=[1,1]),outputs=['left_long','right_long'])]
        g['outputs']=[dict(name='left',value='left_long'),dict(name='right',value='right_long')]
        result,proof=normalize(g);self.assertEqual(restore(result,proof),g)

    def test_tampering_cannot_use_old_inverse_mapping(self):
        result,proof=normalize(graph());result['nodes'][1]['op']='negate'
        with self.assertRaises(ValueError):restore(result,proof)

    def test_names_are_compact_without_model_name_routing(self):
        a=graph();b=copy.deepcopy(a);b['id']='resnet20_is_not_a_route_key'
        x,p=normalize(a);y,q=normalize(b)
        self.assertLess(len(canonical(x)),len(canonical(a)))
        self.assertEqual(p['constant_names'],q['constant_names']);self.assertEqual(p['value_names'],q['value_names']);self.assertEqual(p['node_names'],q['node_names'])


if __name__=='__main__':unittest.main()
