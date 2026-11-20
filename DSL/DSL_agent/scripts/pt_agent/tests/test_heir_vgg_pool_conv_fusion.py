"""Deterministic small pool/conv identities and immutable source graph checks."""
import copy
import os
from types import SimpleNamespace as NS
import unittest

from pt_agent.experiments import heir_vgg_pool_conv_fusion as entry


def fixture(np):
    tensors = {'x':NS(name='x', shape=(1, 2, 32, 32), kind='input')}
    nodes, weights = [], {}
    previous, channels, size = 'x', 2, 32
    for index, (pool_id, conv_id, weight_name) in enumerate(entry.PAIRS):
        active, pooled, output = 'act' + str(index), 'pool' + str(index), 'conv' + str(index)
        next_channels = 3 if index % 2 == 0 else 2
        nodes.append(NS(id='activation' + str(index), op='act', inputs=[previous], outputs=[active], attrs={'fn':'silu'}, src='synthetic activation'))
        tensors[active] = NS(name=active, shape=(1, channels, size, size), kind='activation')
        tensors[pooled] = NS(name=pooled, shape=(1, channels, size//2, size//2), kind='activation')
        nodes.append(NS(id=pool_id, op='avgpool', inputs=[active], outputs=[pooled], attrs=dict(kernel=(2, 2), stride=(2, 2), padding=(0, 0)), src='synthetic pool'))
        weight = (np.arange(next_channels*channels*9, dtype=np.float64).reshape(next_channels, channels, 3, 3) % 9 - 4) / 64
        bias = (np.arange(next_channels, dtype=np.float64)-1) / 8
        weight.flags.writeable = False; bias.flags.writeable = False
        bias_name = weight_name.removesuffix('weight') + 'bias'
        weights.update({weight_name:weight, bias_name:bias})
        tensors[weight_name] = NS(name=weight_name, shape=weight.shape, kind='const')
        tensors[bias_name] = NS(name=bias_name, shape=bias.shape, kind='const')
        nodes.append(NS(id=conv_id, op='conv2d', inputs=[pooled], outputs=[output], attrs=dict(weight=weight_name, bias=bias_name, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1), src='original source retained'))
        size //= 2
        tensors[output] = NS(name=output, shape=(1, next_channels, size, size), kind='activation')
        previous, channels = output, next_channels
    return NS(name=entry.MODEL, nodes=nodes, tensors=tensors, weights=weights, inputs=['x'], outputs=[previous])


def pool(np, x):
    result = np.empty((1, x.shape[1], x.shape[2]//2, x.shape[3]//2), dtype=np.float64)
    for c in range(x.shape[1]):
        for h in range(result.shape[2]):
            for w in range(result.shape[3]):
                result[0,c,h,w] = sum(float(x[0,c,2*h+a,2*w+b]) for a in range(2) for b in range(2)) / 4
    return result


def conv(np, x, weight, bias, stride, padding):
    kh, kw = weight.shape[2:]
    shape = (1, weight.shape[0], (x.shape[2]+2*padding-kh)//stride+1, (x.shape[3]+2*padding-kw)//stride+1)
    result = np.empty(shape, dtype=np.float64)
    for oc in range(shape[1]):
        for h in range(shape[2]):
            for w in range(shape[3]):
                total = 0.
                for ic in range(x.shape[1]):
                    for a in range(kh):
                        for b in range(kw):
                            ih, iw = h*stride+a-padding, w*stride+b-padding
                            if 0 <= ih < x.shape[2] and 0 <= iw < x.shape[3]:
                                total += float(x[0,ic,ih,iw]) * float(weight[oc,ic,a,b])
                result[0,oc,h,w] = total + float(bias[oc])
    return result


def evaluate(np, graph, x):
    values = {'x':x}
    for node in graph.nodes:
        value = values[node.inputs[0]]
        if node.op == 'act':output = value / (1 + np.exp(-value))
        elif node.op == 'avgpool':output = pool(np, value)
        elif node.op == 'conv2d':
            output = conv(np, value, graph.weights[node.attrs['weight']], graph.weights[node.attrs['bias']], node.attrs['stride'][0], node.attrs['padding'][0])
        else:raise AssertionError('Unexpected synthetic operation')
        values[node.outputs[0]] = output
    return values[graph.outputs[0]]


class PoolConvFusionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        cls.np = np

    def strict(self, expected, observed):
        np = self.np
        self.assertEqual(expected.shape, observed.shape)
        self.assertTrue(np.isfinite(expected).all() and np.isfinite(observed).all())
        self.assertTrue(np.all(np.abs(expected-observed) <= 1e-12+1e-12*np.abs(expected)))

    def test_single_pair_derives_exact_blocks_and_readonly_independent_storage(self):
        np = self.np
        original = np.arange(54, dtype=np.float64).reshape(3, 2, 3, 3)/8-3
        before = original.copy()
        derived, metadata = entry.derive_weight(original)
        for a in range(2):
            for b in range(2):np.testing.assert_array_equal(derived[:,:,a::2,b::2], original/4)
        np.testing.assert_array_equal(original, before)
        self.assertFalse(np.shares_memory(original, derived));self.assertFalse(derived.flags.writeable)
        self.assertEqual(metadata['derived_bytes'], original.nbytes*4)

    def test_signed_channels_nonsquare_boundaries_and_bias_once_match_original_math(self):
        np = self.np
        for height, width in ((2, 2), (4, 6), (6, 8)):
            x = ((np.arange(2*height*width, dtype=np.float64).reshape(1, 2, height, width)%19)-9)/8
            x[:,1] *= -1.5
            weight = (np.arange(54, dtype=np.float64).reshape(3, 2, 3, 3)%13-6)/16
            bias = np.array([.5, -.25, 1.25], dtype=np.float64)
            fused, _ = entry.derive_weight(weight)
            expected = conv(np, pool(np, x), weight, bias, 1, 1)
            observed = conv(np, x, fused, bias, 2, 2)
            self.strict(expected, observed)
            self.strict(np.broadcast_to(bias[None,:,None,None], expected.shape), conv(np, np.zeros_like(x), fused, bias, 2, 2))

    def test_four_pairs_fuse_without_mutating_original_nodes_tensors_or_weights(self):
        np = self.np;graph = fixture(np)
        old_nodes, old_tensors = copy.deepcopy(graph.nodes), copy.deepcopy(graph.tensors)
        weights = dict(graph.weights);snapshots = {name:value.copy() for name,value in weights.items()}
        fused, metadata = entry.fuse_graph(graph)
        self.assertEqual(graph.nodes, old_nodes);self.assertEqual(graph.tensors, old_tensors)
        self.assertEqual(set(graph.weights), set(weights));self.assertEqual(len(fused.nodes), len(graph.nodes)-4)
        self.assertEqual(metadata['pair_count'], 4);self.assertEqual(metadata['derived_constant_count'], 4)
        self.assertIsNone(metadata['binary64_strict_math_passed']);self.assertIsNone(metadata['HEIR_layout_success'])
        for name, value in weights.items():
            self.assertIs(fused.weights[name], value);np.testing.assert_array_equal(value, snapshots[name])
        for row in metadata['pairs']:
            node = next(n for n in fused.nodes if n.id == row['conv_id'])
            self.assertEqual(node.inputs, [row['input']]);self.assertEqual(node.attrs['bias'], row['source_bias'])
            self.assertEqual(node.attrs['stride'], (2, 2));self.assertEqual(node.attrs['padding'], (2, 2))
            self.assertNotIn(row['removed_tensor'], fused.tensors)
            self.assertFalse(fused.weights[row['derived_weight']].flags.writeable)
        self.assertEqual([n.id for n in graph.nodes if n.op=='act'], [n.id for n in fused.nodes if n.op=='act'])

    def test_complete_tiny_synthetic_graph_preserves_all_final_outputs(self):
        np = self.np;graph = fixture(np);fused, _ = entry.fuse_graph(graph)
        x = (np.arange(2048, dtype=np.float64).reshape(1, 2, 32, 32)%23-11)/32
        self.strict(evaluate(np, graph, x), evaluate(np, fused, x))

    def test_metadata_counts_and_payload_accounting_are_consistent(self):
        graph = fixture(self.np);fused, metadata = entry.fuse_graph(graph)
        self.assertNotIn('avgpool', metadata['fused_operator_counts'])
        self.assertEqual(metadata['original_operator_counts']['conv2d'], metadata['fused_operator_counts']['conv2d'])
        self.assertEqual(metadata['new_graph_additional_storage_bytes'], sum(fused.weights[row['derived_weight']].nbytes for row in metadata['pairs']))
        self.assertEqual(metadata['net_fused_weight_expansion_bytes'], 3*metadata['source_fused_weights_bytes'])
        self.assertEqual(metadata['estimated_net_export_tensor_bytes'], metadata['net_fused_weight_expansion_bytes']-metadata['removed_export_pool_constant_bytes'])

    def test_reject_other_model_missing_pool_or_renamed_source_pair(self):
        for mutation in ('alex', 'missing', 'renamed'):
            graph = fixture(self.np)
            if mutation=='alex':graph.name='alexnet_silu_avgpool'
            elif mutation=='missing':graph.nodes=[n for n in graph.nodes if n.id!='n6']
            else:next(n for n in graph.nodes if n.id=='n7').id='other'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):entry.find_pairs(graph)

    def test_reject_overlap_pad_ceil_changed_divisor_and_unknown_pool_attributes(self):
        for key, value in [('kernel',(3,3)),('stride',(1,1)),('padding',(1,1)),('ceil_mode',True),('count_include_pad',False),('divisor_override',4),('new_option',0)]:
            graph = fixture(self.np);next(n for n in graph.nodes if n.id=='n6').attrs[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):entry.find_pairs(graph)

    def test_reject_odd_extent_or_inconsistent_tensor_parameter_shapes(self):
        for mutation in ('odd','pool_shape','conv_shape','weight_shape','bias_shape'):
            graph = fixture(self.np);pool_node=next(n for n in graph.nodes if n.id=='n6');conv_node=next(n for n in graph.nodes if n.id=='n7')
            if mutation=='odd':graph.tensors[pool_node.inputs[0]].shape=(1,2,31,32)
            elif mutation=='pool_shape':graph.tensors[pool_node.outputs[0]].shape=(1,2,15,16)
            elif mutation=='conv_shape':graph.tensors[conv_node.outputs[0]].shape=(1,3,15,16)
            elif mutation=='weight_shape':graph.tensors[conv_node.attrs['weight']].shape=(3,2,5,5)
            else:graph.weights[conv_node.attrs['bias']]=self.np.zeros(2,dtype=self.np.float64)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):entry.find_pairs(graph)

    def test_reject_fanout_pool_graph_output_and_intervening_activation(self):
        for mutation in ('fanout','output','intervening'):
            graph = fixture(self.np)
            if mutation=='fanout':graph.nodes.append(NS(id='extra',op='act',inputs=['pool0'],outputs=['extra'],attrs={},src='fixture'))
            elif mutation=='output':graph.outputs=['pool0']
            else:graph.nodes.insert(2,NS(id='intermediate',op='act',inputs=['pool0'],outputs=['mid'],attrs={},src='fixture'))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):entry.find_pairs(graph)

    def test_reject_changed_conv_geometry_groups_and_bias_source(self):
        for key,value in [('stride',(2,2)),('padding',(0,0)),('dilation',(2,2)),('groups',2),('groups',True),('bias',None),('weight','foreign')]:
            graph=fixture(self.np);next(n for n in graph.nodes if n.id=='n7').attrs[key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):entry.find_pairs(graph)

    def test_reject_nonfinite_wrong_dtype_weights_and_existing_derived_names(self):
        np=self.np
        for weight in (np.ones((1,1,3,3),dtype=np.float32),np.full((1,1,3,3),np.nan),np.ones((1,1,2,2))):
            with self.assertRaises(ValueError):entry.derive_weight(weight)
        graph=fixture(np);name='derived-exact-pool-conv:n6:n7';graph.weights[name]=np.zeros(1)
        with self.assertRaises(ValueError):entry.find_pairs(graph)
        graph=fixture(np);graph.weights[entry.PAIRS[0][2]].flags.writeable=True
        with self.assertRaises(ValueError):entry.fuse_graph(graph)


if __name__ == '__main__':unittest.main()
