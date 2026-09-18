"""Independent axis indexing, periodic routing, FX signatures and contract guards."""
import copy
import itertools
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from tensor_permutation import axes,transpose_axes,source_order,routing,reference
from permutation_model_cases import cases
from packed_input_abi import ABI,binding
from packed_model import validate,test_inputs
from fx_to_hecate import translate,Emitter,CipherValue
from model_graph import build_graph_model,evaluate_reference


class TensorPermutationTests(unittest.TestCase):
    def test_frozen_manifest(self):
        from custom_batch_manifest import load_manifest
        _,data,_=load_manifest(Path(__file__).with_name('cases')/'tensor-permutation-12-manifest.json')
        self.assertEqual(data,dict(schema=1,cases=cases()))

    def test_all_axis_orders_and_padding_do_not_leak(self):
        for shape in ([5],[2,3],[2,3,5],[2,3,2,4],[16,16],[2,4,4,8]):
            n=math.prod(shape);x=np.arange(n,dtype=np.float64).reshape(shape);period=binding(shape)['slot_period']
            for dims in itertools.permutations(range(len(shape))):
                with self.subTest(shape=shape,dims=dims):
                    expected=x.transpose(dims);out,order,masks=routing(shape,dims,period)
                    self.assertEqual(out,expected.shape)
                    np.testing.assert_array_equal(np.asarray(reference(x.tolist(),shape,dims)),expected)
                    np.testing.assert_array_equal(x.ravel()[order],expected.ravel())
                    # Nonzero padding and repeated periods make wrap/slot errors visible.
                    source=np.tile(np.pad(x.ravel(),(0,period-n),constant_values=97),3)
                    actual=sum(np.roll(source,-step)*np.tile(mask,3) for step,mask in masks.items())
                    np.testing.assert_array_equal(actual.reshape(3,period)[:,:n],np.tile(expected.ravel(),(3,1)))
                    np.testing.assert_array_equal(actual.reshape(3,period)[:,n:],0.)

    def test_original_models_torch_reference_and_candidate_contract(self):
        from run_model_batch import prepare_case
        from candidate_contract import make_request,validate_candidate
        from deepseek_provider import public_request
        for d in cases():
            with self.subTest(case=d['id']),tempfile.TemporaryDirectory() as tmp:
                output=validate(d)['output_shape'];model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
                self.assertEqual(p['layout']['output_shape'],output)
                request=make_request(p,d,'a'*64,str(model))
                validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=p['hecate_source']),request)
                sent=public_request(request);self.assertEqual(sent['model'],d)
                for name in ('hecate_source','reference','inputs','logical_inputs'):self.assertNotIn(name,sent)
                with np.load(Path(tmp)/'arrays.npz',allow_pickle=False) as data:
                    for i,x in enumerate(data['logical_inputs']):
                        expected=evaluate_reference(d,x.tolist())
                        np.testing.assert_array_equal(data['reference'][i],expected)
                        with torch.no_grad():actual=model(torch.from_numpy(x)).numpy().ravel()
                        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_method_function_variadic_and_keyword_forms(self):
        class A(torch.nn.Module):
            def forward(self,x):return x.permute(2,0,1)
        class B(torch.nn.Module):
            def forward(self,x):return x.permute(dims=(2,0,1))
        class C(torch.nn.Module):
            def forward(self,x):return torch.permute(x,dims=(2,0,1))
        class D(torch.nn.Module):
            def forward(self,x):return x.transpose(dim0=0,dim1=-1)
        class E(torch.nn.Module):
            def forward(self,x):return torch.transpose(x,0,dim1=-1)
        manifest=dict(execution_abi=ABI,input_shape=[2,3,5])
        for group in ((A,B,C),(D,E)):
            payloads=[translate(cls().eval(),manifest) for cls in group]
            x=torch.arange(30,dtype=torch.float64).reshape(2,3,5)
            expected=group[0]().eval()(x)
            for cls in group:torch.testing.assert_close(cls().eval()(x),expected,rtol=0,atol=0)
            for p in payloads[1:]:
                for field in ('hecate_source','public_constants','layout'):self.assertEqual(p[field],payloads[0][field])

    def test_scalar_layout_and_identity_do_not_add_fhe_ops(self):
        e=Emitter(slot_period=8,packed_abi=True)
        v=e.permute(CipherValue(('a','b','c','d','e','f'),(2,3),False),(1,0),'scalar')
        self.assertEqual(v.names,('a','d','b','e','c','f'));self.assertEqual(e.lines,[])
        for shape,dims in [((2,3),(0,1)),((1,6),(1,0)),((1,2,1,3),(2,1,0,3))]:
            e=Emitter(slot_period=8,packed_abi=True);v=e.permute(CipherValue(('x',),shape,True),dims,'identity')
            self.assertEqual(e.lines,[]);self.assertEqual(v.names,('x',))

    def test_rotation_only_in_allowed_positive_powers(self):
        import ast
        from test_fx_to_hecate import interpret_fragment
        for d in cases()[:5]:
            model,shape=build_graph_model(d);p=translate(model,dict(execution_abi=ABI,input_shape=shape))
            period=p['layout']['input_slot_period']
            shifts=[n.args[0].value for n in ast.walk(ast.parse(p['hecate_source'])) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='rotate']
            self.assertTrue(shifts);self.assertTrue(all(s>0 and s<period and s&(s-1)==0 for s in shifts))
            for x in test_inputs(shape):
                padded=np.pad(x.ravel(),(0,period-x.size),constant_values=71.)
                np.testing.assert_allclose(interpret_fragment(p,padded),evaluate_reference(d,x.tolist()),atol=1e-12,rtol=1e-12)

    def test_not_reshape_and_inverse_roundtrip(self):
        x=np.arange(6,dtype=np.float64).reshape(2,3)
        self.assertFalse(np.array_equal(reference(x.tolist(),[2,3],[1,0]),x.reshape(3,2)))
        for dims in itertools.permutations(range(3)):
            x=np.arange(30).reshape(2,3,5);forward=reference(x.tolist(),list(x.shape),dims)
            shape=[x.shape[d] for d in dims];inverse=[dims.index(d) for d in range(3)]
            np.testing.assert_array_equal(reference(forward,shape,inverse),x)

    def test_invalid_axes_and_legacy_isolation(self):
        for dims in ([0,0],[0],[0,2],[True,1],[0.,1],[0,-3]):
            with self.assertRaises(ValueError):axes([2,3],dims)
        for a,b in ((True,1),(0,2),(0,-3),(0.,1)):
            with self.assertRaises(ValueError):transpose_axes([2,3],a,b)
        d=cases()[0];d['schema']=2
        with self.assertRaises(ValueError):validate_graph_legacy(d)
        class Legacy(torch.nn.Module):
            def forward(self,x):return x.transpose(0,1)
        with self.assertRaises(ValueError):translate(Legacy().eval(),[2,2])
        for field,bad in [('dims',[0,0,2]),('dims',[2,0]),('extra',1)]:
            d=cases()[1];d['nodes'][0][field]=bad
            with self.assertRaises(ValueError):validate(d)


def validate_graph_legacy(d):
    from model_graph import validate_graph
    return validate_graph(d)


if __name__=='__main__':unittest.main()
