"""Frozen-statistics BN and branch/layout composition, not just final labels."""
import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from packed_composition_cases import cases
from packed_model import validate,test_inputs
from model_graph import build_graph_model,evaluate_reference
from fx_to_hecate import translate
from packed_input_abi import ABI


class PackedCompositionTests(unittest.TestCase):
    def test_frozen_manifest(self):
        from custom_batch_manifest import load_manifest
        _,data,_=load_manifest(Path(__file__).with_name('cases')/'packed-composition-12-manifest.json')
        self.assertEqual(data,dict(schema=1,cases=cases()))

    def test_all_models_torch_independent_reference_and_contract(self):
        from run_model_batch import prepare_case
        from candidate_contract import make_request,validate_candidate
        from deepseek_provider import public_request
        for d in cases():
            with self.subTest(case=d['id']),tempfile.TemporaryDirectory() as tmp:
                info=validate(d);model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
                self.assertEqual(p['layout']['output_shape'],info['output_shape'])
                request=make_request(p,d,'a'*64,str(model))
                validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=p['hecate_source']),request)
                self.assertEqual(public_request(request)['model'],d)
                with np.load(Path(tmp)/'arrays.npz',allow_pickle=False) as data:
                    for i,x in enumerate(data['logical_inputs']):
                        expected=evaluate_reference(d,x.tolist())
                        np.testing.assert_array_equal(data['reference'][i],expected)
                        with torch.no_grad():actual=model(torch.from_numpy(x)).numpy().ravel()
                        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_bn_module_equivalence_and_frozen_state(self):
        from fx_to_hecate import state_arrays
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.BatchNorm2d(4,dtype=torch.float64,eps=.125)
            def forward(self,x):return self.norm(x)
        d=cases()[3];model=Model().eval()
        with torch.no_grad():
            for field,key in [('running_mean','mean'),('running_var','variance'),('weight','gamma'),('bias','beta')]:
                getattr(model.norm,field).copy_(torch.tensor(d['constants'][key],dtype=torch.float64))
        before=state_arrays(model,max_elements=4096);functional,shape=build_graph_model(d)
        manifest=dict(execution_abi=ABI,input_shape=shape)
        a=translate(model,manifest);b=translate(functional,manifest)
        for field in ('hecate_source','public_constants','layout'):self.assertEqual(a[field],b[field])
        for key,value in before.items():np.testing.assert_array_equal(value,state_arrays(model,max_elements=4096)[key])
        for x in test_inputs(d['input_shape']):
            with torch.no_grad():actual=model(torch.from_numpy(x)).numpy().ravel()
            np.testing.assert_allclose(actual,evaluate_reference(d,x.tolist()),atol=1e-12,rtol=1e-12)
        with self.assertRaises(ValueError):translate(model.train(),manifest)
        model.eval();model.norm.track_running_stats=False
        with self.assertRaises(ValueError):translate(model,manifest)

    def test_wrong_channel_epsilon_and_concat_order_observable(self):
        d=cases()[1];x=test_inputs(d['input_shape'])[1].tolist();expected=evaluate_reference(d,x)
        wrong=copy.deepcopy(d);wrong['constants']['mean'].reverse()
        self.assertNotEqual(expected,evaluate_reference(wrong,x))
        wrong=copy.deepcopy(d);wrong['nodes'][0]['eps']=.25
        self.assertNotEqual(expected,evaluate_reference(wrong,x))
        for index in (8,9,10,11):
            d=cases()[index];wrong=copy.deepcopy(d)
            next(n for n in wrong['nodes'] if n['op']=='concat')['inputs'].reverse()
            x=test_inputs(d['input_shape'])[1].tolist()
            self.assertNotEqual(evaluate_reference(d,x),evaluate_reference(wrong,x))

    def test_concat_mapping_all_axes_and_duplicate_branches(self):
        from concat_ops import source_order,reference
        for shape in ([3],[2,3],[1,2,2],[1,1,2,2]):
            first=np.arange(np.prod(shape)).reshape(shape);second=first+100
            for axis in range(-len(shape),len(shape)):
                out,order=source_order([shape,shape],axis,max_elements=16)
                actual=np.array([(first,second)[b].ravel()[i] for b,i in order]).reshape(out)
                np.testing.assert_array_equal(actual,np.concatenate([first,second],axis=axis))
                np.testing.assert_array_equal(actual,reference([first.tolist(),second.tolist()],axis))
        d=cases()[8];d['nodes'][-1]['inputs']=['x','x'];model,shape=build_graph_model(d)
        manifest=dict(execution_abi=ABI,input_shape=shape)
        p=translate(model,manifest);self.assertEqual(p['layout']['output_shape'],[2,6])

    def test_zero_gamma_uses_trusted_encrypted_zero(self):
        for index in (5,6):
            model,shape=build_graph_model(cases()[index]);p=translate(model,dict(execution_abi=ABI,input_shape=shape))
            self.assertEqual(p['layout']['auxiliary_ciphertexts'][0]['dsl_name'],'zero_ct')
        d=cases()[5];outputs=[evaluate_reference(d,x.tolist()) for x in test_inputs(d['input_shape'])]
        for values in outputs[1:]:self.assertEqual(values,outputs[0])

    def test_legacy_bounds_and_reject_invalid_models(self):
        from batch_norm_ops import coefficients
        from concat_ops import geometry
        with self.assertRaises(ValueError):coefficients((2,3,5),[0.]*3,[1.]*3,None,None,.1)
        with self.assertRaises(ValueError):geometry([(2,3),(2,3)],1)
        for field,bad in [('eps',True),('eps',-1.),('running_mean',None),('training',False)]:
            d=cases()[0];d['nodes'][0][field]=bad
            with self.assertRaises(ValueError):validate(d)
        for key,bad in [('mean',[0.]),('variance',[-1.]*5),('gamma',[float('inf')]*5)]:
            d=cases()[0];d['constants'][key]=bad
            with self.assertRaises(ValueError):validate(d)
        for axis in (True,2,-3):
            d=cases()[8];d['nodes'][-1]['axis']=axis
            with self.assertRaises(ValueError):validate(d)
        d=cases()[8];d['nodes'][-1]['inputs']=['x']*3
        with self.assertRaises(ValueError):validate(d)
        d=cases()[8];d['constants']['public']=[[1.]*3]*2;d['nodes'][-1]['inputs'][1]='public'
        with self.assertRaises(ValueError):validate(d)


if __name__=='__main__':unittest.main()
