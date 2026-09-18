"""Independent windows vs PyTorch, compile lowering and batch/group guards."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from packed_spatial_cases import paid_cases as cases,conv,pool
from packed_model import validate,test_inputs as inputs
from model_graph import build_graph_model,evaluate_reference
from packed_spatial import lowering,geometry


class PackedSpatialTests(unittest.TestCase):
    def test_frozen_manifests(self):
        from custom_batch_manifest import load_manifest
        from packed_spatial_cases import cases as base_cases
        for name,expected in [('packed-spatial-14-manifest.json',base_cases()),('packed-spatial-16-manifest.json',cases())]:
            _,data,_=load_manifest(Path(__file__).with_name('cases')/name)
            self.assertEqual(data,dict(schema=1,cases=expected))

    def test_nested_torch_modules_match_functional_fx_lowering(self):
        from fx_to_hecate import translate
        from packed_input_abi import ABI
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv=torch.nn.Conv2d(1,1,3,stride=2).double()
                self.pool=torch.nn.AvgPool2d(2,count_include_pad=False)
                with torch.no_grad():
                    self.conv.weight.copy_(torch.arange(9,dtype=torch.float64).reshape(1,1,3,3)/32)
                    self.conv.bias.fill_(.03125)
            def forward(self,x):return self.pool(self.conv(x))
        model=Model().eval();shape=[2,1,5,5]
        d=conv(shape,2,1,[3,3],[2,2],[0,0],'module-equivalence')
        d['constants']['weight']=model.conv.weight.detach().tolist();d['constants']['bias']=[.03125]
        d['nodes'].append(dict(id='pool',op='avg_pool2d',inputs=['out'],kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=False));d['output']='pool'
        functional,_=build_graph_model(d)
        a=translate(model,dict(execution_abi=ABI,input_shape=shape))
        b=translate(functional,dict(execution_abi=ABI,input_shape=shape))
        self.assertEqual(a['hecate_source'],b['hecate_source']);self.assertEqual(a['public_constants'],b['public_constants'])
        self.assertEqual(a['layout'],b['layout'])
        for x in inputs(shape):
            with torch.no_grad():actual=model(torch.from_numpy(x)).numpy().ravel()
            np.testing.assert_allclose(actual,evaluate_reference(d,x.tolist()),atol=1e-12,rtol=1e-12)
        model.conv.padding_mode='reflect'
        with self.assertRaises(ValueError):translate(model,dict(execution_abi=ABI,input_shape=shape))

    def test_original_reference_and_candidate_contract(self):
        from run_model_batch import prepare_case
        from fx_to_hecate import translate
        from candidate_contract import make_request,validate_candidate
        from deepseek_provider import public_request
        for d in cases():
            with self.subTest(case=d['id']),tempfile.TemporaryDirectory() as tmp:
                shape=validate(d)['output_shape'];model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
                self.assertEqual(p['layout']['output_shape'],shape)
                request=make_request(p,d,'a'*64,str(model))
                validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=p['hecate_source']),request)
                self.assertEqual(public_request(request)['model'],d)
                with np.load(Path(tmp)/'arrays.npz',allow_pickle=False) as data:
                    for i,x in enumerate(data['logical_inputs']):
                        np.testing.assert_allclose(model(torch.from_numpy(x)).numpy().ravel(),data['reference'][i],atol=1e-12,rtol=1e-12)
                        np.testing.assert_array_equal(data['reference'][i],evaluate_reference(d,x.tolist()))

    def test_matrix_vs_independent_windows_and_pytorch(self):
        for d in cases():
            if len(d['nodes'])!=1:continue
            node=d['nodes'][0];op=node['op'];is_conv=op.startswith('conv')
            weight=d['constants'].get('weight');bias=d['constants'].get('bias')
            ws=None if weight is None else tuple(np.asarray(weight).shape)
            kernel=ws[2:] if is_conv else node['kernel']
            matrix,biases,shape=lowering(op,tuple(d['input_shape']),weight,ws,bias,kernel,
                node['stride'],node['padding'],node.get('count_include_pad',True),
                dilation=node.get('dilation'),groups=node.get('groups',1))
            for x in inputs(d['input_shape']):
                np.testing.assert_allclose(np.asarray(matrix)@x.ravel()+biases,evaluate_reference(d,x.tolist()),atol=1e-12,rtol=1e-12)
            self.assertEqual(list(shape),validate(d)['output_shape'])

    def test_batch_and_group_do_not_leak_into_each_other(self):
        for d in (cases()[1],cases()[5],cases()[6]):
            shape=d['input_shape'];x=np.zeros(shape);x.reshape(2,-1)[1,-1]=1
            actual=np.asarray(evaluate_reference(d,x.tolist())).reshape(validate(d)['output_shape'])
            base=np.asarray(evaluate_reference(d,np.zeros(shape).tolist())).reshape(actual.shape)
            np.testing.assert_array_equal(actual[0],base[0])
            self.assertFalse(np.array_equal(actual[1],base[1]))

    def test_pool_padding_divisor_and_kernel_orientation_are_observable(self):
        for a,b in ((cases()[7],cases()[8]),(cases()[9],cases()[13])):
            x=np.ones(a['input_shape'])
            left=evaluate_reference(a,x.tolist());right=evaluate_reference(b,x.tolist())
            self.assertNotEqual(left,right)
            self.assertTrue(all(abs(v-1)<1e-12 for v in right))
        d=cases()[0];wrong=copy.deepcopy(d);wrong['constants']['weight'][0][0].reverse()
        x=inputs(d['input_shape'])[1].tolist()
        self.assertNotEqual(evaluate_reference(d,x),evaluate_reference(wrong,x))

    def test_old_profile_and_invalid_geometry_rejected(self):
        import spatial_ops
        with self.assertRaises(ValueError):spatial_ops.geometry('conv1d',(1,17),(1,1,3),(3,),(2,),(1,))
        for field,value in [('groups',True),('groups',3),('dilation',[0]),('padding',[-1]),('stride',[0]),('padding','same')]:
            d=cases()[1];d['nodes'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate(d)
        with self.assertRaises(ValueError):validate(conv([2,2,8],1,4,[1],[1],[0],'too-many'))
        with self.assertRaises(ValueError):validate(pool([1,9],1,[2],[1],[2],False,'bad-pad'))
        for field,value in [('ceil_mode',True),('divisor_override',3),('count_include_pad',1)]:
            d=cases()[7];d['nodes'][0][field]=value
            with self.assertRaises(ValueError):validate(d)


if __name__=='__main__':unittest.main()
