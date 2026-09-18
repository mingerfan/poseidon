"""No API: rank-preserving operations, row-major IO and last-axis Linear."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tensor_model_cases import cases,linear,affine
from packed_model import validate,test_inputs as inputs
from model_graph import build_graph_model,evaluate_reference
from packed_input_abi import decode_output


class TensorModelTests(unittest.TestCase):
    def test_manifest_is_fixed(self):
        from custom_batch_manifest import load_manifest
        _,data,_=load_manifest(Path(__file__).with_name('cases')/'tensor-input-12-manifest.json')
        self.assertEqual(data,dict(schema=1,cases=cases()))

    def test_original_pytorch_vs_independent_reference_and_dsl_contract(self):
        from run_model_batch import prepare_case
        from fx_to_hecate import translate
        from candidate_contract import make_request,validate_candidate
        from deepseek_provider import public_request
        for d in cases():
            with self.subTest(case=d['id']),tempfile.TemporaryDirectory() as tmp:
                shape=validate(d)['output_shape'];folder=Path(tmp)
                model,manifest=prepare_case(d,folder);payload=translate(model,manifest)
                self.assertEqual(payload['layout']['output_shape'],shape)
                request=make_request(payload,d,'a'*64,str(model))
                validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=payload['hecate_source']),request)
                self.assertEqual(public_request(request)['model'],d)
                with np.load(folder/'arrays.npz',allow_pickle=False) as data:
                    self.assertEqual(data['reference_logical'].shape,(4,*shape))
                    np.testing.assert_array_equal(decode_output(data['reference'],payload['layout']),data['reference_logical'])
                    for i,x in enumerate(data['logical_inputs']):
                        np.testing.assert_allclose(model(torch.from_numpy(x)).numpy(),data['reference_logical'][i],atol=1e-12,rtol=1e-12)
                        np.testing.assert_array_equal(data['reference'][i],evaluate_reference(d,x.tolist()))

    def test_broadcast_axes_and_inferred_reshape(self):
        from logical_reshape import reshape_shape
        self.assertEqual(reshape_shape((3,5),(5,-1),max_elements=256),(5,3))
        with self.assertRaises(ValueError):reshape_shape((3,5),(5,-1))
        for source,target in [((3,5),(4,-1)),((3,5),(-1,-1)),((16,16),(0,256)),((3,5),(1,1,1,1,15))]:
            with self.assertRaises(ValueError):reshape_shape(source,target,max_elements=256)
        for plain_shape in ([1],[5],[3,1],[1,5],[1,1],[3,5]):
            d=affine([3,5],plain_shape,'axis');validate(d)
            model,_=build_graph_model(d)
            for x in inputs([3,5]):
                np.testing.assert_allclose(model(torch.from_numpy(x)).numpy().ravel(),evaluate_reference(d,x.tolist()),atol=1e-12,rtol=1e-12)
        for plain_shape in ([3],[5,1],[1,3,5]):
            with self.assertRaises(ValueError):validate(affine([3,5],plain_shape,'bad'))

    def test_last_axis_linear_keeps_groups_distinct(self):
        from fx_to_hecate import translate
        from run_model_batch import prepare_case
        for shape in ([2,8],[2,2,8],[1,2,2,16]):
            d=linear(shape,2,'basis');width=shape[-1];groups=math.prod(shape[:-1])
            bias=np.tile(d['constants']['bias'],groups)
            for group in range(groups):
                x=np.zeros(shape,dtype=np.float64);x.reshape(-1)[group*width+width-1]=1
                expected=bias.copy();expected[group*2:(group+1)*2]+=np.asarray(d['constants']['weight'])[:,-1]
                np.testing.assert_array_equal(evaluate_reference(d,x.tolist()),expected)
            with tempfile.TemporaryDirectory() as tmp:
                model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
                rows=[value for key,value in p['public_constants'].items() if '.weight[' in p['constant_origins'][key]]
                self.assertEqual(len(rows),groups*2)
                for i,row in enumerate(rows):
                    nonzero=np.nonzero(np.asarray(row))[0]
                    self.assertTrue(np.all(nonzero//width==i//2))

    def test_scalar_layout_reshape_and_broadcast(self):
        d=linear([2,8],2,'scalar-broadcast')
        d['constants']['gain']=[.25,-.5]
        d['nodes'].append(dict(id='scaled',op='multiply',inputs=['projected','gain']));d['output']='scaled'
        self.assertEqual(validate(d)['output_shape'],[2,2])
        model,_=build_graph_model(d)
        for x in inputs([2,8]):
            np.testing.assert_allclose(model(torch.from_numpy(x)).numpy().ravel(),evaluate_reference(d,x.tolist()),atol=1e-12,rtol=1e-12)

    def test_reject_wrong_weight_axis_and_output_budget(self):
        d=linear([2,8],2,'bad');d['constants']['weight']=[[.25]*16]*2
        with self.assertRaises(ValueError):validate(d)
        with self.assertRaises(ValueError):validate(linear([3,8],6,'too-many-outputs'))
        d=cases()[0];d=copy.deepcopy(d);d['nodes'].append(dict(id='wrong',op='reshape',inputs=['out'],shape=[4,4]));d['output']='wrong'
        with self.assertRaises(ValueError):validate(d)


if __name__=='__main__':unittest.main()
