"""Logical reference versus independent block lowering; not FHE evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from chunked_input_abi import TASK, binding, pack_inputs, validate_request_binding
from chunked_model import lower, test_inputs as inputs
from chunked_model_cases import case, cases
from model_graph import validate_graph, build_graph_model, evaluate_reference


class ChunkedModelTests(unittest.TestCase):
    def test_frozen_custom_manifest(self):
        from custom_batch_manifest import load_manifest
        rows,data,_=load_manifest(Path(__file__).with_name('cases')/'chunked-input-10-manifest.json')
        self.assertEqual(data,dict(schema=1,cases=cases()))
        self.assertEqual([r['descriptor'] for r in rows],cases())

    def test_all_lengths_and_ranks_keep_every_input_element(self):
        for n in range(5,17):
            for shape in ([n],[1,n],[1,n,1],[1,1,n,1]):
                d = case(shape)
                logical = inputs(shape)
                packed = pack_inputs(logical,shape)
                graph, plan = lower(d)
                model, _ = build_graph_model(d)
                physical, _ = build_graph_model(graph)
                for batch in range(4):
                    reference = evaluate_reference(d,logical[batch].tolist())
                    feed = {s['name']: packed[batch,i].tolist() for i,s in enumerate(graph['inputs'])}
                    np.testing.assert_allclose(evaluate_reference(graph,feed),reference,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(model(torch.from_numpy(logical[batch])).numpy(),reference,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(physical(*[torch.from_numpy(v) for v in packed[batch]]).numpy(),reference,atol=1e-12,rtol=1e-12)
                np.testing.assert_array_equal(packed.reshape(4,-1)[:,:n],logical.reshape(4,n))
                self.assertTrue(np.all(packed.reshape(4,-1)[:,n:] == 0))
                self.assertEqual(plan['logical_elements'],n)

    def test_mlp_fanout_residual_and_zero_rows(self):
        for d in cases():
            graph,_ = lower(d)
            original = inputs(d['input_shape'])
            packed = pack_inputs(original,d['input_shape'])
            for i in range(4):
                feed = {s['name']:packed[i,j].tolist() for j,s in enumerate(graph['inputs'])}
                np.testing.assert_allclose(evaluate_reference(graph,feed),
                    evaluate_reference(d,original[i].tolist()),atol=1e-12,rtol=1e-12)

    def test_linear_basis_vectors_cross_every_chunk_boundary(self):
        for n in range(5,17):
            d=case([n]);graph,_=lower(d)
            for column in range(n):
                logical=np.zeros((4,n),dtype=np.float64)
                logical[:,column]=1.0
                packed=pack_inputs(logical,[n])
                feed={s['name']:packed[0,i].tolist() for i,s in enumerate(graph['inputs'])}
                expected=[row[column]+b for row,b in zip(d['constants']['weight'],d['constants']['bias'])]
                np.testing.assert_allclose(evaluate_reference(graph,feed),expected,atol=0,rtol=0)

    def test_input_padding_does_not_contribute_after_public_offset(self):
        d = case([7],'fanout')
        graph,_ = lower(d)
        original = inputs([7])
        packed = pack_inputs(original,[7])
        # Trusted runtime uses zero padding. Perturb here to check weight masks
        # independently, not as a substitute for verifying the runtime mapping.
        packed[:,-1,-1] = 37.0
        for i in range(4):
            actual = evaluate_reference(graph,{s['name']:packed[i,j].tolist() for j,s in enumerate(graph['inputs'])})
            np.testing.assert_allclose(actual,evaluate_reference(d,original[i].tolist()),atol=1e-12,rtol=1e-12)

    def test_prepare_reference_and_provider_boundary(self):
        from run_model_batch import prepare_case
        from fx_to_hecate import translate
        from candidate_contract import make_request, validate_candidate, request_input_names
        from deepseek_provider import public_request
        for d in cases():
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)
                model,shape=prepare_case(d,folder)
                payload=translate(model,shape)
                request=make_request(payload,d,'a'*64,str(model))
                self.assertEqual(request['task'],TASK)
                validate_request_binding(request)
                response=dict(schema=1,request_id=request['request_id'],hecate_source=payload['hecate_source'])
                validate_candidate(response,request)
                sent=public_request(request)
                self.assertEqual(sent['model'],d)
                self.assertNotIn('hecate_source',sent)
                self.assertNotIn('reference',sent)
                self.assertNotIn('logical_inputs',sent)
                with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:
                    self.assertEqual(arrays['logical_inputs'].shape,(4,*d['input_shape']))
                    np.testing.assert_array_equal(arrays['inputs'],pack_inputs(arrays['logical_inputs'],d['input_shape']))
                    expected=[evaluate_reference(d,x.tolist()) for x in arrays['logical_inputs']]
                    np.testing.assert_allclose(arrays['reference'],expected,atol=0,rtol=0)
                changed=copy.deepcopy(request)
                changed['layout']['model_input_binding']['chunks'][0]['flat_stop']=3
                with self.assertRaises(ValueError):request_input_names(changed)
                changed=copy.deepcopy(request)
                changed['layout']['inputs'].reverse()
                with self.assertRaises(ValueError):request_input_names(changed)

    def test_invalid_shapes_ops_arity_and_runtime_overrides(self):
        for shape in ([4],[17],[],[True,8],[1,1,1,1,8],[0,8],[-1,8],[2.0,4]):
            d=case([8]);d['input_shape']=shape
            with self.subTest(shape=shape),self.assertRaises(ValueError):validate_graph(d)
        for op in ('rotate','conv2d','relu','reshape','bootstrap','eval'):
            d=case([8]);d['nodes'][0]['op']=op
            with self.subTest(op=op),self.assertRaises(ValueError):validate_graph(d)
        for changes in ({'weight':'unknown'},{'bias':True},{'inputs':['future']},{'id':'weight'}, {'runtime':'override'}):
            d=case([8]);d['nodes'][1].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_graph(d)
        d=case([8]);d['output']='flat'
        with self.assertRaises(ValueError):validate_graph(d)

    def test_discarding_last_chunk_changes_reference(self):
        for d in cases():
            x=inputs(d['input_shape'])[1]
            wrong=x.copy().reshape(-1); wrong[-1]=0
            self.assertFalse(np.allclose(evaluate_reference(d,x.tolist()),
                evaluate_reference(d,wrong.reshape(x.shape).tolist()),atol=1e-10,rtol=0))


if __name__ == '__main__':unittest.main()
