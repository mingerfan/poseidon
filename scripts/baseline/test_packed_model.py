"""Offline variable-period contracts and independent plaintext references."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from packed_input_abi import ABI,TASK,CONTRACT,binding,pack_inputs,rotations
from packed_model import test_inputs as inputs
from packed_model_cases import case,cases
from model_graph import validate_graph,build_graph_model,evaluate_reference


class PackedModelTests(unittest.TestCase):
    def test_frozen_manifest(self):
        from custom_batch_manifest import load_manifest
        rows,data,_=load_manifest(Path(__file__).with_name('cases')/'packed-input-12-manifest.json')
        self.assertEqual(data,dict(schema=1,cases=cases()))
        self.assertEqual([r['descriptor'] for r in rows],cases())

    def test_artifact_policy_is_opt_in_and_period_specific(self):
        import struct
        from test_seal_cpu_golden import artifact,EMPTY_CST
        from seal_artifact_gate import inspect_artifacts
        program=artifact(ops=((1,0,0,16),))
        options=dict(rotation_steps=rotations(32),execution_abi=ABI,input_period=32)
        gate=inspect_artifacts(program,EMPTY_CST,**options)
        self.assertEqual(gate['rotation_steps'],[16])
        with self.assertRaises(ValueError):inspect_artifacts(program,EMPTY_CST)
        with self.assertRaises(ValueError):inspect_artifacts(program,EMPTY_CST,input_period=32)
        for n in (1,32):
            cst=struct.pack('<qq',1,n)+struct.pack('<'+str(n)+'d',*([.25]*n))
            inspect_artifacts(program,cst,**options)
        for n in (2,4,16,64):
            cst=struct.pack('<qq',1,n)+struct.pack('<'+str(n)+'d',*([.25]*n))
            with self.assertRaises(ValueError):inspect_artifacts(program,cst,**options)
        for op in (5,10,12345):
            with self.assertRaises(ValueError):inspect_artifacts(artifact(ops=((op,0,0,0),)),EMPTY_CST,**options)

    def test_period_boundaries_and_rank(self):
        for n in (1,3,4,5,8,9,16,17,31,32,33,63,64,65,127,128,129,255,256):
            for shape in ([n],[1,n],[1,n,1],[1,1,n,1]):
                d=case(shape);validate_graph(d)
                plan=binding(shape);original=inputs(shape);packed=pack_inputs(original,shape)
                self.assertEqual(plan['slot_period'],max(4,1<<(n-1).bit_length()))
                np.testing.assert_array_equal(packed[:,:n],original.reshape(4,n))
                self.assertTrue(np.all(packed[:,n:]==0))
                model,_=build_graph_model(d)
                for i in range(4):
                    np.testing.assert_allclose(model(torch.from_numpy(original[i])).numpy(),
                        evaluate_reference(d,original[i].tolist()),atol=1e-12,rtol=1e-12)

    def test_translation_request_privacy_and_original_reference(self):
        from run_model_batch import prepare_case
        from fx_to_hecate import translate
        from candidate_contract import make_request,validate_candidate,request_input_names
        from deepseek_provider import public_request
        for d in cases():
            with self.subTest(case=d['id']),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);model,shape=prepare_case(d,folder)
                payload=translate(model,shape)
                request=make_request(payload,d,'a'*64,str(model))
                self.assertEqual(request['task'],TASK)
                check=validate_candidate(dict(schema=1,request_id=request['request_id'],
                    hecate_source=payload['hecate_source']),request)
                sent=public_request(request)
                self.assertEqual(sent['model'],d)
                self.assertIn('NOT always four',sent['semantic_guidance']['layout'])
                self.assertIn('Rebinding and +=/-=/*=',sent['semantic_guidance']['ssa_aliases'])
                for field in ('hecate_source','reference','inputs','logical_inputs'):
                    self.assertNotIn(field,sent)
                with np.load(folder/'arrays.npz',allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['inputs'],pack_inputs(data['logical_inputs'],d['input_shape']))
                    np.testing.assert_array_equal(data['reference'],[evaluate_reference(d,x.tolist()) for x in data['logical_inputs']])
                for key,value in [('input_slot_period',4),('output_selectors',[[0,0]])]:
                    changed=copy.deepcopy(request);changed['layout'][key]=value
                    with self.assertRaises(ValueError):request_input_names(changed)
                changed=copy.deepcopy(request);changed['task']='hecate-synthesis-v0'
                with self.assertRaises(ValueError):request_input_names(changed)

    def test_all_slots_contribute_and_padding_does_not(self):
        # Independent direct dot products, including each last logical column.
        for n in (17,31,64,127,128,255,256):
            d=case([n]);w=np.asarray(d['constants']['weight']);b=np.asarray(d['constants']['bias'])
            for column in (0,n//2,n-1):
                x=np.zeros(n);x[column]=1
                np.testing.assert_array_equal(evaluate_reference(d,x.tolist()),w[:,column]+b)
            padded=np.pad(w,((0,0),(0,binding([n])['slot_period']-n)))
            x=np.full(padded.shape[1],37.0);x[:n]=inputs([n])[1]
            np.testing.assert_allclose(padded@x+b,evaluate_reference(d,x[:n].tolist()),atol=1e-12,rtol=1e-12)

    def test_reject_shapes_and_unimplemented_ops(self):
        for shape in ([],[0],[257],[True],[2.0],[1,1,1,1,4],[17,17]):
            with self.subTest(shape=shape),self.assertRaises(ValueError):binding(shape)
        for op in ('conv2d','relu','rotate','reshape','bootstrap','eval'):
            d=case([17]);d['nodes'][0]['op']=op
            with self.subTest(op=op),self.assertRaises(ValueError):validate_graph(d)
        for period in (3,17,512,True):
            with self.assertRaises(ValueError):rotations(period)

    def test_period_specific_static_contract_and_legacy_isolation(self):
        from hecate_contract import validate_function
        for p in (4,8,16,32,64,128,256):
            source='@hc.func("c")\ndef golden(x):\n    y = x.rotate('+str(p//2)+')\n    return y\n'
            validate_function(source,{},contract=CONTRACT,slot_period=p)
            for step in (-1,3,p):
                with self.assertRaises(ValueError):
                    validate_function(source.replace(str(p//2)+')',str(step)+')'),{},contract=CONTRACT,slot_period=p)
        with self.assertRaises(ValueError):
            validate_function('@hc.func("c")\ndef golden(x):\n    return x\n',{},slot_period=32)


if __name__=='__main__':unittest.main()
