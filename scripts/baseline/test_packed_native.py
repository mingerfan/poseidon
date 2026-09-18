"""Native syntax + variable-period ABI, legacy isolation, privacy and bad keys."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from candidate_contract import make_request,validate_candidate,request_rotations,request_input_names,PACKED_NATIVE_GUIDANCE
from deepseek_provider import public_request
from decorated_functions import validate,register
from fx_to_hecate import translate
from run_model_batch import prepare_case
from model_graph import evaluate_reference
from packed_input_abi import NATIVE_TASK,NATIVE_CONTRACT,rotations
from packed_native_cases import cases,candidate

class PackedNativeTests(unittest.TestCase):
    def test_manifest(self):
        self.assertEqual(json.loads((Path(__file__).parent/'cases/packed-native-6-manifest.json').read_text()),
                         dict(schema=1,cases=cases()))

    def test_models_reference_and_native_contract(self):
        periods=[];outputs=[]
        for i,d in enumerate(cases()):
            with self.subTest(case=d['id']),tempfile.TemporaryDirectory() as tmp:
                model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
                r=make_request(p,d,'a'*64,str(model),native_array_mutation=True)
                self.assertEqual(r['task'],NATIVE_TASK);self.assertEqual(r['semantic_guidance'],PACKED_NATIVE_GUIDANCE)
                self.assertEqual(public_request(r),r)
                for name in ('inputs','logical_inputs','reference','hecate_source'):self.assertNotIn(name,r)
                check=validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=candidate(p,i)),r)
                period=r['layout']['input_slot_period'];periods.append(period);outputs.append(r['layout']['output_ciphertexts'])
                self.assertEqual(check['contract'],NATIVE_CONTRACT)
                self.assertEqual(request_rotations(r),rotations(period))
                self.assertTrue(set(check['rotation_steps'])<=set(rotations(period)))
                self.assertEqual(check['native_functions']['packed_binding']['slot_period'],period)
                self.assertEqual(check['inputs'][0]['slot_period'],period)
                self.assertEqual(check['native_functions']['contract'],'decorated-functions-core-v7')
                if i==0:self.assertTrue(check['native_functions']['array_mutation']['sites'])
                if i==1:
                    self.assertTrue(check['native_functions']['public_loops'])
                    self.assertTrue(check['native_functions']['scalar_augmented']['sites'])
                if i==3:self.assertEqual(request_input_names(r),('x','zero_ct'))
                with np.load(Path(tmp)/'arrays.npz',allow_pickle=False) as data:
                    for x,ref in zip(data['logical_inputs'],data['reference']):
                        np.testing.assert_array_equal(evaluate_reference(d,x.tolist()),ref)
                        with torch.no_grad():actual=model(torch.from_numpy(x)).numpy().ravel()
                        np.testing.assert_allclose(actual,ref,atol=1e-12,rtol=1e-12)
        self.assertEqual(set(periods),{8,16,32,64,128,256})
        self.assertIn(6,outputs);self.assertIn(16,outputs)

    def test_old_contract_limits_and_wrong_period_keys_rejected(self):
        source='@hc.func("c")\ndef golden(x):\n    return x.rotate(8)*weight\n'
        with self.assertRaises(ValueError):validate(source,{'weight':[.25]*16},array_mutation=True)
        validate(source,{'weight':[.25]*16},array_mutation=True,slot_period=16)
        for period in (True,3,512):
            with self.assertRaises(ValueError):validate(source,{},array_mutation=True,slot_period=period)
        class Untouched:
            def __getattr__(self,n):raise AssertionError('frontend touched before rejection')
        for step in (-1,3,16,True):
            invalid=source.replace('rotate(8)','rotate('+repr(step)+')')
            with self.assertRaises(ValueError):
                register(invalid,{'weight':[.25]*16},Untouched(),array_mutation=True,slot_period=16)
        for data in ([.25]*4,[.25]*32,[float('nan')]):
            with self.assertRaises(ValueError):validate(source,{'weight':data},array_mutation=True,slot_period=16)
        wide='@hc.func("c")\ndef golden(x):\n    return ['+','.join(['x']*16)+']\n'
        validate(wide,{},16,array_mutation=True,slot_period=256)
        with self.assertRaises(ValueError):validate(wide,{},16,array_mutation=True)

    def test_versioned_binding_and_legacy_grammar_do_not_mix(self):
        d=cases()[0]
        with tempfile.TemporaryDirectory() as tmp:
            model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
            for options in (dict(native_functions=True),dict(native_arrays=True),dict(native_public_loops=True),
                            dict(native_array_mutation=True,public_construction=True)):
                with self.assertRaises(ValueError):make_request(p,d,'a'*64,**options)
            r=make_request(p,d,'a'*64,native_array_mutation=True)
            for mutate in ('period','task','guidance'):
                bad=copy.deepcopy(r)
                if mutate=='period':bad['layout']['input_slot_period']=4
                elif mutate=='task':bad['task']='hecate-native-function-synthesis-v10'
                else:bad['semantic_guidance']['schema']=True
                with self.assertRaises(ValueError):public_request(bad)
            for index in (0,2):
                current=cases()[index]
                model,manifest=prepare_case(current,Path(tmp));p=translate(model,manifest)
                r=make_request(p,current,'a'*64,native_array_mutation=True)
                # Legal but intentionally wrong semantics; numerical gate must reject.
                validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=candidate(p,index,True)),r)

if __name__=='__main__':unittest.main()
