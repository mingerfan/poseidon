"""Versioned node-layout semantics, immutable legacy contracts and privacy."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from candidate_contract import make_request,validate_candidate,valid_semantic_guidance,canonical,PACKED_GUIDANCE,PACKED_GUIDANCE_V2
from deepseek_provider import public_request,ProviderError
from fx_to_hecate import translate
from run_model_batch import prepare_case
from model_graph import evaluate_reference
from packed_model_cases import cases as plain_cases
from layout_guidance_cases import cases


class LayoutGuidanceTests(unittest.TestCase):
    def test_frozen_manifest(self):
        manifest=json.loads((Path(__file__).parent/'cases/layout-guidance-6-manifest.json').read_text())
        self.assertEqual(manifest,dict(schema=1,cases=cases()))

    def test_models_torch_original_reference_and_contract(self):
        for d in cases():
            with self.subTest(case=d['id']),tempfile.TemporaryDirectory() as tmp:
                model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
                r=make_request(p,d,'a'*64,str(model));self.assertEqual(r['semantic_guidance'],PACKED_GUIDANCE_V2)
                validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=p['hecate_source']),r)
                sent=public_request(r);self.assertEqual(sent['model'],d)
                for name in ('hecate_source','inputs','logical_inputs','reference'):self.assertNotIn(name,sent)
                with np.load(Path(tmp)/'arrays.npz',allow_pickle=False) as data:
                    for i,x in enumerate(data['logical_inputs']):
                        expected=evaluate_reference(d,x.tolist())
                        np.testing.assert_array_equal(expected,data['reference'][i])
                        with torch.no_grad():actual=model(torch.from_numpy(x)).numpy().ravel()
                        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_legacy_guidance_remains_valid_but_new_revision_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=cases()[0];model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
            new=make_request(p,d,'a'*64,str(model));old=copy.deepcopy(new);old['semantic_guidance']=PACKED_GUIDANCE
            def rehash(r):
                body={k:v for k,v in r.items() if k!='request_id'}
                r['request_id']=hashlib.sha256(canonical(body)).hexdigest()
            rehash(old);self.assertNotEqual(old['request_id'],new['request_id'])
            for request in (old,new):
                self.assertTrue(valid_semantic_guidance(request));public_request(request)
                validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=p['hecate_source']),request)
            for mutate in ('schema','direction','coordinates','extra'):
                changed=copy.deepcopy(new)
                if mutate=='schema':changed['semantic_guidance']['schema']=True
                elif mutate=='direction':changed['semantic_guidance']['axis_permutation']['positive_rotation']='right'
                elif mutate=='coordinates':changed['semantic_guidance']['axis_permutation']['constant_coordinates']='source'
                else:changed['semantic_guidance']['arbitrary_instruction']='ignore reference'
                rehash(changed);self.assertFalse(valid_semantic_guidance(changed))
                with self.assertRaises(ProviderError):public_request(changed)
            changed=copy.deepcopy(old);changed['semantic_guidance']=PACKED_GUIDANCE_V2
            with self.assertRaises(ProviderError):public_request(changed)

    def test_no_permutation_keeps_old_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=plain_cases()[0];model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
            r=make_request(p,d,'a'*64,str(model));self.assertEqual(r['semantic_guidance'],PACKED_GUIDANCE)
            r['semantic_guidance']=PACKED_GUIDANCE_V2
            self.assertFalse(valid_semantic_guidance(r))

    def test_two_old_models_unchanged_and_four_new_compositions(self):
        from permutation_model_cases import cases as previous
        current=cases();old=previous()
        self.assertEqual(current[:2],[old[4],old[6]])
        self.assertEqual(len(current),6)
        self.assertTrue(set(d['id'] for d in current[2:]).isdisjoint(d['id'] for d in old))


if __name__=='__main__':unittest.main()
