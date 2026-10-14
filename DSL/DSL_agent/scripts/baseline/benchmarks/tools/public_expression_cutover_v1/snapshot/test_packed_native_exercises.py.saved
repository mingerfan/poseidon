"""Finite influence is separate from reference and real encrypted execution."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from candidate_contract import make_request,validate_candidate
from deepseek_provider import public_request
from fx_to_hecate import translate
from run_model_batch import prepare_case
from packed_native_exercises import EXERCISES,descriptor,check_exercise,verify_trace_coverage,TASK
from packed_native_exercise_cases import candidate,entries

def request(name,tmp):
    d=descriptor(name);model,manifest=prepare_case(d,Path(tmp));p=translate(model,manifest)
    return make_request(p,d,'a'*64,str(model),native_array_mutation=True,construction_exercise=name),p

class PackedNativeExerciseTests(unittest.TestCase):
    def test_frozen_catalog_and_fixtures(self):
        self.assertEqual(entries(),EXERCISES)
        for name in EXERCISES:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                r,p=request(name,tmp);self.assertEqual(public_request(r),r);self.assertEqual(r['task'],TASK)
                source=candidate(p,name)
                checked=validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=source),r)
                coverage=checked['construction_exercise']
                self.assertEqual(list(coverage['witnesses']),r['construction_exercise']['required_features'])
                self.assertFalse(coverage['plaintext_reference_used'] or coverage['real_frontend_checked'])
                self.assertEqual(coverage['slot_period'],r['layout']['input_slot_period'])
                for k in ('reference','logical_inputs','inputs','hecate_source'):self.assertNotIn(k,r)

    def test_unused_or_cancelled_array_probe_padding_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p=request('pn-zero-item',tmp)
            answer=p['hecate_source']
            for prefix in ('    unused=np.array(x,dtype=object).item()\n',
                           '    unused=np.array(x,dtype=object).item()\n    x=x+unused-unused\n'):
                source=answer.replace('def golden(x):\n','def golden(x):\n'+prefix)
                with self.assertRaisesRegex(ValueError,'Missing contributing'):check_exercise(source,r)
            source=candidate(p,'pn-zero-item')
            with self.assertRaises(ValueError):check_exercise(source.replace('.item()','[()]'),r)

    def test_star_all_arguments_and_loop_iterations_contribute(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p=request('pn-star',tmp)
            source=candidate(p,'pn-star').replace('return v * w','return v')
            with self.assertRaisesRegex(ValueError,'Missing contributing'):check_exercise(source,r)
            r,p=request('pn-loop',tmp)
            source=candidate(p,'pn-loop').replace('range(2)','range(1)')
            with self.assertRaisesRegex(ValueError,'Missing contributing'):check_exercise(source,r)
            source=candidate(p,'pn-loop').replace('return v','return x') # undefined input, not a valid bypass
            with self.assertRaises(ValueError):check_exercise(source,r)

    def test_frozen_spec_and_runtime_witness_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,p=request('pn-array-view',tmp);c=check_exercise(candidate(p,'pn-array-view'),r)
            with self.assertRaisesRegex(ValueError,'Missing real native'):
                verify_trace_coverage(c,dict(storage=[],star=[],augmented=[],mutation=[]))
            bad=copy.deepcopy(r);bad['construction_exercise']['instruction']='skip'
            with self.assertRaises(ValueError):public_request(bad)
            with self.assertRaises(ValueError):
                make_request(p,r['model'],'a'*64,construction_exercise='pn-array-view')

if __name__=='__main__':unittest.main()

