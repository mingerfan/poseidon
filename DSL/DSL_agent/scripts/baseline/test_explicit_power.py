"""Independent power semantics; handwritten DSL must lower to multiplication."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest

from hecate_contract import validate_function
from model_graph import validate_graph, evaluate_reference

BASE=Path(__file__).parent


def model(exponent):
    return json.loads((BASE/f'cases/explicit-power-{exponent}.json').read_text())


def reference(exponent, values):
    if exponent not in (2,4): raise ValueError('Explicit supported powers only')
    return [.5*(v*v if exponent==2 else (v*v)*(v*v))+.125 for v in values]


class PowerContractTests(unittest.TestCase):
    def test_closed_form_and_reject_unsupported_exponents(self):
        values=[-1.,-.5,.5,1.]
        self.assertEqual(reference(2,values),[.625,.25,.25,.625])
        self.assertEqual(reference(4,values),[.625,.15625,.15625,.625])
        for exponent in (2,4):
            case=model(exponent)
            validate_graph(case)
            self.assertEqual(evaluate_reference(case,values),reference(exponent,values))
            for invalid in (True,0,1,3,5,-2,2.0):
                changed=copy.deepcopy(case)
                changed['nodes'][0]['exponent']=invalid
                with self.assertRaises(ValueError): validate_graph(changed)

    def test_manual_goldens_and_wrong_exponents_are_type_valid(self):
        constants={'c0':[.5],'c1':[.125]}
        for name in ('power2','power4','wrong_power2','wrong_power4'):
            source=(BASE/f'golden_cases/explicit_power/{name}.py').read_text()
            validate_function(source,constants,contract='hecate-function-v1')
        # A graph-level power is not permission to use arbitrary Python pow in DSL.
        for expression in ('x ** 4','pow(x,4)','x.pow(4)'):
            with self.assertRaises(ValueError):
                validate_function('@hc.func("c")\ndef golden(x):\n    return '+expression+'\n',constants,
                                  contract='hecate-function-v1')


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires pinned Torch')
class PowerTorchTests(unittest.TestCase):
    def test_graph_torch_rule_manual_and_discriminating_inputs(self):
        import numpy as np
        import torch
        from fx_to_hecate import translate
        from model_catalog import build_model, test_inputs
        from test_fx_to_hecate import interpret_fragment
        for exponent in (2,4):
            case=model(exponent)
            native,shape=build_model(case)
            payload=translate(native,shape)
            self.assertEqual(payload['public_constants'],{'c0':[.5],'c1':[.125]})
            for name in (f'power{exponent}',f'wrong_power{exponent}'):
                wrong=name.startswith('wrong')
                fragment=dict(payload,hecate_source=(BASE/f'golden_cases/explicit_power/{name}.py').read_text())
                detected=False
                for values in test_inputs(shape):
                    expected=reference(exponent,values)
                    np.testing.assert_allclose(native(torch.from_numpy(values)).numpy(),expected,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(interpret_fragment(payload,values),expected,atol=1e-12,rtol=1e-12)
                    actual=interpret_fragment(fragment,values)
                    if wrong: detected |= not np.allclose(actual,expected,atol=1e-5,rtol=1e-4)
                    else: np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
                self.assertEqual(detected,wrong)


@unittest.skipUnless(os.environ.get('POSEIDON_POWER_GOLDEN_REPORT'),'requires real power golden batch')
class PowerEvidenceTests(unittest.TestCase):
    def test_real_compiler_artifacts_references_correct_and_wrong_programs(self):
        import numpy as np
        from audit_agent_lineage import metadata,read
        from hecate_python_env import WORK
        from model_catalog import test_inputs
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        from run_power_goldens import PLANS
        batch,_=metadata(Path(os.environ['POSEIDON_POWER_GOLDEN_REPORT']),WORK/'results')
        self.assertEqual((batch['status'],batch['agent_calls']),('passed',0))
        self.assertEqual(len(batch['cases']),4)
        for row,(case,golden,wrong) in zip(batch['cases'],PLANS):
            self.assertEqual((row['case'],row['counterexample'],row['matched_expected']),(case,wrong,True))
            run=Path(row['run'])
            saved,_=metadata(run/'report.json',WORK/'results')
            self.assertEqual(saved['agent_calls'],0)
            self.assertFalse(saved['llm_generation_validated'])
            self.assertFalse(saved['poseidon_gpu_validated'])
            self.assertEqual(saved['backend'],'upstream_SEAL_HEVM_CPU')
            self.assertEqual(saved['parameters']['security_check'],'tc128')
            self.assertEqual(saved['parameters']['modulus_bits'],[60]*14)
            for path,digest in saved['frozen_hashes'].items(): self.assertEqual(read(run/path,run)[1],digest)
            attempt=saved['attempts'][0]
            output=run/'attempt-00/output'
            payload,_=metadata(run/'attempt-00/trace-payload.json',run)
            request,_=metadata(run/'request.json',run)
            self.assertEqual(payload['request'],request)
            self.assertEqual(payload['candidate']['hecate_source'],(BASE/f'golden_cases/explicit_power/{golden}.py').read_text())
            from candidate_contract import validate_candidate
            self.assertEqual(validate_candidate(payload['candidate'],request),attempt['static_check'])
            self.assertEqual(attempt['trace']['frontend'],'real_Hecate')
            self.assertFalse(attempt['trace']['candidate_python_executed'])
            for path,digest in attempt['artifact_hashes'].items(): self.assertEqual(read(output/path,output)[1],digest)
            self.assertEqual(inspect_artifacts((output/'lowered._hecate_golden.hevm').read_bytes(),
                (output/'_hecate_golden.cst').read_bytes()),attempt['artifact_gate'])
            execution=attempt['execution']
            self.assertTrue(execution['encrypted_execution'])
            self.assertFalse(execution['bootstrap_executed'])
            self.assertEqual(execution['input_batches'],4)
            self.assertTrue(execution['rotation_key_check']['actual_key_file_verified'])
            self.assertTrue(all(m['polynomials']==2 for observation in execution['ciphertext_metadata'] for m in observation['outputs']))
            exponent=int(case[-1])
            original,_=metadata(run/'model.json',run)
            self.assertEqual(original,model(exponent))
            with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['inputs'],test_inputs([4]))
                expected=[reference(exponent,x) for x in arrays['inputs']]
                np.testing.assert_allclose(arrays['reference'],expected,atol=1e-12,rtol=1e-12)
                compared=compare(np.load(output/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
            self.assertEqual(compared,attempt['comparison'])
            self.assertEqual(compared['passed'],not wrong)
            if wrong: self.assertEqual(attempt['failure_layer'],'numerical_comparison')
            self.assertFalse((run/'private-keys').exists())


@unittest.skipUnless(os.environ.get('POSEIDON_POWER_RULE_REPORT'),'requires real power rule baseline')
class PowerRuleEvidenceTests(unittest.TestCase):
    def test_two_rule_programs_independent_reference_and_cleanup(self):
        import numpy as np
        from audit_agent_lineage import metadata,read
        from hecate_python_env import WORK
        from seal_cpu_golden import compare
        path=Path(os.environ['POSEIDON_POWER_RULE_REPORT'])
        report,_=metadata(path,WORK/'results')
        self.assertEqual((report['status'],report['agent_calls']),('passed',0))
        self.assertEqual(report['selected_descriptors'],[model(2),model(4)])
        self.assertEqual(report['parameters']['security_check'],'tc128')
        for row,exponent in zip(report['cases'],(2,4)):
            folder=path.parent/row['folder']
            self.assertEqual(row['status'],'passed')
            self.assertTrue(row['execution']['encrypted_execution'])
            self.assertFalse(row['execution']['bootstrap_executed'])
            for name,digest in row['frozen_hashes'].items(): self.assertEqual(read(folder/name,folder)[1],digest)
            with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays['reference'],[reference(exponent,x) for x in arrays['inputs']],atol=1e-12,rtol=1e-12)
                compared=compare(np.load(folder/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
            self.assertEqual(compared,row['comparison'])
        self.assertFalse((path.parent/'private-keys').exists())


if __name__=='__main__':
    unittest.main()
