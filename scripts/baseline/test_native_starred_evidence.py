"""Audit saved real starred-call goldens; never call a model or rerun FHE."""
import ast
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_STARRED_GOLDENS'),'requires actual starred-call golden evidence')
class NativeStarredEvidenceTests(unittest.TestCase):
    def test_response_trace_arguments_artifacts_and_independent_reference(self):
        import numpy as np
        from candidate_contract import canonical,strict_json,validate_candidate
        from deepseek_provider import public_request
        from hecate_python_env import digest
        from native_function_rules import STAR_TASK
        from run_native_starred_goldens import CASES
        from seal_cpu_golden import compare
        root=Path(os.environ['POSEIDON_NATIVE_STARRED_GOLDENS'])
        report=json.loads((root/'report.json').read_text())
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['generator'],'manual_golden')
        self.assertEqual(report['agent_calls'],0)
        self.assertEqual([Path(r['command'][r['command'].index('--golden-file')+1]).stem for r in report['cases']],list(CASES))
        self.assertEqual([r['counterexample'] for r in report['cases']],[False]*8+[True])
        for row in report['cases']:
            with self.subTest(run=row['run']):
                run=Path(row['run']);result=json.loads((run/'report.json').read_text())
                self.assertTrue(row['matched_expected'])
                self.assertEqual(result['agent_calls'],0)
                self.assertFalse(result['llm_generation_validated'])
                self.assertEqual(result['provider'],'scripted_replay')
                self.assertEqual(result['backend'],'upstream_SEAL_HEVM_CPU')
                self.assertFalse(result['poseidon_gpu_validated'])
                self.assertEqual(result['tolerance'],dict(atol=1e-5,rtol=1e-4))
                self.assertEqual(result['parameters']['security_check'],'tc128')
                self.assertEqual(result['parameters']['polynomial_degree'],32768)
                self.assertEqual(result['parameters']['modulus_bits'],[60]*14)
                for name,value in result['frozen_hashes'].items(): self.assertEqual(digest(run/name),value)
                request=json.loads((run/'request.json').read_text())
                self.assertEqual(request['task'],STAR_TASK);self.assertEqual(public_request(request),request)
                attempt=result['attempts'][0];folder=run/'attempt-00';out=folder/'output'
                payload=json.loads((folder/'trace-payload.json').read_text())
                candidate=strict_json((folder/'response.txt').read_text())
                self.assertEqual(payload,dict(request=request,candidate=candidate))
                self.assertEqual((folder/'candidate.py').read_text(),candidate['hecate_source'])
                checked=validate_candidate(candidate,request)
                self.assertEqual(canonical(checked),canonical(attempt['static_check']))
                required={'native-function-plan.json','native-call-events.json','trace-evidence.json',
                          'candidate_trace.mlir','lowered.ckks.mlir','lowered._hecate_golden.hevm','_hecate_golden.cst'}
                self.assertTrue(required <= set(attempt['artifact_hashes']))
                for name,value in attempt['artifact_hashes'].items(): self.assertEqual(digest(out/name),value)
                self.assertEqual(json.loads((out/'native-function-plan.json').read_text()),
                                 json.loads(canonical(checked['native_functions'])))
                tree=ast.parse(candidate['hecate_source']);names={n.name for n in tree.body}
                expected=[dict(caller=fn.name,callee=n.func.id,span=[n.lineno,n.col_offset,n.end_lineno,n.end_col_offset])
                          for fn in tree.body for n in ast.walk(fn)
                          if type(n) is ast.Call and type(n.func) is ast.Name and n.func.id in names]
                self.assertTrue(any(type(n) is ast.Starred for n in ast.walk(tree)))
                self.assertCountEqual(json.loads((out/'native-call-events.json').read_text()),expected)
                self.assertTrue(attempt['compiled'] and attempt['executed'])
                trace=json.loads((out/'trace-evidence.json').read_text())
                self.assertEqual(trace,attempt['trace']);self.assertFalse(trace['candidate_python_executed'])
                self.assertEqual(trace['frontend'],'real_Hecate')
                execution=json.loads((out/'execution.json').read_text());self.assertEqual(execution,attempt['execution'])
                self.assertTrue(execution['encrypted_execution']);self.assertFalse(execution['bootstrap_executed'])
                self.assertEqual((execution['input_batches'],execution['encrypted_input_count']),(4,1))
                with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
                    reference,inputs=arrays['reference'],arrays['inputs']
                np.testing.assert_allclose(reference,inputs*.5+inputs+.375,atol=1e-15,rtol=0)
                comparison=compare(np.load(out/'decrypted.npy',allow_pickle=False),reference,1e-5,1e-4)
                self.assertEqual(comparison,attempt['comparison'])
                self.assertEqual(comparison['passed'],not row['counterexample'])
                if row['counterexample']: self.assertEqual(attempt['failure_layer'],'numerical_comparison')
                self.assertFalse((run/'private-keys').exists())
                self.assertTrue(json.loads((run/'key-cleanup-outcome.json').read_text())['complete'])


if __name__=='__main__': unittest.main()
