"""Saved native loop evidence: real FHE, independent sums and bound counterexample."""
import ast
import json
import os
from pathlib import Path
import unittest

@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_LOOP_GOLDENS'),'requires completed native loop FHE evidence')
class NativeLoopEvidenceTests(unittest.TestCase):
    def test_expanded_calls_real_artifacts_and_independent_reductions(self):
        import numpy as np
        from candidate_contract import canonical,strict_json,validate_candidate
        from deepseek_provider import public_request
        from hecate_python_env import digest
        from native_function_rules import LOOP_TASK
        from native_public_loops import expand
        from run_native_loop_goldens import CASES
        from seal_cpu_golden import compare
        root=Path(os.environ['POSEIDON_NATIVE_LOOP_GOLDENS'])
        batch=json.loads((root/'report.json').read_text())
        # Freeze the actual partial result. A green evidence audit must NOT
        # relabel the correct sum4 program's numerical failure as acceptance.
        self.assertEqual((batch['status'],batch['generator'],batch['agent_calls']),('failed','manual_golden',0))
        self.assertEqual([Path(r['command'][r['command'].index('--golden-file')+1]).stem for r in batch['cases']],list(CASES))
        self.assertEqual([r['counterexample'] for r in batch['cases']],[False]*8+[True])
        for name,row in zip(CASES,batch['cases']):
            with self.subTest(name=name):
                run=Path(row['run']);report=json.loads((run/'report.json').read_text())
                self.assertEqual(row['matched_expected'],name!='sum4')
                self.assertEqual(report['provider'],'scripted_replay')
                self.assertEqual(report['agent_calls'],0)
                self.assertFalse(report['llm_generation_validated'] or report['poseidon_gpu_validated'])
                self.assertEqual(report['backend'],'upstream_SEAL_HEVM_CPU')
                self.assertEqual(report['tolerance'],dict(atol=1e-5,rtol=1e-4))
                p=report['parameters']
                self.assertEqual((p['seal_version'],p['security_check'],p['polynomial_degree']),('4.0.0','tc128',32768))
                self.assertEqual(p['modulus_bits'],[60]*14)
                for file,value in report['frozen_hashes'].items(): self.assertEqual(digest(run/file),value)
                request=json.loads((run/'request.json').read_text())
                self.assertEqual(request['task'],LOOP_TASK)
                self.assertEqual(public_request(request),request)
                attempt=report['attempts'][0];folder=run/'attempt-00';out=folder/'output'
                payload=json.loads((folder/'trace-payload.json').read_text())
                candidate=strict_json((folder/'response.txt').read_text())
                self.assertEqual(payload,dict(request=request,candidate=candidate))
                self.assertEqual((folder/'candidate.py').read_text(),candidate['hecate_source'])
                checked=validate_candidate(candidate,request)
                self.assertEqual(canonical(checked),canonical(attempt['static_check']))
                required={'native-function-plan.json','native-call-events.json','trace-evidence.json',
                          'candidate_trace.mlir','lowered.ckks.mlir','lowered._hecate_golden.hevm','_hecate_golden.cst'}
                self.assertTrue(required<=set(attempt['artifact_hashes']))
                for file,value in attempt['artifact_hashes'].items(): self.assertEqual(digest(out/file),value)
                self.assertEqual(json.loads((out/'native-function-plan.json').read_text()),
                                 json.loads(canonical(checked['native_functions'])))
                tree,loops=expand(ast.parse(candidate['hecate_source']),request['public_constants'])
                self.assertEqual(loops,checked['native_functions']['public_loops'])
                self.assertFalse(any(type(n) is ast.For for n in ast.walk(tree)))
                names={f.name for f in tree.body}
                expected=[dict(caller=fn.name,callee=n.func.id,span=[n.lineno,n.col_offset,n.end_lineno,n.end_col_offset])
                          for fn in tree.body for n in ast.walk(fn)
                          if type(n) is ast.Call and type(n.func) is ast.Name and n.func.id in names]
                self.assertCountEqual(json.loads((out/'native-call-events.json').read_text()),expected)
                self.assertTrue(attempt['compiled'] and attempt['executed'])
                trace=json.loads((out/'trace-evidence.json').read_text())
                self.assertEqual(trace,attempt['trace'])
                self.assertFalse(trace['candidate_python_executed'])
                self.assertEqual(trace['frontend'],'real_Hecate')
                execution=json.loads((out/'execution.json').read_text())
                self.assertEqual(execution,attempt['execution'])
                self.assertTrue(execution['encrypted_execution'])
                self.assertFalse(execution['bootstrap_executed'])
                self.assertEqual((execution['input_batches'],execution['encrypted_input_count']),(4,1))
                with np.load(run/'arrays.npz',allow_pickle=False) as data:
                    inputs,reference=data['inputs'],data['reference']
                if name in ('sum4','descending','wrong_bound'):
                    width=4 if name=='sum4' else 3
                    independent=sum(np.roll(inputs,-step,axis=-1) for step in range(width))
                    self.assertEqual(checked['rotation_steps'],[1,2,3] if name!='descending' else [1,2])
                else: independent=inputs*1.5+.375
                np.testing.assert_allclose(reference,independent,atol=1e-15,rtol=0)
                actual=np.load(out/'decrypted.npy',allow_pickle=False)
                comparison=compare(actual,reference,1e-5,1e-4)
                self.assertEqual(comparison,attempt['comparison'])
                self.assertEqual(comparison['passed'],name not in ('sum4','wrong_bound'))
                if name=='sum4':
                    self.assertEqual(attempt['failure_layer'],'numerical_comparison')
                    self.assertGreater(comparison['max_absolute_error'],1e-5)
                    self.assertFalse(comparison['elementwise_pass'][0][0])
                if row['counterexample']:
                    self.assertEqual(attempt['failure_layer'],'numerical_comparison')
                    wrong=sum(np.roll(inputs,-step,axis=-1) for step in range(4))
                    # The wrong program also crosses the precision gate relative
                    # to its own mathematical formula. Preserve both failures;
                    # never increase atol to make the diagnostic comparison green.
                    residual=compare(actual,wrong,1e-5,1e-4)
                    self.assertFalse(residual['passed'])
                    self.assertGreater(residual['max_absolute_error'],1e-5)
                    self.assertEqual(float(np.max(np.abs(wrong-reference))),1.0)
                    np.testing.assert_allclose(actual-reference,
                        (wrong-reference)+(actual-wrong),atol=1e-15,rtol=0)
                self.assertFalse((run/'private-keys').exists())
                self.assertTrue(json.loads((run/'key-cleanup-outcome.json').read_text())['complete'])

if __name__=='__main__': unittest.main()
