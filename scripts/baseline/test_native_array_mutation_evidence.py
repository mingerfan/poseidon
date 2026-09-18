"""Independent audit of saved real native inplace/view FHE experiments."""
import ast
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_ARRAY_MUTATION_GOLDENS') and os.environ.get('POSEIDON_NATIVE_ARRAY_LAYOUT_GOLDENS'),
                     'requires actual native mutation and order-K FHE batches')
class NativeArrayMutationEvidenceTests(unittest.TestCase):
    def test_actual_alias_changes_compiler_artifacts_and_independent_math(self):
        from run_native_array_mutation_goldens import CASES
        self.check_batch('POSEIDON_NATIVE_ARRAY_MUTATION_GOLDENS',CASES,2)

    def test_order_k_constructor_and_ufunc_real_encrypted_results(self):
        from run_native_array_mutation_goldens import LAYOUT_CASES
        self.check_batch('POSEIDON_NATIVE_ARRAY_LAYOUT_GOLDENS',LAYOUT_CASES,0)

    def check_batch(self,environment,cases,negative_count):
        import numpy as np
        from candidate_contract import canonical,strict_json,validate_candidate,request_rotations,request_input_names
        from compiler_configuration import configuration,verify_artifact_configuration
        from deepseek_provider import public_request
        from hecate_python_env import digest
        from native_array_alias import trace_projection
        from native_function_rules import MUTATION_TASK
        from native_public_loops import expand
        from seal_artifact_gate import inspect_artifacts
        from seal_cpu_golden import compare,PROFILE,WATERLINE
        root=Path(os.environ[environment])
        batch=json.loads((root/'report.json').read_text())
        self.assertEqual((batch['status'],batch['generator'],batch['agent_calls']),('passed','manual_golden',0))
        self.assertEqual([Path(r['command'][r['command'].index('--golden-file')+1]).stem for r in batch['cases']],list(cases))
        self.assertEqual([r['counterexample'] for r in batch['cases']],[False]*(len(cases)-negative_count)+[True]*negative_count)
        self.assertEqual(WATERLINE,40)
        for name,row in zip(cases,batch['cases']):
            with self.subTest(name=name):
                run=Path(row['run']);r=json.loads((run/'report.json').read_text())
                self.assertTrue(row['matched_expected'])
                self.assertEqual((r['provider'],r['agent_calls'],r['waterline']),('scripted_replay',0,45))
                self.assertEqual(r['backend'],'upstream_SEAL_HEVM_CPU')
                self.assertFalse(r['llm_generation_validated'] or r['poseidon_gpu_validated'])
                self.assertEqual(r['compiler_configuration'],configuration('seal-cpu-eva-w45-v1'))
                self.assertEqual(r['tolerance'],dict(atol=1e-5,rtol=1e-4))
                p=r['parameters']
                self.assertEqual((p['seal_version'],p['security_check'],p['polynomial_degree']),('4.0.0','tc128',32768))
                self.assertEqual(p['modulus_bits'],[60]*14)
                for file,value in r['frozen_hashes'].items():self.assertEqual(digest(run/file),value)
                req=json.loads((run/'request.json').read_text());self.assertEqual(req['task'],MUTATION_TASK)
                self.assertEqual(public_request(req),req)
                a=r['attempts'][0];folder=run/'attempt-00';out=folder/'output'
                candidate=strict_json((folder/'response.txt').read_text())
                self.assertEqual(json.loads((folder/'trace-payload.json').read_text()),dict(request=req,candidate=candidate))
                self.assertEqual((folder/'candidate.py').read_text(),candidate['hecate_source'])
                checked=validate_candidate(candidate,req)
                self.assertEqual(canonical(checked),canonical(a['static_check']))
                self.assertEqual(json.loads((out/'native-function-plan.json').read_text()),
                                 json.loads(canonical(checked['native_functions'])))
                required={'native-function-plan.json','native-call-events.json','native-array-mutation-events.json',
                    'trace-evidence.json','candidate_trace.mlir','lowered.ckks.mlir','lowered._hecate_golden.hevm','_hecate_golden.cst'}
                self.assertTrue(required<=set(a['artifact_hashes']))
                for file,value in a['artifact_hashes'].items():self.assertEqual(digest(out/file),value)
                artifact=inspect_artifacts((out/'lowered._hecate_golden.hevm').read_bytes(),(out/'_hecate_golden.cst').read_bytes(),
                    rotation_steps=request_rotations(req),expected_inputs=len(request_input_names(req)))
                self.assertEqual(canonical(artifact),canonical(a['artifact_gate']))
                verify_artifact_configuration(req,artifact,digest(PROFILE))
                tree,loops=expand(ast.parse(candidate['hecate_source']),req['public_constants'],scalar_augmented=True)
                self.assertEqual(loops,checked['native_functions']['public_loops'])
                functions={f.name for f in tree.body}
                calls=[dict(caller=f.name,callee=n.func.id,span=[n.lineno,n.col_offset,n.end_lineno,n.end_col_offset])
                    for f in tree.body for n in ast.walk(f) if type(n) is ast.Call and type(n.func) is ast.Name and n.func.id in functions]
                self.assertCountEqual(json.loads((out/'native-call-events.json').read_text()),calls)
                sites=checked['native_functions']['array_mutation']['sites'];self.assertTrue(sites)
                events=json.loads((out/'native-array-mutation-events.json').read_text())
                self.assertCountEqual(events,json.loads(canonical(trace_projection(sites))))
                # Python wrapper classes do NOT expose actual native p/c IR types.
                self.assertTrue(all('kinds' not in binding for event in events for binding in event['before']))
                if name=='empty':self.assertTrue(all(not r['indices'] for e in events for r in e['changed_cells']))
                else:self.assertTrue(any(r['indices'] for e in events for r in e['changed_cells']))
                if name=='callee_fresh':
                    self.assertEqual({r['name']:r['kinds'] for r in sites[0]['after']},dict(first=['c'],second=['p']))
                    self.assertIn('public_identity',[r['callee'] for r in calls])
                self.assertTrue(a['compiled'] and a['executed'])
                self.assertFalse(a['trace']['candidate_python_executed'])
                self.assertEqual(json.loads((out/'trace-evidence.json').read_text()),a['trace'])
                execution=json.loads((out/'execution.json').read_text());self.assertEqual(execution,a['execution'])
                self.assertTrue(execution['encrypted_execution']);self.assertFalse(execution['bootstrap_executed'])
                self.assertEqual((execution['input_batches'],execution['encrypted_input_count']),(4,1))
                with np.load(run/'arrays.npz',allow_pickle=False) as data:inputs,reference=data['inputs'],data['reference']
                np.testing.assert_allclose(reference,inputs*1.5+.375,atol=1e-15,rtol=0)
                actual=np.load(out/'decrypted.npy',allow_pickle=False)
                comparison=compare(actual,reference,1e-5,1e-4)
                self.assertEqual(comparison,a['comparison'])
                self.assertEqual(comparison['passed'],not row['counterexample'])
                if row['counterexample']:
                    self.assertEqual(a['failure_layer'],'numerical_comparison')
                    wrong=inputs*2+.375 if name=='wrong_alias' else inputs*5.5+.375
                    self.assertTrue(compare(actual,wrong,1e-5,1e-4)['passed'])
                    self.assertGreater(comparison['max_absolute_error'],.49)
                self.assertFalse((run/'private-keys').exists())
                self.assertTrue(json.loads((run/'key-cleanup-outcome.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
