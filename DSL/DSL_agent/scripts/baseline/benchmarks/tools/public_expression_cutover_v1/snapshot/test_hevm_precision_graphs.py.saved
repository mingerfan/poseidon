"""Audit paired multi-graph precision evidence without API or FHE execution."""
import ast
import json
import os
from pathlib import Path
import unittest

import probe_hevm_precision_graphs as probe


class PrecisionGraphPlanTests(unittest.TestCase):
    def test_fixed_nonadaptive_scope(self):
        self.assertEqual((probe.SCALES, probe.TRIALS), ((40,45),3))
        self.assertEqual([s[0] for s in probe.SOURCES],
                         ['square','linear','mlp2','mlp3','fanout','residual','wide_mlp8','sum4'])
        self.assertEqual(len({s[1] for s in probe.SOURCES}),8)
        self.assertTrue(all(len(s[2]) == 64 for s in probe.SOURCES))
        self.assertEqual(probe.WATERLINE,40)
        tree=ast.parse(Path(probe.__file__).read_text())
        self.assertFalse(any(type(n) is ast.While for n in ast.walk(tree)))
        self.assertFalse(any(type(n) is ast.Call and type(n.func) is ast.Name and
                             n.func.id in ('exec','eval') for n in ast.walk(tree)))

    def test_summary_keeps_failures_and_fixed_denominator(self):
        c=dict(compared_values=8,passed=False,max_absolute_error=.1,mae=.01)
        r=dict(compiled=[dict(status='compile_failed',waterline=45)],
               trials=[dict(executions=[dict(waterline=40,status='numerical_failure',comparison=c),
                                       dict(waterline=45,status='execution_failed')])])
        a,b=probe.summarize(r)
        self.assertEqual((a['planned_program_executions'],a['executed'],a['passed']),(24,1,0))
        self.assertEqual((a['compared_values'],a['max_absolute_error']),(8,.1))
        self.assertEqual((b['planned_program_executions'],b['executed'],b['compile_failures'],
                          b['execution_failures']),(24,0,1,1))
        self.assertIsNone(b['mae'])
        self.assertIsNone(b['max_absolute_error'])


@unittest.skipUnless(os.environ.get('POSEIDON_HEVM_GRAPH_PRECISION'), 'requires completed 8-graph diagnostic')
class PrecisionGraphEvidenceTests(unittest.TestCase):
    def test_sources_frozen_and_independent_reference(self):
        import numpy as np
        from hecate_python_env import digest
        root=Path(os.environ['POSEIDON_HEVM_GRAPH_PRECISION'])
        r=json.loads((root/'report.json').read_text())
        self.assertEqual(r['status'],'diagnostic_completed')
        self.assertEqual(r['source_specs'],[list(s) for s in probe.SOURCES])
        self.assertEqual((r['waterlines'],r['planned_key_sets']),([40,45],3))
        self.assertEqual((r['agent_calls'],r['production_waterline']),(0,40))
        self.assertEqual(r['threshold'],dict(atol=1e-5,rtol=1e-4))
        for name in ('production_profile_changed','threshold_changed','llm_generation_validated',
                     'poseidon_gpu_validated','all_semantics_proven'):
            self.assertFalse(r[name])
        for field in ('source_hashes','runtime_hashes'):
            for name,value in r[field].items(): self.assertEqual(digest(Path(name)),value)
        for name,value in r['producer_snapshots'].items(): self.assertEqual(digest(root/name),value)
        self.assertEqual(len(r['sources']),8)
        for spec, source in zip(probe.SOURCES,r['sources']):
            original,_,_,_=probe.checked_source(spec)
            self.assertEqual(probe.security_parameters(original['parameters']),r['security_parameters'])
            self.assertEqual(source['originally_passed'],spec[0] != 'sum4')
            for name,value in source['hashes'].items(): self.assertEqual(digest(root/name),value)
            inputs,reference,graph=probe.reference_for(Path(source['run']))
            with np.load(root/source['name']/'reference-arrays.npz',allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['inputs'],inputs)
                np.testing.assert_array_equal(arrays['reference'],reference)
            self.assertEqual(json.loads((root/source['name']/'independent-model.json').read_text()),graph)

    def test_actual_compiler_runtime_metadata_same_keys_and_numerics(self):
        import numpy as np
        from hecate_python_env import digest
        from seal_artifact_gate import inspect_artifacts
        from seal_cpu_golden import compare
        root=Path(os.environ['POSEIDON_HEVM_GRAPH_PRECISION'])
        r=json.loads((root/'report.json').read_text())
        self.assertEqual(len(r['compiled']),16)
        expected={(s[0],scale) for s in probe.SOURCES for scale in probe.SCALES}
        self.assertEqual({(t['name'],t['waterline']) for t in r['compiled']},expected)
        templates={}
        for t in r['compiled']:
            self.assertEqual(t['status'],'compiled')
            out=Path(t['directory'])
            for name,value in t['hashes'].items(): self.assertEqual(digest(out/name),value)
            gate=inspect_artifacts((out/'lowered._hecate_golden.hevm').read_bytes(),
                                  (out/'_hecate_golden.cst').read_bytes(),rotation_steps=probe.ROTATIONS)
            self.assertEqual(gate,t['gate'])
            self.assertEqual((gate['arg_scale'],gate['arg_level']),([t['waterline']],[13]))
            templates[t['name'],t['waterline']]=t
        self.assertEqual(len(r['trials']),3)
        fingerprints=[]
        for trial in r['trials']:
            run=Path(trial['run']); saved=json.loads((run/'report.json').read_text())
            self.assertEqual(saved['executions'],trial['executions'])
            self.assertEqual(probe.security_parameters(saved['parameters']),r['security_parameters'])
            self.assertEqual(saved['parameters']['rotation_steps'],list(probe.ROTATIONS))
            fingerprints.append(tuple(sorted(saved['same_key_set_hashes'].items())))
            self.assertFalse((run/'private-keys').exists())
            cleanup=json.loads((run/'key-cleanup-outcome.json').read_text())
            self.assertTrue(cleanup['complete']); self.assertEqual(cleanup,trial['cleanup'])
            self.assertEqual(len(trial['executions']),16)
            self.assertEqual({(v['name'],v['waterline']) for v in trial['executions']},expected)
            for item in trial['executions']:
                self.assertIn(item['status'],('passed','numerical_failure'))
                name,scale=item['name'],item['waterline'];out=Path(item['directory'])
                t=templates[name,scale];gate=t['gate']
                for fname,value in t['hashes'].items(): self.assertEqual(digest(out/fname),value)
                with np.load(root/name/'reference-arrays.npz',allow_pickle=False) as frozen:
                    inputs,reference=frozen['inputs'],frozen['reference']
                with np.load(out/'arrays.npz',allow_pickle=False) as data:
                    self.assertEqual(data.files,['inputs'])
                    np.testing.assert_array_equal(data['inputs'],inputs)
                for filename,field in (('decrypted.npy','decrypted_sha256'),('execution.json','execution_sha256')):
                    self.assertEqual(digest(out/filename),item[field])
                comparison=compare(np.load(out/'decrypted.npy',allow_pickle=False),reference,1e-5,1e-4)
                self.assertEqual(comparison,item['comparison'])
                self.assertEqual(item['status']=='passed',comparison['passed'])
                e=json.loads((out/'execution.json').read_text()); self.assertEqual(e,item['execution'])
                self.assertTrue(e['encrypted_execution']); self.assertFalse(e['bootstrap_executed'])
                self.assertEqual((e['input_batches'],e['encrypted_input_count']),(4,1))
                self.assertTrue(e['rotation_key_check']['actual_key_file_verified'])
                self.assertEqual(e['rotation_key_check']['required_steps'],gate['rotation_steps'])
                for batch in e['ciphertext_metadata']:
                    self.assertEqual((batch['input']['log2_scale'],batch['input']['data_modulus_count']),(scale,13))
                    self.assertEqual(len(batch['outputs']),len(gate['res_scale']))
                    for i,value in enumerate(batch['outputs']):
                        self.assertAlmostEqual(value['log2_scale'],gate['res_scale'][i],places=6)
                        self.assertEqual(value['data_modulus_count'],gate['res_level'][i])
                        self.assertEqual(value['polynomials'],2)
        self.assertEqual(len(set(fingerprints)),3)
        self.assertEqual(r['summary'],probe.summarize(r))
        self.assertEqual(sum(s['compared_values'] for s in r['summary']),576)
        self.assertEqual(sum(s['input_executions'] for s in r['summary']),192)


if __name__ == '__main__':
    unittest.main()
