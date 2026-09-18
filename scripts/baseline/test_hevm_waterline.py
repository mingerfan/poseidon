"""Offline checks for the bounded waterline experiment; no API/runtime calls."""
import ast
from pathlib import Path
import struct
import unittest
import probe_hevm_waterline as probe

class WaterlinePlanTests(unittest.TestCase):
    def test_fixed_nonadaptive_scope_and_unchanged_default(self):
        from seal_cpu_golden import WATERLINE
        self.assertEqual(probe.SCALES,(40,45,50))
        self.assertEqual(probe.TRIALS,3)
        self.assertEqual(WATERLINE,40)
        self.assertEqual(len(probe.SOURCE_REPORT_SHA),64)
        tree=ast.parse(Path(probe.__file__).read_text())
        self.assertFalse(any(type(n) is ast.While for n in ast.walk(tree)))
        self.assertFalse(any(type(n) is ast.Call and type(n.func) is ast.Name and
                             n.func.id in ('eval','exec') for n in ast.walk(tree)))

    def test_hevm_instruction_comparison_ignores_only_uninitialized_empty(self):
        def raw(scale,padding,offset=1):
            code=[(65535,*padding),(1,1,0,offset),(6,2,0,1)]
            header=struct.pack('<IIQQ',0x4845564d,24,1,1)
            body=struct.pack('<10Q',80,len(code),3,0,13,scale,13,scale,1,2)
            return header+body+b''.join(struct.pack('<4H',*op) for op in code)
        a=probe.semantic_instructions(raw(40,(1,2,3)))
        b=probe.semantic_instructions(raw(50,(4,5,6)))
        self.assertEqual(a,b)
        self.assertNotEqual(a,probe.semantic_instructions(raw(50,(4,5,6),offset=2)))

@unittest.skipUnless(__import__('os').environ.get('POSEIDON_HEVM_WATERLINE_DIAGNOSTIC'),
                     'requires completed controlled HEVM precision experiment')
class WaterlineEvidenceTests(unittest.TestCase):
    def test_matched_keys_identical_program_frozen_reference_and_actual_scale(self):
        import json
        import os
        import numpy as np
        from hecate_python_env import digest
        from seal_artifact_gate import inspect_artifacts
        from seal_cpu_golden import compare,WATERLINE
        root=Path(os.environ['POSEIDON_HEVM_WATERLINE_DIAGNOSTIC'])
        report=json.loads((root/'report.json').read_text())
        self.assertEqual(report['status'],'diagnostic_completed')
        self.assertEqual((report['planned_key_sets'],report['waterlines']),(3,[40,45,50]))
        self.assertEqual(report['agent_calls'],0)
        self.assertTrue(report['original_failure_preserved'])
        for field in ('production_profile_changed','threshold_changed','llm_generation_validated',
                      'poseidon_gpu_validated','all_semantics_proven'):
            self.assertFalse(report[field])
        self.assertEqual(WATERLINE,40)
        self.assertEqual(report['threshold'],dict(atol=1e-5,rtol=1e-4))
        for field in ('source_hashes','runtime_hashes'):
            for path,value in report[field].items(): self.assertEqual(digest(Path(path)),value)
        for path,value in report['producer_snapshots'].items(): self.assertEqual(digest(root/path),value)
        self.assertEqual(digest(root/'reference-arrays.npz'),report['reference_arrays_sha256'])
        self.assertEqual(digest(root/'trace-payload.json'),report['payload_sha256'])
        original=json.loads((Path(report['source_run'])/'report.json').read_text())
        self.assertFalse(original['attempts'][0]['comparison']['passed'])
        with np.load(root/'reference-arrays.npz',allow_pickle=False) as data:
            inputs,reference=data['inputs'],data['reference']
        np.testing.assert_allclose(reference,sum(np.roll(inputs,-i,axis=-1) for i in range(4)),atol=1e-15,rtol=0)
        templates={t['waterline']:t for t in report['compiled']}
        op_sequences=[]
        for scale,t in templates.items():
            folder=Path(t['directory'])
            for name,value in t['hashes'].items(): self.assertEqual(digest(folder/name),value)
            raw=(folder/'lowered._hecate_golden.hevm').read_bytes()
            gate=inspect_artifacts(raw,(folder/'_hecate_golden.cst').read_bytes(),rotation_steps=(-3,-2,-1,1,2,3))
            self.assertEqual(gate,t['gate'])
            self.assertEqual((gate['arg_scale'],gate['res_scale']),([scale],[scale]))
            self.assertEqual((gate['arg_level'],gate['res_level']),([13],[1]))
            sequence=probe.semantic_instructions(raw);op_sequences.append(sequence)
            self.assertEqual([list(op) for op in sequence],t['semantic_instructions'])
        self.assertTrue(all(ops==op_sequences[0] for ops in op_sequences))
        self.assertEqual(len(report['trials']),3)
        fingerprints=[]
        for trial in report['trials']:
            run=Path(trial['run']);saved=json.loads((run/'report.json').read_text())
            self.assertEqual(saved['scales'],trial['scales'])
            self.assertEqual(saved['parameters'],original['parameters'])
            self.assertEqual([s['waterline'] for s in trial['scales']],[40,45,50])
            fingerprints.append(tuple(sorted(saved['same_key_set_hashes'].items())))
            self.assertFalse((run/'private-keys').exists())
            cleanup=json.loads((run/'key-cleanup-outcome.json').read_text())
            self.assertTrue(cleanup['complete']);self.assertEqual(cleanup,trial['cleanup'])
            for item in trial['scales']:
                out=Path(item['directory']);scale=item['waterline']
                for name,value in templates[scale]['hashes'].items(): self.assertEqual(digest(out/name),value)
                with np.load(out/'arrays.npz',allow_pickle=False) as data:
                    self.assertEqual(data.files,['inputs'])
                    np.testing.assert_array_equal(data['inputs'],inputs)
                self.assertEqual(digest(out/'decrypted.npy'),item['decrypted_sha256'])
                actual=np.load(out/'decrypted.npy',allow_pickle=False)
                self.assertEqual(compare(actual,reference,1e-5,1e-4),item['comparison'])
                self.assertEqual(digest(out/'execution.json'),item['execution_sha256'])
                e=json.loads((out/'execution.json').read_text());self.assertEqual(e,item['execution'])
                self.assertTrue(e['encrypted_execution']);self.assertFalse(e['bootstrap_executed'])
                self.assertEqual((e['input_batches'],e['encrypted_input_count']),(4,1))
                self.assertTrue(e['rotation_key_check']['actual_key_file_verified'])
                self.assertEqual(e['rotation_key_check']['required_steps'],[1,2,3])
                for batch in e['ciphertext_metadata']:
                    self.assertEqual((batch['input']['log2_scale'],batch['input']['data_modulus_count']),(scale,13))
                    self.assertEqual((batch['outputs'][0]['log2_scale'],batch['outputs'][0]['data_modulus_count']),(scale,1))
        self.assertEqual(len(set(fingerprints)),3)

    def test_recomputed_summary_and_finite_precision_improvement_only(self):
        import json
        import os
        root=Path(os.environ['POSEIDON_HEVM_WATERLINE_DIAGNOSTIC'])
        r=json.loads((root/'report.json').read_text())
        for summary in r['summary']:
            values=[s['comparison'] for t in r['trials'] for s in t['scales'] if s['waterline']==summary['waterline']]
            self.assertEqual(summary['passed_key_sets'],sum(v['passed'] for v in values))
            self.assertEqual((summary['key_sets'],summary['input_executions'],summary['compared_values']),(3,12,48))
            self.assertEqual(summary['max_absolute_error'],max(v['max_absolute_error'] for v in values))
            self.assertEqual(summary['mae'],sum(v['mae']*v['compared_values'] for v in values)/48)
        for t in r['trials']:
            c={s['waterline']:s['comparison'] for s in t['scales']}
            self.assertTrue(c[45]['passed'] and c[50]['passed'])
            self.assertLess(c[45]['max_absolute_error'],c[40]['max_absolute_error']/10)
            self.assertLess(c[50]['max_absolute_error'],c[45]['max_absolute_error']/10)
        self.assertFalse(r['all_semantics_proven'])

if __name__=='__main__': unittest.main()
