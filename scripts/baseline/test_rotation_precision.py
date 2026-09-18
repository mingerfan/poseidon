"""Re-audit the bounded direct-SEAL diagnostic, not a replacement runtime."""
import json
import math
import os
from pathlib import Path
import unittest

@unittest.skipUnless(os.environ.get('POSEIDON_ROTATION_PRECISION_DIAGNOSTIC'),'requires completed precision diagnostic')
class RotationPrecisionEvidenceTests(unittest.TestCase):
    def report(self):
        root=Path(os.environ['POSEIDON_ROTATION_PRECISION_DIAGNOSTIC'])
        return root,json.loads((root/'report.json').read_text())

    def test_fixed_scope_producer_hashes_and_stage_arithmetic(self):
        from hecate_python_env import digest
        root,r=self.report()
        self.assertEqual(r['status'],'diagnostic_completed')
        self.assertEqual((r['trials'],r['input_vectors'],r['scale_log2'],r['polynomial_degree']), (2,3,40,32768))
        self.assertEqual(r['modulus_bits'],[60]*14)
        self.assertEqual((r['security'],r['seal_version']),('tc128','4.0.0'))
        for name in ('original_failed_key_reproduced','keys_written_to_disk','compiler_parameters_changed',
                     'threshold_changed','registered_execution_backend'):
            self.assertFalse(r[name])
        self.assertEqual(r['agent_calls'],0)
        for path,value in r['source_hashes'].items(): self.assertEqual(digest(Path(path)),value)
        self.assertEqual(digest(root/'stages.jsonl'),r['stages_sha256'])
        rows=[json.loads(x) for x in (root/'stages.jsonl').read_text().splitlines()]
        self.assertEqual(rows,r['rows'])
        self.assertEqual(len(rows),72)
        self.assertEqual(len({(v['trial'],v['batch'],v['stage']) for v in rows}),72)
        for v in rows:
            self.assertEqual(v['log2_scale'],40)
            self.assertTrue(all(math.isfinite(e) for e in v['error_first4']))
            self.assertEqual(v['max_abs_first4'],max(abs(e) for e in v['error_first4']))
            self.assertAlmostEqual(v['mae_first4'],sum(abs(e) for e in v['error_first4'])/4,places=18)
            self.assertGreaterEqual(v['max_abs_all_slots'],v['max_abs_first4'])
        maxima={s:max(v['max_abs_first4'] for v in rows if v['stage']==s) for s in r['max_abs_first4_by_stage']}
        self.assertEqual(maxima,r['max_abs_first4_by_stage'])

    def test_observed_error_appears_at_rotation_not_modswitch(self):
        _,r=self.report()
        rows={(v['trial'],v['batch'],v['stage']):v for v in r['rows']}
        for trial in range(2):
            for batch in range(3):
                a=rows[trial,batch,'encrypted_input'];b=rows[trial,batch,'modswitch_to_level1']
                self.assertEqual((a['data_moduli'],b['data_moduli']),(13,1))
                for x,y in zip(a['error_first4'],b['error_first4']): self.assertAlmostEqual(x,y,places=12)
                a=rows[trial,batch,'sum_level13'];b=rows[trial,batch,'sum_high_then_modswitch']
                for x,y in zip(a['error_first4'],b['error_first4']): self.assertAlmostEqual(x,y,places=12)
        m=r['max_abs_first4_by_stage']
        self.assertLess(m['encode_decode'],1e-10)
        self.assertLess(m['encrypted_input'],1e-7)
        self.assertGreater(max(m['rotate_level1_'+str(i)] for i in (1,2,3)),m['encrypted_input']*100)
        self.assertGreater(m['sum_level13'],m['sum_level1'])
        # This establishes this diagnostic's behavior, not a stochastic bound.

if __name__=='__main__': unittest.main()
