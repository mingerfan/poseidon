"""Offline checks over retained real artifacts; does not re-execute ciphertexts."""
import json,struct,unittest
from pathlib import Path
from compiler_artifact_evidence import lineage,audit
from hecate_python_env import WORK

class SyntheticLineageTests(unittest.TestCase):
    def artifact(self,operations,result_scale=20,result_level=12,result_dst=2):
        from benchmark_suite import Builder
        from unified_graph_contract import prepare
        from compiler_configuration import PROFILE_SHA256
        b=Builder([(4,)]);model=b.finish(b.node('multiply',['input0',b.const(2.)]))
        request=prepare(model,PROFILE_SHA256)
        raw=struct.pack('<IIQQ',0x4845564D,24,2,1)+struct.pack('<5Q',96,len(operations),3,1,13)
        raw+=struct.pack('<7Q',40,40,13,13,result_scale,result_level,result_dst)
        raw+=b''.join(struct.pack('<4H',*op) for op in operations)
        cst=struct.pack('<qqd',1,1,2.)
        return raw,cst,request
    def test_rehashed_foreign_profile_is_rejected(self):
        from benchmark_graph import digest
        raw,cst,r=self.artifact([(0,0,0,(13<<10)|40),(9,2,0,0),(3,2,2,0)])
        r['compiler_profile_sha256']='0'*64;r['request_id']=digest({k:v for k,v in r.items() if k!='request_id'})
        with self.assertRaisesRegex(ValueError,'fixed 60-bit profile'):lineage(raw,cst,r)
    def test_cipher_plain_multiply_then_rescale_nominal_units(self):
        value=lineage(*self.artifact([(0,0,0,(13<<10)|40),(9,2,0,0),(3,2,2,0)]))
        self.assertEqual(value['events'][1]['output']['scale'],80)
        self.assertEqual(value['outputs'][0]['scale'],20)
        self.assertEqual(value['outputs'][0]['level'],12)
        self.assertFalse(value['error_bound_proven'])
    def test_modswitch_cannot_impersonate_rescale(self):
        with self.assertRaisesRegex(ValueError,'scale lineage mismatch'):
            lineage(*self.artifact([(0,0,0,(13<<10)|40),(9,2,0,0),(4,2,2,1)]))
    def test_stock_add_scale_assignment_updates_aliased_source_register(self):
        # Runtime addcp mutates LHS scale even when destination is another register.
        value=lineage(*self.artifact([(0,0,0,(13<<10)|30),(7,2,0,0)],30,13,0))
        self.assertEqual(value['outputs'][0]['scale'],30)
        self.assertEqual(value['scale_metadata_overrides'],[dict(index=1,before=40,after=30)])

class CompilerEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.failed=WORK/'results/candidate-replay-g2fr2jh4'
        cls.controls=WORK/'results/upstream-concat-r18-controls/report.json'
        if not cls.failed.is_dir() or not cls.controls.exists():raise unittest.SkipTest('Requires retained r18 real artifact evidence')
        from hecate_python_env import ROOT
        import hashlib
        hashes=json.loads((cls.failed/'report.json').read_text())['source_hashes']
        if any(hashlib.sha256((ROOT/'scripts/baseline'/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):
            raise unittest.SkipTest('Historical r18 execution sources differ; use fresh compiler-evidence audit for current source')
        rows=json.loads(cls.controls.read_text())['rows']
        cls.w45=Path(next(r for r in rows if r['id']=='public_w45_0')['evidence'])
    def data(self,folder):
        out=folder/'attempt-00/output'
        return (out/'lowered._hecate_golden.hevm').read_bytes(),(out/'_hecate_golden.cst').read_bytes(),json.loads((folder/'request.json').read_text())
    def test_nominal_scale_path_matches_real_runtime(self):
        for folder,levels,scales in [(self.failed,[2,1],[60,40]),(self.w45,[2,2],[75,60])]:
            with self.subTest(folder=str(folder)):
                result=audit(folder);trace=result['lineage']
                self.assertEqual([x['level'] for x in trace['outputs']],levels)
                self.assertEqual([x['scale'] for x in trace['outputs']],scales)
                self.assertEqual(trace['scale_metadata_overrides'],[])
                self.assertFalse(trace['error_bound_proven']);self.assertFalse(trace['automatically_changes_acceptance'])
        self.assertFalse(audit(self.failed)['numerical_passed'])
    def test_rehashed_valid_range_result_scale_is_still_rejected(self):
        raw,cst,r=self.data(self.failed);nargs=struct.unpack_from('<Q',raw,8)[0]
        raw=bytearray(raw);offset=64+16*nargs
        scale=struct.unpack_from('<Q',raw,offset)[0];struct.pack_into('<Q',raw,offset,scale+1)
        with self.assertRaisesRegex(ValueError,'scale lineage mismatch'):lineage(bytes(raw),cst,r)
    def test_bootstrap_unknown_opcode_and_truncation_rejected(self):
        raw,cst,r=self.data(self.failed);offset=24+struct.unpack_from('<Q',raw,24)[0]
        for opcode in (5,10,1234):
            changed=bytearray(raw);struct.pack_into('<H',changed,offset,opcode)
            with self.subTest(opcode=opcode),self.assertRaisesRegex(ValueError,'Forbidden opcode'):lineage(bytes(changed),cst,r)
        with self.assertRaises(ValueError):lineage(raw[:-1],cst,r)
    def test_inconsistent_input_scale_propagation_is_rejected(self):
        raw,cst,r=self.data(self.failed);changed=bytearray(raw);struct.pack_into('<Q',changed,64,41);struct.pack_into('<Q',changed,72,41)
        with self.assertRaisesRegex(ValueError,'scale lineage mismatch'):lineage(bytes(changed),cst,r)
    def test_fused_relinearization_is_not_claimed_without_mulcc(self):
        result=audit(self.failed);relin=result['compiler_evidence']['relinearization']
        self.assertEqual(relin['fused_mulcc_operations'],0)
        self.assertFalse(relin['per_operation_runtime_observed'])
        self.assertEqual(result['compiler_evidence']['rescale']['artifact_operations'],4)
        self.assertTrue(result['compiler_evidence']['rotation_keys']['actual_key_file_verified'])

if __name__=='__main__':unittest.main()
