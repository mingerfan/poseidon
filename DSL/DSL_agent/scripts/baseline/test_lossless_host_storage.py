"""New bounded storage and expanded semantic binding controls."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from application_policy import prepare
from delivery_component import canonical
from unified_request_storage import compact_request
import lossless_host_bound_delivery as host


class LosslessHostStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='lossless_storage_fixture',
            inputs=[dict(name='input0',shape=[1,7])],constants=dict(w=[[.125]*7,[-.25]*7],b=[.03125,-.0625]),
            nodes=[dict(id='linear0',op='linear',inputs=['input0','w','b'],attrs={},outputs=['result0'])],
            outputs=[dict(name='output0',value='result0')])
        cls.full,_=prepare(model,'poseidon_cpu','numerical',dict(compiler_configuration='poseidon-cpu-eva-w45-current-v1',chunk_period=128,
            public_weight_policy='descriptors-compact-origins-v3',generation_guidance=None))
        cls.encoded=compact_request(cls.full)
        assert canonical(cls.full)!=canonical(cls.encoded)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def asset(self,name,value):
        p=self.root/name;p.write_bytes(canonical(value));return p

    def test_raw_and_encoded_assets_share_semantics_but_each_bind_exact_bytes(self):
        raw=self.asset('raw.json',self.full);encoded=self.asset('encoded.json',self.encoded)
        a=host.freeze_host_request(raw);b=host.freeze_host_request(encoded)
        for key in ['request_id','host_canonical_sha256','public_canonical_sha256']:self.assertEqual(a[key],b[key])
        self.assertNotEqual(a['sha256'],b['sha256']);self.assertEqual(host.resolve_host(b),self.full)

    def test_decoder_roundtrip_preserves_original_numeric_payload(self):
        self.assertEqual(canonical(host._decode_json(canonical(self.encoded))),canonical(self.full))
        value=host._decode_json(b'{"v":[0,-0.0,0.0,1.0]}')['v']
        self.assertIs(type(value[0]),int);self.assertEqual(math.copysign(1.,value[1]),-1.);self.assertEqual(math.copysign(1.,value[2]),1.)

    def test_resolve_reads_the_asset_once(self):
        p=self.asset('encoded.json',self.encoded);binding=host.freeze_host_request(p)
        with patch.object(host,'read_bytes',wraps=host.read_bytes)as reader:
            resolved=host.resolve_host(binding)
        self.assertEqual(resolved,self.full);self.assertEqual(reader.call_count,1)

    def test_changed_stored_bytes_cannot_reuse_old_binding(self):
        p=self.asset('encoded.json',self.encoded);binding=host.freeze_host_request(p)
        p.write_bytes(canonical(self.encoded)+b'\n')
        with self.assertRaises(ValueError):host.resolve_host(binding)

    def test_invalid_runs_and_unregistered_period_are_rejected(self):
        broken=copy.deepcopy(self.encoded)
        record=next(v for v in broken['public_constants'].values()if type(v)is dict and 'runs'in v)
        record['length']+=1
        with self.assertRaises(ValueError):host._decode_json(canonical(broken))
        broken=copy.deepcopy(self.encoded);broken['layout']['input_slot_period']=8192
        with self.assertRaises(ValueError):host._decode_json(canonical(broken))

    def test_duplicate_and_nonfinite_json_is_rejected(self):
        for raw in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}']:
            with self.assertRaises(ValueError):host._decode_json(raw)

    def test_original_semantic_validation_rejects_changed_weights(self):
        broken=copy.deepcopy(self.encoded);broken['model']['constants']['w'][0][0]=.25
        with self.assertRaises(ValueError):host.freeze_host_request(self.asset('wrong.json',broken))

    def test_raw_file_still_has_the_original_four_MiB_bound(self):
        p=self.root/'oversize.json';p.write_bytes(b' '*(4*1024**2+1))
        with self.assertRaises(ValueError):host.freeze_host_request(p)
        with self.assertRaises(ValueError):host._json(p)


if __name__=='__main__':unittest.main()
