"""Offline range/prefix boundaries; no network or official-data substitution."""
import hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import public_assets
import resume_cifar as resume

class ResumeTests(unittest.TestCase):
    def directory(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);return Path(temporary.name)
    def test_exact_range_and_total_required(self):
        self.assertEqual(resume.response_mode(206,'bytes 4-9/10','6',None,4),('prefix-plus-range',10))
        self.assertEqual(resume.response_mode(206,'bytes 4-9/10',None,'identity',4),('prefix-plus-range',10))
    def test_full_same_response_never_requires_second_request(self):
        self.assertEqual(resume.response_mode(200,None,'10',None,4),('complete-response',10))
        self.assertEqual(resume.response_mode(200,None,None,None,4),('complete-response',None))
    def test_status_encoding_length_and_range_mismatch_rejected(self):
        for args in [(302,None,'10',None,4),(206,'bytes 3-9/10','7',None,4),
            (206,'bytes 4-8/10','5',None,4),(206,'bytes 4-9/*','6',None,4),
            (206,'bytes 4-9/10','7',None,4),(206,'bytes 4-9/10','6','gzip',4),
            (200,'bytes 4-9/10','10',None,4),(200,None,'-1',None,4),
            (200,None,str(resume.MAX_BYTES+1),None,4)]:
            with self.assertRaises(ValueError):resume.response_mode(*args)
    def test_verified_prefix_join_and_changed_prefix_rejection(self):
        folder=self.directory();prefix=b'original-prefix\x00\xff';tail=b'original-tail'
        class Interrupted:
            calls=0
            def read(self,size):
                self.calls+=1
                if self.calls==1:return prefix
                raise TimeoutError('fixture incomplete archive')
        with self.assertRaises(TimeoutError):
            public_assets.store_stream(Interrupted(),folder/'prefix',65536,checkpoint=lambda:None)
        value=json.loads((folder/'prefix/incomplete-stream.json').read_text())
        reader=resume.JoinedReader(resume.prefix_reader(folder/'prefix',value),io.BytesIO(tail))
        chunks=[]
        while True:
            block=reader.read(3)
            if not block:break
            self.assertLessEqual(len(block),3);chunks.append(block)
        self.assertEqual(b''.join(chunks),prefix+tail)
        (folder/'prefix/part-00000.bin').write_bytes(b'x'*len(prefix))
        with self.assertRaisesRegex(ValueError,'Chunk digest'):resume.prefix_reader(folder/'prefix',value)
    def test_missing_or_old_approval_precedes_network_and_output(self):
        spec=resume.specification();out=self.directory()/'download'
        with patch('http.client.HTTPSConnection',side_effect=AssertionError('network forbidden')):
            for approval in [None,{},dict(format='poseidon-explicit-public-asset-approval-v1',status='approved')]:
                with self.assertRaisesRegex(ValueError,'matching resume approval'):
                    resume.fetch(spec,approval,out,checkpoint=lambda:None)
        self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
