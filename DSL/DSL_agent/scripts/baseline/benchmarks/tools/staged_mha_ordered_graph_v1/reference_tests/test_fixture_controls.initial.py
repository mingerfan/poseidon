"""Fixture integrity and version separation, standard library only."""
import ast,hashlib,json,math,struct,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'source_fixtures'
class FixtureControls(unittest.TestCase):
 def manifests(self):
  for row in json.loads((ROOT/'index.json').read_text())['cases']:
   p=Path(__file__).resolve().parent/row['manifest'];yield row,p,json.loads(p.read_text())
 def test_nine_explicit_versioned_manifests(self):
  rows=list(self.manifests());self.assertEqual(len(rows),9)
  for row,p,m in rows:
   self.assertEqual(m['format'],'poseidon-trusted-pytorch-v16');self.assertEqual(m['entry'],'model.make')
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['manifest_sha256'])
   self.assertEqual(set(m['approximations']),{'format','defaults','modules','nodes'})
   self.assertEqual(set(m['approximations']['defaults']),{'exp','reciprocal'})
 def test_public_weight_header_bytes_hash_and_finiteness(self):
  for row,p,m in self.manifests():
   for name,w in m['weights'].items():
    raw=(p.parent/w['file']).read_bytes();self.assertEqual(len(raw),w['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),w['sha256'])
    self.assertEqual(raw[:8],b'\x93NUMPY\x01\x00');hlen=int.from_bytes(raw[8:10],'little');h=ast.literal_eval(raw[10:10+hlen].decode());self.assertEqual(list(h['shape']),w['shape']);self.assertEqual(h['descr'],w['dtype']);self.assertFalse(h['fortran_order'])
    data=raw[10+hlen:];count=math.prod(w['shape'])
    if w['dtype']=='<f8':
     values=struct.unpack('<'+'d'*count,data);self.assertTrue(all(math.isfinite(v) and abs(v)<=1024 for v in values))
    else:self.assertEqual(len(data),count);self.assertTrue(all(v in (0,1) for v in data))
 def test_shallow_profiles_share_exact_original_source_and_weights(self):
  rows={row['id']:(row,p,m) for row,p,m in self.manifests()}
  for name,(row,p,m) in rows.items():
   if '__declared_shallow_v1' in name:
    _,op,om=rows[name.split('__')[0]];self.assertEqual(m['files'],om['files']);self.assertEqual(m['weights'],om['weights'])
    self.assertEqual(m['inputs'],om['inputs']);self.assertEqual(m['outputs'],om['outputs'])
    self.assertEqual(m['approximations']['defaults']['exp']['degree'],1)
    self.assertEqual(om['approximations']['defaults']['exp']['degree'],6)
 def test_source_is_self_contained_and_hash_bound(self):
  for row,p,m in self.manifests():
   source=p.parent/'model.py';text=source.read_text();self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),m['files']['model.py']);ast.parse(text)
   self.assertNotIn('benchmarks',text);self.assertNotIn('eval(',text);self.assertNotIn('exec(',text);self.assertNotIn('torch.load',text)
 def test_none_contexts_retain_v16_and_tensors_only_named_outputs(self):
  for row,p,m in self.manifests():
   if 'none' in row['id']:
    self.assertIn('None',(p.parent/'model.py').read_text());self.assertEqual(m['format'],'poseidon-trusted-pytorch-v16')
   self.assertTrue(all(type(o) is str for o in m['outputs']))
if __name__=='__main__':unittest.main()
