"""Fail-closed tests for historical acceptance revalidation."""
import copy
import unittest
from unittest.mock import patch
import retained_acceptance_context_r152 as target

class ContextTests(unittest.TestCase):
 def test_real_source_lineage(self):
  result=target.context()
  self.assertFalse(result['old_executions_rebound'])
  self.assertEqual(len(result['changed_files']),3)
 def test_unreviewed_reference_change_rejected(self):
  sources=target.runtime_sources();sources['scripts/baseline/benchmark_math.py']='0'*64
  with patch.object(target,'runtime_sources',return_value=sources):
   with self.assertRaisesRegex(ValueError,'Current source identity'):target.context()
 def test_added_runtime_module_rejected(self):
  sources=target.runtime_sources();sources['scripts/baseline/unreviewed.py']='0'*64
  with patch.object(target,'runtime_sources',return_value=sources):
   with self.assertRaises(ValueError):target.context()
 def test_damaged_historical_evidence_rejected(self):
  original=target.sha
  def changed(path):
   if str(path).endswith('stage2-gates-r51/report.json'):return '0'*64
   return original(path)
  with patch.object(target,'sha',side_effect=changed):
   with self.assertRaisesRegex(ValueError,'Parent changed'):target.context()
 def test_incomplete_compatibility_rejected(self):
  original=target.bound
  def incomplete(path):
   result=copy.deepcopy(original(path))
   if path.name=='stage2-directed-guidance-acceptance-r146.json':result['old_requests_identical']=2029
   return result
  with patch.object(target,'bound',side_effect=incomplete):
   with self.assertRaisesRegex(ValueError,'Compatibility evidence'):target.context()
if __name__=='__main__':unittest.main()
