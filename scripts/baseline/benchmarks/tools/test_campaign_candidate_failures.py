import copy,json,tempfile,unittest
from pathlib import Path
from campaign_candidate_failures import transparent_failure,MESSAGE
from stage2_agent_pilot_plan import PAID
# Real terminal evidence is an integration fixture, never synthesized as a pass.
SOURCE=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-deepseek-1xehwd5a")
class Failures(unittest.TestCase):
 def setUp(self):
  self.r=json.loads((SOURCE/"report.json").read_text())
 def test_real_terminal_failure_remains_failed(self):
  result=transparent_failure(SOURCE,self.r,PAID)
  self.assertEqual(result["kind"],"candidate_transparent_ciphertext")
  self.assertTrue(result["remains_failed"]);self.assertFalse(result["automatic_candidate_retry"])
 def test_running_not_classified(self):
  self.r["status"]="running";self.assertIsNone(transparent_failure(SOURCE,self.r,PAID))
 def test_other_failure_not_classified(self):
  self.r["attempts"][-1]["diagnostic"]="timeout"
  self.assertIsNone(transparent_failure(SOURCE,self.r,PAID))
 def test_unknown_stderr_cannot_be_classified(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d);(f/"key-cleanup-outcome.json").write_text('{"complete":true}')
   a=f/"attempt-03";a.mkdir();(a/"execute.log").write_text("segmentation fault")
   self.assertIsNone(transparent_failure(f,self.r,PAID))
 def test_incomplete_cleanup_stops(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d);(f/"key-cleanup-outcome.json").write_text('{"complete":false}')
   with self.assertRaisesRegex(ValueError,"cleanup"):transparent_failure(f,self.r,PAID)
 def test_symlink_stops(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d);(f/"key-cleanup-outcome.json").write_text('{"complete":true}')
   a=f/"attempt-03";a.mkdir();(a/"execute.log").symlink_to(SOURCE/"attempt-03/execute.log")
   with self.assertRaisesRegex(ValueError,"Unsafe"):transparent_failure(f,self.r,PAID)
 def test_missing_artifact_stops(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d);(f/"key-cleanup-outcome.json").write_text('{"complete":true}')
   a=f/"attempt-03";a.mkdir();(a/"execute.log").write_text(MESSAGE)
   with self.assertRaises((ValueError,FileNotFoundError)):transparent_failure(f,self.r,PAID)
if __name__=="__main__":unittest.main()
