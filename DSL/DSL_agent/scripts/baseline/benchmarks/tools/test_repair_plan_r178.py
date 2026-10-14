"""No-API authorization, serial native and terminal-classification gates."""
import ast,copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from stage2_agent_repair_plan_r178 import *
from workspace_paths import RESULTS
class PlanTests(unittest.TestCase):
 def plan(self):return build(REVIEW,RESULTS/AUTH_NAME,RESULTS/NAME)
 def test_exact_scope_budget_and_no_manual_source(self):
  p=self.plan()
  self.assertEqual(p["planned"],19);self.assertEqual(len(p["shards"]),4)
  self.assertEqual((p["maximum_generations"],p["maximum_http_attempts"]),(76,304))
  self.assertEqual((p["api_workers"],p["native_workers"]),(4,1))
  self.assertEqual(sum(s["plan"]["limits"]["maximum_generations"] for s in p["shards"]),76)
  self.assertEqual(sum(s["plan"]["limits"]["maximum_http_attempts"] for s in p["shards"]),304)
  prepared=json.loads((RESULTS/"stage2-noncompiler-r177/execution/prepared.json").read_text())
  for row in prepared:
   source=json.loads((RESULTS/"stage2-noncompiler-r177/execution"/row["id"]/"job.json").read_text())["candidate"]["hecate_source"]
   def strings(value):
    if type(value) is str:yield value
    elif type(value) is dict:
     for v in value.values():yield from strings(v)
    elif type(value) is list:
     for v in value:yield from strings(v)
   self.assertFalse(any(source in v for v in strings(p["shards"])))
   self.assertTrue(any(source in v for v in strings([dict(hecate_source=source)])))
  for shard in p["shards"]:
   for row in shard["plan"]["cases"]:
    self.assertIn("explicit-v7",row["candidate_arguments"])
    self.assertNotIn("--inside",row["candidate_arguments"])
    self.assertEqual(row["request"]["compiler_configuration"]["name"],"seal-cpu-eva-w45-v1")
 def test_resealed_changed_scope_still_rejected(self):
  p=self.plan();p["maximum_generations"]+=1;p=sealed(p)
  with self.assertRaises(ValueError):verify(p,RESULTS/NAME)
 def test_no_alternative_output(self):
  with self.assertRaises(ValueError):build(REVIEW,RESULTS/AUTH_NAME,RESULTS/"other")
 def test_terminal_provider_priority(self):
  import importlib.util
  s=importlib.util.spec_from_file_location("reporter",ROOT/"scripts/stage2_agent_repair_report_r179.py")
  module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
  self.assertEqual(module.completion_layer(dict(provider_status="provider_failed",terminal_failure_layer="static_check")),"provider")
  self.assertEqual(module.completion_layer(dict(terminal_failure_layer=None)),"unknown")
 def test_serial_slot_reservation_released_and_blocks_second(self):
  import fcntl
  from stage2_agent_repair_r178 import serial_native_reservation
  from native_execution_slots import native_slot
  with tempfile.TemporaryDirectory() as td,patch("hecate_python_env.WORK",Path(td)):
   slots=Path(td)/"cache/agent-native-slots"
   with serial_native_reservation():
    with native_slot(slots,timeout=.1):
     with self.assertRaises(TimeoutError):
      with native_slot(slots,timeout=.02):pass
   with native_slot(slots,timeout=.1):
    with native_slot(slots,timeout=.1):pass
 def test_runner_and_reporter_syntax_and_limits(self):
  for f in ["scripts/stage2_agent_repair_job_r178.py","scripts/stage2_agent_repair_report_r179.py"]:
   ast.parse((ROOT/f).read_text())
  from stage2_agent_repair_r178 import candidate_command
  p=self.plan();shard=p["shards"][0]["plan"];cmd=candidate_command(shard,shard["cases"][0],Path("/public"))
  self.assertNotIn("--inside",cmd);self.assertNotIn("--replay",cmd);self.assertIn("--live",cmd)
if __name__=="__main__":unittest.main()
