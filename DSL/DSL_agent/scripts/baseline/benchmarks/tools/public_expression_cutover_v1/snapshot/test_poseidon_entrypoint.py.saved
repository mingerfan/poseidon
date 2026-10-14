"""Entry selection preserves traced computation and immutable compiler versions."""
import copy,json,tempfile,unittest
from pathlib import Path
from compiler_configuration import configuration,validate_configuration
from poseidon_runtime_compile import entrypoint_trace,prepare_entrypoint,audit_entrypoint,command,ENTRY_POLICY
TRACE = '''module {
  "func.func"() <{function_type = () -> (), sym_name = "_hecate_unused"}> ({
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x!earth.ci<0 * 0>>, sym_name = "_hecate_golden"}> ({
    %0 = "earth.constant"() {value = 0.125 : f64} : () -> tensor<1x!earth.ci<0 * 0>>
    "func.return"(%0) : (tensor<1x!earth.ci<0 * 0>>) -> ()
  }) : () -> ()
}
'''
class EntrypointTests(unittest.TestCase):
 def test_v2_and_v3_component_requests_reconstruct_exactly(self):
  from benchmark_suite import Builder
  from unified_graph_contract import prepare
  from component_contract import reconstruct_request
  b=Builder([(2,)])
  model=b.finish(b.node("square",["input0"]))
  for schema in (2,3):
   cfg=configuration("poseidon-cpu-eva-w45-v1",schema=schema)
   request=prepare(model,cfg["ckks_config_sha256"],cfg)
   self.assertEqual(reconstruct_request(request),request)
 def test_real_frontend_module_symbol_is_preserved(self):
  trace=TRACE.replace("module {", '\"builtin.module\"() <{sym_name = \"/out/candidate_trace.mlir\"}> ({',1)
  selected,receipt=entrypoint_trace(trace)
  self.assertTrue(selected.startswith('\"builtin.module\"() <{sym_name = \"/out/candidate_trace.mlir\"}> ({'))
  self.assertEqual(receipt["private_functions"],["_hecate_unused"])
  self.assertEqual(selected.replace(', sym_visibility = "private"',''),trace)
 def test_only_helper_visibility_changes(self):
  selected,receipt=entrypoint_trace(TRACE)
  self.assertEqual(selected.replace(', sym_visibility = "private"',''),TRACE)
  self.assertEqual(receipt["private_functions"],["_hecate_unused"])
  self.assertNotIn('sym_name = "_hecate_golden", sym_visibility',selected)
 def test_empty_helpers_are_kept_in_original_trace(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t);(out/"candidate_trace.mlir").write_text(TRACE)
   receipt=prepare_entrypoint(out,configuration("poseidon-cpu-eva-w45-v1"))
   self.assertEqual((out/"candidate_trace.mlir").read_text(),TRACE)
   self.assertEqual(receipt["format"],ENTRY_POLICY)
 def test_missing_or_duplicate_entry_rejected(self):
  for source in [TRACE.replace("_hecate_golden","other"),TRACE+TRACE]:
   with self.subTest(source=source[:20]),self.assertRaises(ValueError):entrypoint_trace(source)
 def test_symbol_references_and_calls_rejected(self):
  for extra in ['func.call @helper()', '"func.call"()', 'symbol = @helper','"earth.call"()']:
   with self.subTest(extra=extra),self.assertRaises(ValueError):entrypoint_trace(TRACE+extra)
 def test_ambiguous_visibility_or_declaration_rejected(self):
  for source in [TRACE.replace('sym_name = "_hecate_unused"','sym_name = "_hecate_unused", sym_visibility = "private"'), TRACE.replace('  "func.func"()', '    "func.func"()',1), TRACE+'sym_name = "shadow"']:
   with self.assertRaises(ValueError):entrypoint_trace(source)
 def test_receipt_and_selected_trace_tampering_rejected(self):
  cfg=configuration("poseidon-cpu-eva-w45-v1");request=dict(compiler_configuration=cfg)
  with tempfile.TemporaryDirectory() as t:
   out=Path(t);(out/"candidate_trace.mlir").write_text(TRACE)
   receipt=prepare_entrypoint(out,cfg);argv=command(45,entry_policy=ENTRY_POLICY)
   audit_entrypoint(out,request,receipt,argv)
   wrong=copy.deepcopy(receipt);wrong["private_functions"]=[]
   with self.assertRaises(ValueError):audit_entrypoint(out,request,wrong,argv)
   with self.assertRaises(ValueError):audit_entrypoint(out,request,receipt,command(45))
   (out/"candidate_trace.entry.mlir").write_text((out/"candidate_trace.entry.mlir").read_text().replace("0.125","0.25"))
   with self.assertRaises(ValueError):audit_entrypoint(out,request,receipt,argv)
 def test_old_configuration_is_exactly_reconstructible(self):
  old=configuration("poseidon-cpu-eva-w45-v1",schema=2)
  new=configuration("poseidon-cpu-eva-w45-v1")
  self.assertEqual(old["schema"],2);self.assertEqual(new["schema"],3)
  self.assertNotIn("runtime_plan_entry_policy",old)
  self.assertNotEqual(old["identity_sha256"],new["identity_sha256"])
  self.assertEqual(validate_configuration(old,old["ckks_config_sha256"]),old)
  self.assertEqual(validate_configuration(new,new["ckks_config_sha256"]),new)
 def test_rehashed_policy_and_boolean_version_rejected(self):
  cfg=configuration("poseidon-cpu-eva-w45-v1")
  cfg["runtime_plan_entry_policy"]="ignore-symbol-uses"
  with self.assertRaises(ValueError):validate_configuration(cfg,cfg["ckks_config_sha256"])
  with self.assertRaises(ValueError):configuration("poseidon-cpu-eva-w45-v1",schema=True)
 def test_historical_configuration_never_silently_selects_entry(self):
  old=configuration("poseidon-cpu-eva-w45-v1",schema=2)
  with tempfile.TemporaryDirectory() as t:self.assertIsNone(prepare_entrypoint(Path(t),old))
  self.assertNotIn("--symbol-dce",command(45))
  with self.assertRaises(ValueError):audit_entrypoint(Path("."),dict(compiler_configuration=old),{},command(45))
 def test_cuda_configuration_does_not_inherit_cpu_acceptance(self):
  self.assertEqual(configuration("poseidon-cuda-eva-w45-v1")["schema"],2)
  with self.assertRaises(ValueError):configuration("poseidon-cuda-eva-w45-v1",schema=3)
if __name__=="__main__":unittest.main()
