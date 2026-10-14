from pathlib import Path
import sys,unittest,json
BASE=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
modules=["test_unified_output_aliases","test_unified_capacity","test_unified_capacity_extended",
"test_unified_capacity_wide","test_application_capabilities","test_agent_component",
"test_unified_generation_guidance","test_source_cli_package_policy","test_program_execution",
"test_capacity_output_identity"]
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
record=dict(tests=result.testsRun,failures=[(str(t),s)for t,s in result.failures],
errors=[(str(t),s)for t,s in result.errors],skipped=[(str(t),s)for t,s in result.skipped],modules=modules,
paid_calls=0,compilations=0,encrypted_calls=0)
(HERE/"controls-pinned.json").write_text(json.dumps(record,sort_keys=True))
print(json.dumps({k:v if type(v)is int else len(v)for k,v in record.items()}))
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
