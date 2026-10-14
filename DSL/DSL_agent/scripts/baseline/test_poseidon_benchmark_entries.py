"""Poseidon batch boundaries, frozen-release identity and numerical proof requirements."""
import copy,contextlib,hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmark_suite import Builder
from benchmark_graph import digest
import semantic_benchmark_execution as execution

def model():
    b=Builder([(2,)])
    return b.finish(b.node("square",["input0"]))

def accepted_report():
    return dict(backend="poseidon_cpu",compiler_configuration=dict(name=execution.CONFIG),
                status="passed",validation_level="numerical",
                attempts=[dict(status="passed",compiled=True,executed=True,numerically_correct=True,
                    comparison=dict(passed=True),execution=dict(backend="poseidon_cpu",
                    encrypted_execution=True,executor_secret_key=False))])

class ConfigurationTests(unittest.TestCase):
    def test_cpu_configurations_only(self):
        for name in execution.CPU_CONFIGURATIONS:
            self.assertEqual(execution.cpu_configuration(name)["backend"],"poseidon_cpu")
        for name in ["seal-cpu-eva-w45-v1","poseidon-cuda-eva-w45-v1","unknown"]:
            with self.assertRaises(ValueError):execution.cpu_configuration(name)

    def test_preflight_uses_selected_poseidon_profile(self):
        from unified_graph_contract import prepare
        g=model();row=dict(model=g,model_sha256=digest(g),category="test")
        ids=[]
        for name in execution.CPU_CONFIGURATIONS:
            cfg=execution.cpu_configuration(name)
            result=execution.preflight([row],need_rule=False,compiler_configuration=name)[0]
            expected=prepare(g,cfg["ckks_config_sha256"],cfg)
            self.assertEqual(result["status"],"ready")
            self.assertEqual(result["request_id"],expected["request_id"])
            ids.append(result["request_id"])
        self.assertNotEqual(*ids)

    def test_cli_rejects_legacy_before_environment_entry(self):
        with patch("sys.argv",["benchmark.py","baseline","--compiler-configuration","seal-cpu-eva-w45-v1"]),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:execution.main()
            self.assertEqual(error.exception.code,2)

    def test_runtime_binding_includes_actual_frontend(self):
        prefix="third_party/ckks-runtime-agent/third_party/dacapo/python/"
        entries=execution.runtime_sources()
        self.assertTrue(any(f.startswith(prefix+"hecate/") for f in entries))
        self.assertTrue(any(f.startswith(prefix+"poly/") for f in entries))
        for f in entries:
            if f.startswith(prefix):
                self.assertEqual(entries[f],hashlib.sha256((execution.ROOT/f).read_bytes()).hexdigest())

class ReleaseTests(unittest.TestCase):
    def test_default_preserves_strict_source_validation(self):
        with patch.object(execution,"load",return_value=([],{})) as loader:
            execution.load_execution_suite(Path("/example"))
            loader.assert_called_once_with(Path("/example"))

    def test_explicit_release_hash_pins_old_dataset_and_new_execution(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/"index.json").write_text('{"source_hashes":{"old":"identity"}}')
            h=hashlib.sha256((p/"index.json").read_bytes()).hexdigest()
            with patch.object(execution,"load",return_value=([],{})) as loader:
                execution.load_execution_suite(p,h)
                loader.assert_called_once_with(p,check_sources=False)
                (p/"index.json").write_text('{}')
                with self.assertRaisesRegex(ValueError,"hash mismatch"):execution.load_execution_suite(p,h)

    def test_release_symlink_and_malformed_hash_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/"real.json").write_text("{}");(p/"index.json").symlink_to(p/"real.json")
            with self.assertRaises(ValueError):execution.load_execution_suite(p,"0"*64)
            for h in ["0"*63,"G"*64,False]:
                with self.assertRaises(ValueError):execution.load_execution_suite(p,h)

class NumericalEvidenceTests(unittest.TestCase):
    def test_real_poseidon_numerical_metadata_accepted(self):
        execution.verify_numerical_report(accepted_report(),execution.CONFIG)

    def test_old_backend_or_other_waterline_rejected(self):
        for field,value in [("backend","upstream_SEAL_HEVM_CPU"),("compiler_configuration",dict(name="poseidon-cpu-eva-w40-v1"))]:
            report=accepted_report();report[field]=value
            with self.assertRaises(ValueError):execution.verify_numerical_report(report,execution.CONFIG)

    def test_compile_only_not_numerical_success(self):
        report=accepted_report();report["validation_level"]="compiled"
        with self.assertRaises(ValueError):execution.verify_numerical_report(report,execution.CONFIG)

    def test_missing_execution_or_failed_comparison_rejected(self):
        mutations=[
            lambda r:r["attempts"][0].update(compiled=False),
            lambda r:r["attempts"][0].update(executed=False),
            lambda r:r["attempts"][0]["comparison"].update(passed=False),
            lambda r:r["attempts"][0]["execution"].update(encrypted_execution=False),
            lambda r:r["attempts"][0]["execution"].update(executor_secret_key=True)]
        for change in mutations:
            report=accepted_report();change(report)
            with self.assertRaises(ValueError):execution.verify_numerical_report(report,execution.CONFIG)

    def test_current_dual_reference_finite_inputs(self):
        result=execution.reference_check(model())
        self.assertEqual(result["groups"],16);self.assertEqual(result["references"],2)
        self.assertLessEqual(result["max_absolute_error"],1e-12)

    def test_reference_disagreement_stops_execution(self):
        import numpy as np
        with patch("benchmark_torch.evaluate",return_value={"output0":np.zeros((3,))}):
            with self.assertRaises(ValueError):execution.reference_check(model())

class OperatorEntryTests(unittest.TestCase):
    def test_plan_is_poseidon_and_selected_cases_only(self):
        import check_operator_regression as regression
        args=["check_operator_regression.py","--output","/unused-output","--case-id","broadcast_0",
              "--compiler-configuration","poseidon-cpu-eva-w40-v1"]
        out=io.StringIO()
        with patch("sys.argv",args),patch.object(regression,"runtime_sources",return_value={}),contextlib.redirect_stdout(out):
            self.assertEqual(regression.main(),0)
        plan=json.loads(out.getvalue())
        self.assertEqual(plan["backend"],"poseidon_cpu")
        self.assertEqual(plan["compiler_configuration"],"poseidon-cpu-eva-w40-v1")
        self.assertEqual([x["id"] for x in plan["cases"]],["broadcast_0"])
        self.assertEqual(plan["paid_calls"],0)

    def test_legacy_operator_option_rejected(self):
        import check_operator_regression as regression
        with patch("sys.argv",["regression.py","--output","/unused","--compiler-configuration","seal-cpu-eva-w45-v1"]),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):regression.main()

if __name__=="__main__":unittest.main()
