"""Static CLI boundaries and read-only audit of real Poseidon artifacts."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import poseidon_benchmark_entries as entry
from benchmark_graph import digest
SUITE=entry.ROOT/"scripts/baseline/benchmarks/semantic-v2-ledger-r38"
INDEX="ee7a0868cd5ea68e7ada18e43a52249b6d1a474a4c13e006fb7285cbe506a6a2"
class HelperPlanTests(unittest.TestCase):
    def test_frozen_tasks_migrate_to_cpu_with_original_waterlines(self):
        wanted=["upstream_silu_frozen_0","upstream_bn_asymmetric_0","upstream_concat_frozen_1"]
        rows,defs,index=entry.helper_selection(SUITE,INDEX,wanted,3,None)
        self.assertEqual([r["task"]["id"] for r in rows],wanted)
        self.assertEqual([r["compiler_configuration"]["waterline"] for r in rows],[40,45,45])
        for r in rows:
            self.assertEqual(r["status"],"ready")
            self.assertEqual(r["compiler_configuration"]["backend"],"poseidon_cpu")
            self.assertTrue(r["request_id"])
            self.assertEqual(r["task"]["model_sha256"],digest(defs[r["task"]["id"]]["model"]))

    def test_explicit_precision_changes_request_binding(self):
        a=entry.helper_selection(SUITE,INDEX,["upstream_silu_frozen_0"],1,None)[0][0]
        b=entry.helper_selection(SUITE,INDEX,["upstream_silu_frozen_0"],1,"poseidon-cpu-eva-w45-v1")[0][0]
        self.assertNotEqual(a["request_id"],b["request_id"])

    def test_unknown_duplicate_and_overlarge_selection_rejected(self):
        for names,limit in [(["missing"],1),(["upstream_silu_frozen_0"]*2,2),
                           (["upstream_silu_frozen_0","upstream_bn_asymmetric_0"],1)]:
            with self.assertRaises(ValueError):entry.helper_selection(SUITE,INDEX,names,limit,None)

    def test_frozen_task_source_mutation_rejected(self):
        original=entry.tasks()
        changed=copy.deepcopy(original);changed[0]["required_helpers"]=[]
        with patch.object(entry,"tasks",return_value=changed):
            with self.assertRaisesRegex(ValueError,"definitions changed"):
                entry.helper_selection(SUITE,INDEX,None,1,None)

    def test_completed_row_matches_actual_helper_evidence(self):
        report=json.loads((entry.WORK/"results/poseidon-benchmark-tools-r276/helpers/report.json").read_text())
        row=report["rows"][0];definition=next(r for r in entry.cases() if "upstream_"+r["name"]==row["id"])
        entry.verify_helper_result(row,definition,report["binding"],recorded=True)
        bad=copy.deepcopy(row);bad["actual_calls"]+=1
        with self.assertRaisesRegex(ValueError,"differs from audit"):
            entry.verify_helper_result(bad,definition,report["binding"],recorded=True)
        bad=copy.deepcopy(row);bad.pop("audited")
        with self.assertRaisesRegex(ValueError,"differs from audit"):
            entry.verify_helper_result(bad,definition,report["binding"],recorded=True)

    def test_no_paid_cli_or_seal_option(self):
        for args in [["helper-directed","--live"],["helper-directed","--compiler-configuration","seal-cpu-eva-w45-v1"]]:
            with self.assertRaises(SystemExit):entry.helper_main(args)

class AuditTests(unittest.TestCase):
    def test_non_poseidon_report_rejected_before_auditing(self):
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            folder=Path(t);(folder/"report.json").write_text('{"backend":"upstream_SEAL_HEVM_CPU"}')
            with patch.object(entry,"verify_candidate") as audit:
                with self.assertRaisesRegex(ValueError,"non-Poseidon"):entry.audit_poseidon(folder)
                audit.assert_not_called()

    def test_direct_platform_path_only(self):
        with self.assertRaisesRegex(ValueError,"Platform-local"):entry.result_path(Path("/tmp/outside-results"))
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            p=Path(t);(p/"linked").symlink_to(p/"missing")
            with self.assertRaises(ValueError):entry.result_path(p/"linked")

    def test_read_only_runtimeplan_audit_of_native_ciphertext_evidence(self):
        p=entry.WORK/"results/poseidon-benchmark-entries-r273/baseline/report.json"
        report=json.loads(p.read_text())
        rows=[entry.audit_poseidon(Path(r["evidence"])) for r in report["reports"]]
        self.assertEqual(len(rows),3)
        for r in rows:
            self.assertEqual(r["backend"],"poseidon_cpu");self.assertTrue(r["numerical_passed"])
            self.assertEqual(r["new_encrypted_executions"],0)
            self.assertFalse(r["per_instruction_native_trace"])
            self.assertEqual(r["compiler_evidence"]["security_parameters"]["actual_generated_parameters"]["security"],"tc128")
        self.assertGreater(sum(r["compiler_evidence"]["rescale"]["artifact_operations"] for r in rows),0)
        self.assertGreater(sum(r["compiler_evidence"]["relinearization"]["artifact_operations"] for r in rows),0)

    def test_tampered_runtimeplan_rejected_by_original_artifact_binding(self):
        import shutil
        p=entry.WORK/"results/poseidon-benchmark-entries-r273/baseline/report.json"
        original=Path(json.loads(p.read_text())["reports"][0]["evidence"])
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            folder=Path(t)/"case";shutil.copytree(original,folder)
            plan=folder/"attempt-00/output/bound/plan.json";data=json.loads(plan.read_text())
            data["values"][0]["level"]=0;plan.write_text(json.dumps(data))
            with self.assertRaises(ValueError):entry.audit_poseidon(folder)

    def test_audit_manifest_count_boundary(self):
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            args=["compiler-evidence","--output",str(Path(t)/"new.json")]
            for i in range(9):args+=["--batch-report",str(Path(t)/(str(i)+".json"))]
            with self.assertRaisesRegex(ValueError,"manifest bounds"):entry.compiler_main(args)
            self.assertFalse((Path(t)/"new.json").exists())

if __name__=="__main__":unittest.main()
