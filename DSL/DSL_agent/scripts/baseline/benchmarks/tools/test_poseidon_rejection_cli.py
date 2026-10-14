"""Static migration, immutable fixtures and previously verified native controls."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import poseidon_rejection_entries as entry
from benchmark_graph import signature
SUITE=entry.ROOT/"scripts/baseline/benchmarks/semantic-v2-ledger-r38"
INDEX="ee7a0868cd5ea68e7ada18e43a52249b6d1a474a4c13e006fb7285cbe506a6a2"

class RejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog,_=entry.selection(SUITE,INDEX,0,48)
        cfg=entry.cpu_configuration("poseidon-cpu-eva-w45-v1")
        cls.requests={c:entry.prepare(entry.model(c),cfg["ckks_config_sha256"],cfg) for c in range(3)}
        paths=[];tops=set()
        for f,field in (("poseidon-benchmark-entries-r273/baseline/report.json","reports"),
                        ("poseidon-benchmark-entries-r273/operators/report.json","records")):
            path=entry.WORK/"results"/f
            if not path.is_file():continue
            report=json.loads(path.read_text())
            rows=report.get(field,report.get("rows",[]))
            for row in rows:
                if row.get("status")!="passed" or not row.get("evidence"):continue
                folder=Path(row["evidence"]);model=json.loads((folder/"model.json").read_text())
                top=signature(model,True)
                if top not in tops:paths.append(folder);tops.add(top)
                if len(paths)==3:break
            if len(paths)==3:break
        if len(paths)!=3:
            # Existing real-evidence replay is separate from portable static tests.
            cls.control_paths=[];cls.control_rows=[]
        else:
            cls.control_paths=paths;cls.control_rows=entry.controls(paths)

    def test_original_frozen_catalog_and_three_contexts(self):
        self.assertEqual(len(self.catalog),48)
        by_requirement={}
        for t in self.catalog:by_requirement.setdefault(t["requirement"],set()).add(t["context"])
        self.assertEqual(len(by_requirement),16)
        self.assertTrue(all(c=={0,1,2} for c in by_requirement.values()))

    def test_explicit_corpus_hash_and_shard_bounds(self):
        with self.assertRaisesRegex(ValueError,"index hash mismatch"):entry.selection(SUITE,"0"*64,0,48)
        for offset,limit in ((-1,1),(0,49),(48,1),(0,0)):
            with self.assertRaises(ValueError):entry.selection(SUITE,INDEX,offset,limit)

    def test_changed_frozen_task_definitions_rejected(self):
        bad=copy.deepcopy(self.catalog);bad[0]["expected_reason"]="different"
        with patch.object(entry,"tasks",return_value=bad):
            with self.assertRaisesRegex(ValueError,"definitions changed"):entry.selection(SUITE,INDEX,0,48)

    def test_changed_historical_fixture_rejected(self):
        t=next(t for t in self.catalog if t["layer"]=="candidate_ast_contract")
        fixture=entry.fixture(t["requirement"].removeprefix("reject."),t["context"])
        bad=copy.deepcopy(fixture);bad["positive"][1]["hecate_source"]+="\n"
        with patch.object(entry,"fixture",return_value=bad):
            with self.assertRaisesRegex(ValueError,"Historical rejection fixture"):
                entry.mapped_fixture(t,self.requests,[],None)

    def test_all_graph_candidate_and_request_controls(self):
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            directory=Path(t)
            rows=[entry.run_task(task,self.requests,[],directory) for task in self.catalog if task["layer"]!="artifact_preflight"]
            self.assertEqual(len(rows),42)
            for row in rows:
                self.assertEqual(row["status"],"passed")
                self.assertTrue(row["positive_control_accepted"] and row["negative_rejected"])
                self.assertFalse(row["actual_compile"] or row["actual_ciphertext_execution"] or row["actual_sandbox_execution"])
            self.assertEqual(list(directory.iterdir()),[])

    def test_missing_artifact_controls_are_blocked(self):
        with patch.object(entry,"inspect") as inspect:
            rows=[entry.run_task(t,self.requests,[],None) for t in self.catalog if t["layer"]=="artifact_preflight"]
            self.assertEqual(len(rows),6)
            self.assertTrue(all(r["status"]=="blocked" for r in rows))
            inspect.assert_not_called()

    def test_native_control_runtimeplan_positive_and_negative(self):
        if not self.control_rows:self.skipTest("Three retained Poseidon controls unavailable")
        before={str(p/"report.json"):entry.sha(p/"report.json") for p in self.control_paths}
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            rows=[entry.run_task(task,self.requests,self.control_rows,Path(t)) for task in self.catalog if task["layer"]=="artifact_preflight"]
            self.assertEqual(len(rows),6)
            self.assertEqual(len({r["positive_control_topology"] for r in rows}),3)
            for row in rows:
                self.assertEqual(row["status"],"passed")
                self.assertIn("unsupported capability",row["reason"])
                self.assertFalse(row["actual_ciphertext_execution"])
            self.assertEqual(len(list(Path(t).glob("*.json"))),12)
        self.assertEqual(before,{str(p/"report.json"):entry.sha(p/"report.json") for p in self.control_paths})

    def test_artifact_control_cardinality_and_duplicates(self):
        for paths in ([Path("one")],[Path("one"),Path("two")],[Path("one")]*3,[Path(str(i)) for i in range(4)]):
            with self.assertRaisesRegex(ValueError,"zero or three"):entry.controls(paths)

    def test_artifact_controls_need_distinct_topologies(self):
        if not self.control_rows:self.skipTest("Three retained Poseidon controls unavailable")
        with patch.object(entry,"signature",return_value="same"):
            with self.assertRaisesRegex(ValueError,"different topologies"):entry.controls(self.control_paths)

    def test_artifact_control_identity_change_rejected(self):
        if not self.control_rows:self.skipTest("Three retained Poseidon controls unavailable")
        bad=copy.deepcopy(self.control_rows);bad[0]["plan_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"identity changed"):entry.validate_controls_unchanged(bad)

    def test_bad_rejection_reason_is_not_success(self):
        task=self.catalog[0]
        def check(value):
            if value=="negative":raise ValueError("unrelated environment failure")
        with patch.object(entry,"mapped_fixture",return_value=("positive","negative",check,"expected","test",{})):
            with self.assertRaisesRegex(ValueError,"Unexpected rejection reason"):entry.run_task(task,self.requests,[],None)

    def test_unexpected_acceptance_is_not_success(self):
        with patch.object(entry,"mapped_fixture",return_value=("positive","negative",lambda v:None,"expected","test",{})):
            with self.assertRaisesRegex(ValueError,"unexpectedly accepted"):entry.run_task(self.catalog[0],self.requests,[],None)

    def test_fresh_direct_output_preserves_existing_files(self):
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            directory=Path(t);existing=directory/"report.json";existing.write_text("retained")
            with self.assertRaises(ValueError):entry.output_path(existing)
            self.assertEqual(existing.read_text(),"retained")
            (directory/"linked").symlink_to(directory,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,"Symlinked output parent"):entry.output_path(directory/"linked"/"new.json")
            self.assertFalse((directory/"new.json").exists())

    def test_paid_and_legacy_backend_cli_unavailable(self):
        for argv in (["rejections","--live"],["rejections","--compiler-configuration","seal-cpu-eva-w45-v1"]):
            with self.assertRaises(SystemExit):entry.main(argv)

    def test_platform_bound_paths(self):
        with self.assertRaisesRegex(ValueError,"Platform-local"):entry.output_path(Path("/tmp/forbidden-rejection.json"))

    def test_integrity_fault_keeps_completed_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            out=Path(t)/"partial.json";first={"bound":"original"};drift={"bound":"changed"};calls=[0]
            def changing_sources():
                calls[0]+=1
                return first if calls[0]<=2 else drift
            with patch.object(entry,"pinned",return_value=True),patch.object(entry,"selection",return_value=(self.catalog[:2],self.catalog)),patch.object(entry,"sources",side_effect=changing_sources),patch.object(entry,"backend_identity",return_value={"backend":"poseidon_cpu"}):
                status=entry.main(["rejections","--execute","--output",str(out)])
            self.assertEqual(status,1)
            report=json.loads(out.read_text())
            self.assertEqual(report["status"],"stopped");self.assertEqual(report["passed"],1)
            self.assertEqual(report["not_run"],1);self.assertEqual(len(report["records"]),1)
            self.assertEqual(report["terminal_error"]["failure_layer"],"integrity")
            self.assertFalse(report["sources_unchanged"])

    def test_missing_controls_report_is_not_all_passed(self):
        with tempfile.TemporaryDirectory(dir=entry.WORK/"results") as t:
            out=Path(t)/"blocked.json"
            with patch.object(entry,"pinned",return_value=True):
                status=entry.main(["rejections","--execute","--suite",str(SUITE),
                    "--frozen-suite-index-sha256",INDEX,"--output",str(out)])
            self.assertEqual(status,1)
            report=json.loads(out.read_text())
            self.assertEqual(report["status"],"blocked")
            self.assertEqual((report["passed"],report["failed"],report["blocked"],report["not_run"]),(42,0,6,0))
            self.assertEqual(report["new_encrypted_executions"],0)

if __name__=="__main__":unittest.main()
