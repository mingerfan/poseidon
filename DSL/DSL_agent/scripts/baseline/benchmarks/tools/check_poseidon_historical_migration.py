"""Static origin/package rejection tests and active backend guards; no new executions."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import unittest
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(Path(__file__).resolve().parent)]

def main():
    from hecate_python_env import WORK,VENV,enter_nix
    p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not a.inside:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--output",str(a.output)])
        return enter_nix(cmd,seconds=400,environment="poseidon-cpu")
    from benchmark_graph import require,canonical,digest
    from check_poseidon_helpers import snapshot
    from audit_historical_delivery import verify_delivery
    from requalify_historical_agent_sources import verify_frozen_files,historical,rebind,compare_arrays,sha
    source=WORK/"results/poseidon-historical-agent-replay-r371"
    require(not a.output.exists(),"Fresh acceptance")
    a.output.mkdir();frozen=snapshot()
    report=json.loads((source/"report.json").read_text());audit_rows=[]
    for row in report["rows"]:
        require(row["status"]=="passed","Successful source cohort")
        audit_rows.append(verify_delivery(source/row["id"]))
    origin=source/"linear"
    def fixture(name):
        target=a.output/name;shutil.copytree(origin,target);return target
    class MigrationTests(unittest.TestCase):
        def test_actual_provenance_rebind(self):
            old=historical(Path(report["rows"][0]["historical_evidence"]))
            req,cand=rebind(old)
            self.assertNotEqual(req["request_id"],old["request"]["request_id"])
            self.assertEqual(cand["hecate_source"],old["candidate"]["hecate_source"])
            self.assertEqual(req["public_constants"],old["request"]["public_constants"])
        def test_historical_candidate_change_rejected(self):
            case=fixture("changed-history")
            with (case/"historical/candidate.py").open("a") as f:f.write("#changed\n")
            with self.assertRaisesRegex(ValueError,"Historical delivery file changed"):verify_delivery(case)
        def test_program_candidate_change_rejected(self):
            case=fixture("changed-program")
            with (case/"program/candidate.py").open("a") as f:f.write("#changed\n")
            with self.assertRaisesRegex(ValueError,"Program source was changed"):verify_delivery(case)
        def test_new_generation_claim_rejected_after_rebinding(self):
            case=fixture("wrong-origin")
            e=json.loads((case/"delivery.json").read_text());e["new_agent_generation"]=True
            e.pop("binding");e["binding"]=digest(e);(case/"delivery.json").write_bytes(canonical(e))
            with self.assertRaisesRegex(ValueError,"Delivery origin category"):verify_delivery(case)
        def test_missing_response_rejected(self):
            case=fixture("missing-response")
            (case/"historical/response.txt").rename(case/"historical/response-retained.txt")
            with self.assertRaisesRegex(ValueError,"Historical delivery file changed"):verify_delivery(case)
        def test_symlink_historical_source_rejected(self):
            case=fixture("symlink-history")
            path=case/"historical/candidate.py";path.rename(case/"historical/candidate-retained.py")
            path.symlink_to("candidate-retained.py")
            with self.assertRaisesRegex(ValueError,"Historical delivery file changed"):verify_delivery(case)
        def test_unsafe_frozen_path_rejected(self):
            with self.assertRaisesRegex(ValueError,"Historical evidence path"):
                verify_frozen_files(origin,{"../report.json":"0"*64})
        def test_old_cli_backend_is_not_an_execution_choice(self):
            from run_candidate import parse_args
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as c:
                parse_args(["--case",str(origin/"historical/model.json"),
                            "--compiler-configuration","seal-cpu-eva-w45-v1","--self-test"])
            self.assertEqual(c.exception.code,2)
        def test_component_rejects_legacy_before_execution(self):
            from component_backend import qualify
            req=json.loads((origin/"historical/request.json").read_text())
            cand=json.loads((origin/"historical/response.txt").read_text())
            with self.assertRaisesRegex(ValueError,"Historical backend execution is retired"):qualify(req,cand)
        def test_reference_and_weights_preserved(self):
            compare_arrays(Path(report["rows"][0]["historical_evidence"]),origin/"reference-check")
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MigrationTests))
    require(snapshot()==frozen,"Core drift during static acceptance")
    out=dict(format="poseidon-historical-migration-integrity-v1",tests=result.testsRun,
             passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
             failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
             delivery_audits=audit_rows,source_hashes=frozen,
             driver_sha256=sha(Path(__file__)),audit_sha256=sha(Path(__file__).with_name("audit_historical_delivery.py")),
             actual_compile=False,encrypted_execution=False,paid_calls=0)
    (a.output/"report.json").write_bytes(canonical(out))
    return int(not result.wasSuccessful())

if __name__=="__main__":raise SystemExit(main())
