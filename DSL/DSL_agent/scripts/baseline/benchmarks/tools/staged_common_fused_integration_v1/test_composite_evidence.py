
"""Synthetic control tests; no backend, approval creation or source cutover."""
import copy
import hashlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import composite_evidence as c


def seal(value):
    value = copy.deepcopy(value)
    value.pop("binding", None)
    value["binding"] = c.digest(value)
    return value


class Fixture:
    def __init__(self):
        names = c.original.CASES
        req = {n: {"request_id": c.digest(n)} for n in names}
        refs = {n: {"model_sha256": c.digest(n + "-model"),
                    "original_sha256": c.digest(n + "-original")} for n in names}
        original_sources = {"synthetic/%03d.py" % i: "1" * 64 for i in range(704)}
        inside_sources = dict(original_sources, **{"synthetic/new-a.py": "2"*64, "synthetic/new-b.py": "3"*64})
        self.integration = seal(dict(format="poseidon-common-fused-integration-v1",
            stage="prepared_pending_backend_qualification", production_applied=False,
            root=str(c.original.ROOT), original_production_hashes=original_sources,
            inside_source_hashes=inside_sources, protected_sources={}))
        common = dict(original_production=original_sources, inside_sources=inside_sources,
                      protected_sources={"synthetic/protected.py":"4"*64},
                      backend_identity={"format":"synthetic-control-only"},
                      combined_plan_binding=self.integration["binding"])
        self.contexts = {}
        self.cases = {}
        self.files = {}
        for label, scope in (("original",c.original),("capacity",c.capacity)):
            bindings = copy.deepcopy(common)
            bindings["requests"] = {n: copy.deepcopy(req[n]) for n in scope.CASES}
            bindings["source_reference"] = {n: copy.deepcopy(refs[n]) for n in scope.CASES}
            if label == "capacity":
                bindings["candidate_hashes"] = {n:c.digest(n+"-candidate") for n in scope.CASES}
            proposal = seal(dict(format=scope.VERSION,cases=list(scope.CASES),phases=list(scope.PHASES),
                candidate_origin=scope.CANDIDATE_ORIGIN,new_agent_generation=False,production_applied=False,
                bootstrap=False,cuda=False,bindings=bindings,scope=copy.deepcopy(scope.SCOPE),
                candidate_strategy="lower-balanced-scalar-v1" if label=="original" else c.capacity.STRATEGY))
            runtime = seal(dict({k:v for k,v in proposal.items() if k!="binding"},
                started_wall=1,deadline_wall=3601,initial_result_directories=[],readonly_result_entries=[]))
            metadata = {str(scope.RUN / f):c.digest(label+f) for f in
                        ("plan.json","report.json","staged-sources.json","namespace-receipt.json")}
            metadata.update({str(scope.PROPOSAL):c.digest(label+"proposal"),str(scope.APPROVAL):c.digest(label+"approval")})
            self.files.update(metadata)
            rows = [dict(case=n,phase=p,status="passed",exit_code=0)
                    for n in c.selected_cases(label) for p in scope.PHASES]
            if label == "original":
                rows += [dict(case=n,phase="qualification",status="failed",exit_code=1)
                         for n in c.capacity.CASES]
            report = dict(format="poseidon-ordered-common-offline-report-v1",binding=runtime["binding"],
                provider_calls=0,http_attempts=0,new_agent_generation=False,production_applied=False,
                candidate_origin=scope.CANDIDATE_ORIGIN,status="failed" if label=="original" else "passed",
                rows=rows,pending=[] if label=="capacity" else [{"status":"not_run"}],
                completed_phases=3 if label=="original" else 33,compilations=1 if label=="original" else 11)
            self.contexts[label] = dict(label=label,scope=scope,proposal=proposal,runtime=runtime,
                                       report=report,reader=types.SimpleNamespace(files=metadata))
            for n in c.selected_cases(label):
                evidence = {str(scope.RUN/n/(p+".json")):c.digest(n+p) for p in scope.PHASES}
                self.cases[n] = dict(case=n,status="passed",candidate_origin=scope.CANDIDATE_ORIGIN,
                    runtime_binding=runtime["binding"],source_count=706,request_id=req[n]["request_id"],
                    backend_identity_sha256=c.digest(common["backend_identity"]),
                    reference_semantics="frozen_reviewed_runner",
                    compilations=1,qualification_encrypted_phases=1,new_input_encrypted_phases=1,
                    cache_generations=0,provider_calls=0,http_attempts=0,
                    candidate_sha256=c.digest(n+"-candidate"),evidence_files=evidence)
        self.original_report_before = copy.deepcopy(self.contexts["original"]["report"])

    def value(self):
        return c.assemble(self.contexts,self.integration,self.cases,copy.deepcopy(self.files))


class CompositeControls(unittest.TestCase):
    def setUp(self):
        self.f=Fixture()

    def test_complete_two_source_index_preserves_failed_original(self):
        value=self.f.value()
        self.assertEqual(value["passed"],12)
        self.assertEqual(value["phases"],36)
        self.assertEqual(value["source_runs"]["original"]["selected_cases"],["common_boolean"])
        self.assertEqual(value["source_runs"]["capacity"]["selected_cases"],list(c.capacity.CASES))
        self.assertEqual(self.f.contexts["original"]["report"],self.f.original_report_before)
        self.assertEqual(self.f.contexts["original"]["report"]["status"],"failed")

    def test_every_case_retains_its_own_origin_and_runtime(self):
        value=self.f.value()
        for name, row in value["cases"].items():
            label="original" if name=="common_boolean" else "capacity"
            self.assertEqual(row["candidate_origin"],self.f.contexts[label]["runtime"]["candidate_origin"])
            self.assertEqual(row["runtime_binding"],value["source_runs"][label]["runtime_binding"])

    def test_missing_case_cannot_be_called_complete(self):
        self.f.cases.pop(c.capacity.CASES[-1])
        with self.assertRaisesRegex(ValueError,"twelve_cases"):self.f.value()

    def test_missing_run_cannot_be_called_complete(self):
        self.f.contexts.pop("capacity")
        with self.assertRaisesRegex(ValueError,"both_composite"):self.f.value()

    def test_backend_source_and_reference_identity_mismatch_rejected(self):
        for field in c.SHARED_BINDINGS:
            f=Fixture()
            f.contexts["capacity"]["runtime"]["bindings"][field]={"different":True}
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"identity_mismatch"):f.value()
        for field in ("requests","source_reference"):
            f=Fixture();f.contexts["capacity"]["runtime"]["bindings"][field][c.capacity.CASES[0]]={"different":True}
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"request_mismatch"):f.value()

    def test_extra_or_missing_request_rejected(self):
        new=self.f.contexts["capacity"]["runtime"]["bindings"]["requests"]
        new["common_boolean"]=copy.deepcopy(self.f.contexts["original"]["runtime"]["bindings"]["requests"]["common_boolean"])
        with self.assertRaisesRegex(ValueError,"exact_request_inventory"):self.f.value()

    def test_case_cannot_change_provenance_to_agent(self):
        self.f.cases[c.capacity.CASES[0]]["candidate_origin"]="agent"
        with self.assertRaisesRegex(ValueError,"case_source_binding"):self.f.value()

    def test_case_cannot_claim_other_runtime_or_sdk(self):
        for key in ("runtime_binding","backend_identity_sha256","request_id","reference_semantics"):
            f=Fixture();f.cases[c.capacity.CASES[0]][key]="different"
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"case_source_binding"):f.value()

    def test_candidate_hash_cannot_be_substituted(self):
        self.f.cases[c.capacity.CASES[0]]["candidate_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"frozen_capacity_candidate"):self.f.value()

    def test_boolean_integer_counts_and_unobserved_phases_rejected(self):
        for key in ("compilations","qualification_encrypted_phases","new_input_encrypted_phases",
                    "cache_generations","provider_calls","http_attempts"):
            f=Fixture();f.cases[c.capacity.CASES[0]][key]=True
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"case_accounting"):f.value()
        self.f.cases[c.capacity.CASES[0]]["evidence_files"].popitem()
        with self.assertRaisesRegex(ValueError,"phase_evidence"):self.f.value()

    def test_evidence_hash_conflicts_rejected(self):
        shared="/synthetic/shared.py"
        self.f.cases["common_boolean"]["evidence_files"][shared]="a"*64
        self.f.cases[c.capacity.CASES[0]]["evidence_files"][shared]="b"*64
        with self.assertRaisesRegex(ValueError,"conflicting_composite"):self.f.value()

    def test_illegal_evidence_entries_rejected(self):
        for entries in ({"relative.py":"a"*64},{"/absolute.py":"wrong"},{" /absolute.py":"a"*64},{}):
            with self.subTest(entries=entries),self.assertRaises(ValueError):c.merge_files({},entries)

    def test_current_index_does_not_claim_new_execution(self):
        value=self.f.value()
        self.assertEqual([value[k] for k in ("provider_calls","compiler_calls","encrypted_executions")],[0,0,0])
        self.assertTrue(value["evidence_observed_only"])
        self.assertFalse(value["agent_generated"]);self.assertFalse(value["production_applied"])

    def test_capacity_requires_all_thirty_three_passed_phases(self):
        f=self.f;report=f.contexts["capacity"]["report"];runtime=f.contexts["capacity"]["runtime"]
        c.validate_batch(report,runtime,"capacity")
        for mutate in ("missing","duplicate","failed","pending","badcount","running","skip","boolean_exit"):
            bad=copy.deepcopy(report)
            if mutate=="missing":bad["rows"].pop()
            elif mutate=="duplicate":bad["rows"][-1]=bad["rows"][0]
            elif mutate in ("failed","skip"):bad["rows"][0]["status"]=mutate
            elif mutate=="boolean_exit":bad["rows"][0]["exit_code"]=False
            elif mutate=="pending":bad["pending"]=[{"status":"not_run"}]
            elif mutate=="running":bad["status"]="running"
            else:bad["completed_phases"]=True
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):c.validate_batch(bad,runtime,"capacity")

    def test_original_selected_pass_keeps_other_failures(self):
        f=self.f;report=f.contexts["original"]["report"];runtime=f.contexts["original"]["runtime"]
        c.validate_batch(report,runtime,"original")
        self.assertEqual(report["status"],"failed")
        bad=copy.deepcopy(report);bad["status"]="passed"
        with self.assertRaisesRegex(ValueError,"preserved_original"):c.validate_batch(bad,runtime,"original")
        bad=copy.deepcopy(report);bad["rows"][0]["status"]="skipped"
        with self.assertRaisesRegex(ValueError,"selected_phase_not_passed"):c.validate_batch(bad,runtime,"original")

    def test_unknown_source_label_rejected(self):
        with self.assertRaisesRegex(ValueError,"recognized_composite"):c.selected_cases("any")

    def test_runtime_preserves_exact_proposal(self):
        for label in c.LABELS:
            ctx=self.f.contexts[label];scope=ctx["scope"]
            with patch.object(scope,"check_plan") as checker:
                c.validate_runtime_proposal(ctx["runtime"],ctx["proposal"],scope)
                checker.assert_called_once_with(ctx["runtime"])
            changed=copy.deepcopy(ctx["runtime"]);changed["scope"]={"changed":True};changed=seal(changed)
            with self.subTest(label=label),self.assertRaisesRegex(ValueError,"preserves_composite"):
                c.validate_runtime_proposal(changed,ctx["proposal"],scope)

    def test_rehashed_candidate_changes_rejected_by_independent_reaudit(self):
        good=self.f.value()
        with patch.object(c,"qualify",return_value=good) as reread:
            self.assertEqual(c.validate_evidence(good,self.f.integration),good)
            reread.assert_called_once_with(self.f.integration)
        mutations=(
            lambda v:v["cases"].pop(c.capacity.CASES[0]),
            lambda v:v["cases"][c.capacity.CASES[0]].update(candidate_sha256="0"*64),
            lambda v:v["source_runs"]["original"].update(runtime_binding="0"*64),
            lambda v:v["source_runs"]["original"].update(report_sha256="0"*64),
            lambda v:v["evidence_files"].popitem(),
        )
        for mutate in mutations:
            bad=copy.deepcopy(good);mutate(bad);bad=seal(bad)
            with self.subTest(mutate=mutate),patch.object(c,"qualify",return_value=good),self.assertRaisesRegex(ValueError,"independent_evidence_chain"):
                c.validate_evidence(bad,self.f.integration)

    def test_output_provenance_and_boolean_counts_rejected(self):
        good=self.f.value()
        for key,value in (("passed",True),("phases",True),("agent_generated",True),
                          ("production_applied",True),("provider_calls",1),("encrypted_executions",1),
                          ("evidence_observed_only",False),("extra_field",0)):
            bad=copy.deepcopy(good);bad[key]=value;bad=seal(bad)
            with self.subTest(key=key),patch.object(c,"qualify") as reader,self.assertRaisesRegex(ValueError,"qualification_schema"):
                c.validate_evidence(bad,self.f.integration)
            reader.assert_not_called()

    def test_missing_approval_stops_before_reading_run(self):
        reader=Mock()
        with patch.object(c.capacity,"before_launch",side_effect=ValueError("Pending independent offline approval")):
            with self.assertRaisesRegex(ValueError,"Pending independent"):c.load_run("capacity",c.capacity,reader)
        reader.json.assert_not_called()

    def test_namespace_record_requires_actual_isolation(self):
        # Valid before_launch and plan checks are fabricated; this is not approval.
        ctx=self.f.contexts["capacity"];scope=ctx["scope"]
        responses=[ctx["proposal"],{},ctx["runtime"],ctx["runtime"]["bindings"]["inside_sources"],
                   dict(actual_nix_shell="pure",staged_modules=10,project_credential_path_absent=False,
                        production_applied=False,provider_calls=0)]
        runtime=copy.deepcopy(ctx["runtime"]);runtime["bindings"]["files"]={}
        responses[2]=runtime
        with patch.object(scope,"before_launch"),patch.object(scope,"check_approval"),patch.object(c,"validate_runtime_proposal"),patch.object(c,"Reader"):
            reader=Mock();reader.files={str(scope.PROPOSAL):"a"*64};reader.json.side_effect=responses
            with self.assertRaisesRegex(ValueError,"isolated_namespace"):c.load_run("capacity",scope,reader)

    def test_reader_rejects_symlinked_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"original.json";p.write_text("{}");link=Path(t)/"link.json";link.symlink_to(p)
            with self.assertRaisesRegex(ValueError,"regular_evidence"):c.Reader().json(link)

    def test_captured_approval_is_verified_even_after_initial_gate_succeeds(self):
        for label in c.LABELS:
            ctx=self.f.contexts[label];scope=ctx["scope"];proposal=ctx["proposal"]
            proposal_hash="a"*64;tools={"synthetic-tool.py":"b"*64}
            receipt=dict(format="poseidon-common-fused-"+("capacity-" if label=="capacity" else "")+"offline-approval-v1",
                approved=True,proposal_binding=proposal["binding"],proposal_sha256=proposal_hash,
                driver_hashes=tools,human_message="synthetic unit control")
            invalid=[{},dict(receipt,approved=False),dict(receipt,proposal_sha256="c"*64),
                     dict(receipt,proposal_binding="d"*64),dict(receipt,driver_hashes={})]
            for captured in invalid:
                reader=Mock();reader.files={str(scope.PROPOSAL):proposal_hash}
                reader.json.side_effect=[proposal,captured]
                with self.subTest(label=label,captured=captured),patch.object(scope,"before_launch"),\
                     patch.object(scope,"driver_hashes",return_value=tools),self.assertRaises(ValueError):
                    c.load_run(label,scope,reader)
                self.assertEqual(reader.json.call_count,2)
            reader=Mock();reader.files={str(scope.PROPOSAL):proposal_hash}
            reader.json.side_effect=[proposal,receipt,ctx["runtime"]]
            with patch.object(scope,"before_launch"),patch.object(scope,"driver_hashes",return_value=tools),\
                 patch.object(scope,"check_approval",wraps=scope.check_approval) as verify,\
                 patch.object(c,"validate_runtime_proposal",side_effect=ValueError("test stops after captured approval")):
                with self.assertRaisesRegex(ValueError,"test stops after"):
                    c.load_run(label,scope,reader)
                verify.assert_called_once_with(proposal,receipt,proposal_hash,tools)

    def test_early_case_evidence_changed_during_later_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"first-case.json";p.write_text('{"status":"passed"}')
            first=c.Reader();first.json(p);first_files=first.finish()
            # A later case's processing changes a file already validated by the
            # first case's separate reader. Its previous SHA must not be accepted.
            p.write_text('{"status":"failed"}')
            value={"evidence_files":first_files}
            with self.assertRaisesRegex(ValueError,"composite_evidence_changed"):
                c.finalize_files(value,c.Reader())

    def test_change_during_final_union_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/"first.json";a.write_text("{}")
            b=Path(t)/"later.json";b.write_text("{}")
            files={}
            for p in (a,b):
                files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            class MutatingReader(c.Reader):
                def bytes(self,path,limit=c.Reader.bytes.__defaults__[0]):
                    raw=super().bytes(path,limit)
                    if Path(path)==b:
                        a.write_text('{"changed":true}')
                    return raw
            with self.assertRaisesRegex(ValueError,"evidence_changed_during_audit"):
                c.finalize_files({"evidence_files":files},MutatingReader())

    def test_final_union_is_checked_with_bounded_reader(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/"first.json";a.write_text("{}")
            b=Path(t)/"second.json";b.write_text("{}")
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a,b)}
            value={"evidence_files":dict(sorted(files.items()))}
            reader=c.Reader()
            self.assertIs(c.finalize_files(value,reader),value)
            self.assertEqual(reader.finish(),value["evidence_files"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
