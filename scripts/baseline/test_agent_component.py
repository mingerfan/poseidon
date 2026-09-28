"""Component contracts and bundle compatibility; no network and no native execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from benchmark_graph import digest, canonical
from benchmark_suite import Builder
from component_contract import (VERSION, prepare_task, reconstruct_request, runner_options,
                                validate_program, synthesize, failure)
from candidate_contract import ReplayProvider
from candidate_bundle import check_bundle, sha, implementation
from compiler_configuration import configuration

def fixture(public=False, directed=False, chunk=False):
    b=Builder([(6 if chunk else 2,)])
    model=b.finish(b.node("square",["input0"]))
    opts=dict(compiler_configuration="seal-cpu-eva-w45-v1")
    if public:opts["construction_profile"]="hecate-unified-public-v1"
    if directed:opts["construction"]="unified-public-call-list"
    if chunk:opts["chunk_period"]=4
    return prepare_task(model,opts)

def response(request,source=None):
    return dict(schema=1,request_id=request["request_id"],hecate_source=source or
                '@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [x*x]\n')

class ComponentTests(unittest.TestCase):
    def test_reconstruct_all_switches_and_legacy_absence(self):
        for public,chunk in [(False,False),(True,False),(False,True),(True,True)]:
            r=fixture(public,public,chunk)
            self.assertEqual(reconstruct_request(r),r)
            flags=runner_options(r)
            self.assertEqual("--unified-profile" in flags,public)
            self.assertEqual("--unified-chunk-period" in flags,chunk)
        r=prepare_task(fixture()["model"],dict(compiler_configuration=None))
        self.assertNotIn("--compiler-configuration",runner_options(r))
        self.assertEqual(reconstruct_request(r),r)
    def test_directed_failure_is_not_illegal_program(self):
        r=fixture(True,True)
        out=validate_program(response(r),r)
        self.assertEqual(out["semantic_validation"],"accepted")
        self.assertEqual(out["construction_coverage"],"not_verified")
        self.assertEqual(out["status"],"rejected")
        self.assertFalse(out["numerically_validated"])
    def test_legal_static_never_claims_execution(self):
        r=fixture();out=validate_program(response(r),r)
        self.assertEqual(out["status"],"static_accepted_not_executed")
        self.assertFalse(out["encrypted_execution"])
    def test_invalid_candidate_still_rejected(self):
        r=fixture(True,True);out=validate_program(response(r,"import os\n"),r)
        self.assertEqual(out["semantic_validation"],"not_checked")
        self.assertEqual(out["status"],"rejected")
    def test_witness_obtained_still_not_fhe(self):
        r=fixture(True,True)
        out=validate_program(response(r,'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return list([x*x])\n'),r)
        self.assertEqual(out["construction_coverage"],"static_witness_obtained")
        self.assertFalse(out["encrypted_execution"])
    def test_changed_request_rejected(self):
        r=fixture();r["layout"]["input_slot_period"]=8
        with self.assertRaises(ValueError):reconstruct_request(r)
    def test_unrecognized_option_rejected(self):
        with self.assertRaises(ValueError):prepare_task(fixture()["model"],{"shell":"echo bad"})
    def test_loop_cannot_call_static_result_a_success(self):
        r=fixture()
        with self.assertRaisesRegex(ValueError,"encrypted numerical"):
            synthesize(r,ReplayProvider(["{}"]),lambda raw,i:dict(status="passed"),lambda *x:None,max_repairs=0)
    def test_injected_loop_is_offline_and_bounded(self):
        r=fixture()
        out=synthesize(r,ReplayProvider(["{}"]),lambda raw,i:dict(status="failed",public_feedback=dict(status="failed",layer="static_check")),
                       lambda *x:None,max_repairs=0)
        self.assertEqual(out["status"],"repair_budget_exhausted")
        self.assertEqual(out["agent_calls"],0)
    def test_loop_filters_private_evidence_and_uses_qualified_success(self):
        r=fixture();seen=[];recorded=[]
        class Provider:
            kind="scripted_replay";agent_calls=0
            def generate(self,request,history):
                seen.append(history)
                return "{}"
        def evaluate(raw,index):
            if index==0:
                return dict(status="failed",evidence="/private/hidden",reference=[123],
                            public_feedback=dict(status="failed",layer="static_check",diagnostic="Type mismatch"))
            return dict(status="passed",encrypted_execution=True,numerically_validated=True,
                        evidence="/private/hidden",reference=[123])
        out=synthesize(r,Provider(),evaluate,lambda i,raw,feedback:recorded.append(feedback),max_repairs=1)
        self.assertEqual(out["status"],"passed")
        self.assertEqual(seen[1],[dict(status="failed",layer="static_check",diagnostic="Type mismatch")])
        self.assertNotIn("private",json.dumps(recorded))
        self.assertNotIn("reference",json.dumps(recorded))
    def test_loop_rejects_private_fields_in_public_feedback(self):
        r=fixture()
        with self.assertRaises(Exception):
            synthesize(r,ReplayProvider(["{}"]),lambda raw,i:dict(status="failed",
                public_feedback=dict(status="failed",layer="static_check",reference=[123])),lambda *x:None,max_repairs=0)
    def test_worker_protocol_outside_checkout(self):
        script=Path(__file__).resolve().parents[1]/"agent_component.py"
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)/"job.json"
            job.write_text(json.dumps(dict(format=VERSION,action="prepare",model=fixture()["model"],options={})))
            p=subprocess.run([sys.executable,"-B",str(script),"--job",str(job)],cwd=d,capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,0,p.stderr)
            out=json.loads(p.stdout);self.assertEqual(out["status"],"prepared_not_executed")
            job.write_text(json.dumps(dict(format=VERSION,action="live",api_key="not-a-key")))
            p=subprocess.run([sys.executable,"-B",str(script),"--job",str(job)],cwd=d,capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,2);self.assertEqual(json.loads(p.stdout)["paid_calls"],0)
    def test_worker_rejects_changed_job_binding(self):
        script=Path(__file__).resolve().parents[1]/"agent_component.py"
        with tempfile.TemporaryDirectory() as d:
            job=Path(d)/"job.json"
            job.write_text(json.dumps(dict(format=VERSION,action="capabilities")))
            p=subprocess.run([sys.executable,"-B",str(script),"--job",str(job),"--job-sha256","0"*64],
                             capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,2)
            self.assertIn("changed",json.loads(p.stdout)["failure"]["diagnostic"])
    def test_component_import_does_not_change_process_state(self):
        script="import os,sys;before=dict(os.environ);import component_contract;assert before==dict(os.environ);assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,"-B","-c",script],cwd=Path(__file__).parent,check=True,timeout=20)

class BundleV2Tests(unittest.TestCase):
    def bundle(self,folder,request=None):
        r=request or fixture()
        for n,v in [("model.json",r["model"]),("request.json",r)]:
            (folder/n).write_text(json.dumps(v))
        (folder/"candidate.py").write_text(response(r)["hecate_source"])
        m=dict(format="poseidon-dsl-bundle-v2",files={n:sha(folder/n) for n in ["model.json","request.json","candidate.py"]},
               compiler_configuration=r.get("compiler_configuration") or configuration("seal-cpu-eva-w40-v1"),
               origin=dict(request_id=r["request_id"]),agent_generated=False,model_provenance="not_supplied",
               dependencies=dict(compiler_sha256="a"*64,helper_environment=None,implementation=implementation(),validation_sources={}))
        self.save(folder,m);return m
    def save(self,p,m):
        m["binding"]=digest({k:v for k,v in m.items() if k!="binding"})
        (p/"manifest.json").write_text(json.dumps(m))
    def test_public_chunk_context_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.bundle(p,fixture(True,True,True))
            self.assertEqual(check_bundle(p)["format"],"poseidon-dsl-bundle-v2")
    def test_missing_request_rejected_even_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p);del m["files"]["request.json"];self.save(p,m)
            with self.assertRaisesRegex(ValueError,"Unexpected"):check_bundle(p)
    def test_stripped_profile_rejected_even_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p,fixture(True))
            r=json.loads((p/"request.json").read_text());del r["construction_profile"]
            r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
            (p/"request.json").write_text(json.dumps(r));m["files"]["request.json"]=sha(p/"request.json");self.save(p,m)
            with self.assertRaises(ValueError):check_bundle(p)
    def test_model_changed_rejected_even_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p);model=json.loads((p/"model.json").read_text());model["id"]="different"
            (p/"model.json").write_text(json.dumps(model));m["files"]["model.json"]=sha(p/"model.json");self.save(p,m)
            with self.assertRaisesRegex(ValueError,"model/request"):check_bundle(p)
    def test_manifest_not_numerical_proof(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p)
            self.assertFalse(check_bundle(p)["agent_generated"])
            self.assertNotIn("validated",m)
    def test_python_provenance_roundtrip_and_tamper(self):
        from restricted_model_python import parse
        from candidate_bundle import check_provenance
        base=Path(__file__).parent/"cases/operator-decomposition-v1"
        manifest=json.loads((base/"mlp-python.json").read_text())
        files={n:(base/n).read_text() for n in manifest["files"]}
        original,model,binding=parse(manifest,files)
        value=dict(original=original,decomposition=binding,python=dict(manifest=manifest,files=files),
                   reference_report=dict(original_sha256=digest(original),model_sha256=digest(model),
                       reference_status="passed",approximation_accuracy_certified=False))
        check_provenance(value,model)
        value["python"]["files"]["helper.py"]+="\n# tampered"
        with self.assertRaises(ValueError):check_provenance(value,model)
    def test_payload_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p)
            (p/"candidate.py").rename(p/"saved.py");(p/"candidate.py").symlink_to(p/"saved.py")
            with self.assertRaisesRegex(ValueError,"integrity"):check_bundle(p)

if __name__=="__main__":unittest.main()
