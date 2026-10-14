"""Real-identity source/application admission; no compiler or Provider."""
import copy,hashlib,importlib.abc,io,json,os,sys,time,unittest
from pathlib import Path
import numpy,torch
from benchmark_graph import canonical,digest
from application_backend import PoseidonBackend
from application_component import Application
from application_identity import check_manifest,STAGES
from candidate_bundle import check_provenance
from component_contract import reconstruct_request
from public_weight_context import verify_binding,verify_context
from application_model_source import FORMAT,own_identity
from model_source_provenance import build,verify
ROOT=Path("/home/lhohy/Code/Poseidon");OUT=Path("/out")
PREPARED=Path("/prepared");CAPTURES=Path("/captures")
PLAN=json.loads(Path("/tests/plan.json").read_text())
ROWS={};STATES=[];BLOCKED_EVENTS=[]
class NoSourceImport(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in {"source_models","source_helpers"}:
            raise AssertionError("Admission must not import source factories")
sys.meta_path.insert(0,NoSourceImport())
def no_execution(event,args):
    if event in {"subprocess.Popen","os.system","os.posix_spawn","os.posix_spawnp","os.fork","os.forkpty"}:
        BLOCKED_EVENTS.append(event);raise AssertionError("Admission started a process")
sys.addaudithook(no_execution)
for record in json.loads((PREPARED/"report.json").read_text())["request_rows"]:
    key=record["case"];folder=PREPARED/key
    ROWS[key]=dict(row=record,model=json.loads((folder/"model.json").read_text()),
        provenance=json.loads((folder/"model-provenance.json").read_text()),
        request=json.loads((folder/"request.json").read_text()),
        context=json.loads((folder/"generation-context.json").read_text()))
backend=PoseidonBackend();identity=check_manifest(backend.identity_manifest())
(OUT/"application-identities.json").write_bytes(canonical(identity))
app=Application(OUT/"tasks",backend)
def submit(record,provenance=None):
    return app.submit(record["model"],options=record["row"]["preparation_options"],
                      model_provenance=record["provenance"] if provenance is None else provenance)
def source(key,trust=True):
    return dict(format=FORMAT,kind="trusted_pytorch",
        manifest=str(CAPTURES/"source-fixtures"/key/"manifest.json"),trust_local_code=trust)

class ApplicationAdmissionTests(unittest.TestCase):
    def test_actual_backend_identity_binds_combined_agent_bytes(self):
        self.assertIs(type(backend),PoseidonBackend)
        self.assertEqual(identity["backend"]["format"],"poseidon-backend-identity-v2")
        self.assertEqual(identity["backend"]["platform"]["configured_platform"],"aarch64-linux")
        sources={}
        for stage in STAGES:sources.update(identity[stage]["sources"])
        for name,want in PLAN["targets"].items():self.assertEqual(sources[name],want)
        for relative,want in PLAN["inside_source_hashes"].items():
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),want)
        self.assertEqual(len(PLAN["inside_source_hashes"]),706)
        self.assertNotEqual(identity["manifest_sha256"],PLAN["production_identity_sha256"])
        self.assertEqual(identity["backend"],PLAN["backend_identity"])

    def test_nine_normalized_sources_admit_with_real_identity_and_bound_provenance(self):
        for key,record in ROWS.items():
            with self.subTest(case=key):
                saved=canonical(record["model"]),canonical(record["provenance"])
                check_provenance(record["provenance"],record["model"])
                task=submit(record);state=app.status(task)
                self.assertEqual((state["status"],state["phase"]),("queued","queued"))
                payload=app._load(app._dir(task)/"task.json")
                self.assertEqual(payload["identity_policy"],"staged-v1")
                self.assertEqual(payload["identity"],identity)
                self.assertEqual(payload["binding"],digest({k:v for k,v in payload.items() if k!="binding"}))
                self.assertEqual(canonical(payload["request"]),canonical(record["request"]))
                self.assertEqual(canonical(payload["model_provenance"]),canonical(record["provenance"]))
                self.assertEqual(canonical(reconstruct_request(payload["request"])),canonical(payload["request"]))
                self.assertEqual(verify_context(verify_binding(payload["request"])),record["context"])
                self.assertEqual((canonical(record["model"]),canonical(record["provenance"])),saved)
                self.assertEqual((state["generations"],state["http_attempts"]),(0,0))
                self.assertEqual(state["validation_level"],"numerical")
                self.assertIsNone(state["acceptance"]);self.assertIsNone(state["package"])
                STATES.append(dict(case=key,phase="graph_admission",task=task,request_id=payload["request"]["request_id"],
                    identity_sha256=identity["manifest_sha256"],status=state["status"],
                    provenance_sha256=digest(record["provenance"]),reuse_key=app._load(app._dir(task)/"state.json")["reuse_key"]))

    def test_source_snapshots_keep_real_identity_files_weights_and_unverified_grade(self):
        for key,record in ROWS.items():
            with self.subTest(case=key):
                fixture=CAPTURES/"source-fixtures"/key
                task=app.submit_source(source(key),resource_profile="general-tensors-v1",
                    options=record["row"]["preparation_options"])
                state=app.status(task);path=app._dir(task)
                self.assertEqual((state["status"],state["phase"]),("queued","intake_queued"))
                snapshot=app._load(path/"source.json")
                self.assertEqual(snapshot["identity"],identity)
                self.assertEqual(snapshot["binding"],digest({k:v for k,v in snapshot.items() if k!="binding"}))
                self.assertEqual(snapshot["implementation"],own_identity())
                for name,item in snapshot["files"].items():
                    raw=(path/"source"/name).read_bytes()
                    self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),(item["bytes"],item["sha256"]))
                    if name!="manifest.json":self.assertEqual(raw,(fixture/name).read_bytes())
                self.assertEqual(json.loads((path/"source/manifest.json").read_text()),
                                 record["provenance"]["trusted_pytorch"]["manifest"])
                origin=build(snapshot,record["request"],record["provenance"])
                self.assertEqual(verify(origin,record["request"],record["provenance"],
                    controller_sha256=identity["generation"]["sources"]["application_model_source.py"]),origin)
                self.assertEqual(state["validation_level"],"numerical")
                self.assertFalse(state["intake"]["reference_verified"])
                self.assertFalse(state["intake"]["encrypted_execution"])
                self.assertEqual(state["intake"]["agent_calls"],0)
                self.assertEqual((state["generations"],state["http_attempts"]),(0,0))
                self.assertFalse((path/"task.json").exists())
                self.assertEqual(app.cancel(task)["status"],"cancelled")
                self.assertEqual(app.prepare_source(task)["status"],"cancelled")
                STATES.append(dict(case=key,phase="source_snapshot",task=task,files=len(snapshot["files"]),
                    identity_sha256=identity["manifest_sha256"],status="cancelled",origin_binding=origin["binding"]))

    def test_identical_model_request_has_stable_reuse_identity(self):
        for key in ("addmm_public_right","baddbmm_public_right","addbmm_public_right"):
            tasks=[submit(ROWS[key]) for _ in range(2)]
            states=[app._load(app._dir(task)/"state.json") for task in tasks]
            self.assertNotEqual(*tasks);self.assertEqual(states[0]["reuse_key"],states[1]["reuse_key"])
            payloads=[app._load(app._dir(task)/"task.json") for task in tasks]
            self.assertEqual(payloads[0]["binding"],payloads[1]["binding"])
            self.assertEqual(payloads[0]["identity"],identity)
            for task in tasks:self.assertEqual(app.cancel(task)["status"],"cancelled")

    def test_missing_weight_and_wrong_source_hash_are_public_input_failures(self):
        for key,record in ROWS.items():
            for mutation in ("weight","source"):
                with self.subTest(case=key,mutation=mutation):
                    bad=copy.deepcopy(record["provenance"]);package=bad["trusted_pytorch"]
                    if mutation=="weight":
                        if package["weight_files"]:package["weight_files"].pop(next(iter(package["weight_files"])))
                        else:package["weight_files"]["undeclared.npy"]="not-a-declared-weight"
                    else:package["files"][next(iter(package["files"]))]+="\n# changed\n"
                    task=submit(record,bad);state=app.status(task)
                    self.assertEqual((state["status"],state["phase"]),("unsupported","complete"))
                    self.assertEqual(state["failure"]["code"],"input_or_capability_unsupported")
                    self.assertEqual(state["failure"]["layer"],"input");self.assertEqual(state["failure"]["context"],{})
                    self.assertFalse((app._dir(task)/"task.json").exists())
                    self.assertEqual((state["generations"],state["http_attempts"]),(0,0))

    def test_snapshot_rejects_nontrusted_sources_and_changed_weight_bytes(self):
        task=app.submit_source(source("addmm_public_right",False),resource_profile="general-tensors-v1")
        self.assertEqual(app.status(task)["status"],"unsupported")
        fixture=CAPTURES/"source-fixtures/addmm_public_right";bad=OUT/"invalid-weight-fixture";bad.mkdir()
        manifest=ROWS["addmm_public_right"]["provenance"]["trusted_pytorch"]["manifest"]
        names={"manifest.json",*manifest["files"],*(item["file"] for item in manifest["weights"].values())}
        for name in names:(bad/name).write_bytes((fixture/name).read_bytes())
        name=next(iter(manifest["weights"].values()))["file"]
        raw=bytearray((bad/name).read_bytes());raw[-1]^=1;(bad/name).write_bytes(raw)
        item=source("addmm_public_right");item["manifest"]=str(bad/"manifest.json")
        task=app.submit_source(item,resource_profile="general-tensors-v1")
        self.assertEqual(app.status(task)["status"],"unsupported")
        self.assertEqual(app.status(task)["failure"]["code"],"model_source_unsupported")
        self.assertEqual((app.status(task)["generations"],app.status(task)["http_attempts"]),(0,0))

    def test_real_request_cancellation_never_promotes_validation(self):
        task=submit(ROWS["addmm_public_right"])
        self.assertEqual(app.status(task)["status"],"queued")
        status=app.cancel(task)
        self.assertEqual((status["status"],status["phase"]),("cancelled","complete"))
        self.assertEqual(app.cancel(task),status)
        self.assertIsNone(status["acceptance"]);self.assertIsNone(status["generation"])
        self.assertEqual((status["generations"],status["http_attempts"]),(0,0))

    def test_zero_processes_source_imports_and_paid_calls(self):
        self.assertEqual(len(ROWS),9);self.assertFalse(BLOCKED_EVENTS)
        self.assertFalse(any(n.split(".")[0] in ("source_models","source_helpers") for n in sys.modules))
        for key in ("POSEIDON_PROBE_SECRET","DEEPSEEK_API_KEY","IN_NIX_SHELL"):self.assertNotIn(key,os.environ)
        self.assertNotEqual(str(Path("/proc/self/ns/net").readlink()),PLAN["host_network_namespace"])

stream=io.StringIO();started=time.monotonic()
selection=sys.argv[1] if len(sys.argv)==2 else "all"
if selection=="all":suite=unittest.defaultTestLoader.loadTestsFromTestCase(ApplicationAdmissionTests)
elif selection=="integrity":suite=unittest.defaultTestLoader.loadTestsFromName(
    "__main__.ApplicationAdmissionTests.test_missing_weight_and_wrong_source_hash_are_public_input_failures")
else:raise ValueError("Explicit test selection")
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report=dict(format="poseidon-source-real-identity-admission-tests-v1",status="passed" if result.wasSuccessful() else "failed",
 tests=result.testsRun,suite=selection,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),source_models=len(ROWS),records=STATES,
 combined_plan_binding=PLAN["combined_plan_binding"],combined_sources=706,identity_sha256=identity["manifest_sha256"],
 backend_identity=digest(identity["backend"]),identity_type="actual PoseidonBackend, no injected identity",
 runtime=dict(python=list(sys.version_info[:3]),numpy=numpy.__version__,torch=torch.__version__,cuda=torch.version.cuda),
 source_factory_calls=0,compiler_calls=0,encrypted_executions=0,provider_calls=0,
 subprocess_attempts=BLOCKED_EVENTS,new_dependencies=0,production_applied=False,seconds=time.monotonic()-started)
(OUT/"tests.log").write_text(stream.getvalue())
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(stream.getvalue()[-18000:] if not result.wasSuccessful() else stream.getvalue()[-500:])
print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","source_models","identity_sha256")}))
raise SystemExit(0 if result.wasSuccessful() else 1)
