"""Freeze current component/source backend scope. Never launches backend or API."""
import copy, hashlib, json, os, sys
from pathlib import Path

from benchmark_graph import canonical, digest, require
from hecate_python_env import ROOT, WORK, VENV
from semantic_benchmark_execution import runtime_sources
from application_backend import PoseidonBackend
from application_policy import prepare
from candidate_bundle import load_model_provenance, check_provenance
from pytorch_model_intake import source_identity, preparation_options
from trusted_pytorch_import import source_identity as legacy_identity
from operator_graph import implementation_sources as matrix_identity
from operator_arithmetic_graph import implementation_sources as arithmetic_identity, preparation_options as graph_options
from unified_graph_lowering import candidate_source
from model_source_provenance import reuse_identity, verify

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def checked(v):
    require(digest({k:x for k,x in v.items() if k != "binding"}) == v["binding"], "Frozen JSON binding")
    return v

def swap(code, old, new, count=1):
    require(code.count(old) == count, "Driver template mismatch: "+old[:100])
    return code.replace(old, new)

def run():
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV, "Pinned pure source freeze")
    pre=WORK/"results/agent-model-source-integration-r1012"
    old_path=ROOT/"docs/baseline/agent-operator-shape-offline-proposal-r991.json"
    old=checked(json.loads(old_path.read_text()))
    audit_dir=pre/"final-actual-context-r1018"
    intake=checked(json.loads((audit_dir/"report.json").read_text()))
    units=json.loads((pre/"final-units-r1018.json").read_text())
    require(intake["passed"]==6 and intake["failed"]==0 and intake["reference_groups"]==96,
            "Production source intake accepted")
    require(units["run"]==units["passed"]==185 and units["failed"]==units["errors"]==units["skipped"]==0,
            "Current final component tests")
    require(len(old["cases"])==112, "Original unexecuted scope")
    frozen_old={**old["case_hashes"],**old["tool_hashes"],str(old_path):sha(old_path)}
    require(all(sha(p)==h for p,h in frozen_old.items()),"Historical cases, candidates, reports and drivers preserved")
    current_sources=runtime_sources()
    capture=source_identity();legacy=legacy_identity()
    matrices=matrix_identity();arithmetic=arithmetic_identity()
    backend=PoseidonBackend().identity()
    require(capture==old["capture_identity"] and legacy==old["legacy_capture_identity"] and
            matrices==old["matrix_identity"] and arithmetic==old["arithmetic_identity"],
            "No capture, mathematical decomposition or reference source changes")
    require(backend["tools"]==old["backend_identity"]["tools"], "No SDK/library/tool change")
    changed=[p for p,h in old["source_hashes"].items() if sha(ROOT/p)!=h]
    require(set(changed)=={"scripts/baseline/application-source-catalog-v1.json",
        "scripts/baseline/application_backend.py","scripts/baseline/application_component.py",
        "scripts/baseline/candidate_bundle.py"}, "Only approved production integration roots changed")
    extra={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"scripts/baseline/application_execution").rglob("*.py")}
    rows=copy.deepcopy(old["cases"])
    prepared=pre/"prepared-source-r1020";require(not prepared.exists(),"Fresh source preparation")
    prepared.mkdir()
    new_files={}
    for observed in intake["rows"]:
        task=audit_dir/"tasks"/observed["task_id"]
        meta=json.loads((task/"task.json").read_text())
        spec=checked(json.loads((task/"source.json").read_text()))
        folder=task/"model-import"
        model=json.loads((folder/"model.json").read_text())
        provenance=load_model_provenance(folder,model)
        check_provenance(provenance,model)
        verify(meta["model_source"],meta["request"],provenance,
               controller_sha256=intake["implementation"]["controller_sha256"])
        options=preparation_options(provenance,model,spec["requirements"]["options"])
        options=graph_options(provenance["original"],model,options)
        request=prepare(model,"poseidon_cpu","numerical",options)[0]
        require(canonical(request)==canonical(meta["request"]),"Source request unchanged")
        name="source_"+observed["id"]
        target=prepared/name;target.mkdir()
        (target/"request.json").write_bytes(canonical(request))
        source=candidate_source(request)[0]
        (target/"candidate.py").write_text(source)
        kind=spec["kind"]
        if kind=="graph":
            manifest=task/"source/model.json"
            api=dict(format="poseidon-model-source-request-v1",kind=kind,model=json.loads(manifest.read_text()))
        else:
            manifest=task/"source/manifest.json"
            api=dict(format="poseidon-model-source-request-v1",kind=kind,manifest=str(manifest))
            if kind=="trusted_pytorch":api["trust_local_code"]=True
        files={str(p):sha(p) for p in task.rglob("*") if p.is_file()}
        files.update({str(p):sha(p) for p in target.iterdir()})
        new_files.update(files)
        rows.append(dict(id=name,input_kind="component_model_source_v1",manifest=str(manifest),
            imported=str(folder),request=str(target/"request.json"),candidate=str(target/"candidate.py"),
            request_id=request["request_id"],model_sha256=digest(model),candidate_sha256=sha(target/"candidate.py"),
            options=options,source_api=api,
            source_requirements=dict(options=spec["requirements"]["options"],resource_profile=spec["resource_profile"]),
            source_receipt=meta["model_source"],source_reuse_identity=reuse_identity(meta["model_source"]),
            source_origin="model_source_rule_program",historical_status="new_input_not_run",
            historical_batch="Production source intake passed; backend never run",files=files,
            current_binding_reference_checked=True,fresh_reference_probes=16))
    require(len(rows)==118,"112 original cases plus six source contexts")
    for i,row in enumerate(rows):row["shard"]=1+i//48
    tools=ROOT/"scripts/baseline/benchmarks/tools/frozen_model_source_backend"
    require(not tools.exists(),"Fresh current frozen drivers")
    old_tools=ROOT/"scripts/baseline/benchmarks/tools/frozen_operator_shape_backend"
    code={}
    for name in ("backend.py","audit.py","job.py","launch.py"):
        s=(old_tools/name).read_text()
        s=s.replace("agent-pytorch-operator-shape-integration-r985","agent-model-source-integration-r1012")
        s=s.replace("agent-operator-shape-backend-r992","agent-application-source-backend-r1023")
        s=s.replace("agent-operator-shape-offline-proposal-r991.json","agent-application-source-offline-proposal-r1021.json")
        s=s.replace("r992","r1023").replace("r993","r1024")
        s=s.replace("poseidon-arithmetic-matrix-offline","poseidon-application-source-offline")
        s=s.replace("==112,","==118,")
        code[name]=s
    s=code["backend.py"]
    s=swap(s,"proposal=load(proposal_path);scope=","proposal=load(proposal_path);case_by_id={c['id']:c for c in proposal['cases']};scope=")
    s=swap(s,'    require(all(sha(p)==h for p,h in proposal["tool_hashes"].items()),"Frozen batch tools")',
        '    require(all(sha(p)==h for p,h in proposal["tool_hashes"].items()),"Frozen batch tools")\n'
        '    require(all(sha(root/p)==h for p,h in proposal["extra_source_hashes"].items()),"Context adapter source freeze")')
    start=s.index('            if case["input_kind"] in (')
    end=s.index('            row["input_checked"]=True',start)
    old_block=s[start:end]
    prefix='''            if case.get("source_api") is not None:
                from model_source_provenance import verify,reuse_identity
                verify(case["source_receipt"],load(case["request"]),provenance)
                require(reference["reference_status"]=="passed" and reference["probes"]==16,
                        "Current independently referenced source intake")
                require(reuse_identity(case["source_receipt"])==case["source_reuse_identity"],
                        "Frozen semantic source identity")
            else:
'''
    s=s[:start]+prefix+"".join("    "+line if line.strip() else line for line in old_block.splitlines(True))+s[end:]
    s=swap(s,'            row["captured"]=case["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9","trusted_pytorch_v10")',
        '            row["captured"]=(case["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9","trusted_pytorch_v10")\n'
        '                or (case.get("source_api") is not None and case["source_api"]["kind"]!="graph"))')
    s=swap(s,'            require(request==prepare_task(model,options) and candidate_source(request)[0]==source,',
        '            if case.get("source_api") is not None:\n'
        '                from application_policy import prepare as source_prepare\n'
        '                fresh=source_prepare(model,"poseidon_cpu","numerical",options)[0]\n'
        '            else:fresh=prepare_task(model,options)\n'
        '            require(request==fresh and candidate_source(request)[0]==source,')
    s=swap(s,'''        task=app.submit_import(folder,options=options,limits=dict(generations=1,http_attempts=1,
            wall_seconds=int(seconds),api_timeout_seconds=1200,max_tokens=384000))''',
        '''        limits=dict(generations=1,http_attempts=1,wall_seconds=int(seconds),api_timeout_seconds=1200,max_tokens=384000)
        if case.get("source_api") is not None:
            task=app.submit_source(case["source_api"],**case["source_requirements"],limits=limits)
        else:task=app.submit_import(folder,options=options,limits=limits)''')
    s=swap(s,'''        require(sha(program/"candidate.py")==row["candidate_sha256"],"Unchanged rule source")''',
        '''        require(sha(program/"candidate.py")==row["candidate_sha256"],"Unchanged rule source")
        if case.get("source_api") is not None:
            from model_source_provenance import FILE,verify,reuse_identity
            current_origin=load(program/FILE)
            verify(current_origin,request,provenance,controller_sha256=app._checked_task_metadata(task)["model_source"]["implementation"]["controller_sha256"])
            require(reuse_identity(current_origin)==case["source_reuse_identity"],"Compiled package source identity")
            row["source_kind"]=case["source_api"]["kind"]
            row["model_source"]=current_origin''')
    s=swap(s,'''        cached=app.run(app.submit_import(Path(row["imported"]),options=item["options"]),NeverGenerate())''',
        '''        original_case=case_by_id[name]
        if original_case.get("source_api") is not None:
            again=app.submit_source(original_case["source_api"],**original_case["source_requirements"])
        else:again=app.submit_import(Path(row["imported"]),options=item["options"])
        cached=app.run(again,NeverGenerate())''')
    code["backend.py"]=s
    s=code["audit.py"]
    s=swap(s,'plan=load(out/"plan.json");report=load(out/"report.json")',
           'plan=load(out/"plan.json");report=load(out/"report.json")\ncase_by_id={c["id"]:c for c in plan["cases"]}\nproposal=load(root/"docs/baseline/agent-application-source-offline-proposal-r1021.json")\nrequire(all(sha(root/p)==h for p,h in proposal["extra_source_hashes"].items()),"Context adapter source freeze")')
    s=swap(s,'    name=row["id"]','    name=row["id"];case=case_by_id[name]')
    a=s.index('    if row["input_kind"] in (');b=s.index('    original_manual=',a)
    old_block=s[a:b]
    prefix='''    if case.get("source_api") is not None:
        from model_source_provenance import verify,reuse_identity
        verify(case["source_receipt"],load(case["request"]),provenance)
        require(ref["probes"]==16 and ref["reference_status"]=="passed","Independent source intake binding")
    else:
'''
    s=s[:a]+prefix+"".join("    "+line if line.strip() else line for line in old_block.splitlines(True))+s[b:]
    a=s.index('    for edit in ("missing","drift"):');b=s.index('    if not row.get("request_id"):',a)
    old_block=s[a:b]
    prefix='''    if case.get("source_api") is not None:
        for key,value in (("model_sha256","0"*64),("files",{})):
            bad=copy.deepcopy(case["source_receipt"]);bad[key]=value
            bad["binding"]=digest({k:v for k,v in bad.items() if k!="binding"})
            try:verify(bad,load(case["request"]),provenance)
            except ValueError:rejections+=1
            else:raise ValueError("Invalid source origin accepted")
    else:
'''
    s=s[:a]+prefix+"".join("    "+line if line.strip() else line for line in old_block.splitlines(True))+s[b:]
    s=swap(s,'''    require(sha(out/"prepared"/name/"candidate.py")==row["candidate_sha256"] and row["source_origin"]==("trusted_pytorch_rule_program" if row["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9","trusted_pytorch_v10") else "declared_json_rule_program"),"Fixed rule provenance")''',
        '''    expected_origin="model_source_rule_program" if case.get("source_api") is not None else ("trusted_pytorch_rule_program" if row["input_kind"] in ("trusted_pytorch_v7","trusted_pytorch_v8","trusted_pytorch_v9","trusted_pytorch_v10") else "declared_json_rule_program")
    require(sha(out/"prepared"/name/"candidate.py")==row["candidate_sha256"] and row["source_origin"]==expected_origin,"Fixed rule provenance")''')
    s=swap(s,'''        validation=load(out/"tasks"/row["task_id"]/"validation-0.json");evidence=Path(validation["evidence"])''',
        '''        validation_path=out/"tasks"/row["task_id"]/"validation-0.json"
        if not validation_path.exists():
            require(case.get("source_api") is not None and row["result"]["generations"]==0,
                    "Only pre-generation source failure has no backend evidence")
            rows.append(dict(id=name,status="failed",failure=row["result"]["failure"],source_origin=row["source_origin"],
                compiled=False,encrypted_execution=False,failure_layer="model_source_intake"))
            continue
        validation=load(validation_path);evidence=Path(validation["evidence"])''')
    s=swap(s,'''    require(sha(package/"candidate.py")==row["candidate_sha256"],"Fixed rule source")''',
        '''    require(sha(package/"candidate.py")==row["candidate_sha256"],"Fixed rule source")
    if case.get("source_api") is not None:
        from model_source_provenance import FILE,verify,reuse_identity
        origin=load(package/FILE)
        verify(origin,request,provenance,controller_sha256=proposal["source_controller_sha256"])
        require(reuse_identity(origin)==case["source_reuse_identity"],"Requalified source package origin")
        require(manifest["origin"]["model_source"]["file"]==FILE,"Source descriptor in manifest")''')
    code["audit.py"]=s
    # Approval is checked before importing the environment launcher or entering Nix.
    code["launch.py"]='''"""Current exact offline scope launcher. No approval means no environment entry."""
import hashlib,json,shlex,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
pre=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-model-source-integration-r1012")
proposal_path=ROOT/"docs/baseline/agent-application-source-offline-proposal-r1021.json"
receipt=pre/"approved-backend-scope-r1023.json"
if not receipt.is_file():raise RuntimeError("Current offline backend scope requires human approval")
proposal=json.loads(proposal_path.read_text());scope=json.loads(receipt.read_text())
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if hashlib.sha256(canonical({k:v for k,v in proposal.items() if k!="binding"})).hexdigest()!=proposal["binding"]:
    raise RuntimeError("Proposal binding changed")
if scope["approval_answer"]!="approved" or scope["proposal_binding"]!=proposal["binding"]:
    raise RuntimeError("Approval binding mismatch")
for section in ("source_hashes","extra_source_hashes","case_hashes","tool_hashes"):
    for name,h in proposal[section].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(path)!=h:raise RuntimeError("Frozen offline input changed")
for name,value in proposal["scope"].items():
    if scope.get(name)!=value:raise RuntimeError("Approved scope differs from fixed proposal")
if scope["deadline_wall"]<=time.time():raise RuntimeError("Approved offline deadline exhausted")
if (pre/"job-r1023.log").exists() or (pre.parent/"agent-application-source-backend-r1023").exists():
    raise RuntimeError("Preserve existing run; explicit outcome audit required")
import sys
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from hecate_python_env import enter_nix,VENV
started=time.monotonic()
prefix='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONPATH='+shlex.quote(str(ROOT/"scripts/baseline"))+' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.quote(str(VENV/"bin/python"))+' -B '
command=prefix+shlex.quote(str(Path(__file__).with_name("job.py")))+" > "+shlex.quote(str(pre/"job-r1023.log"))+" 2>&1"
code=enter_nix(command,seconds=max(1,min(7200,int(scope["deadline_wall"]-time.time()))),environment="poseidon-cpu")
(pre/"launch-terminal-r1023.json").write_text(json.dumps(dict(exit_code=code,seconds=time.monotonic()-started,paid_calls=0,single_nix_environment=True),indent=2)+"\\n")
raise SystemExit(code)
'''
    for name,s in code.items():compile(s,str(tools/name),"exec")
    tools.mkdir()
    for name,s in code.items():(tools/name).write_text(s)
    old_pre=WORK/"results/agent-pytorch-operator-shape-integration-r985"
    for label in ("backend","references","history","new-history","current-history"):
        name="protected-"+label+".json"
        require(not (pre/name).exists(),"Fresh protection record")
        (pre/name).write_bytes((old_pre/name).read_bytes())
        protected=json.loads((pre/name).read_text())
        require(all(sha(ROOT/p)==h for p,h in protected.items()),"Unchanged protected "+label)
    hashes=dict(frozen_old);hashes.update(new_files)
    for p in (audit_dir/"report.json",pre/"final-units-r1018.json",pre/"final-terminal-r1018.json"):
        hashes[str(p)]=sha(p)
    body=copy.deepcopy(old)
    body.update(revision="r1021",status="awaiting_human_approval",cases=rows,
        shards=[dict(id=1,count=48),dict(id=2,count=48),dict(id=3,count=22)],
        source_hashes=current_sources,extra_source_hashes=extra,case_hashes=hashes,
        backend_identity=backend,tool_hashes={str(tools/name):sha(tools/name) for name in code},
        previous_proposal=dict(binding=old["binding"],file=str(old_path),preserved=True,executed=False,source_stale=True),
        prior_approval="r870 authorized24 scope executed and closed; r991 never approved or executed",
        source_intake_acceptance_binding=intake["binding"],source_controller_sha256=intake["implementation"]["controller_sha256"],
        source_api_context_count=6,original_task_count=112,
        affected_plaintext_capture_count=0,new_source_capture_count=6,
        changed_existing_root_sources=changed,unchanged_capture_and_math=True,
        package_reuse_cases=old["package_reuse_cases"]+["source_graph_adaptive_pool_0_functional_query",
            "source_pytorch_instance_norm_2_functional_query","source_restricted_4"])
    body.pop("binding",None);body["binding"]=digest(body)
    target=ROOT/"docs/baseline/agent-application-source-offline-proposal-r1021.json"
    require(not target.exists(),"Fresh proposal")
    target.write_bytes(canonical(body))
    require(runtime_sources()==current_sources and all(sha(ROOT/p)==h for p,h in extra.items()),"Post preparation source freeze")
    require(all(sha(p)==h for p,h in hashes.items()),"Case and history preservation")
    result=dict(format="poseidon-application-source-backend-preparation-v1",planned=118,original_tasks=112,
        source_contexts=6,package_reuse_checks=18,sources=len(current_sources),extra_sources=len(extra),
        case_files=len(hashes),drivers=4,binding=body["binding"],status="prepared_not_approved",
        original_models_requests_and_candidates_unchanged=True,paid_calls=0,compiler_calls=0,encrypted_execution=False)
    (pre/"prepared-backend-r1020.json").write_bytes(canonical(result))
    print(json.dumps(result))
    return 0

if __name__=="__main__":raise SystemExit(run())
