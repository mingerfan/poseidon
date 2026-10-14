"""Integrate explicit PyTorch v10 after checking the idle source boundary."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
def replace_once(text,old,new):
    require(text.count(old)==1,"Expected one integration seam: "+old[:90])
    return text.replace(old,new,1)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run():
    from hecate_python_env import WORK
    from benchmarks.tools.audit_pytorch_operator_shape_contexts import checked,frozen
    root=ROOT;directory=WORK/"results/agent-pytorch-operator-shape-integration-r985"
    require(not directory.exists(),"Fresh integration recovery directory")
    require(subprocess.check_output(["git","branch","--show-current"],cwd=root,text=True).strip()=="lhy-agent-dsl","Preserved development branch")
    ps=subprocess.check_output(["ps","-eo","pid,ppid,comm,args"],text=True)
    watched=("run_candidate.py","run_agent_batch.py","poseidon_agent_cpu","frozen_arithmetic_shape_backend/launch.py",
             "frozen_arithmetic_shape_backend/job.py","audit_operator_shape_capture_prototype.py","audit_operator_shape_probe_guards.py")
    live=[line for line in ps.splitlines() if any(s in line for s in watched)
          and not line.lstrip().startswith(str(os.getpid())+" ")]
    require(not live,"No shared execution worker at integration boundary")
    proposal=checked(json.loads((root/"docs/baseline/agent-arithmetic-shape-offline-proposal-r976.json").read_text()))
    original_freeze=frozen(proposal)
    require(not (WORK/"results/agent-pytorch-shape-integration-r969/approved-backend-scope-r977.json").exists()
        and not (WORK/"results/agent-arithmetic-shape-backend-r977").exists(),"Old proposal remains unstarted")
    prototype=root/"scripts/baseline/benchmarks/tools/fixtures/operator-shape-capture-prototype-v1"
    staged=checked(json.loads((prototype/"prototype.json").read_text()))
    require(all(sha(root/"scripts/baseline"/name)==h for name,h in staged["original_hashes"].items()),"Validated prototype matches source base")
    original_report=checked(json.loads((WORK/"results/agent-operator-shape-prototype-r982/report.json").read_text()))
    continued=checked(json.loads((WORK/"results/agent-operator-shape-refusal-r983/report.json").read_text()))
    require(original_report["original_failures_passed"]==18 and original_report["controls_passed"]==6
        and continued["total_guard_evidence"]==7 and continued["prototype_binding"]==staged["binding"],"Prerequisite staged evidence")
    names=["pytorch_static_trace.py","pytorch_model_capture.py","pytorch_model_intake.py",
           "application_capabilities.py","application-source-catalog-v1.json"]
    before={n:(root/"scripts/baseline"/n).read_text() for n in names};after={}
    trace=before["pytorch_static_trace.py"]
    trace=replace_once(trace,'ARITHMETIC_POLICY = "signature-static-arithmetic-shape-v1"',
        'ARITHMETIC_POLICY = "signature-static-arithmetic-shape-v1"\nOPERATOR_POLICY = "signature-static-operator-shape-v1"')
    trace=trace.replace("(POLICY,INDEX_POLICY,ARITHMETIC_POLICY)","(POLICY,INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY)")
    trace=trace.replace("(INDEX_POLICY,ARITHMETIC_POLICY)","(INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY)")
    trace=replace_once(trace,'if policy==ARITHMETIC_POLICY:\n                functions|={torch.subtract,torch.multiply}\n                methods|={"add","sub","subtract","mul","multiply"}',
        'if policy in (ARITHMETIC_POLICY,OPERATOR_POLICY):\n                functions|={torch.subtract,torch.multiply}\n                methods|={"add","sub","subtract","mul","multiply"}\n            if policy==OPERATOR_POLICY:\n                functions|={F.instance_norm,F.adaptive_avg_pool1d,F.adaptive_avg_pool2d}')
    staged_trace=(prototype/"pytorch_static_trace.py").read_text()
    begin=staged_trace.index('                if policy==ARITHMETIC_POLICY:\n                    if type(child)')
    end=staged_trace.index('                if type(child) is torch.nn.Flatten:',begin)
    added=staged_trace[begin:end].replace("if policy==ARITHMETIC_POLICY:","if policy==OPERATOR_POLICY:")
    trace=replace_once(trace,'                if type(child) is torch.nn.Flatten:',added+'                if type(child) is torch.nn.Flatten:')
    after["pytorch_static_trace.py"]=trace
    capture=before["pytorch_model_capture.py"]
    capture=replace_once(capture,'\ndef capture(', '\nREGISTERED_FUNCTION_POLICY="registered-v2"\nADAPTIVE_FUNCTION_POLICY="static-adaptive-average-v1"\n\ndef capture(')
    capture=replace_once(capture,'matrix_policy="rank-two-three-v1"):', 'matrix_policy="rank-two-three-v1", functional_policy=REGISTERED_FUNCTION_POLICY):\n    require(functional_policy in (REGISTERED_FUNCTION_POLICY,ADAPTIVE_FUNCTION_POLICY),"Functional capture policy")')
    staged_capture=(prototype/"pytorch_model_capture.py").read_text()
    begin=staged_capture.index('            elif functional_policy=="adaptive-pool-static-prototype-v1"')
    end=staged_capture.index('            elif target is torch.nn.functional.batch_norm:',begin)
    added=staged_capture[begin:end].replace('functional_policy=="adaptive-pool-static-prototype-v1"','functional_policy==ADAPTIVE_FUNCTION_POLICY')
    capture=replace_once(capture,'            elif target is torch.nn.functional.batch_norm:',added+'            elif target is torch.nn.functional.batch_norm:')
    after["pytorch_model_capture.py"]=capture
    intake=before["pytorch_model_intake.py"]
    intake=replace_once(intake,'Both use the same isolated CPU capture boundary.',
        'V10 additionally registers operator result metadata and functional adaptive\naverage pooling. All retain the same isolated CPU capture boundary.')
    intake=replace_once(intake,'SOURCE_FORMATS=(FORMAT,SHAPE_FORMAT)\nPACKAGES={FORMAT:PACKAGE_FORMAT,SHAPE_FORMAT:SHAPE_PACKAGE_FORMAT}\nREFERENCES={FORMAT:REFERENCE_FORMAT,SHAPE_FORMAT:SHAPE_REFERENCE_FORMAT}',
        'OPERATOR_SHAPE_FORMAT="poseidon-trusted-pytorch-v10"\nOPERATOR_SHAPE_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v10"\nOPERATOR_SHAPE_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v10"\nSOURCE_FORMATS=(FORMAT,SHAPE_FORMAT,OPERATOR_SHAPE_FORMAT)\nSOURCE_SHAPE_FORMATS=(SHAPE_FORMAT,OPERATOR_SHAPE_FORMAT)\nPACKAGES={FORMAT:PACKAGE_FORMAT,SHAPE_FORMAT:SHAPE_PACKAGE_FORMAT,OPERATOR_SHAPE_FORMAT:OPERATOR_SHAPE_PACKAGE_FORMAT}\nREFERENCES={FORMAT:REFERENCE_FORMAT,SHAPE_FORMAT:SHAPE_REFERENCE_FORMAT,OPERATOR_SHAPE_FORMAT:OPERATOR_SHAPE_REFERENCE_FORMAT}')
    intake=replace_once(intake,'\ndef source_identity():','\ndef source_shape_policy(spec):\n    from pytorch_static_trace import INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY\n    require(type(spec) is dict and spec.get("format") in SOURCE_FORMATS,"Trusted source shape version")\n    return {FORMAT:INDEX_POLICY,SHAPE_FORMAT:ARITHMETIC_POLICY,OPERATOR_SHAPE_FORMAT:OPERATOR_POLICY}[spec["format"]]\n\ndef source_identity():')
    require(intake.count('if spec["format"]==SHAPE_FORMAT:')==4,"All source shape metadata seams")
    intake=intake.replace('if spec["format"]==SHAPE_FORMAT:','if spec["format"] in SOURCE_SHAPE_FORMATS:')
    intake=replace_once(intake,'from pytorch_static_trace import policy_record,ARITHMETIC_POLICY','from pytorch_static_trace import policy_record')
    intake=replace_once(intake,'policy_record(spec["inputs"],resource_profile,policy=ARITHMETIC_POLICY)',
        'policy_record(spec["inputs"],resource_profile,policy=source_shape_policy(spec))')
    intake=replace_once(intake,'source_policy=ARITHMETIC_POLICY if spec["format"]==SHAPE_FORMAT else INDEX_POLICY','source_policy=source_shape_policy(spec)')
    intake=replace_once(intake,'from pytorch_model_capture import capture\n','from pytorch_model_capture import capture,REGISTERED_FUNCTION_POLICY,ADAPTIVE_FUNCTION_POLICY\n')
    intake=replace_once(intake,'matrix_policy="static-tensor-matmul-v1")',
        'matrix_policy="static-tensor-matmul-v1",\n        functional_policy=ADAPTIVE_FUNCTION_POLICY if spec["format"]==OPERATOR_SHAPE_FORMAT else REGISTERED_FUNCTION_POLICY)')
    after["pytorch_model_intake.py"]=intake
    caps=before["application_capabilities.py"]
    caps=replace_once(caps,'from pytorch_model_intake import SOURCE_FORMATS,SHAPE_FORMAT',
        'from pytorch_model_intake import SOURCE_FORMATS,SHAPE_FORMAT,OPERATOR_SHAPE_FORMAT')
    caps=replace_once(caps,'            functional_input_rules=functional_input_rules()),',
        '            operator_result_shape=dict(version=OPERATOR_SHAPE_FORMAT,\n                policy="signature-static-operator-shape-v1",\n                calls=["nn.InstanceNorm1d","nn.InstanceNorm2d","F.instance_norm",\n                       "nn.AdaptiveAvgPool1d","nn.AdaptiveAvgPool2d","F.adaptive_avg_pool1d","F.adaptive_avg_pool2d",\n                       "nn.ConstantPad1d","nn.ConstantPad2d","nn.ZeroPad2d",\n                       "nn.ReflectionPad1d","nn.ReflectionPad2d","nn.ReplicationPad1d","nn.ReplicationPad2d"],\n                queries=["shape","size","ndim","dim","numel","len"],\n                source="declared signature and registered metadata rules",\n                input_status="registered",compiler_status="unverified",encrypted_status="unverified",agent_status="unverified"),\n            functional_adaptive_average=dict(version=OPERATOR_SHAPE_FORMAT,policy="static-adaptive-average-v1",\n                calls=["F.adaptive_avg_pool1d","F.adaptive_avg_pool2d"],ranks=[3,4],\n                output_size="positive static spatial dimensions, each no larger than its input",\n                input_status="registered",compiler_status="unverified",encrypted_status="unverified",agent_status="unverified"),\n            functional_input_rules=functional_input_rules()),')
    after["application_capabilities.py"]=caps
    catalog=json.loads(before["application-source-catalog-v1.json"])
    matched=[d for d in catalog["dynamic_loads"] if d["module"]=="pytorch_model_intake.py"]
    require(len(matched)==1 and "V8/V9 source" in matched[0]["authority"],"Current dynamic source authority")
    matched[0]["authority"]=matched[0]["authority"].replace("V8/V9 source","V8/V9/V10 source")
    after["application-source-catalog-v1.json"]=json.dumps(catalog,indent=2)+"\n"
    for name,text in after.items():
        require(text!=before[name],"Actual integration change")
        if name.endswith(".py"):compile(text,str(root/"scripts/baseline"/name),"exec")
    directory.mkdir(mode=0o700);(directory/"source-base").mkdir()
    for name,text in before.items():(directory/"source-base"/name).write_text(text)
    status=subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=all"],cwd=root)
    (directory/"git-status-before.txt").write_bytes(status)
    record=dict(format="poseidon-pytorch-v10-integration-boundary-v1",proposal_binding=proposal["binding"],
        old_frozen_inventories=original_freeze,prototype_binding=staged["binding"],original_sources={n:digest_text(t) for n,t in before.items()},
        intended_sources={n:digest_text(t) for n,t in after.items()},changed_files=names,
        backend_changes=0,reference_changes=0,parameter_changes=0,paid_calls=0,
        archived_unexecuted_proposal=True,shared_workers_at_boundary=live)
    record["binding"]=digest(record);(directory/"boundary.json").write_bytes(canonical(record))
    for name,text in after.items():
        path=root/"scripts/baseline"/name
        require(sha(path)==record["original_sources"][name],"Source changed after backup")
        path.write_text(text);require(sha(path)==record["intended_sources"][name],"Integrated file bytes")
    (directory/"integrated.json").write_bytes(canonical(dict(binding=record["binding"],files=record["intended_sources"])))
    print(json.dumps(dict(recovery_directory=str(directory),binding=record["binding"],changed=names,old_proposal="retained, now stale"),ensure_ascii=False))
def digest_text(text):return hashlib.sha256(text.encode()).hexdigest()
if __name__=="__main__":run()
