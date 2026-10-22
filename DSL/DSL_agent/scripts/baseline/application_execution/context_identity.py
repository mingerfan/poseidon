"""Worker SDK identity excludes generation; legacy receipts remain exact."""
import copy,hashlib,re
from pathlib import Path
from application_identity import source_closure,POLICY,DEFAULT_COMPATIBILITY,check_manifest
from model_weight_package import regular_bytes
from application_execution.context import canonical,ContextError
FORMAT="poseidon-worker-context-v2"
POLICY_NAME="staged-sdk-v1"
def authority(root):
    baseline=Path(root)/"scripts/baseline"
    roots=("application_identity.py","application_backend.py","hecate_python_env.py",
           "workspace_paths.py","platform_config.py","poseidon_runtime_paths.py",
           "model_weight_package.py")
    closure=source_closure(baseline,roots)
    own=("application_execution/context.py","application_execution/backend.py",
         "application_execution/context_identity.py")
    sources=dict(closure["sources"])
    for name in own:
        sources[name]=hashlib.sha256(regular_bytes(baseline/name,1024**2)).hexdigest()
    if any(name in sources for name in ("deepseek_provider.py","agent_credentials.py",
                                       "application_component.py")):
        raise ContextError("context_sdk_generation_dependency")
    payload=dict(policy=POLICY_NAME,source_policy=POLICY,sources=sources)
    return dict(payload,sha256=hashlib.sha256(canonical(payload)).hexdigest())
def selected(stages):
    check_manifest(stages)
    return {name:stages[name]["sha256"]for name in DEFAULT_COMPATIBILITY}
def fields(root,backend,stages):
    return dict(format=FORMAT,identity_policy=POLICY_NAME,
        backend_identity=dict(backend="poseidon_cpu",tools=copy.deepcopy(stages["backend"])),
        stage_identity_sha256=selected(stages),source_authority=authority(root))
def validate(value):
    expected={"format","identity_policy","backend","platform","paths","library_environment",
              "python_runtime","backend_identity","stage_identity_sha256","source_authority"}
    if set(value)!=expected or value.get("identity_policy")!=POLICY_NAME:
        raise ContextError("context_staged_record_fields")
    hashes=value["stage_identity_sha256"]
    if type(hashes)is not dict or set(hashes)!=set(DEFAULT_COMPATIBILITY)or any(
        type(h)is not str or re.fullmatch("[0-9a-f]{64}",h)is None for h in hashes.values()):
        raise ContextError("context_staged_stage_hashes")
    a=value["source_authority"]
    if type(a)is not dict or set(a)!={"policy","source_policy","sources","sha256"}or a["policy"]!=POLICY_NAME:
        raise ContextError("context_staged_source_authority")
    if a["sha256"]!=hashlib.sha256(canonical({k:v for k,v in a.items()if k!="sha256"})).hexdigest():
        raise ContextError("context_staged_source_authority")
