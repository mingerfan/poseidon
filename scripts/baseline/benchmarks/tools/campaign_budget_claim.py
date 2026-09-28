"""Explicit paid-configuration task identity; legacy campaign claims remain unchanged."""
from pathlib import Path
from benchmark_graph import digest
from benchmark_runner import strict_file
from campaign_live_state import sealed,check,exclusive_json

def claim(registry,spec,sources,owner,plan_binding,paid_configuration):
    if registry.is_symlink():raise ValueError("Unsafe claim registry")
    token=digest(dict(task_id=spec["id"],model_sha256=spec["model_sha256"],
        request_id=spec["request_id"],runtime_source_sha256=digest(sources),
        paid_configuration=paid_configuration))
    if spec.get("evaluation_identity")!=token:raise ValueError("Evaluation identity changed")
    registry.mkdir(exist_ok=True)
    path=registry/(token+".json")
    value=sealed(dict(format="poseidon-paid-task-claim-v2",evaluation_identity=token,
        task_id=spec["id"],model_sha256=spec["model_sha256"],request_id=spec["request_id"],
        paid_configuration=paid_configuration,owner=str(owner.resolve()),plan_binding=plan_binding,
        state="claimed_before_launch",automatic_retry_allowed=False))
    try:exclusive_json(path,value)
    except FileExistsError:
        old=check(strict_file(path,1024**2))
        if old["evaluation_identity"]!=token or old["task_id"]!=spec["id"]:raise ValueError("Claim identity mismatch")
        return False,old
    return True,value
