"""Package reuse keys over immutable model/task metadata; no generation."""
from benchmark_graph import digest,require
from application_identity import check_manifest,DEFAULT_COMPATIBILITY
class ReuseKeyRejected(ValueError):
    """A matching cached task has an invalid original serialized cache key."""
def legacy_key(meta):
    key=dict(request_id=meta["request"]["request_id"],identity=meta["identity"],
             validation_level=meta["validation_level"])
    if meta.get("model_source")is not None:
        from model_source_provenance import reuse_identity
        key["model_source"]=reuse_identity(meta["model_source"])
    if meta.get("model_provenance")is not None:
        key["model_provenance_sha256"]=digest(meta["model_provenance"])
    return digest(key)
def package_key(meta):
    policy=meta.get("identity_policy","legacy-v1")
    require(policy in ("legacy-v1","staged-v1"),"Package reuse identity policy")
    if policy=="legacy-v1":return legacy_key(meta)
    stages=check_manifest(meta["identity"])
    key=dict(format="poseidon-package-reuse-v1",request_sha256=digest(meta["request"]),
        validation_level=meta["validation_level"],
        stages={name:stages[name]["sha256"]for name in DEFAULT_COMPATIBILITY})
    if meta.get("model_source")is not None:
        from model_source_provenance import reuse_identity
        key["model_source"]=reuse_identity(meta["model_source"])
    if meta.get("model_provenance")is not None:
        key["model_provenance_sha256"]=digest(meta["model_provenance"])
    return digest(key)
def matches(prior,prior_meta,requested):
    # Existing serialized cache fields keep their original representation.
    if package_key(prior_meta)!=package_key(requested):return False
    if prior.get("reuse_key")!=legacy_key(prior_meta):
        raise ReuseKeyRejected("cached_task_binding_rejected")
    return True
