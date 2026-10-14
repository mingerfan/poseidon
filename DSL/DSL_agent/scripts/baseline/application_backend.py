"""Trusted pinned-worker bridge. No benchmark index, batch or historical report."""
from pathlib import Path
import hashlib
from benchmark_graph import digest,require
from application_policy import BACKEND

class PoseidonBackend:
    name=BACKEND
    artifact_monitor_version="poseidon-artifact-monitor-v1"
    def operational_artifact_roots(self):
        from hecate_python_env import WORK
        return (WORK/"results",)
    def identity(self):
        from poseidon_backend_identity import identity
        from hecate_python_env import ROOT
        root=ROOT/"scripts/baseline"
        paths=list(root.glob("*.py"))+list(root.glob("*.json"))+list(root.glob("patches/*.patch"))
        paths+=list(root.glob("upstream_adapters/*.py"))
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(backend=self.name,tools=identity(),
                    implementation_digest=digest({str(p.relative_to(root)):sha(p) for p in paths}))
    def identity_manifest(self):
        from application_identity import current_identity_manifest
        return current_identity_manifest()
    def qualify(self,request,candidate,level,*,artifact_observer=None):
        from component_backend import qualify
        options={} if artifact_observer is None else dict(artifact_observer=artifact_observer)
        return qualify(request,candidate,validation_level=level,**options)
    def export(self,result,destination,level,*,model_provenance=None,generation_provenance=None,model_source=None):
        from candidate_bundle import export_bundle
        return export_bundle(Path(result["evidence"]),destination,model_provenance=model_provenance,validation_level=level,
                             application_identities=self.identity_manifest(), generation_provenance=generation_provenance, model_source=model_source)
    def check_package(self,folder,level):
        from candidate_bundle import check_bundle
        m=check_bundle(folder)
        actual=m.get("validation",{}).get("level","numerical")
        require(actual==level,"Package acceptance level mismatch")
        return m["binding"]


    def execute(self,folder,samples,*,artifact_observer=None,sample_format="named"):
        from program_execution import execute_program
        options={} if artifact_observer is None else dict(artifact_observer=artifact_observer)
        if sample_format!="named":options["sample_format"]=sample_format
        return execute_program(folder,samples,**options)
