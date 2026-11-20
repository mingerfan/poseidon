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
    def check_request(self,request,level):
        """Reject a request/SDK mismatch before any Provider generation."""
        from component_contract import reconstruct_request
        from compiler_configuration import request_configuration,verify_execution_configuration
        from poseidon_backend_identity import identity
        require(level in ("compiled","numerical"),"Explicit validation level required")
        reconstruct_request(request)
        config=request_configuration(request)
        actual=identity()
        require(config is not None and config["backend"]=="poseidon_cpu" and
                config["source_lock_sha256"]==actual["source_lock_sha256"] and
                config["operator_spec_sha256"]==actual["operator_spec_sha256"] and
                config["ckks_config_sha256"]==actual["compiler_profile_sha256"],
                "Request/SDK configuration binding mismatch")
        verify_execution_configuration(request,actual["compiler_profile_sha256"],config["waterline"])
        from native_execution_profile import selected,validate_request
        if selected(request):
            from poseidon_runtime_paths import checked_sdk
            from native_sdk_manifest import ready_for_agent_qualification
            sdk=checked_sdk()
            validate_request(request)
            require(type(sdk)is dict and 'native'in sdk and ready_for_agent_qualification(sdk)and
                    actual.get('execution_profile')=='original-epoch-public-native-v1'and
                    actual.get('native_runtime_sha256')==config['native_runtime_sha256']and
                    actual.get('native_client_sha256')==config['native_client_sha256']and
                    actual.get('native_capability_binding')==config['native_capability_binding']and
                    actual.get('native_key_epoch')==config['native_key_epoch'],'Measured actual Native backend configuration')
        else:
            require('native_sdk_execution_fingerprint'not in actual,'Explicit Native SDK requires its Native configuration')
    def qualify(self,request,candidate,level,*,artifact_observer=None):
        from component_backend import qualify
        options={} if artifact_observer is None else dict(artifact_observer=artifact_observer)
        return qualify(request,candidate,validation_level=level,**options)
    def export(self,result,destination,level,*,model_provenance=None,generation_provenance=None,model_source=None):
        if result.get('native_sdk')is True:
            from native_program_package import export_bundle
            return export_bundle(Path(result['evidence']),destination,model_provenance=model_provenance,
                validation_level=level,application_identities=self.identity_manifest(),
                generation_provenance=generation_provenance,model_source=model_source)
        from candidate_bundle import export_bundle
        return export_bundle(Path(result["evidence"]),destination,model_provenance=model_provenance,validation_level=level,
                             application_identities=self.identity_manifest(), generation_provenance=generation_provenance, model_source=model_source)
    def check_package(self,folder,level):
        from native_program_package import is_native,check_bundle as check_native
        if is_native(folder):
            m=check_native(folder)
            require(m['validation']['level']==level,'Native package acceptance level mismatch')
            return m['binding']
        from candidate_bundle import check_bundle
        m=check_bundle(folder)
        actual=m.get("validation",{}).get("level","numerical")
        require(actual==level,"Package acceptance level mismatch")
        return m["binding"]


    def execute(self,folder,samples,*,artifact_observer=None,sample_format="named"):
        from native_program_package import is_native,execute_program as native_execute
        if is_native(folder):
            return native_execute(folder,samples,artifact_observer=artifact_observer,sample_format=sample_format)
        from program_execution import execute_program
        options={} if artifact_observer is None else dict(artifact_observer=artifact_observer)
        if sample_format!="named":options["sample_format"]=sample_format
        return execute_program(folder,samples,**options)
