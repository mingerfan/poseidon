"""Context-bound adapter over the existing Application and Poseidon backend."""
from pathlib import Path
from application_backend import PoseidonBackend
from application_execution.context import (WorkerContext,check_request_record,
    request_receipt,require)

class ContextBackend(PoseidonBackend):
    def __init__(self,context):
        require(isinstance(context,WorkerContext),"context_backend_type")
        self.context=context.check_current()

    def identity(self):
        self.context.check_current()
        return super().identity()

    def identity_manifest(self):
        self.context.check_current()
        return super().identity_manifest()

    def qualify(self,request,candidate,level,*,artifact_observer=None):
        context=self.context.request(request,level)
        options={} if artifact_observer is None else dict(artifact_observer=artifact_observer)
        result=super().qualify(request,candidate,level,**options)
        context.check(self.context,request,level)
        receipt=context.record();request_receipt(self.context,receipt)
        result["run_context"]=receipt
        return result

    def export(self,result,destination,level,*,model_provenance=None,generation_provenance=None,model_source=None):
        from component_contract import reconstruct_request
        from public_weight_context import read_request
        from model_weight_package import regular_bytes
        self.context.check_current()
        destination=Path(destination)
        require(destination.parent.parent==self.context.task_store and destination.parent==destination.parent.resolve(),
                "context_delivery_scope")
        raw=regular_bytes(Path(result["evidence"])/"request.json",4*1024**2).decode()
        request=reconstruct_request(read_request(raw))
        context=check_request_record(result.get("run_context"),self.context,request,level)
        source_options = {} if model_source is None else dict(model_source=model_source)
        super().export(result,destination,level,model_provenance=model_provenance,
                       generation_provenance=generation_provenance,**source_options)
        context.check(self.context,request,level)

    def check_package(self,folder,level):
        self.context.check_current()
        binding=super().check_package(folder,level)
        self.context.check_current()
        return binding

    def execute(self,folder,samples,*,artifact_observer=None):
        from candidate_bundle import check_execution_dependencies
        manifest,request=check_execution_dependencies(Path(folder))
        context=self.context.request(request,manifest["validation"]["level"])
        options={} if artifact_observer is None else dict(artifact_observer=artifact_observer)
        result=super().execute(folder,samples,**options)
        context.check(self.context,request,manifest["validation"]["level"])
        result["run_context"]=context.record()
        request_receipt(self.context,result["run_context"])
        return result

def application(context):
    """Create the existing component with an explicit pinned worker context."""
    from application_component import Application
    require(isinstance(context,WorkerContext),"context_application_type")
    context.persist()
    return Application(context.task_store,ContextBackend(context))
