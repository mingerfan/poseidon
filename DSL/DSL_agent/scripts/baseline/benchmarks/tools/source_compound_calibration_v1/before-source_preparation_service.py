"""Application adapter for explicit original-source calibration preparation."""
import hashlib
from pathlib import Path
from benchmark_graph import digest,canonical
from model_output_contract import snapshot,is_hash
from source_approximation_preparation import (
    inspect_source,observe_source,freeze,PreparationError)
from source_calibration_contract import CalibrationError
from source_calibration import implementation

REQUEST="poseidon-source-preparation-request-v1"
RESULT="poseidon-source-preparation-result-v1"
MODULE_REQUEST="poseidon-source-preparation-request-v2"
MODULE_RESULT="poseidon-source-preparation-result-v2"
MAX_REQUEST_BYTES=5*1024**2
CODES={"source_preparation_input_rejected","source_preparation_trust_required",
       "source_preparation_binding_changed","source_preparation_environment_unavailable",
       "source_preparation_timed_out","source_preparation_cancelled"}
def identity():
    return digest(dict(adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preparer_sha256=hashlib.sha256(Path(__file__).with_name("source_approximation_preparation.py").read_bytes()).hexdigest(),
        observer=implementation()))

def checked(request):
    request=snapshot(request,limit=MAX_REQUEST_BYTES)
    base={"format","action","manifest","source_signature"}
    action=request.get("action")if type(request)is dict else None
    fields=(base|{"samples","input_dtypes","output_structure","resource_profile"}|
            ({"selection"}if action=="observe"else set()))if action in("inspect","observe")else(
            base|{"calibration_evidence","configuration"})
    if type(request)is not dict or set(request)!=fields or request["format"]not in(REQUEST,MODULE_REQUEST) or action not in("inspect","observe","freeze"):
        raise ValueError("source_preparation_input_rejected")
    manifest=request["manifest"]
    if type(manifest)is not dict or set(manifest)!={"path","sha256"}or type(manifest["path"])is not str or not is_hash(manifest["sha256"]):
        raise ValueError("source_preparation_input_rejected")
    path=Path(manifest["path"])
    if not path.is_absolute()or path!=path.resolve()or path.name!="manifest.json"or path.is_symlink():
        raise ValueError("source_preparation_input_rejected")
    return request

class SourcePreparation:
    """Same preparation operations for host applications and the local CLI.

    This adapter schedules no follow-on action and performs no automatic retries.
    Existing generation/inference tasks are not modified.
    """
    def run(self,request,output_directory,*,trust_local_code=False):
        result=dict(format=RESULT,status="validation_failed",action=None,
            api_calls=0,compiler_calls=0,encrypted_calls=0,
            application_accuracy_certified=False,ckks_numerically_validated=False,
            approximation_fitted=False,automatic_domain_expansion=False)
        try:
            data=checked(request);result["action"]=data["action"]
            result["format"]=MODULE_RESULT if data["format"]==MODULE_REQUEST else RESULT
            result["request_sha256"]=digest(data);result["implementation_identity"]=identity()
            if trust_local_code is not True:
                result.update(status="unsupported",failure=dict(code="source_preparation_trust_required",
                    layer="input",retry_policy="never_automatic"))
            else:
                base=dict(expected_manifest_sha256=data["manifest"]["sha256"],
                    source_signature=data["source_signature"],trust_local_code=True)
                if data["action"]=="freeze":
                    receipt=freeze(data["manifest"]["path"],output_directory,
                        calibration_evidence=data["calibration_evidence"],
                        configuration=data["configuration"],**base)
                else:
                    base.update(input_dtypes=data["input_dtypes"],output_structure=data["output_structure"],
                                resource_profile=data["resource_profile"])
                    if data["action"]=="inspect":
                        receipt=inspect_source(data["manifest"]["path"],data["samples"],output_directory,
                            source_locations=data["format"]==MODULE_REQUEST,**base)
                    else:
                        receipt=observe_source(data["manifest"]["path"],data["samples"],output_directory,
                            selection=data["selection"],expected_selection_sha256=digest(data["selection"]),**base)
                result.update(status=receipt["status"],receipt_binding=receipt["binding"],receipt=receipt,
                              evidence_directory=str(Path(output_directory)),failure=None)
                if data["action"]=="freeze":
                    result["configured_manifest"]=receipt["configured_manifest"]
        except (PreparationError,CalibrationError)as error:
            code=error.code
            status=("unsupported"if code in("approximation_calibration_location_unsupported",
                "calibration_fx_trace_rejected","calibration_selector_unavailable")else
                "timed_out"if code=="calibration_timeout"else
                "service_error"if code in("approximation_preparation_binding_changed",
                   "calibration_binding_changed","calibration_fx_graph_changed",
                   "calibration_environment_unavailable","calibration_worker_protocol_error")else "validation_failed")
            result.update(status=status,failure=dict(code=code,layer="source_preparation",retry_policy="never_automatic"))
        except (ValueError,TypeError,KeyError,OverflowError):
            result.update(status="validation_failed",failure=dict(code="source_preparation_input_rejected",
                layer="input",retry_policy="never_automatic"))
        except (OSError,RuntimeError):
            result.update(status="service_error",failure=dict(code="source_preparation_environment_unavailable",
                layer="environment",retry_policy="never_automatic"))
        result["binding"]=digest(result)
        return result

def public_summary(result):
    """CLI output excludes supplied arrays, source text and reference answers."""
    keys=("format","status","action","request_sha256","implementation_identity","receipt_binding",
          "evidence_directory","configured_manifest","api_calls","compiler_calls","encrypted_calls",
          "application_accuracy_certified","ckks_numerically_validated","approximation_fitted",
          "automatic_domain_expansion","failure")
    summary={k:result[k]for k in keys if k in result}
    receipt=result.get("receipt")
    if type(receipt)is dict:
        graph=receipt.get("fx_graph")
        if type(graph)is dict:
            summary["graph_sha256"]=graph["graph_sha256"]
            summary["node_inventory"]=[{k:n[k]for k in("name","op","target")}for n in graph["nodes"]]
            if graph.get("format")=="poseidon-source-calibration-fx-graph-v2":
                for brief,node in zip(summary["node_inventory"],graph["nodes"]):
                    brief["source"]=node["source"]
        if type(receipt.get("observations"))is dict:summary["observations"]=receipt["observations"]
    summary["binding"]=digest(summary)
    return summary
