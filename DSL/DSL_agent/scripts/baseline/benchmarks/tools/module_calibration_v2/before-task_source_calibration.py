"""Explicit passive calibration over a checked existing Application source task."""
import copy
from pathlib import Path
from benchmark_graph import digest
from model_output_contract import is_hash,snapshot,OutputError
from task_output_adapter import TaskOutputAdapter,source_binding
from source_calibration_contract import CalibrationError,checked,FX_FORMAT
from source_calibration import run,implementation
from source_precision_reference import prepare_source
from program_execution import normalize_samples

FORMAT="poseidon-task-source-calibration-v1"
FX_FRAME="poseidon-task-source-fx-calibration-v1"
class TaskSourceCalibration:
    def __init__(self,application):
        self.application=application;self.outputs=TaskOutputAdapter(application)
    def observe(self,task,samples,output_directory,*,expected_package_binding,selection,
                expected_selection_sha256,input_dtypes,trust_local_code=False):
        self.outputs._task(task);outcome=self.application.result(task)
        frame=dict(format=FORMAT,status=outcome["status"],task_id=task,task_status=outcome["status"],
            encrypted_execution=False,agent_calls=0,compiler_calls=0,
            approximation_modified=False,reference_modified=False,application_accuracy_certified=False,
            calibration_observed=False)
        if outcome["status"]!="succeeded":
            frame["failure"]=copy.deepcopy(outcome.get("failure"));return frame
        if trust_local_code is not True:
            frame.update(status="unsupported",failure=dict(code="calibration_trust_required",
                layer="calibration",retry_policy="never_automatic"));return frame
        try:
            selection=checked(selection)
            if not is_hash(expected_selection_sha256)or digest(selection)!=expected_selection_sha256:
                raise CalibrationError("calibration_binding_changed")
            metadata=self.outputs._metadata(task);package,grade=self.outputs._package(outcome,metadata)
            if not is_hash(expected_package_binding)or package["binding"]!=expected_package_binding:
                raise CalibrationError("calibration_binding_changed")
            origin=metadata.get("model_source")
            if type(origin)is not dict or origin.get("kind")!="trusted_pytorch":
                frame.update(status="unsupported",failure=dict(code="calibration_source_unavailable",
                    layer="calibration",retry_policy="never_automatic"));return frame
            folder=self.application._dir(task);source=self.application._source(folder)
            if source["binding"]!=origin["snapshot_binding"]:
                raise CalibrationError("calibration_binding_changed")
            provenance=metadata.get("model_provenance")
            if type(provenance)is not dict:raise CalibrationError("calibration_binding_changed")
            from candidate_bundle import check_provenance
            check_provenance(provenance,metadata["request"]["model"])
            capture=provenance["decomposition"]["pytorch_capture"]
            if capture.get("trusted_code")is not True:raise CalibrationError("calibration_binding_changed")
            spec,_=prepare_source(folder/"source/manifest.json")
            normalized=normalize_samples(metadata["request"],samples)
            wire=[{n:value.tolist()for n,value in sample.items()}for sample in normalized]
            frozen=implementation()
            report=run(folder/"source/manifest.json",metadata["request"]["model"],wire,output_directory,
                expected_manifest_sha256=digest(spec),selection=selection,
                expected_selection_sha256=expected_selection_sha256,input_dtypes=input_dtypes,
                output_structure=capture["output_structure"],resource_profile=source["resource_profile"],trusted=True)
            if self.application.result(task)!=outcome or self.outputs._metadata(task)!=metadata or\
                self.application._source(folder)!=source or implementation()!=frozen:
                raise CalibrationError("calibration_binding_changed")
            described=selection["format"]==FX_FORMAT and selection["mode"]=="describe"
            frame.update(format=FX_FRAME if selection["format"]==FX_FORMAT else FORMAT,
                status="described"if described else "observed",calibration_observed=not described,request_id=metadata["request"]["request_id"],
                bundle_binding=package["binding"],package_validation=grade,
                source_binding=source_binding(metadata),source_snapshot_binding=source["binding"],
                supplied_input_sha256=digest(wire),selection_sha256=expected_selection_sha256,
                calibration_report_binding=report["binding"],calibration_report=report,
                evidence=str(Path(output_directory)/"calibration.json"),failure=None)
            frame["binding"]=digest(frame)
            return frame
        except CalibrationError as error:
            frame.update(status="unsupported"if error.code=="calibration_fx_trace_rejected"else
                "timed_out"if error.code=="calibration_timeout"else
                "service_error"if error.code in ("calibration_binding_changed","calibration_fx_graph_changed","calibration_environment_unavailable",
                 "calibration_worker_protocol_error")else"validation_failed",
                failure=dict(code=error.code,layer="calibration",retry_policy="never_automatic"))
        except (OutputError,ValueError,TypeError,KeyError,OSError,RuntimeError):
            frame.update(status="service_error",failure=dict(code="calibration_binding_changed",
                layer="calibration",retry_policy="never_automatic"))
        return frame


    def describe_fx(self,task,samples,output_directory,*,expected_package_binding,
                    input_dtypes,trust_local_code=False):
        selection=dict(format=FX_FORMAT,purpose="calibration",mode="describe",
                       graph_sha256=None,selectors=[])
        return self.observe(task,samples,output_directory,
            expected_package_binding=expected_package_binding,selection=selection,
            expected_selection_sha256=digest(selection),input_dtypes=input_dtypes,
            trust_local_code=trust_local_code)
