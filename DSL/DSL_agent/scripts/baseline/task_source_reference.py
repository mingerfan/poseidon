"""Optional source-precision reference over a checked Application task.

The existing component still owns generation, delivery, budgets and package
validation. Source forward is explicit and isolated; the original task is kept.
"""
import copy
from pathlib import Path
from benchmark_graph import digest
from model_output_contract import is_hash,snapshot,OutputError
from task_output_adapter import TaskOutputAdapter,source_binding
from source_precision_reference import run,SourceReferenceError,source_identity
FORMAT="poseidon-task-source-reference-v1"
class TaskSourceReference:
    def __init__(self,application):
        self.outputs=TaskOutputAdapter(application)
        self.application=application
    def reference(self,task,samples,output_directory,*,expected_package_binding,
                  input_dtypes,trust_local_code=False):
        """Run the bound trusted source on 1..16 inputs in its declared native dtypes."""
        self.outputs._task(task)
        outcome=self.application.result(task)
        frame=dict(format=FORMAT,status=outcome["status"],task_id=task,task_status=outcome["status"],
            encrypted_execution=False,api_calls=0,compiler_calls=0,
            source_reference_evaluated=False,application_accuracy_certified=False)
        if outcome["status"]!="succeeded":
            frame["failure"]=copy.deepcopy(outcome.get("failure"));return frame
        if trust_local_code is not True:
            frame.update(status="unsupported",failure=dict(code="source_reference_trust_required",
                layer="source_reference",retry_policy="never_automatic"))
            return frame
        try:
            metadata=self.outputs._metadata(task)
            package,grade=self.outputs._package(outcome,metadata)
            if not is_hash(expected_package_binding) or package["binding"]!=expected_package_binding:
                raise SourceReferenceError("source_reference_binding_changed")
            origin=metadata.get("model_source")
            if type(origin)is not dict or origin.get("kind")!="trusted_pytorch":
                frame.update(status="unsupported",failure=dict(code="source_reference_missing_source",
                    layer="source_reference",retry_policy="never_automatic"))
                return frame
            path=self.application._dir(task)
            submission=self.application._source(path)
            if submission["binding"]!=origin["snapshot_binding"]:
                raise SourceReferenceError("source_reference_binding_changed")
            provenance=metadata.get("model_provenance")
            if type(provenance)is not dict:
                raise SourceReferenceError("source_reference_binding_changed")
            from candidate_bundle import check_provenance
            check_provenance(provenance,metadata["request"]["model"])
            captured=provenance["decomposition"]["pytorch_capture"]
            if captured.get("trusted_code")is not True:
                raise SourceReferenceError("source_reference_binding_changed")
            from source_precision_reference import prepare_source
            manifest=path/"source/manifest.json"
            spec,_=prepare_source(manifest)
            # Detach host values before launching the isolated forward.
            from program_execution import normalize_samples
            normalized=normalize_samples(metadata["request"],samples)
            wire=[{n:a.tolist() for n,a in sample.items()} for sample in normalized]
            frozen=source_identity()
            report=run(manifest,metadata["request"]["model"],wire,output_directory,
                expected_manifest_sha256=digest(spec),input_dtypes=input_dtypes,
                output_structure=captured["output_structure"],resource_profile=submission["resource_profile"],
                trusted=True)
            if (self.application.result(task)!=outcome or
                self.outputs._metadata(task)!=metadata or self.application._source(path)!=submission or
                source_identity()!=frozen):
                raise SourceReferenceError("source_reference_binding_changed")
            frame.update(status="evaluated",source_reference_evaluated=True,
                request_id=metadata["request"]["request_id"],bundle_binding=package["binding"],
                task_metadata_binding=metadata["binding"],source_binding=source_binding(metadata),
                source_snapshot_binding=submission["binding"],package_validation=grade,
                supplied_input_sha256=digest(wire),source_reference=report,
                reference_policy="bound-source-native-dtype-finite-v1",
                evidence=str(Path(output_directory)/"native-reference.json"),failure=None)
            frame["binding"]=digest(frame)
            return frame
        except SourceReferenceError as error:
            frame.update(status="service_error" if error.code in (
                "source_reference_binding_changed","source_reference_environment_unavailable",
                "source_reference_worker_protocol_error") else
                "timed_out" if error.code=="source_reference_timeout" else "validation_failed",
                failure=dict(code=error.code,layer="source_reference",retry_policy="never_automatic"))
        except OutputError as error:
            frame.update(status="validation_failed",failure=error.feedback())
        except (ValueError,TypeError,KeyError,OSError,RuntimeError):
            frame.update(status="service_error",failure=dict(code="source_reference_context_rejected",
                layer="source_reference",retry_policy="never_automatic"))
        return frame
