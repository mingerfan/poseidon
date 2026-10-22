"""Optional structured-output delivery over an existing Application instance.

The host retains the receipt binding. Existing generation, package validation,
execution, budgets and task state remain owned by Application and its backend.
"""
import hashlib
import re
from pathlib import Path

from application_component import Application
from benchmark_graph import digest
from model_output_contract import (
    OutputError, build, check, bind_delivery, verify_delivery,
    structure_execution, snapshot, is_hash,
)

FORMAT = "poseidon-task-output-adapter-v1"
RECEIPT = "poseidon-task-output-receipt-v1"
SOURCE = "poseidon-task-output-source-v1"
TASK_ID = re.compile("[0-9a-f]{32}")


def implementation():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_binding(metadata):
    """Bind checked logical input and provenance without exposing their contents."""
    return digest({
        "format": SOURCE,
        "request_id": metadata["request"]["request_id"],
        "model_sha256": digest(metadata["request"]["model"]),
        "model_source": metadata.get("model_source"),
        "model_provenance": metadata.get("model_provenance"),
    })


class TaskOutputAdapter:
    """Host-owned adapter; attach to the same application used for submit/run."""

    def __init__(self, application):
        check(isinstance(application, Application), "output_source_rejected")
        self.application = application

    def _task(self, task):
        check(type(task) is str and TASK_ID.fullmatch(task) is not None,
              "output_binding_rejected")
        return task

    def _frame(self, outcome):
        value = snapshot(outcome, limit=4 * 1024**2)
        return {"format": FORMAT, "status": value["status"], "task": value}

    def _error(self, outcome, error):
        frame = self._frame(outcome)
        frame.update(status="service_error", failure=error.feedback())
        return frame

    def _metadata(self, task):
        try:
            metadata = self.application._checked_task_metadata(task)
            request = metadata["request"]
            check(type(request) is dict and is_hash(request.get("request_id"))
                  and type(request.get("model")) is dict
                  and is_hash(metadata.get("binding")), "output_binding_rejected")
            return metadata
        except OutputError:
            raise
        except (ValueError, TypeError, KeyError, OSError):
            raise OutputError("output_binding_rejected") from None

    def _package(self, outcome, metadata):
        package = outcome.get("package")
        level = metadata.get("validation_level")
        acceptance = outcome.get("acceptance")
        check(type(package) is dict
              and type(package.get("task_id")) is str
              and TASK_ID.fullmatch(package["task_id"]) is not None
              and is_hash(package.get("binding")),
              "output_execution_identity_rejected")
        check(level in ("compiled", "numerical")
              and type(acceptance) is dict
              and acceptance.get("level") == level
              and acceptance.get("compiled") is True
              and acceptance.get("artifact_checked") is True
              and type(acceptance.get("encrypted_execution")) is bool
              and acceptance["encrypted_execution"] == (level == "numerical")
              and type(acceptance.get("numerically_validated")) is bool
              and acceptance["numerically_validated"] == (level == "numerical"),
              "output_execution_identity_rejected")
        return package, {
            "level": level,
            "encrypted_execution": level == "numerical",
            "numerically_validated": level == "numerical",
        }

    def bind(self, task, tree):
        """Bind an explicit host-selected output projection to a checked task."""
        self._task(task)
        outcome = self.application.result(task)
        frame = self._frame(outcome)
        if outcome["status"] != "succeeded":
            return frame
        try:
            metadata = self._metadata(task)
            package, validation = self._package(outcome, metadata)
            contract = build(metadata["request"]["model"], tree,
                             source_binding=source_binding(metadata))
            delivery = bind_delivery(
                contract, request_id=metadata["request"]["request_id"],
                bundle_binding=package["binding"], package_validation=validation,
            )
            receipt = {
                "format": RECEIPT, "task_id": task,
                "package_task_id": package["task_id"],
                "task_metadata_binding": metadata["binding"],
                "delivery": delivery, "adapter_sha256": implementation(),
            }
            receipt["binding"] = digest(receipt)
            frame["receipt"] = receipt
            return frame
        except OutputError as error:
            return self._error(outcome, error)

    def _verify(self, task, receipt, expected_binding, outcome):
        receipt = snapshot(receipt, limit=4 * 1024**2, code="output_binding_rejected")
        check(type(receipt) is dict and set(receipt) == {
            "format", "task_id", "package_task_id", "task_metadata_binding",
            "delivery", "adapter_sha256", "binding",
        } and receipt["format"] == RECEIPT
              and is_hash(expected_binding) and receipt["binding"] == expected_binding
              and digest({k: v for k, v in receipt.items() if k != "binding"}) == expected_binding
              and receipt["task_id"] == task
              and receipt["adapter_sha256"] == implementation(),
              "output_binding_rejected")
        metadata = self._metadata(task)
        package, validation = self._package(outcome, metadata)
        check(receipt["task_metadata_binding"] == metadata["binding"]
              and receipt["package_task_id"] == package["task_id"],
              "output_binding_rejected")
        delivery = receipt["delivery"]
        check(type(delivery) is dict, "output_binding_rejected")
        clean = verify_delivery(
            delivery, expected_binding=delivery.get("binding"),
            model=metadata["request"]["model"],
            source_binding=source_binding(metadata),
        )
        check(clean["request_id"] == metadata["request"]["request_id"]
              and clean["bundle_binding"] == package["binding"]
              and clean["package_validation"] == validation,
              "output_execution_identity_rejected")
        return receipt

    def result(self, task, receipt, *, expected_binding):
        """Recheck task, program and receipt; no generation or execution."""
        self._task(task)
        outcome = self.application.result(task)
        frame = self._frame(outcome)
        if outcome["status"] != "succeeded":
            return frame
        try:
            frame["receipt"] = self._verify(task, receipt, expected_binding, outcome)
            return frame
        except OutputError as error:
            return self._error(outcome, error)

    def execute(self, task, samples, receipt, *, expected_binding):
        """Use the existing checked executor, then restore caller output structure."""
        self._task(task)
        outcome = self.application.result(task)
        frame = self._frame(outcome)
        if outcome["status"] != "succeeded":
            return frame
        try:
            verified = self._verify(task, receipt, expected_binding, outcome)
            execution = self.application.execute(task, samples)
            structured = structure_execution(
                verified["delivery"], execution,
                expected_binding=verified["delivery"]["binding"],
            )
            frame.update(status=structured["status"], result=structured)
            return frame
        except OutputError as error:
            return self._error(outcome, error)
