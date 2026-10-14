"""Task-bound calibration dataset receipts and weighted module statistics.

No generation, execution loop, fitting or accuracy certification takes place
here. The host schedules bounded TaskSourceCalibration calls explicitly.
"""
import hashlib, math, copy
from pathlib import Path
from benchmark_graph import digest
from model_output_contract import snapshot, is_hash
from assessment_array_dataset import load_dataset
from task_output_adapter import TaskOutputAdapter, source_binding
from source_calibration_contract import checked, CalibrationError, FX_FORMAT
from source_calibration import check_report, implementation as observer_identity
from source_precision_reference import prepare_source, checked_samples
from task_source_calibration import FORMAT as FRAME_FORMAT, FX_FRAME
from strict_json_files import strict_file

FORMAT = "poseidon-task-calibration-dataset-v1"
DECLARATION = "poseidon-calibration-dataset-declaration-v1"
MAX_RECEIPT_BYTES = 64 * 1024**2
MAX_PART_BYTES = 16 * 1024**2

class CalibrationDatasetError(ValueError):
    def __init__(self, code):
        if code not in {
            "calibration_dataset_input_rejected", "calibration_dataset_binding_changed",
            "calibration_dataset_receipt_rejected", "calibration_dataset_duplicate_part",
            "calibration_dataset_receipt_budget_exhausted",
        }:
            raise ValueError("Unknown calibration dataset diagnostic")
        self.code = code
        super().__init__(code)

    def feedback(self):
        return dict(code=self.code, layer="calibration", retry_policy="never_automatic")

def require(value, code="calibration_dataset_input_rejected"):
    if not value:
        raise CalibrationDatasetError(code)

def identity():
    return digest(dict(
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observer=observer_identity(),
        adapter_sha256=hashlib.sha256(Path(__file__).with_name("task_source_calibration.py").read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(Path(__file__).with_name("assessment_array_dataset.py").read_bytes()).hexdigest(),
    ))

def declaration(value, binding):
    value = snapshot(value, limit=131072)
    require(type(value) is dict and set(value) == {"format", "role", "dataset_binding", "source"})
    require(value["format"] == DECLARATION and value["role"] == "calibration")
    require(is_hash(value["dataset_binding"]) and is_hash(binding) and digest(value) == binding,
            "calibration_dataset_binding_changed")
    origin = value["source"]
    require(type(origin) is dict and set(origin) == {"name", "revision"})
    require(all(type(origin[k]) is str and 1 <= len(origin[k]) <= 256 and
                all(ord(c) >= 32 for c in origin[k]) for k in origin))
    return value

def aggregate(selection, observations):
    """Merge population moments, weighting by observed values, not batches."""
    result = {}
    for selector in selection["selectors"]:
        rows = [item[selector["id"]] for item in observations]
        if not rows:
            continue
        total = sum(row["values"] for row in rows)
        mean = math.fsum(row["mean"] * (row["values"] / total) for row in rows)
        variance = math.fsum((row["values"] / total) *
                            (row["std_population"]**2 + (row["mean"] - mean)**2)
                            for row in rows)
        require(math.isfinite(mean) and math.isfinite(variance) and variance >= 0,
                "calibration_dataset_receipt_rejected")
        outside = sum(row["outside_domain_calls"] for row in rows)
        result[selector["id"]] = dict(
            **{key:selector[key]for key in (
                ("node","op","target","point","argument","projection")if "node"in selector else
                ("module","argument","projection"))},declared_domain=selector["domain"],
            samples=sum(row["samples"] for row in rows),
            calls=sum(row["calls"] for row in rows), values=total,
            minimum=min(row["minimum"] for row in rows),
            maximum=max(row["maximum"] for row in rows),
            mean=mean, std_population=math.sqrt(variance),
            dtypes=sorted({v for row in rows for v in row["dtypes"]}),
            shapes=[list(s) for s in sorted({tuple(s) for row in rows for s in row["shapes"]})],
            outside_domain_calls=outside,
            domain_status=("not_declared" if selector["domain"] is None else
                           "outside" if outside else "inside"),
        )
    return result

class TaskCalibrationDataset:
    """Deterministic receipt collector; it never launches or retries a worker.

    Cached receipts must retain their original evidence directories. A fresh
    collector revalidates those artifacts and their input/source identities.
    The host owns execution budgets, cancellation and an attempt journal.
    """
    def __init__(self, application, task, dataset_path, *,
                 expected_dataset_binding, calibration_declaration,
                 expected_declaration_sha256, expected_package_binding,
                 selection, expected_selection_sha256, input_dtypes,
                 part_samples=16, trust_local_code=False,
                 max_receipt_bytes=MAX_RECEIPT_BYTES, cached_receipts=None):
        require(trust_local_code is True)
        require(type(part_samples) is int and 1 <= part_samples <= 16)
        require(type(max_receipt_bytes) is int and 1 <= max_receipt_bytes <= MAX_RECEIPT_BYTES)
        self.application = application
        self.task = task
        self.outputs = TaskOutputAdapter(application)
        self.selection = checked(selection)
        require(self.selection["format"] != FX_FORMAT or self.selection["mode"] == "observe")
        require(is_hash(expected_selection_sha256) and
                digest(self.selection) == expected_selection_sha256,
                "calibration_dataset_binding_changed")
        self.declaration = declaration(calibration_declaration, expected_declaration_sha256)
        require(self.declaration["dataset_binding"] == expected_dataset_binding,
                "calibration_dataset_binding_changed")
        self.input_dtypes = snapshot(input_dtypes, limit=131072)
        self.initial = self._current()
        require(self.initial["bundle_binding"] == expected_package_binding,
                "calibration_dataset_binding_changed")
        self.dataset = load_dataset(self.metadata["request"], dataset_path,
                                    expected_binding=expected_dataset_binding)
        entries = self.metadata["request"]["model"]["inputs"]
        require(type(self.input_dtypes) is dict and
                set(self.input_dtypes) == {entry["name"] for entry in entries} and
                all(v in ("float32", "float64") for v in self.input_dtypes.values()))
        for name, record in self.dataset.manifest["inputs"].items():
            require(record["dtype"] == {"float32": "<f4", "float64": "<f8"}[self.input_dtypes[name]])
        self.parts = {item["id"]: item for item in self.dataset.batches(part_samples)}
        require(1 <= len(self.parts) <= 4096)
        self.plan = dict(
            format=FORMAT, anchor=self.initial,
            dataset_binding=expected_dataset_binding,
            declaration_sha256=expected_declaration_sha256,
            declaration=self.declaration, selection=self.selection,
            selection_sha256=expected_selection_sha256,
            input_dtypes=self.input_dtypes, part_samples=part_samples,
            max_receipt_bytes=max_receipt_bytes, samples=self.dataset.manifest["samples"],
            parts={name: dict(sample_ids=part["sample_ids"], samples=len(part["samples"]),
                             input_sha256=digest([{n: a.tolist() for n, a in sample.items()}
                                                  for sample in part["samples"]]))
                   for name, part in self.parts.items()},
        )
        self.plan["binding"] = digest(self.plan)
        self._receipts = {}
        self._artifacts = {}
        self._receipt_bytes = 0
        cache = {} if cached_receipts is None else snapshot(cached_receipts, limit=max_receipt_bytes)
        require(type(cache) is dict)
        for name, item in cache.items():
            require(type(item) is dict and set(item) ==
                    {"frame", "binding", "plan_binding", "evidence_sha256", "request_sha256"})
            require(item["plan_binding"] == self.plan["binding"],
                    "calibration_dataset_binding_changed")
            self.record(name, item["frame"], expected_binding=item["binding"])
            require(self._receipts[name] == item, "calibration_dataset_binding_changed")

    def _current(self):
        self.outputs._task(self.task)
        outcome = self.application.result(self.task)
        require(outcome["status"] == "succeeded")
        metadata = self.outputs._metadata(self.task)
        package, grade = self.outputs._package(outcome, metadata)
        origin = metadata.get("model_source")
        require(type(origin) is dict and origin.get("kind") == "trusted_pytorch")
        source = self.application._source(self.application._dir(self.task))
        require(source["binding"] == origin["snapshot_binding"],
                "calibration_dataset_binding_changed")
        from candidate_bundle import check_provenance
        check_provenance(metadata["model_provenance"], metadata["request"]["model"])
        capture = metadata["model_provenance"]["decomposition"]["pytorch_capture"]
        require(capture.get("trusted_code") is True)
        self.metadata = metadata
        self.source = source
        self.capture = capture
        return dict(task_id=self.task, request_id=metadata["request"]["request_id"],
                    task_result_sha256=digest(outcome), metadata_binding=metadata["binding"],
                    bundle_binding=package["binding"], package_validation=grade,
                    source_binding=source_binding(metadata), source_snapshot_binding=source["binding"],
                    implementation_identity=identity())

    def _verify_current(self):
        require(self.plan["binding"] == digest({k: v for k, v in self.plan.items() if k != "binding"}) and
                self.plan["anchor"] == self.initial and self.plan["selection"] == self.selection and
                digest(self.selection) == self.plan["selection_sha256"] and
                self.plan["input_dtypes"] == self.input_dtypes and self._current() == self.initial,
                "calibration_dataset_binding_changed")

    def batches(self):
        self._verify_current()
        return copy.deepcopy(list(self.parts.values()))

    @property
    def receipts(self):
        self._verify_current()
        return snapshot(self._receipts, limit=self.plan["max_receipt_bytes"])

    def record(self, part_id, frame, *, expected_binding):
        require(type(part_id) is str and part_id in self.parts)
        require(part_id not in self._receipts, "calibration_dataset_duplicate_part")
        self._verify_current()
        frame = snapshot(frame, limit=MAX_PART_BYTES)
        require(type(frame) is dict and is_hash(expected_binding) and
                frame.get("binding") == expected_binding and
                digest({k: v for k, v in frame.items() if k != "binding"}) == expected_binding,
                "calibration_dataset_binding_changed")
        require(frame.get("format") == (FX_FRAME if self.selection["format"]==FX_FORMAT else FRAME_FORMAT) and
                frame.get("status") == "observed" and
                frame.get("calibration_observed") is True and frame.get("encrypted_execution") is False and
                frame.get("approximation_modified") is False and frame.get("reference_modified") is False and
                frame.get("application_accuracy_certified") is False and
                type(frame.get("agent_calls")) is int and frame["agent_calls"] == 0 and
                type(frame.get("compiler_calls")) is int and frame["compiler_calls"] == 0,
                "calibration_dataset_receipt_rejected")
        for key in ("task_id", "request_id", "bundle_binding", "package_validation",
                    "source_binding", "source_snapshot_binding"):
            require(frame.get(key) == self.initial[key], "calibration_dataset_binding_changed")
        require(frame.get("selection_sha256") == self.plan["selection_sha256"],
                "calibration_dataset_binding_changed")
        from hecate_python_env import WORK
        evidence = Path(frame.get("evidence", ""))
        require(evidence.is_absolute() and evidence == evidence.resolve() and
                evidence.is_relative_to(WORK/"results") and evidence.name == "calibration.json" and
                not evidence.is_symlink(), "calibration_dataset_receipt_rejected")
        report = strict_file(evidence, MAX_PART_BYTES)
        request = strict_file(evidence.parent/"request.json", MAX_PART_BYTES)
        require(frame.get("calibration_report") == report and
                frame.get("calibration_report_binding") == report.get("binding"),
                "calibration_dataset_binding_changed")
        spec, _ = prepare_source(self.application._dir(self.task)/"source/manifest.json")
        part = self.parts[part_id]
        wire = [{name: value.tolist() for name, value in sample.items()} for sample in part["samples"]]
        require(digest(wire) == self.plan["parts"][part_id]["input_sha256"],
                "calibration_dataset_binding_changed")
        wire, dtypes, tree, shapes = checked_samples(
            spec, self.metadata["request"]["model"], wire, self.input_dtypes,
            self.capture["output_structure"], self.source["resource_profile"])
        require(request.get("samples") == wire and request.get("manifest") == spec and
                request.get("input_dtypes") == dtypes and request.get("output_structure") == tree and
                request.get("output_shapes") == shapes and request.get("selection") == self.selection and
                request.get("implementation") == observer_identity() and
                frame.get("supplied_input_sha256") == digest(wire),
                "calibration_dataset_binding_changed")
        from trusted_pytorch_import import runtime_identity
        require(request.get("runtime") == runtime_identity(), "calibration_dataset_binding_changed")
        check_report(report, request)
        item = dict(frame=frame, binding=expected_binding, plan_binding=self.plan["binding"],
                    evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                    request_sha256=hashlib.sha256((evidence.parent/"request.json").read_bytes()).hexdigest())
        from benchmark_graph import canonical
        size = len(canonical(item))
        require(self._receipt_bytes + size <= self.plan["max_receipt_bytes"],
                "calibration_dataset_receipt_budget_exhausted")
        self._receipts[part_id] = item
        self._receipt_bytes += size
        self._artifacts[part_id] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (evidence, evidence.parent/"request.json")
        }

    def summary(self):
        self._verify_current()
        from model_weight_package import regular_bytes
        for artifact in self._artifacts.values():
            for filename, expected in artifact.items():
                require(hashlib.sha256(regular_bytes(Path(filename), MAX_PART_BYTES)).hexdigest() == expected,
                        "calibration_dataset_binding_changed")
        selected = [self._receipts[name]["frame"]["calibration_report"]["observations"]
                    for name in self.parts if name in self._receipts]
        complete = len(self._receipts) == len(self.parts)
        result = dict(
            format=FORMAT, status="observed" if complete else "partial",
            plan_binding=self.plan["binding"], dataset_binding=self.plan["dataset_binding"],
            declaration_sha256=self.plan["declaration_sha256"], data_role="declared_calibration",
            source_provenance_independently_certified=False,
            package_validation=self.initial["package_validation"],
            total_parts=len(self.parts), completed_parts=len(self._receipts), complete=complete,
            planned_samples=self.plan["samples"],
            observed_samples=sum(len(self.parts[n]["samples"]) for n in self._receipts),
            observations=aggregate(self.selection, selected),
            receipts={n: v["binding"] for n, v in self._receipts.items()},
            receipt_bytes=self._receipt_bytes, encrypted_execution=False, new_agent_calls=0,
            new_compiler_calls=0, new_encrypted_calls=0, approximation_modified=False,
            reference_modified=False, application_accuracy_certified=False,
            universal_domain_guarantee=False, approximation_fitted=False,
        )
        result["binding"] = digest(result)
        return result
