"""Receipt collection and role/identity controls; injected source, no execution."""
import copy, json, math, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from benchmark_graph import canonical, digest
from assessment_array_dataset import export_dataset
from test_model_output_contract import model, tensor, VALIDATION
import test_source_calibration as observer_fixture
import task_calibration_dataset as cal
from source_calibration_contract import FORMAT as SELECTION
from task_source_calibration import FORMAT as FRAME

class CalibrationDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        adapter = patch("task_calibration_dataset.TaskOutputAdapter")
        adapter.start();self.addCleanup(adapter.stop)
        self.request = {"model": model(), "request_id": "b"*64}
        self.arrays = {"x": np.arange(6, dtype=np.float64).reshape(3, 2)/8}
        self.manifest = export_dataset(self.request, self.arrays, self.root/"data")
        self.decl = dict(format=cal.DECLARATION, role="calibration",
                         dataset_binding=self.manifest["binding"],
                         source=dict(name="unit-inputs", revision="1"))
        self.selection = dict(format=SELECTION, purpose="calibration",
                              selectors=[dict(id="values", module="", argument=0,
                                              projection=dict(kind="tensor"), domain=[0., .4])])
        self.anchor = dict(task_id="unit-task", request_id="b"*64, task_result_sha256="a"*64,
                           metadata_binding="d"*64, bundle_binding="c"*64, package_validation=VALIDATION,
                           source_binding="e"*64, source_snapshot_binding="f"*64,
                           implementation_identity=cal.identity())
        self.spec = {"format": "unit-observation-source"}
        def current(obj):
            obj.metadata = {"request": self.request}
            obj.source = {"resource_profile": "general-tensors-v1"}
            obj.capture = {"output_structure": tensor()}
            return copy.deepcopy(self.anchor)
        self.current = patch.object(cal.TaskCalibrationDataset, "_current", current)
        self.current.start();self.addCleanup(self.current.stop)
        for target, value in (
            ("hecate_python_env.WORK", self.root),
            ("task_calibration_dataset.prepare_source", (self.spec, {})),
            ("task_calibration_dataset.checked_samples", None),
            ("trusted_pytorch_import.runtime_identity", {"unit": "runtime"}),
        ):
            p = patch(target, return_value=value) if target != "hecate_python_env.WORK" else patch(target, value)
            mocked = p.start();self.addCleanup(p.stop)
            if target.endswith("checked_samples"):
                mocked.side_effect = lambda spec, model, wire, types, tree, profile: (wire, types, tree, {"out": [2]})

    def collector(self, **changes):
        args = dict(expected_dataset_binding=self.manifest["binding"],
                    calibration_declaration=self.decl, expected_declaration_sha256=digest(self.decl),
                    expected_package_binding="c"*64, selection=self.selection,
                    expected_selection_sha256=digest(self.selection),
                    input_dtypes={"x": "float64"}, part_samples=2, trust_local_code=True)
        args.update(changes)
        return cal.TaskCalibrationDataset(object(), "unit-task", self.root/"data/dataset.json", **args)

    def receipt(self, collector, name):
        part = collector.parts[name]
        wire = [{n: a.tolist() for n, a in sample.items()} for sample in part["samples"]]
        values = np.concatenate([sample["x"].reshape(-1) for sample in part["samples"]])
        count = len(wire)
        fixture = observer_fixture.CalibrationReportTests();fixture.setUp()
        report = copy.deepcopy(fixture.report)
        request = dict(manifest=self.spec, samples=wire, input_dtypes={"x": "float64"},
                       output_structure=tensor(), output_shapes={"out": [2]},
                       selection=self.selection, implementation=cal.observer_identity(),
                       runtime={"unit": "runtime"}, host_net_namespace="unit-host")
        observed = dict(module="", argument=0, projection=dict(kind="tensor"), samples=count,
                        calls=count, values=int(values.size), minimum=float(values.min()),
                        maximum=float(values.max()), mean=float(values.mean()),
                        std_population=float(values.std()), dtypes=["float64"], shapes=[[2]],
                        declared_domain=[0., .4], outside_domain_calls=sum(
                            bool((sample["x"]>.4).any()) for sample in part["samples"]))
        observed["domain_status"] = "outside" if observed["outside_domain_calls"] else "inside"
        report.update(request_sha256=digest(request), implementation=request["implementation"],
                      manifest_sha256=digest(self.spec), selection_sha256=digest(self.selection),
                      runtime=request["runtime"], input_dtypes=request["input_dtypes"],
                      samples=count, observations={"values": observed},
                      forward_output_sha256=["0"*64]*count)
        report["sandbox"]["network_namespace"]="unit-worker"
        report["binding"] = digest({k:v for k,v in report.items() if k!="binding"})
        path = self.root/"results"/name;path.mkdir(parents=True)
        (path/"calibration.json").write_bytes(canonical(report))
        (path/"request.json").write_bytes(canonical(request))
        frame = dict(format=FRAME, status="observed", calibration_observed=True,
                     encrypted_execution=False, approximation_modified=False, reference_modified=False,
                     application_accuracy_certified=False, agent_calls=0, compiler_calls=0,
                     **{k: self.anchor[k] for k in ("task_id","request_id","bundle_binding",
                        "package_validation","source_binding","source_snapshot_binding")},
                     selection_sha256=digest(self.selection), supplied_input_sha256=digest(wire),
                     calibration_report_binding=report["binding"], calibration_report=report,
                     evidence=str(path/"calibration.json"))
        frame["binding"]=digest(frame)
        return frame

    def put(self, collector, name):
        frame=self.receipt(collector,name)
        collector.record(name,frame,expected_binding=frame["binding"])
        return frame

    def test_empty_and_partial_never_claim_complete_or_accuracy(self):
        c=self.collector();r=c.summary()
        self.assertEqual(r["status"],"partial");self.assertEqual(r["observations"],{})
        self.put(c,"part_0");r=c.summary()
        self.assertFalse(r["complete"]);self.assertEqual(r["observed_samples"],2)
        self.assertFalse(r["application_accuracy_certified"]);self.assertFalse(r["encrypted_execution"])

    def test_unequal_batches_merge_population_moments(self):
        c=self.collector();self.put(c,"part_0");self.put(c,"part_1")
        r=c.summary();stats=r["observations"]["values"]
        self.assertTrue(r["complete"]);self.assertEqual(stats["values"],6)
        self.assertEqual(stats["mean"],float(self.arrays["x"].mean()))
        self.assertAlmostEqual(stats["std_population"],float(self.arrays["x"].std()),places=15)
        self.assertEqual(stats["outside_domain_calls"],1);self.assertEqual(stats["domain_status"],"outside")
        self.assertEqual(r["data_role"],"declared_calibration")

    def test_cached_receipt_resume_and_old_grade_are_kept(self):
        c=self.collector();self.put(c,"part_0")
        other=self.collector(cached_receipts=c.receipts)
        self.assertEqual(other.summary()["binding"],c.summary()["binding"])
        self.put(other,"part_1")
        self.assertEqual(other.summary()["package_validation"],VALIDATION)
        self.assertEqual(other.summary()["observed_samples"],3)

    def test_duplicate_and_wrong_part_are_rejected(self):
        c=self.collector();f=self.put(c,"part_0")
        with self.assertRaises(cal.CalibrationDatasetError):c.record("part_0",f,expected_binding=f["binding"])
        with self.assertRaises(cal.CalibrationDatasetError):c.record("part_1",f,expected_binding=f["binding"])
        self.assertEqual(c.summary()["completed_parts"],1)

    def test_role_and_unknown_fields_are_rejected(self):
        for role in ("test","validation","training",None):
            value=dict(self.decl,role=role)
            with self.subTest(role=role),self.assertRaises(cal.CalibrationDatasetError):
                self.collector(calibration_declaration=value,expected_declaration_sha256=digest(value))
        value=dict(self.decl,callback="bad")
        with self.assertRaises(cal.CalibrationDatasetError):
            self.collector(calibration_declaration=value,expected_declaration_sha256=digest(value))

    def test_trust_and_budget_bounds_are_rejected(self):
        for opts in (dict(trust_local_code=1),dict(trust_local_code=False),
                     dict(part_samples=17),dict(part_samples=True),
                     dict(max_receipt_bytes=0),dict(max_receipt_bytes=64*1024**2+1)):
            with self.subTest(opts=opts),self.assertRaises(cal.CalibrationDatasetError):self.collector(**opts)

    def test_wrong_dtype_and_declaration_binding_are_rejected(self):
        for opts in (dict(input_dtypes={"x":"float32"}),dict(input_dtypes={"x":"int64"}),
                     dict(expected_declaration_sha256="0"*64),dict(expected_dataset_binding="0"*64),
                     dict(expected_package_binding="0"*64)):
            with self.subTest(opts=opts),self.assertRaises(cal.CalibrationDatasetError):self.collector(**opts)

    def test_receipt_budget_preserves_the_recorded_prefix(self):
        c=self.collector(max_receipt_bytes=1);f=self.receipt(c,"part_0")
        with self.assertRaises(cal.CalibrationDatasetError)as error:
            c.record("part_0",f,expected_binding=f["binding"])
        self.assertEqual(error.exception.code,"calibration_dataset_receipt_budget_exhausted")
        self.assertEqual(c.summary()["completed_parts"],0)

    def test_task_or_implementation_drift_is_rejected(self):
        c=self.collector();self.anchor["bundle_binding"]="0"*64
        with self.assertRaises(cal.CalibrationDatasetError):c.summary()
        self.anchor["bundle_binding"]="c"*64
        self.anchor["implementation_identity"]="0"*64
        with self.assertRaises(cal.CalibrationDatasetError):c.batches()

    def test_plan_and_input_tampering_are_rejected(self):
        c=self.collector();c.plan["selection"]["selectors"][0]["domain"]=[-1.,1.]
        with self.assertRaises(cal.CalibrationDatasetError):c.summary()
        self.selection["selectors"][0]["domain"]=[0.,.4]
        c=self.collector();f=self.receipt(c,"part_0");c.parts["part_0"]["samples"][0]["x"][0]=99.
        with self.assertRaises(cal.CalibrationDatasetError):c.record("part_0",f,expected_binding=f["binding"])

    def test_evidence_tampering_is_rejected_even_when_parsing_still_works(self):
        c=self.collector();f=self.put(c,"part_0")
        Path(f["evidence"]).write_bytes(Path(f["evidence"]).read_bytes()+b" ")
        with self.assertRaises(cal.CalibrationDatasetError):c.summary()

    def test_frame_report_and_input_hash_forgery_are_rejected(self):
        c=self.collector();f=self.receipt(c,"part_0")
        for key,value in (("supplied_input_sha256","0"*64),("bundle_binding","0"*64),
                          ("reference_modified",True),("agent_calls",1)):
            bad=copy.deepcopy(f);bad[key]=value;bad["binding"]=digest({k:v for k,v in bad.items()if k!="binding"})
            with self.subTest(key=key),self.assertRaises(cal.CalibrationDatasetError):
                c.record("part_0",bad,expected_binding=bad["binding"])

    def test_batches_and_cached_frames_are_detached(self):
        c=self.collector();given=c.batches();given[0]["samples"][0]["x"][0]=99.
        self.assertNotEqual(c.parts["part_0"]["samples"][0]["x"][0],99.)
        self.put(c,"part_0");cache=c.receipts;cache["part_0"]["frame"]["task_id"]="bad"
        self.assertEqual(c.summary()["completed_parts"],1)

    def test_array_content_change_invalidates_a_new_collector(self):
        c=self.collector();file=self.root/"data/input_0.npy"
        raw=file.read_bytes();file.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        self.assertEqual(len(c.batches()),2)
        with self.assertRaises(ValueError):self.collector()

    def test_wrong_receipt_binding_is_rejected(self):
        c=self.collector();f=self.receipt(c,"part_0")
        with self.assertRaises(cal.CalibrationDatasetError):c.record("part_0",f,expected_binding="0"*64)

if __name__=="__main__":unittest.main()
