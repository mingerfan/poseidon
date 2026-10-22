"""Task-bound calibration rejection controls; no source/FHE/API execution."""
import unittest,copy
from unittest.mock import patch
import test_task_numerical_assessment as fixture
from task_source_calibration import TaskSourceCalibration
from source_calibration_contract import FORMAT
from benchmark_graph import digest
from test_model_output_contract import model

class TaskCalibrationTests(unittest.TestCase):
    setUp=fixture.AssessmentTests.setUp
    ready=fixture.AssessmentTests.ready
    def selection(self):
        return dict(format=FORMAT,purpose="calibration",selectors=[dict(id="root_input",module="",
            argument=0,projection=dict(kind="tensor"),domain=None)])
    def call(self,task,outcome,**kwargs):
        selection=kwargs.pop("selection",self.selection())
        return TaskSourceCalibration(self.app).observe(task,self.samples,self.tmp.name+"/observation",
            expected_package_binding=kwargs.pop("binding",outcome["package"]["binding"]),
            selection=selection,expected_selection_sha256=kwargs.pop("selection_binding",digest(selection)),
            input_dtypes={"x":"float64"},trust_local_code=kwargs.pop("trusted",True),**kwargs)
    def test_nonterminal_and_cancelled_do_not_call_source(self):
        task=self.app.submit(model())
        with patch("task_source_calibration.run")as worker:
            value=TaskSourceCalibration(self.app).observe(task,self.samples,self.tmp.name+"/observation",
                expected_package_binding="a"*64,selection=self.selection(),
                expected_selection_sha256=digest(self.selection()),input_dtypes={"x":"float64"},trust_local_code=True)
            self.assertEqual(value["status"],"queued");worker.assert_not_called()
            self.app.cancel(task)
            value=TaskSourceCalibration(self.app).observe(task,self.samples,self.tmp.name+"/observation",
                expected_package_binding="a"*64,selection=self.selection(),
                expected_selection_sha256=digest(self.selection()),input_dtypes={"x":"float64"},trust_local_code=True)
            self.assertEqual(value["status"],"cancelled");worker.assert_not_called()
    def test_explicit_trust_is_required(self):
        task,_,outcome=self.ready()
        for trusted in (False,1,"yes",None):
            with self.subTest(trusted=trusted),patch("task_source_calibration.run")as worker:
                result=self.call(task,outcome,trusted=trusted)
                self.assertEqual(result["status"],"unsupported")
                self.assertEqual(result["failure"]["code"],"calibration_trust_required");worker.assert_not_called()
    def test_input_source_and_package_binding_rejections_do_not_run(self):
        task,_,outcome=self.ready()
        with patch("task_source_calibration.run")as worker:
            value=self.call(task,outcome,binding="0"*64)
            self.assertEqual(value["failure"]["code"],"calibration_binding_changed")
            value=self.call(task,outcome,selection_binding="0"*64)
            self.assertEqual(value["failure"]["code"],"calibration_binding_changed")
            value=self.call(task,outcome)
            self.assertEqual(value["status"],"unsupported")
            self.assertEqual(value["failure"]["code"],"calibration_source_unavailable")
            worker.assert_not_called()
        self.assertEqual(self.backend.execute_calls,0)
    def test_test_role_and_callback_fields_are_rejected(self):
        task,_,outcome=self.ready()
        for value in (dict(self.selection(),purpose="test"),dict(self.selection(),extra="callback"),
                      dict(self.selection(),purpose="validation")):
            with patch("task_source_calibration.run")as worker:
                result=self.call(task,outcome,selection=value)
                self.assertEqual(result["failure"]["code"],"calibration_input_rejected")
                worker.assert_not_called()
    def test_selection_detached_and_original_grade_kept(self):
        task,provider,outcome=self.ready();value=self.selection();before=copy.deepcopy(value)
        with patch("task_source_calibration.run")as worker:self.call(task,outcome,selection=value)
        self.assertEqual(value,before);self.assertEqual(self.app.result(task),outcome)
        self.assertEqual(provider.calls,1);self.assertEqual(self.backend.execute_calls,0)
        worker.assert_not_called()
if __name__=="__main__":unittest.main()
