"""Real adapter call boundaries with injected stops, not compilation/FHE."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from application_backend import PoseidonBackend
from application_execution.backend import ContextBackend
from application_execution.context import WorkerContext
from application_artifact_budget import OperationalStopped, MONITOR
from application_policy import prepare
from benchmark_suite import Builder
from hecate_python_env import WORK
from test_application_component import Provider

class ArtifactAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=WORK/"results",prefix="artifact-adapter-unit-")
        b=Builder([(2,)])
        cls.model=b.finish(b.node("square",["input0"]))
        cls.request=prepare(cls.model,"poseidon_cpu","numerical",None)[0]
        cls.candidate=json.loads(Provider().generate(cls.request,[]))
        cls.context=WorkerContext.capture(Path(cls.tmp.name)/"tasks")
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    def test_poseidon_backend_declares_fixed_evidence_roots(self):
        backend=PoseidonBackend()
        self.assertEqual(backend.artifact_monitor_version,MONITOR)
        self.assertEqual(backend.operational_artifact_roots(),(WORK/"results",))
    def test_poseidon_qualify_forwards_observer(self):
        observer=lambda p:None
        with patch("component_backend.qualify",return_value={}) as call:
            PoseidonBackend().qualify(self.request,self.candidate,"numerical",artifact_observer=observer)
        call.assert_called_once_with(self.request,self.candidate,validation_level="numerical",
                                     artifact_observer=observer)
    def test_poseidon_legacy_qualify_omits_optional_keyword(self):
        with patch("component_backend.qualify",return_value={}) as call:
            PoseidonBackend().qualify(self.request,self.candidate,"numerical")
        call.assert_called_once_with(self.request,self.candidate,validation_level="numerical")
    def test_context_qualify_preserves_observer_and_receipt(self):
        observer=lambda p:None
        with patch("application_backend.PoseidonBackend.qualify",return_value={"status":"failed"}) as call:
            result=ContextBackend(self.context).qualify(self.request,self.candidate,"numerical",
                                                       artifact_observer=observer)
        call.assert_called_once_with(self.request,self.candidate,"numerical",artifact_observer=observer)
        self.assertIn("run_context",result)
    def test_component_adapter_passes_observer_to_session(self):
        from component_backend import qualify
        observer=lambda p:None
        report=dict(status="failed",attempts=[],failure_layer="compiler")
        with patch("qualification_session.QualificationSession.qualify",
                   return_value=(1,report,Path(self.tmp.name))) as call:
            result=qualify(self.request,self.candidate,artifact_observer=observer)
        call.assert_called_once_with(self.request,self.candidate,validation_level="numerical",
                                     artifact_observer=observer)
        self.assertEqual(result["status"],"failed")
    def test_qualification_registers_fresh_directory_before_reference(self):
        from qualification_session import QualificationSession
        captured=[]
        def observer(folder):
            self.assertEqual(list(folder.iterdir()),[])
            captured.append(folder)
            raise OperationalStopped("artifact_bytes")
        with patch("unified_graph_prepare.prepare_case") as prepare_case:
            with self.assertRaises(OperationalStopped):
                QualificationSession.qualify(self.request,self.candidate,artifact_observer=observer)
        prepare_case.assert_not_called()
        self.assertEqual(len(captured),1)
        report=json.loads((captured[0]/"report.json").read_text())
        self.assertEqual(report["status"],"budget_exhausted")
        self.assertEqual(report["attempts"],[])
    def test_poseidon_execute_forwards_observer(self):
        observer=lambda p:None
        with patch("program_execution.execute_program",return_value={}) as call:
            PoseidonBackend().execute(Path(self.tmp.name),[],artifact_observer=observer)
        call.assert_called_once_with(Path(self.tmp.name),[],artifact_observer=observer)
    def test_execution_registers_before_copy_or_native_runner(self):
        from program_execution import execute_program
        captured=[]
        def observer(folder):
            self.assertEqual(list(folder.iterdir()),[])
            captured.append(folder)
            raise OperationalStopped("artifact_bytes")
        with patch("candidate_bundle.check_execution_dependencies",return_value=({},self.request)), \
             patch("program_execution.package_execution_metadata",return_value={
                 "original_model":self.model,"output_structure":None}), \
             patch("poseidon_package.copy_checked_artifacts") as copy_artifacts, \
             patch("poseidon_validation.execute") as execute:
            with self.assertRaises(OperationalStopped):
                execute_program(Path(self.tmp.name),[{"input0":[0.,1.]}],artifact_observer=observer)
        self.assertEqual(len(captured),1)
        copy_artifacts.assert_not_called();execute.assert_not_called()

if __name__=="__main__":unittest.main()
