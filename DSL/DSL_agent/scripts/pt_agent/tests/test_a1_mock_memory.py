import gc
import sys
import unittest
import weakref
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pt_agent.backends.mock import MockBackend
from pt_agent.experiments.a1_mock_memory import retained_size, snapshot
from pt_agent.dsl.parser import parse
from pt_agent import transformer_validation as validation
from pt_agent.experiments import a1_transformer_agent as experiment


SOURCE='''@fhe.program(slots=1024, refresh="compiler", placement="depth-dp")
def small(x: Cipher[1, 4]):
    a = pack(x, layout="flat", gap=1)
    y = act(a, fn="relu", method="cheb", deg=7, domain=(-2.0, 2.0))
    return y
'''


class TestMockMemoryDiagnostics(unittest.TestCase):
    def check_batch_release(self, *, fail=False):
        references=[]
        def backend(*args,**kwargs):
            value=MockBackend(*args,**kwargs)
            references.append(weakref.ref(value))
            return value
        values=np.array([[-1.0,-0.5,0.5,1.0],[-0.8,-0.2,0.2,0.8]])
        expected=values.copy()
        def observer(*args):
            if fail:raise RuntimeError('observer failure')
        with patch.object(validation,'MockBackend',side_effect=backend):
            if fail:
                with self.assertRaisesRegex(RuntimeError,'observer failure'):
                    validation.execute(parse(SOURCE),{},dict(x=values),engine='mock',observer=observer)
            else:
                actual=validation.execute(parse(SOURCE),{},dict(x=values),engine='mock',observer=observer)
                self.assertEqual(actual.shape,values.shape)
                self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.array_equal(values,expected))
        self.assertTrue(all(reference() is None for reference in references))

    def test_mock_batch_release_on_return(self):
        self.check_batch_release()

    def test_mock_batch_release_on_failure(self):
        self.check_batch_release(fail=True)

    def test_shared_array_counted_once(self):
        value = np.ones(12)
        values = [value, value]
        self.assertEqual(retained_size(values), sys.getsizeof(values)+sys.getsizeof(value))

    def test_snapshot_does_not_keep_backend_or_node(self):
        backend = MockBackend(8, batch=2)
        backend.bind([np.ones((2, 8))])
        value = backend.input(0)
        backend_ref, value_ref = weakref.ref(backend), weakref.ref(value)
        previous = backend._nid
        measured = snapshot()
        self.assertGreaterEqual(measured['live_mock_nodes'], 1)
        self.assertGreaterEqual(measured['live_mock_node_array_bytes'], value.value.nbytes)
        self.assertEqual(backend._nid, previous)
        self.assertTrue(np.array_equal(value.value, np.ones((2, 8))))
        del value, backend
        gc.collect()
        self.assertIsNone(backend_ref())
        self.assertIsNone(value_ref())


class TestSavedMockReport(unittest.TestCase):
    def run_report(self, *, accepted):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);program=root/'program.ptdsl';program.write_text(SOURCE)
            frozen=root/'FROZEN.json'
            frozen.write_text(json.dumps(dict(program=str(program),split_version='split-v2')))
            saved=root/'saved';saved.mkdir()
            values=np.ones((100,4))
            np.savez_compressed(saved/'outputs.npz',mock=values)
            (saved/'REPORT.json').write_text(json.dumps(dict(complete=True,program=str(program),
                engine='mock',report_images=100,release_checks=dict(live_backends=accepted),
                validation=dict(agree=100,total=100,diverged_indices=[]))))
            info=SimpleNamespace(split=dict(version='split-v2',report=list(range(100))))
            ctx=SimpleNamespace(val_images=np.ones((100,4)),val_exact_logits=values)
            calls=[]
            def validate(*args,**kwargs):
                calls.append(kwargs['engine'])
                self.assertEqual(kwargs['engine'],'tensor')
                kwargs['output_callback'](values.copy())
                return dict(agree=100,total=100,diverged_indices=[])
            with patch('pt_agent.planners.agent_tools.build_context',return_value=ctx), \
                 patch.object(experiment,'split_context',return_value=ctx), \
                 patch('pt_agent.dsl.interface.require_interface'), \
                 patch.object(validation,'validate',side_effect=validate):
                if accepted:
                    result=experiment.evaluate_frozen(info,{},None,program,root/'report',
                        frozen=frozen,prompt_version='r8-v8',mock_report=saved/'REPORT.json')
                    self.assertTrue(result['complete'])
                    self.assertEqual(calls,['tensor'])
                else:
                    with self.assertRaisesRegex(ValueError,'batch-release report'):
                        experiment.evaluate_frozen(info,{},None,program,root/'report',
                            frozen=frozen,prompt_version='r8-v8',mock_report=saved/'REPORT.json')
                    self.assertEqual(calls,[])

    def test_completed_mock_is_reused_without_execution(self):
        self.run_report(accepted=True)

    def test_failed_release_report_is_rejected(self):
        self.run_report(accepted=False)


if __name__ == '__main__':
    unittest.main()
