"""Multi-ciphertext package interfaces and check-time spatial diagnostics."""

from pathlib import Path
from tempfile import TemporaryDirectory
import importlib
import json
import subprocess
import sys
import unittest

import numpy as np
import torch
from torch import nn

from pt_agent.backends import EmitBackend, MockBackend
from pt_agent.dsl.parser import parse
from pt_agent.export import export_package
from pt_agent.intake import import_module
from pt_agent.intake.coverage import check
from pt_agent.intake.probe import model_info
from pt_agent.lower import lower, pack_inputs, decode_outputs, LoweringError
from pt_agent.tensor_sim import run_tensor
from pt_agent.tests.test_p2_export import emit_pair, mock_reference, package_modules


class PackageTests(unittest.TestCase):
    def program(self):
        return ('@fhe.program(slots=1024)\ndef multi_example(x: Cipher[1,1025]):\n'
                ' a = pack(x, layout="flat")\n'
                ' squared = act(a, fn="square", method="exact")\n'
                ' result = linear(squared, w=W["weight"], b=W["bias"], algo="bsgs")\n return result,squared\n')

    def test_interface_emit_numpy_and_cli_slots(self):
        rng = np.random.default_rng(1025)
        text = self.program(); p = parse(text)
        x = rng.normal(size=(2, 1025))
        weights = dict(weight=rng.normal(size=(513, 1025)), bias=rng.normal(size=513))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = export_package(text, p, weights, root)
            (old, oc), (new, nc) = emit_pair(text, weights, package)
            self.assertEqual(new, old)
            self.assertEqual(list(nc), list(oc))
            for name in oc:
                np.testing.assert_array_equal(nc[name], oc[name])
            interface = json.loads((package/'interfaces/default.json').read_text())
            self.assertEqual(interface['inputs'][0]['ciphertexts'], [0, 1, 2])
            self.assertEqual([entry['ciphertexts'] for entry in interface['outputs']], [[0, 1], [2, 3, 4]])
            backend = MockBackend(1024, batch=2); backend.bind(pack_inputs(p, [x]))
            values, layouts = lower(p, weights, backend)
            expected = decode_outputs([value.value for value in values], layouts)
            with package_modules(package) as runtime:
                actual = runtime.run_numpy(weights, [x])
                for a, z in zip(actual, expected):
                    np.testing.assert_array_equal(a, z)
                self.assertEqual(len(runtime.pack_arrays([x])), 3)
                # The fake frontend exercises the actual hc.func signature.
                from pt_agent.tests.fixtures import fake_hecate_expr as fake
                class HC:
                    def func(self, signature):
                        self.signature = signature
                        return lambda fn: fn
                hc = HC(); runtime.register(hc, weights)
                self.assertEqual(hc.signature, 'c,c,c')
                output_slots = [v.value for v in values]
                runtime.save_arrays(root/'decrypted.npz', output_slots, runtime.ciphertext_entries(interface['outputs']))
            np.save(root/'inputs.npy', x)
            subprocess.run([sys.executable, '-B', str(package/'prepare_inputs.py'), '--inputs', str(root/'inputs.npy'),
                            '--output', str(root/'packed.npz')], check=True)
            with np.load(root/'packed.npz') as data:
                self.assertEqual(len(data.files), 3)
                for i, packed in enumerate(pack_inputs(p, [x])):
                    np.testing.assert_array_equal(data['x__ct'+str(i)], packed)
            subprocess.run([sys.executable, '-B', str(package/'unpack_outputs.py'), '--signature',
                            str(package/'interfaces/default.json'), '--slots', str(root/'decrypted.npz'),
                            '--output', str(root/'unpacked.npz')], check=True)
            with np.load(root/'unpacked.npz') as data:
                for entry, value in zip(interface['outputs'], expected):
                    np.testing.assert_array_equal(data[entry['name']], value)

    def test_lower_tensor_conv_mean_add(self):
        rng = np.random.default_rng(55)
        for ci, co, stride in ((2, 9, 1), (9, 2, 1), (13, 9, 2)):
            text = (f'@fhe.program(slots=1024)\ndef f(x: Cipher[1,{ci},16,16]):\n'
                    ' a = pack(x, layout="mux")\n'
                    f' convolved = conv2d(a, w=W["w"], stride=({stride},{stride}), pad=(1,1))\n'
                    ' c = act(convolved, fn="square", method="exact")\n d = add(c,c)\n'
                    ' e = mean(d, axis=(2,3), keepdim=False)\n return e\n')
            p = parse(text); weights = dict(w=rng.normal(size=(co, ci, 3, 3)))
            x = rng.normal(size=(2, ci, 16, 16))
            actual = mock_reference(p, weights, x)
            expected = run_tensor(p, weights, x, cache=False)
            self.assertTrue(np.all(np.abs(actual-expected) <= 1e-9*(1+np.abs(expected))))
            with TemporaryDirectory() as directory:
                package = export_package(text, p, weights, directory)
                (old, oc), (new, nc) = emit_pair(text, weights, package)
                self.assertEqual(old, new)
                self.assertEqual(list(oc), list(nc))
                for name in oc:
                    np.testing.assert_array_equal(oc[name], nc[name])

    def test_spatial_import_check_and_lower_error(self):
        module = nn.Conv2d(3, 4, 3, padding=1).double().eval()
        graph = import_module(module, (torch.zeros(1, 3, 224, 224, dtype=torch.float64),), 'large', frontend='aten')
        checked = check(graph)
        self.assertFalse(checked['ok'])
        diag = next(d for d in checked['diags'] if 'spatial tiling not supported' in d['reason'])
        self.assertIsNotNone(diag['op_id']); self.assertGreater(diag['lineno'], 0)
        info = model_info(dict(type='torch'), graph, split={}, has_labels=False)
        self.assertIsNone(info.inputs[0]['ciphertexts'])
        self.assertIn('spatial tiling not supported', info.inputs[0]['layout']['limitation'])
        text = ('@fhe.program(slots=16384)\ndef f(x: Cipher[1,3,224,224]):\n'
                ' a = pack(x, layout="mux")\n return a\n')
        from pt_agent.backends import CountBackend
        with self.assertRaises(LoweringError) as caught:
            lower(parse(text), {}, CountBackend(16384))
        self.assertEqual(caught.exception.op_id, 0)
        self.assertEqual(caught.exception.lineno, 3)
        self.assertIn('spatial tiling not supported', caught.exception.msg)

    def test_model_info_summary(self):
        from pt_agent.models.info import ModelInfo
        info = ModelInfo({}, None, {}, [dict(name='x', shape=[1, 3072])], None, 'regression', '', {}, {})
        info.describe_layouts(4096)
        self.assertEqual(info.inputs[0]['ciphertexts'], 2)
        self.assertEqual(info.inputs[0]['layout']['kind'], 'multi_flat')
        with TemporaryDirectory() as directory:
            info.save(directory)
            self.assertEqual(ModelInfo.load(directory).inputs, info.inputs)


if __name__ == '__main__':
    unittest.main()
