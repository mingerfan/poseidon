"""U3a section 9: ordered BN folding and temporary unavailable-import stubs."""

import gc
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pt_agent.backends import CountBackend
from pt_agent.dsl.parser import parse
from pt_agent.export import export_package
from pt_agent.export.codegen import generate
from pt_agent.lower import lower
from pt_agent.models import dacapo_builders as builders
from pt_agent.paths import DACAPO_UPSTREAM
from pt_agent.tensor_sim import run_tensor
from pt_agent.tests._artifacts import DATA, require
from pt_agent.tests.test_p2_export import emit_pair, package_modules
from pt_agent.tests.test_u3_kernels import mock, source


class BNFoldingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(99)

    def bn(self, channels):
        return {'bn.weight': self.rng.normal(size=channels), 'bn.bias': self.rng.normal(size=channels),
                'bn.running_mean': self.rng.normal(size=channels),
                'bn.running_var': self.rng.uniform(.5,2,size=channels)}

    def expected_bn(self, value, weights):
        return F.batch_norm(torch.from_numpy(value),torch.from_numpy(weights['bn.running_mean']),
            torch.from_numpy(weights['bn.running_var']),torch.from_numpy(weights['bn.weight']),
            torch.from_numpy(weights['bn.bias']),training=False).numpy()

    def package_equal(self, text, weights, images, expected, folded):
        program = parse(text)
        # The fixture's square is already method="exact"; program execution
        # therefore gives exact arithmetic without the activation override.
        self.assertTrue(all(op.kind != 'act' or op.attrs['method'] == 'exact' for op in program.ops))
        np.testing.assert_allclose(mock(program,weights,images),expected,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(run_tensor(program,weights,images,cache=False),expected,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(run_tensor(program,weights,images,cache=False),mock(program,weights,images),atol=1e-9,rtol=1e-9)
        code = generate(program)
        self.assertEqual("pre_bn='bn'" in code,folded)
        self.assertEqual('ops.batchnorm' in code,not folded)
        with TemporaryDirectory() as tmp:
            package = export_package(text,program,weights,Path(tmp))
            (a,ac),(b,bc) = emit_pair(text,weights,package)
            self.assertEqual(a,b)
            self.assertEqual(list(ac),list(bc))
            for name in ac:
                np.testing.assert_array_equal(ac[name],bc[name])
            with package_modules(package) as runtime:
                np.testing.assert_array_equal(runtime.run_numpy(weights,[images])[0],mock(program,weights,images))

    def test_bn_into_following_linear_single_multi(self):
        for size in (8,1025):
            with self.subTest(size=size):
                images = self.rng.normal(size=(2,size))
                weights = self.bn(size)
                weights.update(w=self.rng.normal(size=(4,size)),b=self.rng.normal(size=4))
                text = (f'@fhe.program(slots=1024)\ndef bn_head(x:Cipher[1,{size}]):\n'
                    ' p = pack(x,layout="flat")\n'
                    ' a = act(p,fn="square",method="exact")\n'
                    ' n = batchnorm(a,bn="bn")\n'
                    ' y = linear(n,w=W["w"],b=W["b"])\n return y\n')
                expected = self.expected_bn(images*images,weights)@weights['w'].T+weights['b']
                self.package_equal(text,weights,images,expected,True)

    def test_bn_before_conv_padding_zero_and_nonzero(self):
        for slots in (4096,1024):
            for kernel,pad in ((1,0),(3,1)):
                with self.subTest(slots=slots,pad=pad):
                    images = self.rng.normal(size=(2,8,16,16))
                    weights = self.bn(8)
                    weights.update(w=self.rng.normal(size=(4,8,kernel,kernel)),b=self.rng.normal(size=4))
                    text = source(slots,8,'    a = act(p,fn="square",method="exact")\n'
                        '    n = batchnorm(a,bn="bn")\n'
                        f'    y = conv2d(n,w=W["w"],b=W["b"],pad={pad})\n    return y')
                    before = self.expected_bn(images*images,weights)
                    expected = F.conv2d(torch.from_numpy(before),torch.from_numpy(weights['w']),
                                        torch.from_numpy(weights['b']),padding=pad).numpy()
                    self.package_equal(text,weights,images,expected,pad==0)
                    backend=CountBackend(slots)
                    lower(parse(text),weights,backend)
                    self.assertEqual(backend.stats()['max_depth'],3 if pad==0 else 4)


class StubTests(unittest.TestCase):
    def test_constructor_and_forward_access_fail_and_imports_cleaned(self):
        text = ('import torch\nfrom torchvision import datasets,transforms\n'
                'class Model(torch.nn.Module):\n'
                ' def __init__(self,use=False):\n'
                '  super().__init__()\n'
                '  if use: datasets.CIFAR10()\n'
                ' def forward(self,x): return transforms.Compose()(x)\n')
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.py';path.write_text(text)
            module=builders._load_definition(path,'StubFixture')
            for name in ('torchvision','torchvision.datasets','torchvision.transforms'):
                self.assertNotIsInstance(sys.modules.get(name),builders._TorchvisionStub)
            with self.assertRaisesRegex(RuntimeError,builders.TORCHVISION_ERROR):
                module.Model(True)
            with self.assertRaisesRegex(RuntimeError,builders.TORCHVISION_ERROR):
                module.Model()(torch.ones(1))


ASSETS = [DACAPO_UPSTREAM/'examples/data'/builders.VARIANTS[name][2] for name in
          ('alexnet_silu_avgpool','vgg16_silu_avgpool','mobilenet_silu','squeezenet_silu_avgpool')]


@require(DATA,ASSETS)
class OriginalBuilderTests(unittest.TestCase):
    def test_four_families_construct_weights_and_forward(self):
        from pt_agent.data.cifar import calibration_set
        torch.set_num_threads(1)
        images,_=calibration_set()
        for name in ('alexnet_silu_avgpool','vgg16_silu_avgpool','mobilenet_silu','squeezenet_silu_avgpool'):
            with self.subTest(model=name):
                model=getattr(builders,name)().double().eval()
                with torch.no_grad():
                    output=model(torch.from_numpy(images[:1]))
                self.assertEqual(tuple(output.shape),(1,10))
                self.assertTrue(torch.isfinite(output).all())
                for key in ('torchvision','torchvision.datasets','torchvision.transforms'):
                    self.assertNotIsInstance(sys.modules.get(key),builders._TorchvisionStub)
                del model,output;gc.collect()


if __name__=='__main__':
    unittest.main()
