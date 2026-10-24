"""U3a package, approximation-point and unsupported pool variant contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn

from pt_agent.analyzer import calibrate
from pt_agent.backends import CountBackend
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.export import export_package
from pt_agent.export.codegen import generate
from pt_agent.intake import import_module
from pt_agent.intake.coverage import check as check_graph
from pt_agent.planners.rule import plan
from pt_agent.planners.agent_tools import build_context, validate, _exact_check
from pt_agent.planners.tuner import tune
from pt_agent.tensor_sim import run_tensor, clear_caches
from pt_agent.tests.test_p2_export import emit_pair, package_modules
from pt_agent.tests.test_u3_kernels import mock, source


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        clear_caches()

    def test_package_cnn_emit_numpy_and_folded_linear_bn(self):
        rng = np.random.default_rng(52)
        weights = dict(dw=rng.normal(size=(8,1,3,3)), left=rng.normal(size=(3,8,1,1)),
                       right=rng.normal(size=(5,8,1,1)), fc=rng.normal(size=(4,128)))
        weights.update({'bn.weight': rng.normal(size=4), 'bn.bias': rng.normal(size=4),
                        'bn.running_mean': rng.normal(size=4), 'bn.running_var': rng.uniform(.5,2,size=4)})
        body = ('    d = conv2d(p,w=W["dw"],groups=8,pad=1)\n'
                '    a = avgpool(d,k=2,stride=2,pad=0)\n'
                '    l = conv2d(a,w=W["left"],pad=0)\n'
                '    r = conv2d(a,w=W["right"],pad=0)\n'
                '    c = concat(l,r,axis=1)\n'
                '    m = maxpool(c,k=3,stride=2,pad=1,method="cheb",deg=7,domain=(-100.,100.))\n'
                '    f = flatten(m)\n'
                '    y = linear(f,w=W["fc"])\n'
                '    z = batchnorm(y,bn="bn")\n    return z')
        images = rng.normal(size=(2,8,16,16))
        for slots in (4096,1024):
            with self.subTest(slots=slots), TemporaryDirectory() as tmp:
                text = source(slots,8,body); program = parse(text)
                package = export_package(text,program,weights,Path(tmp))
                (a,ac),(b,bc) = emit_pair(text,weights,package)
                self.assertEqual(a,b)
                self.assertEqual(list(ac),list(bc))
                for name in ac:
                    np.testing.assert_array_equal(ac[name],bc[name])
                expected = mock(program,weights,images)
                with package_modules(package) as runtime:
                    actual = runtime.run_numpy(weights,[images])[0]
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_allclose(run_tensor(program,weights,images,cache=False),expected,atol=1e-9,rtol=1e-9)
                generated = generate(program)
                self.assertIn("post_bn='bn'",generated)
                self.assertNotIn('ops.batchnorm',generated)

    def test_pool_variants_rejected_at_check_with_locations(self):
        inputs = (torch.arange(64,dtype=torch.float64).reshape(1,1,8,8),)
        for kwargs in (dict(kernel_size=3,stride=2,padding=1,ceil_mode=True),
                       dict(kernel_size=2,dilation=2,stride=1),dict(kernel_size=2,return_indices=True)):
            with self.subTest(kwargs=kwargs):
                graph = import_module(nn.MaxPool2d(**kwargs),inputs,'unsupported_pool',frontend='aten')
                checked = check_graph(graph)
                self.assertFalse(checked['ok'])
                for diag in checked['diags']:
                    self.assertIn('暂不支持',diag['reason'])
                    self.assertTrue(diag['op_id'])
                    self.assertGreater(diag['lineno'],0)

    def test_grouped_conv_check_boundary(self):
        text = source(1024,4,'    y = conv2d(p,w=W["w"],groups=2,pad=1)\n    return y',height=4,width=4)
        result = check(parse(text),{'w':(6,2,3,3)})
        self.assertFalse(result.ok)
        self.assertTrue(any('暂不支持' in diag.msg for diag in result.diags))

    def test_aten_adaptive_global_pool_then_flatten(self):
        rng = np.random.default_rng(61)
        images = rng.normal(size=(2,2,16,16))
        model = nn.Sequential(nn.Conv2d(2,8,1),nn.AdaptiveAvgPool2d(1),nn.Flatten()).double().eval()
        graph = import_module(model,(torch.from_numpy(images[:1]),),'global_pool_fixture',frontend='aten')
        means = [node for node in graph.nodes if node.op == 'mean']
        self.assertEqual(len(means),1)
        self.assertEqual(set(means[0].attrs['axes']),{2,3})
        self.assertTrue(means[0].attrs['keepdim'])
        calibration = calibrate(graph,{graph.inputs[0]:images})
        with torch.no_grad():
            expected = model(torch.from_numpy(images)).numpy()
        for slots in (4096,1024):
            text = plan(graph,calibration,deg=127,margin=1.5,slots=slots)
            program = parse(text)
            self.assertTrue(check(program,{name:value.shape for name,value in graph.weights.items()}).ok)
            actual = mock(program,graph.weights,images)
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(run_tensor(program,graph.weights,images,cache=False),actual,atol=1e-9,rtol=1e-9)
            with TemporaryDirectory() as tmp:
                package = export_package(text,program,graph.weights,Path(tmp))
                (a,ac),(b,bc) = emit_pair(text,graph.weights,package)
                self.assertEqual(a,b)
                self.assertEqual(list(ac),list(bc))
                for name in ac:
                    np.testing.assert_array_equal(ac[name],bc[name])
                with package_modules(package) as runtime:
                    np.testing.assert_array_equal(runtime.run_numpy(graph.weights,[images])[0],actual)

    def test_pool_calibration_validation_and_tuning_enumeration(self):
        from unittest.mock import patch
        # Analytic operator fixture, with original inputs supplied to the importer.
        # Real-model calibration uses the existing frozen CIFAR training selection.
        model = nn.Sequential(nn.MaxPool2d(2),nn.Flatten(),nn.Linear(12,2)).double().eval()
        with torch.no_grad():
            model[2].weight.fill_(.001)
            model[2].bias.copy_(torch.tensor([10.,0.],dtype=torch.float64))
        images = np.sin(np.arange(4*3*4*4,dtype=np.float64)).reshape(4,3,4,4)
        graph = import_module(model,(torch.from_numpy(images[:1]),),'pool_fixture',frontend='aten')
        arrays = {graph.inputs[0]:images}
        calibration = calibrate(graph,arrays)
        with patch('pt_agent.planners.agent_tools.heldout_set',return_value=(images,np.zeros(4,dtype=np.int64))):
            ctx = build_context(graph,calibration,n_val=4)
        program = parse(plan(graph,calibration,deg=127,margin=1.5))
        self.assertTrue(_exact_check(program,ctx)['ok'])
        a,b = validate(program,ctx,engine='mock'),validate(program,ctx,engine='tensor')
        self.assertEqual(len(a['rows']),1)
        self.assertEqual(a['agree'],b['agree'])
        self.assertAlmostEqual(a['max_u'],b['max_u'],places=10)
        cached = validate(program,ctx,engine='tensor')
        self.assertEqual(cached,b)
        result = tune(program,ctx,budget_s=60,max_sweeps=1)
        pool = next(op for op in result.program.ops if op.kind=='maxpool')
        self.assertTrue(any(change['line']==pool.lineno for change in result.changes))
        pool.attrs['domain']=(-.01,.01)
        for engine in ('mock','tensor'):
            bad = validate(result.program,ctx,engine=engine,cache=False)
            self.assertTrue(bad['diverged'])
            self.assertGreater(bad['max_u'],1)


if __name__=='__main__':
    unittest.main()
