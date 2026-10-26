"""U5 independent coefficient-only P2 package and prompt version contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.backends import MockBackend
from pt_agent.capabilities import render_capabilities
from pt_agent.dsl.parser import parse
from pt_agent.export import export_package
from pt_agent.lower import lower,pack_inputs,decode_outputs
from pt_agent.planners.prompts import PROMPT_VERSION
from pt_agent.tests.test_p2_export import package_modules,emit_pair


MIXED='''@fhe.program(slots=16384)
def mixed(x:Cipher[1,9],image:Cipher[1,5,2,2]):
    p=pack(x,layout="flat")
    a=act(p,fn="exp",method="remez",deg=7,domain=(-2.,2.))
    inverse=act(a,fn="inv",method="newton",deg0=3,iters=2,domain=(1.,10.))
    c=act(inverse,fn="relu",method="composite",degrees=(3,3),eps=0.2,domain=(-2.,2.))
    q=pack(image,layout="mux")
    r=act(q,fn="exp",method="remez",deg=7,domain=(-2.,2.))
    s=maxpool(r,k=2,stride=1,pad=0,method="composite",degrees=(3,3),eps=0.2,domain=(-4.,4.))
    return c,s
'''


class PackageTests(unittest.TestCase):
    def test_mixed_package_has_coefficients_and_never_refits(self):
        p=parse(MIXED)
        arrays=[np.linspace(0.5,1.5,18).reshape(2,9),np.linspace(-1,1,40).reshape(2,5,2,2)]
        b=MockBackend(p.header.slots,batch=2)
        b.bind(pack_inputs(p,arrays))
        values,layouts=lower(p,{},b)
        expected=decode_outputs([v.value for v in values],layouts)
        with TemporaryDirectory() as directory:
            package=export_package(MIXED,p,{},Path(directory))
            entries=json.loads((package/'config/approximations.json').read_text())
            self.assertEqual({a['method'] for a in entries.values()},{'remez','composite','newton'})
            for a in entries.values():
                self.assertIn('coefficients',a)
                if a['method']=='newton':
                    self.assertEqual(a['iters'],2)
                    self.assertEqual(a['iteration'],'inv_y_2_minus_xy')
                if a['method']=='composite':
                    self.assertEqual(a['eps'],0.2)
                    self.assertEqual(len(a['coefficients']),len(a['degrees']))
            evaluator=(package/p.name/'core/approx.py').read_text()
            self.assertNotIn('linalg',evaluator)
            self.assertNotIn('def remez',evaluator)
            with package_modules(package) as runtime,patch('numpy.linalg.solve',side_effect=AssertionError('refit')):
                actual=runtime.run_numpy({},arrays)
            for x,y in zip(actual,expected):
                np.testing.assert_array_equal(x,y)
            a,b=emit_pair(MIXED,{},package)
            self.assertEqual(a[0],b[0])
            self.assertEqual(list(a[1]),list(b[1]))
            for name in a[1]:
                np.testing.assert_array_equal(a[1][name],b[1][name])

    def test_new_prompt_is_registered_and_default_is_unchanged(self):
        self.assertEqual(PROMPT_VERSION,'r8-v4')
        for version in ('r8-v4','r8-v5'):
            self.assertNotIn('method="newton"',render_capabilities(version))
            self.assertNotIn('method="remez"',render_capabilities(version))
        text=render_capabilities('r8-v6')
        for term in ('method="remez"','method="composite"','method="newton"','deg0','eps','iters','3 per rsqrt'):
            self.assertIn(term,text)


if __name__=='__main__':
    unittest.main()
