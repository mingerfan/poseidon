"""U5 DSL contracts and actual lowering in single and multiple ciphertexts."""

import unittest

import numpy as np

from pt_agent.backends import CountBackend,MockBackend
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.dsl.typecheck import check
from pt_agent.kernels import approx
from pt_agent.kernels.layout import FlatLayout,layout_blocks
from pt_agent.kernels.poly import act_cheb
from pt_agent.lower import lower,pack_inputs,decode_outputs
from pt_agent.tensor_sim import run_tensor


SETTINGS = (
    'fn="exp",method="remez",deg=7,domain=(-2.0,2.0)',
    'fn="relu",method="composite",degrees=(3,3),eps=0.2,domain=(-2.0,2.0)',
    'fn="inv",method="newton",deg0=3,iters=2,domain=(0.25,2.0)',
    'fn="rsqrt",method="newton",deg0=3,iters=2,domain=(0.25,2.0)',
)


def source(settings,*,slots=4096,layout='flat',pool=False,wide=False):
    shape = ('1,1025' if wide else '1,9') if layout == 'flat' else ('1,257,2,2' if wide else '1,5,2,2')
    call = ('maxpool(p,k=2,stride=1,pad=0,'+settings+')' if pool else 'act(p,'+settings+')')
    return (f'@fhe.program(slots={slots})\ndef sample(x:Cipher[{shape}]):\n'
            f'    p=pack(x,layout="{layout}")\n    y={call}\n    return y\n')


class LowerTests(unittest.TestCase):
    def test_parser_printer_and_typecheck(self):
        for settings in SETTINGS:
            p=parse(source(settings))
            self.assertTrue(check(p,{}).ok)
            printed=print_program(p)
            self.assertEqual(print_program(parse(printed)),printed)

    def test_bad_combinations_are_typecheck_errors(self):
        bad=(
            'fn="exp",method="newton",deg0=3,iters=1,domain=(0.1,2.0)',
            'fn="inv",method="newton",deg0=3,iters=1,domain=(0.0,2.0)',
            'fn="relu",method="composite",degrees=(4,),eps=0.2,domain=(-2.,2.)',
            'fn="relu",method="composite",degrees=(3,),eps=0.0,domain=(-2.,2.)',
            'fn="silu",method="composite",degrees=(3,),eps=0.2,domain=(-2.,2.)',
            'fn="relu",method="remez",deg=0,domain=(-2.,2.)')
        for settings in bad:
            p=parse(source(settings))
            result=check(p,{})
            self.assertFalse(result.ok)
            self.assertTrue(any(d.code=='E_APPROX' for d in result.diags))
        p=parse(source('method="newton",deg0=3,iters=1,domain=(0.1,2.)',layout='mux',pool=True))
        self.assertTrue(any(d.code=='E_APPROX' for d in check(p,{}).diags))

    def test_single_multi_mux_flat_and_zero_tails(self):
        for settings in SETTINGS:
            for layout in ('flat','mux'):
                for slots in (4096,1024):
                    with self.subTest(settings=settings,layout=layout,slots=slots):
                        p=parse(source(settings,slots=slots,layout=layout,wide=True))
                        shape=(2,*p.params[0].shape[1:])
                        positive='method="newton"' in settings
                        x=np.linspace(0.5 if positive else -1.5,1.5,np.prod(shape)).reshape(shape)
                        b=MockBackend(slots,batch=2)
                        b.bind(pack_inputs(p,[x]))
                        values,layouts=lower(p,{},b)
                        mock=decode_outputs([v.value for v in values],layouts)[0]
                        tensor=run_tensor(p,{},x,cache=False)
                        self.assertTrue(np.all(np.abs(mock-tensor) <= 1e-9*(1+np.abs(mock))))
                        attrs=p.ops[-1].attrs
                        coefficients=approx.fitted_coefficients(attrs)
                        if attrs['method']=='composite':
                            sign=x/attrs['domain'][1]
                            for c in coefficients:
                                sign=np.polynomial.chebyshev.chebval(sign,c)
                            theoretical=x*(sign+0.5)
                        else:
                            lo,hi=attrs['domain']
                            y=np.polynomial.chebyshev.chebval(x*(2/(hi-lo))-(hi+lo)/(hi-lo),coefficients)
                            if attrs['method']=='newton':
                                x_half=x*0.5
                                for _ in range(attrs['iters']):
                                    y=y*(2-x*y) if attrs['fn']=='inv' else y*(1.5-x_half*(y*y))
                            theoretical=y
                        self.assertTrue(np.all(np.abs(mock-theoretical) <= 1e-9*(1+np.abs(theoretical))))
                        for value,(_,_,block) in zip(values,layout_blocks(layouts[0])):
                            if isinstance(block,FlatLayout):
                                self.assertTrue(np.all(value.value[...,block.size:]==0.0))
                        if slots==1024:
                            self.assertGreater(len(values),1)

    def test_exact_reference_uses_valid_slots(self):
        for settings in SETTINGS:
            for layout in ('flat','mux'):
                p=parse(source(settings,slots=1024,layout=layout,wide=True))
                shape=(2,*p.params[0].shape[1:])
                x=np.linspace(0.5,1.5,np.prod(shape)).reshape(shape)
                b=MockBackend(1024,batch=2)
                b.bind(pack_inputs(p,[x]))
                values,layouts=lower(p,{},b,act_mode='exact')
                actual=decode_outputs([v.value for v in values],layouts)[0]
                expected=approx.function(p.ops[-1].attrs['fn'],x)
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(run_tensor(p,{},x,act_mode='exact'),expected)

    def test_count_depth(self):
        for settings in SETTINGS:
            p=parse(source(settings))
            b=CountBackend(p.header.slots)
            lower(p,{},b)
            attrs=p.ops[-1].attrs
            reference=CountBackend(p.header.slots)
            if attrs['method']=='remez':
                result=act_cheb(reference,reference.input(0),attrs['fn'],attrs['deg'],attrs['domain'])
                self.assertEqual(b.stats()['max_depth'],result.depth)
            elif attrs['method']=='newton':
                result=approx.act_remez(reference,reference.input(0),attrs['fn'],attrs['deg0'],attrs['domain'])
                expected=result.depth+(2 if attrs['fn']=='inv' else 3)*attrs['iters']
                self.assertEqual(b.stats()['max_depth'],expected)

    def test_maxpool_new_methods(self):
        from pt_agent.kernels.cnn import maxpool_tensor
        for settings in ('method="remez",deg=7,domain=(-4.,4.)',
                         'method="composite",degrees=(3,3),eps=0.2,domain=(-4.,4.)'):
            p=parse(source(settings,slots=1024,layout='mux',pool=True,wide=True))
            self.assertTrue(check(p,{}).ok)
            shape=(2,*p.params[0].shape[1:])
            x=np.linspace(-1,1,np.prod(shape)).reshape(shape)
            b=MockBackend(1024,batch=2)
            b.bind(pack_inputs(p,[x]))
            values,layouts=lower(p,{},b)
            mock=decode_outputs([v.value for v in values],layouts)[0]
            tensor=run_tensor(p,{},x,cache=False)
            np.testing.assert_allclose(mock,tensor,rtol=1e-9,atol=1e-9)


if __name__=='__main__':
    unittest.main()
