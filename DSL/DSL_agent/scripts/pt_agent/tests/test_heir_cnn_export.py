"""Deterministic small arithmetic and source checks for the HEIR CNN adapter."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


def small_fixture():
    import numpy as np
    def Node(id,op,inputs,outputs,attrs,src):
        return SimpleNamespace(id=id,op=op,inputs=inputs,outputs=outputs,attrs=attrs,src=src)
    shapes = {'x':(1,2,4,4),'conv':(1,2,4,4),'bn':(1,2,4,4),
              'act':(1,2,4,4),'residual':(1,2,4,4),'rows':(1,2,2,4),
              'cols':(1,2,2,2),'padded':(1,4,2,2),'mean':(1,4),'out':(1,3)}
    nodes = [Node('n0','conv2d',['x'],['conv'],dict(weight='conv.weight',stride=(1,1),padding=(1,1),dilation=(1,1),groups=1),'fixture'),
             Node('n1','batchnorm',['conv'],['bn'],dict(weight='bn.weight',bias='bn.bias',mean='bn.mean',var='bn.var',eps=1e-5),'fixture'),
             Node('n2','act',['bn'],['act'],dict(fn='silu'),'fixture'),
             Node('n3','add',['act','x'],['residual'],{},'fixture'),
             Node('n4','slice',['residual'],['rows'],dict(axis=2,start=0,end=4,step=2),'fixture'),
             Node('n5','slice',['rows'],['cols'],dict(axis=3,start=0,end=4,step=2),'fixture'),
             Node('n6','pad',['cols'],['padded'],dict(pads=(0,0,0,0,1,1)),'fixture'),
             Node('n7','mean',['padded'],['mean'],dict(axes=(2,3),keepdim=False),'fixture'),
             Node('n8','linear',['mean'],['out'],dict(weight='linear.weight',bias='linear.bias'),'fixture')]
    weights = {'conv.weight':(np.arange(36,dtype=np.float64).reshape(2,2,3,3)-17.5)/100,
               'bn.weight':np.array([.5,1.25]),'bn.bias':np.array([-.1,.2]),
               'bn.mean':np.array([.1,-.1]),'bn.var':np.array([.75,1.5]),
               'linear.weight':np.arange(12,dtype=np.float64).reshape(3,4)/20,
               'linear.bias':np.array([.1,-.2,.3])}
    graph = SimpleNamespace(name='small_cnn',nodes=nodes,tensors={n:SimpleNamespace(name=n,shape=s,kind='features') for n,s in shapes.items()},weights=weights,inputs=['x'],outputs=['out'])
    program = '@fhe.program(slots=16384)\ndef small(x: Cipher[1,2,4,4]):\n    y = act(x, fn="silu", method="cheb", deg=63, domain=(-4.0,4.0))\n    return y\n'
    return graph,program


class ExportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        cls.np = np

    def test_scalar_tape_retains_fixed_polynomial_arithmetic(self):
        from pt_agent.experiments.heir_cnn_export import chebyshev_tape
        from pt_agent.kernels.poly import cheb_eval
        np = self.np
        for domain in ((-4.,4.),(-8.378011205301343,10.933806383831946),(-27.35102599948144,39.52667325340186)):
            tape,out,c = chebyshev_tape('silu',63,domain)
            x = np.linspace(*domain,257,dtype=np.float64)
            lo,hi = domain
            reference = cheb_eval(None,x*(2/(hi-lo)) + (-(hi+lo)/(hi-lo)),c)
            observed = tape.evaluate(x,out)
            self.assertEqual(observed.dtype,np.float64)
            self.assertTrue(np.isfinite(observed).all())
            self.assertTrue(np.all(np.abs(reference-observed) <= 1e-12+1e-12*np.abs(reference)))
            self.assertEqual(float(np.abs(reference-observed).max()),0.)

    def test_small_graph_covers_original_cnn_arithmetic(self):
        from pt_agent.experiments.heir_cnn_export import export_graph
        graph,program = small_fixture()
        text,record = export_graph(graph,program,function_name='small_cnn')
        self.assertEqual(record['graph_nodes'],9)
        self.assertEqual(record['input_shape'],[1,2,4,4])
        self.assertEqual(record['output_shape'],[1,3])
        self.assertEqual(len(record['approximations']),1)
        self.assertEqual(record['approximations'][0]['deg'],63)
        self.assertFalse(record['approximation_selected_by_heir'])
        self.assertIn('tensor.pad',text)
        self.assertEqual(text.count('tensor.extract_slice'),2)
        self.assertIn('linalg.conv_2d_nchw_fchw',text)
        self.assertIn('linalg.matvec',text)
        self.assertIn('"parallel", "parallel", "reduction"',text)
        self.assertIn('[[0], [1], [2, 3]]',text)
        self.assertIn('tensor<1x2x4x4xf64> {secret.secret}',text)
        self.assertNotIn('fastmath',text)
        out = os.environ.get('HEIR_CNN_SMALL_OUTPUT')
        if out:
            p = Path(out);p.mkdir(parents=True,exist_ok=True)
            (p/'small_cnn.mlir').write_text(text)
            (p/'small_cnn_export.json').write_text(json.dumps(record,indent=2)+'\n')

    def test_small_convolution_and_batchnorm_have_strict_math(self):
        from pt_agent.kernels.conv import fold_bn
        np = self.np
        graph,_ = small_fixture()
        x = np.arange(32,dtype=np.float64).reshape(1,2,4,4)/32 - .5
        w = graph.weights['conv.weight']
        padded = np.pad(x,((0,0),(0,0),(1,1),(1,1)))
        observed = np.einsum('nchwkl,ockl->nohw',np.lib.stride_tricks.sliding_window_view(padded,(3,3),axis=(2,3)),w)
        reference = np.zeros((1,2,4,4),dtype=np.float64)
        for o in range(2):
            for h in range(4):
                for j in range(4):
                    for c in range(2):
                        for kh in range(3):
                            for kw in range(3):reference[0,o,h,j] += padded[0,c,h+kh,j+kw]*w[o,c,kh,kw]
        self.assertTrue(np.all(np.abs(reference-observed)<=1e-12+1e-12*np.abs(reference)))
        scale,shift = fold_bn(*(graph.weights['bn.'+n] for n in ('weight','bias','mean','var')),1e-5)
        observed = observed*scale[None,:,None,None]+shift[None,:,None,None]
        reference = reference*scale[None,:,None,None]+shift[None,:,None,None]
        self.assertTrue(np.all(np.abs(reference-observed)<=1e-12+1e-12*np.abs(reference)))

    def test_rejects_nonfinite_or_changed_activation_and_shapes(self):
        from pt_agent.experiments.heir_cnn_export import export_graph, scalar_literal
        graph,program = small_fixture()
        with self.assertRaises(ValueError): scalar_literal(float('nan'))
        with self.assertRaises(ValueError): export_graph(graph,program.replace('deg=63','deg=31'))
        with self.assertRaises(ValueError): export_graph(graph,program.replace('1,2,4,4','1,2,8,8'))

    def test_exports_independent_native_operator_cases(self):
        from pt_agent.experiments.heir_cnn_export import Emitter,Tensor,tensor_type
        graph,_ = small_fixture()
        np=self.np
        cases={}
        for kind,shape in (('conv',(1,2,4,4)),('bn',(1,2,4,4)),('poly',(1,2,4,4)),
                           ('shortcut',(1,2,4,4)),('mean',(1,4,2,2)),('linear',(1,4))):
            e=Emitter();x=Tensor('%arg0',shape)
            if kind=='conv':y=e.conv(x,graph.weights['conv.weight'],dict(padding=(1,1),stride=(1,1)),shape)
            elif kind=='bn':
                y=e.elementwise(x,e.constant(np.array([.5,1.25])),'mulf',channel=True)
                y=e.elementwise(y,e.constant(np.array([-.1,.2])),'addf',channel=True)
            elif kind=='poly':y,_=e.activation(x,dict(fn='silu',deg=63,domain=(-4.,4.)))
            elif kind=='shortcut':
                sliced=e.fresh((1,2,2,2))
                e.emit(f'{sliced.name} = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 2, 2, 2] [1, 1, 2, 2] : {x.ty} to {sliced.ty}')
                y=e.pad(sliced,(0,1,0,0),(0,1,0,0))
            elif kind=='mean':y=e.mean(x,(2,3))
            elif kind=='linear':y=e.linear(x,graph.weights['linear.weight'],graph.weights['linear.bias'])
            text='\n'.join(['module {',f'  func.func @{kind}(%arg0: {tensor_type(shape)} {{secret.secret}}) -> {y.ty} {{',*e.lines,f'    return {y.name} : {y.ty}','  }','}',''])
            cases[kind]=text
        self.assertEqual(len(cases),6)
        out=os.environ.get('HEIR_CNN_SMALL_OUTPUT')
        if out:
            p=Path(out);p.mkdir(parents=True,exist_ok=True)
            for kind,text in cases.items():(p/(kind+'_only.mlir')).write_text(text)

    def test_emitted_tensor_constants_retain_numerical_values(self):
        from pt_agent.experiments.heir_cnn_export import export_graph
        from pt_agent.experiments.heir_cnn_math_check import check_constants
        graph,program=small_fixture()
        text,_=export_graph(graph,program,function_name='small_cnn')
        checks=check_constants(graph,text)
        self.assertEqual(len(checks),5)
        self.assertTrue(all(c['passed'] and c['numerically_equal'] for c in checks))
        changed=text.replace('-1.74999999999999989e-01','-1.24999999999999989e-01',1)
        with self.assertRaises(ValueError):check_constants(graph,changed)

    def test_math_contract_keeps_dtype_shape_and_original_tolerance(self):
        from pt_agent.experiments.heir_cnn_math_check import check_math
        np=self.np
        reference=np.array([[1.,-2.]],dtype=np.float64)
        self.assertTrue(check_math(reference,reference.copy())['passed'])
        self.assertFalse(check_math(reference,reference+1e-8)['passed'])
        with self.assertRaises(ValueError):check_math(reference,reference.astype(np.float32))
        with self.assertRaises(ValueError):check_math(reference,reference.reshape(2))
        invalid=reference.copy();invalid[0,0]=np.inf
        with self.assertRaises(ValueError):check_math(reference,invalid)


if __name__ == '__main__':
    unittest.main()
