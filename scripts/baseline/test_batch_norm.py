"""Independent inference-BN semantics; plaintext diagnostics are not FHE."""
import ast
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from model_graph import validate_graph,evaluate_reference,build_graph_model
from fx_to_hecate import translate,state_arrays
from batch_norm_ops import coefficients,expand_channels
from model_catalog import test_inputs
from test_fx_to_hecate import interpret_fragment
BASE=Path(__file__).parent
NAMES=('bn-batched','bn-spatial','bn-no-affine','bn-zero-gamma','bn-polynomial','bn-linear')

def descriptor(name):
    return json.loads((BASE/'cases'/(name+'.json')).read_text())

class BatchNormTests(unittest.TestCase):
    def test_channel_axis_and_epsilon_are_explicit(self):
        g,b=coefficients((2,2),[.5,-.25],[.75,3.75],[-1.,4.],[.125,.375],.25)
        self.assertEqual(g,[-1.,2.]);self.assertEqual(b,[.625,.875])
        self.assertEqual(expand_channels((2,2),g),[-1.,2.,-1.,2.])
        self.assertEqual(expand_channels((1,2,2),g),[-1.,-1.,2.,2.])
        self.assertEqual(evaluate_reference(descriptor('bn-batched'),[[0.,0.],[1.,1.]]),[.625,.875,-.375,2.875])

    def test_six_graphs_torch_reference_and_rule(self):
        for name in NAMES:
            d=descriptor(name);model,shape=build_graph_model(d);payload=translate(model,shape)
            for v in test_inputs(shape):
                expected=evaluate_reference(d,v.tolist())
                np.testing.assert_allclose(model(torch.from_numpy(v)).detach().numpy().reshape(-1),expected,atol=1e-12,rtol=1e-12)
                if name=='bn-zero-gamma':
                    self.assertEqual(payload['layout']['auxiliary_ciphertexts'][0]['dsl_name'],'zero_ct')
                    p=copy.deepcopy(payload);p['hecate_source']=p['hecate_source'].replace('zero_ct','x')
                    actual=interpret_fragment(p,np.zeros(4))
                else:actual=interpret_fragment(payload,v.reshape(-1))
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_module_function_keywords_and_counter_state(self):
        class Model(torch.nn.Module):
            def __init__(self,affine=True,track=True):
                super().__init__();self.bn=torch.nn.BatchNorm1d(2,affine=affine,track_running_stats=track,dtype=torch.float64)
            def forward(self,x):return torch.flatten(self.bn(x))
        for affine in (False,True):
            m=Model(affine).eval()
            with torch.no_grad():
                m.bn.running_mean.copy_(torch.tensor([.5,-.25]))
                m.bn.running_var.copy_(torch.tensor([.75,3.75]))
            before=state_arrays(m);p=translate(m,[2,2])
            self.assertEqual(int(before['bn.num_batches_tracked']),0)
            for x in test_inputs([2,2]):
                np.testing.assert_allclose(interpret_fragment(p,x.reshape(-1)),m(torch.from_numpy(x)).detach().numpy(),atol=1e-12,rtol=1e-12)
            for k,v in before.items():np.testing.assert_array_equal(v,state_arrays(m)[k])
        with self.assertRaises(ValueError):translate(Model().train(),[2,2])
        with self.assertRaisesRegex(ValueError,'running statistics'):translate(Model(track=False).eval(),[2,2])
        class Functional(Model):
            def forward(self,x):
                return torch.flatten(F.batch_norm(x,running_mean=self.bn.running_mean,running_var=self.bn.running_var,
                    weight=self.bn.weight,bias=self.bn.bias,training=False,eps=.25))
        translate(Functional().eval(),[2,2])
        class Training(Functional):
            def forward(self,x):
                return torch.flatten(F.batch_norm(x,self.bn.running_mean,self.bn.running_var,training=True))
        with self.assertRaisesRegex(ValueError,'training'):translate(Training().eval(),[2,2])

    def test_invalid_parameters_fail_closed(self):
        for field,bad in [('eps',True),('eps',float('nan')),('eps',-.1),('eps',2.),
                          ('running_mean',None),('weight','absent')]:
            d=descriptor('bn-batched');d['nodes'][0][field]=bad
            with self.subTest(field=field,bad=bad),self.assertRaises(ValueError):validate_graph(d)
        for key,value in [('variance',[-1.,1.]),('mean',[.5]),('gamma',[float('inf'),1.]),('beta',[[1.,2.]])]:
            d=descriptor('bn-batched');d['constants'][key]=value
            with self.assertRaises(ValueError):validate_graph(d)
        d=descriptor('bn-batched');d['constants']['variance']=[0.,0.];d['nodes'][0]['eps']=0.
        with self.assertRaises(ValueError):validate_graph(d)
        d=descriptor('bn-batched');d['nodes'][0]['training']=False
        with self.assertRaises(ValueError):validate_graph(d)

    def test_upstream_abstract_bn_formula_not_packing_helper(self):
        path=BASE.parents[1]/'third_party/dacapo/python/poly/poly/MPCB.py'
        tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='abstractBN')
        space={'torch':torch}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),space)
        bn=torch.nn.BatchNorm1d(2,dtype=torch.float64).eval()
        with torch.no_grad():
            bn.weight.copy_(torch.tensor([-1.,4.]));bn.bias.copy_(torch.tensor([.125,.375]))
            bn.running_var.copy_(torch.tensor([.75,3.75]));bn.running_mean.copy_(torch.tensor([.5,-.25]))
        bn.eps=.25;g,b=space['abstractBN'](bn)
        np.testing.assert_array_equal(g.detach().numpy(),[-1.,2.])
        np.testing.assert_array_equal(b.detach().numpy(),[.625,.875])

if __name__=='__main__':unittest.main()
