"""C-order metadata reshape and Conv->BatchNorm layout composition."""
import copy,json,unittest
from pathlib import Path
import numpy as np
import torch
from logical_reshape import reshape_shape
from model_graph import build_graph_model,evaluate_reference,validate_graph
from fx_to_hecate import translate
from model_catalog import test_inputs
from test_fx_to_hecate import interpret_fragment
BASE=Path(__file__).parent
NAMES=('reshape-bn','reshape-conv1-bn','reshape-conv2-bn')

class ReshapeTests(unittest.TestCase):
    def test_static_shape_inference_and_invalid_sizes(self):
        self.assertEqual(reshape_shape((4,),(1,2,-1)),(1,2,2))
        self.assertEqual(reshape_shape((2,2),(4,)),(4,))
        for shape in [(),(0,4),(True,4),(-1,-1),(3,-1),(8,),(1,1,1,1,4),(2.,2),('4',)]:
            with self.subTest(shape=shape),self.assertRaises(ValueError):reshape_shape((4,),shape)

    def test_three_models_torch_and_independent_reference(self):
        for name in NAMES:
            d=json.loads((BASE/'cases'/(name+'.json')).read_text());model,shape=build_graph_model(d)
            p=translate(model,shape)
            for x in test_inputs(shape):
                expected=evaluate_reference(d,x.tolist())
                np.testing.assert_allclose(model(torch.from_numpy(x)).detach().numpy(),expected,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(interpret_fragment(p,x.reshape(-1)),expected,atol=1e-12,rtol=1e-12)
        d=json.loads((BASE/'cases/reshape-conv1-bn.json').read_text())
        # conv=[x0+.25,x2+.25,2*x0-.125,2*x2-.125], BN has separate channel factors.
        self.assertEqual(evaluate_reference(d,[[1.,2.,3.,4.]]),[-.625,-2.625,4.625,12.625])

    def test_method_forms_do_not_emit_cipher_operations(self):
        class M(torch.nn.Module):
            def forward(self,x):
                a=x.reshape(1,2,2)
                b=a.reshape((2,-1))
                return torch.reshape(b,(4,))
        p=translate(M().eval(),[4])
        self.assertEqual(p['public_constants'],{})
        self.assertIn('return x',p['hecate_source'])
        self.assertNotIn('rotate',p['hecate_source'])
        np.testing.assert_array_equal(interpret_fragment(p,np.array([1.,2.,3.,4.])),[1.,2.,3.,4.])
        class Transpose(torch.nn.Module):
            def forward(self,x):return x.reshape(2,2).transpose(0,1).reshape(4)
        with self.assertRaises(ValueError):translate(Transpose().eval(),[4])

    def test_invalid_graph_and_no_cipher_driven_shapes(self):
        d=json.loads((BASE/'cases/reshape-bn.json').read_text())
        for target in ([3,-1],[0,4],[True,4],['x',4],[8]):
            bad=copy.deepcopy(d);bad['nodes'][0]['shape']=target
            with self.assertRaises(ValueError):validate_graph(bad)
        class M(torch.nn.Module):
            def forward(self,x):return x.reshape(x[0],-1)
        with self.assertRaises(ValueError):translate(M().eval(),[4])

if __name__=='__main__':unittest.main()
